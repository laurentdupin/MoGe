"""Derive a deterministic pickle-free MoGe-2 safetensors artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

CONVERTER = "moge2-export-pytorch-v1"
FORMAT = "MOGE2_SAFE_F32_V1"


def save_deterministic(tensors: dict[str, torch.Tensor], metadata: dict[str, str], output: Path) -> None:
    """Write the simple contiguous F32 SafeTensors subset used by the DLL."""
    header: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().to(torch.float32).contiguous()
        payload = tensor.numpy().tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(
        header, sort_keys=True, ensure_ascii=True,
        separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with output.open("wb") as destination:
        destination.write(struct.pack("<Q", len(encoded)))
        destination.write(encoded)
        for payload in payloads:
            destination.write(payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("vits-normal",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = sha256(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model_config", "model"}:
        raise ValueError("unsupported MoGe-2 checkpoint envelope")
    config = checkpoint["model_config"]
    if config.get("encoder", {}).get("backbone") != "dinov2_vits14" or \
            "normal_head" not in config:
        raise ValueError("checkpoint is not MoGe-2 ViT-S normal")
    tensors = {}
    for name in sorted(checkpoint["model"]):
        value = checkpoint["model"][name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"non-tensor state entry: {name}")
        tensors[name] = value.detach().cpu().to(torch.float32).contiguous()
    encoder = config["encoder"]
    embedding = int(encoder["dim_out"])
    backbone_embedding = int(tensors["encoder.backbone.cls_token"].shape[-1])
    block_indices = sorted({
        int(name.split(".")[3]) for name in tensors
        if name.startswith("encoder.backbone.blocks.")
    })
    captures = list(encoder["intermediate_layers"])
    if embedding != 384 or len(captures) != 2 or block_indices != list(range(len(block_indices))):
        raise ValueError("unsupported MoGe-2 encoder topology")
    tensors["moge2.config.encoder"] = torch.tensor([
        backbone_embedding, backbone_embedding // 64,
        len(block_indices), captures[0], captures[1]], dtype=torch.float32)
    metadata = {
        "format": FORMAT,
        "format_version": "1",
        "variant": args.variant,
        "canonical_sha256": canonical,
        "converter": CONVERTER,
        "model_config": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "cache_key": f"moge2:{canonical}:{CONVERTER}:1:{args.variant}",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_deterministic(tensors, metadata, args.output)
    print(json.dumps({
        "format": FORMAT,
        "variant": args.variant,
        "canonical_sha256": canonical,
        "derived_sha256": sha256(args.output),
        "tensor_count": len(tensors),
        "converter": CONVERTER,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
