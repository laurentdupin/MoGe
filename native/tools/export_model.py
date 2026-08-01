"""Derive a deterministic pickle-free MoGe-2 safetensors artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

CONVERTER = "moge2-export-pytorch-v1"
FORMAT = "MOGE2_SAFE_F32_V1"


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
    save_file(tensors, args.output, metadata=metadata)
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
