#include "encoder_gpu.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace moge2_native {
namespace {
constexpr std::uint32_t kEmbedding = 384u;
constexpr std::uint32_t kHeads = 6u;
constexpr std::uint32_t kBlocks = 12u;
constexpr const char* kPrefix = "encoder.backbone.";

const da3_native::VulkanBuffer& weight(
    const da3_native::GpuModel& model, const std::string& name) {
    return model.tensor(name).buffer;
}

std::string block_name(std::uint32_t block, const char* suffix) {
    return std::string(kPrefix) + "blocks." + std::to_string(block) + suffix;
}
}  // namespace

EncoderOutput encode_vits(
    da3_native::VulkanContext& context,
    da3_native::GpuModel& model,
    da3_native::VulkanOperators& operators,
    da3_native::VulkanBuffer image,
    std::uint32_t width,
    std::uint32_t height) {
    if (!width || !height || width % 14u || height % 14u)
        throw std::invalid_argument("MoGe-2 encoder input must be divisible by 14");
    const std::uint32_t token_width = width / 14u;
    const std::uint32_t token_height = height / 14u;
    const std::uint32_t patches = token_width * token_height;
    const std::uint32_t tokens = patches + 1u;
    const std::uint64_t elements = std::uint64_t(tokens) * kEmbedding;
    const VkDeviceSize bytes = elements * sizeof(float);
    da3_native::VulkanBuffer state = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer next = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer normalized = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer attended = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer projected = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer qkv = context.create_device_buffer(bytes * 3u);
    da3_native::VulkanBuffer hidden = context.create_device_buffer(bytes * 4u);
    da3_native::VulkanBuffer scores = context.create_device_buffer(
        std::uint64_t(kHeads) * tokens * tokens * sizeof(float));
    da3_native::VulkanBuffer feature_a = context.create_device_buffer(
        std::uint64_t(patches) * kEmbedding * sizeof(float));
    da3_native::VulkanBuffer feature_b = context.create_device_buffer(
        std::uint64_t(patches) * kEmbedding * sizeof(float));
    EncoderOutput output{
        token_width, token_height,
        context.create_device_buffer(
            std::uint64_t(patches) * kEmbedding * sizeof(float)),
        context.create_device_buffer(kEmbedding * sizeof(float))};

    context.batch([&] {
        operators.prepare_tokens(
            state, image,
            weight(model, std::string(kPrefix) + "patch_embed.proj.weight"),
            weight(model, std::string(kPrefix) + "patch_embed.proj.bias"),
            weight(model, std::string(kPrefix) + "cls_token"),
            weight(model, std::string(kPrefix) + "pos_embed"),
            width, height, kEmbedding);
        for (std::uint32_t block = 0; block < kBlocks; ++block) {
            operators.layer_norm(
                normalized, state,
                weight(model, block_name(block, ".norm1.weight")),
                weight(model, block_name(block, ".norm1.bias")),
                tokens, kEmbedding, 1.0e-6f);
            operators.linear(
                qkv, normalized,
                weight(model, block_name(block, ".attn.qkv.weight")),
                weight(model, block_name(block, ".attn.qkv.bias")),
                tokens, kEmbedding, kEmbedding * 3u, false);
            operators.attention_head64(
                attended, qkv, tokens, kHeads, &scores);
            operators.linear(
                projected, attended,
                weight(model, block_name(block, ".attn.proj.weight")),
                weight(model, block_name(block, ".attn.proj.bias")),
                tokens, kEmbedding, kEmbedding, false);
            operators.add_scaled(
                next, state, projected,
                weight(model, block_name(block, ".ls1.gamma")),
                static_cast<std::uint32_t>(elements), kEmbedding);
            std::swap(state, next);
            operators.layer_norm(
                normalized, state,
                weight(model, block_name(block, ".norm2.weight")),
                weight(model, block_name(block, ".norm2.bias")),
                tokens, kEmbedding, 1.0e-6f);
            operators.linear(
                hidden, normalized,
                weight(model, block_name(block, ".mlp.fc1.weight")),
                weight(model, block_name(block, ".mlp.fc1.bias")),
                tokens, kEmbedding, kEmbedding * 4u, true);
            operators.linear(
                projected, hidden,
                weight(model, block_name(block, ".mlp.fc2.weight")),
                weight(model, block_name(block, ".mlp.fc2.bias")),
                tokens, kEmbedding * 4u, kEmbedding, false);
            operators.add_scaled(
                next, state, projected,
                weight(model, block_name(block, ".ls2.gamma")),
                static_cast<std::uint32_t>(elements), kEmbedding);
            std::swap(state, next);
            if (block == 5u || block == 11u) {
                operators.layer_norm(
                    normalized, state,
                    weight(model, std::string(kPrefix) + "norm.weight"),
                    weight(model, std::string(kPrefix) + "norm.bias"),
                    tokens, kEmbedding, 1.0e-6f);
                da3_native::VulkanBuffer& feature =
                    block == 5u ? feature_a : feature_b;
                const char* projection = block == 5u ?
                    "encoder.output_projections.0" :
                    "encoder.output_projections.1";
                operators.project_tokens(
                    feature, normalized,
                    weight(model, std::string(projection) + ".weight"),
                    weight(model, std::string(projection) + ".bias"),
                    token_width, token_height, kEmbedding, kEmbedding);
            }
        }
        operators.add(
            output.features, feature_a, feature_b,
            patches * kEmbedding);
    });
    context.copy(
        output.class_token, 0u, normalized, 0u,
        kEmbedding * sizeof(float));
    return output;
}

}  // namespace moge2_native
