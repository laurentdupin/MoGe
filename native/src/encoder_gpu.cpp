#include "encoder_gpu.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moge2_native {
namespace {
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
    const ModelConfig& config,
    da3_native::VulkanBuffer image,
    std::uint32_t width,
    std::uint32_t height) {
    if (!width || !height || width % 14u || height % 14u)
        throw std::invalid_argument("MoGe-2 encoder input must be divisible by 14");
    const std::uint32_t token_width = width / 14u;
    const std::uint32_t token_height = height / 14u;
    const std::uint32_t patches = token_width * token_height;
    const std::uint32_t tokens = patches + 1u;
    const std::uint32_t embedding = config.embedding;
    const std::uint64_t elements = std::uint64_t(tokens) * embedding;
    const VkDeviceSize bytes = elements * sizeof(float);
    da3_native::VulkanBuffer state = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer next = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer normalized = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer attended = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer projected = context.create_device_buffer(bytes);
    da3_native::VulkanBuffer qkv = context.create_device_buffer(bytes * 3u);
    da3_native::VulkanBuffer hidden = context.create_device_buffer(bytes * 4u);
    da3_native::VulkanBuffer scores = context.create_device_buffer(
        std::uint64_t(config.heads) * tokens * tokens * sizeof(float));
    std::vector<da3_native::VulkanBuffer> captured;
    captured.reserve(config.capture_count);
    for (std::uint32_t index = 0u; index < config.capture_count; ++index) {
        captured.push_back(context.create_device_buffer(
            std::uint64_t(patches) * config.decoder_embedding * sizeof(float)));
    }
    EncoderOutput output{
        token_width, token_height,
        context.create_device_buffer(
            std::uint64_t(patches) * config.decoder_embedding * sizeof(float)),
        context.create_device_buffer(embedding * sizeof(float))};

    context.batch([&] {
        operators.prepare_tokens(
            state, image,
            weight(model, std::string(kPrefix) + "patch_embed.proj.weight"),
            weight(model, std::string(kPrefix) + "patch_embed.proj.bias"),
            weight(model, std::string(kPrefix) + "cls_token"),
            weight(model, std::string(kPrefix) + "pos_embed"),
            width, height, embedding);
        for (std::uint32_t block = 0; block < config.blocks; ++block) {
            operators.layer_norm(
                normalized, state,
                weight(model, block_name(block, ".norm1.weight")),
                weight(model, block_name(block, ".norm1.bias")),
                tokens, embedding, 1.0e-6f);
            operators.linear(
                qkv, normalized,
                weight(model, block_name(block, ".attn.qkv.weight")),
                weight(model, block_name(block, ".attn.qkv.bias")),
                tokens, embedding, embedding * 3u, false);
            operators.attention_head64(
                attended, qkv, tokens, config.heads, &scores);
            operators.linear(
                projected, attended,
                weight(model, block_name(block, ".attn.proj.weight")),
                weight(model, block_name(block, ".attn.proj.bias")),
                tokens, embedding, embedding, false);
            operators.add_scaled(
                next, state, projected,
                weight(model, block_name(block, ".ls1.gamma")),
                static_cast<std::uint32_t>(elements), embedding);
            std::swap(state, next);
            operators.layer_norm(
                normalized, state,
                weight(model, block_name(block, ".norm2.weight")),
                weight(model, block_name(block, ".norm2.bias")),
                tokens, embedding, 1.0e-6f);
            operators.linear(
                hidden, normalized,
                weight(model, block_name(block, ".mlp.fc1.weight")),
                weight(model, block_name(block, ".mlp.fc1.bias")),
                tokens, embedding, embedding * 4u, true);
            operators.linear(
                projected, hidden,
                weight(model, block_name(block, ".mlp.fc2.weight")),
                weight(model, block_name(block, ".mlp.fc2.bias")),
                tokens, embedding * 4u, embedding, false);
            operators.add_scaled(
                next, state, projected,
                weight(model, block_name(block, ".ls2.gamma")),
                static_cast<std::uint32_t>(elements), embedding);
            std::swap(state, next);
            for (std::uint32_t capture = 0u;
                capture < config.capture_count; ++capture) {
                if (block != config.captures[capture]) continue;
                operators.layer_norm(
                    normalized, state,
                    weight(model, std::string(kPrefix) + "norm.weight"),
                    weight(model, std::string(kPrefix) + "norm.bias"),
                    tokens, embedding, 1.0e-6f);
                const std::string projection = "encoder.output_projections." +
                    std::to_string(capture);
                operators.project_tokens(
                    captured[capture], normalized,
                    weight(model, projection + ".weight"),
                    weight(model, projection + ".bias"),
                    token_width, token_height, embedding,
                    config.decoder_embedding);
            }
        }
        da3_native::VulkanBuffer accumulator = std::move(captured[0]);
        for (std::uint32_t capture = 1u;
            capture < config.capture_count; ++capture) {
            const bool last = capture + 1u == config.capture_count;
            da3_native::VulkanBuffer sum = last ? da3_native::VulkanBuffer{} :
                context.create_device_buffer(std::uint64_t(patches) *
                    config.decoder_embedding * sizeof(float));
            da3_native::VulkanBuffer& destination = last ? output.features : sum;
            operators.add(destination, accumulator, captured[capture],
                patches * config.decoder_embedding);
            if (!last) accumulator = std::move(sum);
        }
    });
    context.copy(
        output.class_token, 0u, normalized, 0u,
        embedding * sizeof(float));
    return output;
}

}  // namespace moge2_native
