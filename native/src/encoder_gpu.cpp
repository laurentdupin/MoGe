#include "encoder_gpu.h"
#include "inferbridge/native_harness_environment.h"
#include "inferbridge/native_harness_precision.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace moge2_native {
namespace {
constexpr const char* kPrefix = "encoder.backbone.";

const da3_native::VulkanBuffer& weight(
    const da3_native::GpuModel& model, const std::string& name,
    bool half = false) {
    const da3_native::GpuTensor& tensor = model.tensor(name);
    return half ? tensor.half_buffer : tensor.buffer;
}

std::string block_name(std::uint32_t block, const char* suffix) {
    return std::string(kPrefix) + "blocks." + std::to_string(block) + suffix;
}

void transformer_linear(
    da3_native::VulkanOperators& operators,
    const da3_native::GpuModel& model,
    da3_native::VulkanBuffer& output,
    const da3_native::VulkanBuffer& input,
    const std::string& weight_name, const std::string& bias_name,
    std::uint32_t rows, std::uint32_t input_columns,
    std::uint32_t output_columns, bool gelu = false) {
    const auto& tensor = model.tensor(weight_name);
    if (model.uses_int8_weights()) {
        operators.linear_int8(
            output, input, tensor.int8_buffer, tensor.int8_scales,
            model.tensor(bias_name).buffer,
            rows, input_columns, output_columns, gelu);
    } else {
        operators.linear(
            output, input,
            model.uses_half_weights() ? tensor.half_buffer : tensor.buffer,
            model.tensor(bias_name).buffer,
            rows, input_columns, output_columns, gelu, true,
            model.uses_half_weights());
    }
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
    const bool half_weight = model.uses_half_weights();
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
    da3_native::VulkanBuffer hidden = context.create_device_buffer(bytes * 4u);
    const bool alias_qkv =
        inferbridge::native_harness::scratch_aliasing_enabled();
    da3_native::VulkanBuffer qkv_storage = alias_qkv
        ? da3_native::VulkanBuffer{}
        : context.create_device_buffer(bytes * 3u);
    da3_native::VulkanBuffer& qkv = alias_qkv ? hidden : qkv_storage;
    da3_native::VulkanBuffer scores = context.create_device_buffer(
        std::uint64_t(config.heads) * tokens * tokens * sizeof(float));
    EncoderOutput output{
        token_width, token_height,
        context.create_device_buffer(
            std::uint64_t(patches) * config.decoder_embedding * sizeof(float)),
        context.create_device_buffer(embedding * sizeof(float))};
    std::uint32_t captured_count = 0u;

    const auto prepare = [&] {
        operators.prepare_tokens(
            state, image,
            weight(model, std::string(kPrefix) + "patch_embed.proj.weight"),
            weight(model, std::string(kPrefix) + "patch_embed.proj.bias"),
            weight(model, std::string(kPrefix) + "cls_token"),
            weight(model, std::string(kPrefix) + "pos_embed"),
            width, height, embedding);
    };
    const auto run_block = [&](std::uint32_t block) {
            operators.layer_norm(
                normalized, state,
                weight(model, block_name(block, ".norm1.weight")),
                weight(model, block_name(block, ".norm1.bias")),
                tokens, embedding, 1.0e-6f);
            transformer_linear(
                operators, model, qkv, normalized,
                block_name(block, ".attn.qkv.weight"),
                block_name(block, ".attn.qkv.bias"),
                tokens, embedding, embedding * 3u);
            operators.attention_head64(
                attended, qkv, tokens, config.heads, &scores);
            transformer_linear(
                operators, model, projected, attended,
                block_name(block, ".attn.proj.weight"),
                block_name(block, ".attn.proj.bias"),
                tokens, embedding, embedding);
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
            transformer_linear(
                operators, model, hidden, normalized,
                block_name(block, ".mlp.fc1.weight"),
                block_name(block, ".mlp.fc1.bias"),
                tokens, embedding, embedding * 4u, true);
            transformer_linear(
                operators, model, projected, hidden,
                block_name(block, ".mlp.fc2.weight"),
                block_name(block, ".mlp.fc2.bias"),
                tokens, embedding * 4u, embedding);
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
                const bool accumulate = captured_count++ != 0u;
                operators.project_tokens(
                    output.features, normalized,
                    weight(model, projection + ".weight"),
                    weight(model, projection + ".bias"),
                    token_width, token_height, embedding,
                    config.decoder_embedding, false, 1u, accumulate);
            }
    };
#if defined(__ANDROID__)
    // Keep each transformer block in a bounded submission. A single command
    // buffer for the complete encoder exceeds the Adreno watchdog window at
    // MoGe's default token count and the driver kills the queue.
    context.batch(prepare);
    for (std::uint32_t block = 0; block < config.blocks; ++block) {
        context.batch([&] { run_block(block); });
    }
#else
    // Desktop drivers do not have the Adreno watchdog constraint. One command
    // buffer avoids a CPU/GPU round trip for every transformer block.
    context.batch([&] {
        prepare();
        for (std::uint32_t block = 0; block < config.blocks; ++block) {
            run_block(block);
        }
    });
#endif
    context.copy(
        output.class_token, 0u, normalized, 0u,
        embedding * sizeof(float));
    model.retain_transformer_precision(half_weight);
    return output;
}

}  // namespace moge2_native
