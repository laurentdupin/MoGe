#include "graph_gpu.h"

#include "encoder_gpu.h"
#include "moge_operators.h"

#include <inferbridge/native_harness_profile.h>

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace moge2_native {
namespace {
using da3_native::GpuModel;
using da3_native::VulkanBuffer;
using da3_native::VulkanContext;
using da3_native::VulkanOperators;

std::array<std::uint32_t, 5> decoder_channels(const ModelConfig& config) {
    return {config.decoder_embedding, 256u, 128u, 64u, 32u};
}

const VulkanBuffer& tensor(const GpuModel& model, const std::string& name) {
    return model.tensor(name).buffer;
}

const VulkanBuffer& weight(const GpuModel& model, const std::string& name) {
    const da3_native::GpuTensor& value = model.tensor(name);
    return value.half_buffer.handle() != VK_NULL_HANDLE
        ? value.half_buffer : value.buffer;
}

bool half_weight(const GpuModel& model, const std::string& name) {
    return model.tensor(name).half_buffer.handle() != VK_NULL_HANDLE;
}

VulkanBuffer allocate(
    VulkanContext& context, std::uint32_t width, std::uint32_t height,
    std::uint32_t channels) {
    return context.create_device_buffer(
        std::uint64_t(width) * height * channels * sizeof(float));
}

VulkanBuffer conv(
    VulkanContext& context, VulkanOperators& operators,
    const GpuModel& model, const VulkanBuffer& input,
    const std::string& prefix, std::uint32_t width, std::uint32_t height,
    std::uint32_t input_channels, std::uint32_t output_channels,
    std::uint32_t kernel = 1u, std::uint32_t padding = 0u) {
    VulkanBuffer output = allocate(context, width, height, output_channels);
    const std::string weight_name = prefix + ".weight";
    operators.conv2d(output, input, weight(model, weight_name),
        tensor(model, prefix + ".bias"), width, height, input_channels,
        output_channels, kernel, 1u, padding, true, false,
        half_weight(model, weight_name));
    return output;
}

VulkanBuffer residual(
    VulkanContext& context, VulkanOperators& operators,
    MoGeOperators& moge, const GpuModel& model, const VulkanBuffer& input,
    const std::string& prefix, std::uint32_t width, std::uint32_t height,
    std::uint32_t channels) {
    const std::uint32_t count = width * height * channels;
    VulkanBuffer activated = allocate(context, width, height, channels);
    VulkanBuffer hidden = allocate(context, width, height, channels);
    VulkanBuffer second_activated = allocate(context, width, height, channels);
    VulkanBuffer transformed = allocate(context, width, height, channels);
    VulkanBuffer output = allocate(context, width, height, channels);
    operators.relu(activated, input, count);
    moge.conv2d_replicate(hidden, activated,
        tensor(model, prefix + ".layers.2.weight"),
        tensor(model, prefix + ".layers.2.bias"),
        width, height, channels, channels, 3u, 1u);
    operators.relu(second_activated, hidden, count);
    moge.conv2d_replicate(transformed, second_activated,
        tensor(model, prefix + ".layers.5.weight"),
        tensor(model, prefix + ".layers.5.bias"),
        width, height, channels, channels, 3u, 1u);
    operators.add(output, input, transformed, count);
    return output;
}

VulkanBuffer resample(
    VulkanContext& context, VulkanOperators& operators,
    MoGeOperators& moge, const GpuModel& model, const VulkanBuffer& input,
    const std::string& stack, std::uint32_t level,
    std::uint32_t width, std::uint32_t height,
    const std::array<std::uint32_t, 5>& channels) {
    const std::uint32_t intermediate_channels =
        level < 3u ? channels[level + 1u] : channels[level];
    VulkanBuffer upsampled = allocate(
        context, width * 2u, height * 2u, intermediate_channels);
    if (level < 3u) {
        operators.conv_transpose_nonoverlap(upsampled, input,
            weight(model, stack + ".resamplers." + std::to_string(level) + ".0.weight"),
            tensor(model, stack + ".resamplers." + std::to_string(level) + ".0.bias"),
            width, height, channels[level], channels[level + 1u], 2u,
            half_weight(model, stack + ".resamplers." + std::to_string(level) + ".0.weight"));
    } else {
        moge.bilinear(upsampled, input, width, height,
            width * 2u, height * 2u, channels[level]);
    }
    VulkanBuffer output = allocate(
        context, width * 2u, height * 2u, channels[level + 1u]);
    moge.conv2d_replicate(output, upsampled,
        tensor(model, stack + ".resamplers." + std::to_string(level) + ".1.weight"),
        tensor(model, stack + ".resamplers." + std::to_string(level) + ".1.bias"),
        width * 2u, height * 2u, intermediate_channels,
        channels[level + 1u], 3u, 1u);
    return output;
}

std::array<VulkanBuffer, 5> neck(
    VulkanContext& context, VulkanOperators& operators, MoGeOperators& moge,
    const GpuModel& model, const VulkanBuffer& encoder,
    std::uint32_t token_width, std::uint32_t token_height, float aspect,
    const std::array<std::uint32_t, 5>& channels,
    std::uint32_t residual_blocks) {
    std::array<VulkanBuffer, 5> outputs;
    VulkanBuffer carried;
    for (std::uint32_t level = 0; level < 5u; ++level) {
        const std::uint32_t width = token_width << level;
        const std::uint32_t height = token_height << level;
        const std::uint32_t input_channels = level == 0u ? channels[0] : 0u;
        VulkanBuffer uv = allocate(context, width, height, input_channels + 2u);
        moge.concat_uv(uv, encoder, width, height, input_channels, aspect);
        VulkanBuffer feature = conv(context, operators, model, uv,
            "neck.input_blocks." + std::to_string(level), width, height,
            input_channels + 2u, channels[level]);
        VulkanBuffer state;
        if (level == 0u) {
            state = std::move(feature);
        } else {
            state = allocate(context, width, height, channels[level]);
            operators.add(state, carried, feature,
                width * height * channels[level]);
        }
        if (level >= 1u && level <= 3u) {
            for (std::uint32_t block = 0u; block < residual_blocks; ++block) {
                state = residual(context, operators, moge, model, state,
                    "neck.res_blocks." + std::to_string(level) + "." +
                        std::to_string(block),
                    width, height, channels[level]);
            }
        }
        outputs[level] = std::move(state);
        if (level < 4u) {
            carried = resample(context, operators, moge, model, outputs[level],
                "neck", level, width, height, channels);
        }
    }
    return outputs;
}

VulkanBuffer head(
    VulkanContext& context, VulkanOperators& operators, MoGeOperators& moge,
    const GpuModel& model, const std::array<VulkanBuffer, 5>& features,
    const std::string& stack, std::uint32_t token_width,
    std::uint32_t token_height, std::uint32_t output_channels,
    const std::array<std::uint32_t, 5>& channels,
    std::uint32_t residual_blocks) {
    VulkanBuffer carried;
    VulkanBuffer state;
    for (std::uint32_t level = 0; level < 5u; ++level) {
        const std::uint32_t width = token_width << level;
        const std::uint32_t height = token_height << level;
        VulkanBuffer feature = conv(context, operators, model, features[level],
            stack + ".input_blocks." + std::to_string(level), width, height,
            channels[level], channels[level]);
        if (level == 0u) {
            state = std::move(feature);
        } else {
            state = allocate(context, width, height, channels[level]);
            operators.add(state, carried, feature,
                width * height * channels[level]);
        }
        if (level >= 1u && level <= 3u) {
            for (std::uint32_t block = 0u; block < residual_blocks; ++block) {
                state = residual(context, operators, moge, model, state,
                    stack + ".res_blocks." + std::to_string(level) + "." +
                        std::to_string(block),
                    width, height, channels[level]);
            }
        }
        if (level < 4u) {
            carried = resample(context, operators, moge, model, state,
                stack, level, width, height, channels);
        }
    }
    return conv(context, operators, model, state, stack + ".output_blocks.4",
        token_width * 16u, token_height * 16u, 32u, output_channels);
}

VulkanBuffer metric_scale(
    VulkanContext& context, VulkanOperators& operators,
    const GpuModel& model, const VulkanBuffer& class_token,
    std::uint32_t embedding) {
    VulkanBuffer a = context.create_device_buffer(embedding * sizeof(float));
    VulkanBuffer b = context.create_device_buffer(embedding * sizeof(float));
    VulkanBuffer c = context.create_device_buffer(embedding * sizeof(float));
    VulkanBuffer d = context.create_device_buffer(embedding * sizeof(float));
    VulkanBuffer result = context.create_device_buffer(sizeof(float));
    operators.linear(a, class_token, weight(model, "scale_head.0.weight"),
        tensor(model, "scale_head.0.bias"), 1u, embedding, embedding, false,
        false, half_weight(model, "scale_head.0.weight"));
    operators.relu(b, a, embedding);
    operators.linear(c, b, weight(model, "scale_head.2.weight"),
        tensor(model, "scale_head.2.bias"), 1u, embedding, embedding, false,
        false, half_weight(model, "scale_head.2.weight"));
    operators.relu(d, c, embedding);
    operators.linear(result, d, weight(model, "scale_head.4.weight"),
        tensor(model, "scale_head.4.bias"), 1u, embedding, 1u, false,
        false, half_weight(model, "scale_head.4.weight"));
    operators.exponential(result, 1u);
    return result;
}
}  // namespace

DepthOutput infer_vits_normal(
    VulkanContext& context, GpuModel& model, VulkanOperators& operators,
    MoGeOperators& moge, const ModelConfig& config,
    VulkanBuffer normalized_encoder_image,
    std::uint32_t encoder_width,
    std::uint32_t encoder_height, std::uint32_t output_width,
    std::uint32_t output_height, float background_distance_metres,
    da3_native::VulkanImage* output_image) {
    if (!output_width || !output_height) throw std::invalid_argument("invalid output shape");
    EncoderOutput encoded;
    {
        inferbridge::native::ProfileStage stage(
            "INFERBRIDGE_MOGE_PROFILE", "moge", "encoder");
        encoded = encode_vits(context, model, operators, config,
            std::move(normalized_encoder_image), encoder_width, encoder_height);
    }
    const auto channels = decoder_channels(config);
    const float aspect = float(output_width) / float(output_height);
    const std::uint32_t pixels = output_width * output_height;
    DepthOutput output{output_width, output_height,
        output_image == nullptr ? context.create_device_buffer(
            std::uint64_t(pixels) * sizeof(float)) : VulkanBuffer{},
        {}, context.create_device_buffer(2u * sizeof(float)), {}, {}, {}, {}, {}};
    VulkanBuffer scale;
    VulkanBuffer points;
    VulkanBuffer mask;
    std::array<VulkanBuffer, 5> features;
    VulkanBuffer points_low;
    VulkanBuffer mask_low;
    {
        inferbridge::native::ProfileStage stage(
            "INFERBRIDGE_MOGE_PROFILE", "moge", "neck");
        context.batch([&] {
            features = neck(context, operators, moge, model, encoded.features,
                encoded.token_width, encoded.token_height, aspect, channels,
                config.neck_residual_blocks);
        });
    }
    {
        inferbridge::native::ProfileStage stage(
            "INFERBRIDGE_MOGE_PROFILE", "moge", "points_head");
        context.batch([&] {
            points_low = head(context, operators, moge, model, features,
                "points_head", encoded.token_width, encoded.token_height, 3u,
                channels, config.head_residual_blocks);
        });
    }
    {
        inferbridge::native::ProfileStage stage(
            "INFERBRIDGE_MOGE_PROFILE", "moge", "mask_head");
        context.batch([&] {
            mask_low = head(context, operators, moge, model, features,
                "mask_head", encoded.token_width, encoded.token_height, 1u,
                channels, config.head_residual_blocks);
        });
    }
    {
        inferbridge::native::ProfileStage stage(
            "INFERBRIDGE_MOGE_PROFILE", "moge", "resize_remap");
        context.batch([&] {
            VulkanBuffer points_resized = allocate(context, output_width, output_height, 3u);
            VulkanBuffer mask_resized = allocate(context, output_width, output_height, 1u);
            moge.bilinear(points_resized, points_low,
                encoded.token_width * 16u, encoded.token_height * 16u,
                output_width, output_height, 3u);
            moge.bilinear(mask_resized, mask_low,
                encoded.token_width * 16u, encoded.token_height * 16u,
                output_width, output_height, 1u);
            points = allocate(context, output_width, output_height, 3u);
            mask = allocate(context, output_width, output_height, 1u);
            moge.remap_points_mask(points, mask, points_resized, mask_resized, pixels);
        });
    }
    {
        inferbridge::native::ProfileStage stage(
            "INFERBRIDGE_MOGE_PROFILE", "moge", "metric_solve");
        context.batch([&] {
            scale = metric_scale(context, operators, model, encoded.class_token,
                config.embedding);
            moge.solve_focal_shift(
                output.focal_shift, points, mask, output_width, output_height);
        });
    }
    {
        inferbridge::native::ProfileStage stage(
            "INFERBRIDGE_MOGE_PROFILE", "moge", "final_depth");
        context.batch([&] {
            if (output_image == nullptr) {
                moge.final_depth(output.depth, points, mask, output.focal_shift,
                    scale, pixels, background_distance_metres);
            } else {
                moge.final_depth_image(*output_image, points, mask,
                    output.focal_shift, scale, output_width, output_height,
                    background_distance_metres);
            }
        });
    }
    output.neck_features = std::move(features);
    output.points_low = std::move(points_low);
    output.mask_low = std::move(mask_low);
    output.metric_scale = std::move(scale);
    output.points = std::move(points);
    output.mask = std::move(mask);
    return output;
}

}  // namespace moge2_native
