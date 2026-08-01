#include "moge_operators.h"

#include "bilinear_align_false_spv.h"
#include "concat_uv_spv.h"
#include "conv2d_replicate_spv.h"
#include "final_depth_spv.h"
#include "remap_points_mask_spv.h"
#include "solve_focal_shift_spv.h"

#include <stdexcept>

namespace moge2_native {
namespace {
std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}
}

MoGeOperators::MoGeOperators(da3_native::VulkanContext& context)
    : context_(context),
      conv2d_replicate_(context.create_pipeline(
          da3_conv2d_replicate_spv, da3_conv2d_replicate_spv_size, 4, 48)),
      bilinear_(context.create_pipeline(
          da3_bilinear_align_false_spv, da3_bilinear_align_false_spv_size, 2, 24)),
      concat_uv_(context.create_pipeline(
          da3_concat_uv_spv, da3_concat_uv_spv_size, 2, 16)),
      remap_points_mask_(context.create_pipeline(
          da3_remap_points_mask_spv, da3_remap_points_mask_spv_size, 4, 4)),
      solve_focal_shift_(context.create_pipeline(
          da3_solve_focal_shift_spv, da3_solve_focal_shift_spv_size, 3, 8)),
      final_depth_(context.create_pipeline(
          da3_final_depth_spv, da3_final_depth_spv_size, 5, 4)) {
    conv2d_replicate_.set_debug_name("moge2_conv2d_replicate");
    bilinear_.set_debug_name("moge2_bilinear_align_false");
    concat_uv_.set_debug_name("moge2_concat_uv");
    remap_points_mask_.set_debug_name("moge2_remap_points_mask");
    solve_focal_shift_.set_debug_name("moge2_solve_focal_shift");
    final_depth_.set_debug_name("moge2_final_depth");
}

void MoGeOperators::conv2d_replicate(
    da3_native::VulkanBuffer& output,
    const da3_native::VulkanBuffer& input,
    const da3_native::VulkanBuffer& weight,
    const da3_native::VulkanBuffer& bias,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t input_channels, std::uint32_t output_channels,
    std::uint32_t kernel, std::uint32_t padding) {
    struct Parameters {
        std::uint32_t input_width, input_height, input_channels;
        std::uint32_t output_width, output_height, output_channels;
        std::uint32_t kernel, stride;
        std::int32_t padding;
        std::uint32_t has_bias, batches, output_channel_blocks;
    } parameters{width, height, input_channels, width, height,
        output_channels, kernel, 1u, static_cast<std::int32_t>(padding),
        1u, 1u, divide_up(output_channels, 4u)};
    context_.dispatch(conv2d_replicate_, {&output, &input, &weight, &bias},
        &parameters, sizeof(parameters), divide_up(width, 8u),
        divide_up(height, 8u), parameters.output_channel_blocks);
}

void MoGeOperators::bilinear(
    da3_native::VulkanBuffer& output,
    const da3_native::VulkanBuffer& input,
    std::uint32_t input_width, std::uint32_t input_height,
    std::uint32_t output_width, std::uint32_t output_height,
    std::uint32_t channels) {
    struct Parameters { std::uint32_t iw, ih, ow, oh, channels, batches; }
        parameters{input_width, input_height, output_width, output_height,
            channels, 1u};
    context_.dispatch(bilinear_, {&output, &input}, &parameters,
        sizeof(parameters), divide_up(output_width, 8u),
        divide_up(output_height, 8u), channels);
}

void MoGeOperators::concat_uv(
    da3_native::VulkanBuffer& output,
    const da3_native::VulkanBuffer& input,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t input_channels, float aspect_ratio) {
    struct Parameters { std::uint32_t width, height, channels; float aspect; }
        parameters{width, height, input_channels, aspect_ratio};
    context_.dispatch(concat_uv_, {&output, &input}, &parameters,
        sizeof(parameters), divide_up(width, 8u), divide_up(height, 8u),
        input_channels + 2u);
}

void MoGeOperators::remap_points_mask(
    da3_native::VulkanBuffer& points_output,
    da3_native::VulkanBuffer& mask_output,
    const da3_native::VulkanBuffer& points_input,
    const da3_native::VulkanBuffer& mask_input,
    std::uint32_t pixels) {
    context_.dispatch(remap_points_mask_,
        {&points_output, &mask_output, &points_input, &mask_input},
        &pixels, sizeof(pixels), divide_up(pixels, 256u), 1u, 1u);
}

void MoGeOperators::solve_focal_shift(
    da3_native::VulkanBuffer& result,
    const da3_native::VulkanBuffer& points,
    const da3_native::VulkanBuffer& mask,
    std::uint32_t width, std::uint32_t height) {
    const std::uint32_t parameters[2]{width, height};
    context_.dispatch(solve_focal_shift_, {&result, &points, &mask},
        parameters, sizeof(parameters), 1u, 1u, 1u);
}

void MoGeOperators::final_depth(
    da3_native::VulkanBuffer& depth,
    const da3_native::VulkanBuffer& points,
    const da3_native::VulkanBuffer& mask,
    const da3_native::VulkanBuffer& focal_shift,
    const da3_native::VulkanBuffer& metric_scale,
    std::uint32_t pixels) {
    context_.dispatch(final_depth_,
        {&depth, &points, &mask, &focal_shift, &metric_scale},
        &pixels, sizeof(pixels), divide_up(pixels, 256u), 1u, 1u);
}

}  // namespace moge2_native
