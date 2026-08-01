#pragma once

#include "vulkan.h"

#include <cstdint>

namespace moge2_native {

class MoGeOperators {
public:
    explicit MoGeOperators(da3_native::VulkanContext& context);

    void conv2d_replicate(
        da3_native::VulkanBuffer& output,
        const da3_native::VulkanBuffer& input,
        const da3_native::VulkanBuffer& weight,
        const da3_native::VulkanBuffer& bias,
        std::uint32_t width, std::uint32_t height,
        std::uint32_t input_channels, std::uint32_t output_channels,
        std::uint32_t kernel, std::uint32_t padding);
    void bilinear(
        da3_native::VulkanBuffer& output,
        const da3_native::VulkanBuffer& input,
        std::uint32_t input_width, std::uint32_t input_height,
        std::uint32_t output_width, std::uint32_t output_height,
        std::uint32_t channels);
    void concat_uv(
        da3_native::VulkanBuffer& output,
        const da3_native::VulkanBuffer& input,
        std::uint32_t width, std::uint32_t height,
        std::uint32_t input_channels, float aspect_ratio);
    void remap_points_mask(
        da3_native::VulkanBuffer& points_output,
        da3_native::VulkanBuffer& mask_output,
        const da3_native::VulkanBuffer& points_input,
        const da3_native::VulkanBuffer& mask_input,
        std::uint32_t pixels);
    void solve_focal_shift(
        da3_native::VulkanBuffer& result,
        const da3_native::VulkanBuffer& points,
        const da3_native::VulkanBuffer& mask,
        std::uint32_t width, std::uint32_t height);
    void final_depth(
        da3_native::VulkanBuffer& depth,
        const da3_native::VulkanBuffer& points,
        const da3_native::VulkanBuffer& mask,
        const da3_native::VulkanBuffer& focal_shift,
        const da3_native::VulkanBuffer& metric_scale,
        std::uint32_t pixels);
    void final_depth_image(
        da3_native::VulkanImage& depth,
        const da3_native::VulkanBuffer& points,
        const da3_native::VulkanBuffer& mask,
        const da3_native::VulkanBuffer& focal_shift,
        const da3_native::VulkanBuffer& metric_scale,
        std::uint32_t width, std::uint32_t height);

private:
    da3_native::VulkanContext& context_;
    da3_native::VulkanPipeline conv2d_replicate_;
    da3_native::VulkanPipeline bilinear_;
    da3_native::VulkanPipeline concat_uv_;
    da3_native::VulkanPipeline remap_points_mask_;
    da3_native::VulkanPipeline solve_focal_shift_;
    da3_native::VulkanPipeline final_depth_;
    da3_native::VulkanPipeline final_depth_image_;
};

}  // namespace moge2_native
