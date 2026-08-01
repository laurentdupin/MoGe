#pragma once

#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <array>

namespace moge2_native {

struct DepthOutput {
    std::uint32_t width;
    std::uint32_t height;
    da3_native::VulkanBuffer depth;
    da3_native::VulkanBuffer metric_scale;
    da3_native::VulkanBuffer focal_shift;
    da3_native::VulkanBuffer points;
    da3_native::VulkanBuffer mask;
    std::array<da3_native::VulkanBuffer, 5> neck_features;
    da3_native::VulkanBuffer points_low;
    da3_native::VulkanBuffer mask_low;
};

DepthOutput infer_vits_normal(
    da3_native::VulkanContext& context,
    da3_native::GpuModel& model,
    da3_native::VulkanOperators& operators,
    da3_native::VulkanBuffer normalized_encoder_image,
    std::uint32_t encoder_width,
    std::uint32_t encoder_height,
    std::uint32_t output_width,
    std::uint32_t output_height);

}  // namespace moge2_native
