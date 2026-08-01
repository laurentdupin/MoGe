#pragma once

#include "gpu_model.h"
#include "operators.h"

#include <cstdint>

namespace moge2_native {

struct EncoderOutput {
    std::uint32_t token_width = 0;
    std::uint32_t token_height = 0;
    da3_native::VulkanBuffer features;
    da3_native::VulkanBuffer class_token;
};

EncoderOutput encode_vits(
    da3_native::VulkanContext& context,
    da3_native::GpuModel& model,
    da3_native::VulkanOperators& operators,
    da3_native::VulkanBuffer image,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace moge2_native
