#pragma once

#include "vulkan.h"

#include <cstdint>

namespace moge2_native {

class GpuPreprocessor {
public:
    explicit GpuPreprocessor(da3_native::VulkanContext& context);
    void run_texture(
        da3_native::VulkanBuffer& destination,
        const da3_native::VulkanImage& source,
        std::uint32_t width, std::uint32_t height);
private:
    da3_native::VulkanContext& context_;
    da3_native::VulkanPipeline pipeline_;
};

}  // namespace moge2_native
