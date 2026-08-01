#include "gpu_preprocess.h"

#include "preprocess_moge_texture_spv.h"

#include <stdexcept>

namespace moge2_native {

GpuPreprocessor::GpuPreprocessor(da3_native::VulkanContext& context)
    : context_(context),
      pipeline_(context.create_pipeline(
          da3_preprocess_moge_texture_spv,
          da3_preprocess_moge_texture_spv_size,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}, 8)) {
    pipeline_.set_debug_name("moge2_preprocess_texture");
}

void GpuPreprocessor::run_texture(
    da3_native::VulkanBuffer& destination,
    const da3_native::VulkanImage& source,
    std::uint32_t width, std::uint32_t height) {
    if (!width || !height || destination.size() <
        std::uint64_t(width) * height * 3u * sizeof(float))
        throw std::invalid_argument("invalid MoGe-2 preprocessing dimensions");
    const std::uint32_t parameters[2]{width, height};
    context_.dispatch_image_to_buffer(pipeline_, source, destination,
        parameters, sizeof(parameters), (width + 7u) / 8u,
        (height + 7u) / 8u, 1u);
}

}  // namespace moge2_native
