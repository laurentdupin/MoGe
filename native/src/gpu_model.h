#pragma once

#include "safetensors.h"
#include "vulkan.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace da3_native {

struct GpuTensor {
    VulkanBuffer buffer;
    VulkanBuffer half_buffer;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

class GpuModel {
public:
    GpuModel(const SafeTensors& model, VulkanContext& context);

    const GpuTensor& tensor(std::string_view name) const;
    const VulkanBuffer& zero_bias() const { return zero_bias_; }
    void retain_transformer_precision(bool half_weight);
    void retain_dpt_precision(bool half_weight);
    std::size_t tensor_count() const { return tensors_.size(); }

private:
    VulkanContext& context_;
    VulkanBuffer zero_bias_;
    std::unordered_map<std::string_view, GpuTensor> tensors_;
};

}  // namespace da3_native
