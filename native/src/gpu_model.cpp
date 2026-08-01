#include "gpu_model.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace da3_native {
namespace {

bool use_half_weight(std::string_view name) {
    constexpr std::string_view suffix = ".weight";
    const bool weight =
        name.size() >= suffix.size() &&
        name.substr(name.size() - suffix.size()) == suffix;
    if (!weight) {
        return false;
    }
    if (name.rfind("depth_head.", 0) == 0) {
        return true;
    }
    if (name.rfind("pretrained.blocks.", 0) != 0) {
        return false;
    }
    return name.find(".attn.") != std::string_view::npos ||
        name.find(".mlp.") != std::string_view::npos;
}

std::uint16_t float_to_half(float input) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &input, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<std::uint16_t>(
            sign | (mantissa != 0 ? 0x7e00u : 0x7c00u));
    }
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const std::uint32_t shift =
            static_cast<std::uint32_t>(14 - half_exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u) != 0)) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);
    }
    std::uint32_t rounded_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded_mantissa & 1u) != 0)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0;
            if (half_exponent + 1 >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
            return static_cast<std::uint16_t>(
                sign | ((half_exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(half_exponent) << 10) |
        rounded_mantissa);
}

}  // namespace

GpuModel::GpuModel(const SafeTensors& model, VulkanContext& context)
    : context_(context) {
    zero_bias_ = context_.create_device_buffer(sizeof(float));
    const float zero = 0.0f;
    context_.upload(zero_bias_, &zero, sizeof(zero));
    tensors_.reserve(model.tensor_count());
    for (std::string_view name : model.tensor_names()) {
        const TensorView& source = model.tensor(name);
        if (source.elements >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error(
                "model tensor is too large for this process: " +
                std::string(name));
        }
        const std::size_t bytes =
            static_cast<std::size_t>(source.elements) * sizeof(float);
        GpuTensor destination{
            context.create_device_buffer(bytes),
            {},
            source.dimensions,
            source.rank,
            source.elements,
        };
        context.upload(destination.buffer, source.data, bytes);
        if (use_half_weight(name)) {
            const auto* floats =
                static_cast<const float*>(source.data);
            std::vector<std::uint32_t> packed(
                static_cast<std::size_t>((source.elements + 1) / 2),
                0);
            for (std::uint64_t index = 0;
                 index < source.elements;
                 ++index) {
                packed[static_cast<std::size_t>(index / 2)] |=
                    static_cast<std::uint32_t>(
                        float_to_half(floats[index])) <<
                    ((index & 1u) * 16u);
            }
            destination.half_buffer = context.create_device_buffer(
                packed.size() * sizeof(std::uint32_t));
            context.upload(
                destination.half_buffer,
                packed.data(),
                packed.size() * sizeof(std::uint32_t));
        }
        if (!tensors_.emplace(name, std::move(destination)).second) {
            throw std::runtime_error(
                "duplicate GPU tensor name: " + std::string(name));
        }
    }
}

const GpuTensor& GpuModel::tensor(std::string_view name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "GPU model is missing tensor: " + std::string(name));
    }
    return found->second;
}

void GpuModel::retain_transformer_precision(bool half_weight) {
    for (auto& entry : tensors_) {
        const std::string_view name = entry.first;
        if (name.rfind("pretrained.blocks.", 0) != 0 ||
            name.size() < 7 ||
            name.substr(name.size() - 7) != ".weight" ||
            (name.find(".attn.") == std::string_view::npos &&
             name.find(".mlp.") == std::string_view::npos)) {
            continue;
        }
        GpuTensor& tensor = entry.second;
        context_.discard(
            half_weight ? tensor.buffer : tensor.half_buffer);
    }
}

void GpuModel::retain_dpt_precision(bool half_weight) {
    for (auto& entry : tensors_) {
        const std::string_view name = entry.first;
        if (name.rfind("depth_head.", 0) != 0 ||
            name.size() < 7 ||
            name.substr(name.size() - 7) != ".weight") {
            continue;
        }
        GpuTensor& tensor = entry.second;
        context_.discard(
            half_weight ? tensor.buffer : tensor.half_buffer);
    }
}

}  // namespace da3_native
