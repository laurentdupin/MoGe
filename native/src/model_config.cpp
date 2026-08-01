#include "model_config.h"

#include <cmath>
#include <stdexcept>

namespace moge2_native {

ModelConfig read_model_config(const da3_native::SafeTensors& model) {
    const auto& tensor = model.tensor("moge2.config.encoder");
    if (tensor.rank != 1u || tensor.elements != 5u)
        throw std::runtime_error("invalid MoGe-2 encoder configuration");
    ModelConfig result{};
    auto value = [&](std::uint32_t index) {
        const float raw = tensor.data[index];
        if (!std::isfinite(raw) || raw < 0.0f || raw > 4096.0f ||
            std::floor(raw) != raw)
            throw std::runtime_error("invalid MoGe-2 encoder configuration value");
        return static_cast<std::uint32_t>(raw);
    };
    result.embedding = value(0u);
    result.heads = value(1u);
    result.blocks = value(2u);
    result.capture_first = value(3u);
    result.capture_second = value(4u);
    if (!result.embedding || !result.heads || result.embedding / result.heads != 64u ||
        !result.blocks || result.capture_first >= result.capture_second ||
        result.capture_second >= result.blocks)
        throw std::runtime_error("unsupported MoGe-2 encoder configuration");
    return result;
}

}  // namespace moge2_native
