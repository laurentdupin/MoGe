#include "model_config.h"

#include <cmath>
#include <stdexcept>

namespace moge2_native {

ModelConfig read_model_config(const da3_native::SafeTensors& model) {
    const auto& tensor = model.tensor("moge2.config.encoder");
    if (tensor.rank != 1u || (tensor.elements != 5u &&
        (tensor.elements < 9u || tensor.elements > 11u)))
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
    if (tensor.elements == 5u) {
        result.decoder_embedding = 384u;
        result.capture_count = 2u;
        result.captures[0] = value(3u);
        result.captures[1] = value(4u);
    } else {
        result.decoder_embedding = value(3u);
        result.capture_count = value(4u);
        if (!result.capture_count || result.capture_count > result.captures.size() ||
            tensor.elements != 7u + result.capture_count)
            throw std::runtime_error("invalid MoGe-2 capture configuration");
        for (std::uint32_t index = 0u; index < result.capture_count; ++index)
            result.captures[index] = value(5u + index);
        result.neck_residual_blocks = value(5u + result.capture_count);
        result.head_residual_blocks = value(6u + result.capture_count);
    }
    if (!result.embedding || !result.heads || result.embedding / result.heads != 64u ||
        !result.blocks || !result.decoder_embedding ||
        !result.neck_residual_blocks || !result.head_residual_blocks)
        throw std::runtime_error("unsupported MoGe-2 encoder configuration");
    for (std::uint32_t index = 0u; index < result.capture_count; ++index) {
        if (result.captures[index] >= result.blocks ||
            (index && result.captures[index - 1u] >= result.captures[index]))
            throw std::runtime_error("unsupported MoGe-2 capture configuration");
    }
    return result;
}

}  // namespace moge2_native
