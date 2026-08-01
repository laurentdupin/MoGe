#pragma once

#include "safetensors.h"

#include <array>
#include <cstdint>

namespace moge2_native {

struct ModelConfig {
    std::uint32_t embedding = 0u;
    std::uint32_t heads = 0u;
    std::uint32_t blocks = 0u;
    std::uint32_t decoder_embedding = 0u;
    std::uint32_t capture_count = 0u;
    std::array<std::uint32_t, 4> captures{};
    std::uint32_t neck_residual_blocks = 1u;
    std::uint32_t head_residual_blocks = 1u;
};

ModelConfig read_model_config(const da3_native::SafeTensors& model);

}  // namespace moge2_native
