#pragma once

#include "safetensors.h"

#include <cstdint>

namespace moge2_native {

struct ModelConfig {
    std::uint32_t embedding = 0u;
    std::uint32_t heads = 0u;
    std::uint32_t blocks = 0u;
    std::uint32_t capture_first = 0u;
    std::uint32_t capture_second = 0u;
};

ModelConfig read_model_config(const da3_native::SafeTensors& model);

}  // namespace moge2_native
