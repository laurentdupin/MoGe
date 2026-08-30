#include "external_gpu.h"

#include "inferbridge/native_harness_precision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

bool compare(const char* model_path, inferbridge::native::Precision precision) {
    std::shared_ptr<moge2_native::ExternalGpu> vulkan;
    std::shared_ptr<moge2_native::ExternalGpu> metal;
    {
        const inferbridge::native::ScopedPrecisionRequest scope(precision);
        vulkan = moge2_native::create_external_gpu(model_path, 0u);
        metal = moge2_native::create_metal_gpu(model_path);
    }
    constexpr std::uint32_t width = 96u;
    constexpr std::uint32_t height = 64u;
    std::vector<std::uint8_t> pixels(width * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t offset =
                (static_cast<std::size_t>(y) * width + x) * 4u;
            pixels[offset] = static_cast<std::uint8_t>((x * 11u + y) & 255u);
            pixels[offset + 1u] = static_cast<std::uint8_t>((x + y * 7u) & 255u);
            pixels[offset + 2u] = static_cast<std::uint8_t>((x * 3u + y * 5u) & 255u);
            pixels[offset + 3u] = 255u;
        }
    std::vector<float> reference(width * height);
    std::vector<float> output(width * height);
    vulkan->infer_host(pixels.data(), width, height, width * 4u, false,
        64u, 50.0f, reference.data());
    metal->infer_host(pixels.data(), width, height, width * 4u, false,
        64u, 50.0f, output.data());
    double error = 0.0;
    double magnitude = 0.0;
    float maximum = 0.0f;
    float minimum_output = INFINITY;
    float maximum_output = -INFINITY;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) return false;
        const float difference = std::abs(output[i] - reference[i]);
        error += difference;
        magnitude += std::abs(reference[i]);
        maximum = std::max(maximum, difference);
        minimum_output = std::min(minimum_output, output[i]);
        maximum_output = std::max(maximum_output, output[i]);
    }
    const double relative = error / std::max(magnitude, 1.0e-30);
    std::printf(
        "precision=%s relative_l1=%.9g maximum_absolute=%.9g range=%.9g\n",
        precision == inferbridge::native::Precision::fp16 ? "fp16" : "fp32",
        relative, maximum, maximum_output - minimum_output);
    return maximum_output > minimum_output && relative <= 0.02;
}

int main() {
    const char* model_path = std::getenv("MOGE2_MODEL");
    if (!model_path || !*model_path) return 77;
    return compare(model_path, inferbridge::native::Precision::fp32) &&
        compare(model_path, inferbridge::native::Precision::fp16) ? 0 : 1;
}
