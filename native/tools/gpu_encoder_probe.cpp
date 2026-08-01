#include "encoder_gpu.h"
#include "gpu_model.h"
#include "operators.h"
#include "safetensors.h"
#include "vulkan.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read(const char* path, std::size_t count) {
    std::vector<float> values(count);
    std::ifstream source(path, std::ios::binary);
    source.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!source || source.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("invalid probe input");
    return values;
}
void write(const char* path, const std::vector<float>& values) {
    std::ofstream destination(path, std::ios::binary);
    destination.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!destination) throw std::runtime_error("failed to write probe output");
}
}

int main(int argc, char** argv) try {
    if (argc != 8) return 2;
    const auto width = static_cast<std::uint32_t>(std::stoul(argv[5]));
    const auto height = static_cast<std::uint32_t>(std::stoul(argv[6]));
    const auto device = static_cast<std::uint32_t>(std::stoul(argv[7]));
    auto image = read(argv[2], std::uint64_t(width) * height * 3u);
    da3_native::SafeTensors weights(argv[1]);
    da3_native::VulkanContext context(device);
    da3_native::GpuModel model(weights, context);
    da3_native::VulkanOperators operators(context);
    auto input = context.create_device_buffer(image.size() * sizeof(float));
    context.upload(input, image.data(), image.size() * sizeof(float));
    auto output = moge2_native::encode_vits(
        context, model, operators, std::move(input), width, height);
    std::vector<float> features(
        std::uint64_t(output.token_width) * output.token_height * 384u);
    std::vector<float> class_token(384u);
    context.download(output.features, features.data(),
        features.size() * sizeof(float));
    context.download(output.class_token, class_token.data(),
        class_token.size() * sizeof(float));
    write(argv[3], features);
    write(argv[4], class_token);
    return 0;
} catch (const std::exception& exception) {
    std::fprintf(stderr, "%s\n", exception.what());
    return 1;
}
