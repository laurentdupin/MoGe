#include "gpu_model.h"
#include "graph_gpu.h"
#include "operators.h"
#include "safetensors.h"
#include "vulkan.h"

#include <cstdint>
#include <algorithm>
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
}

int main(int argc, char** argv) try {
    if (argc != 8) return 2;
    const auto width = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto height = static_cast<std::uint32_t>(std::stoul(argv[5]));
    const auto output_width = static_cast<std::uint32_t>(std::stoul(argv[6]));
    const auto output_height = static_cast<std::uint32_t>(std::stoul(argv[7]));
    auto image = read(argv[2], std::uint64_t(width) * height * 3u);
    da3_native::SafeTensors weights(argv[1]);
    da3_native::VulkanContext context(0u);
    da3_native::GpuModel model(weights, context);
    da3_native::VulkanOperators operators(context);
    auto input = context.create_device_buffer(image.size() * sizeof(float));
    context.upload(input, image.data(), image.size() * sizeof(float));
    auto output = moge2_native::infer_vits_normal(context, model, operators,
        std::move(input), width, height, output_width, output_height);
    std::vector<float> depth(std::uint64_t(output_width) * output_height);
    context.download(output.depth, depth.data(), depth.size() * sizeof(float));
    float scalars[3]{};
    context.download(output.metric_scale, scalars, sizeof(float));
    context.download(output.focal_shift, scalars + 1, 2u * sizeof(float));
    std::fprintf(stderr, "metric_scale=%g focal=%g shift=%g\n",
        scalars[0], scalars[1], scalars[2]);
    std::vector<float> points(depth.size() * 3u);
    std::vector<float> mask(depth.size());
    context.download(output.points, points.data(), points.size() * sizeof(float));
    context.download(output.mask, mask.data(), mask.size() * sizeof(float));
    const auto z_begin = points.begin() + static_cast<std::ptrdiff_t>(depth.size() * 2u);
    std::fprintf(stderr, "points_z=%g..%g mask=%g..%g\n",
        *std::min_element(z_begin, points.end()),
        *std::max_element(z_begin, points.end()),
        *std::min_element(mask.begin(), mask.end()),
        *std::max_element(mask.begin(), mask.end()));
    {
        std::ofstream points_file(std::string(argv[3]) + ".points", std::ios::binary);
        points_file.write(reinterpret_cast<const char*>(points.data()),
            static_cast<std::streamsize>(points.size() * sizeof(float)));
        std::ofstream mask_file(std::string(argv[3]) + ".mask", std::ios::binary);
        mask_file.write(reinterpret_cast<const char*>(mask.data()),
            static_cast<std::streamsize>(mask.size() * sizeof(float)));
    }
    for (std::uint32_t level = 0; level < 5u; ++level) {
        const std::uint32_t channels[] = {384u, 256u, 128u, 64u, 32u};
        const std::size_t count = std::size_t(width / 14u << level) *
            (height / 14u << level) * channels[level];
        std::vector<float> values(count);
        context.download(output.neck_features[level], values.data(),
            values.size() * sizeof(float));
        std::ofstream file(std::string(argv[3]) + ".neck" + std::to_string(level),
            std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    for (const auto& item : {std::pair<const char*, const da3_native::VulkanBuffer*>{
             ".points_low", &output.points_low}, {".mask_low", &output.mask_low}}) {
        const std::size_t channels = item.first[1] == 'p' ? 3u : 1u;
        std::vector<float> values(std::size_t(width / 14u * 16u) *
            (height / 14u * 16u) * channels);
        context.download(*item.second, values.data(), values.size() * sizeof(float));
        std::ofstream file(std::string(argv[3]) + item.first, std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    std::ofstream destination(argv[3], std::ios::binary);
    destination.write(reinterpret_cast<const char*>(depth.data()),
        static_cast<std::streamsize>(depth.size() * sizeof(float)));
    if (!destination) throw std::runtime_error("failed to write depth");
    return 0;
} catch (const std::exception& exception) {
    std::fprintf(stderr, "%s\n", exception.what());
    return 1;
}
