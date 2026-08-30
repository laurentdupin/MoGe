#include "external_gpu.h"
#include "model_config.h"
#include "safetensors.h"

#include "inferbridge/native_harness_host_image.h"
#include "inferbridge/native_harness_precision.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace moge2_native {
namespace {

using da3_native::SafeTensors;
using da3_native::TensorView;

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t i = 0; i < tensor.rank; ++i)
        [result addObject:@(tensor.dimensions[i])];
    return result;
}

float cubic(float distance) {
    constexpr float a = -0.75f;
    distance = std::abs(distance);
    if (distance <= 1.0f)
        return ((a + 2.0f) * distance - (a + 3.0f)) *
            distance * distance + 1.0f;
    if (distance < 2.0f)
        return ((a * distance - 5.0f * a) * distance + 8.0f * a) *
            distance - 4.0f * a;
    return 0.0f;
}

std::vector<float> position_embedding(
    const TensorView& source, int patch_height, int patch_width,
    int embedding) {
    std::vector<float> result(
        static_cast<std::size_t>(patch_height * patch_width + 1) * embedding);
    std::copy_n(source.data, embedding, result.data());
    for (int y = 0; y < patch_height; ++y) {
        const float sy = (y + 0.5f) * 37.0f / (patch_height + 0.1f) - 0.5f;
        const int by = static_cast<int>(std::floor(sy));
        for (int x = 0; x < patch_width; ++x) {
            const float sx = (x + 0.5f) * 37.0f / (patch_width + 0.1f) - 0.5f;
            const int bx = static_cast<int>(std::floor(sx));
            float* destination = result.data() +
                static_cast<std::size_t>(1 + y * patch_width + x) * embedding;
            for (int channel = 0; channel < embedding; ++channel) {
                float value = 0.0f;
                for (int oy = -1; oy <= 2; ++oy) {
                    const int py = std::clamp(by + oy, 0, 36);
                    const float wy = cubic(sy - (by + oy));
                    for (int ox = -1; ox <= 2; ++ox) {
                        const int px = std::clamp(bx + ox, 0, 36);
                        value += wy * cubic(sx - (bx + ox)) * source.data[
                            static_cast<std::size_t>(1 + py * 37 + px) *
                                embedding + channel];
                    }
                }
                destination[channel] = value;
            }
        }
    }
    return result;
}

std::vector<float> uv_coordinates(int height, int width, float aspect) {
    const float denominator = std::sqrt(1.0f + aspect * aspect);
    const float span_x = aspect / denominator;
    const float span_y = 1.0f / denominator;
    std::vector<float> result(static_cast<std::size_t>(2) * height * width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * width + x;
            result[p] = ((2.0f * x + 1.0f) / width - 1.0f) * span_x;
            result[static_cast<std::size_t>(height) * width + p] =
                ((2.0f * y + 1.0f) / height - 1.0f) * span_y;
        }
    }
    return result;
}

class GraphBuilder {
public:
    GraphBuilder(const SafeTensors& model, const ModelConfig& config,
                 int encoder_width, int encoder_height,
                 int output_width, int output_height, bool fp16)
        : model_(model), config_(config), encoder_width_(encoder_width),
          encoder_height_(encoder_height), output_width_(output_width),
          output_height_(output_height), patch_width_(encoder_width / 14),
          patch_height_(encoder_height / 14),
          tokens_(patch_width_ * patch_height_ + 1), fp16_(fp16),
          graph_([MPSGraph new]) {}

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    NSArray<MPSGraphTensor*>* outputs() const {
        return @[points_, mask_, scale_];
    }

    void build() {
        input_ = [graph_ placeholderWithShape:
            shape({1, 3, encoder_height_, encoder_width_})
            dataType:MPSDataTypeFloat32 name:@"normalized_rgb_chw"];
        MPSGraphTensor* value = fp16_ ? [graph_ castTensor:input_
            toType:MPSDataTypeFloat16 name:nil] : input_;
        value = conv(value, "encoder.backbone.patch_embed.proj", 14, 0);
        value = [graph_ reshapeTensor:value withShape:
            shape({1, config_.embedding, patch_height_ * patch_width_}) name:nil];
        value = [graph_ transposeTensor:value dimension:1 withDimension:2 name:nil];
        value = [graph_ concatTensors:@[
            constant("encoder.backbone.cls_token"), value] dimension:1 name:nil];
        const auto positions = position_embedding(
            model_.tensor("encoder.backbone.pos_embed"), patch_height_,
            patch_width_, config_.embedding);
        value = add(value, owned(positions,
            shape({1, tokens_, config_.embedding})));

        std::vector<MPSGraphTensor*> captures;
        MPSGraphTensor* final_normalized = nil;
        for (std::uint32_t block = 0; block < config_.blocks; ++block) {
            const std::string p = "encoder.backbone.blocks." +
                std::to_string(block) + ".";
            MPSGraphTensor* normalized = layer_norm(value, p + "norm1", 1.0e-6f);
            MPSGraphTensor* qkv = [graph_ reshapeTensor:
                linear(normalized, p + "attn.qkv") withShape:
                shape({1, tokens_, 3, config_.heads, 64}) name:nil];
            qkv = [graph_ transposeTensor:qkv
                permutation:@[@2, @0, @3, @1, @4] name:nil];
            MPSGraphTensor* query = [graph_ reshapeTensor:slice(qkv, 0, 0, 1)
                withShape:shape({1, config_.heads, tokens_, 64}) name:nil];
            MPSGraphTensor* key = [graph_ reshapeTensor:slice(qkv, 0, 1, 1)
                withShape:shape({1, config_.heads, tokens_, 64}) name:nil];
            MPSGraphTensor* attention_value = [graph_ reshapeTensor:
                slice(qkv, 0, 2, 1)
                withShape:shape({1, config_.heads, tokens_, 64}) name:nil];
            key = [graph_ transposeTensor:key dimension:2 withDimension:3 name:nil];
            MPSGraphTensor* scores = [graph_ matrixMultiplicationWithPrimaryTensor:
                multiply(query, scalar(0.125f)) secondaryTensor:key name:nil];
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attended = [graph_ matrixMultiplicationWithPrimaryTensor:
                scores secondaryTensor:attention_value name:nil];
            attended = [graph_ transposeTensor:attended dimension:1 withDimension:2 name:nil];
            attended = [graph_ reshapeTensor:attended
                withShape:shape({1, tokens_, config_.embedding}) name:nil];
            value = add(value, multiply(linear(attended, p + "attn.proj"),
                constant(p + "ls1.gamma")));
            normalized = layer_norm(value, p + "norm2", 1.0e-6f);
            MPSGraphTensor* hidden = gelu(linear(normalized, p + "mlp.fc1"));
            value = add(value, multiply(linear(hidden, p + "mlp.fc2"),
                constant(p + "ls2.gamma")));
            for (std::uint32_t capture = 0;
                 capture < config_.capture_count; ++capture) {
                if (block != config_.captures[capture]) continue;
                final_normalized = layer_norm(
                    value, "encoder.backbone.norm", 1.0e-6f);
                MPSGraphTensor* patches = slice(
                    final_normalized, 1, 1, tokens_ - 1);
                patches = [graph_ transposeTensor:patches
                    dimension:1 withDimension:2 name:nil];
                patches = [graph_ reshapeTensor:patches withShape:
                    shape({1, config_.embedding, patch_height_, patch_width_})
                    name:nil];
                captures.push_back(conv(patches,
                    "encoder.output_projections." + std::to_string(capture),
                    1, 0));
            }
        }
        if (captures.size() != config_.capture_count || final_normalized == nil)
            throw std::runtime_error("Metal MoGe encoder capture mismatch");
        MPSGraphTensor* encoded = captures.front();
        for (std::size_t i = 1; i < captures.size(); ++i)
            encoded = add(encoded, captures[i]);
        MPSGraphTensor* class_token = [graph_ reshapeTensor:
            slice(final_normalized, 1, 0, 1)
            withShape:shape({1, config_.embedding}) name:nil];
        const std::array<int, 5> channels{
            static_cast<int>(config_.decoder_embedding), 256, 128, 64, 32};
        const float aspect = static_cast<float>(output_width_) / output_height_;
        const auto features = neck(encoded, channels, aspect);
        MPSGraphTensor* points = head(features, "points_head", 3, channels);
        MPSGraphTensor* mask = head(features, "mask_head", 1, channels);
        points = resize(points, output_height_, output_width_, false);
        mask = resize(mask, output_height_, output_width_, false);
        MPSGraphTensor* z = [graph_ exponentWithTensor:
            slice(points, 1, 2, 1) name:nil];
        points_ = [graph_ concatTensors:@[
            multiply(slice(points, 1, 0, 1), z),
            multiply(slice(points, 1, 1, 1), z), z] dimension:1 name:nil];
        mask_ = [graph_ sigmoidWithTensor:mask name:nil];
        MPSGraphTensor* scale = [graph_ reLUWithTensor:
            linear(class_token, "scale_head.0") name:nil];
        scale = [graph_ reLUWithTensor:linear(scale, "scale_head.2") name:nil];
        scale_ = [graph_ exponentWithTensor:linear(scale, "scale_head.4") name:nil];
        if (fp16_) {
            points_ = [graph_ castTensor:points_ toType:MPSDataTypeFloat32 name:nil];
            mask_ = [graph_ castTensor:mask_ toType:MPSDataTypeFloat32 name:nil];
            scale_ = [graph_ castTensor:scale_ toType:MPSDataTypeFloat32 name:nil];
        }
    }

private:
    MPSGraphTensor* constant(const std::string& name) {
        const TensorView& tensor = model_.tensor(name);
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<float*>(tensor.data)
            length:tensor.elements * sizeof(float) freeWhenDone:NO];
        MPSGraphTensor* result = [graph_ constantWithData:data
            shape:shape(tensor) dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:result
            toType:MPSDataTypeFloat16 name:nil] : result;
    }
    MPSGraphTensor* owned(const std::vector<float>& values, MPSShape* dimensions) {
        NSData* data = [NSData dataWithBytes:values.data()
            length:values.size() * sizeof(float)];
        MPSGraphTensor* result = [graph_ constantWithData:data
            shape:dimensions dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:result
            toType:MPSDataTypeFloat16 name:nil] : result;
    }
    MPSGraphTensor* scalar(float value) {
        MPSGraphTensor* result = [graph_ constantWithScalar:value
            dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:result
            toType:MPSDataTypeFloat16 name:nil] : result;
    }
    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* multiply(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* slice(MPSGraphTensor* value, NSUInteger dimension,
                          NSInteger start, NSInteger length) {
        return [graph_ sliceTensor:value dimension:dimension start:start
            length:length name:nil];
    }
    MPSGraphTensor* linear(MPSGraphTensor* value, const std::string& p) {
        MPSGraphTensor* weight = constant(p + ".weight");
        if (weight.shape.count == 4u)
            weight = [graph_ reshapeTensor:weight withShape:shape({
                static_cast<NSInteger>(model_.tensor(p + ".weight").dimensions[0]),
                static_cast<NSInteger>(model_.tensor(p + ".weight").dimensions[1])})
                name:nil];
        weight = [graph_ transposeTensor:weight dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* result = [graph_ matrixMultiplicationWithPrimaryTensor:value
            secondaryTensor:weight name:nil];
        if (model_.contains(p + ".bias")) result = add(result, constant(p + ".bias"));
        return result;
    }
    MPSGraphTensor* layer_norm(MPSGraphTensor* value, const std::string& p,
                               float epsilon) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:value axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:value
            meanTensor:mean axes:axes name:nil];
        return [graph_ normalizationWithTensor:value meanTensor:mean
            varianceTensor:variance gammaTensor:constant(p + ".weight")
            betaTensor:constant(p + ".bias") epsilon:epsilon name:nil];
    }
    MPSGraphTensor* gelu(MPSGraphTensor* value) {
        MPSGraphTensor* error = [graph_ erfWithTensor:
            multiply(value, scalar(0.7071067811865475f)) name:nil];
        return multiply(multiply(value, scalar(0.5f)), add(error, scalar(1.0f)));
    }
    MPSGraphConvolution2DOpDescriptor* descriptor(int stride, int padding) {
        return [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride
            strideInY:stride dilationRateInX:1 dilationRateInY:1 groups:1
            paddingLeft:padding paddingRight:padding paddingTop:padding
            paddingBottom:padding paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }
    MPSGraphTensor* conv(MPSGraphTensor* value, const std::string& p,
                         int stride, int padding) {
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:value
            weightsTensor:constant(p + ".weight") descriptor:descriptor(stride, padding)
            name:nil];
        const auto& bias = model_.tensor(p + ".bias");
        return add(result, [graph_ reshapeTensor:constant(p + ".bias")
            withShape:shape({1, static_cast<NSInteger>(bias.elements), 1, 1}) name:nil]);
    }
    MPSGraphTensor* replicate_conv(MPSGraphTensor* value, const std::string& p) {
        value = [graph_ padTensor:value withPaddingMode:MPSGraphPaddingModeClampToEdge
            leftPadding:shape({0, 0, 1, 1}) rightPadding:shape({0, 0, 1, 1})
            constantValue:0.0 name:nil];
        return conv(value, p, 1, 0);
    }
    MPSGraphTensor* transpose_conv(MPSGraphTensor* value, const std::string& p,
                                    int channels, int h, int w) {
        MPSGraphTensor* result = [graph_ convolutionTranspose2DWithSourceTensor:value
            weightsTensor:constant(p + ".weight") outputShape:shape({1, channels, h * 2, w * 2})
            descriptor:descriptor(2, 0) name:nil];
        return add(result, [graph_ reshapeTensor:constant(p + ".bias")
            withShape:shape({1, channels, 1, 1}) name:nil]);
    }
    MPSGraphTensor* resize(MPSGraphTensor* value, int height, int width,
                           bool align_corners) {
        return [graph_ resizeTensor:value size:shape({height, width})
            mode:MPSGraphResizeBilinear centerResult:align_corners ? NO : YES
            alignCorners:align_corners ? YES : NO
            layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
    }
    MPSGraphTensor* residual(MPSGraphTensor* value, const std::string& p) {
        MPSGraphTensor* result = [graph_ reLUWithTensor:value name:nil];
        result = replicate_conv(result, p + ".layers.2");
        result = [graph_ reLUWithTensor:result name:nil];
        return add(value, replicate_conv(result, p + ".layers.5"));
    }
    MPSGraphTensor* resample(MPSGraphTensor* value, const std::string& stack,
                             int level, int h, int w,
                             const std::array<int, 5>& channels) {
        const std::string p = stack + ".resamplers." + std::to_string(level);
        const int intermediate = level < 3 ? channels[level + 1] : channels[level];
        MPSGraphTensor* upsampled = level < 3
            ? transpose_conv(value, p + ".0", intermediate, h, w)
            : resize(value, h * 2, w * 2, false);
        return replicate_conv(upsampled, p + ".1");
    }
    std::array<MPSGraphTensor*, 5> neck(
        MPSGraphTensor* encoded, const std::array<int, 5>& channels,
        float aspect) {
        std::array<MPSGraphTensor*, 5> output{};
        MPSGraphTensor* carried = nil;
        for (int level = 0; level < 5; ++level) {
            const int w = patch_width_ << level;
            const int h = patch_height_ << level;
            MPSGraphTensor* uv = owned(uv_coordinates(h, w, aspect),
                shape({1, 2, h, w}));
            MPSGraphTensor* input = level == 0
                ? [graph_ concatTensors:@[encoded, uv] dimension:1 name:nil]
                : uv;
            MPSGraphTensor* state = conv(input,
                "neck.input_blocks." + std::to_string(level), 1, 0);
            if (level != 0) state = add(carried, state);
            if (level >= 1 && level <= 3)
                for (std::uint32_t block = 0;
                     block < config_.neck_residual_blocks; ++block)
                    state = residual(state, "neck.res_blocks." +
                        std::to_string(level) + "." + std::to_string(block));
            output[level] = state;
            if (level < 4) carried = resample(
                state, "neck", level, h, w, channels);
        }
        return output;
    }
    MPSGraphTensor* head(const std::array<MPSGraphTensor*, 5>& features,
                         const std::string& stack, int output_channels,
                         const std::array<int, 5>& channels) {
        MPSGraphTensor* state = nil;
        MPSGraphTensor* carried = nil;
        for (int level = 0; level < 5; ++level) {
            const int w = patch_width_ << level;
            const int h = patch_height_ << level;
            state = conv(features[level], stack + ".input_blocks." +
                std::to_string(level), 1, 0);
            if (level != 0) state = add(carried, state);
            if (level >= 1 && level <= 3)
                for (std::uint32_t block = 0;
                     block < config_.head_residual_blocks; ++block)
                    state = residual(state, stack + ".res_blocks." +
                        std::to_string(level) + "." + std::to_string(block));
            if (level < 4) carried = resample(
                state, stack, level, h, w, channels);
        }
        return conv(state, stack + ".output_blocks.4", 1, 0);
    }

    const SafeTensors& model_;
    ModelConfig config_;
    int encoder_width_, encoder_height_, output_width_, output_height_;
    int patch_width_, patch_height_, tokens_;
    bool fp16_;
    MPSGraph* graph_ = nil;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* points_ = nil;
    MPSGraphTensor* mask_ = nil;
    MPSGraphTensor* scale_ = nil;
};

struct Plan {
    MPSGraph* graph = nil;
    MPSGraphTensor* input = nil;
    MPSGraphExecutable* executable = nil;
};

std::array<float, 2> solve_focal_shift(
    const float* points, const float* mask, std::uint32_t width,
    std::uint32_t height) {
    const std::uint32_t pixels = width * height;
    const float aspect = static_cast<float>(width) / height;
    const float span_x = aspect / std::sqrt(1.0f + aspect * aspect);
    const float span_y = 1.0f / std::sqrt(1.0f + aspect * aspect);
    float f = 1.0f;
    float s = 0.0f;
    for (int iteration = 0; iteration < 20; ++iteration) {
        double a00 = 0.0, a01 = 0.0, a11 = 0.0, b0 = 0.0, b1 = 0.0;
        for (std::uint32_t sample = 0; sample < 4096; ++sample) {
            const std::uint32_t ox = sample & 63u;
            const std::uint32_t oy = sample >> 6u;
            const std::uint32_t x = std::min(ox * width / 64u, width - 1u);
            const std::uint32_t y = std::min(oy * height / 64u, height - 1u);
            const std::uint32_t p = y * width + x;
            if (mask[p] <= 0.5f) continue;
            const float denominator = points[2u * pixels + p] + s;
            if (std::abs(denominator) < 1.0e-5f) continue;
            const float px = points[p] / denominator;
            const float py = points[pixels + p] / denominator;
            const float u = ((2.0f * x + 1.0f) / width - 1.0f) * span_x;
            const float v = ((2.0f * y + 1.0f) / height - 1.0f) * span_y;
            const float ex = f * px - u;
            const float ey = f * py - v;
            const float jsx = -f * px / denominator;
            const float jsy = -f * py / denominator;
            a00 += px * px + py * py;
            a01 += px * jsx + py * jsy;
            a11 += jsx * jsx + jsy * jsy;
            b0 += px * ex + py * ey;
            b1 += jsx * ex + jsy * ey;
        }
        const double m00 = a00 + 1.0e-4;
        const double m11 = a11 + 1.0e-4;
        const double determinant = m00 * m11 - a01 * a01;
        if (std::abs(determinant) > 1.0e-12) {
            const float df = static_cast<float>((-m11 * b0 + a01 * b1) / determinant);
            const float ds = static_cast<float>((a01 * b0 - m00 * b1) / determinant);
            f = std::max(1.0e-4f, f + std::clamp(df, -0.5f, 0.5f));
            s += std::clamp(ds, -1.0f, 1.0f);
        }
    }
    return {f, s};
}

class MetalGpu final : public ExternalGpu {
public:
    explicit MetalGpu(const std::string& path)
        : model_(path), config_(read_model_config(model_)) {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (device_ == nil || queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("Metal is unavailable for MoGe-2");
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("MoGe-2 Metal does not support INT8");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
    }
    ExternalGpuCapabilities capabilities() const override { return {}; }
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest&) override {
        throw std::runtime_error("MoGe-2 Metal texture binding is not implemented");
    }
    void infer_host(const std::uint8_t* pixels, std::uint32_t width,
        std::uint32_t height, std::size_t row_stride, bool rgba,
        std::uint32_t num_tokens, float background_distance_metres,
        float* output) override {
        if (!pixels || !output || !width || !height || num_tokens < 16u)
            throw std::invalid_argument("invalid MoGe-2 Metal host request");
        const float aspect = static_cast<float>(width) / height;
        const std::uint32_t token_height = std::max(1u,
            static_cast<std::uint32_t>(std::nearbyint(std::sqrt(num_tokens / aspect))));
        const std::uint32_t token_width = std::max(1u,
            static_cast<std::uint32_t>(std::nearbyint(std::sqrt(num_tokens * aspect))));
        const std::uint32_t encoder_width = token_width * 14u;
        const std::uint32_t encoder_height = token_height * 14u;
        constexpr inferbridge::native_harness::ImageNormalization normalization{
            {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
        auto normalized = inferbridge::native_harness::resize_bgra8_to_normalized_chw(
            pixels, width, height, row_stride, encoder_width, encoder_height,
            rgba, normalization);
        const std::uint64_t key = (static_cast<std::uint64_t>(token_width) << 48u) |
            (static_cast<std::uint64_t>(token_height) << 32u) |
            (static_cast<std::uint64_t>(width) << 16u) | height;
        std::lock_guard<std::mutex> lock(mutex_);
        @autoreleasepool {
            Plan& plan = get_plan(key, encoder_width, encoder_height, width, height);
            id<MTLBuffer> buffer = [device_ newBufferWithBytes:normalized.data()
                length:normalized.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            MPSGraphTensorData* data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buffer shape:shape({1, 3,
                    static_cast<NSInteger>(encoder_height),
                    static_cast<NSInteger>(encoder_width)})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* descriptor =
                [MPSGraphExecutableExecutionDescriptor new];
            descriptor.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:@[data]
                resultsArray:nil executionDescriptor:descriptor];
            if (results.count != 3u)
                throw std::runtime_error("MoGe-2 Metal graph returned invalid outputs");
            const std::size_t count = static_cast<std::size_t>(width) * height;
            std::vector<float> points(count * 3u);
            std::vector<float> mask(count);
            float scale = 0.0f;
            [results[0].mpsndarray readBytes:points.data() strideBytes:nil];
            [results[1].mpsndarray readBytes:mask.data() strideBytes:nil];
            [results[2].mpsndarray readBytes:&scale strideBytes:nil];
            const auto focal_shift = solve_focal_shift(
                points.data(), mask.data(), width, height);
            for (std::size_t i = 0; i < count; ++i) {
                const float value = points[2u * count + i] + focal_shift[1];
                output[i] = mask[i] > 0.5f && value > 0.0f
                    ? value * scale : background_distance_metres;
            }
            upload_bytes_.fetch_add(normalized.size() * sizeof(float));
            download_bytes_.fetch_add((points.size() + mask.size() + 1u) * sizeof(float));
        }
    }
    void transfer_counters(std::uint64_t& upload,
                           std::uint64_t& download) const override {
        upload = upload_bytes_.load();
        download = download_bytes_.load();
    }

private:
    Plan& get_plan(std::uint64_t key, int encoder_width, int encoder_height,
                   int output_width, int output_height) {
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        GraphBuilder builder(model_, config_, encoder_width, encoder_height,
            output_width, output_height, fp16_);
        builder.build();
        MPSGraphShapedType* type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, encoder_height, encoder_width})
            dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* descriptor = [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        descriptor.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [builder.graph() compileWithDevice:graph_device_
            feeds:@{builder.input(): type} targetTensors:builder.outputs()
            targetOperations:nil compilationDescriptor:descriptor];
        if (executable == nil)
            throw std::runtime_error("failed to compile MoGe-2 Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key,
            Plan{builder.graph(), builder.input(), executable}).first->second;
    }

    SafeTensors model_;
    ModelConfig config_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<std::uint64_t, Plan> plans_;
    std::mutex mutex_;
    std::atomic<std::uint64_t> upload_bytes_{0u};
    std::atomic<std::uint64_t> download_bytes_{0u};
};

}  // namespace

std::shared_ptr<ExternalGpu> create_metal_gpu(const std::string& path) {
    return std::make_shared<MetalGpu>(path);
}

}  // namespace moge2_native
