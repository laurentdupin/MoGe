#pragma once

#include "vulkan.h"
#include <inferbridge/native_harness_int8_workspace.h>

#include <cstdint>

namespace da3_native {

class VulkanOperators {
public:
    explicit VulkanOperators(VulkanContext& context);

    void linear(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t input_columns,
        std::uint32_t output_columns,
        bool gelu,
        bool block16 = false,
        bool half_weight = false);
    void linear_int8(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& packed_weight,
        const VulkanBuffer& weight_scales,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t input_columns,
        std::uint32_t output_columns,
        bool gelu = false);

    void layer_norm(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t columns,
        float epsilon);

    void add_scaled(
        VulkanBuffer& output,
        const VulkanBuffer& residual,
        const VulkanBuffer& addend,
        const VulkanBuffer& scale,
        std::uint32_t count,
        std::uint32_t columns);

    void attention_head64(
        VulkanBuffer& output,
        const VulkanBuffer& qkv,
        std::uint32_t tokens,
        std::uint32_t heads,
        VulkanBuffer* score_scratch = nullptr,
        bool half_scores = false,
        std::uint32_t batches = 1);

    void prepare_tokens(
        VulkanBuffer& output,
        const VulkanBuffer& image,
        const VulkanBuffer& patch_weight,
        const VulkanBuffer& patch_bias,
        const VulkanBuffer& class_token,
        const VulkanBuffer& position,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t embedding,
        std::uint32_t batches = 1);

    void project_tokens(
        VulkanBuffer& output,
        const VulkanBuffer& tokens,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t embedding,
        std::uint32_t output_channels,
        bool half_weight = false,
        std::uint32_t batches = 1);

    void conv2d(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t input_channels,
        std::uint32_t output_channels,
        std::uint32_t kernel,
        std::uint32_t stride,
        std::uint32_t padding,
        bool has_bias,
        bool block8 = false,
        bool half_weight = false,
        std::uint32_t batches = 1);

    void conv_transpose_nonoverlap(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t input_channels,
        std::uint32_t output_channels,
        std::uint32_t kernel,
        bool half_weight = false,
        std::uint32_t batches = 1);

    void bilinear_align_true(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height,
        std::uint32_t channels,
        std::uint32_t batches = 1);
    void bilinear_align_true_image(
        VulkanImage& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height);

    void relu(VulkanBuffer& output, const VulkanBuffer& input, std::uint32_t count);

    void add(
        VulkanBuffer& output,
        const VulkanBuffer& left,
        const VulkanBuffer& right,
        std::uint32_t count);

    void replace_token(
        VulkanBuffer& state, const VulkanBuffer& token,
        std::uint32_t columns);
    void qk_norm_rope(
        VulkanBuffer& qkv,
        const VulkanBuffer& q_scale, const VulkanBuffer& q_bias,
        const VulkanBuffer& k_scale, const VulkanBuffer& k_bias,
        std::uint32_t tokens, std::uint32_t heads,
        std::uint32_t patch_width, std::uint32_t mode);
    void capture_concat(
        VulkanBuffer& output, const VulkanBuffer& local,
        const VulkanBuffer& global, std::uint32_t patches,
        std::uint32_t embedding);
    void add_uv(
        VulkanBuffer& values, std::uint32_t width, std::uint32_t height,
        std::uint32_t channels, std::uint32_t image_width,
        std::uint32_t image_height);
    void exponential(VulkanBuffer& values, std::uint32_t count);
    void tokens_to_nchw(
        VulkanBuffer& output, const VulkanBuffer& input,
        std::uint32_t patches, std::uint32_t channels);

private:
    VulkanContext& context_;
    inferbridge::native::Int8ActivationWorkspace<VulkanBuffer> int8_workspace_;
    VulkanPipeline linear_;
    VulkanPipeline linear16_;
    VulkanPipeline linear_vec8_;
    VulkanPipeline quantize_rows_int8_;
    VulkanPipeline linear_int8_tiled_;
    VulkanPipeline linear_half_;
    VulkanPipeline linear16_half_;
    VulkanPipeline gelu_;
    VulkanPipeline layer_norm_;
    VulkanPipeline add_scaled_;
    VulkanPipeline bmm_;
    VulkanPipeline bmm_score_half_;
    VulkanPipeline bmm_value_half_;
    VulkanPipeline softmax_lastdim_;
    VulkanPipeline softmax_lastdim_half_;
    VulkanPipeline prepare_tokens_;
    VulkanPipeline position_bicubic_;
    VulkanPipeline add_position_;
    VulkanPipeline add_;
    VulkanPipeline project_tokens_;
    VulkanPipeline project_tokens_half_;
    VulkanPipeline conv2d_;
    VulkanPipeline conv2d8_;
    VulkanPipeline conv2d_half_;
    VulkanPipeline conv2d8_half_;
    VulkanPipeline conv_transpose_nonoverlap_;
    VulkanPipeline conv_transpose_nonoverlap_half_;
    VulkanPipeline bilinear_align_true_;
    VulkanPipeline bilinear_align_true_image_;
    VulkanPipeline relu_;
    VulkanPipeline replace_token_;
    VulkanPipeline qk_rope_;
    VulkanPipeline capture_concat_;
    VulkanPipeline add_uv_;
    VulkanPipeline exponential_;
    VulkanPipeline tokens_to_nchw_;
};

}  // namespace da3_native
