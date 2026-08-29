#include "operators.h"
#include "inferbridge/native_harness_precision.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bilinear_align_true_image_spv.h"
#include "bmm_spv.h"
#include "bmm_score_half_spv.h"
#include "bmm_value_half_spv.h"
#include "conv2d_spv.h"
#include "conv2d8_spv.h"
#include "conv2d_half_spv.h"
#include "conv2d8_half_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "conv_transpose_nonoverlap_half_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "linear_spv.h"
#include "linear16_spv.h"
#include "linear_vec8_spv.h"
#include "linear_half_spv.h"
#include "linear16_half_spv.h"
#include "linear_int8_tiled_spv.h"
#include "quantize_rows_int8_spv.h"
#include "prepare_tokens_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "project_tokens_half_spv.h"
#include "relu_spv.h"
#include "softmax_lastdim_spv.h"
#include "softmax_lastdim_half_spv.h"
#include "replace_token_spv.h"
#include "qk_rope_spv.h"
#include "capture_concat_spv.h"
#include "add_uv_spv.h"
#include "exponential_spv.h"
#include "tokens_to_nchw_spv.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace da3_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

void require_bytes(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        buffer.size() < elements * sizeof(float)) {
        throw std::invalid_argument(
            std::string(name) + " Vulkan buffer is too small");
    }
}

void require_half_elements(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    const std::uint64_t words = (elements + 1) / 2;
    if (words > std::numeric_limits<std::uint64_t>::max() /
            sizeof(std::uint32_t) ||
        buffer.size() < words * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            std::string(name) + " packed-half Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      linear_(context.create_pipeline(
          da3_linear_spv, da3_linear_spv_size, 4, 12)),
      linear16_(context.create_pipeline(
          da3_linear16_spv, da3_linear16_spv_size, 4, 12)),
      linear_vec8_(context.create_pipeline(
          da3_linear_vec8_spv, da3_linear_vec8_spv_size, 4, 12)),
      quantize_rows_int8_(context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(
              da3_quantize_rows_int8_spv,
              da3_quantize_rows_int8_spv_size, 3, 4)
          : VulkanPipeline{}),
      linear_int8_tiled_(context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(
              da3_linear_int8_tiled_spv,
              da3_linear_int8_tiled_spv_size, 6, 28)
          : VulkanPipeline{}),
      linear_half_(context.create_pipeline(
          da3_linear_half_spv, da3_linear_half_spv_size, 4, 12)),
      linear16_half_(context.create_pipeline(
          da3_linear16_half_spv,
          da3_linear16_half_spv_size,
          4,
          12)),
      gelu_(context.create_pipeline(
          da3_gelu_spv, da3_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          da3_layer_norm_spv, da3_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          da3_add_scaled_spv, da3_add_scaled_spv_size, 4, 8)),
      bmm_(context.create_pipeline(
          da3_bmm_spv, da3_bmm_spv_size, 3, 44)),
      bmm_score_half_(context.create_pipeline(
          da3_bmm_score_half_spv,
          da3_bmm_score_half_spv_size,
          2,
          8)),
      bmm_value_half_(context.create_pipeline(
          da3_bmm_value_half_spv,
          da3_bmm_value_half_spv_size,
          3,
          8)),
      softmax_lastdim_(context.create_pipeline(
          da3_softmax_lastdim_spv,
          da3_softmax_lastdim_spv_size,
          2,
          8)),
      softmax_lastdim_half_(context.create_pipeline(
          da3_softmax_lastdim_half_spv,
          da3_softmax_lastdim_half_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {
              VK_ACCESS_SHADER_READ_BIT |
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          8)),
      prepare_tokens_(context.create_pipeline(
          da3_prepare_tokens_spv,
          da3_prepare_tokens_spv_size,
          5,
          24)),
      position_bicubic_(context.create_pipeline(
          da3_position_bicubic_spv,
          da3_position_bicubic_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          0)),
      add_position_(context.create_pipeline(
          da3_add_position_spv,
          da3_add_position_spv_size,
          4,
          20)),
      add_(context.create_pipeline(
          da3_add_spv, da3_add_spv_size, 3, 4)),
      project_tokens_(context.create_pipeline(
          da3_project_tokens_spv,
          da3_project_tokens_spv_size,
          4,
          20)),
      project_tokens_half_(context.create_pipeline(
          da3_project_tokens_half_spv,
          da3_project_tokens_half_spv_size,
          4,
          20)),
      conv2d_(context.create_pipeline(
          da3_conv2d_spv, da3_conv2d_spv_size, 4, 48)),
      conv2d8_(context.create_pipeline(
          da3_conv2d8_spv, da3_conv2d8_spv_size, 4, 48)),
      conv2d_half_(context.create_pipeline(
          da3_conv2d_half_spv, da3_conv2d_half_spv_size, 4, 48)),
      conv2d8_half_(context.create_pipeline(
          da3_conv2d8_half_spv,
          da3_conv2d8_half_spv_size,
          4,
          48)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          da3_conv_transpose_nonoverlap_spv,
          da3_conv_transpose_nonoverlap_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap_half_(context.create_pipeline(
          da3_conv_transpose_nonoverlap_half_spv,
          da3_conv_transpose_nonoverlap_half_spv_size,
          4,
          24)),
      bilinear_align_true_(context.create_pipeline(
          da3_bilinear_align_true_spv,
          da3_bilinear_align_true_spv_size,
          2,
          24)),
      bilinear_align_true_image_(context.create_pipeline(
          da3_bilinear_align_true_image_spv,
          da3_bilinear_align_true_image_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          16)),
      relu_(context.create_pipeline(
          da3_relu_spv, da3_relu_spv_size, 2, 4)),
      replace_token_(context.create_pipeline(
          da3_replace_token_spv, da3_replace_token_spv_size, 2, 4)),
      qk_rope_(context.create_pipeline(
          da3_qk_rope_spv, da3_qk_rope_spv_size, 5, 16)),
      capture_concat_(context.create_pipeline(
          da3_capture_concat_spv, da3_capture_concat_spv_size, 3, 8)),
      add_uv_(context.create_pipeline(
          da3_add_uv_spv, da3_add_uv_spv_size, 1, 20)),
      exponential_(context.create_pipeline(
          da3_exponential_spv, da3_exponential_spv_size, 1, 4)),
      tokens_to_nchw_(context.create_pipeline(
          da3_tokens_to_nchw_spv, da3_tokens_to_nchw_spv_size, 2, 8)) {
    linear_.set_debug_name("linear");
    linear16_.set_debug_name("linear16");
    linear_vec8_.set_debug_name("linear_vec8");
    if (context.supports_packed_int8_dot()) {
        quantize_rows_int8_.set_debug_name("quantize_rows_int8");
        linear_int8_tiled_.set_debug_name("linear_int8_tiled");
    }
    linear_half_.set_debug_name("linear_half");
    linear16_half_.set_debug_name("linear16_half");
    gelu_.set_debug_name("gelu");
    layer_norm_.set_debug_name("layer_norm");
    add_scaled_.set_debug_name("add_scaled");
    bmm_.set_debug_name("bmm");
    bmm_score_half_.set_debug_name("bmm_score_half");
    bmm_value_half_.set_debug_name("bmm_value_half");
    softmax_lastdim_.set_debug_name("softmax_lastdim");
    softmax_lastdim_half_.set_debug_name(
        "softmax_lastdim_half");
    prepare_tokens_.set_debug_name("prepare_tokens");
    position_bicubic_.set_debug_name("position_bicubic");
    add_position_.set_debug_name("add_position");
    add_.set_debug_name("add");
    project_tokens_.set_debug_name("project_tokens");
    project_tokens_half_.set_debug_name(
        "project_tokens_half");
    conv2d_.set_debug_name("conv2d");
    conv2d8_.set_debug_name("conv2d8");
    conv2d_half_.set_debug_name("conv2d_half");
    conv2d8_half_.set_debug_name("conv2d8_half");
    conv_transpose_nonoverlap_.set_debug_name(
        "conv_transpose_nonoverlap");
    conv_transpose_nonoverlap_half_.set_debug_name(
        "conv_transpose_nonoverlap_half");
    bilinear_align_true_.set_debug_name(
        "bilinear_align_true");
    bilinear_align_true_image_.set_debug_name(
        "bilinear_align_true_image");
    relu_.set_debug_name("relu");
    replace_token_.set_debug_name("replace_token");
    qk_rope_.set_debug_name("qk_rope");
    capture_concat_.set_debug_name("capture_concat");
    add_uv_.set_debug_name("add_uv");
    exponential_.set_debug_name("exponential");
    tokens_to_nchw_.set_debug_name("tokens_to_nchw");
}

void VulkanOperators::linear_int8(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& packed_weight,
    const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu) {
    if (!context_.supports_packed_int8_dot() || input_columns % 4u != 0u)
        throw std::runtime_error("accelerated packed INT8 linear is unavailable");
    VulkanBuffer packed_input = context_.create_device_buffer(
        std::uint64_t(rows) * (input_columns / 4u) * sizeof(std::uint32_t));
    VulkanBuffer input_scales = context_.create_device_buffer(
        std::uint64_t(rows) * sizeof(float));
    context_.dispatch(
        quantize_rows_int8_, {&input, &packed_input, &input_scales},
        &input_columns, sizeof(input_columns), rows);
    const std::uint32_t parameters[7] = {
        rows, input_columns, output_columns, 0u, output_columns, 0u, 1u};
    context_.dispatch(
        linear_int8_tiled_,
        {&output, &packed_input, &packed_weight, &input_scales,
         &weight_scales, &bias},
        parameters, sizeof(parameters),
        divide_up(output_columns, 64u), divide_up(rows, 56u));
    if (gelu) {
        const std::uint32_t count = rows * output_columns;
        context_.dispatch(
            gelu_, {&output, &output}, &count, sizeof(count),
            divide_up(count, 256u));
    }
}

void VulkanOperators::linear(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu,
    bool block16,
    bool half_weight) {
    if (rows == 0 || input_columns == 0 || output_columns == 0) {
        throw std::invalid_argument("linear dimensions cannot be zero");
    }
    require_bytes(input, std::uint64_t(rows) * input_columns, "input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_columns) * input_columns;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
    require_bytes(bias, output_columns, "bias");
    require_bytes(
        output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    VulkanPipeline& pipeline = half_weight
        ? (block16 ? linear16_half_ : linear_half_)
        : linear_vec8_;
    const std::uint32_t groups_x = half_weight
        ? divide_up(divide_up(output_columns, 4), 8)
        : divide_up(output_columns, 64);
    const std::uint32_t groups_y = half_weight
        ? divide_up(divide_up(rows, 4), 8)
        : divide_up(rows, 56);
    context_.dispatch(
        pipeline,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        groups_x,
        groups_y);
    if (gelu) {
        struct GeluParameters {
            std::uint32_t count;
        } gelu_parameters{rows * output_columns};
        context_.dispatch(
            gelu_,
            {&output, &output},
            &gelu_parameters,
            sizeof(gelu_parameters),
            divide_up(gelu_parameters.count, 256));
    }
}

void VulkanOperators::layer_norm(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t columns,
    float epsilon) {
    if (rows == 0 || columns == 0 || epsilon <= 0.0f) {
        throw std::invalid_argument("invalid layer norm parameters");
    }
    require_bytes(input, std::uint64_t(rows) * columns, "input");
    require_bytes(output, std::uint64_t(rows) * columns, "output");
    require_bytes(weight, columns, "weight");
    require_bytes(bias, columns, "bias");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t columns;
        float epsilon;
    } parameters{rows, columns, epsilon};
    context_.dispatch(
        layer_norm_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        rows);
}

void VulkanOperators::add_scaled(
    VulkanBuffer& output,
    const VulkanBuffer& residual,
    const VulkanBuffer& addend,
    const VulkanBuffer& scale,
    std::uint32_t count,
    std::uint32_t columns) {
    if (count == 0 || columns == 0 || count % columns != 0) {
        throw std::invalid_argument("invalid add-scaled dimensions");
    }
    require_bytes(output, count, "output");
    require_bytes(residual, count, "residual");
    require_bytes(addend, count, "addend");
    require_bytes(scale, columns, "scale");
    struct Parameters {
        std::uint32_t count;
        std::uint32_t columns;
    } parameters{count, columns};
    context_.dispatch(
        add_scaled_,
        {&output, &addend, &scale, &residual},
        &parameters,
        sizeof(parameters),
        divide_up(count, 256));
}

void VulkanOperators::attention_head64(
    VulkanBuffer& output,
    const VulkanBuffer& qkv,
    std::uint32_t tokens,
    std::uint32_t heads,
    VulkanBuffer* score_scratch,
    bool half_scores,
    std::uint32_t batches) {
    if (tokens == 0 || heads == 0 || batches == 0) {
        throw std::invalid_argument("invalid attention dimensions");
    }
    if (half_scores && batches != 1) {
        throw std::invalid_argument(
            "batched half-score attention is not implemented");
    }
    const std::uint64_t elements =
        std::uint64_t(batches) * tokens * heads * 64;
    require_bytes(output, elements, "attention output");
    require_bytes(qkv, elements * 3, "QKV");
    const std::uint64_t score_elements =
        std::uint64_t(batches) * heads * tokens * tokens;
    const std::uint64_t score_bytes = half_scores
        ? std::uint64_t(batches) * heads * tokens *
            ((std::uint64_t(tokens) + 1) / 2) *
            sizeof(std::uint32_t)
        : score_elements * sizeof(float);
    VulkanBuffer owned_scores;
    if (score_scratch == nullptr) {
        owned_scores = context_.create_device_buffer(score_bytes);
        score_scratch = &owned_scores;
    } else if (score_scratch->size() < score_bytes) {
        throw std::invalid_argument(
            "attention score scratch Vulkan buffer is too small");
    } else if (!half_scores) {
        require_bytes(
            *score_scratch, score_elements, "attention score scratch");
    }
    VulkanBuffer& scores = *score_scratch;
    if (half_scores) {
        struct HalfParameters {
            std::uint32_t tokens;
            std::uint32_t heads;
        } parameters{tokens, heads};
        context_.dispatch(
            bmm_score_half_,
            {&scores, &qkv},
            &parameters,
            sizeof(parameters),
            divide_up(divide_up(tokens, 4), 8),
            divide_up(divide_up(tokens, 8), 8),
            heads);
        struct SoftmaxParameters {
            std::uint32_t rows;
            std::uint32_t columns;
        } softmax_parameters{heads * tokens, tokens};
        context_.dispatch(
            softmax_lastdim_half_,
            {&scores},
            &softmax_parameters,
            sizeof(softmax_parameters),
            softmax_parameters.rows);
        context_.dispatch(
            bmm_value_half_,
            {&output, &scores, &qkv},
            &parameters,
            sizeof(parameters),
            divide_up(divide_up(64, 4), 8),
            divide_up(divide_up(tokens, 8), 8),
            heads);
        return;
    }
    struct BmmParameters {
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t inner;
        std::uint32_t batches;
        std::uint32_t weight_transposed;
        std::uint32_t output_token_major;
        std::uint32_t qkv_embedding;
        std::uint32_t input_qkv_query;
        std::uint32_t weight_qkv_kind;
        std::uint32_t qkv_heads;
        std::uint32_t qkv_tokens;
    } score_parameters{
        tokens, tokens, 64, batches * heads, 0, 0,
        heads * 64, 1, 1, heads, tokens};
    context_.dispatch(
        bmm_,
        {&scores, &qkv, &qkv},
        &score_parameters,
        sizeof(score_parameters),
        divide_up(divide_up(tokens, 4), 8),
        divide_up(divide_up(tokens, 8), 8),
        batches * heads);
    struct SoftmaxParameters {
        std::uint32_t rows;
        std::uint32_t columns;
    } softmax_parameters{batches * heads * tokens, tokens};
    context_.dispatch(
        softmax_lastdim_,
        {&scores, &scores},
        &softmax_parameters,
        sizeof(softmax_parameters),
        softmax_parameters.rows);
    BmmParameters value_parameters{
        tokens, 64, tokens, batches * heads, 0, 1,
        heads * 64, 0, 2, heads, tokens};
    context_.dispatch(
        bmm_,
        {&output, &scores, &qkv},
        &value_parameters,
        sizeof(value_parameters),
        divide_up(divide_up(64, 4), 8),
        divide_up(divide_up(tokens, 8), 8),
        batches * heads);
}

void VulkanOperators::prepare_tokens(
    VulkanBuffer& output,
    const VulkanBuffer& image,
    const VulkanBuffer& patch_weight,
    const VulkanBuffer& patch_bias,
    const VulkanBuffer& class_token,
    const VulkanBuffer& position,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t embedding,
    std::uint32_t batches) {
    if (input_width == 0 || input_height == 0 ||
        input_width % 14 != 0 || input_height % 14 != 0 ||
        embedding == 0 || batches == 0) {
        throw std::invalid_argument("invalid patch embedding dimensions");
    }
    const std::uint32_t patch_width = input_width / 14;
    const std::uint32_t patch_height = input_height / 14;
    const std::uint64_t tokens =
        std::uint64_t(patch_width) * patch_height + 1;
    require_bytes(
        image,
        std::uint64_t(batches) * input_width * input_height * 3,
        "image");
    require_bytes(
        patch_weight, std::uint64_t(embedding) * 3 * 14 * 14,
        "patch weight");
    require_bytes(patch_bias, embedding, "patch bias");
    require_bytes(class_token, embedding, "class token");
    require_bytes(
        position, std::uint64_t(1370) * embedding, "position");
    require_bytes(
        output, std::uint64_t(batches) * tokens * embedding,
        "token output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t patch_width;
        std::uint32_t patch_height;
        std::uint32_t embedding;
        std::uint32_t batches;
    } parameters{
        input_width,
        input_height,
        patch_width,
        patch_height,
        embedding,
        batches,
    };
    VulkanBuffer interpolated =
        context_.create_device_buffer(tokens * embedding * sizeof(float));
    context_.dispatch(
        prepare_tokens_,
        {
            &output,
            &image,
            &patch_weight,
            &patch_bias,
            &class_token,
        },
        &parameters,
        sizeof(parameters),
        divide_up(embedding, 8),
        divide_up(static_cast<std::uint32_t>(tokens), 8),
        batches);
    struct BufferMetadata {
        std::uint32_t logical_sizes[4];
        std::uint32_t logical_strides[4];
        std::uint32_t physical_strides[4];
        std::uint32_t info[4];
    };
    const std::uint32_t spatial = patch_width * patch_height;
    const BufferMetadata output_metadata{
        {patch_width, patch_height, embedding, 1},
        {1, patch_width, spatial, spatial * embedding},
        {1, patch_width, spatial, spatial * embedding},
        {4, spatial * embedding, spatial * embedding, 0},
    };
    const BufferMetadata input_metadata{
        {37, 37, embedding, 1},
        {1, 37, 37 * 37, 37 * 37 * embedding},
        {embedding, 37 * embedding, 1, 1370 * embedding},
        {4, 1369 * embedding, 1370 * embedding, embedding},
    };
    struct PositionBlock {
        std::int32_t info[4];
        float scale[4];
    };
    const PositionBlock position_block{
        {36, 36,
         static_cast<std::int32_t>(patch_width),
         static_cast<std::int32_t>(patch_height)},
        {
            patch_width == 37 && patch_height == 37
                ? 1.0f
                : static_cast<float>(
                    1.0 /
                    ((static_cast<double>(patch_width) + 0.1) / 37.0)),
            patch_width == 37 && patch_height == 37
                ? 1.0f
                : static_cast<float>(
                    1.0 /
                    ((static_cast<double>(patch_height) + 0.1) / 37.0)),
            0.0f,
            0.0f,
        },
    };
    VulkanBuffer output_metadata_buffer =
        context_.create_host_buffer(sizeof(output_metadata));
    VulkanBuffer input_metadata_buffer =
        context_.create_host_buffer(sizeof(input_metadata));
    VulkanBuffer position_block_buffer =
        context_.create_host_buffer(sizeof(position_block));
    context_.write_host(
        output_metadata_buffer, &output_metadata, sizeof(output_metadata));
    context_.write_host(
        input_metadata_buffer, &input_metadata, sizeof(input_metadata));
    context_.write_host(
        position_block_buffer, &position_block, sizeof(position_block));
    context_.dispatch(
        position_bicubic_,
        {
            &interpolated,
            &output_metadata_buffer,
            &position,
            &input_metadata_buffer,
            &position_block_buffer,
        },
        nullptr,
        0,
        divide_up(spatial * embedding, 256));
    struct AddPositionParameters {
        std::uint32_t patch_width;
        std::uint32_t patch_height;
        std::uint32_t embedding;
        std::uint32_t count;
        std::uint32_t tokens;
    } add_parameters{
        patch_width,
        patch_height,
        embedding,
        static_cast<std::uint32_t>(
            std::uint64_t(batches) * tokens * embedding),
        static_cast<std::uint32_t>(tokens),
    };
    context_.dispatch(
        add_position_,
        {&output, &output, &interpolated, &position},
        &add_parameters,
        sizeof(add_parameters),
        divide_up(add_parameters.count, 256));
}

void VulkanOperators::project_tokens(
    VulkanBuffer& output,
    const VulkanBuffer& tokens,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t embedding,
    std::uint32_t output_channels,
    bool half_weight,
    std::uint32_t batches) {
    if (width == 0 || height == 0 || embedding == 0 ||
        output_channels == 0 || batches == 0) {
        throw std::invalid_argument("invalid token projection dimensions");
    }
    require_bytes(
        tokens,
        std::uint64_t(batches) *
            (std::uint64_t(width) * height + 1) * embedding,
        "tokens");
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * embedding;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
    require_bytes(bias, output_channels, "bias");
    require_bytes(
        output,
        std::uint64_t(batches) * width * height * output_channels,
        "output");
    struct Parameters {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t embedding;
        std::uint32_t output_channels;
        std::uint32_t batches;
    } parameters{
        width, height, embedding, output_channels, batches};
    context_.dispatch(
        half_weight ? project_tokens_half_ : project_tokens_,
        {&output, &tokens, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_channels, 32),
        divide_up(width * height, 32),
        batches);
}

void VulkanOperators::conv2d(
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
    bool block8,
    bool half_weight,
    std::uint32_t batches) {
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || stride == 0 ||
        input_width + 2 * padding < kernel ||
        input_height + 2 * padding < kernel ||
        batches == 0) {
        throw std::invalid_argument("invalid convolution dimensions");
    }
    const std::uint32_t output_width =
        (input_width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input_height + 2 * padding - kernel) / stride + 1;
    require_bytes(
        input,
        std::uint64_t(batches) * input_width *
            input_height * input_channels,
        "convolution input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * input_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(
            weight, weight_elements, "convolution weight");
    } else {
        require_bytes(weight, weight_elements, "convolution weight");
    }
    require_bytes(bias, has_bias ? output_channels : 1, "convolution bias");
    require_bytes(
        output,
        std::uint64_t(batches) * output_width *
            output_height * output_channels,
        "convolution output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t output_channels;
        std::uint32_t kernel;
        std::uint32_t stride;
        std::int32_t padding;
        std::uint32_t has_bias;
        std::uint32_t batches;
        std::uint32_t output_channel_blocks;
    };
    const std::uint32_t output_channel_blocks =
        divide_up(output_channels, block8 ? 8 : 4);
    const Parameters parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel, stride, static_cast<std::int32_t>(padding),
        has_bias ? 1u : 0u,
        batches, output_channel_blocks,
    };
    context_.dispatch(
        half_weight
            ? (block8 ? conv2d8_half_ : conv2d_half_)
            : (block8 ? conv2d8_ : conv2d_),
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        output_channel_blocks * batches);
}

void VulkanOperators::conv_transpose_nonoverlap(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    bool half_weight,
    std::uint32_t batches) {
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || batches == 0) {
        throw std::invalid_argument("invalid transposed convolution dimensions");
    }
    const std::uint32_t output_width = input_width * kernel;
    const std::uint32_t output_height = input_height * kernel;
    require_bytes(
        input,
        std::uint64_t(batches) * input_width *
            input_height * input_channels,
        "transposed convolution input");
    const std::uint64_t weight_elements =
        std::uint64_t(input_channels) * output_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(
            weight, weight_elements, "transposed convolution weight");
    } else {
        require_bytes(
            weight, weight_elements, "transposed convolution weight");
    }
    require_bytes(bias, output_channels, "transposed convolution bias");
    require_bytes(
        output,
        std::uint64_t(batches) * output_width *
            output_height * output_channels,
        "transposed convolution output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_channels;
        std::uint32_t kernel;
        std::uint32_t batches;
    } parameters{
        input_width, input_height, input_channels,
        output_channels, kernel, batches};
    context_.dispatch(
        half_weight
            ? conv_transpose_nonoverlap_half_
            : conv_transpose_nonoverlap_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        output_channels * batches);
}

void VulkanOperators::bilinear_align_true(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height,
    std::uint32_t channels,
    std::uint32_t batches) {
    if (input_width == 0 || input_height == 0 || output_width == 0 ||
        output_height == 0 || channels == 0 || batches == 0) {
        throw std::invalid_argument("invalid bilinear dimensions");
    }
    require_bytes(
        input,
        std::uint64_t(batches) * input_width *
            input_height * channels,
        "bilinear input");
    require_bytes(
        output,
        std::uint64_t(batches) * output_width *
            output_height * channels,
        "bilinear output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t channels;
        std::uint32_t batches;
    } parameters{
        input_width, input_height, output_width, output_height,
        channels, batches};
    context_.dispatch(
        bilinear_align_true_,
        {&output, &input},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        channels * batches);
}

void VulkanOperators::bilinear_align_true_image(
    VulkanImage& output,
    const VulkanBuffer& input,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    if (input_width == 0 || input_height == 0 ||
        output_width == 0 || output_height == 0 ||
        output.width() != output_width ||
        output.height() != output_height ||
        output.format() != VK_FORMAT_R32_SFLOAT) {
        throw std::invalid_argument(
            "invalid bilinear image dimensions or format");
    }
    require_bytes(
        input,
        std::uint64_t(input_width) * input_height,
        "bilinear image input");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t output_width;
        std::uint32_t output_height;
    } parameters{
        input_width, input_height, output_width, output_height};
    context_.dispatch_buffer_to_image(
        bilinear_align_true_image_,
        input,
        output,
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8));
}

void VulkanOperators::relu(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t count) {
    require_bytes(output, count, "ReLU output");
    require_bytes(input, count, "ReLU input");
    context_.dispatch(
        relu_, {&output, &input}, &count, sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::add(
    VulkanBuffer& output,
    const VulkanBuffer& left,
    const VulkanBuffer& right,
    std::uint32_t count) {
    require_bytes(output, count, "add output");
    require_bytes(left, count, "add left");
    require_bytes(right, count, "add right");
    context_.dispatch(
        add_, {&output, &left, &right}, &count, sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::replace_token(
    VulkanBuffer& state, const VulkanBuffer& token,
    std::uint32_t columns) {
    require_bytes(state, columns, "state");
    require_bytes(token, columns, "token");
    context_.dispatch(
        replace_token_, {&state, &token}, &columns, sizeof(columns),
        divide_up(columns, 256));
}

void VulkanOperators::qk_norm_rope(
    VulkanBuffer& qkv,
    const VulkanBuffer& q_scale, const VulkanBuffer& q_bias,
    const VulkanBuffer& k_scale, const VulkanBuffer& k_bias,
    std::uint32_t tokens, std::uint32_t heads,
    std::uint32_t patch_width, std::uint32_t mode) {
    if (tokens == 0 || heads == 0 || patch_width == 0 ||
        (mode != 1 && mode != 2)) {
        throw std::invalid_argument("invalid QK/RoPE parameters");
    }
    require_bytes(qkv, std::uint64_t(tokens) * heads * 64 * 3, "QKV");
    require_bytes(q_scale, 64, "Q scale");
    require_bytes(q_bias, 64, "Q bias");
    require_bytes(k_scale, 64, "K scale");
    require_bytes(k_bias, 64, "K bias");
    struct Parameters {
        std::uint32_t tokens, heads, patch_width, mode;
    } parameters{tokens, heads, patch_width, mode};
    context_.dispatch(
        qk_rope_, {&qkv, &q_scale, &q_bias, &k_scale, &k_bias},
        &parameters, sizeof(parameters), tokens * heads * 2);
}

void VulkanOperators::capture_concat(
    VulkanBuffer& output, const VulkanBuffer& local,
    const VulkanBuffer& global, std::uint32_t patches,
    std::uint32_t embedding) {
    const std::uint64_t output_count =
        std::uint64_t(patches) * embedding * 2;
    require_bytes(output, output_count, "capture output");
    require_bytes(local, std::uint64_t(patches + 1) * embedding, "local");
    require_bytes(global, std::uint64_t(patches + 1) * embedding, "global");
    struct Parameters {
        std::uint32_t patches, embedding;
    } parameters{patches, embedding};
    context_.dispatch(
        capture_concat_, {&output, &local, &global},
        &parameters, sizeof(parameters),
        divide_up(static_cast<std::uint32_t>(output_count), 256));
}

void VulkanOperators::add_uv(
    VulkanBuffer& values, std::uint32_t width, std::uint32_t height,
    std::uint32_t channels, std::uint32_t image_width,
    std::uint32_t image_height) {
    if (width == 0 || height == 0 || channels == 0 ||
        channels % 4 != 0 || image_width == 0 || image_height == 0) {
        throw std::invalid_argument("invalid UV dimensions");
    }
    const std::uint64_t count =
        std::uint64_t(width) * height * channels;
    require_bytes(values, count, "UV values");
    struct Parameters {
        std::uint32_t width, height, channels, image_width, image_height;
    } parameters{width, height, channels, image_width, image_height};
    context_.dispatch(
        add_uv_, {&values}, &parameters, sizeof(parameters),
        divide_up(static_cast<std::uint32_t>(count), 256));
}

void VulkanOperators::exponential(
    VulkanBuffer& values, std::uint32_t count) {
    require_bytes(values, count, "exponential values");
    context_.dispatch(
        exponential_, {&values}, &count, sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::tokens_to_nchw(
    VulkanBuffer& output, const VulkanBuffer& input,
    std::uint32_t patches, std::uint32_t channels) {
    const std::uint64_t count = std::uint64_t(patches) * channels;
    require_bytes(output, count, "NCHW output");
    require_bytes(input, count, "token input");
    struct Parameters {
        std::uint32_t patches, channels;
    } parameters{patches, channels};
    context_.dispatch(
        tokens_to_nchw_, {&output, &input},
        &parameters, sizeof(parameters),
        divide_up(static_cast<std::uint32_t>(count), 256));
}

}  // namespace da3_native
