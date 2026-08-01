#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
    uint inner;
    uint batches;
    uint weight_transposed;
    uint output_token_major;
    uint qkv_embedding;
    uint input_qkv_query;
    uint weight_qkv_kind;
    uint qkv_heads;
    uint qkv_tokens;
} parameters;

shared float input_tile[64 * 16];
shared float weight_tile[32 * 16];

void main() {
    const uint column_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint row_base =
        gl_WorkGroupID.y * 64 + gl_LocalInvocationID.y * 8;
    const uint batch = gl_GlobalInvocationID.z;
    const uint qkv_head = parameters.qkv_heads == 0
        ? batch
        : batch % parameters.qkv_heads;
    const uint qkv_frame = parameters.qkv_heads == 0
        ? 0
        : batch / parameters.qkv_heads;
    float sums[8][4];
    for (uint row = 0; row < 8; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    for (uint inner_base = 0;
         inner_base < parameters.inner;
         inner_base += 16) {
        for (uint index = lane; index < 64 * 16; index += 64) {
            const uint tile_row = index / 16;
            const uint inner = inner_base + index % 16;
            const uint output_row =
                gl_WorkGroupID.y * 64 + tile_row;
            if (batch < parameters.batches &&
                output_row < parameters.rows &&
                inner < parameters.inner) {
                if (parameters.input_qkv_query != 0) {
                    precise float scaled_query =
                        input_buffer.data[
                            (qkv_frame * parameters.qkv_tokens +
                                output_row) *
                                    parameters.qkv_embedding * 3 +
                            qkv_head * 64 + inner] * 0.125;
                    input_tile[index] = scaled_query;
                } else {
                    input_tile[index] = input_buffer.data[
                        (batch * parameters.rows + output_row) *
                            parameters.inner + inner];
                }
            } else {
                input_tile[index] = 0.0;
            }
        }
        for (uint index = lane; index < 32 * 16; index += 64) {
            const uint tile_column = index / 16;
            const uint inner = inner_base + index % 16;
            const uint output_column =
                gl_WorkGroupID.x * 32 + tile_column;
            if (batch < parameters.batches &&
                output_column < parameters.columns &&
                inner < parameters.inner) {
                if (parameters.weight_qkv_kind != 0) {
                    const uint token =
                        parameters.weight_qkv_kind == 1
                        ? output_column
                        : inner;
                    const uint feature =
                        parameters.weight_qkv_kind == 1
                        ? inner
                        : output_column;
                    weight_tile[index] = weight_buffer.data[
                        (qkv_frame * parameters.qkv_tokens + token) *
                            parameters.qkv_embedding * 3 +
                        parameters.weight_qkv_kind *
                            parameters.qkv_embedding +
                        qkv_head * 64 + feature];
                } else {
                    weight_tile[index] =
                        parameters.weight_transposed != 0
                    ? weight_buffer.data[
                          (batch * parameters.columns + output_column) *
                              parameters.inner + inner]
                    : weight_buffer.data[
                          (batch * parameters.inner + inner) *
                              parameters.columns + output_column];
                }
            } else {
                weight_tile[index] = 0.0;
            }
        }
        barrier();
        const uint inner_count =
            min(16, parameters.inner - inner_base);
        for (uint inner = 0; inner < inner_count; ++inner) {
            float input_values[8];
            float weight_values[4];
            for (uint row = 0; row < 8; ++row) {
                input_values[row] = input_tile[
                    (gl_LocalInvocationID.y * 8 + row) * 16 + inner];
            }
            for (uint column = 0; column < 4; ++column) {
                weight_values[column] = weight_tile[
                    (gl_LocalInvocationID.x * 4 + column) * 16 + inner];
            }
            for (uint row = 0; row < 8; ++row) {
                for (uint column = 0; column < 4; ++column) {
                    sums[row][column] +=
                        input_values[row] * weight_values[column];
                }
            }
        }
        barrier();
    }
    for (uint row = 0; row < 8; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.rows) continue;
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            if (output_column < parameters.columns) {
                const uint output_index =
                    parameters.output_token_major != 0 &&
                        parameters.qkv_heads != 0
                    ? ((qkv_frame * parameters.qkv_tokens + output_row) *
                          parameters.qkv_embedding +
                        qkv_head * 64 + output_column)
                    : parameters.output_token_major != 0
                    ? (output_row * parameters.batches + batch) *
                        parameters.columns + output_column
                    : (batch * parameters.rows + output_row) *
                          parameters.columns + output_column;
                output_buffer.data[output_index] = sums[row][column];
            }
        }
    }
}
