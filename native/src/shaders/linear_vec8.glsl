#version 450 core

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    vec4 data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    vec4 data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(set = 0, binding = 4, std430) readonly buffer Residual {
    float data[];
} residual_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Scale {
    float data[];
} scale_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint input_columns;
    uint output_columns;
    uint residual;
} parameters;

#define K_VECTORS 8
#define K_STRIDE 9
shared vec4 input_tile[56 * K_STRIDE];
shared vec4 weight_tile[64 * K_STRIDE];

void main() {
    const uint column_base =
        gl_WorkGroupID.x * 64 + gl_LocalInvocationID.x * 4;
    const uint row_base =
        gl_WorkGroupID.y * 56 + gl_LocalInvocationID.y * 7;
    float sums[7][4];
    for (uint row = 0; row < 7; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    const uint input_vectors = parameters.input_columns / 4;
    for (uint inner_base = 0;
         inner_base < input_vectors;
         inner_base += K_VECTORS) {
        for (uint index = lane; index < 56 * K_VECTORS; index += 128) {
            const uint tile_row = index / K_VECTORS;
            const uint inner = inner_base + index % K_VECTORS;
            const uint output_row =
                gl_WorkGroupID.y * 56 + tile_row;
            input_tile[tile_row * K_STRIDE + index % K_VECTORS] =
                output_row < parameters.rows && inner < input_vectors
                ? input_buffer.data[
                      output_row * input_vectors + inner]
                : vec4(0.0);
        }
        for (uint index = lane; index < 64 * K_VECTORS; index += 128) {
            const uint tile_column = index / K_VECTORS;
            const uint inner = inner_base + index % K_VECTORS;
            const uint output_column =
                gl_WorkGroupID.x * 64 + tile_column;
            weight_tile[tile_column * K_STRIDE + index % K_VECTORS] =
                output_column < parameters.output_columns &&
                    inner < input_vectors
                ? weight_buffer.data[
                      output_column * input_vectors + inner]
                : vec4(0.0);
        }
        barrier();
        const uint count =
            min(K_VECTORS, input_vectors - inner_base);
        for (uint inner = 0; inner < count; ++inner) {
            vec4 input_values[7];
            vec4 weight_values[4];
            for (uint row = 0; row < 7; ++row) {
                input_values[row] = input_tile[
                    (gl_LocalInvocationID.y * 7 + row) * K_STRIDE +
                    inner];
            }
            for (uint column = 0; column < 4; ++column) {
                weight_values[column] = weight_tile[
                    (gl_LocalInvocationID.x * 4 + column) * K_STRIDE +
                    inner];
            }
            for (uint row = 0; row < 7; ++row) {
                for (uint column = 0; column < 4; ++column) {
                    sums[row][column] +=
                        dot(input_values[row], weight_values[column]);
                }
            }
        }
        barrier();
    }
    for (uint row = 0; row < 7; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.rows) {
            continue;
        }
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            if (output_column < parameters.output_columns) {
                float value =
                    sums[row][column] + bias_buffer.data[output_column];
                if (parameters.residual != 0) {
                    precise float scaled =
                        value * scale_buffer.data[output_column];
                    precise float result = residual_buffer.data[
                        output_row * parameters.output_columns +
                        output_column] + scaled;
                    value = result;
                }
                output_buffer.data[
                    output_row * parameters.output_columns + output_column] =
                    value;
            }
        }
    }
}
