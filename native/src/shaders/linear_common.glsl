layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
#if defined(HALF_WEIGHT)
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
float read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return (index & 1) == 0 ? values.x : values.y;
}
#else
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
float read_weight(uint index) {
    return weight_buffer.data[index];
}
#endif
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint input_columns;
    uint output_columns;
} parameters;

#define K_STRIDE (K_TILE + 1)
shared float input_tile[32 * K_STRIDE];
shared float weight_tile[32 * K_STRIDE];

void main() {
    const uint column_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint row_base =
        gl_WorkGroupID.y * 32 + gl_LocalInvocationID.y * 4;
    if (column_base >= parameters.output_columns ||
        row_base >= parameters.rows) {
        // The invocation must still participate in workgroup barriers.
    }
    float sums[4][4];
    for (uint row = 0; row < 4; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    for (uint inner_base = 0;
         inner_base < parameters.input_columns;
         inner_base += K_TILE) {
        for (uint index = lane; index < 32 * K_TILE; index += 64) {
            const uint tile_row = index / K_TILE;
            const uint inner = inner_base + index % K_TILE;
            const uint output_row =
                gl_WorkGroupID.y * 32 + tile_row;
            input_tile[tile_row * K_STRIDE + index % K_TILE] =
                output_row < parameters.rows &&
                    inner < parameters.input_columns
                ? input_buffer.data[
                      output_row * parameters.input_columns + inner]
                : 0.0;
        }
        for (uint index = lane; index < 32 * K_TILE; index += 64) {
            const uint tile_column = index / K_TILE;
            const uint inner = inner_base + index % K_TILE;
            const uint output_column =
                gl_WorkGroupID.x * 32 + tile_column;
            weight_tile[tile_column * K_STRIDE + index % K_TILE] =
                output_column < parameters.output_columns
                    && inner < parameters.input_columns
                ? read_weight(
                    output_column * parameters.input_columns + inner)
                : 0.0;
        }
        barrier();
        const uint inner_count =
            min(K_TILE, parameters.input_columns - inner_base);
        for (uint inner = 0; inner < inner_count; ++inner) {
            float input_values[4];
            float weight_values[4];
            for (uint row = 0; row < 4; ++row) {
                input_values[row] = input_tile[
                    (gl_LocalInvocationID.y * 4 + row) * K_STRIDE + inner];
            }
            for (uint column = 0; column < 4; ++column) {
                weight_values[column] = weight_tile[
                    (gl_LocalInvocationID.x * 4 + column) * K_STRIDE + inner];
            }
            for (uint row = 0; row < 4; ++row) {
                for (uint column = 0; column < 4; ++column) {
                    sums[row][column] +=
                        input_values[row] * weight_values[column];
                }
            }
        }
        barrier();
    }
    for (uint row = 0; row < 4; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.rows) {
            continue;
        }
        for (uint column = 0; column < 4; ++column) {
            const uint output_column = column_base + column;
            if (output_column < parameters.output_columns) {
                output_buffer.data[
                    output_row * parameters.output_columns + output_column] =
                    sums[row][column] + bias_buffer.data[output_column];
            }
        }
    }
}
