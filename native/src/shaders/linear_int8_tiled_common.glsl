#define TILE_ROWS 56
#define ROWS_PER_LANE 7
#define INVOCATIONS 128
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    int data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    int data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer InputScale {
    float data[];
} input_scale_buffer;
layout(set = 0, binding = 4, std430) readonly buffer WeightScale {
    float data[];
} weight_scale_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint rows;
    uint input_columns;
    uint output_columns;
    uint output_row_offset;
    uint output_row_stride;
    uint output_transposed;
    uint has_bias;
} parameters;
shared int input_tile[TILE_ROWS * K_STRIDE];
shared int weight_tile[64 * K_STRIDE];

void main() {
    const uint output_column_base =
        gl_WorkGroupID.x * 64u + gl_LocalInvocationID.x * 4u;
    const uint output_row_base =
        gl_WorkGroupID.y * TILE_ROWS +
        gl_LocalInvocationID.y * ROWS_PER_LANE;
    int sums[ROWS_PER_LANE][4];
    for (uint row = 0u; row < ROWS_PER_LANE; ++row)
        for (uint column = 0u; column < 4u; ++column)
            sums[row][column] = 0;
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    const uint packed_columns = parameters.input_columns / 4u;
    for (uint inner_base = 0u;
         inner_base < packed_columns;
         inner_base += K_PACKED) {
        for (uint index = lane; index < TILE_ROWS * K_PACKED;
             index += INVOCATIONS) {
            const uint tile_row = index / K_PACKED;
            const uint inner = inner_base + index % K_PACKED;
            const uint row = gl_WorkGroupID.y * TILE_ROWS + tile_row;
            input_tile[tile_row * K_STRIDE + index % K_PACKED] =
                row < parameters.rows && inner < packed_columns
                ? input_buffer.data[row * packed_columns + inner] : 0;
        }
        for (uint index = lane; index < 64u * K_PACKED;
             index += INVOCATIONS) {
            const uint tile_column = index / K_PACKED;
            const uint inner = inner_base + index % K_PACKED;
            const uint output_column = gl_WorkGroupID.x * 64u + tile_column;
            weight_tile[tile_column * K_STRIDE + index % K_PACKED] =
                output_column < parameters.output_columns &&
                    inner < packed_columns
                ? weight_buffer.data[
                    output_column * packed_columns + inner] : 0;
        }
        barrier();
        const uint count = min(K_PACKED, packed_columns - inner_base);
        for (uint inner = 0u; inner < count; ++inner) {
            int input_values[ROWS_PER_LANE];
            int weight_values[4];
            for (uint row = 0u; row < ROWS_PER_LANE; ++row)
                input_values[row] = input_tile[
                    (gl_LocalInvocationID.y * ROWS_PER_LANE + row) *
                        K_STRIDE + inner];
            for (uint column = 0u; column < 4u; ++column)
                weight_values[column] = weight_tile[
                    (gl_LocalInvocationID.x * 4u + column) *
                        K_STRIDE + inner];
            for (uint row = 0u; row < ROWS_PER_LANE; ++row)
                for (uint column = 0u; column < 4u; ++column)
                    sums[row][column] += dotPacked4x8EXT(
                        input_values[row], weight_values[column]);
        }
        barrier();
    }
    for (uint row_index = 0u; row_index < ROWS_PER_LANE; ++row_index) {
        const uint row = output_row_base + row_index;
        if (row >= parameters.rows) continue;
        for (uint column_index = 0u; column_index < 4u; ++column_index) {
            const uint output_column = output_column_base + column_index;
            if (output_column >= parameters.output_columns) continue;
            const uint destination_row = row + parameters.output_row_offset;
            const uint destination_index = parameters.output_transposed != 0u
                ? output_column * parameters.output_row_stride + destination_row
                : destination_row * parameters.output_columns + output_column;
            output_buffer.data[destination_index] =
                float(sums[row_index][column_index]) *
                    input_scale_buffer.data[row] *
                    weight_scale_buffer.data[output_column] +
                (parameters.has_bias != 0u
                    ? bias_buffer.data[output_column] : 0.0);
        }
    }
}
