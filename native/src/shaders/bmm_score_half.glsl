#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    uint data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Qkv {
    float data[];
} qkv_buffer;

layout(push_constant) uniform Parameters {
    uint tokens;
    uint heads;
} parameters;

shared float query_tile[64 * 16];
shared float key_tile[32 * 16];

void main() {
    const uint column_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint row_base =
        gl_WorkGroupID.y * 64 + gl_LocalInvocationID.y * 8;
    const uint head = gl_GlobalInvocationID.z;
    const uint embedding = parameters.heads * 64;
    float sums[8][4];
    for (uint row = 0; row < 8; ++row) {
        for (uint column = 0; column < 4; ++column) {
            sums[row][column] = 0.0;
        }
    }
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    for (uint inner_base = 0; inner_base < 64; inner_base += 16) {
        for (uint index = lane; index < 64 * 16; index += 64) {
            const uint tile_row = index / 16;
            const uint inner = inner_base + index % 16;
            const uint token = gl_WorkGroupID.y * 64 + tile_row;
            query_tile[index] =
                head < parameters.heads && token < parameters.tokens
                ? qkv_buffer.data[
                    token * embedding * 3 + head * 64 + inner] * 0.125
                : 0.0;
        }
        for (uint index = lane; index < 32 * 16; index += 64) {
            const uint tile_column = index / 16;
            const uint inner = inner_base + index % 16;
            const uint token = gl_WorkGroupID.x * 32 + tile_column;
            key_tile[index] =
                head < parameters.heads && token < parameters.tokens
                ? qkv_buffer.data[
                    token * embedding * 3 + embedding +
                    head * 64 + inner]
                : 0.0;
        }
        barrier();
        for (uint inner = 0; inner < 16; ++inner) {
            float query_values[8];
            float key_values[4];
            for (uint row = 0; row < 8; ++row) {
                query_values[row] = query_tile[
                    (gl_LocalInvocationID.y * 8 + row) * 16 + inner];
            }
            for (uint column = 0; column < 4; ++column) {
                key_values[column] = key_tile[
                    (gl_LocalInvocationID.x * 4 + column) * 16 + inner];
            }
            for (uint row = 0; row < 8; ++row) {
                for (uint column = 0; column < 4; ++column) {
                    sums[row][column] +=
                        query_values[row] * key_values[column];
                }
            }
        }
        barrier();
    }
    const uint packed_columns = (parameters.tokens + 1) >> 1;
    for (uint row = 0; row < 8; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.tokens ||
            head >= parameters.heads) {
            continue;
        }
        for (uint column = 0; column < 4; column += 2) {
            const uint output_column = column_base + column;
            if (output_column < parameters.tokens) {
                const float high =
                    output_column + 1 < parameters.tokens
                    ? sums[row][column + 1]
                    : 0.0;
                output_buffer.data[
                    (head * parameters.tokens + output_row) *
                        packed_columns +
                    output_column / 2] =
                    packHalf2x16(vec2(sums[row][column], high));
            }
        }
    }
}
