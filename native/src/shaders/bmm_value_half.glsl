#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Scores {
    uint data[];
} score_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Qkv {
    float data[];
} qkv_buffer;

layout(push_constant) uniform Parameters {
    uint tokens;
    uint heads;
} parameters;

shared float score_tile[64 * 16];
shared float value_tile[32 * 16];

float read_score(uint row, uint column, uint head) {
    const uint words = (parameters.tokens + 1) >> 1;
    const vec2 pair = unpackHalf2x16(
        score_buffer.data[
            (head * parameters.tokens + row) * words + column / 2]);
    return (column & 1) == 0 ? pair.x : pair.y;
}

void main() {
    const uint feature_base =
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
    for (uint inner_base = 0;
         inner_base < parameters.tokens;
         inner_base += 16) {
        for (uint index = lane; index < 64 * 16; index += 64) {
            const uint tile_row = index / 16;
            const uint source = inner_base + index % 16;
            const uint row = gl_WorkGroupID.y * 64 + tile_row;
            score_tile[index] =
                head < parameters.heads &&
                    row < parameters.tokens &&
                    source < parameters.tokens
                ? read_score(row, source, head)
                : 0.0;
        }
        for (uint index = lane; index < 32 * 16; index += 64) {
            const uint tile_column = index / 16;
            const uint source = inner_base + index % 16;
            const uint feature =
                gl_WorkGroupID.x * 32 + tile_column;
            value_tile[index] =
                head < parameters.heads &&
                    feature < 64 &&
                    source < parameters.tokens
                ? qkv_buffer.data[
                    source * embedding * 3 + embedding * 2 +
                    head * 64 + feature]
                : 0.0;
        }
        barrier();
        const uint inner_count =
            min(16, parameters.tokens - inner_base);
        for (uint inner = 0; inner < inner_count; ++inner) {
            float score_values[8];
            float value_values[4];
            for (uint row = 0; row < 8; ++row) {
                score_values[row] = score_tile[
                    (gl_LocalInvocationID.y * 8 + row) * 16 + inner];
            }
            for (uint column = 0; column < 4; ++column) {
                value_values[column] = value_tile[
                    (gl_LocalInvocationID.x * 4 + column) * 16 + inner];
            }
            for (uint row = 0; row < 8; ++row) {
                for (uint column = 0; column < 4; ++column) {
                    sums[row][column] +=
                        score_values[row] * value_values[column];
                }
            }
        }
        barrier();
    }
    for (uint row = 0; row < 8; ++row) {
        const uint output_row = row_base + row;
        if (output_row >= parameters.tokens ||
            head >= parameters.heads) {
            continue;
        }
        for (uint column = 0; column < 4; ++column) {
            const uint feature = feature_base + column;
            if (feature < 64) {
                output_buffer.data[
                    (output_row * parameters.heads + head) * 64 +
                    feature] = sums[row][column];
            }
        }
    }
}
