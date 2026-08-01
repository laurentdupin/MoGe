#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Tokens {
    float data[];
} token_buffer;
layout(set = 0, binding = 2, std430) readonly buffer PatchPosition {
    float data[];
} patch_position_buffer;
layout(set = 0, binding = 3, std430) readonly buffer SourcePosition {
    float data[];
} source_position_buffer;

layout(push_constant) uniform Parameters {
    uint patch_width;
    uint patch_height;
    uint embedding;
    uint count;
    uint tokens;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) return;
    const uint token =
        (index / parameters.embedding) % parameters.tokens;
    const uint feature = index % parameters.embedding;
    float position;
    if (token == 0) {
        position = source_position_buffer.data[feature];
    } else {
        const uint spatial = token - 1;
        position = patch_position_buffer.data[
            feature * parameters.patch_width * parameters.patch_height +
            spatial];
    }
    output_buffer.data[index] =
        token_buffer.data[index] + position;
}
