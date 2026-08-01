#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint count;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) {
        return;
    }
    const float value = input_buffer.data[index];
    const float cube = value * value * value;
    const float inner =
        0.7978845608028654 * (value + 0.044715 * cube);
    output_buffer.data[index] =
        0.5 * value *
        (1.0 + tanh(clamp(inner, -15.0, 15.0)));
}
