#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float values[];
} input_buffer;

layout(set = 0, binding = 1, std430) writeonly buffer OutputBuffer {
    float values[];
} output_buffer;

layout(push_constant) uniform Parameters {
    uint count;
    float scale;
    float bias;
    uint relu;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) {
        return;
    }
    float value = input_buffer.values[index] * parameters.scale + parameters.bias;
    if (parameters.relu != 0u) {
        value = max(value, 0.0);
    }
    output_buffer.values[index] = value;
}
