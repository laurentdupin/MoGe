#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Addend {
    float data[];
} addend_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Scale {
    float data[];
} scale_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Residual {
    float data[];
} residual_buffer;

layout(push_constant) uniform Parameters {
    uint count;
    uint columns;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index < parameters.count) {
        precise float scaled =
            addend_buffer.data[index] *
            scale_buffer.data[index % parameters.columns];
        precise float result = residual_buffer.data[index] + scaled;
        output_buffer.data[index] = result;
    }
}
