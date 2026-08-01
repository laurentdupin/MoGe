#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer State { float data[]; } state_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Token { float data[]; } token_buffer;
layout(push_constant) uniform Parameters { uint columns; } parameters;
void main() {
    uint column = gl_GlobalInvocationID.x;
    if (column < parameters.columns) {
        state_buffer.data[column] = token_buffer.data[column];
    }
}
