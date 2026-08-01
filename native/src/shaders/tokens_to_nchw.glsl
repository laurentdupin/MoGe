#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters { uint patches; uint channels; } p;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= p.patches * p.channels) return;
    uint patch_index = index / p.channels;
    uint channel = index % p.channels;
    output_buffer.data[channel * p.patches + patch_index] =
        input_buffer.data[index];
}
