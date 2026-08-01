#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Local { float data[]; } local_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Global { float data[]; } global_buffer;
layout(push_constant) uniform Parameters { uint patches; uint embedding; } p;
void main() {
    uint index = gl_GlobalInvocationID.x;
    uint count = p.patches * p.embedding * 2;
    if (index >= count) return;
    uint patch_index = index / (p.embedding * 2);
    uint lane = index % (p.embedding * 2);
    uint channel = lane % p.embedding;
    output_buffer.data[index] = lane < p.embedding
        ? local_buffer.data[(patch_index + 1) * p.embedding + channel]
        : global_buffer.data[(patch_index + 1) * p.embedding + channel];
}
