#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer Values { float data[]; } values;
layout(push_constant) uniform Parameters { uint count; } p;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < p.count) values.data[index] = exp(values.data[index]);
}
