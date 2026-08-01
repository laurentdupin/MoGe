#version 450 core
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) buffer Qkv { float data[]; } qkv;
layout(set = 0, binding = 1, std430) readonly buffer QScale { float data[]; } qs;
layout(set = 0, binding = 2, std430) readonly buffer QBias { float data[]; } qb;
layout(set = 0, binding = 3, std430) readonly buffer KScale { float data[]; } ks;
layout(set = 0, binding = 4, std430) readonly buffer KBias { float data[]; } kb;
layout(push_constant) uniform Parameters {
    uint tokens; uint heads; uint patch_width; uint mode;
} p;
shared float sum[64];
shared float variance[64];
void main() {
    uint channel = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.x;
    uint head = row % p.heads;
    uint q_or_k = (row / p.heads) % 2;
    uint token = row / (p.heads * 2);
    uint embedding = p.heads * 64;
    uint index = token * 3 * embedding + q_or_k * embedding + head * 64 + channel;
    float value = qkv.data[index];
    sum[channel] = value;
    barrier();
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (channel < stride) sum[channel] += sum[channel + stride];
        barrier();
    }
    float mean = sum[0] / 64.0;
    float delta = value - mean;
    variance[channel] = delta * delta;
    barrier();
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (channel < stride) variance[channel] += variance[channel + stride];
        barrier();
    }
    value = delta * inversesqrt(variance[0] / 64.0 + 1.0e-5);
    value = value * (q_or_k == 0 ? qs.data[channel] : ks.data[channel]) +
        (q_or_k == 0 ? qb.data[channel] : kb.data[channel]);
    qkv.data[index] = value;
    barrier();
    uint coordinate = 0;
    if (token != 0) {
        uint patch_index = token - 1;
        coordinate = p.mode == 2 ? 1 :
            (channel < 32 ? patch_index / p.patch_width + 1 :
             patch_index % p.patch_width + 1);
    }
    uint half_channel = channel & 31u;
    uint lane = half_channel & 15u;
    uint pair_channel = (channel & ~31u) + (lane + ((half_channel < 16) ? 16 : 0));
    float pair = qkv.data[token * 3 * embedding + q_or_k * embedding + head * 64 + pair_channel];
    float angle = float(coordinate) / pow(100.0, float(lane * 2) / 32.0);
    float result = half_channel < 16
        ? value * cos(angle) - pair * sin(angle)
        : value * cos(angle) + pair * sin(angle);
    qkv.data[index] = result;
}
