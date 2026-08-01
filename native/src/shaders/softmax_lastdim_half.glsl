#version 450 core

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Values {
    uint data[];
} values;

layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
} parameters;

shared float partials[128];

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint row = gl_WorkGroupID.x;
    if (row >= parameters.rows) return;
    const uint words = (parameters.columns + 1) >> 1;
    float maximum = -3.402823466e+38;
    for (uint word = lane; word < words; word += 128) {
        const vec2 pair =
            unpackHalf2x16(values.data[row * words + word]);
        maximum = max(maximum, pair.x);
        if (word * 2 + 1 < parameters.columns) {
            maximum = max(maximum, pair.y);
        }
    }
    partials[lane] = maximum;
    barrier();
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partials[lane] =
                max(partials[lane], partials[lane + stride]);
        }
        barrier();
    }
    maximum = partials[0];
    float denominator = 0.0;
    for (uint word = lane; word < words; word += 128) {
        const uint index = row * words + word;
        const vec2 pair = unpackHalf2x16(values.data[index]);
        vec2 exponential = exp(pair - vec2(maximum));
        if (word * 2 + 1 >= parameters.columns) {
            exponential.y = 0.0;
        }
        values.data[index] = packHalf2x16(exponential);
        denominator += exponential.x + exponential.y;
    }
    partials[lane] = denominator;
    barrier();
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        barrier();
    }
    const float inverse = 1.0 / max(partials[0], 1.0e-20);
    for (uint word = lane; word < words; word += 128) {
        const uint index = row * words + word;
        values.data[index] = packHalf2x16(
            unpackHalf2x16(values.data[index]) * inverse);
    }
}
