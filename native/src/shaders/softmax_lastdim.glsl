#version 450 core

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
} parameters;

shared float partials[128];

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint row = gl_WorkGroupID.x;
    if (row >= parameters.rows) return;
    float maximum = -3.402823466e+38;
    for (uint column = lane;
         column < parameters.columns;
         column += 128) {
        maximum = max(
            maximum,
            input_buffer.data[row * parameters.columns + column]);
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
    barrier();
    float denominator = 0.0;
    for (uint column = lane;
         column < parameters.columns;
         column += 128) {
        const uint index = row * parameters.columns + column;
        const float exponential =
            exp(input_buffer.data[index] - maximum);
        output_buffer.data[index] = exponential;
        denominator += exponential;
    }
    memoryBarrierBuffer();
    partials[lane] = denominator;
    barrier();
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partials[lane] += partials[lane + stride];
        }
        barrier();
    }
    const float inverse =
        1.0 / max(partials[0], 1.0e-20);
    for (uint column = lane;
         column < parameters.columns;
         column += 128) {
        const uint index = row * parameters.columns + column;
        output_buffer.data[index] *= inverse;
    }
}
