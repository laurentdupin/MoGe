#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint columns;
    float epsilon;
} parameters;

shared float partial_sum[64];
shared float partial_variance[64];
shared float row_mean;

void main() {
    const uint row = gl_WorkGroupID.x;
    if (row >= parameters.rows) {
        return;
    }

    const uint base = row * parameters.columns;
    float sum = 0.0;
    for (uint column = gl_LocalInvocationID.x;
         column < parameters.columns;
         column += gl_WorkGroupSize.x) {
        const float value = input_buffer.data[base + column];
        sum += value;
    }
    const uint lane = gl_LocalInvocationID.x;
    partial_sum[lane] = sum;
    barrier();
    for (uint width = 32; width > 0; width >>= 1) {
        if (lane < width) {
            partial_sum[lane] += partial_sum[lane + width];
        }
        barrier();
    }
    const float denominator = max(float(parameters.columns), 1.0);
    if (lane == 0) row_mean = partial_sum[0] / denominator;
    barrier();
    float variance_sum = 0.0;
    for (uint column = lane;
         column < parameters.columns;
         column += gl_WorkGroupSize.x) {
        const float difference = input_buffer.data[base + column] - row_mean;
        variance_sum += difference * difference;
    }
    partial_variance[lane] = variance_sum;
    barrier();
    for (uint width = 32; width > 0; width >>= 1) {
        if (lane < width) {
            partial_variance[lane] += partial_variance[lane + width];
        }
        barrier();
    }
    const float variance = partial_variance[0] / denominator;
    const float inverse_deviation =
        inversesqrt(variance + parameters.epsilon);
    for (uint column = lane;
         column < parameters.columns;
         column += gl_WorkGroupSize.x) {
        output_buffer.data[base + column] =
            (input_buffer.data[base + column] - row_mean) *
                inverse_deviation * weight_buffer.data[column] +
            bias_buffer.data[column];
    }
}
