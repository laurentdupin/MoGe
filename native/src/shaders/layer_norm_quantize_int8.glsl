#version 450 core

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(set = 0, binding = 3, std430) writeonly buffer PackedOutput {
    uint data[];
} output_buffer;
layout(set = 0, binding = 4, std430) writeonly buffer Scale {
    float data[];
} scale_buffer;
layout(push_constant) uniform Parameters {
    uint columns;
    float epsilon;
} parameters;

shared float reduction[64];
shared float row_mean;
shared float inverse_deviation;
shared float quantization_scale;

float normalized_value(uint base, uint column) {
    return (input_buffer.data[base + column] - row_mean) *
        inverse_deviation * weight_buffer.data[column] +
        bias_buffer.data[column];
}

void reduce_sum(uint lane) {
    for (uint width = 32; width > 0; width >>= 1) {
        if (lane < width) reduction[lane] += reduction[lane + width];
        barrier();
    }
}

void reduce_maximum(uint lane) {
    for (uint width = 32; width > 0; width >>= 1) {
        if (lane < width)
            reduction[lane] = max(reduction[lane], reduction[lane + width]);
        barrier();
    }
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint lane = gl_LocalInvocationID.x;
    const uint base = row * parameters.columns;
    float sum = 0.0;
    for (uint column = lane; column < parameters.columns; column += 64)
        sum += input_buffer.data[base + column];
    reduction[lane] = sum;
    barrier();
    reduce_sum(lane);
    const float denominator = max(float(parameters.columns), 1.0);
    if (lane == 0) row_mean = reduction[0] / denominator;
    barrier();

    float variance_sum = 0.0;
    for (uint column = lane; column < parameters.columns; column += 64) {
        const float difference = input_buffer.data[base + column] - row_mean;
        variance_sum += difference * difference;
    }
    reduction[lane] = variance_sum;
    barrier();
    reduce_sum(lane);
    if (lane == 0)
        inverse_deviation = inversesqrt(
            reduction[0] / denominator + parameters.epsilon);
    barrier();

    float maximum = 0.0;
    for (uint column = lane; column < parameters.columns; column += 64)
        maximum = max(maximum, abs(normalized_value(base, column)));
    reduction[lane] = maximum;
    barrier();
    reduce_maximum(lane);
    if (lane == 0) {
        quantization_scale = reduction[0] > 0.0
            ? reduction[0] / 127.0
            : 1.0;
        scale_buffer.data[row] = quantization_scale;
    }
    barrier();

    const uint packed_columns = (parameters.columns + 3) / 4;
    for (uint packed = lane; packed < packed_columns; packed += 64) {
        uint value = 0;
        for (uint component = 0; component < 4; ++component) {
            const uint column = packed * 4 + component;
            const int quantized = column < parameters.columns
                ? int(round(clamp(
                    normalized_value(base, column) / quantization_scale,
                    -127.0, 127.0)))
                : 0;
            value |= (uint(quantized) & 0xffu) << (component * 8);
        }
        output_buffer.data[row * packed_columns + packed] = value;
    }
}
