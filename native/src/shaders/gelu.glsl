#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint count;
} parameters;

float erf_high_accuracy(float value) {
    const float magnitude = abs(value);
    const float t = 1.0 / (1.0 + 0.3275911 * magnitude);
    const float polynomial =
        (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
            0.284496736) * t + 0.254829592) * t;
    const float result = 1.0 - polynomial * exp(-magnitude * magnitude);
    return value < 0.0 ? -result : result;
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) {
        return;
    }
    const float value = input_buffer.data[index];
    output_buffer.data[index] =
        0.5 * value * (1.0 + erf_high_accuracy(value * 0.7071067811865476));
}
