#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Depth { float data[]; } depth;
layout(set = 0, binding = 1, std430) readonly buffer Points { float data[]; } points;
layout(set = 0, binding = 2, std430) readonly buffer Mask { float data[]; } mask;
layout(set = 0, binding = 3, std430) readonly buffer FocalShift { float data[]; } focal_shift;
layout(set = 0, binding = 4, std430) readonly buffer MetricScale { float data[]; } metric_scale;
layout(push_constant) uniform Parameters {
    uint pixels;
    float background_distance_metres;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.pixels) return;
    const float value = points.data[2 * parameters.pixels + index] + focal_shift.data[1];
    depth.data[index] = mask.data[index] > 0.5 && value > 0.0
        ? value * metric_scale.data[0]
        : parameters.background_distance_metres;
}
