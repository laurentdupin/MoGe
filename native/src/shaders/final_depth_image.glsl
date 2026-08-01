#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, r32f) uniform writeonly image2D depth_image;
layout(set = 0, binding = 1, std430) readonly buffer Points { float data[]; } points;
layout(set = 0, binding = 2, std430) readonly buffer Mask { float data[]; } mask;
layout(set = 0, binding = 3, std430) readonly buffer FocalShift { float data[]; } focal_shift;
layout(set = 0, binding = 4, std430) readonly buffer MetricScale { float data[]; } metric_scale;
layout(push_constant) uniform Parameters { uint width; uint height; } parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x >= parameters.width || y >= parameters.height) return;
    const uint pixels = parameters.width * parameters.height;
    const uint index = y * parameters.width + x;
    const float value = points.data[2 * pixels + index] + focal_shift.data[1];
    const float output_value = mask.data[index] > 0.5 && value > 0.0
        ? value * metric_scale.data[0]
        : uintBitsToFloat(0x7f800000u);
    imageStore(depth_image, ivec2(x, y), vec4(output_value));
}
