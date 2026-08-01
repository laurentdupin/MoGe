#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer PointsOut { float data[]; } points_out;
layout(set = 0, binding = 1, std430) writeonly buffer MaskOut { float data[]; } mask_out;
layout(set = 0, binding = 2, std430) readonly buffer PointsIn { float data[]; } points_in;
layout(set = 0, binding = 3, std430) readonly buffer MaskIn { float data[]; } mask_in;
layout(push_constant) uniform Parameters { uint pixels; } parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.pixels) return;
    const float z = exp(points_in.data[2 * parameters.pixels + index]);
    points_out.data[index] = points_in.data[index] * z;
    points_out.data[parameters.pixels + index] =
        points_in.data[parameters.pixels + index] * z;
    points_out.data[2 * parameters.pixels + index] = z;
    mask_out.data[index] = 1.0 / (1.0 + exp(-mask_in.data[index]));
}
