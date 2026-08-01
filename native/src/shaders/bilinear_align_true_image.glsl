#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, r32f) uniform writeonly image2D output_image;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
} parameters;

float read_value(uint x, uint y) {
    return input_buffer.data[y * parameters.input_width + x];
}

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    if (output_x >= parameters.output_width ||
        output_y >= parameters.output_height) {
        return;
    }
    const float source_x = parameters.output_width > 1
        ? float(output_x) *
            float(parameters.input_width - 1) /
            float(parameters.output_width - 1)
        : 0.0;
    const float source_y = parameters.output_height > 1
        ? float(output_y) *
            float(parameters.input_height - 1) /
            float(parameters.output_height - 1)
        : 0.0;
    const uint x0 = uint(floor(source_x));
    const uint y0 = uint(floor(source_y));
    const uint x1 = min(x0 + 1, parameters.input_width - 1);
    const uint y1 = min(y0 + 1, parameters.input_height - 1);
    const float x_fraction = source_x - float(x0);
    const float y_fraction = source_y - float(y0);
    const float top =
        read_value(x0, y0) * (1.0 - x_fraction) +
        read_value(x1, y0) * x_fraction;
    const float bottom =
        read_value(x0, y1) * (1.0 - x_fraction) +
        read_value(x1, y1) * x_fraction;
    imageStore(
        output_image,
        ivec2(output_x, output_y),
        vec4(top * (1.0 - y_fraction) + bottom * y_fraction));
}
