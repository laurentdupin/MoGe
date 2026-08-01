#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint channels;
    uint batches;
} parameters;

float read_value(uint x, uint y, uint batch, uint channel) {
    return input_buffer.data[
        ((batch * parameters.channels + channel) *
            parameters.input_height + y) *
            parameters.input_width +
        x];
}

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint batch = gl_GlobalInvocationID.z / parameters.channels;
    const uint channel = gl_GlobalInvocationID.z % parameters.channels;
    if (output_x >= parameters.output_width ||
        output_y >= parameters.output_height ||
        channel >= parameters.channels ||
        batch >= parameters.batches) {
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
        read_value(x0, y0, batch, channel) * (1.0 - x_fraction) +
        read_value(x1, y0, batch, channel) * x_fraction;
    const float bottom =
        read_value(x0, y1, batch, channel) * (1.0 - x_fraction) +
        read_value(x1, y1, batch, channel) * x_fraction;
    output_buffer.data[
        ((batch * parameters.channels + channel) *
            parameters.output_height + output_y) *
            parameters.output_width +
        output_x] =
        top * (1.0 - y_fraction) + bottom * y_fraction;
}
