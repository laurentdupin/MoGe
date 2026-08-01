#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint channels;
    uint batches;
} parameters;

float value_at(uint batch, uint channel, int x, int y) {
    const uint cx = uint(clamp(x, 0, int(parameters.input_width) - 1));
    const uint cy = uint(clamp(y, 0, int(parameters.input_height) - 1));
    return input_buffer.data[((batch * parameters.channels + channel) *
        parameters.input_height + cy) * parameters.input_width + cx];
}

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint slice = gl_GlobalInvocationID.z;
    const uint batch = slice / parameters.channels;
    const uint channel = slice % parameters.channels;
    if (x >= parameters.output_width || y >= parameters.output_height ||
        batch >= parameters.batches) return;
    const float sx = (float(x) + 0.5) * float(parameters.input_width) /
        float(parameters.output_width) - 0.5;
    const float sy = (float(y) + 0.5) * float(parameters.input_height) /
        float(parameters.output_height) - 0.5;
    const int x0 = int(floor(sx));
    const int y0 = int(floor(sy));
    const float tx = sx - float(x0);
    const float ty = sy - float(y0);
    const float top = mix(value_at(batch, channel, x0, y0),
        value_at(batch, channel, x0 + 1, y0), tx);
    const float bottom = mix(value_at(batch, channel, x0, y0 + 1),
        value_at(batch, channel, x0 + 1, y0 + 1), tx);
    output_buffer.data[((batch * parameters.channels + channel) *
        parameters.output_height + y) * parameters.output_width + x] =
        mix(top, bottom, ty);
}
