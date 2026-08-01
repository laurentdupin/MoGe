#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters {
    uint width;
    uint height;
    uint input_channels;
    float aspect_ratio;
} parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel = gl_GlobalInvocationID.z;
    if (x >= parameters.width || y >= parameters.height ||
        channel >= parameters.input_channels + 2) return;
    const uint pixels = parameters.width * parameters.height;
    const uint pixel = y * parameters.width + x;
    if (channel < parameters.input_channels) {
        output_buffer.data[channel * pixels + pixel] =
            input_buffer.data[channel * pixels + pixel];
        return;
    }
    const float span_x = parameters.aspect_ratio /
        sqrt(1.0 + parameters.aspect_ratio * parameters.aspect_ratio);
    const float span_y = 1.0 /
        sqrt(1.0 + parameters.aspect_ratio * parameters.aspect_ratio);
    const float coordinate = channel == parameters.input_channels
        ? ((2.0 * float(x) + 1.0) / float(parameters.width) - 1.0) * span_x
        : ((2.0 * float(y) + 1.0) / float(parameters.height) - 1.0) * span_y;
    output_buffer.data[channel * pixels + pixel] = coordinate;
}
