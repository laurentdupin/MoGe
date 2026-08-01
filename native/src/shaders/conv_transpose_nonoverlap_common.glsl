layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
#if defined(HALF_WEIGHT)
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
float read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return (index & 1) == 0 ? values.x : values.y;
}
#else
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
float read_weight(uint index) {
    return weight_buffer.data[index];
}
#endif
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_channels;
    uint kernel;
    uint batches;
} parameters;

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint batch =
        gl_GlobalInvocationID.z / parameters.output_channels;
    const uint output_channel =
        gl_GlobalInvocationID.z % parameters.output_channels;
    const uint output_width = parameters.input_width * parameters.kernel;
    const uint output_height = parameters.input_height * parameters.kernel;
    if (output_x >= output_width || output_y >= output_height ||
        output_channel >= parameters.output_channels ||
        batch >= parameters.batches) {
        return;
    }
    const uint input_x = output_x / parameters.kernel;
    const uint input_y = output_y / parameters.kernel;
    const uint kernel_x = output_x % parameters.kernel;
    const uint kernel_y = output_y % parameters.kernel;
    float sum = 0.0;
    for (uint input_channel = 0;
         input_channel < parameters.input_channels;
         ++input_channel) {
        const uint input_index =
            ((batch * parameters.input_channels +
                input_channel) * parameters.input_height + input_y) *
                parameters.input_width +
            input_x;
        const uint weight_index =
            ((input_channel * parameters.output_channels +
                output_channel) *
                parameters.kernel +
                kernel_y) *
                parameters.kernel +
            kernel_x;
        sum += input_buffer.data[input_index] *
            read_weight(weight_index);
    }
    output_buffer.data[
        ((batch * parameters.output_channels +
            output_channel) * output_height + output_y) *
            output_width +
        output_x] = sum + bias_buffer.data[output_channel];
}
