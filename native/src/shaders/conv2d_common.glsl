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
    uint output_width;
    uint output_height;
    uint output_channels;
    uint kernel;
    uint stride;
    int padding;
    uint has_bias;
    uint batches;
    uint output_channel_blocks;
#if defined(REPLICATE_PADDING)
    uint input_relu;
#endif
} parameters;

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint batch =
        gl_GlobalInvocationID.z / parameters.output_channel_blocks;
    const uint output_channel_base =
        (gl_GlobalInvocationID.z %
            parameters.output_channel_blocks) *
        OUTPUT_CHANNEL_BLOCK;
    if (output_x >= parameters.output_width ||
        output_y >= parameters.output_height ||
        output_channel_base >= parameters.output_channels ||
        batch >= parameters.batches) {
        return;
    }
    float sums[OUTPUT_CHANNEL_BLOCK];
    for (uint output_offset = 0;
         output_offset < OUTPUT_CHANNEL_BLOCK;
         ++output_offset) {
        sums[output_offset] = 0.0;
    }
    for (uint input_channel = 0;
         input_channel < parameters.input_channels;
         ++input_channel) {
        for (uint kernel_y = 0; kernel_y < parameters.kernel; ++kernel_y) {
            int input_y =
                int(output_y * parameters.stride + kernel_y) -
                parameters.padding;
#if defined(REPLICATE_PADDING)
            input_y = clamp(input_y, 0, int(parameters.input_height) - 1);
#else
            if (input_y < 0 || input_y >= int(parameters.input_height)) {
                continue;
            }
#endif
            for (uint kernel_x = 0;
                 kernel_x < parameters.kernel;
                 ++kernel_x) {
                int input_x =
                    int(output_x * parameters.stride + kernel_x) -
                    parameters.padding;
#if defined(REPLICATE_PADDING)
                input_x = clamp(input_x, 0, int(parameters.input_width) - 1);
#else
                if (input_x < 0 ||
                    input_x >= int(parameters.input_width)) {
                    continue;
                }
#endif
                const uint input_index =
                    ((batch * parameters.input_channels +
                        input_channel) * parameters.input_height +
                        uint(input_y)) *
                        parameters.input_width +
                    uint(input_x);
                float input_value = input_buffer.data[input_index];
#if defined(REPLICATE_PADDING)
                if (parameters.input_relu != 0) {
                    input_value = max(input_value, 0.0);
                }
#endif
                for (uint output_offset = 0;
                     output_offset < OUTPUT_CHANNEL_BLOCK;
                     ++output_offset) {
                    const uint output_channel =
                        output_channel_base + output_offset;
                    if (output_channel < parameters.output_channels) {
                        const uint weight_index =
                            ((output_channel *
                                parameters.input_channels +
                                input_channel) *
                                parameters.kernel +
                                kernel_y) *
                                parameters.kernel +
                            kernel_x;
                        sums[output_offset] += input_value *
                            read_weight(weight_index);
                    }
                }
            }
        }
    }
    for (uint output_offset = 0;
         output_offset < OUTPUT_CHANNEL_BLOCK;
         ++output_offset) {
        const uint output_channel =
            output_channel_base + output_offset;
        if (output_channel < parameters.output_channels) {
            float sum = sums[output_offset];
            if (parameters.has_bias != 0) {
                sum += bias_buffer.data[output_channel];
            }
            output_buffer.data[
                ((batch * parameters.output_channels +
                    output_channel) * parameters.output_height + output_y) *
                    parameters.output_width +
                output_x] = sum;
        }
    }
}
