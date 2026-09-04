#version 450 core
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel; uint stride; int padding;
    uint has_bias; uint batches; uint output_channel_blocks; uint input_relu;
} p;
shared float input_tile[1440];
shared vec4 weight_tile[144];
void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint block = gl_GlobalInvocationID.z % p.output_channel_blocks;
    uint batch = gl_GlobalInvocationID.z / p.output_channel_blocks;
    uint output_base = block * 8;
    bool valid = x < p.output_width && y < p.output_height &&
        output_base < p.output_channels && batch < p.batches;
    vec4 sums0 = vec4(0);
    vec4 sums1 = vec4(0);
    uint lane = gl_LocalInvocationID.y * 16 + gl_LocalInvocationID.x;
    int tile_x = int(gl_WorkGroupID.x * 16) - 1;
    int tile_y = int(gl_WorkGroupID.y * 8) - 1;
    uint input_batch = batch * p.input_channels * p.input_height * p.input_width;
    uint output_batch = batch * p.output_channels * p.output_height * p.output_width;
    for (uint input_base = 0; input_base < p.input_channels; input_base += 8) {
        for (uint index = lane; index < 1440; index += 128) {
            uint channel_offset = index / 180;
            uint tile_index = index % 180;
            uint channel = input_base + channel_offset;
            int input_x = clamp(
                tile_x + int(tile_index % 18), 0, int(p.input_width) - 1);
            int input_y = clamp(
                tile_y + int(tile_index / 18), 0, int(p.input_height) - 1);
            float value = channel < p.input_channels
                ? input_buffer.data[input_batch +
                    (channel * p.input_height + uint(input_y)) *
                    p.input_width + uint(input_x)]
                : 0.0;
            input_tile[index] = p.input_relu != 0 ? max(value, 0.0) : value;
        }
        for (uint index = lane; index < 144; index += 128) {
            uint channel_offset = index / 18;
            uint packed_index = index % 18;
            uint output_group = packed_index / 9;
            uint kernel_index = packed_index % 9;
            uint input_channel = input_base + channel_offset;
            uint output_channel = output_base + output_group * 4;
            weight_tile[index] = input_channel >= p.input_channels
                ? vec4(0)
                : vec4(
                    output_channel < p.output_channels
                        ? weight_buffer.data[(output_channel * p.input_channels + input_channel) * 9 + kernel_index] : 0,
                    output_channel + 1 < p.output_channels
                        ? weight_buffer.data[((output_channel + 1) * p.input_channels + input_channel) * 9 + kernel_index] : 0,
                    output_channel + 2 < p.output_channels
                        ? weight_buffer.data[((output_channel + 2) * p.input_channels + input_channel) * 9 + kernel_index] : 0,
                    output_channel + 3 < p.output_channels
                        ? weight_buffer.data[((output_channel + 3) * p.input_channels + input_channel) * 9 + kernel_index] : 0);
        }
        barrier();
        if (valid) {
            for (uint channel_offset = 0;
                 channel_offset < 8 &&
                 input_base + channel_offset < p.input_channels;
                 ++channel_offset) {
                for (uint kernel_y = 0; kernel_y < 3; ++kernel_y) {
                    for (uint kernel_x = 0; kernel_x < 3; ++kernel_x) {
                        float value = input_tile[
                            channel_offset * 180 +
                            (gl_LocalInvocationID.y + kernel_y) * 18 +
                            gl_LocalInvocationID.x + kernel_x];
                        uint kernel_index = kernel_y * 3 + kernel_x;
                        sums0 += value * weight_tile[
                            channel_offset * 18 + kernel_index];
                        sums1 += value * weight_tile[
                            channel_offset * 18 + 9 + kernel_index];
                    }
                }
            }
        }
        barrier();
    }
    if (!valid) return;
    for (uint output_offset = 0; output_offset < 8; ++output_offset) {
        uint output_channel = output_base + output_offset;
        if (output_channel < p.output_channels) {
            output_buffer.data[output_batch +
                (output_channel * p.output_height + y) * p.output_width + x] =
                (output_offset < 4
                    ? sums0[output_offset]
                    : sums1[output_offset - 4]) +
                bias_buffer.data[output_channel];
        }
    }
}
