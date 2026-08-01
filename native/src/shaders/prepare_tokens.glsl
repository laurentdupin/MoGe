#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer TokenOutput {
    float data[];
} token_output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Image {
    float data[];
} image_buffer;
layout(set = 0, binding = 2, std430) readonly buffer PatchWeight {
    float data[];
} patch_weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer PatchBias {
    float data[];
} patch_bias_buffer;
layout(set = 0, binding = 4, std430) readonly buffer ClassToken {
    float data[];
} class_token_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint patch_width;
    uint patch_height;
    uint embedding;
    uint batches;
} parameters;

void main() {
    const uint feature = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    const uint batch = gl_GlobalInvocationID.z;
    const uint patch_count =
        parameters.patch_width * parameters.patch_height;
    if (feature >= parameters.embedding || token > patch_count ||
        batch >= parameters.batches) {
        return;
    }
    const uint token_base =
        batch * (patch_count + 1) * parameters.embedding;
    if (token == 0) {
        token_output_buffer.data[token_base + feature] =
            class_token_buffer.data[feature];
        return;
    }

    const uint patch_index = token - 1;
    const uint patch_x = patch_index % parameters.patch_width;
    const uint patch_y = patch_index / parameters.patch_width;
    float sum = 0.0;
    for (uint channel = 0; channel < 3; ++channel) {
        for (uint kernel_y = 0; kernel_y < 14; ++kernel_y) {
            for (uint kernel_x = 0; kernel_x < 14; ++kernel_x) {
                const uint image_x = patch_x * 14 + kernel_x;
                const uint image_y = patch_y * 14 + kernel_y;
                const uint image_index =
                    ((batch * 3 + channel) *
                        parameters.input_height + image_y) *
                        parameters.input_width +
                    image_x;
                const uint weight_index =
                    ((feature * 3 + channel) * 14 + kernel_y) * 14 +
                    kernel_x;
                sum += image_buffer.data[image_index] *
                    patch_weight_buffer.data[weight_index];
            }
        }
    }
    token_output_buffer.data[
        token_base + token * parameters.embedding + feature] =
        sum + patch_bias_buffer.data[feature];
}
