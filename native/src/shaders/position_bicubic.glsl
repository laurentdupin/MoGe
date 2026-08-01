#version 450 core

// Adapted from PyTorch's upsample_bicubic2d_buffer_align_false.glsl.
// See THIRD_PARTY_NOTICES.md.
#define PRECISION highp
#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    float data[];
} uOutput;

layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
    uvec4 logical_sizes;
    uvec4 logical_strides;
    uvec4 physical_strides;
    uvec4 info;
} uOutMeta;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
    float data[];
} uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
    uvec4 logical_sizes;
    uvec4 logical_strides;
    uvec4 physical_strides;
    uvec4 info;
} uInMeta;

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
    ivec4 info;
    vec4 scale;
} uBlock;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

float cubic_convolution1(float x, float a) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

float cubic_convolution2(float x, float a) {
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

vec4 cubic_coeffs(float t) {
    const float a = -0.75;
    const float x1 = t;
    const float x2 = 1.0 - t;
    return vec4(
        cubic_convolution2(x1 + 1.0, a),
        cubic_convolution1(x1, a),
        cubic_convolution1(x2, a),
        cubic_convolution2(x2 + 1.0, a));
}

float cubic_interp1d(float x0, float x1, float x2, float x3, float t) {
    const vec4 coeffs = cubic_coeffs(t);
    return x0 * coeffs.x + x1 * coeffs.y +
        x2 * coeffs.z + x3 * coeffs.w;
}

float fetch_bounded(ivec2 pos_xy, uint channel, uint batch) {
    const ivec2 clamped_xy = clamp(pos_xy, ivec2(0, 0), uBlock.info.xy);
    const uvec4 coord =
        uvec4(uint(clamped_xy.x), uint(clamped_xy.y), channel, batch);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + uInMeta.info.w;
    if (read_idx >= uInMeta.info.z) {
        return 0.0;
    }
    return uInput.data[read_idx];
}

void main() {
    const uint write_idx = gl_GlobalInvocationID.x;
    if (write_idx >= uOutMeta.info.y) {
        return;
    }
    const uvec4 out_coord = idx_to_coord(
        write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
    const vec2 src =
        (vec2(out_coord.xy) + vec2(0.5, 0.5)) * uBlock.scale.xy -
        vec2(0.5, 0.5);
    const ivec2 base = ivec2(floor(src));
    const vec2 t = src - vec2(base);

    float rows[4];
    for (int k = 0; k < 4; ++k) {
        const int y = base.y - 1 + k;
        rows[k] = cubic_interp1d(
            fetch_bounded(ivec2(base.x - 1, y), out_coord.z, out_coord.w),
            fetch_bounded(ivec2(base.x + 0, y), out_coord.z, out_coord.w),
            fetch_bounded(ivec2(base.x + 1, y), out_coord.z, out_coord.w),
            fetch_bounded(ivec2(base.x + 2, y), out_coord.z, out_coord.w),
            t.x);
    }
    const uint actual_write_idx =
        coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
    if (actual_write_idx < uOutMeta.info.z) {
        uOutput.data[actual_write_idx] =
            cubic_interp1d(rows[0], rows[1], rows[2], rows[3], t.y);
    }
}
