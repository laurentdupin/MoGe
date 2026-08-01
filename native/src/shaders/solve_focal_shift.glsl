#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Result { float data[]; } result;
layout(set = 0, binding = 1, std430) readonly buffer Points { float data[]; } points;
layout(set = 0, binding = 2, std430) readonly buffer Mask { float data[]; } mask;
layout(push_constant) uniform Parameters { uint width; uint height; } parameters;
shared float r00[256];
shared float r01[256];
shared float r11[256];
shared float rb0[256];
shared float rb1[256];
shared float state_f;
shared float state_s;

void main() {
    const uint lane = gl_LocalInvocationID.x;
    if (lane == 0) { state_f = 1.0; state_s = 0.0; }
    barrier();
    const uint pixels = parameters.width * parameters.height;
    const float aspect = float(parameters.width) / float(parameters.height);
    const float span_x = aspect / sqrt(1.0 + aspect * aspect);
    const float span_y = 1.0 / sqrt(1.0 + aspect * aspect);
    for (uint iteration = 0; iteration < 20; ++iteration) {
        float a00 = 0.0, a01 = 0.0, a11 = 0.0, b0 = 0.0, b1 = 0.0;
        for (uint sample_index = lane; sample_index < 4096; sample_index += 256) {
            const uint ox = sample_index & 63u;
            const uint oy = sample_index >> 6;
            const uint x = min((ox * parameters.width) / 64u, parameters.width - 1u);
            const uint y = min((oy * parameters.height) / 64u, parameters.height - 1u);
            const uint p = y * parameters.width + x;
            if (mask.data[p] <= 0.5) continue;
            const float z = points.data[2 * pixels + p];
            const float denominator = z + state_s;
            if (abs(denominator) < 1.0e-5) continue;
            const float px = points.data[p] / denominator;
            const float py = points.data[pixels + p] / denominator;
            const float u = ((2.0 * float(x) + 1.0) /
                float(parameters.width) - 1.0) * span_x;
            const float v = ((2.0 * float(y) + 1.0) /
                float(parameters.height) - 1.0) * span_y;
            const float ex = state_f * px - u;
            const float ey = state_f * py - v;
            const float jsx = -state_f * px / denominator;
            const float jsy = -state_f * py / denominator;
            a00 += px * px + py * py;
            a01 += px * jsx + py * jsy;
            a11 += jsx * jsx + jsy * jsy;
            b0 += px * ex + py * ey;
            b1 += jsx * ex + jsy * ey;
        }
        r00[lane] = a00; r01[lane] = a01; r11[lane] = a11;
        rb0[lane] = b0; rb1[lane] = b1;
        barrier();
        for (uint stride = 128; stride != 0; stride >>= 1) {
            if (lane < stride) {
                r00[lane] += r00[lane + stride];
                r01[lane] += r01[lane + stride];
                r11[lane] += r11[lane + stride];
                rb0[lane] += rb0[lane + stride];
                rb1[lane] += rb1[lane + stride];
            }
            barrier();
        }
        if (lane == 0) {
            const float damping = 1.0e-4;
            const float m00 = r00[0] + damping;
            const float m11 = r11[0] + damping;
            const float determinant = m00 * m11 - r01[0] * r01[0];
            if (abs(determinant) > 1.0e-12) {
                const float df = (-m11 * rb0[0] + r01[0] * rb1[0]) / determinant;
                const float ds = (r01[0] * rb0[0] - m00 * rb1[0]) / determinant;
                state_f = max(1.0e-4, state_f + clamp(df, -0.5, 0.5));
                state_s += clamp(ds, -1.0, 1.0);
            }
        }
        barrier();
    }
    if (lane == 0) { result.data[0] = state_f; result.data[1] = state_s; }
}
