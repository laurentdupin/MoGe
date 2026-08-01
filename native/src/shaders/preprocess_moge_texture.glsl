#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(binding = 0) uniform sampler2D source_texture;
layout(std430, binding = 1) writeonly buffer Destination { float values[]; } destination;
layout(push_constant) uniform Parameters { uint width; uint height; } parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x >= parameters.width || y >= parameters.height) return;
    const vec2 uv = (vec2(x, y) + vec2(0.5)) /
        vec2(parameters.width, parameters.height);
    const vec3 rgb = texture(source_texture, uv).rgb;
    const float means[3] = float[3](0.485, 0.456, 0.406);
    const float deviations[3] = float[3](0.229, 0.224, 0.225);
    const uint pixel = y * parameters.width + x;
    const uint plane = parameters.width * parameters.height;
    destination.values[pixel] = (rgb.r - means[0]) / deviations[0];
    destination.values[plane + pixel] = (rgb.g - means[1]) / deviations[1];
    destination.values[2 * plane + pixel] = (rgb.b - means[2]) / deviations[2];
}
