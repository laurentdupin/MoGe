#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer Values { float data[]; } values;
layout(push_constant) uniform Parameters {
    uint width; uint height; uint channels; uint image_width; uint image_height;
} p;
void main() {
    uint index = gl_GlobalInvocationID.x;
    uint spatial = p.width * p.height;
    if (index >= p.channels * spatial) return;
    uint channel = index / spatial;
    uint position = index % spatial;
    uint x = position % p.width;
    uint y = position / p.width;
    uint direction_dim = p.channels / 2;
    uint frequency_count = direction_dim / 2;
    uint direction = channel / direction_dim;
    uint component = channel % direction_dim;
    uint frequency = component % frequency_count;
    float aspect = float(p.image_width) / float(p.image_height);
    float diagonal = sqrt(aspect * aspect + 1.0);
    float span = direction == 0 ? aspect / diagonal : 1.0 / diagonal;
    uint size = direction == 0 ? p.width : p.height;
    uint coordinate_index = direction == 0 ? x : y;
    float left = -span * float(size - 1) / float(size);
    float right = span * float(size - 1) / float(size);
    float coordinate = size == 1 ? left :
        left + (right - left) * float(coordinate_index) / float(size - 1);
    float omega = 1.0 / pow(100.0, float(frequency) / float(frequency_count));
    float angle = coordinate * omega;
    values.data[index] += 0.1 * (component < frequency_count ? sin(angle) : cos(angle));
}
