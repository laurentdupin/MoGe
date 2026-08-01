/*
 * Tensor indexing helpers adapted from PyTorch's Vulkan backend.
 * See THIRD_PARTY_NOTICES.md.
 */
uvec4 idx_to_coord(const uint idx, const uvec4 strides, const uvec4 sizes) {
    return uvec4(
        (idx / strides.x) % sizes.x,
        (idx / strides.y) % sizes.y,
        (idx / strides.z) % sizes.z,
        (idx / strides.w) % sizes.w);
}

uint coord_to_idx(const uvec4 coord, const uvec4 strides) {
    const uvec4 linear_terms = coord * strides;
    return linear_terms.x + linear_terms.y + linear_terms.z + linear_terms.w;
}
