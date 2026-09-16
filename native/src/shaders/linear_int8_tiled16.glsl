#version 460
#extension GL_EXT_integer_dot_product : require

#define K_PACKED 16
#define K_STRIDE 17
#include "linear_int8_tiled_common.glsl"
