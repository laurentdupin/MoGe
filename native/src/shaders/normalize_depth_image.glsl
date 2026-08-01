#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(binding=0,r32f) uniform writeonly image2D output_image;
layout(std430,binding=1) readonly buffer Source { float data[]; } source_data;
layout(std430,binding=2) readonly buffer Range { float data[]; } range_data;
layout(push_constant) uniform Parameters { uint width; uint height; } p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;if(x>=p.width||y>=p.height)return;float lo=range_data.data[0],r=range_data.data[1]-lo;float v=r>0.0?1.0-(source_data.data[y*p.width+x]-lo)/r:0.0;imageStore(output_image,ivec2(x,y),vec4(v));}
