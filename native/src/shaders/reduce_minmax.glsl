#version 450 core
layout(local_size_x=256,local_size_y=1,local_size_z=1) in;
layout(std430,binding=0) readonly buffer Source { float data[]; } source_data;
layout(std430,binding=1) writeonly buffer Range { float data[]; } range_data;
layout(push_constant) uniform Parameters { uint count; } p;
shared float minima[256]; shared float maxima[256];
void main(){
 uint lane=gl_LocalInvocationID.x;float lo=1.0/0.0,hi=-1.0/0.0;
 for(uint i=lane;i<p.count;i+=256u){float v=source_data.data[i];lo=min(lo,v);hi=max(hi,v);}
 minima[lane]=lo;maxima[lane]=hi;barrier();
 for(uint step=128u;step>0u;step>>=1u){if(lane<step){minima[lane]=min(minima[lane],minima[lane+step]);maxima[lane]=max(maxima[lane],maxima[lane+step]);}barrier();}
 if(lane==0u){range_data.data[0]=minima[0];range_data.data[1]=maxima[0];}
}
