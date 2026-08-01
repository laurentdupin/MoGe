#version 450 core
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
layout(std430,binding=0) readonly buffer Source { float data[]; } source_data;
layout(std430,binding=1) writeonly buffer Destination { float data[]; } destination_data;
layout(push_constant) uniform Parameters { uint sw; uint sh; uint width; uint height; } p;
float value(uint x,uint y){return source_data.data[y*p.sw+x];}
void main(){
 uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;if(x>=p.width||y>=p.height)return;
 float fx=p.width>1u?float(x)*float(p.sw-1u)/float(p.width-1u):0.0;
 float fy=p.height>1u?float(y)*float(p.sh-1u)/float(p.height-1u):0.0;
 uint x0=uint(floor(fx)),y0=uint(floor(fy)),x1=min(x0+1u,p.sw-1u),y1=min(y0+1u,p.sh-1u);
 float top=mix(value(x0,y0),value(x1,y0),fx-float(x0));
 float bottom=mix(value(x0,y1),value(x1,y1),fx-float(x0));
 destination_data.data[y*p.width+x]=mix(top,bottom,fy-float(y0));
}
