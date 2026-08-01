#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(binding = 0) uniform sampler2D source_texture;
layout(std430, binding = 1) writeonly buffer Destination { uint values[]; } destination_data;
layout(push_constant) uniform Parameters {
    uint source_width;
    uint source_height;
    uint destination_width;
    uint destination_height;
} parameters;

float channel_value(ivec2 coordinate, uint channel) {
    const vec4 pixel = texelFetch(source_texture, coordinate, 0);
    return (channel == 0 ? pixel.b : (channel == 1 ? pixel.g : pixel.r)) * 255.0;
}
float cubic1(float x) { const float a=-0.75; return ((a+2.0)*x-(a+3.0))*x*x+1.0; }
float cubic2(float x) { const float a=-0.75; return ((a*x-5.0*a)*x+8.0*a)*x-4.0*a; }
float coefficient(int tap, float fraction) {
    if (tap == 0) return cubic2(fraction + 1.0);
    if (tap == 1) return cubic1(fraction);
    if (tap == 2) return cubic1(1.0 - fraction);
    return cubic2(2.0 - fraction);
}
float cubic_sample(uint dx, uint dy, uint channel) {
    const float sx=(float(dx)+0.5)*float(parameters.source_width)/float(parameters.destination_width)-0.5;
    const float sy=(float(dy)+0.5)*float(parameters.source_height)/float(parameters.destination_height)-0.5;
    const int bx=int(floor(sx)); const int by=int(floor(sy));
    const float fx=sx-float(bx); const float fy=sy-float(by);
    float result=0.0;
    for(int ty=0;ty<4;++ty){
        const int iy=clamp(by-1+ty,0,int(parameters.source_height)-1);
        float row=0.0;
        for(int tx=0;tx<4;++tx){
            const int ix=clamp(bx-1+tx,0,int(parameters.source_width)-1);
            row += channel_value(ivec2(ix,iy),channel)*coefficient(tx,fx);
        }
        result += row*coefficient(ty,fy);
    }
    return result;
}
float area_sample(uint dx, uint dy, uint channel) {
    const float scale_x=float(parameters.source_width)/float(parameters.destination_width);
    const float scale_y=float(parameters.source_height)/float(parameters.destination_height);
    const float xb=float(dx)*scale_x; const float xe=float(dx+1)*scale_x;
    const float yb=float(dy)*scale_y; const float ye=float(dy+1)*scale_y;
    float sum=0.0;
    for(int iy=int(floor(yb));iy<int(ceil(ye));++iy){
        if(iy<0||iy>=int(parameters.source_height)) continue;
        const float wy=max(0.0,min(ye,float(iy+1))-max(yb,float(iy)));
        for(int ix=int(floor(xb));ix<int(ceil(xe));++ix){
            if(ix<0||ix>=int(parameters.source_width)) continue;
            const float wx=max(0.0,min(xe,float(ix+1))-max(xb,float(ix)));
            sum += channel_value(ivec2(ix,iy),channel)*wx*wy;
        }
    }
    return sum/(scale_x*scale_y);
}
void main(){
    const uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;
    if(x>=parameters.destination_width||y>=parameters.destination_height) return;
    const bool upscale=parameters.destination_width>parameters.source_width||
        parameters.destination_height>parameters.source_height;
    uint packed=0;
    for(uint channel=0;channel<3;++channel){
        const float resized=upscale?cubic_sample(x,y,channel):area_sample(x,y,channel);
        const uint byte_value=uint(clamp(roundEven(resized),0.0,255.0));
        packed |= byte_value << (channel*8);
    }
    destination_data.values[y*parameters.destination_width+x]=packed;
}
