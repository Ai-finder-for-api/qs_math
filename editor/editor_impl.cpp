// qs_math/editor/editor_impl.cpp (C++20)
#include "include/qs_math_editor.hpp"
#include <cmath>
#include <algorithm>
namespace qs::math::editor {
Ray screen_to_ray(f32 sx,f32 sy,f32 sw,f32 sh,const Mat4<f32>& ivp) noexcept {
    // NDC in [-1,1]
    const f32 nx=(2*sx/sw)-1, ny=1-(2*sy/sh);
    // Unproject near and far
    auto unproj=[&](f32 z)->Vec3<f32>{
        Vec4<f32> clip{nx,ny,z,1};
        Vec4<f32> world=ivp*clip;
        const f32 w=world.w<1e-7f?1e-7f:world.w;
        return {world.x/w,world.y/w,world.z/w};
    };
    Vec3<f32> near=unproj(-1),far=unproj(1);
    Ray r; r.origin=near; r.dir=(far-near).normalized(); return r;
}
Vec3<f32> world_to_screen(Vec3<f32> world,const Mat4<f32>& vp,f32 sw,f32 sh) noexcept {
    Vec4<f32> clip=vp*Vec4<f32>{world.x,world.y,world.z,1};
    if(std::abs(clip.w)<1e-7f) return {0,0,-1};
    const f32 inv=1/clip.w;
    const f32 nx=clip.x*inv,ny=clip.y*inv,nz=clip.z*inv;
    return {(nx+1)*0.5f*sw,(1-ny)*0.5f*sh,nz};
}
f32 axis_handle_hit(Ray ray,Vec3<f32> origin,Vec3<f32> axis,f32 len,f32 rad) noexcept {
    // Closest point between ray and axis segment
    const Vec3<f32> axis_end=origin+axis*len;
    const Vec3<f32> d1=ray.dir, d2=axis;
    const Vec3<f32> r=ray.origin-origin;
    const f32 a=d1.dot(d1),e=d2.dot(d2),f2=d2.dot(r);
    const f32 c=d1.dot(r),b=d1.dot(d2);
    const f32 denom=a*e-b*b;
    f32 sc=std::abs(denom)<1e-7f?0:std::max(0.0f,std::min(1.0f,(b*f2-c*e)/denom));
    f32 tc=std::max(0.0f,std::min(1.0f,(b*sc+f2)/std::max(e,1e-7f)));
    const Vec3<f32> closest_ray=ray.origin+d1*sc;
    const Vec3<f32> closest_ax=origin+d2*tc;
    const f32 dist=(closest_ray-closest_ax).length();
    return dist<rad?sc:-1;
}
Vec4<f32> arcball_delta(Vec2<f32> p0,Vec2<f32> p1,f32 r,Vec3<f32> center) noexcept {
    auto to_sphere=[&](Vec2<f32> p)->Vec3<f32>{
        const f32 len2=p.x*p.x+p.y*p.y;
        if(len2<=r*r*0.5f) return {p.x,p.y,std::sqrt(r*r-len2)};
        return {p.x,p.y,r*r*0.5f/std::sqrt(len2)};
    };
    Vec3<f32> v0=to_sphere(p0).normalized(),v1=to_sphere(p1).normalized();
    Vec3<f32> ax=v0.cross(v1); const f32 angle=std::acos(std::max(-1.0f,std::min(1.0f,v0.dot(v1))));
    return {ax.x,ax.y,ax.z,angle};
}
std::array<f32,4> aabb_screen_rect(Vec3<f32> mn,Vec3<f32> mx,const Mat4<f32>& vp,f32 w,f32 h) noexcept {
    f32 x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
    for(i32 c=0;c<8;++c){
        Vec3<f32> corner{c&1?mx.x:mn.x,c&2?mx.y:mn.y,c&4?mx.z:mn.z};
        Vec3<f32> s=world_to_screen(corner,vp,w,h);
        x0=std::min(x0,s.x);y0=std::min(y0,s.y);
        x1=std::max(x1,s.x);y1=std::max(y1,s.y);
    }
    return {{x0,y0,x1,y1}};
}
} // namespace qs::math::editor
