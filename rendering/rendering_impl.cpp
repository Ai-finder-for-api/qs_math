
// qs_math/rendering/rendering_impl.cpp (C++20)
#include "include/qs_math_rendering.hpp"
#include <cmath>
#include <algorithm>
namespace qs::math::render {

Vec3<f32> cook_torrance(Vec3<f32> N,Vec3<f32> V,Vec3<f32> L,Vec3<f32> F0,f32 roughness) noexcept {
    const Vec3<f32> H=(V+L).normalized();
    const f32 NdV=std::max(0.0f,N.dot(V));
    const f32 NdL=std::max(0.0f,N.dot(L));
    const f32 HdV=std::max(0.0f,H.dot(V));
    const f32 D=ggx_ndf(N,H,roughness);
    const f32 G=smith_g(NdV,NdL,roughness);
    const Vec3<f32> F=fresnel_schlick(HdV,F0);
    const f32 denom=4.0f*NdV*NdL+1e-4f;
    return F*(D*G/denom);
}
f32 disney_diffuse(f32 NdV,f32 NdL,f32 LdH,f32 roughness) noexcept {
    const f32 fd90=0.5f+2.0f*roughness*LdH*LdH;
    const f32 f_in =1+(fd90-1)*std::pow(1-NdL,5.0f);
    const f32 f_out=1+(fd90-1)*std::pow(1-NdV,5.0f);
    return f_in*f_out*INV_PI;
}
f32 oren_nayar(Vec3<f32> N,Vec3<f32> V,Vec3<f32> L,f32 sigma) noexcept {
    const f32 s2=sigma*sigma;
    const f32 A=1-0.5f*s2/(s2+0.33f);
    const f32 B=0.45f*s2/(s2+0.09f);
    const f32 NdL=std::max(0.0f,N.dot(L));
    const f32 NdV=std::max(0.0f,N.dot(V));
    const f32 theta_i=std::acos(NdL),theta_r=std::acos(NdV);
    const f32 alpha=std::max(theta_i,theta_r),beta=std::min(theta_i,theta_r);
    // Project V and L onto plane perp to N
    Vec3<f32> Lp=(L-N*NdL).length_sq()>1e-6f?(L-N*NdL).normalized():Vec3<f32>{1,0,0};
    Vec3<f32> Vp=(V-N*NdV).length_sq()>1e-6f?(V-N*NdV).normalized():Vec3<f32>{1,0,0};
    const f32 cos_phi=std::max(0.0f,Lp.dot(Vp));
    return INV_PI*NdL*(A+B*cos_phi*std::sin(alpha)*std::tan(beta));
}
Vec3<f32> sample_ggx(Vec3<f32> N,f32 roughness,f32 u1,f32 u2) noexcept {
    const f32 a=roughness*roughness;
    const f32 phi=2*PI_F*u1;
    const f32 cos_theta=std::sqrt((1-u2)/(1+(a*a-1)*u2));
    const f32 sin_theta=std::sqrt(1-cos_theta*cos_theta);
    const Vec3<f32> H{sin_theta*std::cos(phi),sin_theta*std::sin(phi),cos_theta};
    // Build TBN around N
    Vec3<f32> up=std::abs(N.z)<0.999f?Vec3<f32>{0,0,1}:Vec3<f32>{1,0,0};
    Vec3<f32> T=(up.cross(N)).normalized(), B=N.cross(T);
    return (T*H.x+B*H.y+N*H.z).normalized();
}
f32 pdf_ggx(Vec3<f32> N,Vec3<f32> H,Vec3<f32> V,f32 roughness) noexcept {
    const f32 NdH=std::max(0.0f,N.dot(H));
    const f32 HdV=std::max(0.0f,H.dot(V));
    return ggx_ndf(N,H,roughness)*NdH/(4.0f*HdV+1e-4f);
}
Vec3<f32> linear_to_srgb(Vec3<f32> c) noexcept {
    auto f=[](f32 v)->f32{return v<=0.0031308f?12.92f*v:1.055f*std::pow(v,1.0f/2.4f)-0.055f;};
    return{f(c.x),f(c.y),f(c.z)};
}
Vec3<f32> srgb_to_linear(Vec3<f32> c) noexcept {
    auto f=[](f32 v)->f32{return v<=0.04045f?v/12.92f:std::pow((v+0.055f)/1.055f,2.4f);};
    return{f(c.x),f(c.y),f(c.z)};
}
Vec3<f32> rgb_to_hsv(Vec3<f32> c) noexcept {
    const f32 mx=std::max({c.x,c.y,c.z}),mn=std::min({c.x,c.y,c.z}),d=mx-mn;
    f32 h=0;
    if(d>1e-7f){
        if(mx==c.x) h=std::fmod((c.y-c.z)/d,6.0f);
        else if(mx==c.y) h=(c.z-c.x)/d+2;
        else h=(c.x-c.y)/d+4;
        h/=6.0f; if(h<0)h+=1;
    }
    return{h,mx>1e-7f?d/mx:0,mx};
}
Vec3<f32> hsv_to_rgb(Vec3<f32> hsv) noexcept {
    const f32 h=hsv.x*6,s=hsv.y,v=hsv.z;
    const i32 i=(i32)h; const f32 f=h-i,p=v*(1-s),q=v*(1-s*f),t=v*(1-s*(1-f));
    switch(i%6){
    case 0:return{v,t,p};case 1:return{q,v,p};case 2:return{p,v,t};
    case 3:return{p,q,v};case 4:return{t,p,v};default:return{v,p,q};}
}
Vec3<f32> rgb_to_hsl(Vec3<f32> c) noexcept {
    auto hsv=rgb_to_hsv(c);
    const f32 l=(2-hsv.y)*hsv.z/2;
    const f32 s=l<1e-7f||l>1-1e-7f?0:hsv.y*hsv.z/(l<0.5f?2*l:2-2*l);
    return{hsv.x,s,l};
}
Vec3<f32> hsl_to_rgb(Vec3<f32> hsl) noexcept {
    const f32 s2=hsl.y*(hsl.z<0.5f?hsl.z:1-hsl.z);
    return hsv_to_rgb({hsl.x,s2>1e-7f?2*s2/(hsl.z+s2):0,hsl.z+s2});
}
Vec3<f32> rgb_to_ycbcr(Vec3<f32> c) noexcept {
    return{0.299f*c.x+0.587f*c.y+0.114f*c.z,
           -0.169f*c.x-0.331f*c.y+0.5f*c.z+0.5f,
            0.5f*c.x-0.419f*c.y-0.081f*c.z+0.5f};
}
Vec3<f32> ycbcr_to_rgb(Vec3<f32> y) noexcept {
    const f32 Y=y.x,Cb=y.y-0.5f,Cr=y.z-0.5f;
    return{Y+1.402f*Cr,Y-0.344f*Cb-0.714f*Cr,Y+1.772f*Cb};
}
Vec3<f32> rgb_to_xyz(Vec3<f32> c) noexcept {
    Vec3<f32> l=srgb_to_linear(c);
    return{0.4124f*l.x+0.3576f*l.y+0.1805f*l.z,
           0.2126f*l.x+0.7152f*l.y+0.0722f*l.z,
           0.0193f*l.x+0.1192f*l.y+0.9505f*l.z};
}
Vec3<f32> xyz_to_rgb(Vec3<f32> xyz) noexcept {
    Vec3<f32> l{3.2406f*xyz.x-1.5372f*xyz.y-0.4986f*xyz.z,
               -0.9689f*xyz.x+1.8758f*xyz.y+0.0415f*xyz.z,
                0.0557f*xyz.x-0.2040f*xyz.y+1.0570f*xyz.z};
    return linear_to_srgb(l);
}
static const i32 BAYER4[4][4]={{0,8,2,10},{12,4,14,6},{3,11,1,9},{15,7,13,5}};
f32 bayer_dither(i32 x,i32 y,f32 value) noexcept {
    const f32 thresh=(BAYER4[y&3][x&3]+0.5f)/16.0f;
    return value>thresh?1.0f:0.0f;
}
Vec3<f32> ordered_dither_rgb(Vec3<f32> c,i32 x,i32 y) noexcept {
    const f32 thresh=(BAYER4[y&3][x&3]+0.5f)/16.0f-0.5f;
    return{std::max(0.0f,std::min(1.0f,c.x+thresh)),
           std::max(0.0f,std::min(1.0f,c.y+thresh)),
           std::max(0.0f,std::min(1.0f,c.z+thresh))};
}
Vec2<f32> dir_to_equirect(Vec3<f32> d) noexcept {
    return{0.5f+std::atan2(d.z,d.x)/(2*PI_F),
           0.5f+std::asin(std::max(-1.0f,std::min(1.0f,d.y)))/PI_F};
}
Vec3<f32> equirect_to_dir(Vec2<f32> uv) noexcept {
    const f32 phi=(uv.x-0.5f)*2*PI_F,theta=(uv.y-0.5f)*PI_F;
    return{std::cos(theta)*std::cos(phi),std::sin(theta),std::cos(theta)*std::sin(phi)};
}
Vec2<f32> dir_to_octahedron(Vec3<f32> d) noexcept {
    const f32 l=std::abs(d.x)+std::abs(d.y)+std::abs(d.z);
    Vec2<f32> p{d.x/l,d.y/l};
    if(d.z<0){p={( 1-std::abs(p.y))*(p.x>=0?1:-1),(1-std::abs(p.x))*(p.y>=0?1:-1)};}
    return p*0.5f+Vec2<f32>{0.5f,0.5f};
}
Vec3<f32> octahedron_to_dir(Vec2<f32> uv) noexcept {
    Vec2<f32> p=uv*2-Vec2<f32>{1,1};
    Vec3<f32> d{p.x,p.y,1-std::abs(p.x)-std::abs(p.y)};
    if(d.z<0){d.x=(1-std::abs(d.y))*(d.x>=0?1:-1);d.y=(1-std::abs(d.x))*(d.y>=0?1:-1);}
    return d.normalized();
}
f32 pcf_shadow(f32 depth,f32 shadow_depth,f32 bias) noexcept {
    return depth-bias>shadow_depth?0.0f:1.0f;
}
f32 vsm_shadow(Vec2<f32> moments,f32 depth) noexcept {
    if(depth<=moments.x) return 1.0f;
    const f32 var=moments.y-moments.x*moments.x;
    const f32 d=depth-moments.x;
    return var/(var+d*d);
}
Vec3<f32> tone_agx(Vec3<f32> x) noexcept {
    // AgX tone mapping (Sobotka)
    static const f32 mat[9]={0.842479062253094f,0.0423282422610123f,0.0423756549057051f,
        0.0784335999999992f,0.878468636469772f,0.0784336f,
        0.0792237451477643f,0.0791661274605434f,0.879142973975584f};
    Vec3<f32> v{mat[0]*x.x+mat[1]*x.y+mat[2]*x.z,
                mat[3]*x.x+mat[4]*x.y+mat[5]*x.z,
                mat[6]*x.x+mat[7]*x.y+mat[8]*x.z};
    // Sigmoid + log compression
    auto agx_default=[](f32 v2)->f32{
        v2=std::max(1e-10f,v2);
        const f32 l=std::log2(v2);
        const f32 t=std::max(0.0f,std::min(1.0f,(l+12.47393f)/(12.47393f+4.026069f)));
        return t*t*(3-2*t);
    };
    return{agx_default(v.x),agx_default(v.y),agx_default(v.z)};
}
} // namespace qs::math::render
