
// qs_math/noise/noise_ext_impl.cpp (C++20)
#include "include/qs_math_noise.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
namespace qs::math::noise {
constexpr f32 PI = 3.14159265358979323846f;

// ─── Hash utilities ───────────────────────────────────────────────────────────
static u32 hash2(i32 x, i32 y, u32 seed) noexcept {
    u32 h=(u32)x*1619u^(u32)y*31337u^seed*6971u;
    h=(h^(h>>16))*0x45d9f3b; h=(h^(h>>16))*0x45d9f3b; return h^(h>>16);
}
static u32 hash3(i32 x,i32 y,i32 z,u32 seed) noexcept {
    u32 h=(u32)x*1619u^(u32)y*31337u^(u32)z*6971u^seed*1013u;
    h=(h^(h>>16))*0x45d9f3b;h=(h^(h>>16))*0x45d9f3b;return h^(h>>16);
}
static f32 hash_f(u32 h) noexcept { return (h&0xffffff)*(1.0f/16777216.0f); }

// ─── Value noise ─────────────────────────────────────────────────────────────
f32 value_noise_2d(f32 x, f32 y, u32 seed) noexcept {
    i32 xi=(i32)std::floor(x),yi=(i32)std::floor(y);
    f32 fx=x-xi,fy=y-yi;
    auto s=[](f32 t){return t*t*t*(t*(t*6-15)+10);};
    f32 ux=s(fx),uy=s(fy);
    f32 v00=hash_f(hash2(xi,yi,seed));
    f32 v10=hash_f(hash2(xi+1,yi,seed));
    f32 v01=hash_f(hash2(xi,yi+1,seed));
    f32 v11=hash_f(hash2(xi+1,yi+1,seed));
    return v00*(1-ux)*(1-uy)+v10*ux*(1-uy)+v01*(1-ux)*uy+v11*ux*uy;
}
f32 value_noise_3d(f32 x,f32 y,f32 z,u32 seed) noexcept {
    i32 xi=(i32)std::floor(x),yi=(i32)std::floor(y),zi=(i32)std::floor(z);
    f32 fx=x-xi,fy=y-yi,fz=z-zi;
    auto s=[](f32 t){return t*t*t*(t*(t*6-15)+10);};
    f32 ux=s(fx),uy=s(fy),uz=s(fz);
    f32 v[8];
    for(i32 k=0;k<2;++k)for(i32 j=0;j<2;++j)for(i32 i=0;i<2;++i)
        v[k*4+j*2+i]=hash_f(hash3(xi+i,yi+j,zi+k,seed));
    f32 x0=v[0]*(1-ux)+v[1]*ux, x1=v[2]*(1-ux)+v[3]*ux;
    f32 x2=v[4]*(1-ux)+v[5]*ux, x3=v[6]*(1-ux)+v[7]*ux;
    f32 y0=x0*(1-uy)+x1*uy, y1=x2*(1-uy)+x3*uy;
    return y0*(1-uz)+y1*uz;
}
// ─── Worley 2D ────────────────────────────────────────────────────────────────
CellResult worley_2d(f32 x, f32 y, u32 seed) noexcept {
    i32 xi=(i32)std::floor(x),yi=(i32)std::floor(y);
    f32 F1=1e30f,F2=1e30f; Vec2<f32> cc{};
    for(i32 dy=-2;dy<=2;++dy)for(i32 dx=-2;dx<=2;++dx){
        i32 cx=xi+dx,cy=yi+dy;
        u32 h=hash2(cx,cy,seed);
        f32 px=cx+hash_f(h)*0.7f+0.15f;
        f32 py=cy+hash_f(h^0xdeadbeef)*0.7f+0.15f;
        f32 d=(x-px)*(x-px)+(y-py)*(y-py);
        if(d<F1){F2=F1;F1=d;cc={px,py};}
        else if(d<F2) F2=d;
    }
    return{std::sqrt(F1),std::sqrt(F2),cc};
}
f32 worley_3d_f1(f32 x,f32 y,f32 z,u32 seed) noexcept {
    i32 xi=(i32)std::floor(x),yi=(i32)std::floor(y),zi=(i32)std::floor(z);
    f32 F1=1e30f;
    for(i32 dz=-1;dz<=1;++dz)for(i32 dy=-1;dy<=1;++dy)for(i32 dx=-1;dx<=1;++dx){
        u32 h=hash3(xi+dx,yi+dy,zi+dz,seed);
        f32 px=xi+dx+hash_f(h)*0.8f+0.1f;
        f32 py=yi+dy+hash_f(h^0xc001c0de)*0.8f+0.1f;
        f32 pz=zi+dz+hash_f(h^0xdeadcafe)*0.8f+0.1f;
        f32 d=(x-px)*(x-px)+(y-py)*(y-py)+(z-pz)*(z-pz);
        F1=std::min(F1,d);
    }
    return std::sqrt(F1);
}
// Minimal Perlin gradient noise for internal use
static f32 perlin_grad2(u32 h, f32 x, f32 y) noexcept {
    switch(h&3){case 0:return x+y;case 1:return -x+y;case 2:return x-y;default:return -x-y;}
}
static f32 perlin2(f32 x, f32 y, u32 seed) noexcept {
    i32 xi=(i32)std::floor(x),yi=(i32)std::floor(y);
    f32 fx=x-xi,fy=y-yi;
    auto fade=[](f32 t){return t*t*t*(t*(t*6-15)+10);};
    f32 ux=fade(fx),uy=fade(fy);
    f32 g00=perlin_grad2(hash2(xi,yi,seed),fx,fy);
    f32 g10=perlin_grad2(hash2(xi+1,yi,seed),fx-1,fy);
    f32 g01=perlin_grad2(hash2(xi,yi+1,seed),fx,fy-1);
    f32 g11=perlin_grad2(hash2(xi+1,yi+1,seed),fx-1,fy-1);
    return g00*(1-ux)*(1-uy)+g10*ux*(1-uy)+g01*(1-ux)*uy+g11*ux*uy;
}
// ─── Domain warped FBM ───────────────────────────────────────────────────────
f32 fbm_warped(f32 x,f32 y,i32 oct,f32 warp) noexcept {
    // warp pass 1
    f32 qx=x+warp*value_noise_2d(x,y,0);
    f32 qy=y+warp*value_noise_2d(x+5.2f,y+1.3f,1);
    // warp pass 2
    f32 rx=x+warp*value_noise_2d(qx,qy,2);
    f32 ry=y+warp*value_noise_2d(qx+1.7f,qy+9.2f,3);
    // FBM
    f32 v=0,amp=0.5f,freq=1;
    for(i32 i=0;i<oct;++i){v+=amp*perlin2(rx*freq,ry*freq,i);amp*=0.5f;freq*=2;}
    return v;
}
// ─── Curl noise ──────────────────────────────────────────────────────────────
Vec2<f32> curl_noise_2d(f32 x,f32 y,f32 eps) noexcept {
    f32 dpy=(value_noise_2d(x,y+eps)-value_noise_2d(x,y-eps))/(2*eps);
    f32 dpx=(value_noise_2d(x+eps,y)-value_noise_2d(x-eps,y))/(2*eps);
    return{dpy,-dpx};
}
Vec3<f32> curl_noise_3d(f32 x,f32 y,f32 z,f32 eps) noexcept {
    auto p=[&](f32 a,f32 b,f32 c,u32 s){return value_noise_3d(a,b,c,s);};
    Vec3<f32> dPdx{
        (p(x,y+eps,z,1)-p(x,y-eps,z,1))/(2*eps),
        (p(x,y,z+eps,2)-p(x,y,z-eps,2))/(2*eps),
        (p(x+eps,y,z,0)-p(x-eps,y,z,0))/(2*eps)};
    Vec3<f32> dPdy{
        (p(x,y,z+eps,3)-p(x,y,z-eps,3))/(2*eps),
        (p(x+eps,y,z,1)-p(x-eps,y,z,1))/(2*eps),
        (p(x,y+eps,z,2)-p(x,y-eps,z,2))/(2*eps)};
    return{dPdx.y-dPdy.x,dPdx.z-dPdy.y,dPdx.x-dPdy.z};
}
// ─── Blue noise ──────────────────────────────────────────────────────────────
std::vector<Vec2<f32>> blue_noise_2d(i32 n,f32 w,f32 h,u32 seed) noexcept {
    // Mitchell's best-candidate: for each new point, try M candidates, keep farthest
    const i32 M=30;
    std::vector<Vec2<f32>> pts;
    pts.reserve(n);
    u32 rng=seed;
    auto rand01=[&]()->f32{rng=rng*1664525u+1013904223u;return (rng>>8)*(1.0f/16777216.0f);};
    pts.push_back({rand01()*w,rand01()*h});
    for(i32 i=1;i<n;++i){
        Vec2<f32> best{}; f32 best_d=0;
        for(i32 j=0;j<M;++j){
            Vec2<f32> c{rand01()*w,rand01()*h};
            f32 min_d=1e30f;
            for(auto&p:pts) min_d=std::min(min_d,(c-p).length_sq());
            if(min_d>best_d){best_d=min_d;best=c;}
        }
        pts.push_back(best);
    }
    return pts;
}
std::vector<f32> spectral_1d(i32 n,f32 beta,u32 seed) noexcept {
    // Generate white noise in frequency domain, shape by 1/f^beta, IFFT
    std::vector<std::complex<f64>> spec(n/2+1);
    u32 rng=seed;
    auto rand11=[&]()->f64{rng=rng*1664525u+1013904223u;return ((rng>>8)*(1.0/16777216.0))*2-1;};
    for(i32 k=1;k<(i32)spec.size();++k){
        const f64 power=std::pow((f64)k,-beta*0.5);
        const f64 ang=rand11()*3.14159265358979323846*2;
        spec[k]={power*std::cos(ang),power*std::sin(ang)};
    }
    spec[0]={0,0};
    // IRFFT via naive DFT (for simplicity — n is typically small here)
    std::vector<f32> out(n,0);
    for(i32 i=0;i<n;++i){
        f64 s=0;
        for(i32 k=1;k<(i32)spec.size();++k){
            s+=spec[k].real()*std::cos(2*3.14159265358979*k*i/n)
              -spec[k].imag()*std::sin(2*3.14159265358979*k*i/n);
        }
        out[i]=(f32)(s*2.0/n);
    }
    // Normalize to [-1,1]
    f32 mx=*std::max_element(out.begin(),out.end());
    f32 mn=*std::min_element(out.begin(),out.end());
    f32 range=mx-mn; if(range<1e-7f) return out;
    for(auto&v:out) v=(v-mn)/range*2-1;
    return out;
}
FBMResult fbm_deriv_2d(f32 x,f32 y,i32 oct,f32 lac,f32 gain) noexcept {
    f32 v=0,dx=0,dy=0,amp=0.5f,freq=1;
    for(i32 i=0;i<oct;++i){
        const f32 n=perlin2(x*freq,y*freq,(u32)i);
        const f32 ndx=(perlin2(x*freq+0.01f,y*freq,(u32)i)-perlin2(x*freq-0.01f,y*freq,(u32)i))/0.02f;
        const f32 ndy=(perlin2(x*freq,y*freq+0.01f,(u32)i)-perlin2(x*freq,y*freq-0.01f,(u32)i))/0.02f;
        v+=amp*n; dx+=amp*freq*ndx; dy+=amp*freq*ndy;
        amp*=gain; freq*=lac;
    }
    return{v,{dx,dy}};
}
} // namespace qs::math::noise
