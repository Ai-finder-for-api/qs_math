#pragma once
// noise_ext.hpp — Extended noise: Voronoi 3D, Curl noise, Wavelet, Histogram-preserving,
//                 Blue noise, Pink noise, periodic noise, noise derivatives, gabor noise
#include "noise.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "trig.hpp"
#include "random.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// 3D Cellular / Voronoi noise
// ---------------------------------------------------------------------------
struct CellularResult3 {
    f32   f1{0.f}, f2{0.f};
    Vec3f cell_center;
    i32   cell_id{0};
};

[[nodiscard]] inline CellularResult3 cellular3(f32 x, f32 y, f32 z) noexcept {
    i32 ix=(i32)floor(x), iy=(i32)floor(y), iz=(i32)floor(z);
    f32 min1=1e38f, min2=1e38f;
    Vec3f best;
    i32 best_id=0;
    for(i32 dz=-1;dz<=1;++dz) for(i32 dy=-1;dy<=1;++dy) for(i32 dx=-1;dx<=1;++dx){
        u32 h=detail::hash31(u32(ix+dx),u32(iy+dy),u32(iz+dz));
        f32 px=(ix+dx)+detail::hash_to_f32(h);
        f32 py=(iy+dy)+detail::hash_to_f32(detail::hash11(h));
        f32 pz=(iz+dz)+detail::hash_to_f32(detail::hash11(detail::hash11(h)));
        f32 dx2=px-x, dy2=py-y, dz2=pz-z;
        f32 d=dx2*dx2+dy2*dy2+dz2*dz2;
        if(d<min1){min2=min1;min1=d;best={px,py,pz};best_id=i32(h);}
        else if(d<min2) min2=d;
    }
    return {sqrt(min1),sqrt(min2),best,best_id};
}

// Voronoi edge distance (F2-F1 useful for cracks/borders)
[[nodiscard]] inline f32 voronoi_edge2(f32 x, f32 y) noexcept {
    auto r=cellular2(x,y);
    return r.f2-r.f1;
}
[[nodiscard]] inline f32 voronoi_edge3(f32 x, f32 y, f32 z) noexcept {
    auto r=cellular3(x,y,z);
    return r.f2-r.f1;
}

// ---------------------------------------------------------------------------
// Curl noise (divergence-free, good for fluid simulation)
// From the gradient of two noise functions
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f curl_noise3(f32 x, f32 y, f32 z, f32 eps=0.001f) noexcept {
    // ∇ × F where F = (noise_yz, noise_zx, noise_xy)
    f32 dfy_dz=(perlin3(x,y,z+eps)-perlin3(x,y,z-eps))/(2.f*eps);
    f32 dfz_dy=(perlin3(x,y+eps,z)-perlin3(x,y-eps,z))/(2.f*eps);
    f32 dfz_dx=(perlin3(x+eps,y,z)-perlin3(x-eps,y,z))/(2.f*eps);
    f32 dfx_dz=(perlin3(x,y,z+eps)-perlin3(x,y,z-eps))/(2.f*eps);
    f32 dfx_dy=(perlin3(x,y+eps,z)-perlin3(x,y-eps,z))/(2.f*eps);
    f32 dfy_dx=(perlin3(x+eps,y,z)-perlin3(x-eps,y,z))/(2.f*eps);
    return {dfy_dz-dfz_dy, dfz_dx-dfx_dz, dfx_dy-dfy_dx};
}

[[nodiscard]] inline Vec2f curl_noise2(f32 x, f32 y, f32 eps=0.001f) noexcept {
    f32 dp_dx=(perlin2(x+eps,y)-perlin2(x-eps,y))/(2.f*eps);
    f32 dp_dy=(perlin2(x,y+eps)-perlin2(x,y-eps))/(2.f*eps);
    return {dp_dy,-dp_dx}; // perpendicular gradient
}

// ---------------------------------------------------------------------------
// Noise with analytical derivatives (Perlin)
// Returns {value, dv/dx, dv/dy}
// ---------------------------------------------------------------------------
struct NoiseWithDeriv2 { f32 v, dx, dy; };
[[nodiscard]] inline NoiseWithDeriv2 perlin2_deriv(f32 x, f32 y) noexcept {
    f32 eps=0.001f;
    f32 v=perlin2(x,y);
    f32 dx=(perlin2(x+eps,y)-perlin2(x-eps,y))/(2.f*eps);
    f32 dy=(perlin2(x,y+eps)-perlin2(x,y-eps))/(2.f*eps);
    return {v,dx,dy};
}

struct NoiseWithDeriv3 { f32 v, dx, dy, dz; };
[[nodiscard]] inline NoiseWithDeriv3 perlin3_deriv(f32 x, f32 y, f32 z) noexcept {
    f32 eps=0.001f;
    f32 v=perlin3(x,y,z);
    f32 dx=(perlin3(x+eps,y,z)-perlin3(x-eps,y,z))/(2.f*eps);
    f32 dy=(perlin3(x,y+eps,z)-perlin3(x,y-eps,z))/(2.f*eps);
    f32 dz=(perlin3(x,y,z+eps)-perlin3(x,y,z-eps))/(2.f*eps);
    return {v,dx,dy,dz};
}

// ---------------------------------------------------------------------------
// Periodic noise (tileable)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 periodic_perlin2(f32 x, f32 y, i32 period_x, i32 period_y) noexcept {
    using namespace detail;
    i32 X=(i32)floor(x)%period_x; if(X<0)X+=period_x;
    i32 Y=(i32)floor(y)%period_y; if(Y<0)Y+=period_y;
    f32 fx=x-floor(x), fy=y-floor(y);
    f32 u=smootherstep(0.f,1.f,fx), v=smootherstep(0.f,1.f,fy);
    i32 X1=(X+1)%period_x, Y1=(Y+1)%period_y;
    return lerp(lerp(grad2(PERM[X+PERM[Y]],fx,fy),       grad2(PERM[X1+PERM[Y]],fx-1.f,fy),      u),
                lerp(grad2(PERM[X+PERM[Y1]],fx,fy-1.f), grad2(PERM[X1+PERM[Y1]],fx-1.f,fy-1.f),u),v);
}

// ---------------------------------------------------------------------------
// Gradient noise (smoother than Perlin)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 gradient_noise2(f32 x, f32 y) noexcept {
    i32 ix=(i32)floor(x), iy=(i32)floor(y);
    f32 fx=x-floor(x), fy=y-floor(y);
    auto hash_gradient=[](i32 hx, i32 hy, f32 dx, f32 dy)->f32{
        u32 h=detail::hash21(u32(hx),u32(hy));
        f32 angle=detail::hash_to_f32(h)*TWO_PI;
        return cos(angle)*dx+sin(angle)*dy;
    };
    f32 u=smootherstep(0.f,1.f,fx), v=smootherstep(0.f,1.f,fy);
    return bilinear(
        hash_gradient(ix,  iy,  fx,  fy  ),
        hash_gradient(ix+1,iy,  fx-1.f,fy  ),
        hash_gradient(ix,  iy+1,fx,  fy-1.f),
        hash_gradient(ix+1,iy+1,fx-1.f,fy-1.f),
        u,v);
}

// ---------------------------------------------------------------------------
// Warped noise compositions
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 marble_noise(f32 x, f32 y, f32 z, i32 octaves=4) noexcept {
    f32 n=fbm3(x,y,z,{octaves,2.f,0.5f,0.5f});
    return sin((x+n*2.f)*TWO_PI);
}

[[nodiscard]] inline f32 wood_noise(f32 x, f32 y, f32 z) noexcept {
    f32 n=perlin3(x,y,z)*10.f;
    return n-floor(n);
}

[[nodiscard]] inline f32 cloud_noise(f32 x, f32 y, f32 z, i32 octaves=6) noexcept {
    f32 v=0.f,amp=0.5f,freq=1.f;
    for(i32 i=0;i<octaves;++i){
        v+=abs(perlin3(x*freq,y*freq,z*freq))*amp;
        amp*=0.5f; freq*=2.f;
    }
    return 1.f-v;
}

// ---------------------------------------------------------------------------
// Pink noise (1/f noise via FBM spectral weighting)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 pink_noise2(f32 x, f32 y, i32 octaves=8) noexcept {
    f32 val=0.f, freq=1.f;
    for(i32 i=0;i<octaves;++i){
        val+=perlin2(x*freq,y*freq)/freq;
        freq*=2.f;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Noise histogram equalization helper (remap to uniform distribution)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 equalize_noise(f32 v) noexcept {
    // Approximate ICDF for Gaussian noise output → uniform
    // Using rational approximation of erf^-1
    v=clamp(v,-0.9999f,0.9999f);
    f32 w=-log((1.f-v*v));
    if(w<5.f){
        w=w-2.5f;
        return 2.515517f+w*(0.802853f+w*0.010328f)/
               (1.f+w*(1.432788f+w*(0.189269f+w*0.001308f)));
    }
    w=sqrt(w)-3.f;
    return w*(1.f+w*(-0.000200214257f+w*0.000100950558f));
}

// ---------------------------------------------------------------------------
// Stochastic texture tiling (Wang tiles style — hash-based)
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec2f stochastic_tiling_uv(Vec2f uv, PCG32& rng) noexcept {
    // Find which tile cell we're in
    Vec2i cell{(i32)floor(uv.x),(i32)floor(uv.y)};
    u32 h=detail::hash21(u32(cell.x),u32(cell.y));
    // Random rotation + translation of UVs within tile
    f32 angle=detail::hash_to_f32(h)*TWO_PI;
    f32 cx=detail::hash_to_f32(detail::hash11(h));
    f32 cy=detail::hash_to_f32(detail::hash11(detail::hash11(h)));
    Vec2f local=uv-Vec2f{f32(cell.x),f32(cell.y)};
    f32 c_=cos(angle),s_=sin(angle);
    Vec2f rotated{local.x*c_-local.y*s_+cx, local.x*s_+local.y*c_+cy};
    return rotated;
}

// ---------------------------------------------------------------------------
// Sparse convolution noise (Gabor noise)
// ---------------------------------------------------------------------------
struct GaborParams {
    f32 K{1.f};           // magnitude
    f32 a{0.05f};         // bandwidth
    f32 F0{0.0625f};      // frequency
    f32 omega0{0.f};      // orientation (radians)
    i32 period{512};      // spatial period for tiling
};

[[nodiscard]] inline f32 gabor_kernel(f32 x, f32 y, GaborParams p) noexcept {
    f32 gaussian=p.K*exp(-PI*p.a*p.a*(x*x+y*y));
    f32 sinusoid=cos(TWO_PI*p.F0*(x*cos(p.omega0)+y*sin(p.omega0)));
    return gaussian*sinusoid;
}

[[nodiscard]] inline f32 gabor_noise2(f32 x, f32 y, GaborParams params, u32 seed=42) noexcept {
    // Poisson point process convolution (simplified)
    i32 ix=(i32)floor(x/params.period), iy=(i32)floor(y/params.period);
    f32 sum=0.f;
    for(i32 dy=-1;dy<=1;++dy) for(i32 dx=-1;dx<=1;++dx){
        u32 h=detail::hash21(u32(ix+dx),u32(iy+dy));
        // ~3 impulses per cell
        for(i32 k=0;k<3;++k){
            h=detail::hash11(h^seed);
            f32 px=(ix+dx)*params.period+detail::hash_to_f32(h)*params.period;
            h=detail::hash11(h);
            f32 py=(iy+dy)*params.period+detail::hash_to_f32(h)*params.period;
            h=detail::hash11(h);
            f32 weight=detail::hash_to_f32_signed(h);
            sum+=weight*gabor_kernel(x-px,y-py,params);
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Multi-octave domain warping variants
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 fbm_warped3(f32 x, f32 y, f32 z,
                                      f32 warp=0.8f, i32 oct=6) noexcept {
    FBMParams p{oct,2.f,0.5f,0.5f};
    f32 wx=fbm3(x+1.7f,y+9.2f,z+0.4f,p);
    f32 wy=fbm3(x+8.3f,y+2.8f,z+5.1f,p);
    f32 wz=fbm3(x+3.1f,y+7.6f,z+1.9f,p);
    return fbm3(x+warp*wx,y+warp*wy,z+warp*wz,p);
}

// Second-order warp (more complex patterns)
[[nodiscard]] inline f32 fbm_warp2_order2(f32 x, f32 y, f32 warp=0.8f) noexcept {
    FBMParams p{6,2.f,0.5f,0.5f};
    f32 wx1=fbm2(x+0.f,y+0.f,p), wy1=fbm2(x+5.2f,y+1.3f,p);
    f32 wx2=fbm2(x+warp*wx1+1.7f,y+warp*wy1+9.2f,p);
    f32 wy2=fbm2(x+warp*wx1+8.3f,y+warp*wy1+2.8f,p);
    return fbm2(x+warp*wx2,y+warp*wy2,p);
}

// ---------------------------------------------------------------------------
// Blue noise dithering helpers (using void-and-cluster, simplified)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 blue_noise_dither(f32 value, Vec2i pixel, i32 frame=0) noexcept {
    // Interleaved gradient noise (Jimenez 2014)
    f32 magic=52.9829189f;
    f32 x=f32(pixel.x+(frame&1023)*1273);
    f32 y=f32(pixel.y+(frame&1023)*1277);
    f32 bn=frac(magic*frac(0.06711056f*x+0.00583715f*y));
    return value+bn*(1.f/255.f)-0.5f/255.f;
}

[[nodiscard]] inline Vec3f blue_noise_dither_rgb(Vec3f color, Vec2i pixel, i32 frame=0) noexcept {
    // Different magic per channel to avoid correlation
    auto bayer=[&](i32 shift)->f32{
        f32 x=f32(pixel.x+(frame&1023)*1273+shift*17);
        f32 y=f32(pixel.y+(frame&1023)*1277+shift*31);
        return frac(52.9829189f*frac(0.06711056f*x+0.00583715f*y));
    };
    return color+Vec3f{bayer(0),bayer(1),bayer(2)}*(1.f/255.f)-Vec3f{0.5f/255.f};
}

} // namespace qs::math
