#pragma once
// qs_math/noise.hpp — Perlin, Simplex, Value, Cellular, FBM, Ridged, Turbulence, Blue noise utils
#include "vec2.hpp"
#include "vec3.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Hash utilities
// ---------------------------------------------------------------------------
namespace detail {
    [[nodiscard]] constexpr u32 hash11(u32 x) noexcept {
        x=((x>>16)^x)*0x45d9f3b; x=((x>>16)^x)*0x45d9f3b; return (x>>16)^x;
    }
    [[nodiscard]] constexpr u32 hash21(u32 x, u32 y) noexcept {
        return hash11(x^hash11(y));
    }
    [[nodiscard]] constexpr u32 hash31(u32 x, u32 y, u32 z) noexcept {
        return hash11(x^hash11(y^hash11(z)));
    }
    [[nodiscard]] constexpr f32 hash_to_f32(u32 h) noexcept {
        return f32(h&0xFFFFFF)/f32(0xFFFFFF);
    }
    [[nodiscard]] constexpr f32 hash_to_f32_signed(u32 h) noexcept {
        return hash_to_f32(h)*2.f-1.f;
    }

    // Permutation table (classic Ken Perlin)
    inline constexpr u8 PERM[512] = {
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
        8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
        35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
        134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
        55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
        18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
        250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
        189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
        172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
        228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
        107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
        // repeat
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
        8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
        35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
        134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
        55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
        18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
        250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
        189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
        172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
        228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
        107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };

    // Gradient table for Perlin
    [[nodiscard]] constexpr f32 grad2(i32 hash, f32 x, f32 y) noexcept {
        i32 h=hash&7;
        f32 u=h<4?x:y, v=h<4?y:x;
        return ((h&1)?-u:u)+((h&2)?-v:v);
    }
    [[nodiscard]] constexpr f32 grad3(i32 hash, f32 x, f32 y, f32 z) noexcept {
        i32 h=hash&15;
        f32 u=h<8?x:y, v=h<4?y:(h==12||h==14?x:z);
        return ((h&1)?-u:u)+((h&2)?-v:v);
    }
}

// ---------------------------------------------------------------------------
// Perlin Noise 2D
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 perlin2(f32 x, f32 y) noexcept {
    using namespace detail;
    i32 X=(i32)floor(x)&255, Y=(i32)floor(y)&255;
    x-=floor(x); y-=floor(y);
    f32 u=smootherstep(0.f,1.f,x), v=smootherstep(0.f,1.f,y);
    i32 a=PERM[X]+Y, b=PERM[X+1]+Y;
    return lerp(lerp(grad2(PERM[a],  x,  y), grad2(PERM[b],  x-1.f,y  ),u),
                lerp(grad2(PERM[a+1],x,  y-1.f),grad2(PERM[b+1],x-1.f,y-1.f),u),v);
}

// ---------------------------------------------------------------------------
// Perlin Noise 3D
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 perlin3(f32 x, f32 y, f32 z) noexcept {
    using namespace detail;
    i32 X=(i32)floor(x)&255, Y=(i32)floor(y)&255, Z=(i32)floor(z)&255;
    x-=floor(x); y-=floor(y); z-=floor(z);
    f32 u=smootherstep(0.f,1.f,x), v=smootherstep(0.f,1.f,y), w=smootherstep(0.f,1.f,z);
    i32 a=PERM[X]+Y, aa=PERM[a]+Z, ab=PERM[a+1]+Z;
    i32 b=PERM[X+1]+Y, ba=PERM[b]+Z, bb=PERM[b+1]+Z;
    return lerp(lerp(lerp(grad3(PERM[aa],  x,  y,  z),grad3(PERM[ba],  x-1.f,y,  z  ),u),
                     lerp(grad3(PERM[ab],  x,  y-1.f,z),grad3(PERM[bb],x-1.f,y-1.f,z),u),v),
                lerp(lerp(grad3(PERM[aa+1],x,  y,  z-1.f),grad3(PERM[ba+1],x-1.f,y,z-1.f),u),
                     lerp(grad3(PERM[ab+1],x,  y-1.f,z-1.f),grad3(PERM[bb+1],x-1.f,y-1.f,z-1.f),u),v),w);
}

// ---------------------------------------------------------------------------
// Simplex Noise 2D
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 simplex2(f32 x, f32 y) noexcept {
    using namespace detail;
    static constexpr f32 F2=0.5f*(1.7320508f-1.f), G2=(3.f-1.7320508f)/6.f;
    f32 s=(x+y)*F2; i32 i=(i32)floor(x+s), j=(i32)floor(y+s);
    f32 t=(f32)(i+j)*G2, X0=i-t, Y0=j-t;
    f32 x0=x-X0, y0=y-Y0;
    i32 i1=x0>y0?1:0, j1=x0>y0?0:1;
    f32 x1=x0-i1+G2, y1=y0-j1+G2, x2=x0-1.f+2.f*G2, y2=y0-1.f+2.f*G2;
    i32 ii=i&255, jj=j&255;
    f32 n0=0.f,n1=0.f,n2=0.f;
    f32 t0=0.5f-x0*x0-y0*y0; if(t0>=0.f){t0*=t0; n0=t0*t0*grad2(PERM[ii+PERM[jj]],x0,y0);}
    f32 t1=0.5f-x1*x1-y1*y1; if(t1>=0.f){t1*=t1; n1=t1*t1*grad2(PERM[ii+i1+PERM[jj+j1]],x1,y1);}
    f32 t2=0.5f-x2*x2-y2*y2; if(t2>=0.f){t2*=t2; n2=t2*t2*grad2(PERM[ii+1+PERM[jj+1]],x2,y2);}
    return 70.f*(n0+n1+n2);
}

// ---------------------------------------------------------------------------
// Simplex Noise 3D
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 simplex3(f32 x, f32 y, f32 z) noexcept {
    using namespace detail;
    static constexpr f32 F3=1.f/3.f, G3=1.f/6.f;
    f32 s=(x+y+z)*F3;
    i32 i=(i32)floor(x+s), j=(i32)floor(y+s), k=(i32)floor(z+s);
    f32 t=(f32)(i+j+k)*G3;
    f32 x0=x-(i-t), y0=y-(j-t), z0=z-(k-t);
    i32 i1,j1,k1,i2,j2,k2;
    if(x0>=y0){if(y0>=z0){i1=1;j1=0;k1=0;i2=1;j2=1;k2=0;}
    else if(x0>=z0){i1=1;j1=0;k1=0;i2=1;j2=0;k2=1;}
    else{i1=0;j1=0;k1=1;i2=1;j2=0;k2=1;}}
    else{if(y0<z0){i1=0;j1=0;k1=1;i2=0;j2=1;k2=1;}
    else if(x0<z0){i1=0;j1=1;k1=0;i2=0;j2=1;k2=1;}
    else{i1=0;j1=1;k1=0;i2=1;j2=1;k2=0;}}
    f32 x1=x0-i1+G3, y1=y0-j1+G3, z1=z0-k1+G3;
    f32 x2=x0-i2+2.f*G3, y2=y0-j2+2.f*G3, z2=z0-k2+2.f*G3;
    f32 x3=x0-1.f+3.f*G3, y3=y0-1.f+3.f*G3, z3=z0-1.f+3.f*G3;
    i32 ii=i&255, jj=j&255, kk=k&255;
    f32 n0=0.f,n1=0.f,n2=0.f,n3=0.f;
    f32 t0=0.6f-x0*x0-y0*y0-z0*z0;
    if(t0>=0.f){t0*=t0;n0=t0*t0*grad3(PERM[ii+PERM[jj+PERM[kk]]],x0,y0,z0);}
    f32 t1=0.6f-x1*x1-y1*y1-z1*z1;
    if(t1>=0.f){t1*=t1;n1=t1*t1*grad3(PERM[ii+i1+PERM[jj+j1+PERM[kk+k1]]],x1,y1,z1);}
    f32 t2=0.6f-x2*x2-y2*y2-z2*z2;
    if(t2>=0.f){t2*=t2;n2=t2*t2*grad3(PERM[ii+i2+PERM[jj+j2+PERM[kk+k2]]],x2,y2,z2);}
    f32 t3=0.6f-x3*x3-y3*y3-z3*z3;
    if(t3>=0.f){t3*=t3;n3=t3*t3*grad3(PERM[ii+1+PERM[jj+1+PERM[kk+1]]],x3,y3,z3);}
    return 32.f*(n0+n1+n2+n3);
}

// ---------------------------------------------------------------------------
// Value Noise 2D / 3D
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 value2(f32 x, f32 y) noexcept {
    i32 ix=(i32)floor(x), iy=(i32)floor(y);
    f32 fx=smootherstep(0.f,1.f,x-floor(x)), fy=smootherstep(0.f,1.f,y-floor(y));
    auto h=[](i32 a,i32 b)->f32{ return detail::hash_to_f32(detail::hash21(u32(a),u32(b))); };
    return bilinear(h(ix,iy),h(ix+1,iy),h(ix,iy+1),h(ix+1,iy+1),fx,fy);
}

[[nodiscard]] inline f32 value3(f32 x, f32 y, f32 z) noexcept {
    i32 ix=(i32)floor(x), iy=(i32)floor(y), iz=(i32)floor(z);
    f32 fx=smootherstep(0.f,1.f,x-floor(x));
    f32 fy=smootherstep(0.f,1.f,y-floor(y));
    f32 fz=smootherstep(0.f,1.f,z-floor(z));
    auto h=[](i32 a,i32 b,i32 c)->f32{return detail::hash_to_f32(detail::hash31(u32(a),u32(b),u32(c)));};
    return trilinear(h(ix,iy,iz),h(ix+1,iy,iz),h(ix,iy+1,iz),h(ix+1,iy+1,iz),
                     h(ix,iy,iz+1),h(ix+1,iy,iz+1),h(ix,iy+1,iz+1),h(ix+1,iy+1,iz+1),fx,fy,fz);
}

// ---------------------------------------------------------------------------
// Cellular / Worley Noise 2D
// ---------------------------------------------------------------------------
struct CellularResult { f32 f1{0.f}, f2{0.f}; Vec2f cell_center; };
[[nodiscard]] inline CellularResult cellular2(f32 x, f32 y) noexcept {
    i32 ix=(i32)floor(x), iy=(i32)floor(y);
    f32 min_dist=1e38f, min_dist2=1e38f;
    Vec2f best_cell;
    for(i32 cy=-1;cy<=1;++cy) for(i32 cx=-1;cx<=1;++cx){
        u32 h=detail::hash21(u32(ix+cx),u32(iy+cy));
        f32 px=(ix+cx)+detail::hash_to_f32(h);
        f32 py=(iy+cy)+detail::hash_to_f32(detail::hash11(h));
        f32 dx=px-x, dy=py-y, d=dx*dx+dy*dy;
        if(d<min_dist){min_dist2=min_dist; min_dist=d; best_cell={px,py};}
        else if(d<min_dist2) min_dist2=d;
    }
    return {sqrt(min_dist),sqrt(min_dist2),best_cell};
}

// ---------------------------------------------------------------------------
// FBM (Fractional Brownian Motion)
// ---------------------------------------------------------------------------
struct FBMParams {
    i32  octaves{6};
    f32  lacunarity{2.f};
    f32  gain{0.5f};
    f32  initial_amplitude{0.5f};
};

[[nodiscard]] inline f32 fbm2(f32 x, f32 y, FBMParams p={}) noexcept {
    f32 value=0.f, amplitude=p.initial_amplitude, freq=1.f;
    for(i32 i=0;i<p.octaves;++i){
        value+=perlin2(x*freq,y*freq)*amplitude;
        amplitude*=p.gain; freq*=p.lacunarity;
    }
    return value;
}

[[nodiscard]] inline f32 fbm3(f32 x, f32 y, f32 z, FBMParams p={}) noexcept {
    f32 value=0.f, amplitude=p.initial_amplitude, freq=1.f;
    for(i32 i=0;i<p.octaves;++i){
        value+=perlin3(x*freq,y*freq,z*freq)*amplitude;
        amplitude*=p.gain; freq*=p.lacunarity;
    }
    return value;
}

// Ridged FBM
[[nodiscard]] inline f32 ridged_fbm2(f32 x, f32 y, FBMParams p={}) noexcept {
    f32 value=0.f, amplitude=p.initial_amplitude, freq=1.f;
    for(i32 i=0;i<p.octaves;++i){
        f32 n=1.f-abs(perlin2(x*freq,y*freq));
        value+=n*n*amplitude; amplitude*=p.gain; freq*=p.lacunarity;
    }
    return value;
}

// Turbulence
[[nodiscard]] inline f32 turbulence2(f32 x, f32 y, i32 octaves=6) noexcept {
    f32 value=0.f, freq=1.f, amp=1.f;
    for(i32 i=0;i<octaves;++i){
        value+=abs(perlin2(x*freq,y*freq))*amp;
        freq*=2.f; amp*=0.5f;
    }
    return value;
}

// ---------------------------------------------------------------------------
// Domain-warped noise
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 domain_warp2(f32 x, f32 y, f32 warp_strength=1.f, FBMParams p={}) noexcept {
    f32 wx=fbm2(x+0.1f, y+5.7f, p);
    f32 wy=fbm2(x+3.4f, y+9.2f, p);
    return fbm2(x+warp_strength*wx, y+warp_strength*wy, p);
}

} // namespace qs::math
