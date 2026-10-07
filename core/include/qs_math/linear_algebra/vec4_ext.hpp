#pragma once
// vec4_ext.hpp — Extended Vec4 ops: homogeneous, RGBA ops, plane extraction,
//               quaternion-as-vec4 helpers, AoS↔SoA pack
#include "vec4.hpp"

namespace qs::math {

// Homogeneous divide
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> homogeneous_divide(Vec4<T> v) noexcept {
    if(abs(v.w)<T(EPS_LENGTH)) return {v.x,v.y,v.z};
    T inv=T(1)/v.w;
    return {v.x*inv,v.y*inv,v.z*inv};
}

// 3D point → homogeneous
template<FloatScalar T>
[[nodiscard]] constexpr Vec4<T> to_homogeneous_point(Vec3<T> v) noexcept { return {v.x,v.y,v.z,T(1)}; }
template<FloatScalar T>
[[nodiscard]] constexpr Vec4<T> to_homogeneous_dir(Vec3<T> v) noexcept { return {v.x,v.y,v.z,T(0)}; }

// RGBA channel access (semantic aliases)
template<FloatScalar T>
struct RGBA {
    Vec4<T> v;
    [[nodiscard]] constexpr T r() const noexcept { return v.x; }
    [[nodiscard]] constexpr T g() const noexcept { return v.y; }
    [[nodiscard]] constexpr T b() const noexcept { return v.z; }
    [[nodiscard]] constexpr T a() const noexcept { return v.w; }
    [[nodiscard]] constexpr Vec3<T> rgb() const noexcept { return {v.x,v.y,v.z}; }
    [[nodiscard]] constexpr T luminance() const noexcept {
        return T(0.2126)*v.x + T(0.7152)*v.y + T(0.0722)*v.z;
    }
    [[nodiscard]] constexpr RGBA premultiplied() const noexcept {
        return {Vec4<T>{v.x*v.w, v.y*v.w, v.z*v.w, v.w}};
    }
};

// Plane extraction from Vec4 (nx,ny,nz,d) — n·x + d = 0
template<FloatScalar T>
[[nodiscard]] constexpr T plane_distance(Vec4<T> plane, Vec3<T> point) noexcept {
    return plane.x*point.x + plane.y*point.y + plane.z*point.z + plane.w;
}
template<FloatScalar T>
[[nodiscard]] constexpr Vec4<T> normalize_plane(Vec4<T> plane) noexcept {
    T len=Vec3<T>{plane.x,plane.y,plane.z}.length();
    if(len<T(EPS_LENGTH)) return plane;
    T inv=T(1)/len;
    return {plane.x*inv,plane.y*inv,plane.z*inv,plane.w*inv};
}

// AoS (array of Vec4) to SoA (4 float arrays)
template<FloatScalar T>
inline void aos_to_soa(const Vec4<T>* aos, i32 n, T* xs, T* ys, T* zs, T* ws) noexcept {
    for(i32 i=0;i<n;++i){ xs[i]=aos[i].x; ys[i]=aos[i].y; zs[i]=aos[i].z; ws[i]=aos[i].w; }
}
template<FloatScalar T>
inline void soa_to_aos(const T* xs, const T* ys, const T* zs, const T* ws, Vec4<T>* aos, i32 n) noexcept {
    for(i32 i=0;i<n;++i){ aos[i]={xs[i],ys[i],zs[i],ws[i]}; }
}

// Half-precision packing (lossy)
[[nodiscard]] inline u16 float_to_half(f32 v) noexcept {
    u32 x; __builtin_memcpy(&x,&v,4);
    u32 e=((x>>23)&0xFF), m=x&0x7FFFFF;
    u16 s=u16(x>>31);
    if(e==255) return u16((s<<15)|(31<<10)|(m?1:0));
    if(e<103)  return u16(s<<15);
    if(e<113){ u32 frac=(0x800000|m)>>(125-e); return u16((s<<15)|(frac&0x3FF)); }
    return u16((s<<15)|(((e-112)&0x1F)<<10)|((m>>13)&0x3FF));
}
[[nodiscard]] inline f32 half_to_float(u16 h) noexcept {
    u32 e=((h>>10)&0x1F), m=(h&0x3FF);
    u32 s=(h>>15)&1;
    if(e==31) { u32 x=(s<<31)|(0xFF<<23)|(m<<13); f32 r; __builtin_memcpy(&r,&x,4); return r; }
    if(e==0)  { u32 x=(s<<31)|(m<<13); f32 r; __builtin_memcpy(&r,&x,4); return r; }
    u32 x=(s<<31)|((e+112)<<23)|(m<<13); f32 r; __builtin_memcpy(&r,&x,4); return r;
}

// Pack Vec4 as 4 halves (8 bytes)
inline void pack_vec4_half(Vec4f v, u16* out) noexcept {
    out[0]=float_to_half(v.x); out[1]=float_to_half(v.y);
    out[2]=float_to_half(v.z); out[3]=float_to_half(v.w);
}
inline Vec4f unpack_vec4_half(const u16* in) noexcept {
    return {half_to_float(in[0]),half_to_float(in[1]),half_to_float(in[2]),half_to_float(in[3])};
}

// Pack Vec3 normal into 2 bytes (octahedral encoding)
[[nodiscard]] inline Vec2f oct_encode_normal(Vec3f n) noexcept {
    n=n.normalized();
    f32 d=abs(n.x)+abs(n.y)+abs(n.z);
    Vec2f p{n.x/d, n.y/d};
    if(n.z<=0.f) p={((p.x>=0.f?1.f:-1.f)*(1.f-abs(p.y))),
                    ((p.y>=0.f?1.f:-1.f)*(1.f-abs(p.x)))};
    return p;
}
[[nodiscard]] inline Vec3f oct_decode_normal(Vec2f p) noexcept {
    Vec3f n{p.x,p.y,1.f-abs(p.x)-abs(p.y)};
    if(n.z<0.f){ n.x=(1.f-abs(n.y))*(n.x>=0.f?1.f:-1.f); n.y=(1.f-abs(n.x))*(n.y>=0.f?1.f:-1.f); }
    return n.normalized();
}

} // namespace qs::math
