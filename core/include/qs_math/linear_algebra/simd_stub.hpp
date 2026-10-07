#pragma once
// qs_math/simd_stub.hpp — SIMD abstraction stubs (platform implementations in src/simd/)
// These forward-declare the SIMD types and mark them for future platform-specific backends.
// Currently falls back to scalar Vec2/Vec3/Vec4/Mat4/Quat.
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"

namespace qs::math::simd {

// SIMD widths supported
enum class SIMDWidth { Scalar=1, SSE2=4, AVX=8, AVX512=16, NEON=4 };

// Detect at compile time
#if defined(__AVX512F__)
    inline constexpr SIMDWidth NATIVE_WIDTH = SIMDWidth::AVX512;
#elif defined(__AVX__)
    inline constexpr SIMDWidth NATIVE_WIDTH = SIMDWidth::AVX;
#elif defined(__SSE2__) || defined(_M_X64)
    inline constexpr SIMDWidth NATIVE_WIDTH = SIMDWidth::SSE2;
#elif defined(__ARM_NEON)
    inline constexpr SIMDWidth NATIVE_WIDTH = SIMDWidth::NEON;
#else
    inline constexpr SIMDWidth NATIVE_WIDTH = SIMDWidth::Scalar;
#endif

// Batch of 4 Vec3 — SOA layout for SIMD processing
struct Vec3x4 {
    f32 x[4], y[4], z[4];
    constexpr void set(i32 i, Vec3f v) noexcept { x[i]=v.x; y[i]=v.y; z[i]=v.z; }
    constexpr Vec3f get(i32 i) const noexcept { return {x[i],y[i],z[i]}; }
};
struct Vec3x8 {
    f32 x[8], y[8], z[8];
    constexpr void set(i32 i, Vec3f v) noexcept { x[i]=v.x; y[i]=v.y; z[i]=v.z; }
    constexpr Vec3f get(i32 i) const noexcept { return {x[i],y[i],z[i]}; }
};

// Batch dot products (scalar fallback)
inline void dot4(const Vec3x4& a, const Vec3x4& b, f32* out) noexcept {
    for(i32 i=0;i<4;++i) out[i]=a.x[i]*b.x[i]+a.y[i]*b.y[i]+a.z[i]*b.z[i];
}
// Batch normalize (scalar fallback)
inline void normalize4(Vec3x4& v) noexcept {
    for(i32 i=0;i<4;++i){
        f32 len=sqrt(v.x[i]*v.x[i]+v.y[i]*v.y[i]+v.z[i]*v.z[i]);
        if(len>EPS_LENGTH){ v.x[i]/=len; v.y[i]/=len; v.z[i]/=len; }
    }
}
// Batch transform (4 points by one Mat4)
inline void transform_points4(Mat4f m, Vec3x4& pts) noexcept {
    for(i32 i=0;i<4;++i){
        Vec3f p={pts.x[i],pts.y[i],pts.z[i]};
        p=m.transform_point(p);
        pts.x[i]=p.x; pts.y[i]=p.y; pts.z[i]=p.z;
    }
}

} // namespace qs::math::simd

// ─── Scalar fallback for SIMD-style lane operations ───────────────────────
// These mirror the SIMD interface for non-SIMD builds

struct Float4 {
    f32 v[4];
    constexpr Float4() noexcept : v{0,0,0,0} {}
    constexpr Float4(f32 a,f32 b,f32 c,f32 d) noexcept : v{a,b,c,d} {}
    constexpr explicit Float4(f32 s) noexcept : v{s,s,s,s} {}
    constexpr Float4 operator+(Float4 o) const noexcept { return {v[0]+o.v[0],v[1]+o.v[1],v[2]+o.v[2],v[3]+o.v[3]}; }
    constexpr Float4 operator-(Float4 o) const noexcept { return {v[0]-o.v[0],v[1]-o.v[1],v[2]-o.v[2],v[3]-o.v[3]}; }
    constexpr Float4 operator*(Float4 o) const noexcept { return {v[0]*o.v[0],v[1]*o.v[1],v[2]*o.v[2],v[3]*o.v[3]}; }
    constexpr Float4 operator*(f32 s)   const noexcept { return {v[0]*s,v[1]*s,v[2]*s,v[3]*s}; }
    [[nodiscard]] constexpr f32 hsum()  const noexcept { return v[0]+v[1]+v[2]+v[3]; }
    [[nodiscard]] constexpr f32 dot(Float4 o) const noexcept { return (*this*o).hsum(); }
};

struct Int4 {
    i32 v[4];
    constexpr Int4() noexcept : v{0,0,0,0} {}
    constexpr Int4(i32 a,i32 b,i32 c,i32 d) noexcept : v{a,b,c,d} {}
    constexpr explicit Int4(i32 s) noexcept : v{s,s,s,s} {}
    constexpr Int4 operator+(Int4 o) const noexcept { return {v[0]+o.v[0],v[1]+o.v[1],v[2]+o.v[2],v[3]+o.v[3]}; }
    constexpr Int4 operator&(Int4 o) const noexcept { return {v[0]&o.v[0],v[1]&o.v[1],v[2]&o.v[2],v[3]&o.v[3]}; }
    constexpr Int4 operator|(Int4 o) const noexcept { return {v[0]|o.v[0],v[1]|o.v[1],v[2]|o.v[2],v[3]|o.v[3]}; }
    [[nodiscard]] constexpr int mask() const noexcept {
        return int(v[0]<0)|(int(v[1]<0)<<1)|(int(v[2]<0)<<2)|(int(v[3]<0)<<3);
    }
};

[[nodiscard]] inline Float4 float4_load (const f32* p)  noexcept { return {p[0],p[1],p[2],p[3]}; }
inline void                 float4_store(f32* p, Float4 a) noexcept { p[0]=a.v[0];p[1]=a.v[1];p[2]=a.v[2];p[3]=a.v[3]; }
[[nodiscard]] inline Float4 float4_madd (Float4 a,Float4 b,Float4 c) noexcept {
    return {std::fma(a.v[0],b.v[0],c.v[0]),std::fma(a.v[1],b.v[1],c.v[1]),
            std::fma(a.v[2],b.v[2],c.v[2]),std::fma(a.v[3],b.v[3],c.v[3])};
}
[[nodiscard]] inline Float4 float4_min  (Float4 a,Float4 b) noexcept {
    return {a.v[0]<b.v[0]?a.v[0]:b.v[0],a.v[1]<b.v[1]?a.v[1]:b.v[1],
            a.v[2]<b.v[2]?a.v[2]:b.v[2],a.v[3]<b.v[3]?a.v[3]:b.v[3]};
}
[[nodiscard]] inline Float4 float4_max  (Float4 a,Float4 b) noexcept {
    return {a.v[0]>b.v[0]?a.v[0]:b.v[0],a.v[1]>b.v[1]?a.v[1]:b.v[1],
            a.v[2]>b.v[2]?a.v[2]:b.v[2],a.v[3]>b.v[3]?a.v[3]:b.v[3]};
}
