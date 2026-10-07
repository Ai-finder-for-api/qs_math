#pragma once
// vec_compare.hpp — Component-wise comparisons returning BVec,
//                   approximate equality, sorting predicates
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "bvec.hpp"
#include "scalar_math.hpp"

namespace qs::math {

// Component-wise comparisons → BVec
template<MathElement T> [[nodiscard]] constexpr BVec2 equal   (Vec2<T> a,Vec2<T> b) noexcept { return {a.x==b.x,a.y==b.y}; }
template<MathElement T> [[nodiscard]] constexpr BVec2 not_equal(Vec2<T> a,Vec2<T> b) noexcept { return {a.x!=b.x,a.y!=b.y}; }
template<MathElement T> [[nodiscard]] constexpr BVec2 less_than(Vec2<T> a,Vec2<T> b) noexcept { return {a.x<b.x,a.y<b.y}; }
template<MathElement T> [[nodiscard]] constexpr BVec2 greater_than(Vec2<T> a,Vec2<T> b) noexcept { return {a.x>b.x,a.y>b.y}; }
template<MathElement T> [[nodiscard]] constexpr BVec2 less_equal(Vec2<T> a,Vec2<T> b) noexcept { return {a.x<=b.x,a.y<=b.y}; }
template<MathElement T> [[nodiscard]] constexpr BVec2 greater_equal(Vec2<T> a,Vec2<T> b) noexcept { return {a.x>=b.x,a.y>=b.y}; }

template<MathElement T> [[nodiscard]] constexpr BVec3 equal    (Vec3<T> a,Vec3<T> b) noexcept { return {a.x==b.x,a.y==b.y,a.z==b.z}; }
template<MathElement T> [[nodiscard]] constexpr BVec3 not_equal(Vec3<T> a,Vec3<T> b) noexcept { return {a.x!=b.x,a.y!=b.y,a.z!=b.z}; }
template<MathElement T> [[nodiscard]] constexpr BVec3 less_than(Vec3<T> a,Vec3<T> b) noexcept { return {a.x<b.x,a.y<b.y,a.z<b.z}; }
template<MathElement T> [[nodiscard]] constexpr BVec3 greater_than(Vec3<T> a,Vec3<T> b) noexcept { return {a.x>b.x,a.y>b.y,a.z>b.z}; }

template<MathElement T> [[nodiscard]] constexpr BVec4 equal    (Vec4<T> a,Vec4<T> b) noexcept { return {a.x==b.x,a.y==b.y,a.z==b.z,a.w==b.w}; }
template<MathElement T> [[nodiscard]] constexpr BVec4 not_equal(Vec4<T> a,Vec4<T> b) noexcept { return {a.x!=b.x,a.y!=b.y,a.z!=b.z,a.w!=b.w}; }

// Approximate equality (epsilon-based)
template<FloatScalar T>
[[nodiscard]] constexpr BVec2 approx_equal(Vec2<T> a, Vec2<T> b, T eps=EPS_F32) noexcept {
    return {abs(a.x-b.x)<=eps, abs(a.y-b.y)<=eps};
}
template<FloatScalar T>
[[nodiscard]] constexpr BVec3 approx_equal(Vec3<T> a, Vec3<T> b, T eps=EPS_F32) noexcept {
    return {abs(a.x-b.x)<=eps, abs(a.y-b.y)<=eps, abs(a.z-b.z)<=eps};
}
template<FloatScalar T>
[[nodiscard]] constexpr BVec4 approx_equal(Vec4<T> a, Vec4<T> b, T eps=EPS_F32) noexcept {
    return {abs(a.x-b.x)<=eps, abs(a.y-b.y)<=eps, abs(a.z-b.z)<=eps, abs(a.w-b.w)<=eps};
}

// Select component-wise based on mask
template<MathElement T>
[[nodiscard]] constexpr Vec2<T> select(BVec2 mask, Vec2<T> a, Vec2<T> b) noexcept {
    return {mask.x?a.x:b.x, mask.y?a.y:b.y};
}
template<MathElement T>
[[nodiscard]] constexpr Vec3<T> select(BVec3 mask, Vec3<T> a, Vec3<T> b) noexcept {
    return {mask.x?a.x:b.x, mask.y?a.y:b.y, mask.z?a.z:b.z};
}
template<MathElement T>
[[nodiscard]] constexpr Vec4<T> select(BVec4 mask, Vec4<T> a, Vec4<T> b) noexcept {
    return {mask.x?a.x:b.x, mask.y?a.y:b.y, mask.z?a.z:b.z, mask.w?a.w:b.w};
}

// Lexicographic comparators (for sorting)
template<MathElement T>
[[nodiscard]] constexpr bool lex_less(Vec2<T> a, Vec2<T> b) noexcept {
    return a.x!=b.x ? a.x<b.x : a.y<b.y;
}
template<MathElement T>
[[nodiscard]] constexpr bool lex_less(Vec3<T> a, Vec3<T> b) noexcept {
    if(a.x!=b.x) return a.x<b.x;
    if(a.y!=b.y) return a.y<b.y;
    return a.z<b.z;
}

// Clamp vector to AABB
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> clamp_to_aabb(Vec3<T> v, Vec3<T> mn, Vec3<T> mx) noexcept {
    return v.clamp(mn,mx);
}

// Step and clamp operations for shaders
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> step(Vec3<T> edge, Vec3<T> v) noexcept {
    return {qs::math::step(edge.x,v.x), qs::math::step(edge.y,v.y), qs::math::step(edge.z,v.z)};
}
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> smoothstep(Vec3<T> lo, Vec3<T> hi, Vec3<T> v) noexcept {
    return {qs::math::smoothstep(lo.x,hi.x,v.x), qs::math::smoothstep(lo.y,hi.y,v.y), qs::math::smoothstep(lo.z,hi.z,v.z)};
}
template<FloatScalar T>
[[nodiscard]] constexpr Vec4<T> step(Vec4<T> edge, Vec4<T> v) noexcept {
    return {qs::math::step(edge.x,v.x),qs::math::step(edge.y,v.y),
            qs::math::step(edge.z,v.z),qs::math::step(edge.w,v.w)};
}

// Floor/ceil/round/frac for vectors
template<FloatScalar T>
[[nodiscard]] inline Vec2<T> floor(Vec2<T> v) noexcept { return {qs::math::floor(v.x),qs::math::floor(v.y)}; }
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> floor(Vec3<T> v) noexcept { return {qs::math::floor(v.x),qs::math::floor(v.y),qs::math::floor(v.z)}; }
template<FloatScalar T>
[[nodiscard]] inline Vec4<T> floor(Vec4<T> v) noexcept { return {qs::math::floor(v.x),qs::math::floor(v.y),qs::math::floor(v.z),qs::math::floor(v.w)}; }
template<FloatScalar T>
[[nodiscard]] inline Vec2<T> ceil(Vec2<T> v) noexcept { return {qs::math::ceil(v.x),qs::math::ceil(v.y)}; }
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> ceil(Vec3<T> v) noexcept { return {qs::math::ceil(v.x),qs::math::ceil(v.y),qs::math::ceil(v.z)}; }
template<FloatScalar T>
[[nodiscard]] inline Vec2<T> frac(Vec2<T> v) noexcept { return {qs::math::frac(v.x),qs::math::frac(v.y)}; }
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> frac(Vec3<T> v) noexcept { return {qs::math::frac(v.x),qs::math::frac(v.y),qs::math::frac(v.z)}; }
template<FloatScalar T>
[[nodiscard]] inline Vec4<T> frac(Vec4<T> v) noexcept { return {qs::math::frac(v.x),qs::math::frac(v.y),qs::math::frac(v.z),qs::math::frac(v.w)}; }

} // namespace qs::math

// ─── Component-wise min/max/clamp ─────────────────────────────────────────
template<MathElement T> [[nodiscard]] constexpr Vec2<T> min(Vec2<T> a,Vec2<T> b) noexcept { return {a.x<b.x?a.x:b.x,a.y<b.y?a.y:b.y}; }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> min(Vec3<T> a,Vec3<T> b) noexcept { return {a.x<b.x?a.x:b.x,a.y<b.y?a.y:b.y,a.z<b.z?a.z:b.z}; }
template<MathElement T> [[nodiscard]] constexpr Vec4<T> min(Vec4<T> a,Vec4<T> b) noexcept { return {a.x<b.x?a.x:b.x,a.y<b.y?a.y:b.y,a.z<b.z?a.z:b.z,a.w<b.w?a.w:b.w}; }
template<MathElement T> [[nodiscard]] constexpr Vec2<T> max(Vec2<T> a,Vec2<T> b) noexcept { return {a.x>b.x?a.x:b.x,a.y>b.y?a.y:b.y}; }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> max(Vec3<T> a,Vec3<T> b) noexcept { return {a.x>b.x?a.x:b.x,a.y>b.y?a.y:b.y,a.z>b.z?a.z:b.z}; }
template<MathElement T> [[nodiscard]] constexpr Vec4<T> max(Vec4<T> a,Vec4<T> b) noexcept { return {a.x>b.x?a.x:b.x,a.y>b.y?a.y:b.y,a.z>b.z?a.z:b.z,a.w>b.w?a.w:b.w}; }

template<MathElement T> [[nodiscard]] constexpr Vec2<T> clamp(Vec2<T> v,Vec2<T> lo,Vec2<T> hi) noexcept { return min(max(v,lo),hi); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> clamp(Vec3<T> v,Vec3<T> lo,Vec3<T> hi) noexcept { return min(max(v,lo),hi); }
template<MathElement T> [[nodiscard]] constexpr Vec4<T> clamp(Vec4<T> v,Vec4<T> lo,Vec4<T> hi) noexcept { return min(max(v,lo),hi); }
template<MathElement T> [[nodiscard]] constexpr Vec2<T> clamp(Vec2<T> v,T lo,T hi) noexcept { return clamp(v,Vec2<T>{lo},Vec2<T>{hi}); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> clamp(Vec3<T> v,T lo,T hi) noexcept { return clamp(v,Vec3<T>{lo},Vec3<T>{hi}); }
template<MathElement T> [[nodiscard]] constexpr Vec4<T> clamp(Vec4<T> v,T lo,T hi) noexcept { return clamp(v,Vec4<T>{lo},Vec4<T>{hi}); }

// ─── Horizontal operations ─────────────────────────────────────────────────
template<MathElement T> [[nodiscard]] constexpr T hmin(Vec2<T> v) noexcept { return v.x<v.y?v.x:v.y; }
template<MathElement T> [[nodiscard]] constexpr T hmin(Vec3<T> v) noexcept { return v.x<v.y?(v.x<v.z?v.x:v.z):(v.y<v.z?v.y:v.z); }
template<MathElement T> [[nodiscard]] constexpr T hmin(Vec4<T> v) noexcept { T a=v.x<v.y?v.x:v.y,b=v.z<v.w?v.z:v.w; return a<b?a:b; }
template<MathElement T> [[nodiscard]] constexpr T hmax(Vec2<T> v) noexcept { return v.x>v.y?v.x:v.y; }
template<MathElement T> [[nodiscard]] constexpr T hmax(Vec3<T> v) noexcept { return v.x>v.y?(v.x>v.z?v.x:v.z):(v.y>v.z?v.y:v.z); }
template<MathElement T> [[nodiscard]] constexpr T hmax(Vec4<T> v) noexcept { T a=v.x>v.y?v.x:v.y,b=v.z>v.w?v.z:v.w; return a>b?a:b; }
template<MathElement T> [[nodiscard]] constexpr T hsum(Vec2<T> v) noexcept { return v.x+v.y; }
template<MathElement T> [[nodiscard]] constexpr T hsum(Vec3<T> v) noexcept { return v.x+v.y+v.z; }
template<MathElement T> [[nodiscard]] constexpr T hsum(Vec4<T> v) noexcept { return v.x+v.y+v.z+v.w; }
template<MathElement T> [[nodiscard]] constexpr T hprod(Vec2<T> v) noexcept { return v.x*v.y; }
template<MathElement T> [[nodiscard]] constexpr T hprod(Vec3<T> v) noexcept { return v.x*v.y*v.z; }
template<MathElement T> [[nodiscard]] constexpr T hprod(Vec4<T> v) noexcept { return v.x*v.y*v.z*v.w; }

// ─── Component-wise abs/sign/floor/ceil ───────────────────────────────────
template<FloatScalar T> [[nodiscard]] inline Vec2<T> abs_cw(Vec2<T> v) noexcept { return {abs(v.x),abs(v.y)}; }
template<FloatScalar T> [[nodiscard]] inline Vec3<T> abs_cw(Vec3<T> v) noexcept { return {abs(v.x),abs(v.y),abs(v.z)}; }
template<FloatScalar T> [[nodiscard]] inline Vec4<T> abs_cw(Vec4<T> v) noexcept { return {abs(v.x),abs(v.y),abs(v.z),abs(v.w)}; }
template<FloatScalar T> [[nodiscard]] inline Vec2<T> floor_cw(Vec2<T> v) noexcept { return {std::floor(v.x),std::floor(v.y)}; }
template<FloatScalar T> [[nodiscard]] inline Vec3<T> floor_cw(Vec3<T> v) noexcept { return {std::floor(v.x),std::floor(v.y),std::floor(v.z)}; }
template<FloatScalar T> [[nodiscard]] inline Vec2<T> ceil_cw (Vec2<T> v) noexcept { return {std::ceil(v.x),std::ceil(v.y)}; }
template<FloatScalar T> [[nodiscard]] inline Vec3<T> ceil_cw (Vec3<T> v) noexcept { return {std::ceil(v.x),std::ceil(v.y),std::ceil(v.z)}; }
template<FloatScalar T> [[nodiscard]] inline Vec2<T> fract_cw(Vec2<T> v) noexcept { return {fract(v.x),fract(v.y)}; }
template<FloatScalar T> [[nodiscard]] inline Vec3<T> fract_cw(Vec3<T> v) noexcept { return {fract(v.x),fract(v.y),fract(v.z)}; }
