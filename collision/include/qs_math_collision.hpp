#pragma once
// qs_math/collision/include/qs_math_collision.hpp (C++20)
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <array>
namespace qs::math::collision {
using namespace qs::math;
// ─── Contact manifold ────────────────────────────────────────────────────────
struct Contact {
    Vec3<f32> point{};
    Vec3<f32> normal{};
    f32       depth{0};
    bool      valid{false};
};
// ─── GJK support function interface ──────────────────────────────────────────
using SupportFn = Vec3<f32>(*)(const void* shape, Vec3<f32> dir) noexcept;
// ─── AABB broad-phase ─────────────────────────────────────────────────────────
struct AABB3 { Vec3<f32> mn,mx;
    [[nodiscard]] bool overlaps(const AABB3& o) const noexcept {
        return mn.x<=o.mx.x&&mx.x>=o.mn.x&&mn.y<=o.mx.y&&mx.y>=o.mn.y&&mn.z<=o.mx.z&&mx.z>=o.mn.z;
    }
    [[nodiscard]] AABB3 merge(const AABB3& o) const noexcept {return {mn.min(o.mn),mx.max(o.mx)};}
    [[nodiscard]] f32 surface_area() const noexcept {
        Vec3<f32> d=mx-mn; return 2*(d.x*d.y+d.y*d.z+d.z*d.x);
    }
};
// ─── GJK — distance query between two convex shapes ──────────────────────────
struct GJKResult { f32 dist; Vec3<f32> witness_a, witness_b; bool intersecting; };
[[nodiscard]] GJKResult gjk(const void* shapeA, SupportFn supA,
                             const void* shapeB, SupportFn supB,
                             i32 max_iter=64) noexcept;
// ─── EPA — penetration depth after GJK detects intersection ──────────────────
[[nodiscard]] Contact epa(const void* shapeA, SupportFn supA,
                           const void* shapeB, SupportFn supB,
                           const Vec3<f32> simplex[4]) noexcept;
// ─── Sphere support ───────────────────────────────────────────────────────────
struct Sphere { Vec3<f32> center; f32 radius; };
[[nodiscard]] inline Vec3<f32> sphere_support(const void* s, Vec3<f32> d) noexcept {
    const auto& sp=*static_cast<const Sphere*>(s);
    const f32 len=d.length(); if(len<1e-7f) return sp.center;
    return sp.center + d*(sp.radius/len);
}
// ─── Box support ─────────────────────────────────────────────────────────────
struct Box { Vec3<f32> center; Vec3<f32> half_extents; };
[[nodiscard]] inline Vec3<f32> box_support(const void* b, Vec3<f32> d) noexcept {
    const auto& bx=*static_cast<const Box*>(b);
    return bx.center + Vec3<f32>{
        d.x>=0?bx.half_extents.x:-bx.half_extents.x,
        d.y>=0?bx.half_extents.y:-bx.half_extents.y,
        d.z>=0?bx.half_extents.z:-bx.half_extents.z};
}
} // namespace qs::math::collision
