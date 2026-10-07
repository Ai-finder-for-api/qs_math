#pragma once
// qs_math/vec_ops.hpp — Additional vector operations: orthonormal basis, triple products,
//                        slerp for direction vectors, midpoint, barycentric, face normal
#include "vec3.hpp"
#include "vec4.hpp"

namespace qs::math {

// Gram-Schmidt orthogonalization
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> gram_schmidt(Vec3<T> v, Vec3<T> ref) noexcept {
    return (v - ref * dot(v, ref)).normalized();
}

// Triple product: a . (b x c)
template<FloatScalar T>
[[nodiscard]] constexpr T triple_product(Vec3<T> a, Vec3<T> b, Vec3<T> c) noexcept {
    return dot(a, cross(b, c));
}

// Slerp for direction vectors (not quaternions)
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> slerp_dir(Vec3<T> a, Vec3<T> b, T t) noexcept {
    T d = clamp(dot(a,b), T(-1), T(1));
    T theta = safe_acos(d) * t;
    Vec3<T> rel = (b - a*d).normalized();
    T c, s; sincos(theta, s, c);
    return a*c + rel*s;
}

// Midpoint
template<MathElement T>
[[nodiscard]] constexpr Vec2<T> midpoint(Vec2<T> a, Vec2<T> b) noexcept { return (a+b)*T(0.5); }
template<MathElement T>
[[nodiscard]] constexpr Vec3<T> midpoint(Vec3<T> a, Vec3<T> b) noexcept { return (a+b)*T(0.5); }

// Barycentric coordinates: given triangle (a,b,c) and point p, return (u,v,w)
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> barycentric(Vec3<T> a, Vec3<T> b, Vec3<T> c, Vec3<T> p) noexcept {
    Vec3<T> v0=b-a, v1=c-a, v2=p-a;
    T d00=dot(v0,v0), d01=dot(v0,v1), d11=dot(v1,v1);
    T d20=dot(v2,v0), d21=dot(v2,v1);
    T inv=T(1)/(d00*d11-d01*d01);
    T v=(d11*d20-d01*d21)*inv;
    T w=(d00*d21-d01*d20)*inv;
    return {T(1)-v-w, v, w};
}

// Face normal from triangle winding
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> face_normal(Vec3<T> a, Vec3<T> b, Vec3<T> c) noexcept {
    return cross(b-a, c-a).normalized();
}

// Angle between two vectors (0..PI)
template<FloatScalar T>
[[nodiscard]] inline T angle_between(Vec3<T> a, Vec3<T> b) noexcept {
    return safe_acos(clamp(dot(a.normalized(), b.normalized()), T(-1), T(1)));
}
template<FloatScalar T>
[[nodiscard]] inline T angle_between(Vec2<T> a, Vec2<T> b) noexcept {
    return safe_acos(clamp(dot(a.normalized(), b.normalized()), T(-1), T(1)));
}

// Signed angle in 2D (from a to b, CCW positive)
template<FloatScalar T>
[[nodiscard]] inline T signed_angle(Vec2<T> a, Vec2<T> b) noexcept {
    return atan2(cross(a,b), dot(a,b));
}

// Signed angle in 3D around axis
template<FloatScalar T>
[[nodiscard]] inline T signed_angle(Vec3<T> a, Vec3<T> b, Vec3<T> axis) noexcept {
    T unsigned_angle = angle_between(a, b);
    T sign_val = sign(dot(cross(a,b), axis));
    return unsigned_angle * sign_val;
}

// Move towards target with max step
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> move_towards(Vec3<T> from, Vec3<T> to, T max_delta) noexcept {
    Vec3<T> diff = to - from;
    T dist = diff.length();
    if (dist <= max_delta || dist < T(EPS_LENGTH)) return to;
    return from + diff * (max_delta / dist);
}
template<FloatScalar T>
[[nodiscard]] inline Vec2<T> move_towards(Vec2<T> from, Vec2<T> to, T max_delta) noexcept {
    Vec2<T> diff = to - from;
    T dist = diff.length();
    if (dist <= max_delta || dist < T(EPS_LENGTH)) return to;
    return from + diff * (max_delta / dist);
}

} // namespace qs::math

// ─── Projection helpers ────────────────────────────────────────────────────
// Project v onto plane with given normal (n must be unit)
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> project_onto_plane(Vec3<T> v, Vec3<T> n) noexcept {
    return v - n * dot(v, n);
}
// Project v onto line direction d (d must be unit)
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> project_onto_line(Vec3<T> v, Vec3<T> d) noexcept {
    return d * dot(v, d);
}

// ─── Reflect / refract ─────────────────────────────────────────────────────
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> reflect(Vec3<T> i, Vec3<T> n) noexcept {
    return i - n * (T(2) * dot(i, n));
}
// Snell's law refraction (i=incident unit, n=surface normal unit, eta=n1/n2)
// Returns zero vector on total internal reflection
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> refract(Vec3<T> i, Vec3<T> n, T eta) noexcept {
    T cos_i   = -dot(n, i);
    T sin2_t  = eta*eta*(T(1)-cos_i*cos_i);
    if (sin2_t > T(1)) return Vec3<T>{T(0),T(0),T(0)}; // TIR
    T cos_t   = std::sqrt(T(1)-sin2_t);
    return i*eta + n*(eta*cos_i - cos_t);
}

// ─── Winding / handedness ──────────────────────────────────────────────────
// CCW winding of 2D triangle (positive = CCW)
template<FloatScalar T>
[[nodiscard]] constexpr T triangle_winding2d(Vec2<T> a, Vec2<T> b, Vec2<T> c) noexcept {
    return (b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y);
}
template<FloatScalar T>
[[nodiscard]] constexpr bool is_ccw(Vec2<T> a, Vec2<T> b, Vec2<T> c) noexcept {
    return triangle_winding2d(a,b,c) > T(0);
}

// ─── Vector packing/encoding ───────────────────────────────────────────────
// Octahedron normal encoding: unit vec3 -> vec2 [-1,1]^2
[[nodiscard]] inline Vec2f oct_encode(Vec3f n) noexcept {
    f32 l1 = abs(n.x)+abs(n.y)+abs(n.z);
    Vec2f p = {n.x/l1, n.y/l1};
    if(n.z < 0.f) p = {(1.f-abs(p.y))*(p.x>=0.f?1.f:-1.f),
                       (1.f-abs(p.x))*(p.y>=0.f?1.f:-1.f)};
    return p;
}
[[nodiscard]] inline Vec3f oct_decode(Vec2f p) noexcept {
    Vec3f n = {p.x, p.y, 1.f-abs(p.x)-abs(p.y)};
    if(n.z < 0.f){ n.x=(1.f-abs(n.y))*(n.x>=0.f?1.f:-1.f);
                   n.y=(1.f-abs(n.x))*(n.y>=0.f?1.f:-1.f); }
    return n.normalized();
}

// ─── Centroid of n points ──────────────────────────────────────────────────
[[nodiscard]] inline Vec3f centroid(const Vec3f* pts, i32 n) noexcept {
    Vec3f s{0,0,0};
    for(i32 i=0;i<n;++i) s+=pts[i];
    return s*(1.f/f32(n));
}
[[nodiscard]] inline Vec2f centroid2d(const Vec2f* pts, i32 n) noexcept {
    Vec2f s{0,0};
    for(i32 i=0;i<n;++i) s+=pts[i];
    return s*(1.f/f32(n));
}
