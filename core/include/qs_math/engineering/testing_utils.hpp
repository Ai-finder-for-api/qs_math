#pragma once
// qs_math/testing_utils.hpp — Approx equality, comparison helpers, random test data
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include "transform.hpp"
#include "random.hpp"

namespace qs::math::test {

// ---------------------------------------------------------------------------
// Approximate equality (for unit tests)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr bool approx_eq(f32 a, f32 b, f32 eps=1e-5f) noexcept { return abs(a-b)<=eps; }
[[nodiscard]] constexpr bool approx_eq(f64 a, f64 b, f64 eps=1e-10)  noexcept { return abs(a-b)<=eps; }
[[nodiscard]] constexpr bool approx_eq(Vec2f a, Vec2f b, f32 eps=1e-5f) noexcept {
    return approx_eq(a.x,b.x,eps)&&approx_eq(a.y,b.y,eps);
}
[[nodiscard]] constexpr bool approx_eq(Vec3f a, Vec3f b, f32 eps=1e-5f) noexcept {
    return approx_eq(a.x,b.x,eps)&&approx_eq(a.y,b.y,eps)&&approx_eq(a.z,b.z,eps);
}
[[nodiscard]] constexpr bool approx_eq(Vec4f a, Vec4f b, f32 eps=1e-5f) noexcept {
    return approx_eq(a.x,b.x,eps)&&approx_eq(a.y,b.y,eps)&&approx_eq(a.z,b.z,eps)&&approx_eq(a.w,b.w,eps);
}
[[nodiscard]] constexpr bool approx_eq(Quatf a, Quatf b, f32 eps=1e-5f) noexcept {
    return (approx_eq(a.x,b.x,eps)&&approx_eq(a.y,b.y,eps)&&approx_eq(a.z,b.z,eps)&&approx_eq(a.w,b.w,eps))
        || (approx_eq(-a.x,b.x,eps)&&approx_eq(-a.y,b.y,eps)&&approx_eq(-a.z,b.z,eps)&&approx_eq(-a.w,b.w,eps));
}
[[nodiscard]] constexpr bool approx_eq(Mat4f a, Mat4f b, f32 eps=1e-5f) noexcept {
    for(i32 c=0;c<4;++c) for(i32 r=0;r<4;++r) if(!approx_eq(a.col[c][r],b.col[c][r],eps)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Random test data generators (use a fixed seed for reproducibility)
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f rand_vec3(PCG32& rng, f32 scale=1.f) noexcept {
    return {rng.range_f32(-scale,scale),rng.range_f32(-scale,scale),rng.range_f32(-scale,scale)};
}
[[nodiscard]] inline Vec3f rand_unit_vec3(PCG32& rng) noexcept { return rng.unit_sphere(); }
[[nodiscard]] inline Quatf rand_quat(PCG32& rng) noexcept {
    // Shoemake's uniform quaternion distribution
    f32 u1=rng.next_f32(), u2=rng.next_f32(), u3=rng.next_f32();
    f32 s1=sqrt(1.f-u1), s2=sqrt(u1);
    return Quatf{s1*sin(TWO_PI*u2), s1*cos(TWO_PI*u2), s2*sin(TWO_PI*u3), s2*cos(TWO_PI*u3)};
}
[[nodiscard]] inline Mat4f rand_rotation_mat4(PCG32& rng) noexcept {
    return rand_quat(rng).to_mat4();
}
[[nodiscard]] inline Transformf rand_transform(PCG32& rng) noexcept {
    Vec3f t{rng.range_f32(-100.f,100.f),rng.range_f32(-100.f,100.f),rng.range_f32(-100.f,100.f)};
    Quatf r=rand_quat(rng);
    Vec3f s{rng.range_f32(0.1f,10.f),rng.range_f32(0.1f,10.f),rng.range_f32(0.1f,10.f)};
    return {t,r,s};
}

// Check if a matrix is orthogonal
[[nodiscard]] constexpr bool is_orthogonal(Mat3f m, f32 eps=1e-4f) noexcept {
    Mat3f mt=m.transposed();
    Mat3f id=mt*m;
    return approx_eq(Mat4f{id}, Mat4f::identity(), eps);
}

// Numerical gradient check for functions R->R
template<typename Fn>
[[nodiscard]] inline bool gradient_check(Fn&& f, f64 x, f64 expected_grad, f64 tol=1e-5) noexcept {
    f64 h=1e-5;
    f64 approx=(f(x+h)-f(x-h))/(2.0*h);
    return abs(approx-expected_grad)<=tol;
}

} // namespace qs::math::test
