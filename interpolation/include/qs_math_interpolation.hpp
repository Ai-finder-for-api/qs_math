
#pragma once
// qs_math/interpolation/include/qs_math_interpolation.hpp (C++20)
// Extended interpolation: SLERP, squad, multivariate, RBF, barycentric.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/vec4.hpp>
#include <qs_math/quaternion/quat.hpp>
#include <cmath>
#include <vector>
#include <functional>
namespace qs::math::interp {
using namespace qs::math;
// ─── Scalar lerp family ───────────────────────────────────────────────────────
template<typename T> [[nodiscard]] T lerp(T a, T b, f32 t) noexcept { return a+(b-a)*t; }
template<typename T> [[nodiscard]] T smoothstep(T a, T b, f32 t) noexcept {
    t=(t-a)/(b-a); t=t<0?0:(t>1?1:t); return t*t*(3-2*t);
}
template<typename T> [[nodiscard]] T smootherstep(T a, T b, f32 t) noexcept {
    t=(t-a)/(b-a); t=t<0?0:(t>1?1:t); return t*t*t*(t*(t*6-15)+10);
}
// ─── Quaternion SLERP / NLERP / SQUAD ────────────────────────────────────────
[[nodiscard]] Quat<f32> slerp(Quat<f32> a, Quat<f32> b, f32 t) noexcept;
[[nodiscard]] Quat<f32> nlerp(Quat<f32> a, Quat<f32> b, f32 t) noexcept;
[[nodiscard]] Quat<f32> squad(Quat<f32> q0, Quat<f32> q1,
    Quat<f32> s0, Quat<f32> s1, f32 t) noexcept;
[[nodiscard]] Quat<f32> squad_tangent(Quat<f32> prev,
    Quat<f32> curr, Quat<f32> next) noexcept;
// ─── Radial basis function interpolation ─────────────────────────────────────
struct RBFInterp {
    enum class Kernel { Gaussian, Multiquadric, InvMultiquadric, ThinPlate };
    std::vector<Vec2<f32>> centers;
    std::vector<f64> weights;
    f64 epsilon{1.0};
    Kernel kernel{Kernel::Gaussian};
    void fit(const Vec2<f32>* pts, const f64* values, i32 n) noexcept;
    [[nodiscard]] f64 eval(Vec2<f32> p) const noexcept;
private:
    [[nodiscard]] f64 phi(f64 r) const noexcept;
};
// ─── Multivariate linear interpolation (ND tetrahedra) ───────────────────────
[[nodiscard]] f64 trilinear_field(const f64 corners[8], f64 u, f64 v, f64 w) noexcept;
// ─── Bezier surface (bicubic, 4x4 control points) ────────────────────────────
[[nodiscard]] Vec3<f32> bezier_surface(
    const Vec3<f32> ctrl[4][4], f32 u, f32 v) noexcept;
[[nodiscard]] Vec3<f32> bezier_surface_normal(
    const Vec3<f32> ctrl[4][4], f32 u, f32 v) noexcept;
// ─── B-spline surface ─────────────────────────────────────────────────────────
[[nodiscard]] Vec3<f32> bspline_surface(const Vec3<f32>* ctrl,
    i32 nu, i32 nv, i32 pu, i32 pv,
    const f64* ku, const f64* kv, f64 u, f64 v) noexcept;
} // namespace qs::math::interp
