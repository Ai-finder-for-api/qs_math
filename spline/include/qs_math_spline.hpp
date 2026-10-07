
#pragma once
// qs_math/spline/include/qs_math_spline.hpp (C++20)
// Spline module: NURBS surfaces, tensor-product splines, spline fitting.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <functional>
namespace qs::math::spline {
using namespace qs::math;
// ─── Cubic spline segment ────────────────────────────────────────────────────
struct Cubic { f64 a,b,c,d; // y = a + b*t + c*t^2 + d*t^3
    [[nodiscard]] f64 eval(f64 t) const noexcept { return a+t*(b+t*(c+t*d)); }
    [[nodiscard]] f64 deriv(f64 t) const noexcept { return b+t*(2*c+3*t*d); }
    [[nodiscard]] f64 deriv2(f64 t) const noexcept { return 2*c+6*t*d; }
};
// ─── Natural cubic spline ────────────────────────────────────────────────────
struct NaturalSpline {
    std::vector<f64> xs, ys;
    std::vector<Cubic> segs;
    void fit(const f64* x, const f64* y, i32 n) noexcept;
    [[nodiscard]] f64 eval(f64 x) const noexcept;
    [[nodiscard]] f64 deriv(f64 x) const noexcept;
    [[nodiscard]] f64 integrate(f64 a, f64 b) const noexcept;
};
// ─── Clamped cubic spline ─────────────────────────────────────────────────────
struct ClampedSpline : NaturalSpline {
    void fit(const f64* x, const f64* y, i32 n, f64 d0, f64 dn) noexcept;
};
// ─── Periodic spline ──────────────────────────────────────────────────────────
struct PeriodicSpline : NaturalSpline {
    void fit(const f64* x, const f64* y, i32 n) noexcept;
};
// ─── Parametric 3D spline ────────────────────────────────────────────────────
struct Spline3D {
    NaturalSpline sx, sy, sz;
    std::vector<f64> ts;
    void fit(const Vec3<f64>* pts, i32 n) noexcept; // chord-length param
    [[nodiscard]] Vec3<f64> eval(f64 t) const noexcept;
    [[nodiscard]] Vec3<f64> tangent(f64 t) const noexcept;
    [[nodiscard]] Vec3<f64> normal(f64 t) const noexcept;
    [[nodiscard]] f64 arc_length(f64 t0, f64 t1, i32 steps=100) const noexcept;
};
// ─── NURBS curve ──────────────────────────────────────────────────────────────
struct NURBSCurve {
    std::vector<Vec4<f64>> ctrl; // {x,y,z,w}
    std::vector<f64> knots;
    i32 degree{3};
    void set(std::vector<Vec4<f64>> c, std::vector<f64> k, i32 d) noexcept { ctrl=c;knots=k;degree=d; }
    [[nodiscard]] Vec3<f64> eval(f64 t) const noexcept;
    [[nodiscard]] Vec3<f64> tangent(f64 t, f64 h=1e-5) const noexcept;
    [[nodiscard]] f64 arc_length(i32 steps=200) const noexcept;
    [[nodiscard]] f64 t_at_arc_length(f64 s, i32 steps=200) const noexcept;
};
// ─── 2D spline fitting to scattered data ─────────────────────────────────────
[[nodiscard]] NaturalSpline fit_spline_to_fn(
    std::function<f64(f64)> f, f64 a, f64 b, i32 n) noexcept;
// ─── Bezier-to-spline conversion ─────────────────────────────────────────────
void bezier_to_bspline(const Vec3<f64>* bezier_ctrl, i32 n_segs,
    std::vector<Vec3<f64>>& bspline_ctrl,
    std::vector<f64>& knots) noexcept;
} // namespace qs::math::spline
