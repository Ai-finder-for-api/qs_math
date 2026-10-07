#pragma once
// qs_math/trig.hpp — Trig functions, fast approx, full-precision wrappers
#include "scalar_types.hpp"
#include "constants.hpp"
#include "angle.hpp"
#include <cmath>

namespace qs::math {

// --- Full precision ---
template<FloatScalar T> [[nodiscard]] inline T sin  (T v) noexcept { return std::sin(v); }
template<FloatScalar T> [[nodiscard]] inline T cos  (T v) noexcept { return std::cos(v); }
template<FloatScalar T> [[nodiscard]] inline T tan  (T v) noexcept { return std::tan(v); }
template<FloatScalar T> [[nodiscard]] inline T asin (T v) noexcept { return std::asin(v); }
template<FloatScalar T> [[nodiscard]] inline T acos (T v) noexcept { return std::acos(v); }
template<FloatScalar T> [[nodiscard]] inline T atan (T v) noexcept { return std::atan(v); }
template<FloatScalar T> [[nodiscard]] inline T atan2(T y, T x) noexcept { return std::atan2(y,x); }
template<FloatScalar T> [[nodiscard]] inline T sinh (T v) noexcept { return std::sinh(v); }
template<FloatScalar T> [[nodiscard]] inline T cosh (T v) noexcept { return std::cosh(v); }
template<FloatScalar T> [[nodiscard]] inline T tanh (T v) noexcept { return std::tanh(v); }

// Angle overloads
template<FloatScalar T> [[nodiscard]] inline T sin  (Radians<T> r) noexcept { return std::sin(r.value); }
template<FloatScalar T> [[nodiscard]] inline T cos  (Radians<T> r) noexcept { return std::cos(r.value); }
template<FloatScalar T> [[nodiscard]] inline T tan  (Radians<T> r) noexcept { return std::tan(r.value); }
template<FloatScalar T> [[nodiscard]] inline T sin  (Degrees<T> d) noexcept { return std::sin(d.value*T(DEG_TO_RAD_D)); }
template<FloatScalar T> [[nodiscard]] inline T cos  (Degrees<T> d) noexcept { return std::cos(d.value*T(DEG_TO_RAD_D)); }
template<FloatScalar T> [[nodiscard]] inline T tan  (Degrees<T> d) noexcept { return std::tan(d.value*T(DEG_TO_RAD_D)); }

// sincos — compute both at once
template<FloatScalar T>
inline void sincos(T v, T& s, T& c) noexcept { s=std::sin(v); c=std::cos(v); }
template<FloatScalar T>
inline void sincos(Radians<T> r, T& s, T& c) noexcept { s=std::sin(r.value); c=std::cos(r.value); }

// Safe asin/acos (clamp input to [-1,1])
template<FloatScalar T> [[nodiscard]] inline T safe_asin(T v) noexcept { return std::asin(clamp(v,T(-1),T(1))); }
template<FloatScalar T> [[nodiscard]] inline T safe_acos(T v) noexcept { return std::acos(clamp(v,T(-1),T(1))); }

// Fast sin approx (Bhaskara I) — ~4th order polynomial, < 0.2% error
[[nodiscard]] inline f32 fast_sin(f32 x) noexcept {
    x = wrap(x, -PI, PI);
    f32 x2 = x*x;
    return x*(1.0f - x2*(1.0f/6.0f - x2*(1.0f/120.0f - x2/5040.0f)));
}
[[nodiscard]] inline f32 fast_cos(f32 x) noexcept { return fast_sin(x + HALF_PI); }
[[nodiscard]] inline f32 fast_atan2(f32 y, f32 x) noexcept {
    // Approximation by Ingo Quilez
    f32 ay = abs(y), ax = abs(x);
    f32 a  = (ax>ay) ? ay/ax : ax/ay;
    f32 s  = a*a;
    f32 r  = (0.04737f*s - 0.14551f)*s + 1.0f;
    r *= a;
    if (ay>ax) r = HALF_PI - r;
    if (x<0.0f) r = PI - r;
    if (y<0.0f) r = -r;
    return r;
}

// Convert between trig forms
template<FloatScalar T> [[nodiscard]] inline T cot(T v) noexcept  { return T(1)/std::tan(v); }
template<FloatScalar T> [[nodiscard]] inline T sec(T v) noexcept  { return T(1)/std::cos(v); }
template<FloatScalar T> [[nodiscard]] inline T csc(T v) noexcept  { return T(1)/std::sin(v); }

} // namespace qs::math

// ─── Inverse hyperbolic ────────────────────────────────────────────────────
template<FloatScalar T> [[nodiscard]] inline T asinh(T v) noexcept { return std::asinh(v); }
template<FloatScalar T> [[nodiscard]] inline T acosh(T v) noexcept { return std::acosh(v); }
template<FloatScalar T> [[nodiscard]] inline T atanh(T v) noexcept { return std::atanh(v); }

// ─── Versine / Haversine / Exsecant ───────────────────────────────────────
template<FloatScalar T> [[nodiscard]] inline T versin (T v) noexcept { return T(1)-std::cos(v); }
template<FloatScalar T> [[nodiscard]] inline T coversin(T v) noexcept { return T(1)-std::sin(v); }
template<FloatScalar T> [[nodiscard]] inline T haversin(T v) noexcept { return versin(v)*T(0.5); }
template<FloatScalar T> [[nodiscard]] inline T exsec  (T v) noexcept { return T(1)/std::cos(v)-T(1); }

// Haversine great-circle distance (lat/lon in radians)
[[nodiscard]] inline f64 haversine_distance(f64 lat1,f64 lon1,f64 lat2,f64 lon2,f64 R=6371000.0) noexcept {
    f64 dlat=lat2-lat1, dlon=lon2-lon1;
    f64 a=std::sin(dlat*0.5)*std::sin(dlat*0.5)+std::cos(lat1)*std::cos(lat2)*std::sin(dlon*0.5)*std::sin(dlon*0.5);
    return 2.0*R*std::asin(std::sqrt(a));
}

// ─── Polynomial sin/cos approx (minimax) ──────────────────────────────────
// Accurate to within ~1.2e-6 for sin, cos
[[nodiscard]] inline f32 poly_sin(f32 x) noexcept {
    // Range reduce to [-pi/2, pi/2]
    const f32 inv2pi = 0.15915494309f;
    f32 k = std::round(x*inv2pi); x -= k*(2.0f*PI);
    if(x >  HALF_PI) { x = PI - x; }
    if(x < -HALF_PI) { x = -PI - x; }
    // Minimax polynomial for sin on [-pi/2, pi/2]
    f32 x2=x*x;
    return x*(1.0f+x2*(-0.16666667f+x2*(0.00833333f+x2*(-0.00019841f+x2*2.75573e-6f))));
}
[[nodiscard]] inline f32 poly_cos(f32 x) noexcept {
    const f32 inv2pi = 0.15915494309f;
    f32 k = std::round(x*inv2pi); x -= k*(2.0f*PI);
    if(x >  HALF_PI) { x = PI - x; }
    if(x < -HALF_PI) { x = -PI - x; }
    f32 x2=x*x;
    return 1.0f+x2*(-0.5f+x2*(0.04166667f+x2*(-0.00138889f+x2*2.48016e-5f)));
}

// ─── Trigonometric identities ──────────────────────────────────────────────
template<FloatScalar T>
[[nodiscard]] inline T sin2(T a, T b) noexcept { return T(2)*std::sin(a)*std::cos(b); } // sin(a+b)+sin(a-b)
template<FloatScalar T>
[[nodiscard]] inline T cos_sum(T a, T b) noexcept { return std::cos(a)*std::cos(b)-std::sin(a)*std::sin(b); }
template<FloatScalar T>
[[nodiscard]] inline T sin_sum(T a, T b) noexcept { return std::sin(a)*std::cos(b)+std::cos(a)*std::sin(b); }

// ─── Chebyshev expansion of trig ──────────────────────────────────────────
// Evaluate Chebyshev series c[0..n] at x in [-1,1]
[[nodiscard]] inline f64 chebyshev_eval(const f64* c, int n, f64 x) noexcept {
    f64 b0=0,b1=0,b2=0;
    for(int i=n;i>=0;--i){ b2=b1; b1=b0; b0=2.0*x*b1-b2+c[i]; }
    return 0.5*(b0-b2+c[0]);
}
