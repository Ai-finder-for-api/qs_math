
#pragma once
// qs_math/numerical/include/qs_math_numerical.hpp (C++20)
// High-level numerical module: special functions, series, transforms.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <vector>
#include <array>
namespace qs::math::numerical {
using namespace qs::math;
// ─── Special functions ───────────────────────────────────────────────────────
[[nodiscard]] f64 gamma_ln(f64 x) noexcept;
[[nodiscard]] f64 gamma_fn(f64 x) noexcept;
[[nodiscard]] f64 beta_fn(f64 a, f64 b) noexcept;
[[nodiscard]] f64 erf_approx(f64 x) noexcept;
[[nodiscard]] f64 erfc_approx(f64 x) noexcept;
[[nodiscard]] f64 digamma(f64 x) noexcept;
[[nodiscard]] f64 bessel_j0(f64 x) noexcept;
[[nodiscard]] f64 bessel_j1(f64 x) noexcept;
[[nodiscard]] f64 bessel_jn(i32 n, f64 x) noexcept;
[[nodiscard]] f64 legendre_p(i32 n, f64 x) noexcept;
[[nodiscard]] f64 chebyshev_t(i32 n, f64 x) noexcept;
[[nodiscard]] f64 laguerre(i32 n, f64 x) noexcept;
[[nodiscard]] f64 hermite(i32 n, f64 x) noexcept;
// ─── Incomplete beta / gamma (regularised) ───────────────────────────────────
[[nodiscard]] f64 inc_beta(f64 a, f64 b, f64 x) noexcept;
[[nodiscard]] f64 inc_gamma_lower(f64 a, f64 x) noexcept;
// ─── Zeta and related ────────────────────────────────────────────────────────
[[nodiscard]] f64 riemann_zeta(f64 s) noexcept;
[[nodiscard]] f64 eta_fn(f64 s) noexcept;
// ─── Integer sequences ────────────────────────────────────────────────────────
[[nodiscard]] u64 fibonacci(i32 n) noexcept;
[[nodiscard]] u64 catalan(i32 n) noexcept;
[[nodiscard]] u64 binomial_coeff(i32 n, i32 k) noexcept;
[[nodiscard]] f64 stirling_approx(i32 n) noexcept;
// ─── Continued fraction evaluation (Lentz's method) ─────────────────────────
[[nodiscard]] f64 continued_fraction(const f64* a, const f64* b,
    i32 n, f64 tol=1e-12) noexcept;
} // namespace qs::math::numerical
