
#pragma once
// qs_math/probability/include/qs_math_probability.hpp (C++20)
// Probability distributions: PDF/CDF/inverse CDF, Bayesian, Markov chains.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <vector>
#include <functional>
namespace qs::math::prob {
using namespace qs::math;
// ─── Normal distribution ─────────────────────────────────────────────────────
[[nodiscard]] f64 normal_pdf(f64 x, f64 mu=0, f64 sigma=1) noexcept;
[[nodiscard]] f64 normal_cdf(f64 x, f64 mu=0, f64 sigma=1) noexcept;
[[nodiscard]] f64 normal_inv_cdf(f64 p, f64 mu=0, f64 sigma=1) noexcept;
// ─── Student-t ────────────────────────────────────────────────────────────────
[[nodiscard]] f64 student_t_pdf(f64 t, f64 nu) noexcept;
[[nodiscard]] f64 student_t_cdf(f64 t, f64 nu) noexcept;
// ─── Beta distribution ────────────────────────────────────────────────────────
[[nodiscard]] f64 beta_pdf(f64 x, f64 a, f64 b) noexcept;
[[nodiscard]] f64 beta_cdf(f64 x, f64 a, f64 b) noexcept;
// ─── Gamma distribution ───────────────────────────────────────────────────────
[[nodiscard]] f64 gamma_pdf(f64 x, f64 shape, f64 rate=1) noexcept;
[[nodiscard]] f64 gamma_cdf(f64 x, f64 shape, f64 rate=1) noexcept;
// ─── Binomial distribution ────────────────────────────────────────────────────
[[nodiscard]] f64 binomial_pmf(i32 k, i32 n, f64 p) noexcept;
[[nodiscard]] f64 binomial_cdf(i32 k, i32 n, f64 p) noexcept;
// ─── Poisson distribution ────────────────────────────────────────────────────
[[nodiscard]] f64 poisson_pmf(i32 k, f64 lambda) noexcept;
[[nodiscard]] f64 poisson_cdf(i32 k, f64 lambda) noexcept;
// ─── Exponential distribution ─────────────────────────────────────────────────
[[nodiscard]] f64 exponential_pdf(f64 x, f64 rate) noexcept;
[[nodiscard]] f64 exponential_cdf(f64 x, f64 rate) noexcept;
[[nodiscard]] f64 exponential_inv_cdf(f64 p, f64 rate) noexcept;
// ─── Chi-squared distribution ────────────────────────────────────────────────
[[nodiscard]] f64 chi2_pdf(f64 x, f64 k) noexcept;
[[nodiscard]] f64 chi2_cdf(f64 x, f64 k) noexcept;
[[nodiscard]] f64 chi2_test_pvalue(f64 stat, f64 df) noexcept;
// ─── Multivariate normal ──────────────────────────────────────────────────────
[[nodiscard]] f64 mvn_pdf(const f64* x, const f64* mu,
    const f64* Sigma_inv, f64 det_Sigma, i32 n) noexcept;
// ─── Dirichlet distribution ───────────────────────────────────────────────────
[[nodiscard]] f64 dirichlet_pdf(const f64* x, const f64* alpha, i32 k) noexcept;
// ─── Kalman filter ────────────────────────────────────────────────────────────
struct KalmanFilter {
    std::vector<f64> x;   // state mean
    std::vector<f64> P;   // state covariance (n×n)
    i32 n{0};
    KalmanFilter(i32 state_dim) : x(state_dim,0), P(state_dim*state_dim,0), n(state_dim) {}
    void predict(const f64* F, const f64* Q) noexcept;
    void update(const f64* z, const f64* H, const f64* R, i32 m) noexcept;
};
// ─── Markov chain ────────────────────────────────────────────────────────────
[[nodiscard]] std::vector<f64> markov_stationary(
    const f64* trans, i32 n, i32 max_iter=1000, f64 tol=1e-10) noexcept;
// ─── Entropy / information theory ────────────────────────────────────────────
[[nodiscard]] f64 entropy_shannon(const f64* probs, i32 n) noexcept;
[[nodiscard]] f64 kl_divergence(const f64* p, const f64* q, i32 n) noexcept;
[[nodiscard]] f64 mutual_information(const f64* joint, i32 nx, i32 ny) noexcept;
} // namespace qs::math::prob
