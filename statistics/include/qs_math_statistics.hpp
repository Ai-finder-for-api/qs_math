
#pragma once
// qs_math/statistics/include/qs_math_statistics.hpp (C++20)
// Extended statistics: Bayesian inference, bootstrap, ANOVA, time series.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <vector>
#include <functional>
#include <array>
namespace qs::math::stats {
using namespace qs::math;
// ─── Bootstrap CI ─────────────────────────────────────────────────────────────
struct BootstrapCI { f64 lower, upper, mean_est; };
[[nodiscard]] BootstrapCI bootstrap_ci(const f64* data, i32 n,
    std::function<f64(const f64*,i32)> stat,
    i32 B=2000, f64 alpha=0.05, u64 seed=42) noexcept;
// ─── ANOVA (one-way) ──────────────────────────────────────────────────────────
struct ANOVAResult { f64 F_stat, p_value; bool significant; };
[[nodiscard]] ANOVAResult one_way_anova(const std::vector<std::vector<f64>>& groups) noexcept;
// ─── Mann-Whitney U test ──────────────────────────────────────────────────────
struct MWResult { f64 U, p_value; bool significant; };
[[nodiscard]] MWResult mann_whitney(const f64* a, i32 na,
    const f64* b, i32 nb, f64 alpha=0.05) noexcept;
// ─── Time series ─────────────────────────────────────────────────────────────
struct ARModel { std::vector<f64> coefs; f64 intercept;
    [[nodiscard]] f64 predict(const f64* history, i32 p) const noexcept;
};
[[nodiscard]] ARModel fit_ar(const f64* ts, i32 n, i32 p) noexcept;
[[nodiscard]] std::vector<f64> moving_average(const f64* ts, i32 n, i32 window) noexcept;
[[nodiscard]] std::vector<f64> exponential_smoothing(const f64* ts, i32 n, f64 alpha) noexcept;
[[nodiscard]] std::vector<f64> holt_winters(const f64* ts, i32 n,
    f64 alpha, f64 beta, f64 gamma, i32 period) noexcept;
// ─── Principal component analysis (PCA) ──────────────────────────────────────
struct PCAResult {
    std::vector<f64> eigenvalues;
    std::vector<std::vector<f64>> components; // each component is a vector of dim d
    std::vector<f64> explained_variance_ratio;
};
[[nodiscard]] PCAResult pca(const f64* X, i32 n, i32 d, i32 k) noexcept;
// ─── K-means clustering ───────────────────────────────────────────────────────
struct KMeansResult {
    std::vector<i32> labels;
    std::vector<std::vector<f64>> centroids;
    f64 inertia;
};
[[nodiscard]] KMeansResult kmeans(const f64* X, i32 n, i32 d, i32 k,
    i32 max_iter=300, u64 seed=0) noexcept;
// ─── Correlation matrix ───────────────────────────────────────────────────────
[[nodiscard]] std::vector<f64> correlation_matrix(const f64* X, i32 n, i32 d) noexcept;
// ─── Z-score normalization ────────────────────────────────────────────────────
void zscore_normalize(f64* X, i32 n, i32 d) noexcept;
} // namespace qs::math::stats
