
#pragma once
// qs_math/optimization/include/qs_math_optimization.hpp (C++20)
// Numerical optimization: gradient descent, BFGS, Nelder-Mead, simulated annealing, GA.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <functional>
#include <vector>
#include <array>
#include <cmath>
namespace qs::math::opt {
using namespace qs::math;
using ObjFn = std::function<f64(const f64*)>;

struct OptResult { std::vector<f64> x; f64 value; i32 iters; bool converged; };

// ─── Gradient descent with line search ───────────────────────────────────────
OptResult gradient_descent(ObjFn f, const f64* x0, i32 n,
    f64 lr=0.01, f64 tol=1e-8, i32 max_iter=10000) noexcept;

// ─── Adam optimizer ───────────────────────────────────────────────────────────
OptResult adam(ObjFn f, const f64* x0, i32 n,
    f64 lr=0.001, f64 beta1=0.9, f64 beta2=0.999,
    f64 eps=1e-8, f64 tol=1e-8, i32 max_iter=10000) noexcept;

// ─── L-BFGS (limited memory BFGS) ────────────────────────────────────────────
OptResult lbfgs(ObjFn f, const f64* x0, i32 n,
    i32 m=10, f64 tol=1e-8, i32 max_iter=1000) noexcept;

// ─── Nelder-Mead simplex ─────────────────────────────────────────────────────
OptResult nelder_mead(ObjFn f, const f64* x0, i32 n,
    f64 step=0.1, f64 tol=1e-8, i32 max_iter=5000) noexcept;

// ─── Simulated annealing ─────────────────────────────────────────────────────
OptResult simulated_annealing(ObjFn f, const f64* x0, i32 n,
    f64 T0=100, f64 Tf=1e-4, f64 cooling=0.995,
    f64 step=0.1, i32 max_iter=100000, u64 seed=42) noexcept;

// ─── Differential evolution ───────────────────────────────────────────────────
OptResult differential_evolution(ObjFn f, i32 n,
    const f64* lo, const f64* hi,
    i32 pop=20, f64 F=0.8, f64 CR=0.9,
    i32 max_iter=1000, u64 seed=1) noexcept;

// ─── Golden section search (1D) ───────────────────────────────────────────────
[[nodiscard]] f64 golden_section(std::function<f64(f64)> f,
    f64 a, f64 b, f64 tol=1e-10) noexcept;

// ─── Conjugate gradient (Polak-Ribière, unconstrained) ────────────────────────
OptResult conjugate_gradient(ObjFn f, const f64* x0, i32 n,
    f64 tol=1e-8, i32 max_iter=1000) noexcept;
} // namespace qs::math::opt
