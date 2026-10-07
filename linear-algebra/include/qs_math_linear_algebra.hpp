
#pragma once
// qs_math/linear-algebra/include/qs_math_linear_algebra.hpp (C++20)
// Dense/sparse linear algebra: SVD, eigenvalues, iterative solvers, tensor ops.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/mat3.hpp>
#include <qs_math/linear_algebra/mat4.hpp>
#include <cmath>
#include <vector>
#include <array>
namespace qs::math::linalg {
using namespace qs::math;
// ─── Dense matrix (heap, m×n row-major) ──────────────────────────────────────
struct DMatrix {
    i32 m{0},n{0};
    std::vector<f64> data;
    DMatrix()=default;
    DMatrix(i32 m,i32 n,f64 fill=0):m(m),n(n),data(m*n,fill){}
    [[nodiscard]] f64& at(i32 r,i32 c)noexcept{return data[r*n+c];}
    [[nodiscard]] f64  at(i32 r,i32 c)const noexcept{return data[r*n+c];}
    [[nodiscard]] static DMatrix identity(i32 n) noexcept;
    [[nodiscard]] DMatrix operator*(const DMatrix& o) const noexcept;
    [[nodiscard]] DMatrix transposed() const noexcept;
    [[nodiscard]] f64 frobenius_norm() const noexcept;
    [[nodiscard]] f64 trace() const noexcept;
};
// ─── SVD (Golub-Reinsch, m>=n) ────────────────────────────────────────────────
struct SVDResult { DMatrix U,S,Vt; };
[[nodiscard]] SVDResult svd(DMatrix A) noexcept;
// ─── Symmetric eigenvalues (QR algorithm with Householder reduction) ──────────
struct EigenResult { std::vector<f64> values; DMatrix vectors; };
[[nodiscard]] EigenResult eigen_symmetric(DMatrix A, i32 max_iter=200) noexcept;
// ─── Cholesky decomposition (SPD matrix) ─────────────────────────────────────
[[nodiscard]] bool cholesky(const DMatrix& A, DMatrix& L) noexcept;
void cholesky_solve(const DMatrix& L, const f64* b, f64* x, i32 n) noexcept;
// ─── Iterative solvers ────────────────────────────────────────────────────────
void jacobi_iter(const DMatrix& A, const f64* b, f64* x,
    i32 max_iter=1000, f64 tol=1e-10) noexcept;
void gauss_seidel(const DMatrix& A, const f64* b, f64* x,
    i32 max_iter=1000, f64 tol=1e-10) noexcept;
// ─── Tensor outer product (rank-1 update) ─────────────────────────────────────
[[nodiscard]] DMatrix outer_product(const f64* u, const f64* v, i32 m, i32 n) noexcept;
// ─── Rank-k approximation via truncated SVD ───────────────────────────────────
[[nodiscard]] DMatrix low_rank_approx(const DMatrix& A, i32 k) noexcept;
// ─── 3×3 real eigenvalues (Cardano) ──────────────────────────────────────────
[[nodiscard]] std::array<f64,3> eigenvalues_3x3_sym(const Mat3<f64>& A) noexcept;
} // namespace qs::math::linalg
