#pragma once
// qs_math/linear_algebra.hpp — Dense matrix up to 16×16, row ops, rank, RREF (C++20)
#include "scalar_types.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <cassert>
#include <optional>
#include <algorithm>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// DenseMatrix<T> — runtime-sized dense matrix (up to ~16×16 recommended, no hard limit)
// Row-major storage.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T = f32>
struct DenseMatrix {
    u32 rows{0}, cols{0};
    std::vector<T> d; // row-major

    DenseMatrix() = default;
    DenseMatrix(u32 r, u32 c, T fill = T(0)) : rows(r), cols(c), d(r*c, fill) {}

    [[nodiscard]] T& at(u32 r, u32 c)       { return d[r*cols+c]; }
    [[nodiscard]] T  at(u32 r, u32 c) const { return d[r*cols+c]; }
    [[nodiscard]] T& operator()(u32 r,u32 c)       { return at(r,c); }
    [[nodiscard]] T  operator()(u32 r,u32 c) const { return at(r,c); }

    // Identity
    [[nodiscard]] static DenseMatrix identity(u32 n) {
        DenseMatrix M(n,n);
        for(u32 i=0;i<n;++i) M.at(i,i)=T(1);
        return M;
    }
    // Transpose
    [[nodiscard]] DenseMatrix transpose() const {
        DenseMatrix R(cols,rows);
        for(u32 i=0;i<rows;++i) for(u32 j=0;j<cols;++j) R.at(j,i)=at(i,j);
        return R;
    }

    // Arithmetic
    [[nodiscard]] DenseMatrix operator+(const DenseMatrix& b) const {
        assert(rows==b.rows&&cols==b.cols);
        DenseMatrix R=*this; for(u32 i=0;i<u32(d.size());++i) R.d[i]+=b.d[i]; return R;
    }
    [[nodiscard]] DenseMatrix operator-(const DenseMatrix& b) const {
        assert(rows==b.rows&&cols==b.cols);
        DenseMatrix R=*this; for(u32 i=0;i<u32(d.size());++i) R.d[i]-=b.d[i]; return R;
    }
    [[nodiscard]] DenseMatrix operator*(T s) const {
        DenseMatrix R=*this; for(auto& v:R.d) v*=s; return R;
    }
    [[nodiscard]] DenseMatrix operator*(const DenseMatrix& b) const {
        assert(cols==b.rows);
        DenseMatrix R(rows, b.cols);
        for(u32 i=0;i<rows;++i) for(u32 k=0;k<cols;++k) for(u32 j=0;j<b.cols;++j)
            R.at(i,j)+=at(i,k)*b.at(k,j);
        return R;
    }
    [[nodiscard]] std::vector<T> operator*(const std::vector<T>& v) const {
        assert(cols==u32(v.size()));
        std::vector<T> r(rows, T(0));
        for(u32 i=0;i<rows;++i) for(u32 j=0;j<cols;++j) r[i]+=at(i,j)*v[j];
        return r;
    }

    // Row operations
    void row_swap(u32 a, u32 b) {
        for(u32 j=0;j<cols;++j) std::swap(at(a,j),at(b,j));
    }
    void row_scale(u32 r, T s) { for(u32 j=0;j<cols;++j) at(r,j)*=s; }
    void row_add(u32 dst, u32 src, T s) { for(u32 j=0;j<cols;++j) at(dst,j)+=at(src,j)*s; }

    // Norms
    [[nodiscard]] T frobenius_norm() const {
        T s=T(0); for(auto v:d) s+=v*v; return std::sqrt(s);
    }
    [[nodiscard]] T max_abs() const {
        T m=T(0); for(auto v:d) if(std::abs(v)>m) m=std::abs(v); return m;
    }
    [[nodiscard]] T trace() const {
        assert(rows==cols); T s=T(0); for(u32 i=0;i<rows;++i) s+=at(i,i); return s;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian elimination with partial pivoting → RREF, rank, determinant
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
struct GaussResult {
    DenseMatrix<T> rref;
    u32  rank{0};
    T    determinant{1};      // only valid for square matrices
    std::vector<u32> pivot_cols;
};

template<typename T>
[[nodiscard]] GaussResult<T> gaussian_eliminate(DenseMatrix<T> M, T tol = T(1e-9)) {
    GaussResult<T> res;
    res.rref = M;
    auto& A = res.rref;
    u32 r=0;
    T det_sign = T(1);
    for(u32 col=0; col<A.cols && r<A.rows; ++col) {
        // Find pivot
        u32 pivot = r;
        T best = std::abs(A.at(r,col));
        for(u32 i=r+1;i<A.rows;++i) if(std::abs(A.at(i,col))>best) { best=std::abs(A.at(i,col)); pivot=i; }
        if(best < tol) continue;
        if(pivot!=r) { A.row_swap(r,pivot); det_sign=-det_sign; }
        res.determinant *= A.at(r,col);
        res.pivot_cols.push_back(col);
        T inv = T(1)/A.at(r,col);
        A.row_scale(r,inv);
        for(u32 i=0;i<A.rows;++i) {
            if(i==r) continue;
            T factor = A.at(i,col);
            if(std::abs(factor)>tol) A.row_add(i,r,-factor);
        }
        ++r;
    }
    res.rank = r;
    res.determinant *= det_sign;
    return res;
}

// Determinant
template<typename T>
[[nodiscard]] T determinant(const DenseMatrix<T>& M) {
    assert(M.rows==M.cols);
    auto res = gaussian_eliminate(M);
    if(res.rank < M.rows) return T(0);
    return res.determinant;
}

// Matrix inverse via Gauss-Jordan on augmented [A|I]
template<typename T>
[[nodiscard]] std::optional<DenseMatrix<T>> inverse(const DenseMatrix<T>& M, T tol=T(1e-9)) {
    assert(M.rows==M.cols);
    u32 n=M.rows;
    DenseMatrix<T> Aug(n,2*n);
    for(u32 i=0;i<n;++i){ for(u32 j=0;j<n;++j) Aug.at(i,j)=M.at(i,j); Aug.at(i,i+n)=T(1); }
    auto res = gaussian_eliminate(Aug, tol);
    if(res.rank<n) return std::nullopt;
    DenseMatrix<T> Inv(n,n);
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) Inv.at(i,j)=res.rref.at(i,j+n);
    return Inv;
}

// Null space basis (columns spanning ker(M))
template<typename T>
[[nodiscard]] std::vector<std::vector<T>> null_space(const DenseMatrix<T>& M, T tol=T(1e-9)) {
    auto res = gaussian_eliminate(M, tol);
    u32 n=M.cols;
    // Free variables are those not in pivot_cols
    std::vector<bool> is_pivot(n,false);
    for(auto c:res.pivot_cols) is_pivot[c]=true;
    std::vector<std::vector<T>> basis;
    for(u32 free=0;free<n;++free) {
        if(is_pivot[free]) continue;
        std::vector<T> v(n,T(0)); v[free]=T(1);
        for(u32 pi=0;pi<u32(res.pivot_cols.size());++pi) {
            u32 pc=res.pivot_cols[pi];
            v[pc]=-res.rref.at(pi,free);
        }
        basis.push_back(v);
    }
    return basis;
}

// Solve Ax=b via augmented elimination
template<typename T>
[[nodiscard]] std::optional<std::vector<T>> solve(const DenseMatrix<T>& A, const std::vector<T>& b, T tol=T(1e-9)) {
    assert(A.rows==u32(b.size()));
    u32 n=A.rows;
    DenseMatrix<T> Aug(n,n+1);
    for(u32 i=0;i<n;++i){ for(u32 j=0;j<n;++j) Aug.at(i,j)=A.at(i,j); Aug.at(i,n)=b[i]; }
    auto res = gaussian_eliminate(Aug,tol);
    if(res.rank<n) return std::nullopt;
    std::vector<T> x(n);
    for(u32 i=0;i<n;++i) x[i]=res.rref.at(i,n);
    return x;
}

// Condition number estimate (ratio of largest to smallest pivot)
template<typename T>
[[nodiscard]] T condition_number_estimate(const DenseMatrix<T>& M, T tol=T(1e-12)) {
    auto res = gaussian_eliminate(M, tol);
    if(res.rank==0) return std::numeric_limits<T>::infinity();
    // Use diagonal of U after elimination
    T mx=T(0), mn=std::numeric_limits<T>::max();
    for(u32 i=0;i<res.rank;++i) {
        T v=std::abs(M.at(i,i)); // rough estimate
        if(v>mx) mx=v; if(v<mn) mn=v;
    }
    return mn>tol ? mx/mn : std::numeric_limits<T>::infinity();
}

// Build companion matrix for polynomial a[0] + a[1]x + ... + a[n]x^n (monic)
template<typename T>
[[nodiscard]] DenseMatrix<T> companion_matrix(const std::vector<T>& poly) {
    u32 n = u32(poly.size())-1;
    DenseMatrix<T> C(n,n);
    for(u32 i=1;i<n;++i) C.at(i,i-1)=T(1);
    for(u32 i=0;i<n;++i) C.at(i,n-1)=-poly[i]/poly[n];
    return C;
}

} // namespace qs
