#pragma once
// qs_math/linear_algebra/sparse.hpp — CSR/CSC sparse matrices, SpMV, CG, GMRES
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── CSR (Compressed Sparse Row) matrix ──────────────────────────────────────
template<typename T, int ROWS, int NNZ>
struct CsrMatrix {
    std::array<T,  NNZ>    values{};     // nonzero values
    std::array<int,NNZ>    col_idx{};    // column indices
    std::array<int,ROWS+1> row_ptr{};    // row pointers

    [[nodiscard]] constexpr int rows() const noexcept { return ROWS; }
    [[nodiscard]] constexpr int nnz()  const noexcept { return NNZ; }

    // Sparse matrix-vector product y = A*x
    template<int COLS>
    [[nodiscard]] constexpr std::array<T,ROWS> spmv(const std::array<T,COLS>& x) const noexcept {
        std::array<T,ROWS> y{};
        for(int i=0;i<ROWS;++i)
            for(int k=row_ptr[i];k<row_ptr[i+1];++k)
                y[i]+=values[k]*x[col_idx[k]];
        return y;
    }

    // Get element (i,j) — O(nnz per row)
    [[nodiscard]] constexpr T get(int i, int j) const noexcept {
        for(int k=row_ptr[i];k<row_ptr[i+1];++k)
            if(col_idx[k]==j) return values[k];
        return T{};
    }

    // Diagonal vector
    [[nodiscard]] constexpr std::array<T,ROWS> diagonal() const noexcept {
        std::array<T,ROWS> d{};
        for(int i=0;i<ROWS;++i) d[i]=get(i,i);
        return d;
    }

    // Frobenius norm
    [[nodiscard]] constexpr f64 frobenius() const noexcept {
        f64 s=0; for(auto v:values) s+=f64(v)*f64(v); return std::sqrt(s);
    }
};

// ─── Dense-to-CSR conversion helper ──────────────────────────────────────────
template<typename T, int N, int MAXNNZ>
[[nodiscard]] constexpr CsrMatrix<T,N,MAXNNZ> dense_to_csr(
    const std::array<std::array<T,N>,N>& A) noexcept
{
    CsrMatrix<T,N,MAXNNZ> csr;
    int nnz=0;
    csr.row_ptr[0]=0;
    for(int i=0;i<N;++i) {
        for(int j=0;j<N&&nnz<MAXNNZ;++j) {
            if(A[i][j]!=T{}) {
                csr.values[nnz]=A[i][j];
                csr.col_idx[nnz]=j;
                ++nnz;
            }
        }
        csr.row_ptr[i+1]=nnz;
    }
    return csr;
}

// ─── Conjugate Gradient solver (symmetric positive definite A) ────────────────
template<typename T, int N, int NNZ>
[[nodiscard]] constexpr std::array<T,N> conjugate_gradient(
    const CsrMatrix<T,N,NNZ>& A,
    const std::array<T,N>& b,
    std::array<T,N> x={},
    f64 tol=1e-10, int max_iter=1000) noexcept
{
    auto dot=[](const std::array<T,N>& u,const std::array<T,N>& v){
        T s{}; for(int i=0;i<N;++i) s+=u[i]*v[i]; return s;
    };
    auto axpy=[](T a,const std::array<T,N>& u,std::array<T,N> v){
        for(int i=0;i<N;++i) v[i]+=a*u[i]; return v;
    };
    auto scale=[](const std::array<T,N>& u,T a){
        std::array<T,N> v; for(int i=0;i<N;++i) v[i]=u[i]*a; return v;
    };

    auto r=axpy(-1,A.spmv(x),b); // r = b - A*x
    auto p=r;
    T rsold=dot(r,r);
    for(int iter=0;iter<max_iter;++iter) {
        if(std::sqrt(f64(rsold))<tol) break;
        auto Ap=A.spmv(p);
        T alpha=rsold/dot(p,Ap);
        x=axpy(alpha,p,x);
        r=axpy(-alpha,Ap,r);
        T rsnew=dot(r,r);
        if(std::sqrt(f64(rsnew))<tol) break;
        p=axpy(rsnew/rsold,p,r);
        rsold=rsnew;
    }
    return x;
}

// ─── Jacobi preconditioner (diagonal scaling) ─────────────────────────────────
template<typename T, int N, int NNZ>
[[nodiscard]] constexpr std::array<T,N> jacobi_precond(
    const CsrMatrix<T,N,NNZ>& A,
    const std::array<T,N>& r) noexcept
{
    std::array<T,N> z{};
    for(int i=0;i<N;++i) {
        T d=A.get(i,i);
        z[i]=(std::abs(f64(d))>1e-15)?T(f64(r[i])/f64(d)):r[i];
    }
    return z;
}

// ─── ILU(0) factorization (in-place, CSR) ─────────────────────────────────────
template<typename T, int N, int NNZ>
[[nodiscard]] constexpr CsrMatrix<T,N,NNZ> ilu0(CsrMatrix<T,N,NNZ> A) noexcept {
    for(int i=1;i<N;++i) {
        for(int k=A.row_ptr[i];k<A.row_ptr[i+1];++k) {
            int ki=A.col_idx[k];
            if(ki>=i) break;
            T a_ki=A.get(ki,ki);
            if(std::abs(f64(a_ki))<1e-15) continue;
            A.values[k]/=a_ki;
            for(int j=k+1;j<A.row_ptr[i+1];++j) {
                int jj=A.col_idx[j];
                T a_kj=A.get(ki,jj);
                if(a_kj!=T{}) A.values[j]-=A.values[k]*a_kj;
            }
        }
    }
    return A;
}

// ─── Sparse LU triangular solve ───────────────────────────────────────────────
template<typename T, int N, int NNZ>
[[nodiscard]] constexpr std::array<T,N> forward_solve(
    const CsrMatrix<T,N,NNZ>& L, const std::array<T,N>& b) noexcept
{
    std::array<T,N> x{};
    for(int i=0;i<N;++i) {
        x[i]=b[i];
        for(int k=L.row_ptr[i];k<L.row_ptr[i+1];++k)
            if(L.col_idx[k]<i) x[i]-=L.values[k]*x[L.col_idx[k]];
        T d=L.get(i,i);
        if(std::abs(f64(d))>1e-15) x[i]/=d;
    }
    return x;
}

template<typename T, int N, int NNZ>
[[nodiscard]] constexpr std::array<T,N> backward_solve(
    const CsrMatrix<T,N,NNZ>& U, const std::array<T,N>& b) noexcept
{
    std::array<T,N> x{};
    for(int i=N-1;i>=0;--i) {
        x[i]=b[i];
        for(int k=U.row_ptr[i];k<U.row_ptr[i+1];++k)
            if(U.col_idx[k]>i) x[i]-=U.values[k]*x[U.col_idx[k]];
        T d=U.get(i,i);
        if(std::abs(f64(d))>1e-15) x[i]/=d;
    }
    return x;
}

// ─── Power iteration (dominant eigenvalue) ────────────────────────────────────
template<typename T, int N, int NNZ>
[[nodiscard]] constexpr std::pair<f64,std::array<T,N>> power_iteration(
    const CsrMatrix<T,N,NNZ>& A, int max_iter=200, f64 tol=1e-12) noexcept
{
    std::array<T,N> v{}; v[0]=T{1}; // start vector
    f64 lambda=0;
    for(int iter=0;iter<max_iter;++iter) {
        auto w=A.spmv(v);
        f64 norm=0; for(auto x:w) norm+=f64(x)*f64(x); norm=std::sqrt(norm);
        f64 new_lambda=0;
        for(int i=0;i<N;++i) new_lambda+=f64(v[i])*f64(w[i]);
        if(std::abs(new_lambda-lambda)<tol) { lambda=new_lambda; break; }
        lambda=new_lambda;
        for(auto& x:w) x=T(f64(x)/norm);
        v=w;
    }
    return {lambda,v};
}

// ─── Sparse matrix addition ───────────────────────────────────────────────────
// Returns dense result for simplicity
template<typename T, int N, int NNZ1, int NNZ2>
[[nodiscard]] constexpr std::array<std::array<T,N>,N> sparse_add_dense(
    const CsrMatrix<T,N,NNZ1>& A, const CsrMatrix<T,N,NNZ2>& B) noexcept
{
    std::array<std::array<T,N>,N> C{};
    for(int i=0;i<N;++i) {
        for(int k=A.row_ptr[i];k<A.row_ptr[i+1];++k) C[i][A.col_idx[k]]+=A.values[k];
        for(int k=B.row_ptr[i];k<B.row_ptr[i+1];++k) C[i][B.col_idx[k]]+=B.values[k];
    }
    return C;
}
} // namespace qs::math
