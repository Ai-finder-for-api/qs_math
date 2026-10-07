#pragma once
// qs_math/least_squares.hpp — Normal equations, pseudoinverse, NNLS, weighted LS (C++20)
#include "scalar_types.hpp"
#include "linear_algebra.hpp"
#include "eigensolvers.hpp"
#include <vector>
#include <cmath>
#include <optional>
#include <cassert>

namespace qs {

// ─── Ordinary least squares via normal equations: x = (A^T A)^{-1} A^T b ─────
template<typename T>
[[nodiscard]] std::optional<std::vector<T>> least_squares_normal(
    const DenseMatrix<T>& A, const std::vector<T>& b, T tol=T(1e-10))
{
    assert(A.rows==u32(b.size()));
    auto At  = A.transpose();
    auto AtA = At * A;
    // A^T b
    std::vector<T> Atb(A.cols, T(0));
    for(u32 i=0;i<A.cols;++i) for(u32 j=0;j<A.rows;++j) Atb[i]+=At.at(i,j)*b[j];
    return solve(AtA, Atb, tol);
}

// ─── Moore-Penrose pseudoinverse A^+ = V Σ^+ U^T (via Jacobi SVD) ────────────
// Uses eigendecomposition of A^T A for the SVD
template<typename T>
[[nodiscard]] DenseMatrix<f64> pseudoinverse(const DenseMatrix<T>& A, f64 tol=1e-10) {
    u32 m=A.rows, n=A.cols;
    // Compute A^T A and its eigen-decomposition
    DenseMatrix<f64> Ad(m,n); for(u32 i=0;i<m;++i) for(u32 j=0;j<n;++j) Ad.at(i,j)=f64(A.at(i,j));
    auto At = Ad.transpose();
    auto AtA = At * Ad;
    auto eig = jacobi_eigen(AtA);
    // Σ^+ : invert non-tiny singular values (sqrt of eigenvalues)
    u32 r=n;
    DenseMatrix<f64> Sigma_plus(n,m); // n×m
    for(u32 i=0;i<n;++i) {
        f64 sv = eig.eigenvalues[i] > 0 ? std::sqrt(eig.eigenvalues[i]) : 0.0;
        if(i<m && sv>tol) Sigma_plus.at(i,i) = 1.0/sv;
        else if(sv<=tol && i<r) r=i;
    }
    // Compute U = A V Σ^{-1} for columns with non-zero singular values
    auto V = eig.eigenvectors; // n×n
    auto AV = Ad * V;          // m×n
    DenseMatrix<f64> U(m,n);
    for(u32 j=0;j<n;++j) {
        f64 sv = eig.eigenvalues[j]>0 ? std::sqrt(eig.eigenvalues[j]) : 0.0;
        if(sv>tol) for(u32 i=0;i<m;++i) U.at(i,j)=AV.at(i,j)/sv;
    }
    // A^+ = V Σ^+ U^T
    auto Ut = U.transpose();
    return V * (Sigma_plus * Ut);
}

// ─── Weighted least squares: minimise (Ax-b)^T W (Ax-b) ──────────────────────
// W is diagonal, given as a weight vector w[i]
template<typename T>
[[nodiscard]] std::optional<std::vector<T>> weighted_least_squares(
    const DenseMatrix<T>& A, const std::vector<T>& b, const std::vector<T>& w, T tol=T(1e-10))
{
    assert(A.rows==u32(b.size()) && A.rows==u32(w.size()));
    // Form A^T W A and A^T W b
    DenseMatrix<T> AtWA(A.cols, A.cols);
    std::vector<T> AtWb(A.cols, T(0));
    for(u32 i=0;i<A.rows;++i) {
        for(u32 j=0;j<A.cols;++j) {
            AtWb[j] += A.at(i,j)*w[i]*b[i];
            for(u32 k=0;k<A.cols;++k)
                AtWA.at(j,k) += A.at(i,j)*w[i]*A.at(i,k);
        }
    }
    return solve(AtWA, AtWb, tol);
}

// ─── Regularised least squares (Tikhonov): minimise ||Ax-b||^2 + λ||x||^2 ───
template<typename T>
[[nodiscard]] std::optional<std::vector<T>> ridge_regression(
    const DenseMatrix<T>& A, const std::vector<T>& b, T lambda=T(1e-3), T tol=T(1e-10))
{
    u32 n=A.cols;
    auto At  = A.transpose();
    auto AtA = At * A;
    // Add λI
    for(u32 i=0;i<n;++i) AtA.at(i,i)+=lambda;
    std::vector<T> Atb(n,T(0));
    for(u32 i=0;i<n;++i) for(u32 j=0;j<A.rows;++j) Atb[i]+=At.at(i,j)*b[j];
    return solve(AtA, Atb, tol);
}

// ─── NNLS — Non-negative least squares (Lawson-Hanson active set) ─────────────
// Finds x ≥ 0 minimising ||Ax - b||^2
template<typename T>
[[nodiscard]] std::vector<T> nnls(const DenseMatrix<T>& A, const std::vector<T>& b,
    u32 max_iter=1000, T tol=T(1e-8))
{
    u32 m=A.rows, n=A.cols;
    std::vector<T> x(n, T(0));
    std::vector<bool> passive(n, false); // passive set: unconstrained variables
    auto At = A.transpose();

    for(u32 outer=0;outer<max_iter;++outer) {
        // Compute gradient w = A^T(b - Ax)
        auto Ax=A*x;
        std::vector<T> w(n,T(0));
        for(u32 i=0;i<n;++i) for(u32 j=0;j<m;++j) w[i]+=At.at(i,j)*(b[j]-Ax[j]);
        // Check KKT: find max w[i] for i not in passive
        f64 max_w=-1e300; u32 t=n;
        for(u32 i=0;i<n;++i) if(!passive[i] && f64(w[i])>max_w){max_w=f64(w[i]);t=i;}
        if(t==n || max_w<=f64(tol)) break;
        passive[t]=true;
        // Inner loop: solve unconstrained LS over passive set, enforce non-negativity
        while(true) {
            // Build sub-problem for passive set
            std::vector<u32> P; for(u32 i=0;i<n;++i) if(passive[i]) P.push_back(i);
            u32 p=u32(P.size());
            DenseMatrix<T> Ap(m,p);
            for(u32 i=0;i<m;++i) for(u32 j=0;j<p;++j) Ap.at(i,j)=A.at(i,P[j]);
            auto sp_opt=least_squares_normal(Ap,b,tol);
            if(!sp_opt) break;
            auto& sp=*sp_opt;
            // Check all s[j] > 0
            bool all_pos=true;
            for(u32 j=0;j<p;++j) if(sp[j]<=T(0)){all_pos=false;break;}
            if(all_pos) {
                for(u32 j=0;j<p;++j) x[P[j]]=sp[j];
                for(u32 i=0;i<n;++i) if(!passive[i]) x[i]=T(0);
                break;
            }
            // Find alpha: max step before any x hits zero
            T alpha=std::numeric_limits<T>::max();
            for(u32 j=0;j<p;++j) if(sp[j]<=T(0)){
                T a=x[P[j]]/(x[P[j]]-sp[j]);
                if(a<alpha) alpha=a;
            }
            for(u32 j=0;j<p;++j) x[P[j]]+=alpha*(sp[j]-x[P[j]]);
            for(u32 i=0;i<n;++i) if(passive[i] && x[i]<=tol){passive[i]=false;x[i]=T(0);}
        }
    }
    return x;
}

// ─── Total least squares (errors in both A and b) ─────────────────────────────
// Uses SVD of [A|b]; last right singular vector gives solution
template<typename T>
[[nodiscard]] std::optional<std::vector<T>> total_least_squares(
    const DenseMatrix<T>& A, const std::vector<T>& b, T tol=T(1e-10))
{
    u32 m=A.rows, n=A.cols;
    // Augmented matrix [A | b]
    DenseMatrix<T> Ab(m, n+1);
    for(u32 i=0;i<m;++i){ for(u32 j=0;j<n;++j) Ab.at(i,j)=A.at(i,j); Ab.at(i,n)=b[i]; }
    // SVD via eigendecomposition of Ab^T Ab
    auto AbT = Ab.transpose();
    auto C   = AbT * Ab; // (n+1)×(n+1)
    auto eig = jacobi_eigen(C, 500, f64(tol));
    // Smallest singular value → last eigenvector (eigenvalues sorted ascending)
    u32 k=0; // smallest eigenvalue index
    u32 np1=n+1;
    std::vector<T> v(np1);
    for(u32 i=0;i<np1;++i) v[i]=T(eig.eigenvectors.at(i,k));
    if(std::abs(v[n])<T(tol)) return std::nullopt; // rank deficient
    std::vector<T> x(n);
    for(u32 i=0;i<n;++i) x[i]=-v[i]/v[n];
    return x;
}

// ─── Iteratively Reweighted Least Squares (IRLS) for L1/Lp regression ────────
template<typename T>
[[nodiscard]] std::vector<T> irls(const DenseMatrix<T>& A, const std::vector<T>& b,
    T p=T(1), u32 max_iter=50, T tol=T(1e-6))
{
    // Minimise Σ|r_i|^p via repeated WLS, w_i = |r_i|^{p-2}
    u32 m=A.rows, n=A.cols;
    std::vector<T> x(n,T(0));
    auto res=least_squares_normal(A,b,T(1e-10));
    if(res) x=*res;
    for(u32 it=0;it<max_iter;++it) {
        auto Ax=A*x;
        std::vector<T> w(m);
        for(u32 i=0;i<m;++i) {
            T r=std::abs(b[i]-Ax[i]);
            w[i]=std::pow(r+T(1e-8), p-T(2));
        }
        auto xn_opt=weighted_least_squares(A,b,w,T(1e-10));
        if(!xn_opt) break;
        T diff=T(0); for(u32 i=0;i<n;++i) diff+=std::abs((*xn_opt)[i]-x[i]);
        x=*xn_opt;
        if(diff<tol) break;
    }
    return x;
}

} // namespace qs
