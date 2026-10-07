#pragma once
// qs_math/analysis/functional_analysis.hpp — Banach/Hilbert spaces, operators, spectral theory
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── L² inner product on discrete functions ───────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 l2_inner(const std::array<f64,N>& f,
                                      const std::array<f64,N>& g, f64 dx=1.0) noexcept {
    f64 s=0; for(int i=0;i<N;++i) s+=f[i]*g[i]*dx; return s;
}
template<int N>
[[nodiscard]] constexpr f64 l2_norm(const std::array<f64,N>& f, f64 dx=1.0) noexcept {
    return std::sqrt(l2_inner(f,f,dx));
}
// L^p norm
template<int N>
[[nodiscard]] constexpr f64 lp_norm_vec(const std::array<f64,N>& f, f64 p, f64 dx=1.0) noexcept {
    f64 s=0; for(auto v:f) s+=std::pow(std::abs(v),p)*dx; return std::pow(s,1.0/p);
}

// ─── Gram-Schmidt orthogonalization ───────────────────────────────────────────
template<int N, int K>
[[nodiscard]] constexpr std::array<std::array<f64,N>,K> gram_schmidt(
    const std::array<std::array<f64,N>,K>& vecs, f64 dx=1.0) noexcept
{
    std::array<std::array<f64,N>,K> ortho{};
    for(int i=0;i<K;++i) {
        ortho[i]=vecs[i];
        for(int j=0;j<i;++j) {
            f64 proj=l2_inner<N>(vecs[i],ortho[j],dx)/(l2_inner<N>(ortho[j],ortho[j],dx)+1e-15);
            for(int k=0;k<N;++k) ortho[i][k]-=proj*ortho[j][k];
        }
        f64 nrm=l2_norm<N>(ortho[i],dx);
        if(nrm>1e-12) for(auto& v:ortho[i]) v/=nrm;
    }
    return ortho;
}

// ─── Linear operator as matrix (finite-dimensional) ───────────────────────────
template<int N>
struct LinearOperator {
    std::array<std::array<f64,N>,N> mat{};

    [[nodiscard]] constexpr std::array<f64,N> apply(const std::array<f64,N>& v) const noexcept {
        std::array<f64,N> w{};
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) w[i]+=mat[i][j]*v[j];
        return w;
    }
    [[nodiscard]] constexpr f64 operator_norm() const noexcept {
        // Spectral norm ≈ max row sum (1-norm)
        f64 mx=0;
        for(int i=0;i<N;++i){f64 s=0;for(int j=0;j<N;++j)s+=std::abs(mat[i][j]);if(s>mx)mx=s;}
        return mx;
    }
    [[nodiscard]] constexpr f64 frobenius_norm() const noexcept {
        f64 s=0; for(auto& r:mat) for(auto v:r) s+=v*v; return std::sqrt(s);
    }
    [[nodiscard]] constexpr LinearOperator compose(const LinearOperator& B) const noexcept {
        LinearOperator C{};
        for(int i=0;i<N;++i) for(int k=0;k<N;++k) if(mat[i][k])
            for(int j=0;j<N;++j) C.mat[i][j]+=mat[i][k]*B.mat[k][j];
        return C;
    }
    [[nodiscard]] constexpr LinearOperator adjoint() const noexcept {
        LinearOperator At{};
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) At.mat[j][i]=mat[i][j];
        return At;
    }
    [[nodiscard]] constexpr bool is_self_adjoint(f64 tol=1e-9) const noexcept {
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            if(std::abs(mat[i][j]-mat[j][i])>tol) return false;
        return true;
    }
    [[nodiscard]] constexpr bool is_positive_definite() const noexcept {
        // Check all leading principal minors > 0 (Sylvester's criterion)
        for(int k=1;k<=N;++k) {
            // Compute det of k×k leading submatrix
            f64 det=1;
            std::array<std::array<f64,N>,N> sub=mat;
            for(int col=0;col<k;++col) {
                if(std::abs(sub[col][col])<1e-15) return false;
                det*=sub[col][col];
                for(int row=col+1;row<k;++row) {
                    f64 f=sub[row][col]/sub[col][col];
                    for(int j=col;j<k;++j) sub[row][j]-=f*sub[col][j];
                }
            }
            if(det<=0) return false;
        }
        return true;
    }
    // Trace and determinant
    [[nodiscard]] constexpr f64 trace() const noexcept {
        f64 t=0; for(int i=0;i<N;++i) t+=mat[i][i]; return t;
    }
};

// ─── Projection operator onto subspace ───────────────────────────────────────
template<int N, int K>
[[nodiscard]] constexpr LinearOperator<N> projection_operator(
    const std::array<std::array<f64,N>,K>& basis) noexcept
{
    // P = Σ_i |e_i><e_i| (orthonormal basis assumed)
    LinearOperator<N> P{};
    for(int k=0;k<K;++k)
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            P.mat[i][j]+=basis[k][i]*basis[k][j];
    return P;
}

// ─── Spectral decomposition (symmetric, real) ─────────────────────────────────
template<int N>
struct SpectralDecomp {
    std::array<f64,N> eigenvalues{};
    std::array<std::array<f64,N>,N> eigenvectors{};

    // Build from eigendecomposition: A = Σ λ_i |v_i><v_i|
    [[nodiscard]] constexpr LinearOperator<N> reconstruct() const noexcept {
        LinearOperator<N> A{};
        for(int k=0;k<N;++k) if(std::abs(eigenvalues[k])>1e-15)
            for(int i=0;i<N;++i) for(int j=0;j<N;++j)
                A.mat[i][j]+=eigenvalues[k]*eigenvectors[k][i]*eigenvectors[k][j];
        return A;
    }

    // Matrix function f(A) = Σ f(λ_i)|v_i><v_i|
    template<typename Func>
    [[nodiscard]] constexpr LinearOperator<N> matrix_function(Func&& f) const noexcept {
        LinearOperator<N> fA{};
        for(int k=0;k<N;++k) {
            f64 fk=f(eigenvalues[k]);
            for(int i=0;i<N;++i) for(int j=0;j<N;++j)
                fA.mat[i][j]+=fk*eigenvectors[k][i]*eigenvectors[k][j];
        }
        return fA;
    }

    // Matrix square root (positive semidefinite)
    [[nodiscard]] constexpr LinearOperator<N> matrix_sqrt() const noexcept {
        return matrix_function([](f64 x){return x>0?std::sqrt(x):0.0;});
    }
    // Matrix exponential
    [[nodiscard]] constexpr LinearOperator<N> matrix_exp() const noexcept {
        return matrix_function([](f64 x){return std::exp(x);});
    }
    // Matrix log
    [[nodiscard]] constexpr LinearOperator<N> matrix_log() const noexcept {
        return matrix_function([](f64 x){return x>1e-15?std::log(x):-1e15;});
    }
};

// ─── Compact operators (finite rank approximation) ────────────────────────────
template<int N, int R>
[[nodiscard]] constexpr LinearOperator<N> finite_rank_op(
    const std::array<std::array<f64,N>,R>& u,
    const std::array<std::array<f64,N>,R>& v,
    const std::array<f64,R>& sigma) noexcept
{
    // T = Σ σ_k |u_k><v_k|
    LinearOperator<N> T{};
    for(int k=0;k<R;++k) for(int i=0;i<N;++i) for(int j=0;j<N;++j)
        T.mat[i][j]+=sigma[k]*u[k][i]*v[k][j];
    return T;
}

// ─── Resolvent operator R(λ) = (λI - A)^{-1} ─────────────────────────────────
// Resolvent via Neumann series for |λ| > ||A||
template<int N>
[[nodiscard]] constexpr LinearOperator<N> resolvent_neumann(
    const LinearOperator<N>& A, f64 lambda, int terms=20) noexcept
{
    // R(λ) = (1/λ) Σ (A/λ)^n
    LinearOperator<N> R{},An{};
    for(int i=0;i<N;++i){R.mat[i][i]=1.0/lambda;An.mat[i][i]=1.0;}
    f64 inv_lam=1.0/lambda;
    for(int n=1;n<terms;++n) {
        An=An.compose(A);
        f64 coeff=std::pow(inv_lam,n+1);
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) R.mat[i][j]+=coeff*An.mat[i][j];
    }
    return R;
}

// ─── Sobolev space norms (H^s) ────────────────────────────────────────────────
// H^s norm: ||f||_{H^s}² = Σ (1+k²)^s |f_k|²
template<int N>
[[nodiscard]] constexpr f64 sobolev_norm_sq(const std::array<f64,N>& fourier_coeffs, f64 s) noexcept {
    f64 sum=0;
    for(int k=0;k<N;++k) {
        f64 kf=f64(k)-N/2; // centered frequencies
        sum+=std::pow(1+kf*kf,s)*fourier_coeffs[k]*fourier_coeffs[k];
    }
    return sum;
}

// ─── Fixed point theorems (Banach) ───────────────────────────────────────────
// Banach fixed-point: T contraction → unique fixed point
template<int N, typename T>
[[nodiscard]] constexpr std::pair<std::array<f64,N>,int> banach_fixed_point(
    T&& contraction, std::array<f64,N> x0, f64 tol=1e-10, int max_iter=1000) noexcept
{
    std::array<f64,N> x=x0;
    for(int iter=0;iter<max_iter;++iter) {
        auto xn=contraction(x);
        f64 diff=0; for(int i=0;i<N;++i){f64 d=xn[i]-x[i];diff+=d*d;}
        x=xn;
        if(diff<tol*tol) return{x,iter};
    }
    return{x,max_iter};
}
} // namespace qs::math
