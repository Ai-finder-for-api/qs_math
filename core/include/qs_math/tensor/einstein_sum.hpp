#pragma once
// qs_math/tensor/einstein_sum.hpp — Einstein summation, Levi-Civita, index gymnastics
#include "../arithmetic/scalar_types.hpp"
#include "tensor_core.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Levi-Civita symbol ε_{ijk} (3D) ─────────────────────────────────────────
[[nodiscard]] constexpr int levi_civita_3(int i, int j, int k) noexcept {
    if(i==j||j==k||i==k) return 0;
    // Even permutation of (0,1,2) → +1, odd → -1
    return ((j-i)*(k-j)*(k-i)>0)?1:-1;
}

// Levi-Civita symbol ε_{ijkl} (4D)
[[nodiscard]] constexpr int levi_civita_4(int i, int j, int k, int l) noexcept {
    const int idx[4]={i,j,k,l};
    for(int a=0;a<4;++a) for(int b=a+1;b<4;++b) if(idx[a]==idx[b]) return 0;
    // Count inversions
    int inv=0;
    for(int a=0;a<4;++a) for(int b=a+1;b<4;++b) if(idx[a]>idx[b]) ++inv;
    return (inv%2==0)?1:-1;
}

// Generalized Levi-Civita for rank N (runtime)
[[nodiscard]] constexpr int levi_civita_n(const int* idx, int N) noexcept {
    for(int a=0;a<N;++a) for(int b=a+1;b<N;++b) if(idx[a]==idx[b]) return 0;
    int inv=0;
    for(int a=0;a<N;++a) for(int b=a+1;b<N;++b) if(idx[a]>idx[b]) ++inv;
    return (inv%2==0)?1:-1;
}

// Levi-Civita tensor as Tensor<int,3,3,3>
[[nodiscard]] constexpr Tensor<int,3,3,3> levi_civita_tensor_3d() noexcept {
    Tensor<int,3,3,3> eps{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
        eps(i,j,k)=levi_civita_3(i,j,k);
    return eps;
}

// ─── Cross product via Levi-Civita: (a×b)_i = ε_{ijk} a_j b_k ───────────────
template<typename T>
[[nodiscard]] constexpr Tensor<T,3> cross_product(
    const Tensor<T,3>& a, const Tensor<T,3>& b) noexcept
{
    Tensor<T,3> c{};
    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
                c[i]+=T(levi_civita_3(i,j,k))*a[j]*b[k];
    return c;
}

// ─── Metric tensor and index lowering/raising ─────────────────────────────────
// Flat Euclidean metric g_{ij} = delta_{ij}
template<int N>
[[nodiscard]] constexpr Tensor<f64,N,N> metric_euclidean() noexcept {
    Tensor<f64,N,N> g{};
    for(int i=0;i<N;++i) g(i,i)=1.0;
    return g;
}

// Minkowski metric g_{μν} = diag(-1,+1,+1,+1) (signature -+++)
[[nodiscard]] constexpr Tensor<f64,4,4> metric_minkowski() noexcept {
    Tensor<f64,4,4> g{};
    g(0,0)=-1.0; g(1,1)=1.0; g(2,2)=1.0; g(3,3)=1.0;
    return g;
}

// Lower index: v_i = g_{ij} v^j
template<int N>
[[nodiscard]] constexpr Tensor<f64,N> lower_index(
    const Tensor<f64,N>& v, const Tensor<f64,N,N>& g) noexcept
{
    Tensor<f64,N> u{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) u[i]+=g(i,j)*v[j];
    return u;
}

// Raise index: v^i = g^{ij} v_j (g_inv = metric inverse)
template<int N>
[[nodiscard]] constexpr Tensor<f64,N> raise_index(
    const Tensor<f64,N>& v, const Tensor<f64,N,N>& g_inv) noexcept
{
    return lower_index(v,g_inv); // same operation with inverse metric
}

// ─── Christoffel symbols Γ^λ_{μν} = (1/2) g^{λσ}(∂_μ g_{νσ}+∂_ν g_{μσ}-∂_σ g_{μν}) ────
// Given metric evaluated on a grid, finite-difference Christoffel
template<int N>
[[nodiscard]] constexpr Tensor<f64,N,N,N> christoffel_fd(
    const Tensor<f64,N,N>& g_at_x,
    const Tensor<f64,N,N,N>& dg, // dg[sigma][mu][nu] = ∂_σ g_{μν}
    const Tensor<f64,N,N>& g_inv) noexcept
{
    Tensor<f64,N,N,N> Gamma{}; // Gamma[lambda][mu][nu]
    for(int lam=0;lam<N;++lam)
        for(int mu=0;mu<N;++mu)
            for(int nu=0;nu<N;++nu) {
                f64 s=0;
                for(int sig=0;sig<N;++sig)
                    s+=g_inv(lam,sig)*(dg(mu,nu,sig)+dg(nu,mu,sig)-dg(sig,mu,nu));
                Gamma(lam,mu,nu)=0.5*s;
            }
    return Gamma;
}

// ─── Riemann curvature tensor (rank 4) ────────────────────────────────────────
// R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}
// Provide both Γ and ∂Γ (finite-differenced externally)
template<int N>
[[nodiscard]] constexpr Tensor<f64,N,N,N,N> riemann_tensor(
    const Tensor<f64,N,N,N>& Gamma,
    const Tensor<f64,N,N,N,N>& dGamma) // dGamma[mu][rho][nu][sigma] = ∂_μ Γ^ρ_{νσ}
noexcept
{
    Tensor<f64,N,N,N,N> R{};
    for(int rho=0;rho<N;++rho)
        for(int sig=0;sig<N;++sig)
            for(int mu=0;mu<N;++mu)
                for(int nu=0;nu<N;++nu) {
                    f64 v=dGamma(mu,rho,nu,sig)-dGamma(nu,rho,mu,sig);
                    for(int lam=0;lam<N;++lam)
                        v+=Gamma(rho,mu,lam)*Gamma(lam,nu,sig)
                          -Gamma(rho,nu,lam)*Gamma(lam,mu,sig);
                    R(rho,sig,mu,nu)=v;
                }
    return R;
}

// ─── Ricci tensor: R_{μν} = R^ρ_{μρν} ────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr Tensor<f64,N,N> ricci_tensor(
    const Tensor<f64,N,N,N,N>& R) noexcept
{
    Tensor<f64,N,N> Ric{};
    for(int mu=0;mu<N;++mu) for(int nu=0;nu<N;++nu)
        for(int rho=0;rho<N;++rho) Ric(mu,nu)+=R(rho,mu,rho,nu);
    return Ric;
}

// Ricci scalar: R = g^{μν} R_{μν}
template<int N>
[[nodiscard]] constexpr f64 ricci_scalar(
    const Tensor<f64,N,N>& Ric, const Tensor<f64,N,N>& g_inv) noexcept
{
    return double_contract(g_inv,Ric);
}

// ─── Wedge product of 1-forms (p-forms) ──────────────────────────────────────
// (α∧β)_{ij} = α_i β_j - α_j β_i
template<int N>
[[nodiscard]] constexpr Tensor<f64,N,N> wedge_1form(
    const Tensor<f64,N>& a, const Tensor<f64,N>& b) noexcept
{
    Tensor<f64,N,N> w{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) w(i,j)=a[i]*b[j]-a[j]*b[i];
    return w;
}

// Hodge dual of a 2-form in 3D: (*F)_k = (1/2) ε_{ijk} F^{ij}
template<typename T>
[[nodiscard]] constexpr Tensor<T,3> hodge_dual_2form_3d(const Tensor<T,3,3>& F) noexcept {
    Tensor<T,3> dual{};
    for(int k=0;k<3;++k)
        for(int i=0;i<3;++i)
            for(int j=0;j<3;++j)
                dual[k]+=T(levi_civita_3(i,j,k))*F(i,j)*T(0.5);
    return dual;
}

} // namespace qs::math
