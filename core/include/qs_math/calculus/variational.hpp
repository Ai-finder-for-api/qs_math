#pragma once
// qs_math/calculus/variational.hpp — Euler-Lagrange, brachistochrone, geodesics, optimal control
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Euler-Lagrange equation residual ─────────────────────────────────────────
// L(t,y,y') → d/dt ∂L/∂y' - ∂L/∂y = 0
// Numerical check at a point
template<typename L>
[[nodiscard]] constexpr f64 euler_lagrange_residual(L&& lagrangian,
    f64 t, f64 y, f64 dy, f64 dt=1e-5) noexcept
{
    // ∂L/∂y (at current y, dy)
    f64 dLdy=(lagrangian(t,y+dt,dy)-lagrangian(t,y-dt,dy))/(2*dt);
    // d/dt[∂L/∂dy'] via finite time difference (requires y(t±h))
    // Approximate with central difference on ∂L/∂dy'
    f64 dLdyp=(lagrangian(t,y,dy+dt)-lagrangian(t,y,dy-dt))/(2*dt);
    // d/dt[∂L/∂y'] ≈ second mixed partial
    f64 d2Ldtdyp=(lagrangian(t+dt,y,dy)-lagrangian(t-dt,y,dy))/(2*dt*dt)*dt;
    return d2Ldtdyp-dLdy;
}

// ─── Brachistochrone: cycloid parametric ──────────────────────────────────────
// x(θ) = r(θ-sinθ), y(θ) = r(1-cosθ) — time-optimal curve under gravity
[[nodiscard]] constexpr std::pair<f64,f64> brachistochrone_point(f64 r, f64 theta) noexcept {
    return {r*(theta-std::sin(theta)), r*(1-std::cos(theta))};
}
// Descent time: T = π*sqrt(2r/g)
[[nodiscard]] constexpr f64 brachistochrone_time(f64 r, f64 g=9.81) noexcept {
    return 3.14159265358979323846*std::sqrt(2*r/g);
}

// ─── Functional derivative δF/δy(x) ──────────────────────────────────────────
// F[y] = ∫ L(x,y,y') dx
// δF/δy(x) = ∂L/∂y - d/dx(∂L/∂y') (Euler-Lagrange operator)
template<typename L>
[[nodiscard]] constexpr f64 functional_derivative(L&& L_fn,
    f64 x, f64 y, f64 dy, f64 h=1e-5) noexcept
{
    f64 dL_dy=(L_fn(x,y+h,dy)-L_fn(x,y-h,dy))/(2*h);
    // d/dx[∂L/∂y'] ≈ ∂²L/∂x∂y' + ∂²L/∂y∂y'*y' + ∂²L/∂y'² * y''
    // Simplified numerical: just ∂L/∂y - finite-diff of ∂L/∂y'
    f64 dL_dyp_plus=(L_fn(x+h,y+h*dy,dy)-L_fn(x+h,y+h*dy-h,dy))/(h*2);
    f64 dL_dyp_minus=(L_fn(x-h,y-h*dy,dy)-L_fn(x-h,y-h*dy-h,dy))/(h*2);
    f64 d_dx_dL_dyp=(dL_dyp_plus-dL_dyp_minus)/(2*h);
    return dL_dy-d_dx_dL_dyp;
}

// ─── Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q ──────────────────────────
template<int N, typename H>
[[nodiscard]] constexpr std::pair<std::array<f64,N>,std::array<f64,N>>
hamilton_eqs(H&& hamiltonian,
    const std::array<f64,N>& q, const std::array<f64,N>& p,
    f64 h=1e-6) noexcept
{
    std::array<f64,N> dq{},dp{};
    for(int i=0;i<N;++i) {
        std::array<f64,N> pp=p,pm=p,qp=q,qm=q;
        pp[i]+=h;pm[i]-=h;qp[i]+=h;qm[i]-=h;
        dq[i]=(hamiltonian(q,pp)-hamiltonian(q,pm))/(2*h);  // ∂H/∂p
        dp[i]=-(hamiltonian(qp,p)-hamiltonian(qm,p))/(2*h); // -∂H/∂q
    }
    return {dq,dp};
}

// ─── Pontryagin minimum principle ─────────────────────────────────────────────
// Optimal control: min ∫L(x,u)dt s.t. ẋ=f(x,u)
// Costate: λ̇ = -∂H/∂x, where H(x,u,λ) = L(x,u) + λ·f(x,u)
template<int NS, int NC, typename HamFn>
[[nodiscard]] constexpr std::pair<std::array<f64,NS>,std::array<f64,NS>>
pontryagin_costate(HamFn&& H,
    const std::array<f64,NS>& x, const std::array<f64,NC>& u,
    const std::array<f64,NS>& lam, f64 h=1e-6) noexcept
{
    std::array<f64,NS> dx{},dlam{};
    for(int i=0;i<NS;++i) {
        std::array<f64,NS> xp=x,xm=x;
        xp[i]+=h; xm[i]-=h;
        dlam[i]=-(H(xp,u,lam)-H(xm,u,lam))/(2*h);
    }
    // Optimal control: ∂H/∂u = 0 (solved externally)
    // State dynamics from Hamiltonian ∂H/∂λ = f(x,u)
    for(int i=0;i<NS;++i) {
        std::array<f64,NS> lamp=lam,lamm=lam;
        lamp[i]+=h; lamm[i]-=h;
        dx[i]=(H(x,u,lamp)-H(x,u,lamm))/(2*h);
    }
    return {dx,dlam};
}

// ─── Geodesic shooting ────────────────────────────────────────────────────────
// Find geodesic from p to q on Riemannian manifold via shooting method
template<int N, typename MetricFn>
[[nodiscard]] constexpr std::array<f64,N> geodesic_endpoint(
    MetricFn&& g,
    const std::array<f64,N>& p,
    const std::array<f64,N>& initial_vel,
    f64 T=1.0, int steps=100) noexcept
{
    f64 dt=T/steps;
    std::array<f64,N> x=p, v=initial_vel;
    for(int k=0;k<steps;++k) {
        // Christoffel symbols via finite differences of metric
        std::array<f64,N> acc{};
        for(int lam=0;lam<N;++lam) {
            for(int mu=0;mu<N;++mu) for(int nu=0;nu<N;++nu) {
                // Approximate Γ^λ_{μν} via metric derivatives
                std::array<f64,N> xmu_p=x,xmu_m=x,xnu_p=x,xnu_m=x,xlam_p=x,xlam_m=x;
                xmu_p[mu]+=1e-4;xmu_m[mu]-=1e-4;
                xnu_p[nu]+=1e-4;xnu_m[nu]-=1e-4;
                xlam_p[lam]+=1e-4;xlam_m[lam]-=1e-4;
                auto gp=g(xmu_p),gm=g(xmu_m);
                f64 dg_mu_nulam=(gp[nu][lam]-gm[nu][lam])/2e-4; // ∂_μ g_{νλ}
                gp=g(xnu_p);gm=g(xnu_m);
                f64 dg_nu_mulam=(gp[mu][lam]-gm[mu][lam])/2e-4;
                gp=g(xlam_p);gm=g(xlam_m);
                f64 dg_lam_munu=(gp[mu][nu]-gm[mu][nu])/2e-4;
                f64 Gamma=0.5*(dg_mu_nulam+dg_nu_mulam-dg_lam_munu);
                // Raise index with g^{λσ} (approximate with identity metric for simplicity)
                acc[lam]-=Gamma*v[mu]*v[nu];
            }
        }
        for(int i=0;i<N;++i){x[i]+=v[i]*dt;v[i]+=acc[i]*dt;}
    }
    return x;
}

// ─── Isoperimetric problem (constrained variational) ──────────────────────────
// Maximize area with fixed perimeter (circle is optimal)
// Isoperimetric ratio: 4π*Area/Perimeter² ≤ 1
[[nodiscard]] constexpr f64 isoperimetric_ratio(f64 area, f64 perimeter) noexcept {
    return 4*3.14159265358979323846*area/(perimeter*perimeter);
}

// ─── Action functional (Lagrangian mechanics) ─────────────────────────────────
// S[q] = ∫_{t1}^{t2} L(t,q,q̇) dt
template<typename L, int N>
[[nodiscard]] constexpr f64 action(L&& lagrangian,
    const std::array<f64,N>* q_path, // sampled path
    f64 t0, f64 t1, int steps) noexcept
{
    f64 dt=(t1-t0)/steps, S=0;
    for(int k=0;k<steps-1;++k) {
        f64 t=t0+k*dt;
        std::array<f64,N> qdot{};
        for(int i=0;i<N;++i) qdot[i]=(q_path[k+1][i]-q_path[k][i])/dt;
        S+=lagrangian(t,q_path[k],qdot)*dt;
    }
    return S;
}
} // namespace qs::math
