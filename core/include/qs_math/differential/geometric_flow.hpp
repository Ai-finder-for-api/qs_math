#pragma once
// qs_math/differential/geometric_flow.hpp — Mean curvature flow, Ricci flow, eikonal
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Mean curvature flow ──────────────────────────────────────────────────────
// ∂X/∂t = Δ_M X = H*N (evolve surface by mean curvature)
// Discrete: cotangent Laplacian on triangle mesh
template<int NV>
[[nodiscard]] constexpr std::array<std::array<f64,3>,NV> mean_curvature_flow_step(
    const std::array<std::array<f64,3>,NV>& verts,
    const std::array<f64,NV*NV>& L,  // cotangent Laplacian (flattened)
    f64 dt, int nv) noexcept
{
    auto result=verts;
    for(int i=0;i<nv;++i) {
        std::array<f64,3> Lv{};
        for(int j=0;j<nv;++j) {
            f64 w=L[i*NV+j];
            Lv[0]+=w*verts[j][0];Lv[1]+=w*verts[j][1];Lv[2]+=w*verts[j][2];
        }
        result[i][0]+=dt*Lv[0];result[i][1]+=dt*Lv[1];result[i][2]+=dt*Lv[2];
    }
    return result;
}

// ─── Curvature flow on curves (2D) ────────────────────────────────────────────
// ∂γ/∂t = κ*N (evolve curve by curvature → shrinks to point)
template<int N>
constexpr void curve_shortening_step(
    std::array<std::array<f64,2>,N>& gamma, f64 dt) noexcept
{
    auto prev=gamma;
    for(int i=0;i<N;++i) {
        int ip=(i+1)%N, im=(i-1+N)%N;
        // Discrete curvature: Laplacian of position
        f64 kx=prev[ip][0]-2*prev[i][0]+prev[im][0];
        f64 ky=prev[ip][1]-2*prev[i][1]+prev[im][1];
        gamma[i][0]+=dt*kx;
        gamma[i][1]+=dt*ky;
    }
}

// ─── Eikonal equation (fast marching method sketch) ───────────────────────────
// |∇u| = 1/speed, u = distance / time
// 2D update formula: max(u-u_x,0)² + max(u-u_y,0)² = (h/speed)²
[[nodiscard]] constexpr f64 eikonal_update(f64 u_left,f64 u_right,
                                             f64 u_down, f64 u_up,
                                             f64 h, f64 speed=1.0) noexcept
{
    f64 ux=std::min(u_left,u_right);
    f64 uy=std::min(u_down,u_up);
    f64 f=h/speed;
    // Solve (u-ux)² + (u-uy)² = f²
    if(std::abs(ux-uy)>=f) return std::min(ux,uy)+f;
    f64 s=(ux+uy+std::sqrt(2*f*f-(ux-uy)*(ux-uy)))*0.5;
    return s;
}

template<int H, int W>
constexpr void fast_marching_step(
    std::array<std::array<f64,W>,H>& u,
    const std::array<std::array<f64,W>,H>& speed,
    f64 h=1.0) noexcept
{
    auto prev=u;
    for(int y=1;y<H-1;++y) for(int x=1;x<W-1;++x) {
        f64 upd=eikonal_update(prev[y][x-1],prev[y][x+1],
                                prev[y-1][x],prev[y+1][x],h,speed[y][x]);
        u[y][x]=std::min(u[y][x],upd);
    }
}

// ─── Willmore flow (energy ∫H² dA) ───────────────────────────────────────────
// ∂X/∂t = -∇_M E_W = -(ΔH + 2H(H²-K))N (surface diffusion)
// Discrete: biharmonic + nonlinear term
[[nodiscard]] constexpr f64 willmore_energy_vertex(f64 H, f64 K, f64 area) noexcept {
    return H*H*area; // integrate H² over area element
}

// ─── Ricci flow (Thurston geometrization) ────────────────────────────────────
// ∂g_ij/∂t = -2 R_ij (evolve Riemannian metric by Ricci curvature)
// Discrete (2D): evolve conformal factor u where g = e^{2u} g_flat
template<int N>
constexpr void ricci_flow_step_2d(
    std::array<f64,N>& u,              // log conformal factor
    const std::array<f64,N>& K_gauss,  // Gaussian curvature at vertices
    f64 target_K, f64 dt) noexcept
{
    // ∂u/∂t = -(K - K_target) / area (normalized)
    for(int i=0;i<N;++i) u[i]-=dt*(K_gauss[i]-target_K);
}

// ─── Anisotropic diffusion (Perona-Malik) ─────────────────────────────────────
[[nodiscard]] constexpr f64 pm_conductance(f64 grad_mag, f64 k, bool mode=true) noexcept {
    if(mode) return std::exp(-(grad_mag/k)*(grad_mag/k)); // Gaussian
    return 1.0/(1+(grad_mag/k)*(grad_mag/k));             // Lorentzian
}

template<int H, int W>
constexpr void perona_malik_step(
    std::array<std::array<f64,W>,H>& u,
    f64 dt, f64 k, bool mode=true) noexcept
{
    auto prev=u;
    for(int y=1;y<H-1;++y) for(int x=1;x<W-1;++x) {
        // Four directional gradients
        f64 gN=prev[y-1][x]-prev[y][x];
        f64 gS=prev[y+1][x]-prev[y][x];
        f64 gE=prev[y][x+1]-prev[y][x];
        f64 gW=prev[y][x-1]-prev[y][x];
        f64 cN=pm_conductance(std::abs(gN),k,mode);
        f64 cS=pm_conductance(std::abs(gS),k,mode);
        f64 cE=pm_conductance(std::abs(gE),k,mode);
        f64 cW=pm_conductance(std::abs(gW),k,mode);
        u[y][x]+=dt*(cN*gN+cS*gS+cE*gE+cW*gW);
    }
}

// ─── Total variation denoising (ROF model) ────────────────────────────────────
// min_u ∫|∇u| + λ/2*∫(u-f)² dx
// Gradient descent step: ∂u/∂t = div(∇u/|∇u|) - λ(u-f)
template<int H, int W>
constexpr void tv_denoise_step(
    std::array<std::array<f64,W>,H>& u,
    const std::array<std::array<f64,W>,H>& f,
    f64 dt, f64 lambda, f64 eps=1e-3) noexcept
{
    auto prev=u;
    for(int y=1;y<H-1;++y) for(int x=1;x<W-1;++x) {
        // Divergence of normalized gradient (Euler-Lagrange of TV)
        f64 gx_c=(prev[y][x+1]-prev[y][x-1])*0.5;
        f64 gy_c=(prev[y+1][x]-prev[y-1][x])*0.5;
        f64 g_mag=std::sqrt(gx_c*gx_c+gy_c*gy_c+eps*eps);
        // Simplified TV regularizer: Laplacian / |∇u|
        f64 lap=prev[y+1][x]+prev[y-1][x]+prev[y][x+1]+prev[y][x-1]-4*prev[y][x];
        u[y][x]+=dt*(lap/g_mag-lambda*(prev[y][x]-f[y][x]));
    }
}

// ─── Geodesic distance on meshes (heat method sketch) ─────────────────────────
// 1. Evolve heat u_t = Δu from source s for time t
// 2. Compute ∇u, normalize to X = -∇u/|∇u|
// 3. Solve Δφ = div(X)
// Here: just document the key PDE steps
constexpr void heat_method_step1(auto& u, auto& L, f64 t) noexcept {
    // u_{t+dt} = u_t + t*L*u_t (implicit: solve (I - t*L)*u = u_source)
    // Implementation uses linear solve — deferred to caller
}
} // namespace qs::math
