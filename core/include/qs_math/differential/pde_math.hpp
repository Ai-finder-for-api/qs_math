#pragma once
// qs_math/differential/pde_math.hpp — Heat, wave, Laplace, Navier-Stokes FD stencils
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Finite difference stencils ───────────────────────────────────────────────
// 1D first derivative (central, 2nd order): (f_{i+1}-f_{i-1})/(2h)
[[nodiscard]] constexpr f64 fd1_central(f64 fm1, f64 fp1, f64 h) noexcept {
    return (fp1-fm1)/(2*h);
}
// 1D first derivative (forward, 1st order)
[[nodiscard]] constexpr f64 fd1_forward(f64 f0, f64 fp1, f64 h) noexcept {
    return (fp1-f0)/h;
}
// 1D second derivative (central, 2nd order): (f_{i+1}-2f_i+f_{i-1})/h²
[[nodiscard]] constexpr f64 fd2_central(f64 fm1, f64 f0, f64 fp1, f64 h) noexcept {
    return (fp1-2*f0+fm1)/(h*h);
}
// 4th order central: (-f_{i+2}+16f_{i+1}-30f_i+16f_{i-1}-f_{i-2})/(12h²)
[[nodiscard]] constexpr f64 fd2_4th(f64 fm2,f64 fm1,f64 f0,f64 fp1,f64 fp2,f64 h) noexcept {
    return (-fp2+16*fp1-30*f0+16*fm1-fm2)/(12*h*h);
}
// Upwind: first derivative (upwind biased for advection)
[[nodiscard]] constexpr f64 fd1_upwind(f64 fm1, f64 f0, f64 fp1, f64 u, f64 h) noexcept {
    return u>0?(f0-fm1)/h:(fp1-f0)/h;
}

// ─── 1D Heat equation: ∂u/∂t = α ∂²u/∂x² ─────────────────────────────────────
template<int N>
constexpr void heat_explicit(std::array<f64,N>& u, f64 alpha, f64 dt, f64 dx) noexcept {
    f64 r=alpha*dt/(dx*dx);
    std::array<f64,N> un=u;
    for(int i=1;i<N-1;++i)
        un[i]=u[i]+r*(u[i+1]-2*u[i]+u[i-1]);
    u=un; // boundary conditions held fixed
}

// Crank-Nicolson (implicit, unconditionally stable) — tridiagonal solve
template<int N>
constexpr void heat_crank_nicolson(std::array<f64,N>& u, f64 alpha, f64 dt, f64 dx) noexcept {
    f64 r=alpha*dt/(2*dx*dx);
    // RHS: (1-r)*u[i] + r/2*(u[i+1]+u[i-1]) effectively b[i]
    std::array<f64,N> b{}, c_lower{}, c_upper{}, diag{};
    for(int i=1;i<N-1;++i) {
        b[i]=r*u[i-1]+(1-2*r)*u[i]+r*u[i+1];
        diag[i]=1+2*r; c_lower[i]=-r; c_upper[i]=-r;
    }
    // Thomas algorithm
    for(int i=2;i<N-1;++i){
        f64 m=c_lower[i]/diag[i-1];
        diag[i]-=m*c_upper[i-1];
        b[i]-=m*b[i-1];
    }
    u[N-2]=b[N-2]/diag[N-2];
    for(int i=N-3;i>=1;--i) u[i]=(b[i]-c_upper[i]*u[i+1])/diag[i];
}

// ─── 1D Wave equation: ∂²u/∂t² = c² ∂²u/∂x² ─────────────────────────────────
template<int N>
constexpr void wave_explicit(std::array<f64,N>& u, std::array<f64,N>& u_prev,
                              f64 c, f64 dt, f64 dx) noexcept {
    f64 r=c*dt/dx; f64 r2=r*r;
    std::array<f64,N> u_next=u;
    for(int i=1;i<N-1;++i)
        u_next[i]=2*u[i]-u_prev[i]+r2*(u[i+1]-2*u[i]+u[i-1]);
    u_prev=u; u=u_next;
}

// ─── 2D Laplace equation (Gauss-Seidel relaxation) ────────────────────────────
template<int NX, int NY>
constexpr f64 laplace_sor(std::array<std::array<f64,NY>,NX>& u,
                            const std::array<std::array<bool,NY>,NX>& fixed,
                            f64 omega=1.5, int iters=100) noexcept
{
    f64 residual=0;
    for(int iter=0;iter<iters;++iter) {
        residual=0;
        for(int i=1;i<NX-1;++i) for(int j=1;j<NY-1;++j) {
            if(fixed[i][j]) continue;
            f64 new_u=(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1])*0.25;
            f64 delta=omega*(new_u-u[i][j]);
            residual+=delta*delta;
            u[i][j]+=delta;
        }
        if(residual<1e-20) break;
    }
    return std::sqrt(residual);
}

// ─── 2D Poisson: ∇²u = f ─────────────────────────────────────────────────────
template<int NX, int NY>
constexpr void poisson_jacobi(std::array<std::array<f64,NY>,NX>& u,
                               const std::array<std::array<f64,NY>,NX>& f,
                               f64 h, int iters=100) noexcept
{
    std::array<std::array<f64,NY>,NX> un{};
    for(int iter=0;iter<iters;++iter) {
        for(int i=1;i<NX-1;++i) for(int j=1;j<NY-1;++j)
            un[i][j]=(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]-h*h*f[i][j])*0.25;
        for(int i=1;i<NX-1;++i) for(int j=1;j<NY-1;++j) u[i][j]=un[i][j];
    }
}

// ─── Level set equation: ∂φ/∂t + v·∇φ = 0 ────────────────────────────────────
template<int NX, int NY>
constexpr void level_set_advect(std::array<std::array<f64,NY>,NX>& phi,
                                  const std::array<std::array<f64,NY>,NX>& vx,
                                  const std::array<std::array<f64,NY>,NX>& vy,
                                  f64 dt, f64 dx, f64 dy) noexcept
{
    std::array<std::array<f64,NY>,NX> phin=phi;
    for(int i=1;i<NX-1;++i) for(int j=1;j<NY-1;++j) {
        f64 dphi_dx=fd1_upwind(phi[i-1][j],phi[i][j],phi[i+1][j],vx[i][j],dx);
        f64 dphi_dy=fd1_upwind(phi[i][j-1],phi[i][j],phi[i][j+1],vy[i][j],dy);
        phin[i][j]=phi[i][j]-dt*(vx[i][j]*dphi_dx+vy[i][j]*dphi_dy);
    }
    phi=phin;
}

// Reinitialization: ∂φ/∂τ = sign(φ₀)(1-|∇φ|)
template<int NX, int NY>
constexpr void level_set_reinit(std::array<std::array<f64,NY>,NX>& phi,
                                  f64 dt, f64 dx, int iters=5) noexcept
{
    for(int iter=0;iter<iters;++iter) {
        auto phin=phi;
        for(int i=1;i<NX-1;++i) for(int j=1;j<NY-1;++j) {
            f64 dpx=(phi[i+1][j]-phi[i][j])/dx,dmx=(phi[i][j]-phi[i-1][j])/dx;
            f64 dpy=(phi[i][j+1]-phi[i][j])/dx,dmy=(phi[i][j]-phi[i][j-1])/dx;
            f64 grad_mag;
            if(phi[i][j]>0) {
                f64 ax=std::max(dpx,0.0),bx=std::min(dmx,0.0);
                f64 ay=std::max(dpy,0.0),by=std::min(dmy,0.0);
                grad_mag=std::sqrt(std::max(ax*ax,bx*bx)+std::max(ay*ay,by*by));
            } else {
                f64 ax=std::min(dpx,0.0),bx=std::max(dmx,0.0);
                f64 ay=std::min(dpy,0.0),by=std::max(dmy,0.0);
                grad_mag=std::sqrt(std::max(ax*ax,bx*bx)+std::max(ay*ay,by*by));
            }
            f64 s=phi[i][j]/std::sqrt(phi[i][j]*phi[i][j]+dx*dx);
            phin[i][j]=phi[i][j]-dt*s*(grad_mag-1);
        }
        phi=phin;
    }
}

// ─── Allen-Cahn phase field equation: ∂φ/∂t = ε²∇²φ + φ - φ³ ────────────────
template<int NX, int NY>
constexpr void allen_cahn_step(std::array<std::array<f64,NY>,NX>& phi,
                                 f64 eps, f64 dt, f64 h) noexcept
{
    auto phin=phi;
    for(int i=1;i<NX-1;++i) for(int j=1;j<NY-1;++j) {
        f64 lap=(phi[i+1][j]+phi[i-1][j]+phi[i][j+1]+phi[i][j-1]-4*phi[i][j])/(h*h);
        f64 f=phi[i][j]-phi[i][j]*phi[i][j]*phi[i][j];
        phin[i][j]=phi[i][j]+dt*(eps*eps*lap+f);
    }
    phi=phin;
}

// ─── Characteristic method for first-order PDEs ───────────────────────────────
// Solve ∂u/∂t + c∂u/∂x = 0 via method of characteristics
[[nodiscard]] constexpr f64 characteristic_interp(
    const f64* u, int N, f64 x, f64 dx) noexcept
{
    f64 xi=x/dx;
    int i=int(xi); if(i<0)i=0;if(i>=N-1)i=N-2;
    f64 t=xi-i;
    return (1-t)*u[i]+t*u[i+1];
}
template<int N>
constexpr void advect_characteristics(std::array<f64,N>& u, f64 c, f64 dt, f64 dx) noexcept {
    std::array<f64,N> un{};
    for(int i=0;i<N;++i) {
        f64 x=i*dx-c*dt;
        int j=int(x/dx); j=(j%N+N)%N; // periodic
        f64 t=x/dx-int(x/dx);
        un[i]=(1-t)*u[j]+t*u[(j+1)%N];
    }
    u=un;
}
} // namespace qs::math
