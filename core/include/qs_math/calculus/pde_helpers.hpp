#pragma once
// qs_math/pde_helpers.hpp — Finite difference stencils: heat, wave, Poisson PDEs (C++20)
#include "scalar_types.hpp"
#include "sparse_matrix.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <cassert>

namespace qs {

// ─── 1D uniform grid helpers ──────────────────────────────────────────────────
struct Grid1D {
    f64 a,b; u32 N; f64 h;
    Grid1D(f64 a,f64 b,u32 N):a(a),b(b),N(N),h((b-a)/(N+1)){}
    [[nodiscard]] f64 x(u32 i) const { return a+(i+1)*h; }
    [[nodiscard]] u32 size() const { return N; }
};

struct Grid2D {
    f64 ax,bx,ay,by; u32 Nx,Ny; f64 hx,hy;
    Grid2D(f64 ax,f64 bx,u32 Nx,f64 ay,f64 by,u32 Ny)
        :ax(ax),bx(bx),ay(ay),by(by),Nx(Nx),Ny(Ny),hx((bx-ax)/(Nx+1)),hy((by-ay)/(Ny+1)){}
    [[nodiscard]] f64 x(u32 i) const {return ax+(i+1)*hx;}
    [[nodiscard]] f64 y(u32 j) const {return ay+(j+1)*hy;}
    [[nodiscard]] u32 idx(u32 i,u32 j) const {return i*Ny+j;}
    [[nodiscard]] u32 size() const {return Nx*Ny;}
};

// ─── Finite difference stencils (coefficients) ───────────────────────────────

// 1D Laplacian stencil (second derivative): [-1,2,-1]/h^2
// Returns triplet list for sparse assembly
[[nodiscard]] inline std::vector<std::tuple<u32,u32,f64>> laplacian_1d_stencil(const Grid1D& g) {
    std::vector<std::tuple<u32,u32,f64>> trips;
    f64 h2=g.h*g.h;
    for(u32 i=0;i<g.N;++i) {
        trips.emplace_back(i,i,2.0/h2);
        if(i>0)    trips.emplace_back(i,i-1,-1.0/h2);
        if(i<g.N-1)trips.emplace_back(i,i+1,-1.0/h2);
    }
    return trips;
}

// 2D Laplacian (5-point stencil) on Grid2D
[[nodiscard]] inline SparseCSR<f64> laplacian_2d(const Grid2D& g) {
    u32 n=g.size();
    std::vector<std::tuple<u32,u32,f64>> trips;
    f64 hx2=g.hx*g.hx, hy2=g.hy*g.hy;
    for(u32 i=0;i<g.Nx;++i) for(u32 j=0;j<g.Ny;++j) {
        u32 k=g.idx(i,j);
        trips.emplace_back(k,k, 2/hx2+2/hy2);
        if(i>0)    trips.emplace_back(k,g.idx(i-1,j),-1/hx2);
        if(i<g.Nx-1)trips.emplace_back(k,g.idx(i+1,j),-1/hx2);
        if(j>0)    trips.emplace_back(k,g.idx(i,j-1),-1/hy2);
        if(j<g.Ny-1)trips.emplace_back(k,g.idx(i,j+1),-1/hy2);
    }
    return SparseCSR<f64>::from_triplets(n,n,std::move(trips));
}

// 4th-order 1D Laplacian stencil: [-1,16,-30,16,-1]/(12h^2)
[[nodiscard]] inline std::vector<std::tuple<u32,u32,f64>> laplacian_1d_4th(const Grid1D& g) {
    std::vector<std::tuple<u32,u32,f64>> trips;
    f64 h2=g.h*g.h;
    for(u32 i=0;i<g.N;++i) {
        trips.emplace_back(i,i,-30.0/(12*h2));
        if(i>0)    trips.emplace_back(i,i-1,16.0/(12*h2));
        if(i<g.N-1)trips.emplace_back(i,i+1,16.0/(12*h2));
        if(i>1)    trips.emplace_back(i,i-2,-1.0/(12*h2));
        if(i+2<g.N)trips.emplace_back(i,i+2,-1.0/(12*h2));
    }
    return trips;
}

// ─── Heat equation: du/dt = α Δu (explicit FTCS + implicit Crank-Nicolson) ───

// Explicit FTCS time step: u^{n+1} = u^n + α dt/dx² * Lu^n
// Stable for α dt/dx² ≤ 0.5 (CFL)
[[nodiscard]] inline std::vector<f64> heat_1d_explicit_step(
    const std::vector<f64>& u, f64 alpha, f64 dt, f64 dx,
    f64 bc_left=0, f64 bc_right=0)
{
    u32 N=u32(u.size());
    f64 r=alpha*dt/(dx*dx);
    assert(r<=0.5 && "FTCS CFL violated — reduce dt");
    std::vector<f64> u_new(N);
    u_new[0]  =(1-2*r)*u[0]+r*(bc_left+u[1]);
    u_new[N-1]=(1-2*r)*u[N-1]+r*(u[N-2]+bc_right);
    for(u32 i=1;i<N-1;++i) u_new[i]=(1-2*r)*u[i]+r*(u[i-1]+u[i+1]);
    return u_new;
}

// Implicit Crank-Nicolson for heat equation (unconditionally stable)
// Builds tridiagonal system (a,b,c) and solves via Thomas algorithm
[[nodiscard]] inline std::vector<f64> heat_1d_cn_step(
    const std::vector<f64>& u, f64 alpha, f64 dt, f64 dx,
    f64 bc_left=0, f64 bc_right=0)
{
    u32 N=u32(u.size());
    f64 r=alpha*dt/(2*dx*dx);
    // RHS
    std::vector<f64> rhs(N);
    rhs[0]  =r*bc_left+(1-2*r)*u[0]+r*u[1]+r*bc_left;
    rhs[N-1]=r*u[N-2]+(1-2*r)*u[N-1]+r*bc_right+r*bc_right;
    for(u32 i=1;i<N-1;++i) rhs[i]=r*u[i-1]+(1-2*r)*u[i]+r*u[i+1];
    // Thomas algorithm for tridiagonal: sub=-r, diag=1+2r, sup=-r
    std::vector<f64> c_(N,0),d(rhs);
    f64 diag=1+2*r, sub=-r, sup=-r;
    // Forward sweep
    c_[0]=sup/diag; d[0]/=diag;
    for(u32 i=1;i<N;++i) {
        f64 m=diag-sub*c_[i-1];
        c_[i]=sup/m; d[i]=(d[i]-sub*d[i-1])/m;
    }
    // Back substitution
    std::vector<f64> u_new(N);
    u_new[N-1]=d[N-1];
    for(i32 i=i32(N)-2;i>=0;--i) u_new[u32(i)]=d[u32(i)]-c_[u32(i)]*u_new[u32(i)+1];
    return u_new;
}

// ─── Wave equation: d²u/dt² = c² Δu (Leapfrog) ───────────────────────────────
struct WaveState1D { std::vector<f64> u, u_prev; };

[[nodiscard]] inline WaveState1D wave_1d_leapfrog_step(
    const WaveState1D& s, f64 c, f64 dt, f64 dx,
    f64 bc_left=0, f64 bc_right=0)
{
    u32 N=u32(s.u.size());
    f64 r2=(c*dt/dx)*(c*dt/dx);
    assert(r2<=1.0 && "Wave CFL violated");
    WaveState1D ns;
    ns.u.resize(N); ns.u_prev=s.u;
    ns.u[0]  =bc_left;
    ns.u[N-1]=bc_right;
    for(u32 i=1;i<N-1;++i)
        ns.u[i]=2*s.u[i]-s.u_prev[i]+r2*(s.u[i-1]-2*s.u[i]+s.u[i+1]);
    return ns;
}

// ─── Poisson equation: Δu = f (steady-state, 2D) ─────────────────────────────
// Returns the solution vector on interior grid nodes
[[nodiscard]] inline std::vector<f64> poisson_2d(const Grid2D& g,
    const std::function<f64(f64,f64)>& rhs_func,
    const std::function<f64(f64,f64)>& bc_func)
{
    u32 n=g.size();
    auto L=laplacian_2d(g);
    // Build RHS incorporating boundary conditions
    std::vector<f64> b(n,0);
    f64 hx2=g.hx*g.hx, hy2=g.hy*g.hy;
    for(u32 i=0;i<g.Nx;++i) for(u32 j=0;j<g.Ny;++j) {
        u32 k=g.idx(i,j);
        b[k]=rhs_func(g.x(i),g.y(j));
        // Add boundary contributions
        if(i==0)      b[k]+=bc_func(g.ax,g.y(j))/hx2;
        if(i==g.Nx-1) b[k]+=bc_func(g.bx,g.y(j))/hx2;
        if(j==0)      b[k]+=bc_func(g.x(i),g.ay)/hy2;
        if(j==g.Ny-1) b[k]+=bc_func(g.x(i),g.by)/hy2;
    }
    std::vector<f64> u(n,0);
    cg_solve(L,b,u,5000,1e-10);
    return u;
}

// ─── Advection equation: du/dt + c du/dx = 0 (upwind scheme) ─────────────────
[[nodiscard]] inline std::vector<f64> advection_1d_upwind_step(
    const std::vector<f64>& u, f64 c, f64 dt, f64 dx)
{
    u32 N=u32(u.size()); f64 r=c*dt/dx;
    assert(std::abs(r)<=1.0 && "Advection CFL violated");
    std::vector<f64> u_new(N);
    for(u32 i=0;i<N;++i) {
        u32 im1=(i+N-1)%N; // periodic BC
        u32 ip1=(i+1)%N;
        u_new[i]=c>0?u[i]-r*(u[i]-u[im1]):u[i]-r*(u[ip1]-u[i]);
    }
    return u_new;
}

// WENO5 stencil weights for advection (5th order essentially non-oscillatory)
[[nodiscard]] inline std::vector<f64> advection_1d_weno5_step(
    const std::vector<f64>& u, f64 c, f64 dt, f64 dx)
{
    u32 N=u32(u.size()); f64 r=c*dt/dx;
    auto mod=[&](i32 i)->u32{return u32((i%i32(N)+i32(N))%i32(N));};
    auto w=[&](f64 a)->f64{return 1.0/(a*a+1e-40);}; // WENO weight
    std::vector<f64> u_new(N);
    for(u32 i=0;i<N;++i) {
        // Get 5-point stencil
        f64 vm2=u[mod(i32(i)-2)],vm1=u[mod(i32(i)-1)],v0=u[i];
        f64 vp1=u[mod(i32(i)+1)],vp2=u[mod(i32(i)+2)];
        // 3 candidate stencils
        f64 q0=( vm2-2*vm1+v0)/3,q1=(-vm1+2*v0+vp1)/6,q2=(v0+2*vp1-vp2)/3;
        f64 IS0=(13*(vm2-2*vm1+v0)*(vm2-2*vm1+v0)+(vm2-4*vm1+3*v0)*(vm2-4*vm1+3*v0))/12;
        f64 IS1=(13*(vm1-2*v0+vp1)*(vm1-2*v0+vp1)+(vm1-vp1)*(vm1-vp1))/12;
        f64 IS2=(13*(v0-2*vp1+vp2)*(v0-2*vp1+vp2)+(3*v0-4*vp1+vp2)*(3*v0-4*vp1+vp2))/12;
        f64 w0=w(IS0+1e-6),w1=w(IS1+1e-6),w2=w(IS2+1e-6);
        f64 wsum=w0+w1+w2;
        f64 flux=(w0/wsum)*q0+(w1/wsum)*q1+(w2/wsum)*q2;
        // Similar for left-biased flux (simplified: use same here)
        f64 dflux=c*(flux-((w0/wsum)*u[mod(i32(i)-1)]+(w1/wsum)*v0+(w2/wsum)*vp1));
        u_new[i]=u[i]-r*dflux;
    }
    return u_new;
}

// ─── Multigrid V-cycle helper (2-level, 1D) ───────────────────────────────────
[[nodiscard]] inline std::vector<f64> multigrid_vcycle_1d(
    const std::vector<f64>& rhs, u32 N, f64 h,
    u32 pre_smooth=3, u32 post_smooth=3)
{
    // Gauss-Seidel smoother on [-1,2,-1]/h² system
    auto smooth=[&](std::vector<f64> u, const std::vector<f64>& f_, u32 iters)->std::vector<f64>{
        f64 h2=h*h;
        for(u32 it=0;it<iters;++it)
            for(u32 i=0;i<N;++i) {
                f64 ul=i>0?u[i-1]:0, ur=i<N-1?u[i+1]:0;
                u[i]=(f_[i]*h2+ul+ur)*0.5;
            }
        return u;
    };
    auto residual=[&](const std::vector<f64>& u,const std::vector<f64>& f_)->std::vector<f64>{
        std::vector<f64> r(N);
        f64 h2=h*h;
        for(u32 i=0;i<N;++i){f64 ul=i>0?u[i-1]:0,ur=i<N-1?u[i+1]:0;r[i]=f_[i]-(2*u[i]-ul-ur)/h2;}
        return r;
    };
    std::vector<f64> u(N,0);
    u=smooth(u,rhs,pre_smooth);
    auto r=residual(u,rhs);
    // Restrict to coarse (injection)
    u32 Nc=N/2; if(Nc<2) return u;
    std::vector<f64> rc(Nc);
    for(u32 i=0;i<Nc;++i) rc[i]=r[2*i+1];
    // Solve coarse (just smooth here)
    std::vector<f64> ec(Nc,0);
    ec=smooth(ec,rc,20);
    // Prolongate
    std::vector<f64> ef(N,0);
    for(u32 i=0;i<Nc;++i){ef[2*i]+=0.5*ec[i];ef[2*i+1]+=ec[i];if(2*i+2<N)ef[2*i+2]+=0.5*ec[i];}
    for(u32 i=0;i<N;++i) u[i]+=ef[i];
    u=smooth(u,rhs,post_smooth);
    return u;
}

} // namespace qs
