#pragma once
// qs_math/fluid_helpers.hpp — SPH kernels, pressure solve, viscosity, Navier-Stokes (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include <vector>
#include <cmath>
#include <numbers>
#include <cassert>
#include <functional>

namespace qs {

// ─── SPH Kernel functions ─────────────────────────────────────────────────────
// All kernels: W(r,h) — smoothing length h, radius r=|r_ij|

// Poly6 kernel (Müller 2003) — density estimation
[[nodiscard]] inline f32 kernel_poly6(f32 r, f32 h) {
    if(r>h||r<0) return 0;
    f32 h2=h*h, r2=r*r;
    f32 coeff=315.f/(64.f*f32(std::numbers::pi)*h2*h2*h2*h2*h);
    f32 q=h2-r2;
    return coeff*q*q*q;
}
[[nodiscard]] inline Vec3f kernel_poly6_grad(Vec3f r_vec, f32 h) {
    f32 r=length(r_vec);
    if(r>h||r<1e-7f) return {};
    f32 h2=h*h, r2=r*r;
    f32 coeff=-945.f/(32.f*f32(std::numbers::pi)*h2*h2*h2*h2*h);
    return r_vec*(coeff*(h2-r2)*(h2-r2));
}

// Spiky kernel (Desbrun 1996) — pressure force gradient (non-zero at r=0)
[[nodiscard]] inline f32 kernel_spiky(f32 r, f32 h) {
    if(r>h||r<0) return 0;
    f32 q=h-r;
    return 15.f/(f32(std::numbers::pi)*h*h*h*h*h*h)*q*q*q;
}
[[nodiscard]] inline Vec3f kernel_spiky_grad(Vec3f r_vec, f32 h) {
    f32 r=length(r_vec);
    if(r>h||r<1e-7f) return {};
    f32 q=h-r;
    f32 coeff=-45.f/(f32(std::numbers::pi)*h*h*h*h*h*h)*q*q/r;
    return r_vec*coeff;
}

// Viscosity kernel (Müller 2003) — Laplacian for viscosity
[[nodiscard]] inline f32 kernel_viscosity_laplacian(f32 r, f32 h) {
    if(r>h||r<0) return 0;
    return 45.f/(f32(std::numbers::pi)*h*h*h*h*h*h)*(h-r);
}

// Cubic spline kernel (Monaghan) — most accurate, used in many SPH codes
[[nodiscard]] inline f32 kernel_cubic_spline(f32 r, f32 h) {
    f32 q=r/h;
    f32 sigma=1.f/(f32(std::numbers::pi)*h*h*h); // 3D normalisation
    if(q<0) return 0;
    if(q<1.f) return sigma*(1-1.5f*q*q+0.75f*q*q*q);
    if(q<2.f) { f32 t=2-q; return sigma*0.25f*t*t*t; }
    return 0;
}
[[nodiscard]] inline Vec3f kernel_cubic_spline_grad(Vec3f r_vec, f32 h) {
    f32 r=length(r_vec);
    if(r<1e-7f) return {};
    f32 q=r/h;
    f32 sigma=1.f/(f32(std::numbers::pi)*h*h*h*h);
    f32 dW=0;
    if(q<1.f) dW=sigma*(-3*q+2.25f*q*q);
    else if(q<2.f){ f32 t=2-q; dW=-sigma*0.75f*t*t; }
    return r_vec*(dW/r);
}

// Wendland C2 kernel (compact support, no negative lobes)
[[nodiscard]] inline f32 kernel_wendland_c2(f32 r, f32 h) {
    f32 q=r/h;
    if(q>=1) return 0;
    f32 t=1-q, coeff=21.f/(2.f*f32(std::numbers::pi)*h*h*h);
    return coeff*t*t*t*t*(1+4*q);
}

// ─── SPH particle ─────────────────────────────────────────────────────────────
struct SPHParticle {
    Vec3f pos{}, vel{}, acc{};
    f32   rho{0};    // density
    f32   pressure{0};
    f32   mass{1.f};
};

// ─── SPH density estimation ───────────────────────────────────────────────────
inline void sph_compute_density(std::vector<SPHParticle>& particles, f32 h) {
    u32 n=u32(particles.size());
    for(u32 i=0;i<n;++i) {
        particles[i].rho=0;
        for(u32 j=0;j<n;++j) {
            f32 r=length(particles[i].pos-particles[j].pos);
            particles[i].rho+=particles[j].mass*kernel_poly6(r,h);
        }
    }
}

// ─── Equation of state: pressure from density ─────────────────────────────────
// Tait equation: P = k * ((ρ/ρ0)^γ - 1)
[[nodiscard]] inline f32 tait_pressure(f32 rho, f32 rho0=1000.f, f32 k=200.f, f32 gamma=7.f) {
    return k*(std::pow(rho/rho0,gamma)-1.f);
}
// Ideal gas: P = k * ρ
[[nodiscard]] inline f32 ideal_gas_pressure(f32 rho, f32 k=1.f) { return k*rho; }

// ─── SPH forces ────────────────────────────────────────────────────────────────
struct SPHConfig {
    f32 h{0.1f};        // smoothing length
    f32 rho0{1000.f};   // rest density
    f32 k_pressure{200.f};
    f32 mu_viscosity{0.1f};
    f32 gamma{7.f};
    Vec3f gravity{0,-9.81f,0};
};

inline void sph_compute_forces(std::vector<SPHParticle>& p, const SPHConfig& cfg) {
    u32 n=u32(p.size());
    for(u32 i=0;i<n;++i) {
        p[i].pressure=tait_pressure(p[i].rho,cfg.rho0,cfg.k_pressure,cfg.gamma);
        p[i].acc=cfg.gravity;
    }
    for(u32 i=0;i<n;++i) {
        Vec3f f_pressure{}, f_viscosity{};
        for(u32 j=0;j<n;++j) {
            if(i==j) continue;
            Vec3f rij=p[i].pos-p[j].pos;
            f32   r=length(rij);
            if(r>=cfg.h||r<1e-7f) continue;
            // Pressure force: -m_j (P_i/ρ_i² + P_j/ρ_j²) ∇W
            f32 pterm=p[i].pressure/(p[i].rho*p[i].rho)+p[j].pressure/(p[j].rho*p[j].rho);
            Vec3f grad=kernel_spiky_grad(rij,cfg.h);
            f_pressure=f_pressure-grad*(p[j].mass*pterm);
            // Viscosity: μ m_j (v_j-v_i)/ρ_j * ∇²W
            f32 lap=kernel_viscosity_laplacian(r,cfg.h);
            f_viscosity=f_viscosity+(p[j].vel-p[i].vel)*(cfg.mu_viscosity*p[j].mass*lap/p[j].rho);
        }
        p[i].acc=p[i].acc+f_pressure*(1.f/p[i].rho)+f_viscosity*(1.f/p[i].rho);
    }
}

// XSPH velocity correction (smoother particle paths)
inline void sph_xsph_correction(std::vector<SPHParticle>& p, f32 h, f32 eps=0.5f) {
    u32 n=u32(p.size());
    std::vector<Vec3f> dx(n,{});
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) {
        if(i==j) continue;
        f32 r=length(p[i].pos-p[j].pos);
        f32 W=kernel_poly6(r,h);
        dx[i]=dx[i]+(p[j].vel-p[i].vel)*(eps*p[j].mass*W/(0.5f*(p[i].rho+p[j].rho)));
    }
    for(u32 i=0;i<n;++i) p[i].vel=p[i].vel+dx[i];
}

// ─── Navier-Stokes helpers ────────────────────────────────────────────────────

// Vorticity ω = ∇×u (finite difference on grid)
struct NSGrid {
    u32 nx,ny,nz; f32 dx;
    std::vector<Vec3f> vel;
    [[nodiscard]] u32 idx(u32 i,u32 j,u32 k) const { return i*ny*nz+j*nz+k; }
    [[nodiscard]] Vec3f& at(u32 i,u32 j,u32 k) { return vel[idx(i,j,k)]; }
    [[nodiscard]] Vec3f  at(u32 i,u32 j,u32 k) const { return vel[idx(i,j,k)]; }
};

[[nodiscard]] inline Vec3f vorticity(const NSGrid& g, u32 i,u32 j,u32 k) {
    if(i<1||j<1||k<1||i+1>=g.nx||j+1>=g.ny||k+1>=g.nz) return {};
    f32 inv2dx=0.5f/g.dx;
    Vec3f wx{ (g.at(i,j+1,k).z-g.at(i,j-1,k).z)*inv2dx-(g.at(i,j,k+1).y-g.at(i,j,k-1).y)*inv2dx,
              (g.at(i,j,k+1).x-g.at(i,j,k-1).x)*inv2dx-(g.at(i+1,j,k).z-g.at(i-1,j,k).z)*inv2dx,
              (g.at(i+1,j,k).y-g.at(i-1,j,k).y)*inv2dx-(g.at(i,j+1,k).x-g.at(i,j-1,k).x)*inv2dx };
    return wx;
}

// Divergence ∇·u
[[nodiscard]] inline f32 divergence_ns(const NSGrid& g, u32 i,u32 j,u32 k) {
    if(i<1||j<1||k<1||i+1>=g.nx||j+1>=g.ny||k+1>=g.nz) return 0;
    f32 inv2dx=0.5f/g.dx;
    return (g.at(i+1,j,k).x-g.at(i-1,j,k).x)*inv2dx
          +(g.at(i,j+1,k).y-g.at(i,j-1,k).y)*inv2dx
          +(g.at(i,j,k+1).z-g.at(i,j,k-1).z)*inv2dx;
}

// Pressure projection (make velocity divergence-free) — Jacobi iteration
inline void pressure_project(NSGrid& vel_grid, std::vector<f32>& pressure, u32 iters=50) {
    u32 nx=vel_grid.nx,ny=vel_grid.ny,nz=vel_grid.nz;
    f32 dx=vel_grid.dx;
    pressure.assign(nx*ny*nz,0);
    auto pidx=[&](u32 i,u32 j,u32 k){return i*ny*nz+j*nz+k;};
    // Build divergence RHS
    std::vector<f32> div(nx*ny*nz,0);
    for(u32 i=1;i+1<nx;++i) for(u32 j=1;j+1<ny;++j) for(u32 k=1;k+1<nz;++k)
        div[pidx(i,j,k)]=divergence_ns(vel_grid,i,j,k)/dx;
    // Jacobi solve ∇²p = div
    std::vector<f32> p_new(pressure.size(),0);
    for(u32 it=0;it<iters;++it) {
        for(u32 i=1;i+1<nx;++i) for(u32 j=1;j+1<ny;++j) for(u32 k=1;k+1<nz;++k) {
            p_new[pidx(i,j,k)]=(pressure[pidx(i+1,j,k)]+pressure[pidx(i-1,j,k)]
                               +pressure[pidx(i,j+1,k)]+pressure[pidx(i,j-1,k)]
                               +pressure[pidx(i,j,k+1)]+pressure[pidx(i,j,k-1)]
                               -dx*dx*div[pidx(i,j,k)])/6.f;
        }
        pressure=p_new;
    }
    // Subtract pressure gradient from velocity
    f32 inv2dx=0.5f/dx;
    for(u32 i=1;i+1<nx;++i) for(u32 j=1;j+1<ny;++j) for(u32 k=1;k+1<nz;++k) {
        vel_grid.at(i,j,k).x-=(pressure[pidx(i+1,j,k)]-pressure[pidx(i-1,j,k)])*inv2dx;
        vel_grid.at(i,j,k).y-=(pressure[pidx(i,j+1,k)]-pressure[pidx(i,j-1,k)])*inv2dx;
        vel_grid.at(i,j,k).z-=(pressure[pidx(i,j,k+1)]-pressure[pidx(i,j,k-1)])*inv2dx;
    }
}

// ─── Lattice Boltzmann method D2Q9 (2D incompressible flow) ──────────────────
struct LBM_D2Q9 {
    u32 nx,ny;
    static constexpr u32 Q=9;
    std::vector<f32> f, f_eq;
    f32 omega{1.f}; // relaxation: omega = 1/tau, tau = viscosity*3 + 0.5

    // D2Q9 weights and velocities
    static constexpr f32 W[9]={4.f/9,1.f/9,1.f/9,1.f/9,1.f/9,1.f/36,1.f/36,1.f/36,1.f/36};
    static constexpr i32 CX[9]={0,1,0,-1,0,1,-1,-1,1};
    static constexpr i32 CY[9]={0,0,1,0,-1,1,1,-1,-1};

    LBM_D2Q9(u32 nx,u32 ny,f32 viscosity):nx(nx),ny(ny),f(nx*ny*Q,0),f_eq(nx*ny*Q,0),
        omega(1.f/(3.f*viscosity+0.5f)){}

    [[nodiscard]] u32 idx(u32 x,u32 y,u32 q) const { return (y*nx+x)*Q+q; }

    void init_equilibrium(f32 rho0=1.f) {
        for(u32 y=0;y<ny;++y) for(u32 x=0;x<nx;++x)
            for(u32 q=0;q<Q;++q) f[idx(x,y,q)]=W[q]*rho0;
    }

    [[nodiscard]] f32 density(u32 x,u32 y) const {
        f32 rho=0; for(u32 q=0;q<Q;++q) rho+=f[idx(x,y,q)]; return rho;
    }
    [[nodiscard]] Vec2f velocity(u32 x,u32 y) const {
        f32 rho=density(x,y); Vec2f v{};
        for(u32 q=0;q<Q;++q){ f32 fq=f[idx(x,y,q)]; v.x+=CX[q]*fq; v.y+=CY[q]*fq; }
        return v*(1.f/(rho+1e-10f));
    }

    void compute_equilibrium() {
        for(u32 y=0;y<ny;++y) for(u32 x=0;x<nx;++x) {
            f32 rho=density(x,y); Vec2f u=velocity(x,y);
            f32 u2=u.x*u.x+u.y*u.y;
            for(u32 q=0;q<Q;++q) {
                f32 cu=CX[q]*u.x+CY[q]*u.y;
                f_eq[idx(x,y,q)]=W[q]*rho*(1+3*cu+4.5f*cu*cu-1.5f*u2);
            }
        }
    }

    void collide() {
        compute_equilibrium();
        u32 n=nx*ny*Q;
        for(u32 i=0;i<n;++i) f[i]+=omega*(f_eq[i]-f[i]);
    }

    void stream() {
        std::vector<f32> f_new(f.size(),0);
        for(u32 y=0;y<ny;++y) for(u32 x=0;x<nx;++x) for(u32 q=0;q<Q;++q) {
            i32 xn=i32(x)+CX[q], yn=i32(y)+CY[q];
            if(xn<0||xn>=(i32)nx||yn<0||yn>=(i32)ny) continue;
            f_new[idx(u32(xn),u32(yn),q)]=f[idx(x,y,q)];
        }
        f=f_new;
    }

    void step() { collide(); stream(); }

    // No-slip bounce-back on solid nodes
    void bounce_back(const std::vector<bool>& solid) {
        static constexpr u32 OPP[9]={0,3,4,1,2,7,8,5,6};
        for(u32 y=0;y<ny;++y) for(u32 x=0;x<nx;++x) if(solid[y*nx+x])
            for(u32 q=0;q<Q;++q) f[idx(x,y,q)]=f[idx(x,y,OPP[q])];
    }
};

// ─── Fluid surface tension (curvature-based) ──────────────────────────────────
// CSF model: F_surface = σ κ n̂ δ_s where κ = -∇·(∇φ/|∇φ|)
[[nodiscard]] inline f32 curvature_from_levelset(const std::vector<f32>& phi,
    u32 nx, u32 ny, u32 i, u32 j, f32 dx)
{
    auto p=[&](i32 x,i32 y)->f32{
        x=std::clamp(x,0,i32(nx)-1); y=std::clamp(y,0,i32(ny)-1);
        return phi[u32(y)*nx+u32(x)];
    };
    f32 phi_x=(p(i32(i)+1,i32(j))-p(i32(i)-1,i32(j)))/(2*dx);
    f32 phi_y=(p(i32(i),i32(j)+1)-p(i32(i),i32(j)-1))/(2*dx);
    f32 phi_xx=(p(i32(i)+1,i32(j))-2*p(i32(i),i32(j))+p(i32(i)-1,i32(j)))/(dx*dx);
    f32 phi_yy=(p(i32(i),i32(j)+1)-2*p(i32(i),i32(j))+p(i32(i),i32(j)-1))/(dx*dx);
    f32 phi_xy=(p(i32(i)+1,i32(j)+1)-p(i32(i)+1,i32(j)-1)-p(i32(i)-1,i32(j)+1)+p(i32(i)-1,i32(j)-1))/(4*dx*dx);
    f32 mag2=phi_x*phi_x+phi_y*phi_y+1e-10f;
    return -(phi_xx*phi_y*phi_y-2*phi_xy*phi_x*phi_y+phi_yy*phi_x*phi_x)/(mag2*std::sqrt(mag2));
}

} // namespace qs
