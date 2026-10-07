#pragma once
// qs_math/physics/fluid_math.hpp — Fluid simulation math: SPH kernels,
//   pressure/viscosity forces, vorticity confinement, grid advection,
//   divergence-free projection, Navier-Stokes helpers (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <vector>
#include <cmath>
#include <functional>

namespace qs::math::physics {

// ─── SPH smoothing kernels ────────────────────────────────────────────────
// All kernels take (r, h) where r = distance, h = smoothing length.
// Poly6 (density estimation)
[[nodiscard]] inline f32 sph_poly6(f32 r, f32 h) noexcept {
    if(r>h||r<0.f) return 0.f;
    f32 h2=h*h, r2=r*r, d=h2-r2;
    return 315.f/(64.f*f32(PI)*h2*h2*h2*h) * d*d*d;
}

// Gradient of poly6
[[nodiscard]] inline Vec3f sph_poly6_grad(Vec3f rvec, f32 r, f32 h) noexcept {
    if(r>h||r<1e-6f) return {0,0,0};
    f32 h2=h*h,r2=r*r,d=h2-r2;
    f32 coeff=-945.f/(32.f*f32(PI)*h2*h2*h2*h)*d*d;
    return rvec*coeff;
}

// Spiky (pressure gradient — non-zero at origin)
[[nodiscard]] inline f32 sph_spiky(f32 r, f32 h) noexcept {
    if(r>h||r<0.f) return 0.f;
    f32 d=h-r;
    return 15.f/(f32(PI)*h*h*h*h*h*h) * d*d*d;
}

[[nodiscard]] inline Vec3f sph_spiky_grad(Vec3f rvec, f32 r, f32 h) noexcept {
    if(r>h||r<1e-6f) return {0,0,0};
    f32 d=h-r;
    f32 coeff=-45.f/(f32(PI)*h*h*h*h*h*h)*d*d/r;
    return rvec*coeff;
}

// Viscosity (Laplacian — non-negative Laplacian)
[[nodiscard]] inline f32 sph_viscosity_lap(f32 r, f32 h) noexcept {
    if(r>h||r<0.f) return 0.f;
    return 45.f/(f32(PI)*h*h*h*h*h*h) * (h-r);
}

// Wendland C2 (compact, positive-definite)
[[nodiscard]] inline f32 sph_wendland(f32 r, f32 h) noexcept {
    f32 q=r/h; if(q>1.f) return 0.f;
    f32 t=1.f-q;
    return 21.f/(2.f*f32(PI)*h*h*h) * t*t*t*t*(4.f*q+1.f);
}

[[nodiscard]] inline Vec3f sph_wendland_grad(Vec3f rvec, f32 r, f32 h) noexcept {
    f32 q=r/h; if(q>1.f||r<1e-6f) return {0,0,0};
    f32 t=1.f-q;
    f32 coeff=-21.f/(2.f*f32(PI)*h*h*h*h) * 20.f*q*t*t*t/r;
    return rvec*coeff;
}

// ─── SPH particle ─────────────────────────────────────────────────────────
struct SPHParticle {
    Vec3f pos, vel, force;
    f32   density, pressure, mass;
    f32   temperature = 0.f; // for thermal coupling
};

// ─── SPH density computation ──────────────────────────────────────────────
inline void compute_density(std::vector<SPHParticle>& particles, f32 h, f32 rest_density) noexcept {
    int n=int(particles.size());
    for(int i=0;i<n;++i){
        f32 rho=0.f;
        for(int j=0;j<n;++j){
            Vec3f r=particles[i].pos-particles[j].pos;
            f32 dist=r.length();
            rho+=particles[j].mass*sph_poly6(dist,h);
        }
        particles[i].density=max(rho,1e-6f);
    }
}

// ─── SPH pressure (Tait equation of state) ────────────────────────────────
// p = k*((rho/rho0)^gamma - 1),  gamma=7 for water
[[nodiscard]] inline f32 tait_pressure(f32 density, f32 rest_density,
                                        f32 k=1000.f, f32 gamma=7.f) noexcept
{
    return k*(std::pow(density/rest_density, gamma)-1.f);
}

inline void compute_pressure_tait(std::vector<SPHParticle>& p, f32 rho0, f32 k=1000.f) noexcept {
    for(auto& pi:p) pi.pressure=tait_pressure(pi.density,rho0,k);
}

// ─── SPH force accumulation ───────────────────────────────────────────────
inline void accumulate_forces(
    std::vector<SPHParticle>& particles, f32 h,
    f32 viscosity=0.1f, Vec3f gravity={0,-9.81f,0}) noexcept
{
    int n=int(particles.size());
    for(int i=0;i<n;++i) particles[i].force={0,0,0};
    for(int i=0;i<n;++i){
        auto& pi=particles[i];
        Vec3f f_press={0,0,0}, f_visc={0,0,0};
        for(int j=0;j<n;++j){
            if(i==j) continue;
            auto& pj=particles[j];
            Vec3f r=pi.pos-pj.pos;
            f32 dist=r.length();
            if(dist>h||dist<1e-6f) continue;
            // Pressure force: symmetric
            f32 pterm=(pi.pressure/(pi.density*pi.density)+pj.pressure/(pj.density*pj.density));
            f_press=f_press-sph_spiky_grad(r,dist,h)*pj.mass*pterm;
            // Viscosity
            f_visc=f_visc+(pj.vel-pi.vel)*(pj.mass/pj.density*sph_viscosity_lap(dist,h));
        }
        pi.force=f_press*pi.density + f_visc*viscosity + gravity*pi.density;
    }
}

// ─── SPH integration (symplectic Euler) ───────────────────────────────────
inline void integrate_sph(std::vector<SPHParticle>& p, f32 dt,
                           Vec3f bounds_lo={-1,-1,-1},
                           Vec3f bounds_hi={1,1,1}, f32 restitution=0.3f) noexcept
{
    for(auto& pi:p){
        Vec3f acc=pi.force/pi.density;
        pi.vel=pi.vel+acc*dt;
        pi.pos=pi.pos+pi.vel*dt;
        // Boundary collision
        auto bound=[&](f32& pos,f32& vel,f32 lo,f32 hi){
            if(pos<lo){pos=lo;vel=std::abs(vel)*restitution;}
            if(pos>hi){pos=hi;vel=-std::abs(vel)*restitution;}
        };
        bound(pi.pos.x,pi.vel.x,bounds_lo.x,bounds_hi.x);
        bound(pi.pos.y,pi.vel.y,bounds_lo.y,bounds_hi.y);
        bound(pi.pos.z,pi.vel.z,bounds_lo.z,bounds_hi.z);
    }
}

// ─── Eulerian grid fluid helpers ──────────────────────────────────────────
// MAC (Marker-and-Cell) grid: u on x-faces, v on y-faces, w on z-faces, p at centers

struct MacGrid3D {
    int nx,ny,nz;
    f32 dx;
    std::vector<f32> u,v,w,p,div;

    MacGrid3D(int nx,int ny,int nz,f32 dx)
        :nx(nx),ny(ny),nz(nz),dx(dx)
        ,u((nx+1)*ny*nz,0),v(nx*(ny+1)*nz,0),w(nx*ny*(nz+1),0)
        ,p(nx*ny*nz,0),div(nx*ny*nz,0){}

    f32& U(int i,int j,int k){ return u[i*(ny*nz)+j*nz+k]; }
    f32& V(int i,int j,int k){ return v[i*((ny+1)*nz)+j*nz+k]; }
    f32& W(int i,int j,int k){ return w[i*(ny*(nz+1))+j*(nz+1)+k]; }
    f32& P(int i,int j,int k){ return p[i*(ny*nz)+j*nz+k]; }
    f32& DIV(int i,int j,int k){ return div[i*(ny*nz)+j*nz+k]; }
};

// Compute velocity divergence into grid.div
inline void compute_divergence(MacGrid3D& g) noexcept {
    f32 inv_dx=1.f/g.dx;
    for(int i=0;i<g.nx;++i)
        for(int j=0;j<g.ny;++j)
            for(int k=0;k<g.nz;++k)
                g.DIV(i,j,k)=inv_dx*(
                    g.U(i+1,j,k)-g.U(i,j,k)+
                    g.V(i,j+1,k)-g.V(i,j,k)+
                    g.W(i,j,k+1)-g.W(i,j,k));
}

// Pressure projection (Jacobi iteration)
inline void project_pressure(MacGrid3D& g, int iters=40, f32 rho=1.f, f32 dt=0.016f) noexcept {
    f32 dx2=g.dx*g.dx, scale=rho*dx2/dt;
    std::fill(g.p.begin(),g.p.end(),0.f);
    for(int it=0;it<iters;++it){
        auto old_p=g.p;
        for(int i=1;i<g.nx-1;++i)
            for(int j=1;j<g.ny-1;++j)
                for(int k=1;k<g.nz-1;++k){
                    f32 rhs=g.DIV(i,j,k)*scale;
                    g.P(i,j,k)=(old_p[(i-1)*(g.ny*g.nz)+j*g.nz+k]+
                                 old_p[(i+1)*(g.ny*g.nz)+j*g.nz+k]+
                                 old_p[i*(g.ny*g.nz)+(j-1)*g.nz+k]+
                                 old_p[i*(g.ny*g.nz)+(j+1)*g.nz+k]+
                                 old_p[i*(g.ny*g.nz)+j*g.nz+(k-1)]+
                                 old_p[i*(g.ny*g.nz)+j*g.nz+(k+1)]-rhs)/6.f;
                }
    }
    // Subtract gradient to make velocity divergence-free
    f32 inv_dx=1.f/g.dx, inv_rho_dt=dt/rho;
    for(int i=1;i<g.nx;++i)
        for(int j=0;j<g.ny;++j)
            for(int k=0;k<g.nz;++k)
                g.U(i,j,k)-=inv_dx*inv_rho_dt*(g.P(i,j,k)-g.P(i-1,j,k));
    for(int i=0;i<g.nx;++i)
        for(int j=1;j<g.ny;++j)
            for(int k=0;k<g.nz;++k)
                g.V(i,j,k)-=inv_dx*inv_rho_dt*(g.P(i,j,k)-g.P(i,j-1,k));
    for(int i=0;i<g.nx;++i)
        for(int j=0;j<g.ny;++j)
            for(int k=1;k<g.nz;++k)
                g.W(i,j,k)-=inv_dx*inv_rho_dt*(g.P(i,j,k)-g.P(i,j,k-1));
}

// ─── Vorticity confinement ────────────────────────────────────────────────
// Compute cell-centered vorticity curl(u) from MAC grid
[[nodiscard]] inline Vec3f cell_vorticity(MacGrid3D& g, int i, int j, int k) noexcept {
    f32 inv2dx=0.5f/g.dx;
    f32 wx=(g.W(i,j+1,k)-g.W(i,j-1,k)-(g.V(i,j,k+1)-g.V(i,j,k-1)))*inv2dx;
    f32 wy=(g.U(i,j,k+1)-g.U(i,j,k-1)-(g.W(i+1,j,k)-g.W(i-1,j,k)))*inv2dx;
    f32 wz=(g.V(i+1,j,k)-g.V(i-1,j,k)-(g.U(i,j+1,k)-g.U(i,j-1,k)))*inv2dx;
    return {wx,wy,wz};
}

// ─── Navier-Stokes dimensionless numbers ──────────────────────────────────
[[nodiscard]] constexpr f32 reynolds_number(f32 density,f32 vel,f32 length,f32 viscosity) noexcept {
    return density*vel*length/viscosity;
}
[[nodiscard]] constexpr f32 mach_number(f32 vel, f32 speed_of_sound) noexcept {
    return vel/speed_of_sound;
}
[[nodiscard]] constexpr f32 froude_number(f32 vel, f32 g, f32 depth) noexcept {
    return vel/std::sqrt(g*depth);
}
[[nodiscard]] constexpr f32 strouhal_number(f32 freq, f32 length, f32 vel) noexcept {
    return freq*length/vel;
}
[[nodiscard]] constexpr f32 prandtl_number(f32 viscosity, f32 cp, f32 thermal_k) noexcept {
    return viscosity*cp/thermal_k;
}

// ─── Buoyancy force ───────────────────────────────────────────────────────
[[nodiscard]] inline Vec3f buoyancy(f32 rho_fluid, f32 rho_object, f32 volume,
                                     Vec3f g={0,-9.81f,0}) noexcept
{
    return g*((rho_fluid-rho_object)*volume);
}

// ─── Bernoulli / continuity helpers ───────────────────────────────────────
// Bernoulli pressure from velocity (incompressible)
[[nodiscard]] constexpr f32 bernoulli_pressure(f32 rho, f32 v1, f32 p1, f32 v2) noexcept {
    return p1+0.5f*rho*(v1*v1-v2*v2);
}
// Continuity: v2 given A1,A2,v1
[[nodiscard]] constexpr f32 continuity_velocity(f32 A1, f32 v1, f32 A2) noexcept {
    return A1*v1/A2;
}

// ─── Wave equation helpers ────────────────────────────────────────────────
// Dispersion: wave speed in shallow water
[[nodiscard]] inline f32 wave_speed_shallow(f32 depth, f32 g=9.81f) noexcept {
    return std::sqrt(g*depth);
}
// Deep water dispersion: c = sqrt(g*lambda/(2*pi))
[[nodiscard]] inline f32 wave_speed_deep(f32 wavelength, f32 g=9.81f) noexcept {
    return std::sqrt(g*wavelength/f32(TWO_PI));
}
// Gerstner wave displacement
[[nodiscard]] inline Vec3f gerstner_wave(Vec2f xz, f32 t,
                                          Vec2f dir, f32 wavelength, f32 amplitude, f32 steepness) noexcept
{
    dir=Vec2f{dir.x,dir.y}*(1.f/std::sqrt(dir.x*dir.x+dir.y*dir.y+1e-30f));
    f32 k=f32(TWO_PI)/wavelength, c=wave_speed_deep(wavelength);
    f32 phase=k*(dir.x*xz.x+dir.y*xz.y)-c*t;
    f32 a=steepness/k;
    return {a*dir.x*std::cos(phase), amplitude*std::sin(phase), a*dir.y*std::cos(phase)};
}

} // namespace qs::math::physics
