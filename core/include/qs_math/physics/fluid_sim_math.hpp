#pragma once
// qs_math/fluid_sim_math.hpp — SPH, MAC grid, pressure solve, advection (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <array>

namespace qs {

// ─── SPH kernel functions ─────────────────────────────────────────────────────

// Poly6 kernel (density estimation)
[[nodiscard]] inline f32 sph_poly6(f32 r, f32 h) {
    if (r>h) return 0.f;
    f32 t=h*h-r*r;
    return (315.f/(64.f*k_pi*std::pow(h,9.f)))*t*t*t;
}

// Spiky kernel gradient magnitude (pressure)
[[nodiscard]] inline f32 sph_spiky_grad(f32 r, f32 h) {
    if (r>h||r<1e-6f) return 0.f;
    f32 t=h-r;
    return -(45.f/(k_pi*std::pow(h,6.f)))*t*t;
}

// Viscosity kernel Laplacian
[[nodiscard]] inline f32 sph_viscosity_lap(f32 r, f32 h) {
    if (r>h) return 0.f;
    return (45.f/(k_pi*std::pow(h,6.f)))*(h-r);
}

// Cubic spline kernel (Monaghan 1992)
[[nodiscard]] inline f32 sph_cubic(f32 r, f32 h) {
    f32 q=r/h;
    f32 sigma=1.f/(k_pi*h*h*h);
    if (q<1.f) return sigma*(1.f-1.5f*q*q+0.75f*q*q*q);
    if (q<2.f) { f32 t=2.f-q; return sigma*0.25f*t*t*t; }
    return 0.f;
}

// ─── SPH particle ─────────────────────────────────────────────────────────────

struct SPHParticle {
    Vec3f pos{};
    Vec3f vel{};
    Vec3f force{};
    f32   density{};
    f32   pressure{};
    f32   mass{1.f};
};

struct SPHParams {
    f32 h{0.1f};             // smoothing radius
    f32 rest_density{1000.f};
    f32 stiffness{100.f};    // pressure stiffness
    f32 viscosity{0.1f};
    Vec3f gravity{0,-9.81f,0};
    f32 dt{0.005f};
};

// Compute density and pressure for all particles
inline void sph_compute_density(
    std::span<SPHParticle> particles, const SPHParams& p)
{
    u32 N=u32(particles.size());
    for (u32 i=0;i<N;++i) {
        particles[i].density=0.f;
        for (u32 j=0;j<N;++j) {
            f32 r=length(particles[i].pos-particles[j].pos);
            particles[i].density+=particles[j].mass*sph_poly6(r,p.h);
        }
        // Tait equation of state
        f32 ratio=particles[i].density/p.rest_density;
        particles[i].pressure=p.stiffness*(ratio*ratio*ratio*ratio*ratio*ratio*ratio-1.f);
    }
}

// Compute forces (pressure + viscosity + gravity)
inline void sph_compute_forces(
    std::span<SPHParticle> particles, const SPHParams& p)
{
    u32 N=u32(particles.size());
    for (u32 i=0;i<N;++i) {
        particles[i].force=p.gravity*particles[i].mass;
        for (u32 j=0;j<N;++j) {
            if(i==j) continue;
            Vec3f rij=particles[i].pos-particles[j].pos;
            f32 r=length(rij);
            if(r<1e-6f||r>p.h) continue;
            Vec3f rhat=rij/r;
            // Pressure force
            f32 fp=-particles[j].mass*(particles[i].pressure+particles[j].pressure)
                    /(2.f*particles[j].density)*sph_spiky_grad(r,p.h);
            particles[i].force=particles[i].force+rhat*fp;
            // Viscosity force
            Vec3f dv=particles[j].vel-particles[i].vel;
            f32 fv=p.viscosity*particles[j].mass/particles[j].density*sph_viscosity_lap(r,p.h);
            particles[i].force=particles[i].force+dv*fv;
        }
    }
}

inline void sph_integrate(std::span<SPHParticle> particles, f32 dt) {
    for (auto& p : particles) {
        Vec3f acc=p.force*(1.f/p.mass);
        p.vel=p.vel+acc*dt;
        p.pos=p.pos+p.vel*dt;
    }
}

// ─── MAC (Marker-and-Cell) grid ───────────────────────────────────────────────

struct MACGrid {
    u32 nx{},ny{},nz{};
    f32 cell_size{};
    // Staggered velocities: u on x-faces, v on y-faces, w on z-faces
    std::vector<f32> u_vel, v_vel, w_vel;  // (nx+1)*ny*nz etc.
    std::vector<f32> pressure;
    std::vector<u8>  cell_type;  // 0=air, 1=fluid, 2=solid

    void resize(u32 x, u32 y, u32 z, f32 cs) {
        nx=x;ny=y;nz=z;cell_size=cs;
        u_vel.assign((x+1)*y*z,0.f);
        v_vel.assign(x*(y+1)*z,0.f);
        w_vel.assign(x*y*(z+1),0.f);
        pressure.assign(x*y*z,0.f);
        cell_type.assign(x*y*z,0);
    }

    [[nodiscard]] f32& u(i32 x,i32 y,i32 z){ return u_vel[u32(x)+u32(nx+1)*(u32(y)+u32(ny)*u32(z))]; }
    [[nodiscard]] f32& v(i32 x,i32 y,i32 z){ return v_vel[u32(x)+u32(nx)*(u32(y)+u32(ny+1)*u32(z))]; }
    [[nodiscard]] f32& w(i32 x,i32 y,i32 z){ return w_vel[u32(x)+u32(nx)*(u32(y)+u32(ny)*u32(z))]; }
    [[nodiscard]] f32  p(i32 x,i32 y,i32 z)const{return pressure[u32(x)+u32(nx)*(u32(y)+u32(ny)*u32(z))];}

    [[nodiscard]] f32 divergence(i32 x,i32 y,i32 z) const {
        f32 invH=1.f/cell_size;
        return invH*(
            const_cast<MACGrid*>(this)->u(x+1,y,z)-const_cast<MACGrid*>(this)->u(x,y,z)+
            const_cast<MACGrid*>(this)->v(x,y+1,z)-const_cast<MACGrid*>(this)->v(x,y,z)+
            const_cast<MACGrid*>(this)->w(x,y,z+1)-const_cast<MACGrid*>(this)->w(x,y,z));
    }
};

// Simple Jacobi pressure projection (one iteration)
inline void pressure_project_jacobi(MACGrid& g, f32 dt) {
    std::vector<f32> new_p(g.pressure.size(),0.f);
    f32 invH2=1.f/(g.cell_size*g.cell_size);
    for (u32 z=1;z<g.nz-1;++z)
    for (u32 y=1;y<g.ny-1;++y)
    for (u32 x=1;x<g.nx-1;++x) {
        if(g.cell_type[x+g.nx*(y+g.ny*z)]!=1) continue;
        f32 div=g.divergence(x,y,z);
        f32 sum=g.p(x-1,y,z)+g.p(x+1,y,z)+g.p(x,y-1,z)+g.p(x,y+1,z)+g.p(x,y,z-1)+g.p(x,y,z+1);
        new_p[x+g.nx*(y+g.ny*z)]=(sum-div/dt*g.cell_size*g.cell_size)/6.f;
    }
    g.pressure=new_p;
}

// Apply pressure gradient to velocity
inline void apply_pressure(MACGrid& g, f32 dt) {
    f32 invH=1.f/(g.cell_size);
    for (u32 z=1;z<g.nz;++z)
    for (u32 y=1;y<g.ny;++y)
    for (u32 x=1;x<g.nx;++x) {
        g.u(x,y,z)-=dt*invH*(g.p(x,y,z)-g.p(x-1,y,z));
        g.v(x,y,z)-=dt*invH*(g.p(x,y,z)-g.p(x,y-1,z));
        g.w(x,y,z)-=dt*invH*(g.p(x,y,z)-g.p(x,y,z-1));
    }
}

// Semi-Lagrangian advection for scalar field
inline void advect_scalar(
    const std::vector<f32>& q, std::vector<f32>& q_new,
    const MACGrid& g, f32 dt)
{
    for (u32 z=0;z<g.nz;++z)
    for (u32 y=0;y<g.ny;++y)
    for (u32 x=0;x<g.nx;++x) {
        // Sample velocity at cell center
        f32 vx=(const_cast<MACGrid&>(g).u(x,y,z)+const_cast<MACGrid&>(g).u(x+1,y,z))*0.5f;
        f32 vy=(const_cast<MACGrid&>(g).v(x,y,z)+const_cast<MACGrid&>(g).v(x,y+1,z))*0.5f;
        f32 vz=(const_cast<MACGrid&>(g).w(x,y,z)+const_cast<MACGrid&>(g).w(x,y,z+1))*0.5f;
        f32 bx=f32(x)-vx*dt/g.cell_size;
        f32 by=f32(y)-vy*dt/g.cell_size;
        f32 bz=f32(z)-vz*dt/g.cell_size;
        bx=std::clamp(bx,0.f,f32(g.nx-1));
        by=std::clamp(by,0.f,f32(g.ny-1));
        bz=std::clamp(bz,0.f,f32(g.nz-1));
        u32 ix=u32(bx),iy=u32(by),iz=u32(bz);
        f32 fx=bx-f32(ix),fy=by-f32(iy),fz=bz-f32(iz);
        u32 ix1=std::min(ix+1,g.nx-1),iy1=std::min(iy+1,g.ny-1),iz1=std::min(iz+1,g.nz-1);
        auto idx=[&](u32 a,u32 b,u32 c){return a+g.nx*(b+g.ny*c);};
        f32 v000=q[idx(ix,iy,iz)],  v100=q[idx(ix1,iy,iz)];
        f32 v010=q[idx(ix,iy1,iz)], v110=q[idx(ix1,iy1,iz)];
        f32 v001=q[idx(ix,iy,iz1)], v101=q[idx(ix1,iy,iz1)];
        f32 v011=q[idx(ix,iy1,iz1)],v111=q[idx(ix1,iy1,iz1)];
        q_new[idx(x,y,z)]=
            v000*(1-fx)*(1-fy)*(1-fz)+v100*fx*(1-fy)*(1-fz)+
            v010*(1-fx)*fy*(1-fz)+    v110*fx*fy*(1-fz)+
            v001*(1-fx)*(1-fy)*fz+    v101*fx*(1-fy)*fz+
            v011*(1-fx)*fy*fz+        v111*fx*fy*fz;
    }
}

// ─── Smoke/buoyancy ───────────────────────────────────────────────────────────

struct SmokeParams {
    f32 alpha{0.0625f};   // smoke density buoyancy
    f32 beta{0.025f};     // temperature buoyancy
    f32 ambient_temp{0.f};
};

inline void apply_buoyancy(MACGrid& g,
    const std::vector<f32>& smoke_density,
    const std::vector<f32>& temperature,
    const SmokeParams& s, f32 dt)
{
    for (u32 z=0;z<g.nz;++z)
    for (u32 x=0;x<g.nx;++x)
    for (u32 y=1;y<g.ny;++y) {
        u32 idx=x+g.nx*(y+g.ny*z);
        u32 idx_below=x+g.nx*(y-1+g.ny*z);
        f32 buoy=-s.alpha*(smoke_density[idx]+smoke_density[idx_below])*0.5f
                 +s.beta*((temperature[idx]+temperature[idx_below])*0.5f-s.ambient_temp);
        g.v(i32(x),i32(y),i32(z))+=buoy*dt;
    }
}

} // namespace qs
