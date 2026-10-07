#pragma once
// vector_field.hpp — 2D/3D vector field mathematics: divergence, curl, gradient,
//                    Laplacian, stream functions, potential fields, flow analysis,
//                    Helmholtz decomposition, vorticity
#include "vec2.hpp"
#include "vec3.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Continuous vector field operations
// ---------------------------------------------------------------------------
// Gradient of scalar field f at point (finite differences)
template<typename Fn2D>
[[nodiscard]] inline Vec2f gradient2d(Fn2D&& f, Vec2f p, f32 eps=0.001f) noexcept {
    return {(f({p.x+eps,p.y})-f({p.x-eps,p.y}))/(2.f*eps),
            (f({p.x,p.y+eps})-f({p.x,p.y-eps}))/(2.f*eps)};
}

template<typename Fn3D>
[[nodiscard]] inline Vec3f gradient3d(Fn3D&& f, Vec3f p, f32 eps=0.001f) noexcept {
    return {(f({p.x+eps,p.y,p.z})-f({p.x-eps,p.y,p.z}))/(2.f*eps),
            (f({p.x,p.y+eps,p.z})-f({p.x,p.y-eps,p.z}))/(2.f*eps),
            (f({p.x,p.y,p.z+eps})-f({p.x,p.y,p.z-eps}))/(2.f*eps)};
}

// Divergence of vector field F=(Fx,Fy,Fz)
template<typename FxFn, typename FyFn, typename FzFn>
[[nodiscard]] inline f32 divergence3d(FxFn&& Fx, FyFn&& Fy, FzFn&& Fz,
                                       Vec3f p, f32 eps=0.001f) noexcept {
    f32 dFx=(Fx({p.x+eps,p.y,p.z})-Fx({p.x-eps,p.y,p.z}))/(2.f*eps);
    f32 dFy=(Fy({p.x,p.y+eps,p.z})-Fy({p.x,p.y-eps,p.z}))/(2.f*eps);
    f32 dFz=(Fz({p.x,p.y,p.z+eps})-Fz({p.x,p.y,p.z-eps}))/(2.f*eps);
    return dFx+dFy+dFz;
}

// Curl of 3D vector field
template<typename FxFn, typename FyFn, typename FzFn>
[[nodiscard]] inline Vec3f curl3d(FxFn&& Fx, FyFn&& Fy, FzFn&& Fz,
                                   Vec3f p, f32 eps=0.001f) noexcept {
    f32 dFz_dy=(Fz({p.x,p.y+eps,p.z})-Fz({p.x,p.y-eps,p.z}))/(2.f*eps);
    f32 dFy_dz=(Fy({p.x,p.y,p.z+eps})-Fy({p.x,p.y,p.z-eps}))/(2.f*eps);
    f32 dFx_dz=(Fx({p.x,p.y,p.z+eps})-Fx({p.x,p.y,p.z-eps}))/(2.f*eps);
    f32 dFz_dx=(Fz({p.x+eps,p.y,p.z})-Fz({p.x-eps,p.y,p.z}))/(2.f*eps);
    f32 dFy_dx=(Fy({p.x+eps,p.y,p.z})-Fy({p.x-eps,p.y,p.z}))/(2.f*eps);
    f32 dFx_dy=(Fx({p.x,p.y+eps,p.z})-Fx({p.x,p.y-eps,p.z}))/(2.f*eps);
    return {dFz_dy-dFy_dz, dFx_dz-dFz_dx, dFy_dx-dFx_dy};
}

// Laplacian of scalar field
template<typename Fn>
[[nodiscard]] inline f32 laplacian3d(Fn&& f, Vec3f p, f32 eps=0.001f) noexcept {
    f32 fc=f(p);
    return (f({p.x+eps,p.y,p.z})+f({p.x-eps,p.y,p.z})
           +f({p.x,p.y+eps,p.z})+f({p.x,p.y-eps,p.z})
           +f({p.x,p.y,p.z+eps})+f({p.x,p.y,p.z-eps})-6.f*fc)/(eps*eps);
}

// ---------------------------------------------------------------------------
// Discrete vector field (grid-based)
// ---------------------------------------------------------------------------
struct VectorField2D {
    Vec2f* data{nullptr};
    i32    w{0}, h{0};
    f32    dx{1.f}, dy{1.f};
    Vec2f  origin{0.f};

    [[nodiscard]] Vec2f sample(Vec2f p) const noexcept {
        if(!data) return {};
        f32 fx=(p.x-origin.x)/dx, fy=(p.y-origin.y)/dy;
        i32 ix=clamp(i32(fx),0,w-2), iy=clamp(i32(fy),0,h-2);
        f32 bx=fx-ix, by=fy-iy;
        Vec2f v00=data[iy*w+ix],     v10=data[iy*w+ix+1];
        Vec2f v01=data[(iy+1)*w+ix], v11=data[(iy+1)*w+ix+1];
        return lerp(lerp(v00,v10,bx),lerp(v01,v11,bx),by);
    }

    [[nodiscard]] f32 divergence_at(i32 x, i32 y) const noexcept {
        if(!data||x<=0||x>=w-1||y<=0||y>=h-1) return 0.f;
        f32 dvx_dx=(data[y*w+x+1].x-data[y*w+x-1].x)/(2.f*dx);
        f32 dvy_dy=(data[(y+1)*w+x].y-data[(y-1)*w+x].y)/(2.f*dy);
        return dvx_dx+dvy_dy;
    }

    [[nodiscard]] f32 curl_z_at(i32 x, i32 y) const noexcept {
        if(!data||x<=0||x>=w-1||y<=0||y>=h-1) return 0.f;
        f32 dvy_dx=(data[y*w+x+1].y-data[y*w+x-1].y)/(2.f*dx);
        f32 dvx_dy=(data[(y+1)*w+x].x-data[(y-1)*w+x].x)/(2.f*dy);
        return dvy_dx-dvx_dy;
    }

    // Streamline integration (Euler step)
    [[nodiscard]] Vec2f streamline_step(Vec2f p, f32 dt) const noexcept {
        return p+sample(p)*dt;
    }
    // RK4 streamline step
    [[nodiscard]] Vec2f streamline_rk4(Vec2f p, f32 dt) const noexcept {
        Vec2f k1=sample(p);
        Vec2f k2=sample(p+k1*(dt*0.5f));
        Vec2f k3=sample(p+k2*(dt*0.5f));
        Vec2f k4=sample(p+k3*dt);
        return p+(k1+k2*2.f+k3*2.f+k4)*(dt/6.f);
    }
};

struct VectorField3D {
    Vec3f* data{nullptr};
    i32 w{0},h{0},d{0};
    f32 dx{1.f},dy{1.f},dz{1.f};
    Vec3f origin{0.f};

    [[nodiscard]] Vec3f sample(Vec3f p) const noexcept {
        if(!data) return {};
        f32 fx=(p.x-origin.x)/dx, fy=(p.y-origin.y)/dy, fz=(p.z-origin.z)/dz;
        i32 ix=clamp(i32(fx),0,w-2), iy=clamp(i32(fy),0,h-2), iz=clamp(i32(fz),0,d-2);
        f32 bx=fx-ix,by=fy-iy,bz=fz-iz;
        auto g=[&](i32 x,i32 y,i32 z)->Vec3f{return data[(z*h+y)*w+x];};
        return lerp(bilinear(g(ix,iy,iz),g(ix+1,iy,iz),g(ix,iy+1,iz),g(ix+1,iy+1,iz),bx,by),
                    bilinear(g(ix,iy,iz+1),g(ix+1,iy,iz+1),g(ix,iy+1,iz+1),g(ix+1,iy+1,iz+1),bx,by),
                    bz);
    }

    [[nodiscard]] Vec3f streamline_rk4(Vec3f p, f32 dt) const noexcept {
        Vec3f k1=sample(p);
        Vec3f k2=sample(p+k1*(dt*0.5f));
        Vec3f k3=sample(p+k2*(dt*0.5f));
        Vec3f k4=sample(p+k3*dt);
        return p+(k1+k2*2.f+k3*2.f+k4)*(dt/6.f);
    }
};

// ---------------------------------------------------------------------------
// Potential fields
// ---------------------------------------------------------------------------
// Gravitational potential
[[nodiscard]] constexpr f32 gravitational_potential(Vec3f r, f32 mass, f32 G=6.674e-11f) noexcept {
    f32 d=r.length();
    return d<EPS_F32?-F32_INF:-G*mass/d;
}
[[nodiscard]] constexpr Vec3f gravitational_field(Vec3f r, f32 mass, f32 G=6.674e-11f) noexcept {
    f32 d2=r.length_sq();
    if(d2<EPS_F32) return {0.f};
    return r*(-G*mass/(d2*sqrt(d2)));
}

// Electrostatic potential (Coulomb)
[[nodiscard]] constexpr f32 coulomb_potential(Vec3f r, f32 charge, f32 k=8.9875e9f) noexcept {
    f32 d=r.length();
    return d<EPS_F32?F32_INF:k*charge/d;
}
[[nodiscard]] constexpr Vec3f coulomb_field(Vec3f r, f32 charge, f32 k=8.9875e9f) noexcept {
    f32 d2=r.length_sq();
    if(d2<EPS_F32) return {0.f};
    return r*(k*charge/(d2*sqrt(d2)));
}

// Navigation potential field (attractive + repulsive)
[[nodiscard]] constexpr Vec3f potential_field_nav(
    Vec3f pos, Vec3f goal, Vec3f obstacle,
    f32 k_att=1.f, f32 k_rep=1.f, f32 d0=2.f) noexcept
{
    // Attractive force toward goal
    Vec3f att=(goal-pos)*k_att;
    // Repulsive force from obstacle
    Vec3f diff=pos-obstacle;
    f32 d=diff.length();
    Vec3f rep{0.f};
    if(d<d0&&d>EPS_F32){
        f32 r=k_rep*(1.f/d-1.f/d0)/(d*d);
        rep=diff.normalized()*r;
    }
    return att+rep;
}

// ---------------------------------------------------------------------------
// Helmholtz decomposition helpers
// ---------------------------------------------------------------------------
// Every smooth vector field F = -∇φ + ∇×A (potential + rotational)
// Here we provide the irrotational (gradient) component via Poisson solve hint
[[nodiscard]] inline f32 poisson_jacobi_step(
    const f32* f, f32* phi, i32 w, i32 h, i32 x, i32 y, f32 h2) noexcept
{
    if(x<=0||x>=w-1||y<=0||y>=h-1) return phi[y*w+x];
    return (phi[y*w+x+1]+phi[y*w+x-1]+phi[(y+1)*w+x]+phi[(y-1)*w+x]-h2*f[y*w+x])*0.25f;
}

} // namespace qs::math
