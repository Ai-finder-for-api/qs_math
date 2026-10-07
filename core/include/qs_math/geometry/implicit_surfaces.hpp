#pragma once
// qs_math/implicit_surfaces.hpp — Metaballs, RBF implicit, level-set operations (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <functional>
#include <array>

namespace qs {

// ─── Implicit surface base ────────────────────────────────────────────────────

// An implicit surface is defined by f(p) = 0; inside: f < 0, outside: f > 0
using ImplicitFn = std::function<f32(const Vec3f&)>;

[[nodiscard]] inline Vec3f implicit_gradient(const ImplicitFn& f, const Vec3f& p, f32 eps=0.001f) {
    return Vec3f{
        f({p.x+eps,p.y,p.z}) - f({p.x-eps,p.y,p.z}),
        f({p.x,p.y+eps,p.z}) - f({p.x,p.y-eps,p.z}),
        f({p.x,p.y,p.z+eps}) - f({p.x,p.y,p.z-eps})} * (1.f/(2.f*eps));
}

[[nodiscard]] inline Vec3f implicit_normal(const ImplicitFn& f, const Vec3f& p, f32 eps=0.001f) {
    Vec3f g = implicit_gradient(f, p, eps);
    f32 l = length(g);
    return l > 1e-8f ? g/l : Vec3f{0,1,0};
}

// ─── Signed Distance Functions ────────────────────────────────────────────────

[[nodiscard]] inline f32 sdf_sphere(const Vec3f& p, const Vec3f& c, f32 r) {
    return length(p-c)-r;
}

[[nodiscard]] inline f32 sdf_box(const Vec3f& p, const Vec3f& b) {
    Vec3f q{std::abs(p.x)-b.x, std::abs(p.y)-b.y, std::abs(p.z)-b.z};
    Vec3f mx{std::max(q.x,0.f),std::max(q.y,0.f),std::max(q.z,0.f)};
    return length(mx) + std::min(std::max(q.x,std::max(q.y,q.z)),0.f);
}

[[nodiscard]] inline f32 sdf_torus(const Vec3f& p, f32 R, f32 r) {
    Vec2f q{std::sqrt(p.x*p.x+p.z*p.z)-R, p.y};
    return std::sqrt(q.x*q.x+q.y*q.y)-r;
}

[[nodiscard]] inline f32 sdf_capsule(const Vec3f& p, const Vec3f& a, const Vec3f& b, f32 r) {
    Vec3f pa=p-a, ba=b-a;
    f32 t=std::clamp(dot(pa,ba)/dot(ba,ba),0.f,1.f);
    return length(pa-ba*t)-r;
}

[[nodiscard]] inline f32 sdf_plane(const Vec3f& p, const Vec3f& n, f32 d) {
    return dot(p,n)-d;
}

[[nodiscard]] inline f32 sdf_cylinder(const Vec3f& p, f32 r, f32 h) {
    Vec2f d{std::sqrt(p.x*p.x+p.z*p.z)-r, std::abs(p.y)-h};
    return std::min(std::max(d.x,d.y),0.f)+length(Vec2f{std::max(d.x,0.f),std::max(d.y,0.f)});
}

// ─── SDF boolean operations ───────────────────────────────────────────────────

[[nodiscard]] inline f32 sdf_union(f32 a, f32 b) { return std::min(a,b); }
[[nodiscard]] inline f32 sdf_intersect(f32 a, f32 b) { return std::max(a,b); }
[[nodiscard]] inline f32 sdf_subtract(f32 a, f32 b) { return std::max(a,-b); }

// Smooth union (blending)
[[nodiscard]] inline f32 sdf_smooth_union(f32 a, f32 b, f32 k) {
    f32 h=std::clamp(0.5f+0.5f*(b-a)/k,0.f,1.f);
    return std::lerp(b,a,h)-k*h*(1.f-h);
}

[[nodiscard]] inline f32 sdf_smooth_subtract(f32 a, f32 b, f32 k) {
    f32 h=std::clamp(0.5f-0.5f*(a+b)/k,0.f,1.f);
    return std::lerp(a,-b,h)+k*h*(1.f-h);
}

[[nodiscard]] inline f32 sdf_smooth_intersect(f32 a, f32 b, f32 k) {
    f32 h=std::clamp(0.5f-0.5f*(b-a)/k,0.f,1.f);
    return std::lerp(b,a,h)+k*h*(1.f-h);
}

// ─── Metaballs ────────────────────────────────────────────────────────────────

struct Metaball {
    Vec3f center{};
    f32   radius{1.f};
    f32   strength{1.f};
};

// Wyvill polynomial falloff
[[nodiscard]] inline f32 metaball_falloff(f32 r_sq, f32 R) {
    f32 R2=R*R;
    if(r_sq>=R2) return 0.f;
    f32 t=1.f-r_sq/R2;
    return t*t*t;
}

// Metaball field at point p
[[nodiscard]] inline f32 metaball_field(
    const Vec3f& p, std::span<const Metaball> balls, f32 threshold=1.f)
{
    f32 sum=0.f;
    for (const auto& b : balls)
        sum += b.strength * metaball_falloff(length_sq(p-b.center), b.radius);
    return threshold - sum;
}

// ─── RBF implicit surface ────────────────────────────────────────────────────

struct RBFImplicit {
    struct Center {
        Vec3f pos{};
        f32   weight{};
    };
    std::vector<Center> centers;
    f32 eps{1.f}; // RBF shape parameter

    // Thin-plate spline RBF: r^2 * log(r)
    [[nodiscard]] f32 rbf_tps(f32 r) const {
        if(r<1e-10f) return 0.f;
        return r*r*std::log(r);
    }

    // Gaussian RBF
    [[nodiscard]] f32 rbf_gaussian(f32 r) const {
        return std::exp(-eps*eps*r*r);
    }

    [[nodiscard]] f32 eval(const Vec3f& p) const {
        f32 v=0.f;
        for (const auto& c : centers)
            v += c.weight * rbf_gaussian(length(p-c.pos));
        return v;
    }

    [[nodiscard]] Vec3f gradient(const Vec3f& p) const {
        Vec3f g{0,0,0};
        for (const auto& c : centers) {
            Vec3f d=p-c.pos;
            f32 r=length(d);
            if(r<1e-8f) continue;
            f32 dr=-2.f*eps*eps*std::exp(-eps*eps*r*r);
            g=g+d*(c.weight*dr);
        }
        return g;
    }
};

// ─── Level-set operations ─────────────────────────────────────────────────────

// Reinitialize level set via fast sweeping (simplified 2D)
inline void reinitialize_levelset_2d(
    std::vector<f32>& phi, u32 w, u32 h, f32 cell_size=1.f, u32 iters=4)
{
    auto sgn=[](f32 v)->f32{return v>0.f?1.f:v<0.f?-1.f:0.f;};
    for (u32 iter=0;iter<iters;++iter)
    for (u32 y=1;y<h-1;++y)
    for (u32 x=1;x<w-1;++x) {
        f32 ax=std::min(std::abs(phi[y*w+x-1]),std::abs(phi[y*w+x+1]));
        f32 ay=std::min(std::abs(phi[(y-1)*w+x]),std::abs(phi[(y+1)*w+x]));
        f32 s=sgn(phi[y*w+x]);
        f32 d=std::min(ax,ay)+cell_size;
        if(d>std::max(ax,ay)+cell_size) {
            f32 s2=ax*ax+ay*ay;
            d=(ax+ay+std::sqrt(std::max(0.f,2.f*cell_size*cell_size-s2*(1.f))))*.5f;
        }
        phi[y*w+x]=s*std::min(std::abs(phi[y*w+x]),d);
    }
}

// Advect level set via semi-Lagrangian
inline void advect_levelset(
    std::vector<f32>& phi, const std::vector<f32>& vx, const std::vector<f32>& vy,
    u32 w, u32 h, f32 dt, f32 cell_size=1.f)
{
    std::vector<f32> phi_new(phi.size());
    for (u32 y=0;y<h;++y)
    for (u32 x=0;x<w;++x) {
        f32 bx=f32(x)-vx[y*w+x]*dt/cell_size;
        f32 by=f32(y)-vy[y*w+x]*dt/cell_size;
        bx=std::clamp(bx,0.f,f32(w-1));
        by=std::clamp(by,0.f,f32(h-1));
        u32 ix=u32(bx),iy=u32(by);
        f32 fx=bx-f32(ix),fy=by-f32(iy);
        u32 ix1=std::min(ix+1,w-1),iy1=std::min(iy+1,h-1);
        phi_new[y*w+x]=phi[iy*w+ix]*(1-fx)*(1-fy)+phi[iy*w+ix1]*fx*(1-fy)
                      +phi[iy1*w+ix]*(1-fx)*fy+phi[iy1*w+ix1]*fx*fy;
    }
    phi=phi_new;
}

// ─── Sphere tracing (ray marching) ───────────────────────────────────────────

struct SphereTraceResult {
    Vec3f hit_pos{};
    Vec3f normal{};
    f32   dist{};
    u32   steps{};
    bool  hit{};
};

[[nodiscard]] inline SphereTraceResult sphere_trace(
    const Vec3f& ray_origin, const Vec3f& ray_dir,
    const ImplicitFn& sdf,
    f32 max_dist=100.f, u32 max_steps=256, f32 tol=0.001f)
{
    f32 t=0.f;
    for (u32 i=0;i<max_steps;++i) {
        Vec3f p=ray_origin+ray_dir*t;
        f32 d=sdf(p);
        if(d<tol) {
            return {p, implicit_normal(sdf,p), t, i, true};
        }
        t+=d;
        if(t>max_dist) break;
    }
    return {{},{},max_dist,max_steps,false};
}

} // namespace qs
