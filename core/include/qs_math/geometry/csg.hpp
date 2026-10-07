#pragma once
// qs_math/geometry/csg.hpp — Constructive Solid Geometry: SDF operations,
//   boolean union/intersection/difference, smooth blending, domain warping,
//   extrusion, revolution, morphing, CSG tree evaluation (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <functional>
#include <vector>
#include <memory>
#include <cmath>

namespace qs::math::geometry {

// ─── SDF primitive types ──────────────────────────────────────────────────
using SDF3 = std::function<f32(Vec3f)>;
using SDF2 = std::function<f32(Vec2f)>;

// ─── Primitive SDFs ───────────────────────────────────────────────────────
[[nodiscard]] inline SDF3 sdf_sphere(Vec3f center, f32 radius) noexcept {
    return [=](Vec3f p){ return (p-center).length()-radius; };
}

[[nodiscard]] inline SDF3 sdf_box(Vec3f center, Vec3f half) noexcept {
    return [=](Vec3f p){
        Vec3f q={std::abs(p.x-center.x)-half.x,
                 std::abs(p.y-center.y)-half.y,
                 std::abs(p.z-center.z)-half.z};
        return Vec3f{max(q.x,0.f),max(q.y,0.f),max(q.z,0.f)}.length()
               +min(max(q.x,max(q.y,q.z)),0.f);
    };
}

[[nodiscard]] inline SDF3 sdf_rounded_box(Vec3f center, Vec3f half, f32 r) noexcept {
    return [=](Vec3f p){
        Vec3f q={std::abs(p.x-center.x)-half.x+r,
                 std::abs(p.y-center.y)-half.y+r,
                 std::abs(p.z-center.z)-half.z+r};
        return Vec3f{max(q.x,0.f),max(q.y,0.f),max(q.z,0.f)}.length()
               +min(max(q.x,max(q.y,q.z)),0.f)-r;
    };
}

[[nodiscard]] inline SDF3 sdf_torus(Vec3f center, f32 R, f32 r) noexcept {
    return [=](Vec3f p){
        Vec3f d=p-center;
        f32 q=std::sqrt(d.x*d.x+d.z*d.z)-R;
        return std::sqrt(q*q+d.y*d.y)-r;
    };
}

[[nodiscard]] inline SDF3 sdf_cylinder(Vec3f a, Vec3f b, f32 r) noexcept {
    return [=](Vec3f p){
        Vec3f ab=b-a, ap=p-a;
        f32 t=clamp01(dot(ap,ab)/dot(ab,ab));
        Vec3f closest=a+ab*t;
        return (p-closest).length()-r;
    };
}

[[nodiscard]] inline SDF3 sdf_capsule(Vec3f a, Vec3f b, f32 r) noexcept {
    return sdf_cylinder(a,b,r); // same for capsule
}

[[nodiscard]] inline SDF3 sdf_cone(Vec3f tip, Vec3f base_center, f32 base_r) noexcept {
    return [=](Vec3f p){
        Vec3f ab=base_center-tip, ap=p-tip;
        f32 h=ab.length();
        Vec3f abn=ab*(1.f/h);
        f32 t=clamp(dot(ap,abn),0.f,h);
        f32 cone_r=base_r*t/h;
        Vec3f closest=tip+abn*t;
        f32 lateral=(p-closest).length()-cone_r;
        return lateral;
    };
}

[[nodiscard]] inline SDF3 sdf_plane(Vec3f normal, f32 offset) noexcept {
    return [=](Vec3f p){ return dot(p,normal)+offset; };
}

[[nodiscard]] inline SDF3 sdf_ellipsoid(Vec3f center, Vec3f radii) noexcept {
    return [=](Vec3f p){
        Vec3f d={(p.x-center.x)/radii.x,(p.y-center.y)/radii.y,(p.z-center.z)/radii.z};
        f32 len=d.length();
        return len*(len-1.f)/Vec3f{d.x/radii.x,d.y/radii.y,d.z/radii.z}.length();
    };
}

[[nodiscard]] inline SDF3 sdf_octahedron(Vec3f center, f32 s) noexcept {
    return [=](Vec3f p){
        Vec3f q={std::abs(p.x-center.x),std::abs(p.y-center.y),std::abs(p.z-center.z)};
        return (q.x+q.y+q.z-s)*0.57735027f;
    };
}

// ─── Boolean operations ───────────────────────────────────────────────────
[[nodiscard]] inline SDF3 sdf_union(SDF3 a, SDF3 b) noexcept {
    return [a=std::move(a),b=std::move(b)](Vec3f p){ return min(a(p),b(p)); };
}

[[nodiscard]] inline SDF3 sdf_intersection(SDF3 a, SDF3 b) noexcept {
    return [a=std::move(a),b=std::move(b)](Vec3f p){ return max(a(p),b(p)); };
}

[[nodiscard]] inline SDF3 sdf_difference(SDF3 a, SDF3 b) noexcept {
    return [a=std::move(a),b=std::move(b)](Vec3f p){ return max(a(p),-b(p)); };
}

[[nodiscard]] inline SDF3 sdf_negate(SDF3 a) noexcept {
    return [a=std::move(a)](Vec3f p){ return -a(p); };
}

// ─── Smooth boolean operations (k controls blend radius) ──────────────────
[[nodiscard]] inline SDF3 sdf_smooth_union(SDF3 a, SDF3 b, f32 k) noexcept {
    return [a=std::move(a),b=std::move(b),k](Vec3f p){
        f32 da=a(p), db=b(p);
        f32 h=max(k-std::abs(da-db),0.f)/k;
        return min(da,db)-h*h*k*0.25f;
    };
}

[[nodiscard]] inline SDF3 sdf_smooth_intersection(SDF3 a, SDF3 b, f32 k) noexcept {
    return [a=std::move(a),b=std::move(b),k](Vec3f p){
        f32 da=a(p), db=b(p);
        f32 h=max(k-std::abs(da-db),0.f)/k;
        return max(da,db)+h*h*k*0.25f;
    };
}

[[nodiscard]] inline SDF3 sdf_smooth_difference(SDF3 a, SDF3 b, f32 k) noexcept {
    return [a=std::move(a),b=std::move(b),k](Vec3f p){
        f32 da=a(p), nb=-b(p);
        f32 h=max(k-std::abs(da-nb),0.f)/k;
        return max(da,nb)+h*h*k*0.25f;
    };
}

// Polynomial smooth min (C2 continuity)
[[nodiscard]] inline SDF3 sdf_smin_poly(SDF3 a, SDF3 b, f32 k) noexcept {
    return [a=std::move(a),b=std::move(b),k](Vec3f p){
        f32 da=a(p), db=b(p);
        f32 h=clamp01(0.5f+(db-da)/(2.f*k));
        return lerp(db,da,h)-k*h*(1.f-h);
    };
}

// ─── Displacement & warping ───────────────────────────────────────────────
[[nodiscard]] inline SDF3 sdf_displace(SDF3 base, SDF3 displace_field, f32 strength=1.f) noexcept {
    return [b=std::move(base),d=std::move(displace_field),strength](Vec3f p){
        return b(p)+d(p)*strength;
    };
}

[[nodiscard]] inline SDF3 sdf_twist(SDF3 base, f32 k) noexcept {
    return [b=std::move(base),k](Vec3f p){
        f32 c=std::cos(k*p.y), s=std::sin(k*p.y);
        Vec3f q={c*p.x-s*p.z, p.y, s*p.x+c*p.z};
        return b(q);
    };
}

[[nodiscard]] inline SDF3 sdf_bend(SDF3 base, f32 k) noexcept {
    return [b=std::move(base),k](Vec3f p){
        f32 c=std::cos(k*p.x), s=std::sin(k*p.x);
        Vec3f q={c*p.x-s*p.y, s*p.x+c*p.y, p.z};
        return b(q);
    };
}

// Domain repetition (infinite tiling)
[[nodiscard]] inline SDF3 sdf_repeat(SDF3 base, Vec3f period) noexcept {
    return [b=std::move(base),period](Vec3f p){
        Vec3f q={p.x-period.x*std::round(p.x/period.x),
                 p.y-period.y*std::round(p.y/period.y),
                 p.z-period.z*std::round(p.z/period.z)};
        return b(q);
    };
}

// Finite repetition (n copies along each axis)
[[nodiscard]] inline SDF3 sdf_repeat_finite(SDF3 base, Vec3f step, Vec3<i32> count) noexcept {
    return [b=std::move(base),step,count](Vec3f p){
        Vec3f q={
            p.x-step.x*clamp(std::round(p.x/step.x),0.f,f32(count.x-1)),
            p.y-step.y*clamp(std::round(p.y/step.y),0.f,f32(count.y-1)),
            p.z-step.z*clamp(std::round(p.z/step.z),0.f,f32(count.z-1))
        };
        return b(q);
    };
}

// ─── Extrusion & revolution ───────────────────────────────────────────────
// Extrude a 2D SDF along Z by height h
[[nodiscard]] inline SDF3 sdf_extrude(SDF2 sdf2d, f32 h) noexcept {
    return [s=std::move(sdf2d),h](Vec3f p){
        f32 d=s(Vec2f{p.x,p.y});
        Vec2f w={d,std::abs(p.z)-h};
        return min(max(w.x,w.y),0.f)+Vec2f{max(w.x,0.f),max(w.y,0.f)}.length();
    };
}

// Revolve a 2D SDF (in XY) around Y axis
[[nodiscard]] inline SDF3 sdf_revolve(SDF2 sdf2d, f32 offset=0.f) noexcept {
    return [s=std::move(sdf2d),offset](Vec3f p){
        f32 r=std::sqrt(p.x*p.x+p.z*p.z)-offset;
        return s(Vec2f{r,p.y});
    };
}

// ─── Morphing ─────────────────────────────────────────────────────────────
[[nodiscard]] inline SDF3 sdf_lerp(SDF3 a, SDF3 b, f32 t) noexcept {
    return [a=std::move(a),b=std::move(b),t](Vec3f p){
        return lerp(a(p),b(p),t);
    };
}

// ─── Offset / shell ───────────────────────────────────────────────────────
[[nodiscard]] inline SDF3 sdf_offset(SDF3 base, f32 d) noexcept {
    return [b=std::move(base),d](Vec3f p){ return b(p)-d; };
}

[[nodiscard]] inline SDF3 sdf_shell(SDF3 base, f32 thickness) noexcept {
    return [b=std::move(base),thickness](Vec3f p){ return std::abs(b(p))-thickness; };
}

// ─── SDF gradient (central difference normal) ────────────────────────────
[[nodiscard]] inline Vec3f sdf_normal(const SDF3& f, Vec3f p, f32 eps=1e-3f) noexcept {
    return Vec3f{
        f({p.x+eps,p.y,p.z})-f({p.x-eps,p.y,p.z}),
        f({p.x,p.y+eps,p.z})-f({p.x,p.y-eps,p.z}),
        f({p.x,p.y,p.z+eps})-f({p.x,p.y,p.z-eps})
    }.normalized();
}

// Tetrahedron technique normal (4 samples vs 6)
[[nodiscard]] inline Vec3f sdf_normal_tet(const SDF3& f, Vec3f p, f32 eps=1e-3f) noexcept {
    const Vec3f k0={1,-1,-1}, k1={-1,-1,1}, k2={-1,1,-1}, k3={1,1,1};
    return (k0*f(p+k0*eps)+k1*f(p+k1*eps)+k2*f(p+k2*eps)+k3*f(p+k3*eps)).normalized();
}

// ─── Sphere tracing (ray marching) ───────────────────────────────────────
struct MarchResult {
    bool  hit;
    f32   t;      // ray parameter
    Vec3f point;
    Vec3f normal;
    int   steps;
};

[[nodiscard]] inline MarchResult sphere_trace(
    const SDF3& scene, Vec3f origin, Vec3f dir,
    f32 t_min=0.001f, f32 t_max=1000.f,
    f32 eps=1e-4f, int max_steps=256) noexcept
{
    dir=dir.normalized();
    f32 t=t_min;
    for(int i=0;i<max_steps;++i){
        Vec3f p=origin+dir*t;
        f32 d=scene(p);
        if(d<eps) return {true,t,p,sdf_normal_tet(scene,p),i};
        t+=d;
        if(t>t_max) break;
    }
    return {false,t_max,{},Vec3f{0,1,0},max_steps};
}

// Soft shadow (penumbra approximation)
[[nodiscard]] inline f32 soft_shadow(
    const SDF3& scene, Vec3f origin, Vec3f dir,
    f32 t_min, f32 t_max, f32 k=16.f) noexcept
{
    dir=dir.normalized(); f32 t=t_min, shadow=1.f;
    for(int i=0;i<64;++i){
        f32 d=scene(origin+dir*t);
        if(d<1e-4f) return 0.f;
        shadow=min(shadow,k*d/t);
        t+=d; if(t>t_max) break;
    }
    return clamp01(shadow);
}

// Ambient occlusion via SDF stepping
[[nodiscard]] inline f32 sdf_ao(
    const SDF3& scene, Vec3f p, Vec3f n, f32 step=0.1f, int steps=5) noexcept
{
    f32 occ=0,weight=1.f;
    for(int i=1;i<=steps;++i){
        f32 d=step*i;
        occ+=weight*(d-scene(p+n*d))/d;
        weight*=0.5f;
    }
    return clamp01(1.f-1.5f*occ);
}

// ─── 2D SDFs ──────────────────────────────────────────────────────────────
[[nodiscard]] inline SDF2 sdf2_circle(Vec2f c, f32 r) noexcept {
    return [=](Vec2f p){ return (p-c).length()-r; };
}
[[nodiscard]] inline SDF2 sdf2_box(Vec2f c, Vec2f h) noexcept {
    return [=](Vec2f p){
        Vec2f d={std::abs(p.x-c.x)-h.x,std::abs(p.y-c.y)-h.y};
        return Vec2f{max(d.x,0.f),max(d.y,0.f)}.length()+min(max(d.x,d.y),0.f);
    };
}
[[nodiscard]] inline SDF2 sdf2_segment(Vec2f a, Vec2f b, f32 r=0.f) noexcept {
    return [=](Vec2f p){
        Vec2f ab={b.x-a.x,b.y-a.y}, ap={p.x-a.x,p.y-a.y};
        f32 t=clamp01((ap.x*ab.x+ap.y*ab.y)/(ab.x*ab.x+ab.y*ab.y+1e-30f));
        Vec2f d={ap.x-t*ab.x,ap.y-t*ab.y};
        return std::sqrt(d.x*d.x+d.y*d.y)-r;
    };
}
[[nodiscard]] inline SDF2 sdf2_union(SDF2 a, SDF2 b) noexcept {
    return [a=std::move(a),b=std::move(b)](Vec2f p){ return min(a(p),b(p)); };
}
[[nodiscard]] inline SDF2 sdf2_difference(SDF2 a, SDF2 b) noexcept {
    return [a=std::move(a),b=std::move(b)](Vec2f p){ return max(a(p),-b(p)); };
}
[[nodiscard]] inline SDF2 sdf2_smooth_union(SDF2 a, SDF2 b, f32 k) noexcept {
    return [a=std::move(a),b=std::move(b),k](Vec2f p){
        f32 da=a(p),db=b(p),h=max(k-std::abs(da-db),0.f)/k;
        return min(da,db)-h*h*k*0.25f;
    };
}

} // namespace qs::math::geometry
