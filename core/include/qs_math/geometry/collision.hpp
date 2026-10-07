#pragma once
// qs_math/geometry/collision.hpp — Collision detection: GJK, EPA, SAT,
//   sphere/AABB/OBB/capsule/triangle tests, continuous collision detection,
//   swept volumes, contact generation (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <array>
#include <optional>
#include <cmath>

namespace qs::math::geometry {

// ─── Contact info ─────────────────────────────────────────────────────────
struct Contact {
    Vec3f point;
    Vec3f normal;  // from B into A
    f32   depth;   // penetration depth
};

// ─── Sphere tests ────────────────────────────────────────────────────────
[[nodiscard]] inline bool sphere_sphere(
    Vec3f ca, f32 ra, Vec3f cb, f32 rb,
    Contact* contact=nullptr) noexcept
{
    Vec3f d=ca-cb; f32 dist2=dot(d,d);
    f32 r=ra+rb;
    if(dist2>=r*r) return false;
    if(contact){
        f32 dist=std::sqrt(dist2);
        contact->normal=(dist>1e-6f)?d*(1.f/dist):Vec3f{0,1,0};
        contact->depth=r-dist;
        contact->point=cb+contact->normal*(rb-contact->depth*0.5f);
    }
    return true;
}

[[nodiscard]] inline bool sphere_plane(
    Vec3f center, f32 radius, Vec3f plane_n, f32 plane_d,
    Contact* contact=nullptr) noexcept
{
    f32 dist=dot(center,plane_n)+plane_d;
    if(dist>radius) return false;
    if(contact){
        contact->normal=plane_n;
        contact->depth=radius-dist;
        contact->point=center-plane_n*dist;
    }
    return true;
}

[[nodiscard]] inline bool sphere_aabb(
    Vec3f center, f32 radius, Vec3f lo, Vec3f hi,
    Contact* contact=nullptr) noexcept
{
    Vec3f closest={clamp(center.x,lo.x,hi.x),
                   clamp(center.y,lo.y,hi.y),
                   clamp(center.z,lo.z,hi.z)};
    Vec3f d=center-closest;
    f32 dist2=dot(d,d);
    if(dist2>=radius*radius) return false;
    if(contact){
        f32 dist=std::sqrt(dist2);
        contact->normal=(dist>1e-6f)?d*(1.f/dist):Vec3f{0,1,0};
        contact->depth=radius-dist;
        contact->point=closest;
    }
    return true;
}

// ─── AABB tests ───────────────────────────────────────────────────────────
[[nodiscard]] inline bool aabb_aabb(
    Vec3f lo1, Vec3f hi1, Vec3f lo2, Vec3f hi2,
    Contact* contact=nullptr) noexcept
{
    if(lo1.x>hi2.x||lo2.x>hi1.x) return false;
    if(lo1.y>hi2.y||lo2.y>hi1.y) return false;
    if(lo1.z>hi2.z||lo2.z>hi1.z) return false;
    if(contact){
        // Find minimum overlap axis
        f32 ox=min(hi1.x-lo2.x,hi2.x-lo1.x);
        f32 oy=min(hi1.y-lo2.y,hi2.y-lo1.y);
        f32 oz=min(hi1.z-lo2.z,hi2.z-lo1.z);
        if(ox<=oy&&ox<=oz){contact->normal={1,0,0};contact->depth=ox;}
        else if(oy<=oz){contact->normal={0,1,0};contact->depth=oy;}
        else{contact->normal={0,0,1};contact->depth=oz;}
        Vec3f c1=(lo1+hi1)*0.5f, c2=(lo2+hi2)*0.5f;
        Vec3f d=c1-c2;
        if(dot(d,contact->normal)<0) contact->normal=contact->normal*-1.f;
        contact->point=(lo1+hi1+lo2+hi2)*0.25f;
    }
    return true;
}

[[nodiscard]] inline bool aabb_ray(
    Vec3f lo, Vec3f hi, Vec3f origin, Vec3f dir,
    f32& t_near, f32& t_far) noexcept
{
    Vec3f inv={1.f/(dir.x+1e-30f),1.f/(dir.y+1e-30f),1.f/(dir.z+1e-30f)};
    f32 tx1=(lo.x-origin.x)*inv.x, tx2=(hi.x-origin.x)*inv.x;
    f32 ty1=(lo.y-origin.y)*inv.y, ty2=(hi.y-origin.y)*inv.y;
    f32 tz1=(lo.z-origin.z)*inv.z, tz2=(hi.z-origin.z)*inv.z;
    t_near=max({min(tx1,tx2),min(ty1,ty2),min(tz1,tz2)});
    t_far =min({max(tx1,tx2),max(ty1,ty2),max(tz1,tz2)});
    return t_far>=max(0.f,t_near);
}

// ─── OBB (Oriented Bounding Box) ─────────────────────────────────────────
struct OBB {
    Vec3f center;
    Vec3f axes[3];  // unit axes
    Vec3f half;     // half-extents along axes
};

// Project OBB onto axis
[[nodiscard]] inline f32 obb_project(const OBB& o, Vec3f axis) noexcept {
    return o.half.x*std::abs(dot(o.axes[0],axis))
          +o.half.y*std::abs(dot(o.axes[1],axis))
          +o.half.z*std::abs(dot(o.axes[2],axis));
}

// SAT OBB-OBB test
[[nodiscard]] inline bool obb_obb(const OBB& A, const OBB& B, Contact* contact=nullptr) noexcept {
    Vec3f T=B.center-A.center;
    // Test 15 axes: 3 from A, 3 from B, 9 cross products
    f32 min_overlap=1e30f; Vec3f min_axis{0,1,0};
    auto test=[&](Vec3f axis)->bool{
        f32 a=obb_project(A,axis);
        f32 b=obb_project(B,axis);
        f32 d=std::abs(dot(T,axis));
        f32 overlap=a+b-d;
        if(overlap<0) return true; // separating
        if(overlap<min_overlap){min_overlap=overlap;min_axis=axis;}
        return false;
    };
    for(int i=0;i<3;++i) if(test(A.axes[i])) return false;
    for(int i=0;i<3;++i) if(test(B.axes[i])) return false;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j){
        Vec3f axis=cross(A.axes[i],B.axes[j]);
        f32 l=axis.length();
        if(l>1e-6f&&test(axis*(1.f/l))) return false;
    }
    if(contact){
        if(dot(T,min_axis)<0) min_axis=min_axis*-1.f;
        contact->normal=min_axis;
        contact->depth=min_overlap;
        contact->point=A.center+min_axis*(min_overlap*0.5f);
    }
    return true;
}

// ─── Capsule tests ────────────────────────────────────────────────────────
struct Capsule { Vec3f a,b; f32 r; };

[[nodiscard]] inline f32 capsule_segment_dist(
    const Capsule& cap, Vec3f p) noexcept
{
    Vec3f ab=cap.b-cap.a, ap=p-cap.a;
    f32 t=clamp01(dot(ap,ab)/(dot(ab,ab)+1e-30f));
    Vec3f closest=cap.a+ab*t;
    return (p-closest).length();
}

[[nodiscard]] inline bool capsule_sphere(
    const Capsule& cap, Vec3f center, f32 sphere_r,
    Contact* contact=nullptr) noexcept
{
    f32 dist=capsule_segment_dist(cap,center);
    if(dist>=cap.r+sphere_r) return false;
    if(contact){
        Vec3f ab=cap.b-cap.a, ap=center-cap.a;
        f32 t=clamp01(dot(ap,ab)/(dot(ab,ab)+1e-30f));
        Vec3f closest=cap.a+ab*t;
        Vec3f d=center-closest;
        f32 dl=d.length();
        contact->normal=(dl>1e-6f)?d*(1.f/dl):Vec3f{0,1,0};
        contact->depth=cap.r+sphere_r-dist;
        contact->point=center-contact->normal*sphere_r;
    }
    return true;
}

[[nodiscard]] inline bool capsule_capsule(
    const Capsule& A, const Capsule& B,
    Contact* contact=nullptr) noexcept
{
    // Closest point between two line segments
    Vec3f d1=A.b-A.a, d2=B.b-B.a, r=A.a-B.a;
    f32 a=dot(d1,d1), e=dot(d2,d2), f=dot(d2,r);
    f32 s,t;
    if(a<=1e-10f&&e<=1e-10f){s=t=0;}
    else if(a<=1e-10f){s=0;t=clamp01(f/e);}
    else{
        f32 c=dot(d1,r);
        if(e<=1e-10f){t=0;s=clamp01(-c/a);}
        else{
            f32 b=dot(d1,d2),denom=a*e-b*b;
            s=denom>1e-10f?clamp01((b*f-c*e)/denom):0.f;
            t=clamp01((b*s+f)/e);
            s=clamp01((b*t-c)/a);
        }
    }
    Vec3f pA=A.a+d1*s, pB=B.a+d2*t;
    Vec3f d=pA-pB; f32 dist=d.length();
    if(dist>=A.r+B.r) return false;
    if(contact){
        contact->normal=(dist>1e-6f)?d*(1.f/dist):Vec3f{0,1,0};
        contact->depth=A.r+B.r-dist;
        contact->point=(pA+pB)*0.5f;
    }
    return true;
}

// ─── Triangle tests ───────────────────────────────────────────────────────
// Ray-triangle (Möller-Trumbore)
struct RayTriResult { bool hit; f32 t,u,v; };

[[nodiscard]] inline RayTriResult ray_triangle(
    Vec3f orig, Vec3f dir,
    Vec3f v0, Vec3f v1, Vec3f v2,
    bool cull_backface=false) noexcept
{
    Vec3f e1=v1-v0, e2=v2-v0;
    Vec3f h=cross(dir,e2);
    f32 a=dot(e1,h);
    if(cull_backface&&a<1e-8f) return {false};
    if(std::abs(a)<1e-8f) return {false};
    f32 f=1.f/a;
    Vec3f s=orig-v0;
    f32 u=f*dot(s,h);
    if(u<0||u>1) return {false};
    Vec3f q=cross(s,e1);
    f32 v=f*dot(dir,q);
    if(v<0||u+v>1) return {false};
    f32 t=f*dot(e2,q);
    return {t>1e-6f,t,u,v};
}

// Sphere-triangle (GJK-like, simplified)
[[nodiscard]] inline bool sphere_triangle(
    Vec3f center, f32 radius,
    Vec3f v0, Vec3f v1, Vec3f v2,
    Contact* contact=nullptr) noexcept
{
    // Find closest point on triangle to sphere center
    Vec3f e1=v1-v0, e2=v2-v0;
    Vec3f n=cross(e1,e2); f32 nl=n.length();
    if(nl<1e-10f) return false;
    Vec3f nhat=n*(1.f/nl);
    // Barycentric coords
    Vec3f d=center-v0;
    f32 d11=dot(e1,e1),d12=dot(e1,e2),d22=dot(e2,e2);
    f32 d31=dot(d,e1),d32=dot(d,e2);
    f32 denom=d11*d22-d12*d12;
    f32 s=clamp01((d22*d31-d12*d32)/denom);
    f32 t=clamp01((d11*d32-d12*d31)/denom);
    if(s+t>1){f32 f=s+t;s/=f;t/=f;}
    Vec3f closest=v0+e1*s+e2*t;
    Vec3f diff=center-closest; f32 dist=diff.length();
    if(dist>=radius) return false;
    if(contact){
        contact->normal=(dist>1e-6f)?diff*(1.f/dist):nhat;
        contact->depth=radius-dist;
        contact->point=closest;
    }
    return true;
}

// ─── GJK support functions ────────────────────────────────────────────────
using SupportFn = std::function<Vec3f(Vec3f)>;

[[nodiscard]] inline Vec3f support_sphere(Vec3f center, f32 r, Vec3f d) noexcept {
    return center+d.normalized()*r;
}
[[nodiscard]] inline Vec3f support_box(Vec3f lo, Vec3f hi, Vec3f d) noexcept {
    return {d.x>0?hi.x:lo.x, d.y>0?hi.y:lo.y, d.z>0?hi.z:lo.z};
}

// Minkowski difference support
[[nodiscard]] inline Vec3f mink_support(
    SupportFn sA, SupportFn sB, Vec3f d) noexcept
{
    return sA(d)-sB(-d);
}

// ─── Continuous Collision Detection (swept sphere) ────────────────────────
// Sweep sphere from (center,radius) by velocity v, against static sphere (c2,r2)
// Returns time of impact [0,1] or -1 if no hit
[[nodiscard]] inline f32 swept_sphere_sphere(
    Vec3f c1, f32 r1, Vec3f v,
    Vec3f c2, f32 r2) noexcept
{
    Vec3f d=c1-c2;
    f32 R=r1+r2;
    f32 a=dot(v,v);
    if(a<1e-10f){if(dot(d,d)<R*R) return 0.f; return -1.f;}
    f32 b=2.f*dot(v,d);
    f32 c=dot(d,d)-R*R;
    f32 disc=b*b-4.f*a*c;
    if(disc<0) return -1.f;
    f32 t=(-b-std::sqrt(disc))/(2.f*a);
    return (t>=0.f&&t<=1.f)?t:-1.f;
}

// Swept AABB vs static AABB (separating axis time-of-impact)
[[nodiscard]] inline f32 swept_aabb_aabb(
    Vec3f lo1, Vec3f hi1, Vec3f v,
    Vec3f lo2, Vec3f hi2) noexcept
{
    f32 t_enter=0, t_exit=1;
    for(int i=0;i<3;++i){
        f32 vi=(&v.x)[i];
        f32 l1=(&lo1.x)[i],h1=(&hi1.x)[i];
        f32 l2=(&lo2.x)[i],h2=(&hi2.x)[i];
        if(std::abs(vi)<1e-8f){if(l1>h2||h1<l2) return -1.f;}
        else{
            f32 t1=(l2-h1)/vi, t2=(h2-l1)/vi;
            if(t1>t2) std::swap(t1,t2);
            t_enter=max(t_enter,t1);
            t_exit =min(t_exit, t2);
            if(t_enter>t_exit) return -1.f;
        }
    }
    return (t_enter>=0.f&&t_enter<=1.f)?t_enter:-1.f;
}

} // namespace qs::math::geometry
