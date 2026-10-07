#pragma once
// sweep_tests.hpp — Swept shape intersection tests (CCD):
//   sphere sweep, AABB sweep, capsule sweep, moving sphere vs sphere,
//   linear CCD helpers
#include "geometry3d.hpp"
#include "intersection.hpp"

namespace qs::math {

struct SweepResult {
    bool  hit{false};
    f32   t{0.f};         // time of first contact [0,1]
    Vec3f hit_point;
    Vec3f hit_normal;
};

// Moving sphere vs static sphere (CCD)
[[nodiscard]] inline SweepResult sweep_sphere_sphere(
    Spheref moving, Vec3f vel,  // moving sphere center + velocity
    Spheref stationary) noexcept
{
    f32 sum_r=moving.radius+stationary.radius;
    Vec3f rel=moving.center-stationary.center;
    // Solve |rel + vel*t|^2 = sum_r^2
    f32 a=dot(vel,vel);
    if(a<EPS_LENGTH) return {};
    f32 b=2.f*dot(rel,vel);
    f32 c=dot(rel,rel)-sum_r*sum_r;
    f32 disc=b*b-4.f*a*c;
    if(disc<0.f) return {};
    f32 t=(-b-sqrt(disc))/(2.f*a);
    if(t<0.f||t>1.f) return {};
    Vec3f hit_center=moving.center+vel*t;
    Vec3f normal=(hit_center-stationary.center)*(1.f/sum_r);
    return {true,t,hit_center+normal*(-moving.radius),normal};
}

// Moving sphere vs static plane
[[nodiscard]] inline SweepResult sweep_sphere_plane(
    Spheref sphere, Vec3f vel, Planef plane) noexcept
{
    f32 d=plane.signed_distance(sphere.center)-sphere.radius;
    f32 vn=dot(plane.normal,vel);
    if(abs(vn)<EPS_LENGTH) return {};
    f32 t=-d/vn;
    if(t<0.f||t>1.f) return {};
    Vec3f hit=sphere.center+vel*t;
    return {true,t,hit-plane.normal*sphere.radius,plane.normal};
}

// Moving AABB vs static AABB (separating axis / entry-exit)
[[nodiscard]] inline SweepResult sweep_aabb_aabb(
    AABBf moving, Vec3f vel, AABBf stationary) noexcept
{
    Vec3f inv_vel;
    inv_vel.x=abs(vel.x)>EPS_LENGTH?1.f/vel.x:F32_INF;
    inv_vel.y=abs(vel.y)>EPS_LENGTH?1.f/vel.y:F32_INF;
    inv_vel.z=abs(vel.z)>EPS_LENGTH?1.f/vel.z:F32_INF;

    Vec3f t_entry=(stationary.min_pt-moving.max_pt)*Vec3f{inv_vel.x,inv_vel.y,inv_vel.z};
    Vec3f t_exit =(stationary.max_pt-moving.min_pt)*Vec3f{inv_vel.x,inv_vel.y,inv_vel.z};

    // Swap if negative
    if(t_entry.x>t_exit.x) { f32 tmp=t_entry.x; t_entry.x=t_exit.x; t_exit.x=tmp; }
    if(t_entry.y>t_exit.y) { f32 tmp=t_entry.y; t_entry.y=t_exit.y; t_exit.y=tmp; }
    if(t_entry.z>t_exit.z) { f32 tmp=t_entry.z; t_entry.z=t_exit.z; t_exit.z=tmp; }

    f32 t_in  = max(max(t_entry.x,t_entry.y),t_entry.z);
    f32 t_out = min(min(t_exit.x, t_exit.y), t_exit.z);

    if(t_in>t_out||t_out<0.f||t_in>1.f) return {};
    f32 t=max(t_in,0.f);

    // Compute normal from axis with largest entry time
    Vec3f normal{0.f};
    if(t_entry.x>=t_entry.y&&t_entry.x>=t_entry.z) normal.x=vel.x<0.f?1.f:-1.f;
    else if(t_entry.y>=t_entry.z) normal.y=vel.y<0.f?1.f:-1.f;
    else normal.z=vel.z<0.f?1.f:-1.f;

    AABBf hit_box{moving.min_pt+vel*t, moving.max_pt+vel*t};
    return {true,t,hit_box.center(),normal};
}

// Moving capsule vs static sphere
[[nodiscard]] inline SweepResult sweep_capsule_sphere(
    Capsulef cap, Vec3f vel, Spheref sphere) noexcept
{
    // Expand sphere radius by capsule radius, then sweep segment
    Spheref expanded{sphere.center, sphere.radius+cap.radius};
    // Find closest approach of swept segment midpoint
    Vec3f mid=(cap.a+cap.b)*0.5f;
    return sweep_sphere_sphere({mid,cap.radius+(cap.a-cap.b).length()*0.5f},vel,expanded);
}

// Time of impact for continuous collision (linear interpolation helper)
[[nodiscard]] inline f32 find_toi(
    bool (*test_fn)(void*,void*,void*),
    void* ctx_a, void* ctx_b, void* ctx,
    f32 t_lo=0.f, f32 t_hi=1.f, i32 iterations=8) noexcept
{
    for(i32 i=0;i<iterations;++i){
        f32 t_mid=(t_lo+t_hi)*0.5f;
        if(test_fn(ctx_a,ctx_b,ctx)) t_hi=t_mid; else t_lo=t_mid;
    }
    return (t_lo+t_hi)*0.5f;
}

} // namespace qs::math
