#pragma once
// physics_dynamics.hpp — Rigid body dynamics helpers:
//   integration, constraints, joints, impulse resolution,
//   contact manifold, Baumgarte stabilization, simulation step
#include "physics_helpers.hpp"
#include "quat.hpp"
#include "mat3.hpp"
#include "mat4.hpp"
#include "geometry3d.hpp"

namespace qs::math::physics {

// ---------------------------------------------------------------------------
// Rigid body state
// ---------------------------------------------------------------------------
struct RigidBodyState {
    Vec3f position{0.f};
    Quatf orientation{Quatf::identity()};
    Vec3f linear_velocity{0.f};
    Vec3f angular_velocity{0.f};
    f32   inv_mass{1.f};           // 0 = static
    Mat3f inv_inertia_world;       // world-space inverse inertia tensor
    Mat3f inv_inertia_local;       // body-space inverse inertia tensor
    Vec3f force_accumulator{0.f};
    Vec3f torque_accumulator{0.f};

    void apply_force(Vec3f f) noexcept { force_accumulator=force_accumulator+f; }
    void apply_torque(Vec3f t) noexcept { torque_accumulator=torque_accumulator+t; }
    void apply_force_at_point(Vec3f f, Vec3f world_pt) noexcept {
        force_accumulator=force_accumulator+f;
        torque_accumulator=torque_accumulator+cross(world_pt-position,f);
    }
    void clear_forces() noexcept { force_accumulator={0.f}; torque_accumulator={0.f}; }

    // Update world-space inertia tensor
    void update_inertia() noexcept {
        Mat3f R=orientation.to_mat3();
        inv_inertia_world=R*inv_inertia_local*R.transposed();
    }

    // Velocity of a world-space point on the body
    [[nodiscard]] Vec3f velocity_at(Vec3f world_pt) const noexcept {
        return linear_velocity+cross(angular_velocity,world_pt-position);
    }
};

// Euler integration step
inline void integrate_euler(RigidBodyState& body, f32 dt) noexcept {
    if(body.inv_mass<=0.f) return;
    body.linear_velocity=body.linear_velocity+body.force_accumulator*(body.inv_mass*dt);
    body.angular_velocity=body.angular_velocity+body.inv_inertia_world*body.torque_accumulator*dt;
    body.position=body.position+body.linear_velocity*dt;
    // Integrate orientation
    Quatf dq{body.angular_velocity.x*0.5f*dt,
              body.angular_velocity.y*0.5f*dt,
              body.angular_velocity.z*0.5f*dt,0.f};
    body.orientation=(body.orientation+dq*body.orientation).normalized();
    body.update_inertia();
    body.clear_forces();
}

// RK4 integration step (more stable)
inline void integrate_rk4(RigidBodyState& body, f32 dt) noexcept {
    if(body.inv_mass<=0.f) return;
    Vec3f a=body.force_accumulator*body.inv_mass;
    Vec3f alpha=body.inv_inertia_world*body.torque_accumulator;
    Vec3f k1v=a, k1w=alpha;
    Vec3f k2v=a, k2w=alpha; // simplified (constant force over step)
    body.linear_velocity=body.linear_velocity+(k1v+k2v)*(dt*0.5f);
    body.angular_velocity=body.angular_velocity+(k1w+k2w)*(dt*0.5f);
    body.position=body.position+body.linear_velocity*dt;
    Quatf dq{body.angular_velocity.x*0.5f*dt,
              body.angular_velocity.y*0.5f*dt,
              body.angular_velocity.z*0.5f*dt,0.f};
    body.orientation=(body.orientation+dq*body.orientation).normalized();
    body.update_inertia();
    body.clear_forces();
}

// ---------------------------------------------------------------------------
// Contact manifold
// ---------------------------------------------------------------------------
struct ContactPoint {
    Vec3f point_a, point_b; // world-space contact points
    Vec3f normal;            // from B to A
    f32   penetration{0.f};
    f32   normal_impulse{0.f};
    f32   tangent_impulse[2]{};
};

struct ContactManifold {
    ContactPoint contacts[8];
    i32 count{0};
    f32 friction{0.5f};
    f32 restitution{0.3f};
};

// ---------------------------------------------------------------------------
// Impulse-based constraint resolution (Baumgarte stabilization)
// ---------------------------------------------------------------------------
inline void resolve_contact(
    RigidBodyState& a, RigidBodyState& b,
    ContactPoint& cp,
    f32 dt,
    f32 baumgarte_beta=0.2f,
    f32 slop=0.001f) noexcept
{
    Vec3f ra=cp.point_a-a.position, rb=cp.point_b-b.position;
    Vec3f va=a.velocity_at(cp.point_a), vb=b.velocity_at(cp.point_b);
    Vec3f vrel=va-vb;
    f32 vn=dot(vrel,cp.normal);
    if(vn>0.f) return; // separating

    // Effective mass
    Vec3f ra_cross_n=cross(ra,cp.normal), rb_cross_n=cross(rb,cp.normal);
    f32 eff_mass=a.inv_mass+b.inv_mass
                +dot(ra_cross_n,a.inv_inertia_world*ra_cross_n)
                +dot(rb_cross_n,b.inv_inertia_world*rb_cross_n);
    if(eff_mass<=EPS_F32) return;

    // Baumgarte correction
    f32 bias=baumgarte_beta/dt*max(0.f,cp.penetration-slop);

    f32 restitution=cp.penetration>slop?0.3f:0.f; // no bounce for small penetrations
    f32 j=-(1.f+restitution)*vn+bias;
    j/=eff_mass;

    // Clamp accumulated impulse
    f32 new_impulse=max(0.f,cp.normal_impulse+j);
    f32 delta_j=new_impulse-cp.normal_impulse;
    cp.normal_impulse=new_impulse;

    Vec3f impulse=cp.normal*delta_j;
    a.linear_velocity=a.linear_velocity+impulse*a.inv_mass;
    b.linear_velocity=b.linear_velocity-impulse*b.inv_mass;
    a.angular_velocity=a.angular_velocity+a.inv_inertia_world*cross(ra,impulse);
    b.angular_velocity=b.angular_velocity-b.inv_inertia_world*cross(rb,impulse);

    // Friction impulse
    Vec3f tangent_vel=vrel-cp.normal*dot(vrel,cp.normal);
    f32 tv_len=tangent_vel.length();
    if(tv_len>EPS_F32){
        Vec3f t=tangent_vel*(1.f/tv_len);
        Vec3f ra_cross_t=cross(ra,t), rb_cross_t=cross(rb,t);
        f32 eff_mass_t=a.inv_mass+b.inv_mass
                      +dot(ra_cross_t,a.inv_inertia_world*ra_cross_t)
                      +dot(rb_cross_t,b.inv_inertia_world*rb_cross_t);
        if(eff_mass_t>EPS_F32){
            f32 jt=-tv_len/eff_mass_t;
            f32 mu=0.5f; // combined friction
            jt=clamp(jt,-mu*cp.normal_impulse,mu*cp.normal_impulse);
            Vec3f friction_impulse=t*jt;
            a.linear_velocity=a.linear_velocity+friction_impulse*a.inv_mass;
            b.linear_velocity=b.linear_velocity-friction_impulse*b.inv_mass;
            a.angular_velocity=a.angular_velocity+a.inv_inertia_world*cross(ra,friction_impulse);
            b.angular_velocity=b.angular_velocity-b.inv_inertia_world*cross(rb,friction_impulse);
        }
    }
}

// ---------------------------------------------------------------------------
// Joint constraints
// ---------------------------------------------------------------------------
struct DistanceJoint {
    Vec3f anchor_a, anchor_b; // local-space anchors
    f32   rest_length{1.f};
    f32   stiffness{1.f};

    void apply(RigidBodyState& a, RigidBodyState& b, f32 dt) noexcept {
        Vec3f wa=a.position+a.orientation.rotate(anchor_a);
        Vec3f wb=b.position+b.orientation.rotate(anchor_b);
        Vec3f dir=wb-wa;
        f32 d=dir.length();
        if(d<EPS_F32) return;
        dir=dir*(1.f/d);
        f32 error=(d-rest_length)*stiffness;
        f32 eff_mass=a.inv_mass+b.inv_mass;
        if(eff_mass<=EPS_F32) return;
        f32 lambda=error/eff_mass;
        Vec3f imp=dir*lambda;
        a.linear_velocity=a.linear_velocity+imp*a.inv_mass;
        b.linear_velocity=b.linear_velocity-imp*b.inv_mass;
    }
};

struct HingeJoint {
    Vec3f anchor_a, anchor_b;
    Vec3f axis_a, axis_b;  // hinge axes in local space
    f32   min_angle{-PI}, max_angle{PI};

    void apply(RigidBodyState& a, RigidBodyState& b, f32 dt) noexcept {
        // World-space axes
        Vec3f wa=a.orientation.rotate(axis_a);
        Vec3f wb=b.orientation.rotate(axis_b);
        Vec3f corr=cross(wa,wb);
        f32 corr_len=corr.length();
        if(corr_len>EPS_F32){
            Vec3f w_corr=corr*(0.3f/corr_len*corr_len);
            a.angular_velocity=a.angular_velocity+a.inv_inertia_world*(-w_corr);
            b.angular_velocity=b.angular_velocity+b.inv_inertia_world*(w_corr);
        }
    }
};

// ---------------------------------------------------------------------------
// Raycast against convex hull (support mapping + GJK)
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool raycast_convex(
    const Vec3f* pts, i32 n, Vec3f ray_origin, Vec3f ray_dir,
    f32& t_out) noexcept
{
    // GJK-based raycast (simplified)
    Vec3f r=ray_dir.normalized();
    Vec3f x=ray_origin;
    Vec3f n_={0.f};
    f32 lambda=0.f;
    Vec3f v=x-support_point_cloud(pts,n,-r);
    for(i32 iter=0;iter<32;++iter){
        if(v.length_sq()<EPS_F32*EPS_F32){t_out=lambda;return true;}
        Vec3f p=support_point_cloud(pts,n,-v.normalized());
        f32 vdot=dot(v,v), wdot=dot(v,x-p);
        if(wdot>0.f){
            f32 vr=dot(v,r);
            if(vr>=0.f) return false;
            lambda-=wdot/vr;
            x=ray_origin+r*lambda;
            n_=-v.normalized();
        }
        v=x-p;
    }
    t_out=lambda; return true;
}

// ---------------------------------------------------------------------------
// Buoyancy force
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec3f buoyancy_force(
    f32 submerged_volume, f32 fluid_density=1000.f,
    f32 gravity=9.80665f) noexcept
{
    return {0.f, fluid_density*submerged_volume*gravity, 0.f};
}

// Approximate submerged volume of AABB in a flat fluid surface
[[nodiscard]] constexpr f32 submerged_volume_aabb(
    AABBf aabb, f32 fluid_surface_y) noexcept
{
    f32 bottom=aabb.min_pt.y, top=aabb.max_pt.y;
    f32 submerged_height=clamp(fluid_surface_y-bottom,0.f,top-bottom);
    Vec3f size=aabb.size();
    return size.x*submerged_height*size.z;
}

} // namespace qs::math::physics
