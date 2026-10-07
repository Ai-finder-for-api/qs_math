#pragma once
// qs_math/constraint_solver.hpp — rigid body joints, Baumgarte, sequential impulse (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "quat.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace qs {

// ─── Rigid body state ─────────────────────────────────────────────────────────

struct RigidBody {
    Vec3f pos{};
    Quatf rot{0,0,0,1};
    Vec3f vel{};
    Vec3f ang_vel{};
    f32   inv_mass{1.f};
    Mat3f inv_inertia_world{};  // world-space inverse inertia tensor

    [[nodiscard]] Vec3f velocity_at(const Vec3f& r) const {
        return vel + cross(ang_vel, r);
    }
};

// ─── Contact constraint ───────────────────────────────────────────────────────

struct ContactConstraint {
    u32   body_a{}, body_b{};
    Vec3f normal{};      // from a to b
    Vec3f ra{}, rb{};    // contact point relative to COM
    f32   penetration{};
    f32   restitution{0.3f};
    f32   friction{0.4f};

    f32   lambda_n{0.f}; // accumulated normal impulse
    f32   lambda_t1{0.f}, lambda_t2{0.f}; // tangent impulses
};

// ─── Effective mass ───────────────────────────────────────────────────────────

[[nodiscard]] inline f32 compute_effective_mass(
    const RigidBody& a, const RigidBody& b,
    const Vec3f& ra, const Vec3f& rb, const Vec3f& n)
{
    Vec3f raxn = cross(ra, n);
    Vec3f rbxn = cross(rb, n);
    f32 k = a.inv_mass + b.inv_mass
          + dot(raxn, a.inv_inertia_world * raxn)
          + dot(rbxn, b.inv_inertia_world * rbxn);
    return k > 1e-10f ? 1.f/k : 0.f;
}

// ─── Sequential impulse solver ────────────────────────────────────────────────

inline void solve_contact_impulse(
    RigidBody& a, RigidBody& b, ContactConstraint& c,
    f32 dt, f32 baumgarte=0.2f, f32 slop=0.005f)
{
    Vec3f rel_vel = b.velocity_at(c.rb) - a.velocity_at(c.ra);
    f32 vn = dot(rel_vel, c.normal);

    // Baumgarte position correction
    f32 bias = -baumgarte/dt * std::min(0.f, -c.penetration + slop);

    // Restitution
    f32 restitution_bias = -c.restitution * std::min(vn, -1.f);
    f32 desired_delta_vn = -(vn + restitution_bias);

    f32 eff_mass = compute_effective_mass(a,b,c.ra,c.rb,c.normal);
    f32 delta_lambda = eff_mass * (desired_delta_vn + bias);

    // Clamp (non-penetration)
    f32 old_lambda = c.lambda_n;
    c.lambda_n = std::max(0.f, c.lambda_n + delta_lambda);
    delta_lambda = c.lambda_n - old_lambda;

    Vec3f impulse = c.normal * delta_lambda;
    a.vel = a.vel - impulse * a.inv_mass;
    b.vel = b.vel + impulse * b.inv_mass;
    a.ang_vel = a.ang_vel - a.inv_inertia_world * cross(c.ra, impulse);
    b.ang_vel = b.ang_vel + b.inv_inertia_world * cross(c.rb, impulse);

    // Friction
    Vec3f tangent1 = rel_vel - c.normal*dot(rel_vel,c.normal);
    f32 tl = length(tangent1);
    if (tl > 1e-6f) {
        tangent1 = tangent1/tl;
        Vec3f tangent2 = cross(c.normal, tangent1);
        for (auto [tang, lamt] : std::array<std::pair<Vec3f,f32*>,2>{
            {{tangent1,&c.lambda_t1},{tangent2,&c.lambda_t2}}}) {
            f32 vt = dot(b.velocity_at(c.rb)-a.velocity_at(c.ra), tang);
            f32 em_t = compute_effective_mass(a,b,c.ra,c.rb,tang);
            f32 dl_t = em_t * (-vt);
            f32 max_t = c.friction * c.lambda_n;
            f32 old_t = *lamt;
            *lamt = std::clamp(*lamt + dl_t, -max_t, max_t);
            dl_t = *lamt - old_t;
            Vec3f fimp = tang * dl_t;
            a.vel = a.vel - fimp*a.inv_mass;
            b.vel = b.vel + fimp*b.inv_mass;
            a.ang_vel = a.ang_vel - a.inv_inertia_world*cross(c.ra,fimp);
            b.ang_vel = b.ang_vel + b.inv_inertia_world*cross(c.rb,fimp);
        }
    }
}

// ─── Joint constraints ────────────────────────────────────────────────────────

struct BallSocketJoint {
    u32   body_a{}, body_b{};
    Vec3f anchor_a{}, anchor_b{}; // local space
    f32   baumgarte{0.1f};

    void solve(RigidBody& a, RigidBody& b, f32 dt) {
        Vec3f ra = a.rot.rotate(anchor_a);
        Vec3f rb = b.rot.rotate(anchor_b);
        Vec3f err = (a.pos+ra) - (b.pos+rb);
        Vec3f bias = -baumgarte/dt * err;
        // Per-axis: solve 3 normal constraints
        for (u32 ax=0;ax<3;++ax) {
            Vec3f n{ax==0?1.f:0.f, ax==1?1.f:0.f, ax==2?1.f:0.f};
            f32 eff = compute_effective_mass(a,b,ra,rb,n);
            f32 vrel = dot(b.velocity_at(rb)-a.velocity_at(ra), n);
            f32 dl = eff*(-vrel + bias[ax]);
            Vec3f imp = n*dl;
            a.vel = a.vel - imp*a.inv_mass;
            b.vel = b.vel + imp*b.inv_mass;
            a.ang_vel = a.ang_vel - a.inv_inertia_world*cross(ra,imp);
            b.ang_vel = b.ang_vel + b.inv_inertia_world*cross(rb,imp);
        }
    }
};

struct HingeJoint {
    u32   body_a{}, body_b{};
    Vec3f anchor_a{}, anchor_b{};
    Vec3f axis_a{0,1,0};  // local hinge axis on a
    f32   min_angle{-k_pi};
    f32   max_angle{ k_pi};
    f32   baumgarte{0.1f};

    void solve(RigidBody& a, RigidBody& b, f32 dt) {
        // Ball-socket position constraint
        Vec3f ra = a.rot.rotate(anchor_a);
        Vec3f rb = b.rot.rotate(anchor_b);
        Vec3f err = (a.pos+ra)-(b.pos+rb);
        Vec3f bias = -baumgarte/dt * err;
        for (u32 ax=0;ax<3;++ax) {
            Vec3f n{ax==0?1.f:0.f,ax==1?1.f:0.f,ax==2?1.f:0.f};
            f32 eff = compute_effective_mass(a,b,ra,rb,n);
            f32 vrel = dot(b.velocity_at(rb)-a.velocity_at(ra),n);
            Vec3f imp = n*(eff*(-vrel+bias[ax]));
            a.vel=a.vel-imp*a.inv_mass; b.vel=b.vel+imp*b.inv_mass;
            a.ang_vel=a.ang_vel-a.inv_inertia_world*cross(ra,imp);
            b.ang_vel=b.ang_vel+b.inv_inertia_world*cross(rb,imp);
        }
        // Angular constraint: keep perpendicular axes aligned
        Vec3f world_axis = a.rot.rotate(axis_a);
        Vec3f up = (std::abs(world_axis.y)<0.9f)?Vec3f{0,1,0}:Vec3f{1,0,0};
        Vec3f perp1 = normalize(cross(world_axis, up));
        Vec3f perp2 = cross(world_axis, perp1);
        Vec3f b_axis = b.rot.rotate(axis_a);
        for (Vec3f perp : {perp1, perp2}) {
            f32 ang_err = dot(b_axis, perp);
            f32 ang_bias = -baumgarte/dt * ang_err;
            f32 iA = dot(perp, a.inv_inertia_world*perp);
            f32 iB = dot(perp, b.inv_inertia_world*perp);
            f32 eff = (iA+iB)>1e-10f?1.f/(iA+iB):0.f;
            f32 ang_vrel = dot(b.ang_vel-a.ang_vel, perp);
            f32 dl = eff*(-ang_vrel+ang_bias);
            Vec3f aimp = perp*dl;
            a.ang_vel = a.ang_vel - a.inv_inertia_world*aimp;
            b.ang_vel = b.ang_vel + b.inv_inertia_world*aimp;
        }
    }
};

// ─── Position-level correction (XPBD-style) ──────────────────────────────────

inline void position_correction(
    RigidBody& a, RigidBody& b,
    const Vec3f& ra, const Vec3f& rb,
    const Vec3f& correction, f32 compliance=0.f)
{
    f32 len = length(correction);
    if (len < 1e-8f) return;
    Vec3f n = correction/len;
    f32 k = a.inv_mass + b.inv_mass
          + dot(cross(ra,n), a.inv_inertia_world*cross(ra,n))
          + dot(cross(rb,n), b.inv_inertia_world*cross(rb,n))
          + compliance;
    if (k < 1e-10f) return;
    f32 lambda = len/k;
    Vec3f p = n*lambda;
    a.pos = a.pos - p*a.inv_mass;
    b.pos = b.pos + p*b.inv_mass;
    // Angular correction
    Vec3f dq_a = a.inv_inertia_world * cross(ra, -p);
    Vec3f dq_b = b.inv_inertia_world * cross(rb,  p);
    Quatf qa{dq_a.x,dq_a.y,dq_a.z,0.f};
    Quatf qb{dq_b.x,dq_b.y,dq_b.z,0.f};
    a.rot = normalize(a.rot + qa*a.rot*0.5f);
    b.rot = normalize(b.rot + qb*b.rot*0.5f);
}

} // namespace qs
