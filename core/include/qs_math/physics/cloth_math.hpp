#pragma once
// qs_math/cloth_math.hpp — Position-based dynamics, XPBD constraints, bending stiffness (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>

namespace qs {

// ─── Particles ────────────────────────────────────────────────────────────────

struct ClothParticle {
    Vec3f pos{};
    Vec3f prev_pos{};
    Vec3f vel{};
    f32   inv_mass{1.f};   // 0 = pinned
};

// ─── Constraint types ─────────────────────────────────────────────────────────

struct DistanceConstraint {
    u32 i{}, j{};
    f32 rest_length{};
    f32 compliance{0.f}; // 0 = rigid (XPBD: 1/stiffness * dt^2)
};

struct BendConstraint {
    u32 i{}, j{}, k{}, l{}; // two triangles sharing edge j-k
    f32 rest_angle{};
    f32 compliance{1e-4f};
};

struct IsometricBendConstraint {
    u32 verts[4]{};  // p0,p1 = shared edge; p2,p3 = opposite
    f32 Q[4][4]{};   // precomputed stiffness matrix
    f32 compliance{1e-5f};
};

// ─── PBD constraint projection ────────────────────────────────────────────────

inline void solve_distance_constraint(
    ClothParticle& a, ClothParticle& b, const DistanceConstraint& c)
{
    Vec3f d = b.pos - a.pos;
    f32 len = length(d);
    if (len < 1e-8f) return;
    f32 C = len - c.rest_length;
    f32 denom = a.inv_mass + b.inv_mass;
    if (denom < 1e-10f) return;
    Vec3f corr = d * (C / (len * denom));
    a.pos = a.pos + corr *  a.inv_mass;
    b.pos = b.pos + corr * -b.inv_mass;
}

// XPBD distance constraint with Lagrange multiplier
inline void solve_distance_xpbd(
    ClothParticle& a, ClothParticle& b, const DistanceConstraint& c,
    f32& lambda, f32 dt)
{
    Vec3f d = b.pos - a.pos;
    f32 len = length(d);
    if (len < 1e-8f) return;
    Vec3f grad_b = d / len;
    Vec3f grad_a = -grad_b;
    f32 C = len - c.rest_length;
    f32 alpha_tilde = c.compliance / (dt*dt);
    f32 denom = a.inv_mass*dot(grad_a,grad_a) + b.inv_mass*dot(grad_b,grad_b) + alpha_tilde;
    if (denom < 1e-10f) return;
    f32 delta_lambda = (-C - alpha_tilde*lambda) / denom;
    lambda += delta_lambda;
    a.pos = a.pos + grad_a * (a.inv_mass * delta_lambda);
    b.pos = b.pos + grad_b * (b.inv_mass * delta_lambda);
}

// ─── Dihedral bending constraint (Bridson et al.) ─────────────────────────────

[[nodiscard]] inline f32 dihedral_angle(
    const Vec3f& p0, const Vec3f& p1,
    const Vec3f& p2, const Vec3f& p3)
{
    Vec3f e  = p1 - p0;
    Vec3f n1 = cross(p2 - p0, e);
    Vec3f n2 = cross(e, p3 - p0);
    f32 l1 = length(n1), l2 = length(n2);
    if (l1<1e-8f||l2<1e-8f) return 0.f;
    n1 = n1/l1; n2 = n2/l2;
    f32 c = std::clamp(dot(n1,n2),-1.f,1.f);
    return std::acos(c) * (dot(cross(n1,n2),e)<0.f?-1.f:1.f);
}

inline void solve_bend_constraint(
    ClothParticle& p0, ClothParticle& p1,
    ClothParticle& p2, ClothParticle& p3,
    const BendConstraint& bc)
{
    f32 angle = dihedral_angle(p0.pos,p1.pos,p2.pos,p3.pos);
    f32 C = angle - bc.rest_angle;
    if (std::abs(C) < 1e-6f) return;

    Vec3f e = p1.pos - p0.pos;
    f32 el = length(e);
    if (el < 1e-8f) return;

    Vec3f n1 = cross(p2.pos-p0.pos, e);
    Vec3f n2 = cross(e, p3.pos-p0.pos);
    f32 l1=length(n1), l2=length(n2);
    if (l1<1e-8f||l2<1e-8f) return;

    Vec3f q2 = (cross(p3.pos-p0.pos,e) + cross(n2,e/el)) / l2 - (cross(p2.pos-p0.pos,e) + cross(n1,e/el)) / l1;
    Vec3f q3 = (cross(n1,e/el) + cross(n2, p0.pos-p1.pos)) / l2;
    Vec3f q0 = -q2 - q3;
    Vec3f q1 = q2;  // simplified

    f32 denom = (p0.inv_mass*dot(q0,q0) + p1.inv_mass*dot(q1,q1)
               + p2.inv_mass*dot(q2,q2) + p3.inv_mass*dot(q3,q3));
    if (denom < 1e-10f) return;
    f32 s = -std::sqrt(std::max(0.f,1.f-std::cos(angle)*std::cos(angle)))*C / denom;
    p0.pos = p0.pos + q0*(p0.inv_mass*s);
    p1.pos = p1.pos + q1*(p1.inv_mass*s);
    p2.pos = p2.pos + q2*(p2.inv_mass*s);
    p3.pos = p3.pos + q3*(p3.inv_mass*s);
}

// ─── Volume constraint (pressure) ─────────────────────────────────────────────

[[nodiscard]] inline f32 cloth_volume(std::span<const ClothParticle> pts,
                                       std::span<const std::array<u32,3>> tris)
{
    f32 vol = 0.f;
    for (const auto& t : tris) {
        const Vec3f& a = pts[t[0]].pos;
        const Vec3f& b = pts[t[1]].pos;
        const Vec3f& c = pts[t[2]].pos;
        vol += dot(a, cross(b,c)) / 6.f;
    }
    return std::abs(vol);
}

// ─── Self-collision spatial hash ──────────────────────────────────────────────

[[nodiscard]] inline u64 spatial_hash(i32 x, i32 y, i32 z) {
    u64 ux=u64(x+100000), uy=u64(y+100000), uz=u64(z+100000);
    return (ux*73856093ULL) ^ (uy*19349663ULL) ^ (uz*83492791ULL);
}

struct ClothSelfCollision {
    f32 thickness{0.01f};
    f32 cell_size{0.05f};

    [[nodiscard]] std::array<i32,3> to_cell(const Vec3f& p) const {
        return {i32(std::floor(p.x/cell_size)),
                i32(std::floor(p.y/cell_size)),
                i32(std::floor(p.z/cell_size))};
    }

    inline void resolve_particle_pair(
        ClothParticle& a, ClothParticle& b, f32 restitution=0.5f) const
    {
        Vec3f d = b.pos - a.pos;
        f32 dist = length(d);
        if (dist < thickness && dist > 1e-8f) {
            Vec3f n = d / dist;
            f32 pen = thickness - dist;
            f32 total = a.inv_mass + b.inv_mass;
            if (total < 1e-10f) return;
            a.pos = a.pos - n*(pen * a.inv_mass / total);
            b.pos = b.pos + n*(pen * b.inv_mass / total);
        }
    }
};

// ─── Cloth sim step ───────────────────────────────────────────────────────────

struct ClothSimParams {
    Vec3f gravity{0,-9.81f,0};
    f32   dt{1.f/60.f};
    u32   solver_iterations{10};
    f32   damping{0.98f};
    f32   air_resistance{0.01f};
};

inline void cloth_integrate(
    std::span<ClothParticle> particles, const ClothSimParams& p)
{
    for (auto& pt : particles) {
        if (pt.inv_mass == 0.f) continue;
        pt.vel = pt.vel + p.gravity * p.dt;
        pt.vel = pt.vel * p.damping;
        pt.prev_pos = pt.pos;
        pt.pos = pt.pos + pt.vel * p.dt;
    }
}

inline void cloth_update_velocities(
    std::span<ClothParticle> particles, f32 dt)
{
    for (auto& pt : particles) {
        if (pt.inv_mass == 0.f) continue;
        pt.vel = (pt.pos - pt.prev_pos) / dt;
    }
}

// ─── Stiffness fabric presets ─────────────────────────────────────────────────

struct FabricPreset {
    f32 stretch_compliance{};
    f32 shear_compliance{};
    f32 bend_compliance{};
    const char* name{};
};

static constexpr FabricPreset k_fabric_silk    {1e-7f, 1e-6f, 5e-4f, "silk"};
static constexpr FabricPreset k_fabric_cotton  {1e-8f, 1e-7f, 1e-4f, "cotton"};
static constexpr FabricPreset k_fabric_denim   {1e-9f, 1e-8f, 1e-5f, "denim"};
static constexpr FabricPreset k_fabric_leather {1e-10f,1e-9f, 1e-6f, "leather"};
static constexpr FabricPreset k_fabric_rubber  {1e-6f, 1e-5f, 1e-3f, "rubber"};

// ─── Wind force ───────────────────────────────────────────────────────────────

[[nodiscard]] inline Vec3f cloth_wind_force(
    const Vec3f& wind_vel, const Vec3f& face_normal,
    f32 area, f32 drag_coeff=1.2f)
{
    f32 proj = dot(wind_vel, face_normal);
    return face_normal * (proj * area * drag_coeff);
}

// ─── Cloth tearing ────────────────────────────────────────────────────────────

struct TearConstraint {
    DistanceConstraint dc;
    f32 max_stretch{1.3f}; // ratio of current/rest length before tearing

    [[nodiscard]] bool should_tear(const ClothParticle& a, const ClothParticle& b) const {
        return length(b.pos - a.pos) > dc.rest_length * max_stretch;
    }
};

} // namespace qs
