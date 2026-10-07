
// qs_math/physics/physics_more_impl.cpp (C++20)
// More physics: constraint solvers, soft body, cloth, fluid SPH.
#include "include/qs_math_physics.hpp"
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
namespace qs::math::physics {

// ─── PBD cloth simulation ─────────────────────────────────────────────────────
struct ClothParticle {
    Vec3<f64> pos{}, prev_pos{}, vel{};
    f64 inv_mass{1};
    bool pinned{false};
};

struct ClothConstraint {
    i32 a{0}, b{0};
    f64 rest_len{0};
    f64 stiffness{1};
};

struct Cloth {
    std::vector<ClothParticle> particles;
    std::vector<ClothConstraint> constraints;
    i32 solver_iters{10};
    Vec3<f64> gravity{0,-9.81,0};
    f64 damping{0.99};

    void step(f64 dt) noexcept {
        // Predict positions
        for (auto& p : particles) {
            if (p.pinned) continue;
            p.vel = (p.vel + gravity*dt) * damping;
            p.prev_pos = p.pos;
            p.pos = p.pos + p.vel * dt;
        }
        // Solve constraints
        for (i32 iter = 0; iter < solver_iters; ++iter) {
            for (auto& c : constraints) {
                auto& pa = particles[c.a];
                auto& pb = particles[c.b];
                const Vec3<f64> d = pb.pos - pa.pos;
                const f64 len = d.length();
                if (len < 1e-10) continue;
                const f64 diff = (len - c.rest_len) / len;
                const f64 w_sum = pa.inv_mass + pb.inv_mass;
                if (w_sum < 1e-14) continue;
                const Vec3<f64> corr = d * (diff * c.stiffness / w_sum);
                if (!pa.pinned) pa.pos = pa.pos + corr *  pa.inv_mass;
                if (!pb.pinned) pb.pos = pb.pos - corr *  pb.inv_mass;
            }
        }
        // Update velocities
        for (auto& p : particles) {
            if (p.pinned) continue;
            p.vel = (p.pos - p.prev_pos) * (1.0/dt);
        }
    }

    void add_wind(Vec3<f64> wind, const i32* faces, i32 nf) noexcept {
        for (i32 t = 0; t < nf; ++t) {
            auto& pa = particles[faces[t*3]];
            auto& pb = particles[faces[t*3+1]];
            auto& pc = particles[faces[t*3+2]];
            const Vec3<f64> n = (pb.pos-pa.pos).cross(pc.pos-pa.pos).normalized();
            const f64 area = (pb.pos-pa.pos).cross(pc.pos-pa.pos).length()*0.5;
            const Vec3<f64> force = n * n.dot(wind) * area * 0.33;
            pa.vel = pa.vel + force*pa.inv_mass;
            pb.vel = pb.vel + force*pb.inv_mass;
            pc.vel = pc.vel + force*pc.inv_mass;
        }
    }
};

// ─── SPH fluid (smoothed particle hydrodynamics) ─────────────────────────────
struct SPHParticle {
    Vec3<f64> pos{}, vel{}, force{};
    f64 density{0}, pressure{0};
    f64 mass{0.01};
};

struct SPHSolver {
    f64 h{0.1};       // smoothing radius
    f64 rho0{1000};   // rest density
    f64 k{500};       // pressure stiffness
    f64 mu{0.1};      // viscosity
    Vec3<f64> gravity{0,-9.81,0};

    // Poly6 kernel
    [[nodiscard]] f64 W_poly6(f64 r) const noexcept {
        if (r >= h) return 0;
        const f64 d = h*h - r*r;
        return 315.0/(64*M_PI*std::pow(h,9)) * d*d*d;
    }
    // Spiky gradient kernel magnitude
    [[nodiscard]] f64 W_spiky_grad(f64 r) const noexcept {
        if (r < 1e-10 || r >= h) return 0;
        const f64 d = h - r;
        return -45.0/(M_PI*std::pow(h,6)) * d*d;
    }
    // Viscosity Laplacian kernel
    [[nodiscard]] f64 W_visc_lap(f64 r) const noexcept {
        if (r >= h) return 0;
        return 45.0/(M_PI*std::pow(h,6)) * (h - r);
    }

    void compute_density(std::vector<SPHParticle>& p) const noexcept {
        for (auto& pi : p) {
            pi.density = 0;
            for (auto& pj : p) {
                const f64 r = (pi.pos - pj.pos).length();
                pi.density += pj.mass * W_poly6(r);
            }
            pi.pressure = k * (pi.density - rho0);
        }
    }

    void compute_forces(std::vector<SPHParticle>& parts) const noexcept {
        for (i32 i = 0; i < (i32)parts.size(); ++i) {
            Vec3<f64> fp{}, fv{};
            auto& pi = parts[i];
            for (i32 j = 0; j < (i32)parts.size(); ++j) {
                if (i == j) continue;
                auto& pj = parts[j];
                const Vec3<f64> rij = pi.pos - pj.pos;
                const f64 r = rij.length();
                if (r < 1e-10 || r >= h) continue;
                const Vec3<f64> rn = rij / r;
                // Pressure force
                const f64 pterm = -(pi.pressure + pj.pressure) / (2*pj.density+1e-14);
                fp = fp + rn * (pterm * pj.mass * W_spiky_grad(r));
                // Viscosity force
                fv = fv + (pj.vel - pi.vel) * (mu * pj.mass / (pj.density+1e-14) * W_visc_lap(r));
            }
            pi.force = (fp + fv) / (pi.density + 1e-14) + gravity;
        }
    }

    void step(std::vector<SPHParticle>& parts, f64 dt) noexcept {
        compute_density(parts);
        compute_forces(parts);
        for (auto& p : parts) {
            p.vel = p.vel + p.force * dt;
            p.pos = p.pos + p.vel * dt;
            // Simple boundary reflection
            for (i32 ax = 0; ax < 3; ++ax) {
                f64& x = (&p.pos.x)[ax]; f64& v = (&p.vel.x)[ax];
                if (x < 0) { x = -x; v = -v * 0.5; }
                if (x > 1) { x = 2-x; v = -v * 0.5; }
            }
        }
    }
};

// ─── Constraint solver (Gauss-Seidel) for rigid bodies ───────────────────────
struct ContactConstraint {
    Vec3<f64> point, normal;
    f64 penetration{0};
    f64 restitution{0.3}, friction{0.5};
    f64 lambda_n{0}, lambda_t{0}; // accumulated impulses
};

void solve_contact(RigidBody& a, RigidBody& b,
    ContactConstraint& c, f64 dt) noexcept
{
    const Vec3<f64> ra = c.point - a.position;
    const Vec3<f64> rb = c.point - b.position;
    const Vec3<f64> va = a.point_velocity(c.point);
    const Vec3<f64> vb = b.point_velocity(c.point);
    const f64 vrel_n  = c.normal.dot(vb - va);

    // Compute effective mass
    const Vec3<f64> ra_n = ra.cross(c.normal);
    const Vec3<f64> rb_n = rb.cross(c.normal);
    const f64 K = a.mass_inv + b.mass_inv
        + (a.inertia_inv * ra_n).dot(ra_n)
        + (b.inertia_inv * rb_n).dot(rb_n);
    if (K < 1e-14) return;

    // Baumgarte stabilisation
    const f64 beta  = 0.2;
    const f64 slop  = 0.01;
    const f64 bias  = -beta/dt * std::min(0.0, -c.penetration + slop);
    const f64 j_n   = -(1 + c.restitution) * vrel_n + bias;
    const f64 delta_lambda = std::max(-c.lambda_n, j_n / K);
    c.lambda_n += delta_lambda;

    const Vec3<f64> impulse = c.normal * delta_lambda;
    a.apply_impulse(-impulse, ra);
    b.apply_impulse( impulse, rb);
}

// ─── Buoyancy force ───────────────────────────────────────────────────────────
[[nodiscard]] Vec3<f64> buoyancy_force(f64 submerged_volume,
    f64 fluid_density, Vec3<f64> gravity) noexcept
{
    return gravity * (-fluid_density * submerged_volume);
}
[[nodiscard]] f64 submerged_box_volume(Vec3<f64> center, Vec3<f64> half,
    f64 water_level) noexcept
{
    const f64 top    = center.y + half.y;
    const f64 bottom = center.y - half.y;
    if (top <= water_level) return 8*half.x*half.y*half.z; // fully submerged
    if (bottom >= water_level) return 0;
    const f64 depth = water_level - bottom;
    return 4*half.x*half.z*depth;
}

// ─── Ragdoll constraint helpers ───────────────────────────────────────────────
struct HingeConstraint {
    Vec3<f64> anchor_a, anchor_b; // in body-local space
    Vec3<f64> axis;               // hinge axis in body-local space of a
    f64 min_angle{-M_PI*0.5}, max_angle{M_PI*0.5};
    f64 stiffness{1000}, damping{50};

    [[nodiscard]] f64 current_angle(const RigidBody& a, const RigidBody& b) const noexcept {
        (void)a;(void)b;
        return 0; // placeholder
    }
    [[nodiscard]] Vec3<f64> torque(const RigidBody& a, const RigidBody& b) const noexcept {
        const f64 angle = current_angle(a, b);
        const f64 clamped = std::max(min_angle, std::min(max_angle, angle));
        const f64 correction = (clamped - angle) * stiffness;
        return axis * correction;
    }
};

} // namespace qs::math::physics
