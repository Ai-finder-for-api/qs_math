
#pragma once
// qs_math/physics/include/qs_math_physics.hpp (C++20)
// Extended physics math: rigid-body kinematics, springs, fluid, optics, relativity.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/mat3.hpp>
#include <qs_math/quaternion/quat.hpp>
#include <cmath>
#include <array>
namespace qs::math::physics {
using namespace qs::math;
// ─── Rigid body state ────────────────────────────────────────────────────────
struct RigidBody {
    Vec3<f64> position{},  velocity{},  force{};
    Vec3<f64> angular_vel{}, torque{};
    Quat<f64> orientation{1,0,0,0};
    Mat3<f64> inertia_inv{};
    f64 mass_inv{1};
    void integrate(f64 dt) noexcept;
    void apply_force(Vec3<f64> f, Vec3<f64> r={}) noexcept;
    void apply_impulse(Vec3<f64> j, Vec3<f64> r={}) noexcept;
    [[nodiscard]] Vec3<f64> world_point(Vec3<f64> local) const noexcept;
    [[nodiscard]] Vec3<f64> point_velocity(Vec3<f64> world_pt) const noexcept;
};
// ─── Spring-damper ────────────────────────────────────────────────────────────
struct SpringDamper {
    f64 k{100}, c{10}, rest_len{1};
    [[nodiscard]] f64 force(f64 length, f64 vel) const noexcept {
        return -k*(length-rest_len)-c*vel;
    }
};
// ─── Projectile motion ───────────────────────────────────────────────────────
struct Projectile {
    Vec3<f64> pos{}, vel{};
    f64 drag_coef{0}, mass{1};
    void step(f64 dt, Vec3<f64> gravity={0,-9.81,0}) noexcept;
    [[nodiscard]] Vec3<f64> pos_at(f64 t, Vec3<f64> g={0,-9.81,0}) const noexcept {
        return pos + vel*t + g*(0.5*t*t);
    }
    [[nodiscard]] f64 range_flat(f64 angle_rad, f64 speed, f64 g=9.81) const noexcept {
        return speed*speed*std::sin(2*angle_rad)/g;
    }
    [[nodiscard]] f64 time_of_flight(f64 angle_rad, f64 speed, f64 g=9.81) const noexcept {
        return 2*speed*std::sin(angle_rad)/g;
    }
};
// ─── Orbital mechanics ───────────────────────────────────────────────────────
constexpr f64 G_CONST = 6.674e-11;
struct Orbit {
    f64 semi_major{0}, eccentricity{0}, inclination{0};
    f64 period(f64 mu) const noexcept {
        return 2*M_PI*std::sqrt(semi_major*semi_major*semi_major/mu);
    }
    Vec3<f64> state_at_anomaly(f64 true_anomaly, f64 mu) const noexcept;
};
// ─── Fluid dynamics helpers ───────────────────────────────────────────────────
[[nodiscard]] inline f64 reynolds_number(f64 rho,f64 v,f64 L,f64 mu) noexcept { return rho*v*L/mu; }
[[nodiscard]] inline f64 mach_number(f64 v,f64 c) noexcept { return v/c; }
[[nodiscard]] inline f64 bernoulli_velocity(f64 P1,f64 P2,f64 rho) noexcept {
    return std::sqrt(2*std::max(0.0,(P1-P2))/rho);
}
[[nodiscard]] inline f64 drag_coefficient_sphere(f64 Re) noexcept {
    if(Re<1)return 24.0/Re;
    if(Re<1000)return 24.0/Re*(1+0.15*std::pow(Re,0.687));
    return 0.44;
}
// ─── Electromagnetic ──────────────────────────────────────────────────────────
[[nodiscard]] inline Vec3<f64> lorentz_force(Vec3<f64> E,Vec3<f64> B,
    Vec3<f64> v,f64 q) noexcept { return (E + v.cross(B))*q; }
[[nodiscard]] inline f64 larmor_radius(f64 m,f64 v,f64 q,f64 B) noexcept { return m*v/(std::abs(q)*B); }
// ─── Optics ───────────────────────────────────────────────────────────────────
[[nodiscard]] Vec3<f64> snell_refract(Vec3<f64> d,Vec3<f64> n,f64 n1,f64 n2) noexcept;
[[nodiscard]] Vec3<f64> fresnel_reflect(Vec3<f64> d,Vec3<f64> n) noexcept;
[[nodiscard]] f64 fresnel_reflectance(f64 cos_theta_i,f64 n1,f64 n2) noexcept;
// ─── Special relativity ───────────────────────────────────────────────────────
[[nodiscard]] inline f64 lorentz_factor(f64 v,f64 c=2.998e8) noexcept {
    const f64 b=v/c; return 1.0/std::sqrt(1-b*b);
}
[[nodiscard]] inline f64 relativistic_mass(f64 m0,f64 v) noexcept { return m0*lorentz_factor(v); }
[[nodiscard]] inline f64 relativistic_energy(f64 m0,f64 v,f64 c=2.998e8) noexcept {
    return relativistic_mass(m0,v)*c*c;
}
[[nodiscard]] inline Vec3<f64> velocity_addition(Vec3<f64> u,Vec3<f64> v,f64 c=2.998e8) noexcept {
    const f64 v2=v.dot(v),c2=c*c;
    const f64 gamma_v=lorentz_factor(std::sqrt(v2),c);
    const f64 uv=u.dot(v)/c2;
    const f64 denom=1+uv;
    return (u*(1.0/gamma_v)+v+v*(gamma_v*uv/(gamma_v+1)))*(1.0/denom);
}
} // namespace qs::math::physics
