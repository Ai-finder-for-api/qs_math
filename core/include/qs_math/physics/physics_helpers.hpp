#pragma once
// qs_math/physics_helpers.hpp — Inertia tensors, mass, spring, damping, velocity helpers
#include "vec3.hpp"
#include "mat3.hpp"
#include "scalar_math.hpp"

namespace qs::math::physics {

// ---------------------------------------------------------------------------
// Inertia tensors (solid bodies)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Mat3f inertia_sphere(f32 mass, f32 radius) noexcept {
    f32 i=2.f/5.f*mass*radius*radius;
    return Mat3f::scale(Vec3f{i,i,i});
}

[[nodiscard]] constexpr Mat3f inertia_box(f32 mass, Vec3f size) noexcept {
    f32 m12=mass/12.f;
    return Mat3f::scale({m12*(size.y*size.y+size.z*size.z),
                         m12*(size.x*size.x+size.z*size.z),
                         m12*(size.x*size.x+size.y*size.y)});
}

[[nodiscard]] constexpr Mat3f inertia_cylinder(f32 mass, f32 radius, f32 height) noexcept {
    f32 ih=mass*(3.f*radius*radius+height*height)/12.f;
    f32 iz=mass*radius*radius*0.5f;
    return Mat3f::scale({ih,ih,iz});
}

[[nodiscard]] constexpr Mat3f inertia_capsule(f32 mass, f32 radius, f32 cylinder_height) noexcept {
    // Approximate: hemisphere masses + cylinder
    f32 r2=radius*radius, h=cylinder_height;
    f32 m_cyl=mass*(h/(h+4.f/3.f*radius)), m_hemi=mass-m_cyl;
    f32 ix=m_cyl*(3.f*r2+h*h)/12.f+m_hemi*(2.f*r2/5.f+h*h/4.f+3.f*h*radius/8.f);
    f32 iz=(m_cyl*r2*0.5f+m_hemi*2.f*r2/5.f);
    return Mat3f::scale({ix,ix,iz});
}

// Parallel axis theorem
[[nodiscard]] constexpr Mat3f parallel_axis(Mat3f I, f32 mass, Vec3f offset) noexcept {
    f32 d2=offset.length_sq();
    Mat3f outer{
        {offset.x*offset.x, offset.x*offset.y, offset.x*offset.z},
        {offset.x*offset.y, offset.y*offset.y, offset.y*offset.z},
        {offset.x*offset.z, offset.y*offset.z, offset.z*offset.z}
    };
    // I_new = I + m*(d^2*Id - outer)
    Mat3f id=Mat3f::identity()*d2;
    return I + (id-outer)*mass;
}

// ---------------------------------------------------------------------------
// Spring dynamics
// ---------------------------------------------------------------------------
struct SpringState {
    f32 position{0.f};
    f32 velocity{0.f};
};

// Semi-implicit Euler spring step
inline void spring_step(SpringState& s, f32 target, f32 stiffness, f32 damping, f32 dt) noexcept {
    f32 force=-stiffness*(s.position-target)-damping*s.velocity;
    s.velocity+=force*dt;
    s.position+=s.velocity*dt;
}

// Critically damped spring (no oscillation)
inline void critical_spring_step(SpringState& s, f32 target, f32 omega, f32 dt) noexcept {
    f32 diff=s.position-target;
    f32 f=1.f+2.f*dt*omega;
    f32 oo=omega*omega;
    f32 hoo=dt*oo;
    f32 det=1.f/(f+dt*hoo);
    s.position=(f*s.position+dt*s.velocity+hoo*target)*det;
    s.velocity=(s.velocity+hoo*(target-s.position+diff))*det;
}

// ---------------------------------------------------------------------------
// Damping models
// ---------------------------------------------------------------------------
// Linear drag
[[nodiscard]] constexpr Vec3f linear_drag(Vec3f velocity, f32 drag) noexcept {
    return velocity*(-drag);
}
// Quadratic (aerodynamic) drag
[[nodiscard]] constexpr Vec3f quadratic_drag(Vec3f velocity, f32 drag_coeff, f32 frontal_area, f32 air_density=1.225f) noexcept {
    f32 speed=velocity.length();
    if(speed<EPS_LENGTH) return Vec3f{0.f};
    f32 drag_force=0.5f*air_density*drag_coeff*frontal_area*speed*speed;
    return velocity.normalized()*(-drag_force);
}

// ---------------------------------------------------------------------------
// Velocity helpers
// ---------------------------------------------------------------------------
// Velocity needed to reach target height in gravity field
[[nodiscard]] constexpr f32 launch_speed_for_height(f32 target_height, f32 grav=9.80665f) noexcept {
    return sqrt(2.f*grav*max(0.f,target_height));
}
// Time to fall a given height
[[nodiscard]] constexpr f32 fall_time(f32 height, f32 grav=9.80665f) noexcept {
    return sqrt(2.f*max(0.f,height)/grav);
}
// Projectile range
[[nodiscard]] constexpr f32 projectile_range(f32 speed, f32 angle_rad, f32 grav=9.80665f) noexcept {
    return speed*speed*sin(2.f*angle_rad)/grav;
}
// Orbital velocity
[[nodiscard]] inline f32 orbital_velocity(f32 radius, f32 planet_mass, f32 G=6.674e-11f) noexcept {
    return sqrt(G*planet_mass/radius);
}
// Impulse to change velocity
[[nodiscard]] constexpr Vec3f impulse(Vec3f delta_v, f32 mass) noexcept { return delta_v*mass; }

} // namespace qs::math::physics
