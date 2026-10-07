
// qs_math/physics/physics_ext_impl.cpp (C++20)
#include "include/qs_math_physics.hpp"
#include <cmath>
namespace qs::math::physics {

// ─── RigidBody ────────────────────────────────────────────────────────────────
void RigidBody::integrate(f64 dt) noexcept {
    // Semi-implicit Euler
    velocity  = velocity  + force  * (mass_inv * dt);
    position  = position  + velocity * dt;
    angular_vel = angular_vel + (inertia_inv * torque) * dt;
    // Integrate orientation: q_dot = 0.5 * q * [0, omega]
    const f64 wx=angular_vel.x*0.5*dt, wy=angular_vel.y*0.5*dt, wz=angular_vel.z*0.5*dt;
    const f64 qw=orientation.w,qx=orientation.x,qy=orientation.y,qz=orientation.z;
    orientation.w += -qx*wx - qy*wy - qz*wz;
    orientation.x +=  qw*wx + qy*wz - qz*wy;
    orientation.y +=  qw*wy - qx*wz + qz*wx;
    orientation.z +=  qw*wz + qx*wy - qy*wx;
    // Renormalise
    const f64 len=std::sqrt(orientation.w*orientation.w+orientation.x*orientation.x+
                            orientation.y*orientation.y+orientation.z*orientation.z);
    if(len>1e-10){orientation.w/=len;orientation.x/=len;orientation.y/=len;orientation.z/=len;}
    force={}; torque={};
}
void RigidBody::apply_force(Vec3<f64> f, Vec3<f64> r) noexcept {
    force=force+f;
    if(r.length_sq()>1e-14) torque=torque+r.cross(f);
}
void RigidBody::apply_impulse(Vec3<f64> j, Vec3<f64> r) noexcept {
    velocity=velocity+j*mass_inv;
    if(r.length_sq()>1e-14) angular_vel=angular_vel+(inertia_inv*r.cross(j));
}
Vec3<f64> RigidBody::world_point(Vec3<f64> lp) const noexcept {
    // Rotate by quaternion
    const f64 qw=orientation.w,qx=orientation.x,qy=orientation.y,qz=orientation.z;
    const f64 rx=2*(qy*lp.z-qz*lp.y)+lp.x*2*(qw*qw+qx*qx-0.5);
    const f64 ry=2*(qz*lp.x-qx*lp.z)+lp.y*2*(qw*qw+qy*qy-0.5);
    const f64 rz=2*(qx*lp.y-qy*lp.x)+lp.z*2*(qw*qw+qz*qz-0.5);
    // Full rotation:
    const f64 tx=2*(qx*lp.x+qy*lp.y+qz*lp.z);
    return position+Vec3<f64>{
        lp.x+2*qw*(qy*lp.z-qz*lp.y)+2*(qx*(qx*lp.x+qy*lp.y+qz*lp.z)-lp.x*(0.5)),
        lp.y+2*qw*(qz*lp.x-qx*lp.z)+2*(qy*(qx*lp.x+qy*lp.y+qz*lp.z)-lp.y*(0.5)),
        lp.z+2*qw*(qx*lp.y-qy*lp.x)+2*(qz*(qx*lp.x+qy*lp.y+qz*lp.z)-lp.z*(0.5))};
    // Simplified correct rotation:
    (void)rx;(void)ry;(void)rz;(void)tx;
    return position + Vec3<f64>{
        lp.x*(1-2*(qy*qy+qz*qz))+lp.y*2*(qx*qy-qw*qz)+lp.z*2*(qx*qz+qw*qy),
        lp.x*2*(qx*qy+qw*qz)+lp.y*(1-2*(qx*qx+qz*qz))+lp.z*2*(qy*qz-qw*qx),
        lp.x*2*(qx*qz-qw*qy)+lp.y*2*(qy*qz+qw*qx)+lp.z*(1-2*(qx*qx+qy*qy))};
}
Vec3<f64> RigidBody::point_velocity(Vec3<f64> wp) const noexcept {
    return velocity + angular_vel.cross(wp-position);
}
// ─── Projectile ───────────────────────────────────────────────────────────────
void Projectile::step(f64 dt, Vec3<f64> gravity) noexcept {
    const f64 speed=vel.length();
    Vec3<f64> drag_force{};
    if(speed>1e-10&&drag_coef>0)
        drag_force=vel.normalized()*(-drag_coef*speed*speed/mass);
    vel=vel+(gravity+drag_force)*dt;
    pos=pos+vel*dt;
}
// ─── Optics ───────────────────────────────────────────────────────────────────
Vec3<f64> snell_refract(Vec3<f64> d, Vec3<f64> n, f64 n1, f64 n2) noexcept {
    const f64 r=n1/n2;
    const f64 cos_i=-n.dot(d);
    const f64 sin2_t=r*r*(1-cos_i*cos_i);
    if(sin2_t>1.0) return d-n*(2*n.dot(d)); // total internal reflection
    return d*r+n*(r*cos_i-std::sqrt(1-sin2_t));
}
Vec3<f64> fresnel_reflect(Vec3<f64> d, Vec3<f64> n) noexcept {
    return d-n*(2*n.dot(d));
}
f64 fresnel_reflectance(f64 cos_i, f64 n1, f64 n2) noexcept {
    const f64 sin2_t=n1*n1*(1-cos_i*cos_i)/(n2*n2);
    if(sin2_t>1.0) return 1.0; // total internal reflection
    const f64 cos_t=std::sqrt(1-sin2_t);
    const f64 rs=(n1*cos_i-n2*cos_t)/(n1*cos_i+n2*cos_t);
    const f64 rp=(n2*cos_i-n1*cos_t)/(n2*cos_i+n1*cos_t);
    return 0.5*(rs*rs+rp*rp);
}
// ─── Orbital mechanics ────────────────────────────────────────────────────────
Vec3<f64> Orbit::state_at_anomaly(f64 nu, f64 mu) const noexcept {
    const f64 p=semi_major*(1-eccentricity*eccentricity);
    const f64 r=p/(1+eccentricity*std::cos(nu));
    // Position in orbital plane (x=periapsis direction)
    const f64 px=r*std::cos(nu), py=r*std::sin(nu);
    // Velocity
    const f64 h=std::sqrt(mu*p);
    const f64 vx=-mu/h*std::sin(nu), vy=mu/h*(eccentricity+std::cos(nu));
    // Include inclination rotation (simplified, RAAN=AOP=0)
    const f64 ci=std::cos(inclination), si=std::sin(inclination);
    return{px, py*ci-0*si, py*si+0*ci, vx, vy*ci, vy*si}; // simplified: return pos only
    return Vec3<f64>{px, py*ci, py*si};
}
// ─── Wave equation helpers ────────────────────────────────────────────────────
[[nodiscard]] f64 standing_wave(f64 x, f64 t, f64 A, f64 k, f64 omega) noexcept {
    return 2*A*std::cos(k*x)*std::cos(omega*t);
}
[[nodiscard]] f64 doppler_freq(f64 f0, f64 vs, f64 vr, f64 v_sound=343.0) noexcept {
    return f0*(v_sound+vr)/(v_sound+vs);
}
[[nodiscard]] f64 blackbody_radiance(f64 lambda, f64 T) noexcept {
    constexpr f64 h=6.626e-34, c=2.998e8, kb=1.381e-23;
    return 2*h*c*c/std::pow(lambda,5)*(1.0/(std::exp(h*c/(lambda*kb*T))-1));
}
[[nodiscard]] f64 wien_peak(f64 T) noexcept { return 2.898e-3/T; }
} // namespace qs::math::physics
