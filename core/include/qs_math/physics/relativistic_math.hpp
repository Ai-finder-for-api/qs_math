#pragma once
// qs_math/relativistic_math.hpp — Lorentz transforms, 4-vectors, spacetime interval, metric (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include <cmath>
#include <array>

namespace qs {

static constexpr f64 k_c = 299792458.0; // speed of light m/s

// ─── Four-vector ──────────────────────────────────────────────────────────────

struct FourVector {
    f64 t{}, x{}, y{}, z{};   // (ct, x, y, z) or (E/c, px, py, pz)

    [[nodiscard]] f64 spacetime_interval_sq() const {
        return t*t - x*x - y*y - z*z;  // (+,-,-,-)  signature
    }

    [[nodiscard]] f64 proper_time_sq() const { return spacetime_interval_sq(); }

    [[nodiscard]] Vec3d spatial() const { return {x,y,z}; }

    FourVector operator+(const FourVector& o) const { return {t+o.t,x+o.x,y+o.y,z+o.z}; }
    FourVector operator*(f64 s) const { return {t*s,x*s,y*s,z*s}; }

    [[nodiscard]] f64 minkowski_dot(const FourVector& o) const {
        return t*o.t - x*o.x - y*o.y - z*o.z;
    }
};

// ─── Lorentz factor ───────────────────────────────────────────────────────────

[[nodiscard]] inline f64 lorentz_gamma(f64 v, f64 c=k_c) {
    f64 beta=v/c;
    return 1./std::sqrt(std::max(0.,1.-beta*beta));
}

[[nodiscard]] inline f64 lorentz_gamma_vec(const Vec3d& vel, f64 c=k_c) {
    f64 v2=dot(vel,vel)/(c*c);
    return 1./std::sqrt(std::max(0.,1.-v2));
}

[[nodiscard]] inline f64 rapidity(f64 v, f64 c=k_c) {
    f64 beta=v/c;
    return 0.5*std::log((1.+beta)/(1.-beta));
}

// ─── Lorentz boost (along x axis) ────────────────────────────────────────────

[[nodiscard]] inline FourVector lorentz_boost_x(const FourVector& ev, f64 v, f64 c=k_c) {
    f64 gamma=lorentz_gamma(v,c);
    f64 beta=v/c;
    return {gamma*(ev.t-beta*ev.x), gamma*(ev.x-beta*ev.t), ev.y, ev.z};
}

// General Lorentz boost along arbitrary direction n (unit vector)
[[nodiscard]] inline FourVector lorentz_boost(const FourVector& ev, const Vec3d& vel, f64 c=k_c) {
    f64 v=length(vel);
    if(v<1e-15*c) return ev;
    f64 gamma=lorentz_gamma(v,c);
    f64 beta=v/c;
    Vec3d n=vel/v;
    Vec3d r=ev.spatial();
    f64 r_par=dot(r,n);
    Vec3d r_perp=r-n*r_par;
    f64 t_new=gamma*(ev.t-beta*r_par/c*c);
    Vec3d r_new=r_perp+n*(gamma*(r_par-beta*c*ev.t));
    return {t_new,r_new.x,r_new.y,r_new.z};
}

// ─── Relativistic velocity addition ──────────────────────────────────────────

[[nodiscard]] inline Vec3d relativistic_velocity_add(
    const Vec3d& u, const Vec3d& v, f64 c=k_c)
{
    f64 gamma_v=lorentz_gamma_vec(v,c);
    f64 uv=dot(u,v)/(c*c);
    f64 scale=1./(1.+uv);
    Vec3d u_par=v*(dot(u,v)/dot(v,v));
    Vec3d u_perp=u-u_par;
    return (u_par+u_perp/gamma_v)*scale;
}

// ─── Relativistic energy / momentum ──────────────────────────────────────────

[[nodiscard]] inline f64 rest_energy(f64 mass, f64 c=k_c) { return mass*c*c; }

[[nodiscard]] inline f64 total_energy(f64 mass, f64 v, f64 c=k_c) {
    return lorentz_gamma(v,c)*mass*c*c;
}

[[nodiscard]] inline f64 kinetic_energy(f64 mass, f64 v, f64 c=k_c) {
    return total_energy(mass,v,c)-rest_energy(mass,c);
}

[[nodiscard]] inline Vec3d relativistic_momentum(f64 mass, const Vec3d& vel, f64 c=k_c) {
    return vel*lorentz_gamma_vec(vel,c)*mass;
}

// Energy-momentum four-vector
[[nodiscard]] inline FourVector energy_momentum(f64 mass, const Vec3d& vel, f64 c=k_c) {
    f64 gamma=lorentz_gamma_vec(vel,c);
    Vec3d p=vel*(gamma*mass);
    return {gamma*mass*c, p.x, p.y, p.z};
}

// ─── Doppler effect ───────────────────────────────────────────────────────────

// Relativistic Doppler: observed frequency for source moving at velocity v
// theta: angle between velocity and line of sight (observer frame)
[[nodiscard]] inline f64 relativistic_doppler(f64 f0, f64 v, f64 theta, f64 c=k_c) {
    f64 beta=v/c;
    f64 gamma=lorentz_gamma(v,c);
    return f0/(gamma*(1.+beta*std::cos(theta)));
}

// Transverse Doppler (theta = pi/2)
[[nodiscard]] inline f64 transverse_doppler(f64 f0, f64 v, f64 c=k_c) {
    return f0/lorentz_gamma(v,c);
}

// Gravitational Doppler (blueshift falling into gravity well)
[[nodiscard]] inline f64 gravitational_redshift(f64 f0, f64 GM, f64 r, f64 c=k_c) {
    return f0*std::sqrt(1.-2.*GM/(r*c*c));
}

// ─── Minkowski metric tensor ──────────────────────────────────────────────────

// Flat spacetime metric (+,-,-,-)
static constexpr f64 k_metric_flat[4][4] = {
    {1,0,0,0},{0,-1,0,0},{0,0,-1,0},{0,0,0,-1}};

[[nodiscard]] inline f64 minkowski_norm_sq(const FourVector& v) {
    return v.t*v.t-v.x*v.x-v.y*v.y-v.z*v.z;
}

// Schwarzschild metric component g_tt at radius r (mass M, G=1)
[[nodiscard]] inline f64 schwarzschild_gtt(f64 r, f64 GM, f64 c=k_c) {
    return 1.-2.*GM/(r*c*c);
}

// Gravitational time dilation: dt_proper = dt_coord * sqrt(g_tt)
[[nodiscard]] inline f64 gravitational_time_dilation(f64 dt, f64 r, f64 GM, f64 c=k_c) {
    return dt*std::sqrt(schwarzschild_gtt(r,GM,c));
}

// Schwarzschild radius
[[nodiscard]] inline f64 schwarzschild_radius(f64 mass, f64 G=6.674e-11, f64 c=k_c) {
    return 2.*G*mass/(c*c);
}

// ─── Penrose diagram coordinate transforms ────────────────────────────────────

// Kruskal-Szekeres coords from Schwarzschild (outside horizon)
[[nodiscard]] inline std::pair<f64,f64> kruskal_exterior(f64 r, f64 t, f64 rs) {
    f64 T=std::sqrt(r/rs-1.)*std::exp(r/(2.*rs))*std::sinh(t/(2.*rs));
    f64 X=std::sqrt(r/rs-1.)*std::exp(r/(2.*rs))*std::cosh(t/(2.*rs));
    return {T,X};
}

// ─── Aberration of light ──────────────────────────────────────────────────────

[[nodiscard]] inline f64 stellar_aberration(f64 theta_source, f64 v, f64 c=k_c) {
    f64 beta=v/c;
    f64 cos_s=std::cos(theta_source);
    f64 cos_obs=(cos_s-beta)/(1.-beta*cos_s);
    return std::acos(std::clamp(cos_obs,-1.,1.));
}

} // namespace qs
