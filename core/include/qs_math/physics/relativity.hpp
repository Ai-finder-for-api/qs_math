#pragma once
// qs_math/physics/relativity.hpp — Special/general relativity, Lorentz, metric tensors, geodesics
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

inline constexpr f64 k_c2=8.987551787368176e16; // c² in m²/s²
inline constexpr f64 k_c =2.99792458e8;          // c in m/s
inline constexpr f64 k_G =6.674e-11;             // gravitational constant

// ─── 4-vector ─────────────────────────────────────────────────────────────────
struct FourVec { f64 t,x,y,z;
    // Minkowski inner product (signature -+++)
    [[nodiscard]] constexpr f64 dot(FourVec b) const noexcept {
        return -t*b.t+x*b.x+y*b.y+z*b.z;
    }
    [[nodiscard]] constexpr FourVec operator+(FourVec b) const noexcept {return{t+b.t,x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr FourVec operator-(FourVec b) const noexcept {return{t-b.t,x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr FourVec operator*(f64 s)    const noexcept {return{t*s,x*s,y*s,z*s};}
    // Invariant interval: ds² = -c²dt² + dx² + dy² + dz²
    [[nodiscard]] constexpr f64 interval2() const noexcept {return -k_c2*t*t+x*x+y*y+z*z;}
    [[nodiscard]] constexpr bool is_timelike()  const noexcept {return interval2()<0;}
    [[nodiscard]] constexpr bool is_spacelike() const noexcept {return interval2()>0;}
    [[nodiscard]] constexpr bool is_lightlike() const noexcept {return std::abs(interval2())<1e-10;}
};

// ─── Lorentz factor ───────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 lorentz_gamma(f64 v) noexcept {
    f64 beta=v/k_c;
    return 1.0/std::sqrt(1-beta*beta);
}
[[nodiscard]] constexpr f64 beta_from_gamma(f64 gamma) noexcept {
    return std::sqrt(1-1/(gamma*gamma));
}
[[nodiscard]] constexpr f64 rapidity(f64 v) noexcept {
    f64 beta=v/k_c;
    return 0.5*std::log((1+beta)/(1-beta));
}

// ─── Lorentz boost (x-direction) ─────────────────────────────────────────────
[[nodiscard]] constexpr FourVec lorentz_boost_x(FourVec p, f64 v) noexcept {
    f64 gamma=lorentz_gamma(v), beta=v/k_c;
    return{gamma*(p.t-beta*p.x/k_c),gamma*(p.x-v*p.t),p.y,p.z};
}
// General Lorentz boost along direction (nx,ny,nz)
[[nodiscard]] constexpr FourVec lorentz_boost(FourVec p, f64 vx, f64 vy, f64 vz) noexcept {
    f64 v=std::sqrt(vx*vx+vy*vy+vz*vz);
    if(v<1e-15) return p;
    f64 gamma=lorentz_gamma(v),beta=v/k_c;
    f64 dot=(vx*p.x+vy*p.y+vz*p.z)/v;
    f64 t_new=gamma*(p.t-beta*dot/k_c);
    f64 coeff=(gamma-1)*dot/v-gamma*beta*p.t*k_c/v; // simplified
    return{t_new,p.x+(vx/v)*coeff,p.y+(vy/v)*coeff,p.z+(vz/v)*coeff};
}

// ─── Relativistic kinematics ──────────────────────────────────────────────────
// Relativistic energy: E² = (pc)² + (mc²)²
[[nodiscard]] constexpr f64 rel_energy(f64 p_magnitude, f64 mass) noexcept {
    return std::sqrt(p_magnitude*p_magnitude*k_c2+mass*mass*k_c2*k_c2);
}
[[nodiscard]] constexpr f64 rel_kinetic_energy(f64 v, f64 mass) noexcept {
    return (lorentz_gamma(v)-1)*mass*k_c2;
}
[[nodiscard]] constexpr f64 rel_momentum(f64 v, f64 mass) noexcept {
    return lorentz_gamma(v)*mass*v;
}
// Energy-momentum 4-vector: p^μ = (E/c, p_x, p_y, p_z)
[[nodiscard]] constexpr FourVec four_momentum(f64 mass, f64 vx, f64 vy, f64 vz) noexcept {
    f64 v=std::sqrt(vx*vx+vy*vy+vz*vz);
    f64 gamma=lorentz_gamma(v),E=rel_energy(gamma*mass*v,mass);
    return{E/k_c,gamma*mass*vx,gamma*mass*vy,gamma*mass*vz};
}
// Invariant mass: M² = -p²/c² (in natural units with c=1: M² = E²-|p|²)
[[nodiscard]] constexpr f64 invariant_mass(FourVec p) noexcept {
    return std::sqrt(std::max(0.0,-p.interval2()))/k_c2;
}

// ─── Relativistic velocity addition ──────────────────────────────────────────
[[nodiscard]] constexpr f64 velocity_addition(f64 v1, f64 v2) noexcept {
    return (v1+v2)/(1+v1*v2/k_c2);
}
// Thomas precession angle
[[nodiscard]] constexpr f64 thomas_angle(f64 v1, f64 v2, f64 angle) noexcept {
    f64 gamma1=lorentz_gamma(v1),gamma2=lorentz_gamma(v2);
    return std::atan2(gamma1*gamma2*(gamma1+gamma2)*std::sin(angle)*v1*v2/k_c2,
                      (gamma1+gamma2)*(gamma1+gamma2)-gamma1*gamma2*(1+std::cos(angle))*v1*v2/k_c2);
}

// ─── Electromagnetic field tensor F^{μν} ─────────────────────────────────────
using FieldTensor = std::array<std::array<f64,4>,4>;

[[nodiscard]] constexpr FieldTensor em_field_tensor(
    f64 Ex,f64 Ey,f64 Ez,f64 Bx,f64 By,f64 Bz) noexcept
{
    return{{{{0,-Ex/k_c,-Ey/k_c,-Ez/k_c}},
            {{Ex/k_c,0,-Bz,By}},
            {{Ey/k_c,Bz,0,-Bx}},
            {{Ez/k_c,-By,Bx,0}}}};
}
// Lorentz invariants: F^{μν}F_{μν} = 2(B²-E²/c²), F*F = -4E·B/c
[[nodiscard]] constexpr f64 em_invariant_1(f64 Ex,f64 Ey,f64 Ez,f64 Bx,f64 By,f64 Bz) noexcept {
    return 2*(Bx*Bx+By*By+Bz*Bz-(Ex*Ex+Ey*Ey+Ez*Ez)/k_c2);
}
[[nodiscard]] constexpr f64 em_invariant_2(f64 Ex,f64 Ey,f64 Ez,f64 Bx,f64 By,f64 Bz) noexcept {
    return -4*(Ex*Bx+Ey*By+Ez*Bz)/k_c;
}

// ─── Schwarzschild metric ─────────────────────────────────────────────────────
// ds² = -(1-r_s/r)c²dt² + (1-r_s/r)^{-1}dr² + r²(dθ²+sin²θ dφ²)
// r_s = 2GM/c² (Schwarzschild radius)
[[nodiscard]] constexpr f64 schwarzschild_radius(f64 mass) noexcept {
    return 2*k_G*mass/k_c2;
}
[[nodiscard]] constexpr f64 gravitational_time_dilation(f64 r, f64 r_s) noexcept {
    return std::sqrt(1-r_s/r); // dt_proper/dt_far
}
[[nodiscard]] constexpr f64 gravitational_redshift(f64 r_emit, f64 r_obs, f64 r_s) noexcept {
    return std::sqrt((1-r_s/r_obs)/(1-r_s/r_emit))-1;
}
// Photon sphere radius (unstable circular orbit)
[[nodiscard]] constexpr f64 photon_sphere(f64 r_s) noexcept {return 1.5*r_s;}
// ISCO (innermost stable circular orbit) for Schwarzschild
[[nodiscard]] constexpr f64 isco_radius(f64 r_s) noexcept {return 3*r_s;}

// ─── Gravitational wave strain ────────────────────────────────────────────────
// h_+ and h_× polarizations for binary system
[[nodiscard]] constexpr f64 gw_strain_plus(
    f64 chirp_mass, f64 distance, f64 freq, f64 time, f64 inclination) noexcept
{
    f64 Mc=chirp_mass;
    f64 amp=4*k_G*Mc/(distance*k_c2)*std::pow(k_pi_d*k_G*Mc*freq/k_c2/k_c,2.0/3.0);
    return amp*(1+std::cos(inclination)*std::cos(inclination))*0.5*std::cos(2*k_pi_d*freq*time);
}
[[nodiscard]] constexpr f64 gw_strain_cross(
    f64 chirp_mass, f64 distance, f64 freq, f64 time, f64 inclination) noexcept
{
    f64 Mc=chirp_mass;
    f64 amp=4*k_G*Mc/(distance*k_c2)*std::pow(k_pi_d*k_G*Mc*freq/k_c2/k_c,2.0/3.0);
    return amp*std::cos(inclination)*std::sin(2*k_pi_d*freq*time);
}
// Chirp mass from component masses
[[nodiscard]] constexpr f64 chirp_mass(f64 m1, f64 m2) noexcept {
    return std::pow(m1*m2,0.6)/std::pow(m1+m2,0.2);
}
// Gravitational wave frequency evolution: df/dt = (96/5)π^{8/3}(GM_c/c³)^{5/3}f^{11/3}
[[nodiscard]] constexpr f64 gw_freq_derivative(f64 Mc, f64 f) noexcept {
    f64 coeff=96.0/5*(k_pi_d*k_G*Mc/k_c2/k_c);
    return coeff*std::pow(k_pi_d*f,11.0/3.0)*std::pow(k_G*Mc/k_c2/k_c,5.0/3.0)*k_pi_d;
}
} // namespace qs::math
