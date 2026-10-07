#pragma once
// qs_math/physics/electromagnetism.hpp — Maxwell eqs, field math, EM waves, circuits
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

// ─── EM constants ─────────────────────────────────────────────────────────────
inline constexpr f64 k_epsilon0  = 8.8541878128e-12; // F/m
inline constexpr f64 k_mu0       = 1.25663706212e-6; // H/m
inline constexpr f64 k_c_light   = 2.99792458e8;     // m/s
inline constexpr f64 k_e_charge  = 1.602176634e-19;  // C
inline constexpr f64 k_coulomb   = 8.9875517923e9;   // N·m²/C²

// ─── Coulomb's law ────────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 coulomb_force(f64 q1, f64 q2, f64 r) noexcept {
    return k_coulomb*q1*q2/(r*r);
}
[[nodiscard]] constexpr f64 electric_potential(f64 q, f64 r) noexcept {
    return k_coulomb*q/r;
}
[[nodiscard]] constexpr f64 electric_field_point(f64 q, f64 r) noexcept {
    return k_coulomb*q/(r*r);
}

// ─── Electric dipole ─────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 dipole_potential(f64 p, f64 r, f64 theta) noexcept {
    return k_coulomb*p*std::cos(theta)/(r*r);
}
[[nodiscard]] constexpr f64 dipole_field_r(f64 p, f64 r, f64 theta) noexcept {
    return 2*k_coulomb*p*std::cos(theta)/(r*r*r);
}
[[nodiscard]] constexpr f64 dipole_field_theta(f64 p, f64 r, f64 theta) noexcept {
    return k_coulomb*p*std::sin(theta)/(r*r*r);
}

// ─── Gauss's law applications ─────────────────────────────────────────────────
[[nodiscard]] constexpr f64 field_infinite_line(f64 lambda, f64 r) noexcept {
    return lambda/(2*3.14159265358979323846*k_epsilon0*r);
}
[[nodiscard]] constexpr f64 field_infinite_plane(f64 sigma) noexcept {
    return sigma/(2*k_epsilon0);
}
[[nodiscard]] constexpr f64 field_sphere_outside(f64 Q, f64 r) noexcept {
    return Q/(4*3.14159265358979323846*k_epsilon0*r*r);
}
[[nodiscard]] constexpr f64 field_sphere_inside(f64 rho, f64 r) noexcept {
    return rho*r/(3*k_epsilon0);
}

// ─── Magnetic fields ──────────────────────────────────────────────────────────
// Biot-Savart for infinite wire
[[nodiscard]] constexpr f64 b_infinite_wire(f64 I, f64 r) noexcept {
    return k_mu0*I/(2*3.14159265358979323846*r);
}
// Solenoid interior: B = μ₀nI
[[nodiscard]] constexpr f64 b_solenoid(f64 n_turns_per_m, f64 I) noexcept {
    return k_mu0*n_turns_per_m*I;
}
// Toroid: B = μ₀NI/(2πr)
[[nodiscard]] constexpr f64 b_toroid(f64 N, f64 I, f64 r) noexcept {
    return k_mu0*N*I/(2*3.14159265358979323846*r);
}

// ─── EM wave propagation ──────────────────────────────────────────────────────
struct EMWave { f64 E_amp,B_amp,frequency,wavelength,phase_velocity; };

[[nodiscard]] constexpr EMWave em_wave(f64 E0, f64 frequency) noexcept {
    f64 lambda=k_c_light/frequency;
    return {E0, E0/k_c_light, frequency, lambda, k_c_light};
}

// Poynting vector magnitude: S = E²/(μ₀c) for plane wave
[[nodiscard]] constexpr f64 poynting_magnitude(f64 E0) noexcept {
    return E0*E0/(k_mu0*k_c_light);
}
// Wave impedance of free space: η₀ = μ₀c
inline constexpr f64 k_eta0 = k_mu0*k_c_light; // ≈ 376.73 Ω

// Index of refraction: n = c/v_phase
[[nodiscard]] constexpr f64 refractive_index(f64 eps_r, f64 mu_r=1) noexcept {
    return std::sqrt(eps_r*mu_r);
}
// Snell's law: n1*sin(θ1) = n2*sin(θ2)
[[nodiscard]] constexpr f64 snell_angle(f64 n1, f64 n2, f64 theta1) noexcept {
    f64 s=n1*std::sin(theta1)/n2;
    return s>1.0?-1.0:std::asin(s); // -1 = total internal reflection
}
// Fresnel coefficients (s-polarization)
[[nodiscard]] constexpr f64 fresnel_rs(f64 n1, f64 n2, f64 t1, f64 t2) noexcept {
    return (n1*std::cos(t1)-n2*std::cos(t2))/(n1*std::cos(t1)+n2*std::cos(t2));
}
[[nodiscard]] constexpr f64 fresnel_ts(f64 n1, f64 n2, f64 t1, f64 t2) noexcept {
    return 2*n1*std::cos(t1)/(n1*std::cos(t1)+n2*std::cos(t2));
}
// p-polarization
[[nodiscard]] constexpr f64 fresnel_rp(f64 n1, f64 n2, f64 t1, f64 t2) noexcept {
    return (n2*std::cos(t1)-n1*std::cos(t2))/(n2*std::cos(t1)+n1*std::cos(t2));
}

// ─── Capacitors and inductors ─────────────────────────────────────────────────
[[nodiscard]] constexpr f64 capacitance_parallel_plate(f64 eps_r,f64 A,f64 d) noexcept {
    return eps_r*k_epsilon0*A/d;
}
[[nodiscard]] constexpr f64 capacitance_cylindrical(f64 eps_r,f64 L,f64 a,f64 b) noexcept {
    return 2*3.14159265358979323846*eps_r*k_epsilon0*L/std::log(b/a);
}
[[nodiscard]] constexpr f64 inductance_solenoid(f64 mu_r,f64 N,f64 A,f64 l) noexcept {
    return mu_r*k_mu0*N*N*A/l;
}
[[nodiscard]] constexpr f64 inductance_toroid(f64 mu_r,f64 N,f64 A,f64 r) noexcept {
    return mu_r*k_mu0*N*N*A/(2*3.14159265358979323846*r);
}

// ─── RLC circuits ─────────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 resonant_frequency(f64 L, f64 C) noexcept {
    return 1.0/(2*3.14159265358979323846*std::sqrt(L*C));
}
[[nodiscard]] constexpr f64 quality_factor(f64 R, f64 L, f64 C) noexcept {
    return (1.0/R)*std::sqrt(L/C);
}
[[nodiscard]] constexpr f64 rlc_impedance(f64 R, f64 L, f64 C, f64 omega) noexcept {
    f64 X=omega*L-1.0/(omega*C);
    return std::sqrt(R*R+X*X);
}
[[nodiscard]] constexpr f64 rlc_phase(f64 R, f64 L, f64 C, f64 omega) noexcept {
    return std::atan2(omega*L-1.0/(omega*C),R);
}

// ─── Maxwell stress tensor ────────────────────────────────────────────────────
// T_{ij} = ε₀(E_iE_j - δ_{ij}|E|²/2) + (1/μ₀)(B_iB_j - δ_{ij}|B|²/2)
template<int N=3>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N> maxwell_stress_tensor(
    const std::array<f64,N>& E, const std::array<f64,N>& B) noexcept
{
    std::array<std::array<f64,N>,N> T{};
    f64 E2=0,B2=0;
    for(int i=0;i<N;++i){E2+=E[i]*E[i];B2+=B[i]*B[i];}
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
        T[i][j]=k_epsilon0*(E[i]*E[j]-(i==j?0.5*E2:0))
               +(B[i]*B[j]-(i==j?0.5*B2:0))/k_mu0;
    }
    return T;
}
} // namespace qs::math
