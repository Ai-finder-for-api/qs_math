#pragma once
// qs_math/physics/thermodynamics.hpp — Thermo laws, gas models, entropy, heat transfer
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

// ─── Physical constants ───────────────────────────────────────────────────────
inline constexpr f64 k_boltzmann    = 1.380649e-23;   // J/K
inline constexpr f64 k_avogadro     = 6.02214076e23;  // mol^-1
inline constexpr f64 k_gas_constant = 8.314462618;    // J/(mol·K)
inline constexpr f64 k_planck       = 6.62607015e-34; // J·s
inline constexpr f64 k_stefan_boltzmann = 5.670374419e-8; // W/(m²K⁴)

// ─── Ideal gas ────────────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 ideal_gas_pressure(f64 n, f64 T, f64 V) noexcept {
    return n*k_gas_constant*T/V;
}
[[nodiscard]] constexpr f64 ideal_gas_volume(f64 n, f64 T, f64 P) noexcept {
    return n*k_gas_constant*T/P;
}
[[nodiscard]] constexpr f64 ideal_gas_temperature(f64 P, f64 V, f64 n) noexcept {
    return P*V/(n*k_gas_constant);
}
// RMS speed of ideal gas molecules: v_rms = sqrt(3kT/m)
[[nodiscard]] constexpr f64 rms_speed(f64 T, f64 m_kg) noexcept {
    return std::sqrt(3*k_boltzmann*T/m_kg);
}
// Mean speed: v_mean = sqrt(8kT/(πm))
[[nodiscard]] constexpr f64 mean_speed(f64 T, f64 m_kg) noexcept {
    return std::sqrt(8*k_boltzmann*T/(3.14159265358979323846*m_kg));
}
// Most probable speed: v_mp = sqrt(2kT/m)
[[nodiscard]] constexpr f64 most_probable_speed(f64 T, f64 m_kg) noexcept {
    return std::sqrt(2*k_boltzmann*T/m_kg);
}

// ─── Maxwell-Boltzmann distribution ──────────────────────────────────────────
[[nodiscard]] constexpr f64 maxwell_boltzmann(f64 v, f64 T, f64 m_kg) noexcept {
    f64 a=m_kg/(k_boltzmann*T);
    return 4*3.14159265358979323846*v*v*std::pow(a/(2*3.14159265358979323846),1.5)*std::exp(-a*v*v*0.5);
}

// ─── Van der Waals gas ────────────────────────────────────────────────────────
// (p + a/v²)(v - b) = RT, a=attraction, b=excluded volume
[[nodiscard]] constexpr f64 vdw_pressure(f64 T, f64 Vm, f64 a, f64 b) noexcept {
    return k_gas_constant*T/(Vm-b)-a/(Vm*Vm);
}
// Critical constants: Vc=3b, Tc=8a/(27Rb), Pc=a/(27b²)
[[nodiscard]] constexpr f64 vdw_critical_T(f64 a, f64 b) noexcept {
    return 8*a/(27*k_gas_constant*b);
}
[[nodiscard]] constexpr f64 vdw_critical_P(f64 a, f64 b) noexcept { return a/(27*b*b); }
[[nodiscard]] constexpr f64 vdw_critical_V(f64 b) noexcept { return 3*b; }

// ─── Thermodynamic potentials ─────────────────────────────────────────────────
// Internal energy: dU = TdS - PdV
// Helmholtz: A = U - TS
// Enthalpy: H = U + PV
// Gibbs: G = H - TS = U + PV - TS

[[nodiscard]] constexpr f64 helmholtz(f64 U, f64 T, f64 S) noexcept { return U-T*S; }
[[nodiscard]] constexpr f64 enthalpy(f64 U, f64 P, f64 V)  noexcept { return U+P*V; }
[[nodiscard]] constexpr f64 gibbs(f64 H, f64 T, f64 S)     noexcept { return H-T*S; }

// ─── Entropy calculations ─────────────────────────────────────────────────────
// Boltzmann entropy: S = k*ln(W)
[[nodiscard]] constexpr f64 boltzmann_entropy(f64 W) noexcept {
    return k_boltzmann*std::log(W);
}
// Entropy change for ideal gas process
[[nodiscard]] constexpr f64 entropy_change_ideal(f64 n, f64 Cv, f64 T1, f64 T2,
                                                   f64 V1, f64 V2) noexcept {
    return n*(Cv*std::log(T2/T1)+k_gas_constant*std::log(V2/V1));
}
// Clausius inequality: ΔS ≥ Q/T
[[nodiscard]] constexpr f64 clausius_entropy(f64 Q, f64 T) noexcept { return Q/T; }

// ─── Heat transfer ────────────────────────────────────────────────────────────
// Fourier's law: q = -k * dT/dx
[[nodiscard]] constexpr f64 fourier_heat_flux(f64 k, f64 dT_dx) noexcept { return -k*dT_dx; }
// Newton's law of cooling: Q = h*A*(T_s - T_inf)
[[nodiscard]] constexpr f64 newton_cooling(f64 h, f64 A, f64 Ts, f64 Tinf) noexcept {
    return h*A*(Ts-Tinf);
}
// Stefan-Boltzmann radiation: Q = ε*σ*A*T^4
[[nodiscard]] constexpr f64 radiation_flux(f64 eps, f64 T) noexcept {
    return eps*k_stefan_boltzmann*T*T*T*T;
}
// 1D fin temperature distribution: T(x) = T_inf + (T_b-T_inf)*cosh(m*(L-x))/cosh(m*L)
// m = sqrt(h*P/(k*A_c))
[[nodiscard]] constexpr f64 fin_temperature(f64 x, f64 L, f64 m,
                                              f64 T_base, f64 T_inf) noexcept {
    return T_inf+(T_base-T_inf)*std::cosh(m*(L-x))/std::cosh(m*L);
}
// Biot number: Bi = h*L_c/k (lumped capacitance valid for Bi<0.1)
[[nodiscard]] constexpr f64 biot_number(f64 h, f64 Lc, f64 k) noexcept { return h*Lc/k; }
// Lumped capacitance: T(t) = T_inf + (T0-T_inf)*exp(-t/τ), τ = ρcV/(hA)
[[nodiscard]] constexpr f64 lumped_temp(f64 t, f64 tau, f64 T0, f64 T_inf) noexcept {
    return T_inf+(T0-T_inf)*std::exp(-t/tau);
}

// ─── Phase transitions ────────────────────────────────────────────────────────
// Clausius-Clapeyron: dP/dT = L*P/(R*T²) for liquid-vapor
[[nodiscard]] constexpr f64 clausius_clapeyron_slope(f64 L, f64 P, f64 T) noexcept {
    return L*P/(k_gas_constant*T*T);
}
// Latent heat from two-point measurement
[[nodiscard]] constexpr f64 latent_heat(f64 P1,f64 T1,f64 P2,f64 T2) noexcept {
    return k_gas_constant*std::log(P2/P1)/((1.0/T1)-(1.0/T2));
}

// ─── Carnot efficiency ────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 carnot_efficiency(f64 T_cold, f64 T_hot) noexcept {
    return 1.0-T_cold/T_hot;
}
[[nodiscard]] constexpr f64 carnot_cop_refrigerator(f64 T_cold, f64 T_hot) noexcept {
    return T_cold/(T_hot-T_cold);
}

// ─── Adiabatic processes ─────────────────────────────────────────────────────
// T*V^(γ-1) = const, T^γ*P^(1-γ) = const
[[nodiscard]] constexpr f64 adiabatic_T2(f64 T1, f64 V1, f64 V2, f64 gamma) noexcept {
    return T1*std::pow(V1/V2,gamma-1);
}
[[nodiscard]] constexpr f64 adiabatic_P2(f64 P1, f64 V1, f64 V2, f64 gamma) noexcept {
    return P1*std::pow(V1/V2,gamma);
}
// Speed of sound: c = sqrt(γP/ρ) = sqrt(γRT/M)
[[nodiscard]] constexpr f64 speed_of_sound(f64 gamma, f64 T, f64 M_molar) noexcept {
    return std::sqrt(gamma*k_gas_constant*T/M_molar);
}

// ─── Blackbody radiation ──────────────────────────────────────────────────────
// Planck distribution: B(λ,T) = 2hc²/λ⁵ * 1/(exp(hc/λkT)-1)
[[nodiscard]] constexpr f64 planck_spectral(f64 lambda, f64 T) noexcept {
    constexpr f64 c=2.99792458e8,h=k_planck,k=k_boltzmann;
    f64 x=h*c/(lambda*k*T);
    if(x>700) return 0;
    return 2*h*c*c/(lambda*lambda*lambda*lambda*lambda)/(std::exp(x)-1);
}
// Wien's displacement: λ_max*T = b = 2.897771955e-3 m·K
[[nodiscard]] constexpr f64 wien_peak(f64 T) noexcept { return 2.897771955e-3/T; }
// Total power: P = σT⁴
[[nodiscard]] constexpr f64 blackbody_power(f64 T) noexcept {
    return k_stefan_boltzmann*T*T*T*T;
}

// ─── Statistical mechanics ────────────────────────────────────────────────────
// Bose-Einstein distribution: n_i = 1/(exp((E_i-μ)/kT)-1)
[[nodiscard]] constexpr f64 bose_einstein(f64 E, f64 mu, f64 T) noexcept {
    f64 x=(E-mu)/(k_boltzmann*T);
    if(x<-500) return 1e15;
    return 1.0/(std::exp(x)-1.0);
}
// Fermi-Dirac distribution: n_i = 1/(exp((E_i-μ)/kT)+1)
[[nodiscard]] constexpr f64 fermi_dirac(f64 E, f64 mu, f64 T) noexcept {
    f64 x=(E-mu)/(k_boltzmann*T);
    return 1.0/(std::exp(x)+1.0);
}
// Partition function (discrete): Z = Σ exp(-E_i/kT)
template<int N>
[[nodiscard]] constexpr f64 partition_function(const std::array<f64,N>& E, f64 T) noexcept {
    f64 Z=0; for(auto e:E) Z+=std::exp(-e/(k_boltzmann*T)); return Z;
}
// Free energy from partition: F = -kT*ln(Z)
[[nodiscard]] constexpr f64 free_energy(f64 Z, f64 T) noexcept {
    return -k_boltzmann*T*std::log(Z);
}
} // namespace qs::math
