#pragma once
// qs_math/math_constants_ext.hpp — Physical, astronomical, and mathematical constants (C++20)
#include "scalar_types.hpp"
#include <numbers>

namespace qs {

// ─── Mathematical constants (high-precision) ─────────────────────────────────
inline constexpr f64 QS_PI          = std::numbers::pi_v<f64>;
inline constexpr f64 QS_TAU         = 2*std::numbers::pi_v<f64>;
inline constexpr f64 QS_E           = std::numbers::e_v<f64>;
inline constexpr f64 QS_PHI         = std::numbers::phi_v<f64>;       // golden ratio
inline constexpr f64 QS_SQRT2       = std::numbers::sqrt2_v<f64>;
inline constexpr f64 QS_SQRT3       = std::numbers::sqrt3_v<f64>;
inline constexpr f64 QS_LN2         = std::numbers::ln2_v<f64>;
inline constexpr f64 QS_LN10        = std::numbers::ln10_v<f64>;
inline constexpr f64 QS_LOG2E       = std::numbers::log2e_v<f64>;
inline constexpr f64 QS_LOG10E      = std::numbers::log10e_v<f64>;
inline constexpr f64 QS_INV_PI      = std::numbers::inv_pi_v<f64>;
inline constexpr f64 QS_INV_SQRTPI  = std::numbers::inv_sqrtpi_v<f64>;
inline constexpr f64 QS_EGAMMA      = std::numbers::egamma_v<f64>;    // Euler-Mascheroni γ ≈ 0.5772

// Additional mathematical constants
inline constexpr f64 QS_SQRT5       = 2.2360679774997896964091736687747632;
inline constexpr f64 QS_CBRT2       = 1.2599210498948731647672106072782284;
inline constexpr f64 QS_CBRT3       = 1.4422495703074083823216383107801095;
inline constexpr f64 QS_PI2         = QS_PI*QS_PI;                   // π²
inline constexpr f64 QS_PI_SQ       = QS_PI*QS_PI;
inline constexpr f64 QS_HALF_PI     = QS_PI*0.5;
inline constexpr f64 QS_QUARTER_PI  = QS_PI*0.25;
inline constexpr f64 QS_TWO_OVER_PI = 2.0/QS_PI;
inline constexpr f64 QS_FOUR_OVER_PI= 4.0/QS_PI;
inline constexpr f64 QS_DEG_TO_RAD  = QS_PI/180.0;
inline constexpr f64 QS_RAD_TO_DEG  = 180.0/QS_PI;
inline constexpr f64 QS_APERY       = 1.2020569031595942854;         // ζ(3)
inline constexpr f64 QS_CATALAN     = 0.9159655941772190;            // Catalan's constant
inline constexpr f64 QS_KHINCHIN    = 2.6854520010653064453;         // Khinchin constant
inline constexpr f64 QS_FEIGENBAUM1 = 4.6692016091029906718;        // δ (bifurcation)
inline constexpr f64 QS_FEIGENBAUM2 = 2.5029078750958928222;        // α
inline constexpr f64 QS_GLAISHER    = 1.2824271291006226368;        // Glaisher-Kinkelin A
inline constexpr f64 QS_OMEGA       = 0.5671432904097838729;        // Ω constant: Ω e^Ω = 1
inline constexpr f64 QS_TWIN_PRIME  = 0.6601618158468695739278;     // twin prime constant
inline constexpr f64 QS_LANDAU      = 0.7642236535892206;           // Landau-Ramanujan
inline constexpr f64 QS_MEISSEL     = 0.2615101;                    // Meissel-Mertens
inline constexpr f64 QS_LEVY        = 1.7722707580707447866321;     // Lévy constant for CF

// ─── Fundamental physical constants (CODATA 2018) ─────────────────────────────
// Speed of light in vacuum
inline constexpr f64 QS_C           = 2.99792458e8;        // m/s (exact)
// Planck constant
inline constexpr f64 QS_H           = 6.62607015e-34;       // J·s (exact)
inline constexpr f64 QS_HBAR        = QS_H/(2*QS_PI);       // ℏ = h/(2π)
// Elementary charge
inline constexpr f64 QS_E_CHARGE    = 1.602176634e-19;      // C (exact)
// Boltzmann constant
inline constexpr f64 QS_K_B         = 1.380649e-23;         // J/K (exact)
// Avogadro constant
inline constexpr f64 QS_N_A         = 6.02214076e23;        // mol⁻¹ (exact)
// Molar gas constant R = k_B N_A
inline constexpr f64 QS_R_GAS       = QS_K_B*QS_N_A;        // 8.314... J/(mol·K)
// Faraday constant F = e N_A
inline constexpr f64 QS_FARADAY     = QS_E_CHARGE*QS_N_A;   // 96485... C/mol
// Stefan-Boltzmann constant σ = 2π⁵k⁴/(15h³c²)
inline constexpr f64 QS_SIGMA_SB    = 5.670374419e-8;       // W/(m²·K⁴)
// Gravitational constant
inline constexpr f64 QS_G           = 6.67430e-11;          // m³/(kg·s²)
// Vacuum permeability μ₀
inline constexpr f64 QS_MU_0        = 1.25663706212e-6;     // H/m
// Vacuum permittivity ε₀ = 1/(μ₀c²)
inline constexpr f64 QS_EPS_0       = 8.8541878128e-12;     // F/m
// Fine-structure constant α = e²/(4πε₀ℏc)
inline constexpr f64 QS_ALPHA       = 7.2973525693e-3;      // dimensionless ≈ 1/137
// Bohr radius a₀ = ℏ/(mₑ c α)
inline constexpr f64 QS_BOHR_RADIUS = 5.29177210903e-11;    // m
// Electron mass
inline constexpr f64 QS_ME          = 9.1093837015e-31;     // kg
// Proton mass
inline constexpr f64 QS_MP          = 1.67262192369e-27;    // kg
// Neutron mass
inline constexpr f64 QS_MN          = 1.67492749804e-27;    // kg
// Proton-to-electron mass ratio
inline constexpr f64 QS_MP_ME       = QS_MP/QS_ME;          // ≈ 1836.15
// Atomic mass unit
inline constexpr f64 QS_AMU         = 1.66053906660e-27;    // kg
// Rydberg constant R_∞
inline constexpr f64 QS_RYDBERG     = 1.0973731568160e7;    // m⁻¹
// Compton wavelength λ_C = h/(mₑc)
inline constexpr f64 QS_COMPTON     = 2.42631023867e-12;    // m
// Magnetic flux quantum Φ₀ = h/(2e)
inline constexpr f64 QS_PHI_0       = QS_H/(2*QS_E_CHARGE); // Wb
// von Klitzing constant R_K = h/e²
inline constexpr f64 QS_R_K         = QS_H/(QS_E_CHARGE*QS_E_CHARGE); // Ω ≈ 25812.8
// Josephson constant K_J = 2e/h
inline constexpr f64 QS_K_J         = 2*QS_E_CHARGE/QS_H;  // Hz/V
// Standard gravity
inline constexpr f64 QS_G_STD       = 9.80665;              // m/s² (exact)
// Standard atmosphere
inline constexpr f64 QS_ATM         = 101325.0;             // Pa (exact)
// Triple point of water
inline constexpr f64 QS_T_TRIPLE    = 273.16;               // K (exact)
// Molar volume of ideal gas at STP
inline constexpr f64 QS_MOLAR_VOL   = 22.41396954e-3;       // m³/mol

// ─── Astronomical constants ───────────────────────────────────────────────────
inline constexpr f64 QS_AU          = 1.495978707e11;       // m (exact)
inline constexpr f64 QS_LY          = 9.4607304725808e15;   // m
inline constexpr f64 QS_PC          = 3.085677581491367e16; // m
inline constexpr f64 QS_SOLAR_MASS  = 1.98892e30;           // kg
inline constexpr f64 QS_SOLAR_RAD   = 6.957e8;              // m
inline constexpr f64 QS_SOLAR_LUM   = 3.828e26;             // W
inline constexpr f64 QS_SOLAR_TEFF  = 5778.0;               // K
inline constexpr f64 QS_EARTH_MASS  = 5.97219e24;           // kg
inline constexpr f64 QS_EARTH_RAD   = 6.3781e6;             // m (equatorial)
inline constexpr f64 QS_EARTH_J2    = 1.082636e-3;          // oblateness parameter
inline constexpr f64 QS_MOON_MASS   = 7.342e22;             // kg
inline constexpr f64 QS_MOON_RAD    = 1.7374e6;             // m
inline constexpr f64 QS_MOON_DIST   = 3.84400e8;            // m (mean)
inline constexpr f64 QS_SIDEREAL_DAY= 86164.0905;           // s
inline constexpr f64 QS_SOLAR_DAY   = 86400.0;              // s
inline constexpr f64 QS_JULIAN_YEAR = 365.25*86400.0;       // s
inline constexpr f64 QS_HUBBLE      = 70e3/(3.085677581e22);// s⁻¹ (H₀ ≈ 70 km/s/Mpc)

// ─── f32 versions of commonly used constants ─────────────────────────────────
inline constexpr f32 QS_PIf         = f32(QS_PI);
inline constexpr f32 QS_TAUf        = f32(QS_TAU);
inline constexpr f32 QS_Ef          = f32(QS_E);
inline constexpr f32 QS_PHIf        = f32(QS_PHI);
inline constexpr f32 QS_SQRT2f      = f32(QS_SQRT2);
inline constexpr f32 QS_INV_PIf     = f32(QS_INV_PI);
inline constexpr f32 QS_DEG_TO_RADf = f32(QS_DEG_TO_RAD);
inline constexpr f32 QS_RAD_TO_DEGf = f32(QS_RAD_TO_DEG);
inline constexpr f32 QS_G_STDf      = f32(QS_G_STD);

} // namespace qs
