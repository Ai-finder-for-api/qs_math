#pragma once
// qs_math/rendering/volumetric.hpp — Volume rendering, phase functions, transmittance, VDB math
#include "../arithmetic/scalar_types.hpp"
#include "pbr_math.hpp"
#include <cmath>
#include <array>

namespace qs::math {

// ─── Phase functions ──────────────────────────────────────────────────────────
// Henyey-Greenstein: p(cosθ) = (1-g²)/(4π(1+g²-2g*cosθ)^{3/2})
[[nodiscard]] constexpr f32 hg_phase(f32 cos_theta, f32 g) noexcept {
    f32 denom=1+g*g-2*g*cos_theta;
    return (1-g*g)/(4*k_pi_f*std::pow(denom,1.5f)+1e-7f);
}
// Sample HG phase function
[[nodiscard]] constexpr f32 hg_sample_cos(f32 xi, f32 g) noexcept {
    if(std::abs(g)<1e-4f) return 1-2*xi;
    f32 s=(1-g*g)/(1-g+2*g*xi);
    return (1+g*g-s*s)/(2*g);
}
// Multiple scattering blend (Schlick approximation)
[[nodiscard]] constexpr f32 schlick_phase(f32 cos_theta, f32 k) noexcept {
    f32 d=1-k*cos_theta;
    return (1-k*k)/(4*k_pi_f*d*d);
}
// Mie scattering (Lorenz-Mie, simplified for droplets)
[[nodiscard]] constexpr f32 mie_phase_approx(f32 cos_theta) noexcept {
    // Two-lobe HG blend for water droplets
    return 0.5f*hg_phase(cos_theta,0.8f)+0.5f*hg_phase(cos_theta,-0.5f);
}
// Rayleigh scattering: p ∝ (1 + cos²θ)
[[nodiscard]] constexpr f32 rayleigh_phase(f32 cos_theta) noexcept {
    return 3*(1+cos_theta*cos_theta)/(16*k_pi_f);
}

// ─── Beer-Lambert transmittance ───────────────────────────────────────────────
[[nodiscard]] constexpr f32 transmittance(f32 sigma_t, f32 distance) noexcept {
    return std::exp(-sigma_t*distance);
}
[[nodiscard]] constexpr f32 transmittance_rgb(f32 sigma_r,f32 sigma_g,f32 sigma_b,f32 d) noexcept {
    return (transmittance(sigma_r,d)+transmittance(sigma_g,d)+transmittance(sigma_b,d))/3;
}
// Free path sampling: t = -log(ξ)/σ_t
[[nodiscard]] constexpr f32 sample_free_path(f32 xi, f32 sigma_t) noexcept {
    return -std::log(1-xi+1e-7f)/sigma_t;
}
[[nodiscard]] constexpr f32 free_path_pdf(f32 t, f32 sigma_t) noexcept {
    return sigma_t*std::exp(-sigma_t*t);
}

// ─── Volume rendering equation ────────────────────────────────────────────────
// L(x,ω) = ∫_0^{t_max} T(0,t)*σ_s(x_t)*L_scatter(x_t,ω) dt + T(0,t_max)*L_bg
// Single-step transmittance update
[[nodiscard]] constexpr R3 volume_render_step(
    R3 L, R3 T_accum,        // current radiance and transmittance
    R3 emission,              // L_e at sample point
    R3 in_scatter,            // L_i (environment/sun) at sample point
    f32 sigma_a, f32 sigma_s, // absorption and scattering coefficients
    f32 dt) noexcept          // step size
{
    f32 sigma_t=sigma_a+sigma_s;
    f32 T_step=transmittance(sigma_t,dt);
    // Emission contribution: T*(1-T_step)/sigma_t * sigma_a * Le
    R3 contrib_e=emission*(1-T_step)*(sigma_a/(sigma_t+1e-7f));
    // Scatter contribution
    R3 contrib_s=in_scatter*(1-T_step)*(sigma_s/(sigma_t+1e-7f));
    return L+T_accum*(contrib_e+contrib_s);
}

// ─── Delta tracking (Woodcock tracking) ───────────────────────────────────────
// Used for heterogeneous media with majorant σ_maj ≥ σ_t(x)
[[nodiscard]] constexpr bool woodcock_collision(f32 sigma_t, f32 sigma_maj, f32 xi) noexcept {
    // Returns true = real collision, false = null collision
    return xi < sigma_t/sigma_maj;
}

// ─── Spectral volume rendering ────────────────────────────────────────────────
// Hero wavelength sampling: pick wavelength λ_h, scatter others deterministically
[[nodiscard]] constexpr std::array<f32,4> hero_wavelengths(f32 lambda_h, f32 lambda_min=380, f32 lambda_max=780) noexcept {
    f32 range=lambda_max-lambda_min;
    return{lambda_h,
           std::fmod(lambda_h+range*0.25f-lambda_min,range)+lambda_min,
           std::fmod(lambda_h+range*0.50f-lambda_min,range)+lambda_min,
           std::fmod(lambda_h+range*0.75f-lambda_min,range)+lambda_min};
}

// ─── Atmospheric scattering ───────────────────────────────────────────────────
// Rayleigh extinction: σ_R(λ) ∝ 1/λ⁴
[[nodiscard]] constexpr f32 rayleigh_sigma(f32 lambda_nm) noexcept {
    constexpr f32 lambda0=550; // nm reference
    f32 r=lambda0/lambda_nm;
    return r*r*r*r*5.8e-6f; // approximate coefficient at sea level
}
// Mie extinction (approximate, wavelength-independent for large particles)
[[nodiscard]] constexpr f32 mie_sigma(f32 altitude_km, f32 sigma0=21e-6f) noexcept {
    return sigma0*std::exp(-altitude_km/1.2f);
}
// Ozone absorption (peaked around 600nm)
[[nodiscard]] constexpr f32 ozone_sigma(f32 lambda_nm) noexcept {
    f32 d=(lambda_nm-600)/50; return 3.426e-4f*std::exp(-d*d);
}
// Atmospheric density profile (exponential)
[[nodiscard]] constexpr f32 atm_density(f32 altitude_km, f32 scale_height=8.5f) noexcept {
    return std::exp(-altitude_km/scale_height);
}
// Optical depth along path (integrated density)
[[nodiscard]] constexpr f32 optical_depth_rayleigh(f32 cos_zenith, f32 altitude_km) noexcept {
    return rayleigh_sigma(550)*8500*atm_density(altitude_km)/std::max(std::abs(cos_zenith),0.01f);
}

// ─── VDB-style sparse volume ──────────────────────────────────────────────────
// Root → internal → leaf node indices for uniform sparse grids
[[nodiscard]] constexpr std::array<int,3> vdb_coord(f32 x,f32 y,f32 z,f32 dx) noexcept {
    return{int(std::floor(x/dx)),int(std::floor(y/dx)),int(std::floor(z/dx))};
}
[[nodiscard]] constexpr u64 vdb_hash(int ix,int iy,int iz) noexcept {
    return (u64(ix)*73856093)^(u64(iy)*19349663)^(u64(iz)*83492791);
}

// ─── Phase function MIS (balance heuristic for phase+light) ───────────────────
[[nodiscard]] constexpr f32 volume_mis_weight(f32 pdf_phase, f32 pdf_light) noexcept {
    return pdf_phase/(pdf_phase+pdf_light+1e-7f);
}

// ─── Chromatic dispersion ─────────────────────────────────────────────────────
// Cauchy dispersion formula: n(λ) = A + B/λ² + C/λ⁴
[[nodiscard]] constexpr f32 cauchy_ior(f32 lambda_um, f32 A, f32 B, f32 C=0) noexcept {
    return A+B/(lambda_um*lambda_um)+C/(lambda_um*lambda_um*lambda_um*lambda_um);
}
// Sellmeier dispersion
[[nodiscard]] constexpr f32 sellmeier_ior(f32 lambda_um,
    const std::array<f32,3>& B_coeff, const std::array<f32,3>& C_coeff) noexcept
{
    f32 lam2=lambda_um*lambda_um, n2=1;
    for(int i=0;i<3;++i) n2+=B_coeff[i]*lam2/(lam2-C_coeff[i]);
    return std::sqrt(n2);
}
// Common glass: BK7 Sellmeier coefficients
[[nodiscard]] constexpr f32 bk7_ior(f32 lambda_um) noexcept {
    return sellmeier_ior(lambda_um,
        {1.03961212f,0.23179234f,1.01046945f},
        {0.00600069867f,0.0200179144f,103.560653f});
}
} // namespace qs::math
