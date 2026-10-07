#pragma once
// qs_math/rendering/spectral_render.hpp — Spectral rendering: hero wavelength, upsampling,
//                                          fluorescence, XYZ/RGB conversions, blackbody (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec4.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <cmath>
#include <array>

namespace qs::math::render {

// ─── Wavelength range & sampling ──────────────────────────────────────────
inline constexpr f32 LAMBDA_MIN = 360.f;   // nm
inline constexpr f32 LAMBDA_MAX = 830.f;   // nm
inline constexpr f32 LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;

[[nodiscard]] constexpr f32 lambda_to_t(f32 lambda_nm) noexcept {
    return (lambda_nm - LAMBDA_MIN) / LAMBDA_RANGE;
}

// ─── CIE 1931 2° observer (analytic fit — Wyman et al. 2013) ──────────────
[[nodiscard]] inline f32 cie_x(f32 l) noexcept {
    f32 t1 = (l - 442.0f) * (l < 442.f ? 0.0624f : 0.0374f);
    f32 t2 = (l - 599.8f) * (l < 599.8f ? 0.0264f : 0.0323f);
    f32 t3 = (l - 501.1f) * (l < 501.1f ? 0.0490f : 0.0382f);
    return 0.362f*std::exp(-0.5f*t1*t1) + 1.056f*std::exp(-0.5f*t2*t2)
          -0.065f*std::exp(-0.5f*t3*t3);
}
[[nodiscard]] inline f32 cie_y(f32 l) noexcept {
    f32 t1 = (l - 568.8f) * (l < 568.8f ? 0.0213f : 0.0247f);
    f32 t2 = (l - 530.9f) * (l < 530.9f ? 0.0613f : 0.0322f);
    return 0.821f*std::exp(-0.5f*t1*t1) + 0.286f*std::exp(-0.5f*t2*t2);
}
[[nodiscard]] inline f32 cie_z(f32 l) noexcept {
    f32 t1 = (l - 437.0f) * (l < 437.f ? 0.0845f : 0.0278f);
    f32 t2 = (l - 459.0f) * (l < 459.f ? 0.0385f : 0.0725f);
    return 1.217f*std::exp(-0.5f*t1*t1) + 0.681f*std::exp(-0.5f*t2*t2);
}

[[nodiscard]] inline Vec3f wavelength_to_xyz(f32 lambda_nm) noexcept {
    return {cie_x(lambda_nm), cie_y(lambda_nm), cie_z(lambda_nm)};
}

// ─── XYZ <-> sRGB (D65) ───────────────────────────────────────────────────
[[nodiscard]] inline Vec3f xyz_to_srgb_linear(Vec3f xyz) noexcept {
    return {
         3.2404542f*xyz.x - 1.5371385f*xyz.y - 0.4985314f*xyz.z,
        -0.9692660f*xyz.x + 1.8760108f*xyz.y + 0.0415560f*xyz.z,
         0.0556434f*xyz.x - 0.2040259f*xyz.y + 1.0572252f*xyz.z
    };
}
[[nodiscard]] inline Vec3f srgb_linear_to_xyz(Vec3f rgb) noexcept {
    return {
        0.4124564f*rgb.x + 0.3575761f*rgb.y + 0.1804375f*rgb.z,
        0.2126729f*rgb.x + 0.7151522f*rgb.y + 0.0721750f*rgb.z,
        0.0193339f*rgb.x + 0.1191920f*rgb.y + 0.9503041f*rgb.z
    };
}
[[nodiscard]] inline Vec3f xyz_to_srgb_gamma(Vec3f xyz) noexcept {
    Vec3f lin = xyz_to_srgb_linear(xyz);
    auto g = [](f32 c) noexcept -> f32 {
        return c <= 0.0031308f ? 12.92f*c : 1.055f*std::pow(c, 1.f/2.4f) - 0.055f;
    };
    return {g(lin.x), g(lin.y), g(lin.z)};
}

// ─── Hero-wavelength MIS ──────────────────────────────────────────────────
// Samples 4 hero wavelengths with stratified jitter for chromatic effects.
struct HeroWavelength {
    f32 lambda[4];  // nm
    f32 pdf;

    static HeroWavelength sample(f32 u) noexcept {
        HeroWavelength hw;
        hw.pdf = 1.f / LAMBDA_RANGE;
        f32 base = LAMBDA_MIN + u * LAMBDA_RANGE;
        for (int i = 0; i < 4; ++i)
            hw.lambda[i] = LAMBDA_MIN + fmod(base - LAMBDA_MIN + f32(i) * LAMBDA_RANGE * 0.25f, LAMBDA_RANGE);
        return hw;
    }

    // Accumulate 4 spectral radiance samples into XYZ
    [[nodiscard]] Vec3f to_xyz(const f32 radiance[4]) const noexcept {
        Vec3f xyz{0,0,0};
        for (int i = 0; i < 4; ++i) {
            xyz.x += cie_x(lambda[i]) * radiance[i];
            xyz.y += cie_y(lambda[i]) * radiance[i];
            xyz.z += cie_z(lambda[i]) * radiance[i];
        }
        return xyz * (LAMBDA_RANGE / (4.f * pdf * LAMBDA_RANGE));
    }
};

// ─── Spectral upsampling (Smits 1999 / Jakob-Hanika 2019 style) ──────────
// Approximate RGB -> spectral power distribution via sigmoid polynomial.
// Coefficients from Otsu et al. 2018 (3-coefficient model).

struct SpectralCoeffs { f32 c0, c1, c2; };

[[nodiscard]] inline f32 eval_spectral_poly(SpectralCoeffs s, f32 lambda) noexcept {
    // Maps wavelength to [0,1] and evaluates sigmoid(c0 + c1*x + c2*x^2)
    f32 x = (lambda - LAMBDA_MIN) / LAMBDA_RANGE;
    f32 v = s.c0 + x*(s.c1 + x*s.c2);
    return 0.5f + v / (2.f * std::sqrt(1.f + v*v)); // fast sigmoid via smooth division
}

// Approximate inversion: sRGB -> spectral coefficients (simplified)
[[nodiscard]] inline SpectralCoeffs srgb_to_spectral_coeffs(Vec3f rgb) noexcept {
    // Clamp and avoid degenerate black
    rgb.x = max(rgb.x, 1e-4f); rgb.y = max(rgb.y, 1e-4f); rgb.z = max(rgb.z, 1e-4f);
    // Heuristic: use log ratios to determine tilt/curvature
    f32 log_r = std::log(rgb.x), log_g = std::log(rgb.y), log_b = std::log(rgb.z);
    f32 c1 = (log_r - log_b);                   // slope: red vs blue
    f32 c2 = (log_r - 2.f*log_g + log_b)*2.f;   // curvature: green peak
    f32 c0 = log_g - c1*0.5f - c2*0.25f;        // midpoint offset
    return {c0, c1, c2};
}

// ─── Planck blackbody spectral radiance ───────────────────────────────────
// Returns spectral radiance W/(m^2 * sr * m) at wavelength (nm) and temp (K)
[[nodiscard]] inline f64 planck_radiance(f64 lambda_nm, f64 T) noexcept {
    constexpr f64 h  = 6.62607015e-34;   // Planck constant
    constexpr f64 c  = 299792458.0;      // speed of light
    constexpr f64 kb = 1.380649e-23;     // Boltzmann constant
    f64 lambda = lambda_nm * 1e-9;       // m
    f64 c2_over_lambda_T = (h*c) / (kb * lambda * T);
    return (2.0*h*c*c) / (lambda*lambda*lambda*lambda*lambda * (std::exp(c2_over_lambda_T) - 1.0));
}

// Wien's displacement law: peak wavelength in nm
[[nodiscard]] inline f64 wien_peak_nm(f64 T) noexcept {
    return 2.897771955e6 / T; // b/T, b = 2897771.955 nm·K
}

// Blackbody color to linear sRGB (normalized to luminance 1 at peak)
[[nodiscard]] inline Vec3f blackbody_to_srgb(f32 T) noexcept {
    // Sample 16 wavelengths and integrate
    Vec3f xyz{0,0,0};
    f32 norm = 0.f;
    for (int i = 0; i < 16; ++i) {
        f32 l = LAMBDA_MIN + (f32(i) + 0.5f) * LAMBDA_RANGE / 16.f;
        f32 rad = f32(planck_radiance(f64(l), f64(T)));
        xyz.x += cie_x(l) * rad;
        xyz.y += cie_y(l) * rad;
        xyz.z += cie_z(l) * rad;
        norm  += rad;
    }
    if (norm < 1e-30f) return {1,1,1};
    Vec3f rgb = xyz_to_srgb_linear(xyz * (1.f/norm));
    // Clamp negative chromaticities and normalize
    rgb.x = max(0.f, rgb.x); rgb.y = max(0.f, rgb.y); rgb.z = max(0.f, rgb.z);
    f32 peak = max({rgb.x, rgb.y, rgb.z});
    return peak > EPS_F32 ? rgb * (1.f/peak) : Vec3f{1,1,1};
}

// ─── Fluorescence shift ───────────────────────────────────────────────────
// Stokes shift: absorb at lambda_ex, emit at lambda_em (lambda_em > lambda_ex)
struct FluorophoreParams {
    f32 excitation_nm   = 490.f;
    f32 emission_nm     = 525.f;
    f32 bandwidth_ex    = 20.f;
    f32 bandwidth_em    = 30.f;
    f32 quantum_yield   = 0.8f;  // fraction of absorbed photons re-emitted
    f32 molar_extinction= 1.0f;  // relative absorption strength
};

[[nodiscard]] inline f32 fluorophore_excitation(const FluorophoreParams& f, f32 lambda) noexcept {
    f32 d = (lambda - f.excitation_nm) / f.bandwidth_ex;
    return f.molar_extinction * std::exp(-0.5f*d*d);
}
[[nodiscard]] inline f32 fluorophore_emission(const FluorophoreParams& f, f32 lambda) noexcept {
    f32 d = (lambda - f.emission_nm) / f.bandwidth_em;
    return f.quantum_yield * std::exp(-0.5f*d*d);
}

// Contribution of fluorescence to exitant radiance at lambda_em,
// given incident spectral radiance I(lambda_ex).
template<typename IncidentSPD>
[[nodiscard]] inline f32 fluorescent_radiance(
    const FluorophoreParams& fp, f32 lambda_em, IncidentSPD&& spd) noexcept
{
    // Integrate over excitation band (trapezoidal, 8 samples)
    f32 sum = 0.f;
    for (int i = 0; i < 8; ++i) {
        f32 l = fp.excitation_nm - 3.f*fp.bandwidth_ex + f32(i) * (6.f*fp.bandwidth_ex/7.f);
        sum += fluorophore_excitation(fp, l) * spd(l);
    }
    return fluorophore_emission(fp, lambda_em) * sum * (6.f*fp.bandwidth_ex/7.f);
}

// ─── Dispersion (Cauchy equation) ─────────────────────────────────────────
// n(lambda) = A + B/lambda^2 + C/lambda^4 (lambda in micrometers)
struct CauchyCoeffs { f32 A, B, C; };

// Common glass types
inline constexpr CauchyCoeffs GLASS_BK7   = {1.5046f, 0.00420f, 0.f};
inline constexpr CauchyCoeffs GLASS_FUSED_SILICA = {1.4580f, 0.00354f, 0.f};
inline constexpr CauchyCoeffs GLASS_SAPPHIRE     = {1.7512f, 0.00879f, 0.f};
inline constexpr CauchyCoeffs GLASS_DIAMOND      = {2.3780f, 0.01342f, 0.f};

[[nodiscard]] inline f32 cauchy_ior(CauchyCoeffs c, f32 lambda_nm) noexcept {
    f32 l = lambda_nm * 1e-3f; // nm -> um
    return c.A + c.B/(l*l) + c.C/(l*l*l*l);
}

// Compute dispersed refraction direction for each wavelength in hero set
[[nodiscard]] inline void disperse_refract(
    Vec3f incident, Vec3f normal, CauchyCoeffs glass,
    const HeroWavelength& hw, Vec3f out[4]) noexcept
{
    for (int i = 0; i < 4; ++i) {
        f32 eta = 1.f / cauchy_ior(glass, hw.lambda[i]);
        f32 cos_i = -dot(normal, incident);
        f32 sin2_t = eta*eta*(1.f - cos_i*cos_i);
        if (sin2_t > 1.f) { out[i] = {0,0,0}; continue; } // TIR
        f32 cos_t = std::sqrt(1.f - sin2_t);
        out[i] = incident*eta + normal*(eta*cos_i - cos_t);
    }
}

// ─── Spectral irradiance table ─────────────────────────────────────────────
// Piecewise linear SPD stored at 5nm steps (94 samples, 360-830nm)
struct SpectralTable {
    static constexpr int STEPS = 95;
    static constexpr f32 STEP  = 5.f;
    f32 values[STEPS] = {};

    void clear() noexcept { for(auto& v:values) v=0.f; }

    [[nodiscard]] f32 sample(f32 lambda_nm) const noexcept {
        f32 t = (lambda_nm - LAMBDA_MIN) / STEP;
        int i = i32(t); t -= f32(i);
        i = clamp(i, 0, STEPS-2);
        return lerp(values[i], values[i+1], t);
    }

    // Integrate to XYZ using CIE CMFs (trapezoidal)
    [[nodiscard]] Vec3f to_xyz() const noexcept {
        Vec3f xyz{0,0,0};
        for (int i = 0; i < STEPS; ++i) {
            f32 l = LAMBDA_MIN + f32(i) * STEP;
            xyz.x += cie_x(l) * values[i];
            xyz.y += cie_y(l) * values[i];
            xyz.z += cie_z(l) * values[i];
        }
        return xyz * STEP;
    }
};

// D65 illuminant (normalized to Y=1)
[[nodiscard]] inline SpectralTable d65_illuminant() noexcept {
    // CIE D65 relative SPD at 5nm steps, 360-830nm (truncated/normalized)
    static constexpr f32 d65[95] = {
        46.6f,52.1f,49.9f,54.6f,82.8f,91.5f,93.4f,86.7f,104.9f,117.0f,
        117.8f,114.9f,115.9f,108.8f,109.4f,107.8f,104.8f,107.7f,104.4f,104.0f,
        100.0f,96.3f,95.8f,97.4f,98.9f,95.6f,96.6f,97.1f,102.1f,100.8f,
        99.8f,98.0f,99.0f,97.2f,98.0f,100.1f,100.8f,99.0f,99.9f,99.6f,
        100.3f,99.0f,100.0f,99.6f,98.9f,97.8f,97.3f,98.0f,97.9f,96.8f,
        97.4f,97.3f,97.0f,96.4f,96.4f,96.5f,98.5f,98.9f,99.2f,99.0f,
        99.9f,100.6f,101.0f,100.9f,101.1f,100.5f,99.8f,99.0f,98.4f,97.2f,
        96.7f,95.9f,95.0f,93.9f,92.8f,91.6f,90.8f,90.2f,89.5f,88.7f,
        88.3f,87.5f,86.7f,86.3f,85.6f,84.7f,83.7f,82.7f,81.8f,80.6f,
        79.9f,79.1f,78.3f,77.5f,76.7f
    };
    SpectralTable t;
    f32 norm = d65[40]; // normalize at 560nm
    for (int i = 0; i < 95; ++i) t.values[i] = d65[i] / norm;
    return t;
}

} // namespace qs::math::render
