#pragma once
// qs_math/visual_perception.hpp — Contrast sensitivity, eye adaptation, perceptual metrics (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "color.hpp"
#include "color_science.hpp"
#include <cmath>
#include <array>
#include <algorithm>
#include <vector>
#include <span>

namespace qs {

// ─── Eye adaptation ───────────────────────────────────────────────────────────

struct EyeAdaptation {
    f32 adapted_luminance{100.f};  // cd/m² — current adapted level
    f32 tau_dark{0.4f};            // time constant dark adaptation (s)
    f32 tau_bright{0.1f};          // time constant bright adaptation (s)
    f32 min_luminance{0.001f};
    f32 max_luminance{100000.f};

    void update(f32 scene_luminance, f32 dt) {
        f32 tau = scene_luminance < adapted_luminance ? tau_dark : tau_bright;
        f32 alpha = 1.f - std::exp(-dt / tau);
        adapted_luminance += (scene_luminance - adapted_luminance) * alpha;
        adapted_luminance = std::clamp(adapted_luminance, min_luminance, max_luminance);
    }

    // Key value for auto-exposure (Reinhard)
    [[nodiscard]] f32 key_value() const {
        return 1.03f - 2.f / (2.f + std::log10(adapted_luminance + 1.f));
    }

    // Exposure value in EV
    [[nodiscard]] f32 exposure_ev() const {
        return std::log2(adapted_luminance / 12.5f);
    }

    // Linear exposure scale
    [[nodiscard]] f32 exposure_scale() const {
        return key_value() / (adapted_luminance + 1e-6f);
    }
};

// ─── Contrast sensitivity function (CSF) ─────────────────────────────────────

// Barten 1999 CSF model
[[nodiscard]] inline f32 barten_csf(
    f32 spatial_freq_cpd,     // cycles per degree
    f32 luminance_cdm2=100.f, // adaptation luminance
    f32 field_size_deg=10.f)  // stimulus size
{
    f32 a = 440.f*(1.f+0.7f/luminance_cdm2);
    f32 b = 0.3f*(1.f+100.f/luminance_cdm2);
    f32 u = spatial_freq_cpd;
    f32 csf = a * u * std::exp(-b * u) * std::sqrt(1.f + 0.06f * std::exp(b * u));
    // Field size correction
    csf *= std::sqrt(field_size_deg / 10.f);
    return std::max(0.f, csf);
}

// Threshold contrast at spatial frequency
[[nodiscard]] inline f32 threshold_contrast(f32 cpd, f32 luminance=100.f) {
    f32 csf = barten_csf(cpd, luminance);
    return csf > 0.f ? 1.f / csf : 1.f;
}

// Just-noticeable difference (JND) in luminance
[[nodiscard]] inline f32 jnd_luminance(f32 L) {
    // Weber fraction approximation
    if(L < 0.01f) return 0.01f;
    if(L < 1.f)   return 0.02f * std::pow(L, 0.7f);
    if(L < 100.f) return 0.01f * L;
    return 0.007f * L;
}

// ─── CIELAB and perceptual difference ────────────────────────────────────────

// CIE76 color difference (Lab)
[[nodiscard]] inline f32 delta_e_76(const Vec3f& lab1, const Vec3f& lab2) {
    Vec3f d = lab1 - lab2;
    return std::sqrt(dot(d, d));
}

// CIE94 color difference
[[nodiscard]] inline f32 delta_e_94(const Vec3f& lab1, const Vec3f& lab2,
    bool graphic_arts = true)
{
    f32 kL = 1.f, K1 = graphic_arts ? 0.045f : 0.048f, K2 = graphic_arts ? 0.015f : 0.014f;
    f32 dL = lab1.x - lab2.x;
    f32 C1 = std::sqrt(lab1.y*lab1.y + lab1.z*lab1.z);
    f32 C2 = std::sqrt(lab2.y*lab2.y + lab2.z*lab2.z);
    f32 dC = C1 - C2;
    f32 da = lab1.y - lab2.y, db = lab1.z - lab2.z;
    f32 dH2 = da*da + db*db - dC*dC;
    f32 SL = 1.f, SC = 1.f + K1*C1, SH = 1.f + K2*C1;
    return std::sqrt((dL/(kL*SL))*(dL/(kL*SL)) + (dC/SC)*(dC/SC) + dH2/(SH*SH));
}

// CIEDE2000 (full)
[[nodiscard]] inline f32 delta_e_2000(const Vec3f& lab1, const Vec3f& lab2) {
    f32 L1=lab1.x, a1=lab1.y, b1=lab1.z;
    f32 L2=lab2.x, a2=lab2.y, b2=lab2.z;
    f32 C1=std::sqrt(a1*a1+b1*b1), C2=std::sqrt(a2*a2+b2*b2);
    f32 Cm=(C1+C2)*0.5f;
    f32 Cm7=std::pow(Cm,7.f);
    f32 G=0.5f*(1.f-std::sqrt(Cm7/(Cm7+6103515625.f)));
    f32 ap1=a1*(1.f+G), ap2=a2*(1.f+G);
    f32 Cp1=std::sqrt(ap1*ap1+b1*b1), Cp2=std::sqrt(ap2*ap2+b2*b2);
    f32 hp1=std::atan2(b1,ap1), hp2=std::atan2(b2,ap2);
    if(hp1<0.f) hp1+=k_tau; if(hp2<0.f) hp2+=k_tau;
    f32 dLp=L2-L1, dCp=Cp2-Cp1;
    f32 dhp=hp2-hp1;
    if(std::abs(dhp)>k_pi) dhp+=dhp>0?-k_tau:k_tau;
    f32 dHp=2.f*std::sqrt(Cp1*Cp2)*std::sin(dhp*0.5f);
    f32 Lp=(L1+L2)*0.5f, Cp=(Cp1+Cp2)*0.5f;
    f32 hp=(std::abs(hp1-hp2)>k_pi)?(hp1+hp2+k_tau)*0.5f:(hp1+hp2)*0.5f;
    f32 T=1.f-0.17f*std::cos(hp-0.5236f)+0.24f*std::cos(2.f*hp)+0.32f*std::cos(3.f*hp+0.1047f)-0.20f*std::cos(4.f*hp-1.0996f);
    f32 SL=1.f+0.015f*(Lp-50.f)*(Lp-50.f)/std::sqrt(20.f+(Lp-50.f)*(Lp-50.f));
    f32 SC=1.f+0.045f*Cp, SH=1.f+0.015f*Cp*T;
    f32 Cp7=std::pow(Cp,7.f);
    f32 RC=2.f*std::sqrt(Cp7/(Cp7+6103515625.f));
    f32 d_theta=0.5236f*std::exp(-((hp*180.f/k_pi-275.f)/25.f)*((hp*180.f/k_pi-275.f)/25.f));
    f32 RT=-RC*std::sin(2.f*d_theta);
    return std::sqrt((dLp/SL)*(dLp/SL)+(dCp/SC)*(dCp/SC)+(dHp/SH)*(dHp/SH)+RT*(dCp/SC)*(dHp/SH));
}

// ─── Perceptual image metrics ─────────────────────────────────────────────────

// Mean Opinion Score (MOS) from PSNR approximation
[[nodiscard]] inline f32 psnr_to_mos(f32 psnr_db) {
    // Sigmoid-like mapping
    return 1.f + 4.f / (1.f + std::exp(-0.3f*(psnr_db - 35.f)));
}

// PSNR in dB
[[nodiscard]] inline f32 psnr(std::span<const f32> img, std::span<const f32> ref, f32 max_val=1.f) {
    u32 n=u32(std::min(img.size(),ref.size()));
    if(!n) return 0.f;
    f32 mse=0.f;
    for(u32 i=0;i<n;++i){f32 d=img[i]-ref[i]; mse+=d*d;}
    mse/=f32(n);
    return mse>0.f?10.f*std::log10(max_val*max_val/mse):-1.f;
}

// ─── Chromostereopsis (depth from color) ─────────────────────────────────────

// Red appears to advance, blue recedes
[[nodiscard]] inline f32 chromostereopsis_depth(const Vec3f& srgb) {
    // Approximate perceptual depth cue from chromatic aberration of eye lens
    return (srgb.x - srgb.z) * 0.5f; // [-0.5, 0.5]
}

// ─── Saccade and fixation model ───────────────────────────────────────────────

struct GazeModel {
    Vec2f fixation{0.5f, 0.5f}; // screen space [0,1]
    f32   saccade_threshold{0.05f}; // deg (stimuli farther than this trigger saccade)
    f32   saccade_duration{0.04f};  // seconds
    f32   fixation_duration{0.25f}; // typical fixation time (seconds)
    f32   t{0.f};
    bool  in_saccade{false};
    Vec2f saccade_target{};

    void update(f32 dt, Vec2f stimulus) {
        t += dt;
        if(in_saccade) {
            if(t >= saccade_duration) { fixation = saccade_target; in_saccade = false; t = 0.f; }
        } else {
            if(t >= fixation_duration && length(stimulus - fixation) > saccade_threshold) {
                saccade_target = stimulus; in_saccade = true; t = 0.f;
            }
        }
    }

    // Acuity falloff: resolution drops with eccentricity
    [[nodiscard]] f32 acuity_at(Vec2f pos) const {
        f32 ecc = length(pos - fixation) * 50.f; // degrees (approx)
        return 1.f / (1.f + 0.12f * ecc);        // Anstis formula
    }
};

// ─── HDR display mapping ──────────────────────────────────────────────────────

struct HDRDisplayParams {
    f32 peak_luminance{1000.f};   // cd/m² (display peak)
    f32 min_luminance{0.001f};    // black level
    f32 max_content_cll{4000.f};  // content max frame average light level
    f32 reference_white{203.f};   // nits for SDR white in HLG/PQ

    // PQ (ST.2084) encode: linear -> PQ
    [[nodiscard]] f32 pq_encode(f32 L_linear) const {
        static constexpr f32 m1=0.1593017578125f,m2=78.84375f;
        static constexpr f32 c1=0.8359375f,c2=18.8515625f,c3=18.6875f;
        f32 Y=std::clamp(L_linear/10000.f,0.f,1.f);
        f32 Ym=std::pow(Y,m1);
        return std::pow((c1+c2*Ym)/(1.f+c3*Ym),m2);
    }

    [[nodiscard]] f32 pq_decode(f32 N) const {
        static constexpr f32 m1=0.1593017578125f,m2=78.84375f;
        static constexpr f32 c1=0.8359375f,c2=18.8515625f,c3=18.6875f;
        f32 Nm=std::pow(N,1.f/m2);
        return 10000.f*std::pow(std::max(0.f,Nm-c1)/(c2-c3*Nm),1.f/m1);
    }

    // HLG encode
    [[nodiscard]] f32 hlg_encode(f32 L_linear) const {
        static constexpr f32 a=0.17883277f,b=0.28466892f,c=0.55991073f;
        f32 E=L_linear/reference_white;
        return E<=1.f/12.f ? std::sqrt(3.f*E) : a*std::log(12.f*E-b)+c;
    }
};

} // namespace qs
