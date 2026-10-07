#pragma once
// qs_math/spectral_irradiance.hpp — Spectral rendering pipeline, light spectra, sensor response (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "color_science.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <span>
#include <algorithm>
#include <functional>

namespace qs {

// ─── Spectral power distribution (full resolution) ───────────────────────────

struct FullSPD {
    static constexpr u32 N_BINS = 81;           // 360-720nm, 5nm step
    static constexpr f32 LAMBDA_MIN = 360.f;
    static constexpr f32 LAMBDA_MAX = 720.f;
    static constexpr f32 STEP = 5.f;

    f32 data[N_BINS]{};

    [[nodiscard]] f32 eval(f32 lambda) const {
        f32 t=(lambda-LAMBDA_MIN)/STEP;
        if(t<0.f||t>f32(N_BINS-1)) return 0.f;
        u32 i=u32(t); u32 j=std::min(i+1,N_BINS-1);
        return data[i]*(1.f-(t-f32(i)))+data[j]*(t-f32(i));
    }

    void scale(f32 s){ for(auto& v:data) v*=s; }
    void add(const FullSPD& o){ for(u32 i=0;i<N_BINS;++i) data[i]+=o.data[i]; }
    void mul(const FullSPD& o){ for(u32 i=0;i<N_BINS;++i) data[i]*=o.data[i]; }

    [[nodiscard]] f32 total_power() const {
        f32 s=0.f; for(auto v:data) s+=v; return s*STEP;
    }

    // Convert to CIE XYZ
    [[nodiscard]] Vec3f to_xyz() const {
        Vec3f xyz{0,0,0};
        for(u32 i=0;i<N_BINS;++i){
            f32 l=LAMBDA_MIN+f32(i)*STEP;
            xyz.x+=data[i]*cie_x(l)*STEP;
            xyz.y+=data[i]*cie_y(l)*STEP;
            xyz.z+=data[i]*cie_z(l)*STEP;
        }
        return xyz;
    }
};

// ─── Standard illuminants ─────────────────────────────────────────────────────

[[nodiscard]] inline FullSPD illuminant_D65() {
    // CIE D65 tabulated values (360-720nm, 5nm)
    static constexpr f32 d65[FullSPD::N_BINS]={
        82.75f,91.49f,93.43f,86.68f,104.86f,117.01f,117.81f,114.86f,115.92f,108.81f,
        109.35f,107.80f,104.79f,107.69f,104.40f,104.04f,100.00f,96.34f,95.79f,88.69f,
        90.01f,89.60f,87.70f,83.29f,83.70f,80.03f,80.21f,82.28f,78.28f,69.72f,
        71.61f,74.35f,61.60f,69.89f,75.09f,63.59f,46.42f,66.81f,63.38f,64.30f,
        59.45f,51.96f,57.45f,60.31f,62.96f,67.15f,68.70f,66.00f,61.00f,53.30f,
        58.90f,61.90f,62.10f,39.92f,48.00f,52.40f,45.40f,36.90f,36.60f,41.00f,
        37.10f,35.10f,30.90f,30.00f,32.10f,32.40f,28.10f,28.10f,27.80f,27.50f,
        25.60f,25.20f,23.70f,20.40f,20.10f,22.00f,22.20f,23.80f,20.20f,28.40f,33.00f
    };
    FullSPD spd; for(u32 i=0;i<FullSPD::N_BINS;++i) spd.data[i]=d65[i]; return spd;
}

[[nodiscard]] inline FullSPD illuminant_A() {
    // Planckian 2856K
    FullSPD spd;
    for(u32 i=0;i<FullSPD::N_BINS;++i){
        f32 l=(FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP)*1e-9f;
        f32 v=100.f*(std::exp(0.014387769f/(0.5600e-6f))-1.f)
                    /(std::exp(0.014387769f/l)-1.f)*std::pow(0.5600e-6f/l,5.f);
        spd.data[i]=v;
    }
    return spd;
}

[[nodiscard]] inline FullSPD illuminant_blackbody(f32 T_K) {
    FullSPD spd;
    for(u32 i=0;i<FullSPD::N_BINS;++i){
        f32 l=(FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP)*1e-9f;
        f32 c1=3.74183e-16f, c2=1.43878e-2f;
        spd.data[i]=c1/(std::pow(l,5.f)*(std::exp(c2/(l*T_K))-1.f));
    }
    // Normalize to peak
    f32 peak=*std::max_element(spd.data,spd.data+FullSPD::N_BINS);
    if(peak>0.f) for(auto& v:spd.data) v/=peak;
    return spd;
}

// ─── Camera sensor spectral response ─────────────────────────────────────────

struct SensorResponse {
    FullSPD R, G, B; // spectral response curves for each channel

    void set_standard_camera() {
        // Approximate Bayer sensor response (Gaussian approximations)
        for(u32 i=0;i<FullSPD::N_BINS;++i){
            f32 l=FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP;
            f32 rr=(l-620.f)/40.f; R.data[i]=std::exp(-rr*rr*0.5f);
            f32 gr=(l-540.f)/45.f; G.data[i]=std::exp(-gr*gr*0.5f);
            f32 br=(l-450.f)/35.f; B.data[i]=std::exp(-br*br*0.5f);
        }
    }

    [[nodiscard]] Vec3f capture(const FullSPD& illuminant, const FullSPD& reflectance) const {
        Vec3f rgb{0,0,0};
        for(u32 i=0;i<FullSPD::N_BINS;++i){
            f32 s=illuminant.data[i]*reflectance.data[i]*FullSPD::STEP;
            rgb.x+=R.data[i]*s;
            rgb.y+=G.data[i]*s;
            rgb.z+=B.data[i]*s;
        }
        return rgb;
    }
};

// ─── Metamerism ───────────────────────────────────────────────────────────────

// Check if two SPDs are metameric under illuminant (same XYZ within tolerance)
[[nodiscard]] inline bool are_metamers(
    const FullSPD& a, const FullSPD& b,
    const FullSPD& illuminant, f32 tolerance=0.5f)
{
    FullSPD la=illuminant, lb=illuminant;
    la.mul(a); lb.mul(b);
    Vec3f xa=la.to_xyz(), xb=lb.to_xyz();
    Vec3f diff=xa-xb;
    return length(diff)<tolerance;
}

// Metamerism index (CIE illuminant B vs D65)
[[nodiscard]] inline f32 metamerism_index(
    const FullSPD& a, const FullSPD& b)
{
    auto D65=illuminant_D65();
    auto A=illuminant_A();
    FullSPD la=D65, lb=D65, la2=A, lb2=A;
    la.mul(a); lb.mul(b); la2.mul(a); lb2.mul(b);
    Vec3f da=la.to_xyz()-lb.to_xyz();
    Vec3f da2=la2.to_xyz()-lb2.to_xyz();
    return length(da-da2);
}

// ─── Spectral upsampling (Otsu 2018) ─────────────────────────────────────────

// 3-coefficient basis for spectral upsampling
struct SpectralBasis3 {
    Vec3f coeffs{};  // c0, c1, c2

    [[nodiscard]] f32 eval(f32 lambda) const {
        // Sigmoid polynomial basis
        f32 t=(lambda-FullSPD::LAMBDA_MIN)/(FullSPD::LAMBDA_MAX-FullSPD::LAMBDA_MIN)*2.f-1.f;
        return 0.5f + coeffs.x*t + coeffs.y*(2.f*t*t-1.f) + coeffs.z*(4.f*t*t*t-3.f*t);
    }

    [[nodiscard]] FullSPD to_spd() const {
        FullSPD spd;
        for(u32 i=0;i<FullSPD::N_BINS;++i)
            spd.data[i]=std::clamp(eval(FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP),0.f,1.f);
        return spd;
    }
};

// ─── Light source spectral profiles ──────────────────────────────────────────

[[nodiscard]] inline FullSPD led_warm_white() {
    FullSPD spd{};
    // Blue pump peak + phosphor emission
    for(u32 i=0;i<FullSPD::N_BINS;++i){
        f32 l=FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP;
        f32 blue=(l-450.f)/20.f; f32 phosphor=(l-570.f)/60.f;
        spd.data[i]=std::exp(-blue*blue*0.5f)*0.3f+std::exp(-phosphor*phosphor*0.5f);
    }
    return spd;
}

[[nodiscard]] inline FullSPD sodium_vapor() {
    FullSPD spd{};
    // Dominant lines at 589nm (D1/D2)
    for(u32 i=0;i<FullSPD::N_BINS;++i){
        f32 l=FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP;
        f32 d=(l-589.f)/3.f;
        spd.data[i]=std::exp(-d*d*0.5f);
    }
    return spd;
}

[[nodiscard]] inline FullSPD fluorescent_cool_white() {
    FullSPD spd{};
    for(u32 i=0;i<FullSPD::N_BINS;++i){
        f32 l=FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP;
        // Multiple emission peaks typical of FL3
        f32 p1=(l-436.f)/8.f, p2=(l-547.f)/10.f, p3=(l-611.f)/8.f;
        f32 broad=(l-545.f)/80.f;
        spd.data[i]=std::exp(-p1*p1)*0.4f+std::exp(-p2*p2)*0.8f
                   +std::exp(-p3*p3)*0.6f+std::exp(-broad*broad)*0.3f;
    }
    return spd;
}

// ─── Color rendering index (CRI) approximation ───────────────────────────────

[[nodiscard]] inline f32 approximate_cri(const FullSPD& illuminant) {
    // Compare test illuminant to reference (D65 or blackbody at same CCT)
    auto ref=illuminant_D65();
    // Use 8 standard Munsell samples (simplified: random reflectances)
    f32 total_ri=0.f;
    for(u32 s=0;s<8;++s){
        FullSPD sample_refl{};
        for(u32 i=0;i<FullSPD::N_BINS;++i){
            f32 l=FullSPD::LAMBDA_MIN+f32(i)*FullSPD::STEP;
            f32 t=(l-380.f+f32(s)*40.f)/60.f;
            sample_refl.data[i]=std::clamp(std::sin(t*k_pi),0.f,1.f);
        }
        // XYZ under test vs reference
        FullSPD test_lit=illuminant, ref_lit=ref;
        test_lit.mul(sample_refl); ref_lit.mul(sample_refl);
        Vec3f xyz_t=test_lit.to_xyz(), xyz_r=ref_lit.to_xyz();
        // Normalize by Y
        if(xyz_t.y>0.f) xyz_t=xyz_t*(1.f/xyz_t.y);
        if(xyz_r.y>0.f) xyz_r=xyz_r*(1.f/xyz_r.y);
        f32 de=length(xyz_t-xyz_r)*100.f;
        total_ri+=100.f-4.6f*de;
    }
    return std::clamp(total_ri/8.f,0.f,100.f);
}

// Correlated color temperature (CCT) from XYZ
[[nodiscard]] inline f32 xyz_to_cct(const Vec3f& xyz) {
    // Robertson method
    if(xyz.x+xyz.y+xyz.z<1e-6f) return 6504.f;
    f32 xe=xyz.x/(xyz.x+xyz.y+xyz.z);
    f32 ye=xyz.y/(xyz.x+xyz.y+xyz.z);
    f32 n=(xe-0.3320f)/(0.1858f-ye);
    return 449.f*n*n*n+3525.f*n*n+6823.3f*n+5520.33f;
}

} // namespace qs
