#pragma once
// color_science.hpp — Deep color science: spectral, HDR tone mapping,
//                     chromatic adaptation, color matching functions,
//                     ICC profile helpers, color space matrices,
//                     wide gamut (DCI-P3, Rec.2020, ACEScg), exposure
#include "color.hpp"
#include "mat3.hpp"
#include "vec3.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Color space matrices (RGB ↔ XYZ, D65)
// ---------------------------------------------------------------------------
// sRGB → XYZ (D65)
inline constexpr Mat3f sRGB_TO_XYZ{
    Vec3f{0.4124564f,0.2126729f,0.0193339f},
    Vec3f{0.3575761f,0.7151522f,0.1191920f},
    Vec3f{0.1804375f,0.0721750f,0.9503041f}
};
// XYZ → sRGB (D65)
inline constexpr Mat3f XYZ_TO_sRGB{
    Vec3f{ 3.2404542f,-0.9692660f, 0.0556434f},
    Vec3f{-1.5371385f, 1.8760108f,-0.2040259f},
    Vec3f{-0.4985314f, 0.0415560f, 1.0572252f}
};
// ACEScg → XYZ
inline constexpr Mat3f ACES_TO_XYZ{
    Vec3f{0.6624541811f,0.2722287168f,-0.0055746495f},
    Vec3f{0.1340042065f,0.6740817658f, 0.0040607335f},
    Vec3f{0.1561876744f,0.0536895174f, 1.0103391685f}
};
// Rec.2020 → XYZ
inline constexpr Mat3f REC2020_TO_XYZ{
    Vec3f{0.6369580483f,0.2627002120f,0.0000000000f},
    Vec3f{0.1446169065f,0.6779980715f,0.0280726930f},
    Vec3f{0.1688809752f,0.0593017165f,1.0609850577f}
};
// DCI-P3 → XYZ (D65)
inline constexpr Mat3f DCIP3_TO_XYZ{
    Vec3f{0.4865709486f,0.2289745641f,0.0000000000f},
    Vec3f{0.2656676932f,0.6917385218f,0.0451133819f},
    Vec3f{0.1982172852f,0.0792869141f,1.0439443689f}
};

// Color space enum
enum class ColorSpace { sRGB, LinearSRGB, ACEScg, Rec2020, DCIP3, XYZ };

// Convert linear RGB between color spaces
[[nodiscard]] inline Vec3f convert_color_space(
    Vec3f rgb, ColorSpace from, ColorSpace to) noexcept
{
    if(from==to) return rgb;
    // Convert to XYZ first
    auto to_xyz=[](Vec3f c, ColorSpace cs)->Vec3f{
        switch(cs){
            case ColorSpace::sRGB:        return sRGB_TO_XYZ*to_linear({c.x,c.y,c.z}).to_vec3();
            case ColorSpace::LinearSRGB:  return sRGB_TO_XYZ*c;
            case ColorSpace::ACEScg:      return ACES_TO_XYZ*c;
            case ColorSpace::Rec2020:     return REC2020_TO_XYZ*c;
            case ColorSpace::DCIP3:       return DCIP3_TO_XYZ*c;
            case ColorSpace::XYZ:         return c;
            default:                      return c;
        }
    };
    auto from_xyz=[](Vec3f xyz, ColorSpace cs)->Vec3f{
        switch(cs){
            case ColorSpace::sRGB:        return to_srgb(ColorRGB{(XYZ_TO_sRGB*xyz).x,(XYZ_TO_sRGB*xyz).y,(XYZ_TO_sRGB*xyz).z}).to_vec3();
            case ColorSpace::LinearSRGB:  return XYZ_TO_sRGB*xyz;
            case ColorSpace::ACEScg:      return ACES_TO_XYZ.inversed()*xyz;
            case ColorSpace::Rec2020:     return REC2020_TO_XYZ.inversed()*xyz;
            case ColorSpace::DCIP3:       return DCIP3_TO_XYZ.inversed()*xyz;
            case ColorSpace::XYZ:         return xyz;
            default:                      return xyz;
        }
    };
    return from_xyz(to_xyz(rgb,from),to);
}

// ---------------------------------------------------------------------------
// Chromatic adaptation (Bradford method)
// ---------------------------------------------------------------------------
// Adaptation from D65 to D50 (common in ICC workflow)
inline constexpr Mat3f BRADFORD_D65_TO_D50{
    Vec3f{ 1.0478112f,-0.0295424f, 0.0092345f},
    Vec3f{ 0.0228866f, 0.9904844f,-0.0150436f},
    Vec3f{-0.0501270f, 0.0170491f, 0.7521316f}
};

[[nodiscard]] inline Vec3f chromatic_adapt_bradford(Vec3f xyz_d65) noexcept {
    return BRADFORD_D65_TO_D50*xyz_d65;
}

// ---------------------------------------------------------------------------
// HDR Tone Mapping
// ---------------------------------------------------------------------------
// Reinhard tone mapping
[[nodiscard]] constexpr Vec3f tonemap_reinhard(Vec3f hdr, f32 white=1.f) noexcept {
    Vec3f scaled=hdr*(1.f/(white*white));
    return {hdr.x/(1.f+scaled.x), hdr.y/(1.f+scaled.y), hdr.z/(1.f+scaled.z)};
}

// ACES filmic (Hill/Narkowicz approximation)
[[nodiscard]] constexpr Vec3f tonemap_aces_approx(Vec3f x) noexcept {
    const f32 a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
    x=x*0.6f;
    return {saturate((x.x*(a*x.x+b))/(x.x*(c*x.x+d)+e)),
            saturate((x.y*(a*x.y+b))/(x.y*(c*x.y+d)+e)),
            saturate((x.z*(a*x.z+b))/(x.z*(c*x.z+d)+e))};
}

// Uncharted 2 "Filmic" tone map
[[nodiscard]] constexpr f32 uc2_partial(f32 x) noexcept {
    const f32 A=0.15f,B=0.50f,C=0.10f,D=0.20f,E=0.02f,F=0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
[[nodiscard]] constexpr Vec3f tonemap_uncharted2(Vec3f hdr, f32 exposure=2.f) noexcept {
    f32 W=11.2f; hdr=hdr*exposure;
    f32 scale=1.f/uc2_partial(W);
    return {uc2_partial(hdr.x)*scale, uc2_partial(hdr.y)*scale, uc2_partial(hdr.z)*scale};
}

// AgX tone map (simplified)
[[nodiscard]] inline Vec3f tonemap_agx(Vec3f hdr) noexcept {
    // Log space compression
    static const Mat3f agx_mat{
        Vec3f{0.842479062253094f,0.0784335999999992f,0.0792237451477643f},
        Vec3f{0.0423282422610123f,0.878468636469772f,0.0791661274605434f},
        Vec3f{0.0423756549057051f,0.0784336f,0.879142973793104f}
    };
    Vec3f v=agx_mat*hdr;
    v={max(EPS_F32,v.x),max(EPS_F32,v.y),max(EPS_F32,v.z)};
    v={log2(v.x),log2(v.y),log2(v.z)};
    // Sigmoid
    auto sig=[](f32 x)->f32{
        x=clamp(x/14.f+0.18f,-1.f,1.f);
        return 0.5f*(1.f+x*(3.f-x*x));
    };
    return {sig(v.x),sig(v.y),sig(v.z)};
}

// ---------------------------------------------------------------------------
// Exposure
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 ev_to_exposure(f32 ev) noexcept { return exp2(-ev); }
[[nodiscard]] constexpr f32 luminance_to_ev(f32 L, f32 K=12.5f) noexcept { return log2(L*100.f/K); }
[[nodiscard]] constexpr f32 aperture_shutter_iso_to_ev(f32 aperture,f32 shutter,f32 iso) noexcept {
    return log2((aperture*aperture*100.f)/(shutter*iso));
}

// Auto-exposure target luminance
[[nodiscard]] constexpr f32 auto_exposure(f32 avg_luminance, f32 target_ev=0.f) noexcept {
    return exp2(target_ev)/(avg_luminance+EPS_F32);
}

// ---------------------------------------------------------------------------
// CIE color matching functions (1931 2° observer, tabulated at 10nm, 380-780nm)
// Returns XYZ tristimulus for a wavelength in nm
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f cie_cmf(f32 wavelength_nm) noexcept {
    // Analytical approximation (Wyman et al. 2013)
    auto g=[](f32 x,f32 mu,f32 s1,f32 s2)->f32{
        f32 t=(x-mu)/(x<mu?s1:s2);
        return exp(-0.5f*t*t);
    };
    f32 x=1.056f*g(wavelength_nm,599.8f,37.9f,31.0f)
        +0.362f*g(wavelength_nm,442.0f,16.0f,26.7f)
        -0.065f*g(wavelength_nm,501.1f,20.4f,26.2f);
    f32 y=0.821f*g(wavelength_nm,568.8f,46.9f,40.5f)
        +0.286f*g(wavelength_nm,530.9f,16.3f,31.1f);
    f32 z=1.217f*g(wavelength_nm,437.0f,11.8f,36.0f)
        +0.681f*g(wavelength_nm,459.0f,26.0f,13.8f);
    return {max(0.f,x),max(0.f,y),max(0.f,z)};
}

// Spectrum to XYZ (given spectral power distribution sampled at regular intervals)
[[nodiscard]] inline Vec3f spectrum_to_xyz(
    const f32* spd, i32 n_samples, f32 lambda_min, f32 lambda_max) noexcept
{
    Vec3f xyz{0.f};
    f32 dl=(lambda_max-lambda_min)/(n_samples-1);
    for(i32 i=0;i<n_samples;++i){
        f32 lam=lambda_min+i*dl;
        xyz=xyz+cie_cmf(lam)*spd[i]*dl;
    }
    return xyz;
}

// Blackbody to sRGB
[[nodiscard]] inline Vec3f blackbody_to_xyz(f32 kelvin) noexcept {
    // Integrate Planckian locus against CIE CMFs
    constexpr f32 h=6.626e-34f, c=3e8f, k=1.38e-23f;
    Vec3f xyz{0.f};
    for(i32 i=0;i<40;++i){
        f32 lam=(380.f+i*10.f)*1e-9f;
        f32 B=1.f/(exp(h*c/(lam*k*kelvin))-1.f);
        xyz=xyz+cie_cmf(lam*1e9f)*B*10.f;
    }
    return xyz;
}

// ---------------------------------------------------------------------------
// Munsell-like hue rotation
// ---------------------------------------------------------------------------
[[nodiscard]] inline ColorOKLCH hue_shift(ColorOKLCH c, f32 degrees) noexcept {
    return {c.L, c.C, c.h+degrees*DEG_TO_RAD};
}
[[nodiscard]] inline ColorOKLCH saturate_color(ColorOKLCH c, f32 amount) noexcept {
    return {c.L, c.C*amount, c.h};
}
[[nodiscard]] inline ColorOKLCH lighten_color(ColorOKLCH c, f32 amount) noexcept {
    return {clamp(c.L+amount,0.f,1.f), c.C, c.h};
}

// Generate analogous palette
inline void analogous_palette(ColorOKLCH base, f32 angle_step,
                               ColorOKLCH* out, i32 n) noexcept {
    for(i32 i=0;i<n;++i)
        out[i]=hue_shift(base,float(i-n/2)*angle_step);
}

// Complementary
[[nodiscard]] inline ColorOKLCH complementary(ColorOKLCH c) noexcept {
    return hue_shift(c,180.f);
}

// Split-complementary
inline void split_complementary(ColorOKLCH c, f32 split,
                                 ColorOKLCH& a, ColorOKLCH& b) noexcept {
    a=hue_shift(c,180.f-split); b=hue_shift(c,180.f+split);
}

// Triadic
inline void triadic_palette(ColorOKLCH c, ColorOKLCH& a, ColorOKLCH& b) noexcept {
    a=hue_shift(c,120.f); b=hue_shift(c,240.f);
}

// ---------------------------------------------------------------------------
// Color grading — LUT apply (3D LUT, n×n×n)
// ---------------------------------------------------------------------------
[[nodiscard]] inline ColorRGB apply_lut_3d(
    ColorRGB color, const ColorRGB* lut, i32 size) noexcept
{
    f32 s=f32(size-1);
    f32 rx=saturate(color.r)*s, ry=saturate(color.g)*s, rz=saturate(color.b)*s;
    i32 ix=(i32)rx, iy=(i32)ry, iz=(i32)rz;
    ix=min(ix,size-2); iy=min(iy,size-2); iz=min(iz,size-2);
    f32 fx=rx-ix, fy=ry-iy, fz=rz-iz;
    auto idx=[&](i32 x,i32 y,i32 z)->ColorRGB{
        return lut[z*size*size+y*size+x];
    };
    // Trilinear interpolation
    ColorRGB c000=idx(ix,iy,iz),   c100=idx(ix+1,iy,iz);
    ColorRGB c010=idx(ix,iy+1,iz), c110=idx(ix+1,iy+1,iz);
    ColorRGB c001=idx(ix,iy,iz+1), c101=idx(ix+1,iy,iz+1);
    ColorRGB c011=idx(ix,iy+1,iz+1),c111=idx(ix+1,iy+1,iz+1);
    auto bl=[](ColorRGB a,ColorRGB b,f32 t)->ColorRGB{return lerp(a,b,t);};
    return bl(bl(bl(c000,c100,fx),bl(c010,c110,fx),fy),
              bl(bl(c001,c101,fx),bl(c011,c111,fx),fy),fz);
}

// ---------------------------------------------------------------------------
// Spectral upsampling (Smits method — RGB to spectral)
// ---------------------------------------------------------------------------
// Returns coefficient for 3-lobed spectral model at given wavelength
[[nodiscard]] inline f32 smits_spectral(Vec3f rgb, f32 wavelength_nm) noexcept {
    // Very simplified — real Smits uses lookup tables
    f32 r=rgb.x,g=rgb.y,b=rgb.z;
    f32 w=wavelength_nm;
    // Red lobe: 600-700nm, Green: 500-600nm, Blue: 400-500nm
    f32 red_lobe  =exp(-0.5f*((w-650.f)/40.f)*((w-650.f)/40.f));
    f32 green_lobe=exp(-0.5f*((w-550.f)/40.f)*((w-550.f)/40.f));
    f32 blue_lobe =exp(-0.5f*((w-450.f)/40.f)*((w-450.f)/40.f));
    return r*red_lobe+g*green_lobe+b*blue_lobe;
}

} // namespace qs::math
