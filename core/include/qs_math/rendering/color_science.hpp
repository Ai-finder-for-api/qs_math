#pragma once
// qs_math/rendering/color_science.hpp — Color science: color spaces, chromatic
//   adaptation, gamut mapping, tone mapping, color difference (CIEDE2000),
//   color temperature, metamerism, color blindness simulation (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/mat3.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <cmath>
#include <array>

namespace qs::math::render {

// ─── CIE XYZ ↔ sRGB (D65, already in spectral_render.hpp — extended here) ─
// Gamma encode / decode
[[nodiscard]] inline f32 srgb_gamma_encode(f32 c) noexcept {
    return c<=0.0031308f ? 12.92f*c : 1.055f*std::pow(c,1.f/2.4f)-0.055f;
}
[[nodiscard]] inline f32 srgb_gamma_decode(f32 c) noexcept {
    return c<=0.04045f ? c/12.92f : std::pow((c+0.055f)/1.055f,2.4f);
}
[[nodiscard]] inline Vec3f srgb_encode(Vec3f lin) noexcept {
    return {srgb_gamma_encode(lin.x),srgb_gamma_encode(lin.y),srgb_gamma_encode(lin.z)};
}
[[nodiscard]] inline Vec3f srgb_decode(Vec3f enc) noexcept {
    return {srgb_gamma_decode(enc.x),srgb_gamma_decode(enc.y),srgb_gamma_decode(enc.z)};
}

// ─── CIE Lab (CIELAB 1976) ───────────────────────────────────────────────
// Reference white D65: Xn=0.95047, Yn=1.0, Zn=1.08883
inline constexpr Vec3f D65_WHITE = {0.95047f,1.0f,1.08883f};
inline constexpr Vec3f D50_WHITE = {0.96422f,1.0f,0.82521f};

[[nodiscard]] inline f32 lab_f(f32 t) noexcept {
    return t>0.008856f?std::cbrt(t):(7.787f*t+16.f/116.f);
}
[[nodiscard]] inline f32 lab_f_inv(f32 t) noexcept {
    return t>0.20689655f?(t*t*t):(t-16.f/116.f)/7.787f;
}

[[nodiscard]] inline Vec3f xyz_to_lab(Vec3f xyz, Vec3f white=D65_WHITE) noexcept {
    f32 fx=lab_f(xyz.x/white.x), fy=lab_f(xyz.y/white.y), fz=lab_f(xyz.z/white.z);
    return {116.f*fy-16.f, 500.f*(fx-fy), 200.f*(fy-fz)};
}

[[nodiscard]] inline Vec3f lab_to_xyz(Vec3f lab, Vec3f white=D65_WHITE) noexcept {
    f32 fy=(lab.x+16.f)/116.f;
    f32 fx=lab.y/500.f+fy;
    f32 fz=fy-lab.z/200.f;
    return {lab_f_inv(fx)*white.x, lab_f_inv(fy)*white.y, lab_f_inv(fz)*white.z};
}

// ─── CIE LCh (polar Lab) ─────────────────────────────────────────────────
[[nodiscard]] inline Vec3f lab_to_lch(Vec3f lab) noexcept {
    return {lab.x, std::sqrt(lab.y*lab.y+lab.z*lab.z),
            std::atan2(lab.z,lab.y)*180.f/f32(PI)};
}
[[nodiscard]] inline Vec3f lch_to_lab(Vec3f lch) noexcept {
    f32 h=lch.z*f32(PI)/180.f;
    return {lch.x, lch.y*std::cos(h), lch.y*std::sin(h)};
}

// ─── CIEDE2000 color difference ───────────────────────────────────────────
[[nodiscard]] inline f32 ciede2000(Vec3f lab1, Vec3f lab2) noexcept {
    // Unpack
    f32 L1=lab1.x,a1=lab1.y,b1=lab1.z;
    f32 L2=lab2.x,a2=lab2.y,b2=lab2.z;
    // Step 1: C*ab, h*ab
    f32 C1=std::sqrt(a1*a1+b1*b1), C2=std::sqrt(a2*a2+b2*b2);
    f32 Cab=(C1+C2)/2.f;
    f32 Cab7=std::pow(Cab,7.f);
    f32 G=0.5f*(1.f-std::sqrt(Cab7/(Cab7+6103515625.f))); // 25^7
    f32 a1p=a1*(1.f+G), a2p=a2*(1.f+G);
    f32 C1p=std::sqrt(a1p*a1p+b1*b1), C2p=std::sqrt(a2p*a2p+b2*b2);
    f32 h1p=std::atan2(b1,a1p)*180.f/f32(PI); if(h1p<0) h1p+=360.f;
    f32 h2p=std::atan2(b2,a2p)*180.f/f32(PI); if(h2p<0) h2p+=360.f;
    // Step 2: deltas
    f32 dLp=L2-L1;
    f32 dCp=C2p-C1p;
    f32 dhp=0.f;
    if(C1p*C2p>0){
        dhp=h2p-h1p;
        if(dhp>180) dhp-=360; else if(dhp<-180) dhp+=360;
    }
    f32 dHp=2.f*std::sqrt(C1p*C2p)*std::sin(dhp*f32(PI)/360.f);
    // Step 3: CIEDE2000 weighting
    f32 Lbar=(L1+L2)/2.f;
    f32 Cbar=(C1p+C2p)/2.f;
    f32 hbar=h1p+h2p;
    if(C1p*C2p>0){
        if(std::abs(h1p-h2p)<=180) hbar/=2.f;
        else hbar=(hbar+360.f)/2.f;
    }
    f32 T=1.f-0.17f*std::cos((hbar-30.f)*f32(PI)/180.f)
            +0.24f*std::cos(2.f*hbar*f32(PI)/180.f)
            +0.32f*std::cos((3.f*hbar+6.f)*f32(PI)/180.f)
            -0.20f*std::cos((4.f*hbar-63.f)*f32(PI)/180.f);
    f32 SL=1.f+0.015f*(Lbar-50.f)*(Lbar-50.f)/std::sqrt(20.f+(Lbar-50.f)*(Lbar-50.f));
    f32 SC=1.f+0.045f*Cbar;
    f32 SH=1.f+0.015f*Cbar*T;
    f32 Cbar7=std::pow(Cbar,7.f);
    f32 RC=2.f*std::sqrt(Cbar7/(Cbar7+6103515625.f));
    f32 dTheta=(30.f*std::exp(-((hbar-275.f)/25.f)*((hbar-275.f)/25.f)));
    f32 RT=-RC*std::sin(2.f*dTheta*f32(PI)/180.f);
    f32 dE=std::sqrt((dLp/SL)*(dLp/SL)+(dCp/SC)*(dCp/SC)+(dHp/SH)*(dHp/SH)
                    +RT*(dCp/SC)*(dHp/SH));
    return dE;
}

// ─── HSV / HSL ────────────────────────────────────────────────────────────
[[nodiscard]] inline Vec3f rgb_to_hsv(Vec3f rgb) noexcept {
    f32 r=rgb.x,g=rgb.y,b=rgb.z;
    f32 mx=max({r,g,b}), mn=min({r,g,b}), d=mx-mn;
    f32 h=0,s=(mx>1e-6f)?d/mx:0,v=mx;
    if(d>1e-6f){
        if(mx==r) h=std::fmod((g-b)/d,6.f);
        else if(mx==g) h=(b-r)/d+2.f;
        else h=(r-g)/d+4.f;
        h*=60.f; if(h<0) h+=360.f;
    }
    return {h,s,v};
}
[[nodiscard]] inline Vec3f hsv_to_rgb(Vec3f hsv) noexcept {
    f32 h=hsv.x,s=hsv.y,v=hsv.z;
    f32 c=v*s, x=c*(1.f-std::abs(std::fmod(h/60.f,2.f)-1.f)), m=v-c;
    Vec3f rgb{};
    if(h<60) rgb={c,x,0}; else if(h<120) rgb={x,c,0};
    else if(h<180) rgb={0,c,x}; else if(h<240) rgb={0,x,c};
    else if(h<300) rgb={x,0,c}; else rgb={c,0,x};
    return {rgb.x+m,rgb.y+m,rgb.z+m};
}

[[nodiscard]] inline Vec3f rgb_to_hsl(Vec3f rgb) noexcept {
    f32 r=rgb.x,g=rgb.y,b=rgb.z;
    f32 mx=max({r,g,b}),mn=min({r,g,b});
    f32 l=(mx+mn)/2.f,s=0,h=0,d=mx-mn;
    if(d>1e-6f){
        s=d/(l<0.5f?mx+mn:2.f-mx-mn);
        if(mx==r) h=std::fmod((g-b)/d,6.f);
        else if(mx==g) h=(b-r)/d+2.f;
        else h=(r-g)/d+4.f;
        h*=60.f; if(h<0) h+=360.f;
    }
    return {h,s,l};
}
[[nodiscard]] inline Vec3f hsl_to_rgb(Vec3f hsl) noexcept {
    f32 h=hsl.x,s=hsl.y,l=hsl.z;
    f32 c=(1.f-std::abs(2.f*l-1.f))*s;
    f32 x=c*(1.f-std::abs(std::fmod(h/60.f,2.f)-1.f));
    f32 m=l-c/2.f;
    Vec3f rgb{};
    if(h<60) rgb={c,x,0}; else if(h<120) rgb={x,c,0};
    else if(h<180) rgb={0,c,x}; else if(h<240) rgb={0,x,c};
    else if(h<300) rgb={x,0,c}; else rgb={c,0,x};
    return {rgb.x+m,rgb.y+m,rgb.z+m};
}

// ─── Oklab (perceptual, 2020) ─────────────────────────────────────────────
[[nodiscard]] inline Vec3f linear_srgb_to_oklab(Vec3f c) noexcept {
    f32 l=0.4122214708f*c.x+0.5363325363f*c.y+0.0514459929f*c.z;
    f32 m=0.2119034982f*c.x+0.6806995451f*c.y+0.1073969566f*c.z;
    f32 s=0.0883024619f*c.x+0.2817188376f*c.y+0.6299787005f*c.z;
    f32 l_=std::cbrt(l),m_=std::cbrt(m),s_=std::cbrt(s);
    return {0.2104542553f*l_+0.7936177850f*m_-0.0040720468f*s_,
            1.9779984951f*l_-2.4285922050f*m_+0.4505937099f*s_,
            0.0259040371f*l_+0.7827717662f*m_-0.8086757660f*s_};
}
[[nodiscard]] inline Vec3f oklab_to_linear_srgb(Vec3f c) noexcept {
    f32 l_=c.x+0.3963377774f*c.y+0.2158037573f*c.z;
    f32 m_=c.x-0.1055613458f*c.y-0.0638541728f*c.z;
    f32 s_=c.x-0.0894841775f*c.y-1.2914855480f*c.z;
    f32 l=l_*l_*l_,m=m_*m_*m_,s=s_*s_*s_;
    return {+4.0767416621f*l-3.3077115913f*m+0.2309699292f*s,
            -1.2684380046f*l+2.6097574011f*m-0.3413193965f*s,
            -0.0041960863f*l-0.7034186147f*m+1.7076147010f*s};
}

// ─── Chromatic adaptation (Bradford transform) ────────────────────────────
// Adapt XYZ color from source white to destination white
[[nodiscard]] inline Vec3f bradford_adapt(Vec3f xyz, Vec3f src_white, Vec3f dst_white) noexcept {
    // Bradford matrix
    static const f32 M[9]={
         0.8951f, 0.2664f,-0.1614f,
        -0.7502f, 1.7135f, 0.0367f,
         0.0389f,-0.0685f, 1.0296f
    };
    static const f32 Mi[9]={
         0.9869929f,-0.1470543f, 0.1599627f,
         0.4323053f, 0.5183603f, 0.0492912f,
        -0.0085287f, 0.0400428f, 0.9684867f
    };
    // Apply M to src and dst whites
    auto mvmul=[](const f32 m[9],Vec3f v)->Vec3f{
        return {m[0]*v.x+m[1]*v.y+m[2]*v.z,
                m[3]*v.x+m[4]*v.y+m[5]*v.z,
                m[6]*v.x+m[7]*v.y+m[8]*v.z};
    };
    Vec3f ρ_s=mvmul(M,src_white), ρ_d=mvmul(M,dst_white);
    // Scale
    Vec3f ρ_xyz=mvmul(M,xyz);
    Vec3f ρ_scaled={ρ_xyz.x*(ρ_d.x/ρ_s.x),ρ_xyz.y*(ρ_d.y/ρ_s.y),ρ_xyz.z*(ρ_d.z/ρ_s.z)};
    return mvmul(Mi,ρ_scaled);
}

// ─── Tone mapping operators ───────────────────────────────────────────────
// Reinhard (luminance)
[[nodiscard]] inline Vec3f reinhard(Vec3f hdr) noexcept {
    f32 lum=dot(hdr,Vec3f{0.2126f,0.7152f,0.0722f});
    f32 scale=lum/(1.f+lum)/max(lum,1e-6f);
    return hdr*scale;
}

// Reinhard extended (with Lwhite)
[[nodiscard]] inline Vec3f reinhard_ext(Vec3f hdr, f32 Lwhite=1.f) noexcept {
    f32 lum=dot(hdr,Vec3f{0.2126f,0.7152f,0.0722f});
    f32 scale=lum*(1.f+lum/(Lwhite*Lwhite))/(1.f+lum)/max(lum,1e-6f);
    return hdr*scale;
}

// ACES filmic (Hill approximation)
[[nodiscard]] inline Vec3f aces_filmic(Vec3f x) noexcept {
    const f32 a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
    Vec3f r={clamp01((x.x*(a*x.x+b))/(x.x*(c*x.x+d)+e)),
             clamp01((x.y*(a*x.y+b))/(x.y*(c*x.y+d)+e)),
             clamp01((x.z*(a*x.z+b))/(x.z*(c*x.z+d)+e))};
    return r;
}

// Uncharted 2 / Hable
[[nodiscard]] inline Vec3f hable_op(Vec3f x) noexcept {
    const f32 A=0.15f,B=0.50f,C=0.10f,D=0.20f,E=0.02f,F=0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
[[nodiscard]] inline Vec3f hable_tonemap(Vec3f hdr, f32 exposure=2.f) noexcept {
    Vec3f curr=hable_op(hdr*exposure);
    Vec3f white_scale=Vec3f{1,1,1}*(1.f/hable_op(Vec3f{11.2f,11.2f,11.2f}).x);
    return curr*white_scale;
}

// AgX (approximate, 2023)
[[nodiscard]] inline Vec3f agx_tonemap(Vec3f hdr) noexcept {
    // AgX log compression
    hdr = Vec3f{max(hdr.x,1e-6f),max(hdr.y,1e-6f),max(hdr.z,1e-6f)};
    Vec3f lhdr={std::log2(hdr.x),std::log2(hdr.y),std::log2(hdr.z)};
    // Clamp log exposure
    lhdr={clamp(lhdr.x,-12.47393f,4.026069f),clamp(lhdr.y,-12.47393f,4.026069f),clamp(lhdr.z,-12.47393f,4.026069f)};
    // Normalize to [0,1]
    Vec3f x=(lhdr+Vec3f{12.47393f,12.47393f,12.47393f})/(12.47393f+4.026069f);
    // Sigmoid (polynomial approx)
    auto sig=[](f32 t)->f32{
        t=clamp(t,0.f,1.f);
        return t*t*(3.f-2.f*t);
    };
    return {sig(x.x),sig(x.y),sig(x.z)};
}

// ─── Gamut mapping ────────────────────────────────────────────────────────
// Clip to sRGB gamut (hard clip in linear)
[[nodiscard]] inline Vec3f gamut_clip(Vec3f c) noexcept {
    return {clamp01(c.x),clamp01(c.y),clamp01(c.z)};
}

// Chroma compression (preserve luminance, reduce chroma)
[[nodiscard]] inline Vec3f gamut_compress_chroma(Vec3f c, f32 threshold=0.8f) noexcept {
    f32 lum=dot(c,Vec3f{0.2126f,0.7152f,0.0722f});
    Vec3f chroma=c-Vec3f{lum,lum,lum};
    f32 max_ch=max(std::abs(chroma.x),max(std::abs(chroma.y),std::abs(chroma.z)));
    if(max_ch>threshold){
        f32 scale=threshold+(max_ch-threshold)/(1.f+(max_ch-threshold));
        chroma=chroma*(scale/max_ch);
    }
    return Vec3f{lum,lum,lum}+chroma;
}

// ─── Color blindness simulation (Machado 2009) ────────────────────────────
enum class ColorBlindType { Protanopia, Deuteranopia, Tritanopia };

[[nodiscard]] inline Vec3f simulate_color_blindness(Vec3f srgb_lin, ColorBlindType t) noexcept {
    // Simplified Machado matrices (severity=1.0)
    static const f32 PROT[9]={0.152286f,1.052583f,-0.204868f,0.114503f,0.786281f,0.099216f,-0.003882f,-0.048116f,1.051998f};
    static const f32 DEUT[9]={0.367322f,0.860646f,-0.227968f,0.280085f,0.672501f,0.047413f,-0.011820f,0.042940f,0.968881f};
    static const f32 TRIT[9]={1.255528f,-0.076749f,-0.178779f,-0.078411f,0.930809f,0.147602f,0.004733f,0.691367f,0.303900f};
    const f32* M = (t==ColorBlindType::Protanopia)?PROT:(t==ColorBlindType::Deuteranopia)?DEUT:TRIT;
    return {M[0]*srgb_lin.x+M[1]*srgb_lin.y+M[2]*srgb_lin.z,
            M[3]*srgb_lin.x+M[4]*srgb_lin.y+M[5]*srgb_lin.z,
            M[6]*srgb_lin.x+M[7]*srgb_lin.y+M[8]*srgb_lin.z};
}

// ─── CIECAM02 (simplified appearance model) ───────────────────────────────
struct CAM02Params {
    f32 La=64.f;    // adapting luminance
    f32 Yb=20.f;    // background relative luminance
    f32 Yw=100.f;   // white luminance
};

[[nodiscard]] inline Vec3f xyz_to_cam02_jch(Vec3f xyz, const CAM02Params& p=CAM02Params{}) noexcept {
    // Simplified: compute J (lightness), C (chroma), h (hue)
    f32 k=1.f/(5.f*p.La+1.f), k4=k*k*k*k;
    f32 FL=0.2f*k4*(5.f*p.La)+0.1f*(1.f-k4)*(1.f-k4)*std::cbrt(5.f*p.La);
    f32 n=p.Yb/p.Yw, Nbb=0.725f/std::pow(n,0.2f);
    f32 z=1.48f+std::sqrt(50.f*n);
    // Hunt-Pointer-Estevez adapted coords
    Vec3f lms={0.7328f*xyz.x+0.4296f*xyz.y-0.1624f*xyz.z,
               -0.7036f*xyz.x+1.6975f*xyz.y+0.0061f*xyz.z,
                0.0030f*xyz.x+0.0136f*xyz.y+0.9834f*xyz.z};
    auto hpe_adapt=[&](f32 c)->f32{
        f32 fc=(FL*c/100.f); if(fc<0) fc=-std::pow(-fc,0.42f); else fc=std::pow(fc,0.42f);
        return 400.f*fc/(fc+27.13f)+0.1f;
    };
    f32 La_=hpe_adapt(lms.x), Ma_=hpe_adapt(lms.y), Sa_=hpe_adapt(lms.z);
    f32 A=(2.f*La_+Ma_+Sa_/20.f-0.305f)*Nbb;
    f32 J=100.f*std::pow(A/100.f,n*z);
    f32 a=La_-12.f*Ma_/11.f+Sa_/11.f;
    f32 b=(La_+Ma_-2.f*Sa_)/9.f;
    f32 h=std::atan2(b,a)*180.f/f32(PI); if(h<0) h+=360.f;
    f32 et=0.25f*(std::cos(h*f32(PI)/180.f+2.f)+3.8f);
    f32 t=(50000.f/13.f*Nbb*et*std::sqrt(a*a+b*b))/(La_+Ma_+(21.f/20.f)*Sa_);
    f32 C=std::pow(t,0.9f)*std::sqrt(J/100.f)*std::pow(1.64f-std::pow(0.29f,n),0.73f);
    return {J,C,h};
}

// ─── Perceptual color mixing (Oklab) ─────────────────────────────────────
[[nodiscard]] inline Vec3f mix_colors_perceptual(Vec3f a_lin, Vec3f b_lin, f32 t) noexcept {
    Vec3f la=linear_srgb_to_oklab(a_lin);
    Vec3f lb=linear_srgb_to_oklab(b_lin);
    return oklab_to_linear_srgb(la*(1.f-t)+lb*t);
}

// ─── Munsell / Pantone utility (lookup table helpers) ─────────────────────
// Approximate Munsell value from CIE Y (0-100)
[[nodiscard]] inline f32 munsell_value(f32 Y) noexcept {
    // Priest-Gibson-MacNicholas polynomial (1920)
    f32 y=Y/100.f;
    return 2.4266f*std::pow(y,1.f/3.f)-0.9885f;
}

} // namespace qs::math::render
