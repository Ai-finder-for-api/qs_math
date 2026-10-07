#pragma once
// qs_math/rendering/color_math.hpp — Color spaces, chromatic adaptation, spectral upsampling
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

struct RGB { f32 r,g,b;
    [[nodiscard]] constexpr RGB operator+(RGB o) const noexcept{return{r+o.r,g+o.g,b+o.b};}
    [[nodiscard]] constexpr RGB operator*(f32 s) const noexcept{return{r*s,g*s,b*s};}
    [[nodiscard]] constexpr RGB operator*(RGB o) const noexcept{return{r*o.r,g*o.g,b*o.b};}
    [[nodiscard]] constexpr f32 luminance() const noexcept{return 0.2126f*r+0.7152f*g+0.0722f*b;}
};
struct XYZ { f32 X,Y,Z; };
struct Lab { f32 L,a,b; };
struct HSV { f32 h,s,v; };
struct HSL { f32 h,s,l; };
struct Oklab { f32 L,a,b; };

// ─── sRGB ↔ linear ────────────────────────────────────────────────────────────
[[nodiscard]] constexpr f32 srgb_to_linear(f32 c) noexcept {
    return c<=0.04045f?c/12.92f:std::pow((c+0.055f)/1.055f,2.4f);
}
[[nodiscard]] constexpr f32 linear_to_srgb(f32 c) noexcept {
    return c<=0.0031308f?12.92f*c:1.055f*std::pow(c,1.0f/2.4f)-0.055f;
}
[[nodiscard]] constexpr RGB srgb_to_linear_rgb(RGB c) noexcept {
    return{srgb_to_linear(c.r),srgb_to_linear(c.g),srgb_to_linear(c.b)};
}
[[nodiscard]] constexpr RGB linear_to_srgb_rgb(RGB c) noexcept {
    return{linear_to_srgb(c.r),linear_to_srgb(c.g),linear_to_srgb(c.b)};
}

// ─── RGB ↔ XYZ (D65, sRGB primaries) ─────────────────────────────────────────
[[nodiscard]] constexpr XYZ rgb_to_xyz(RGB c) noexcept {
    return {
        0.4124564f*c.r+0.3575761f*c.g+0.1804375f*c.b,
        0.2126729f*c.r+0.7151522f*c.g+0.0721750f*c.b,
        0.0193339f*c.r+0.1191920f*c.g+0.9503041f*c.b
    };
}
[[nodiscard]] constexpr RGB xyz_to_rgb(XYZ c) noexcept {
    return {
        3.2404542f*c.X-1.5371385f*c.Y-0.4985314f*c.Z,
       -0.9692660f*c.X+1.8760108f*c.Y+0.0415560f*c.Z,
        0.0556434f*c.X-0.2040259f*c.Y+1.0572252f*c.Z
    };
}

// ─── XYZ ↔ CIE L*a*b* ────────────────────────────────────────────────────────
inline constexpr f32 k_Xn=0.95047f,k_Yn=1.0f,k_Zn=1.08883f; // D65
[[nodiscard]] constexpr f32 lab_f(f32 t) noexcept {
    return t>0.008856f?std::cbrt(t):(7.787f*t+16.0f/116.0f);
}
[[nodiscard]] constexpr Lab xyz_to_lab(XYZ c) noexcept {
    f32 fx=lab_f(c.X/k_Xn),fy=lab_f(c.Y/k_Yn),fz=lab_f(c.Z/k_Zn);
    return {116*fy-16, 500*(fx-fy), 200*(fy-fz)};
}
[[nodiscard]] constexpr XYZ lab_to_xyz(Lab c) noexcept {
    f32 fy=(c.L+16)/116,fx=c.a/500+fy,fz=fy-c.b/200;
    auto inv=[](f32 t){return t>0.206897f?t*t*t:(t-16.0f/116.0f)/7.787f;};
    return {k_Xn*inv(fx),k_Yn*inv(fy),k_Zn*inv(fz)};
}
// ΔE76 color difference
[[nodiscard]] constexpr f32 delta_e76(Lab a, Lab b) noexcept {
    f32 dL=a.L-b.L,da=a.a-b.a,db=a.b-b.b;
    return std::sqrt(dL*dL+da*da+db*db);
}
// ΔE2000 (simplified, skipping RT term)
[[nodiscard]] constexpr f32 delta_e2000(Lab a, Lab b) noexcept {
    f32 L_bar=(a.L+b.L)*0.5f;
    f32 C1=std::sqrt(a.a*a.a+a.b*a.b),C2=std::sqrt(b.a*b.a+b.b*b.b);
    f32 C_bar=(C1+C2)*0.5f, C7=std::pow(C_bar,7.f);
    f32 G=0.5f*(1-std::sqrt(C7/(C7+6103515625.f))); // 25^7
    f32 a1p=a.a*(1+G), a2p=b.a*(1+G);
    f32 C1p=std::sqrt(a1p*a1p+a.b*a.b), C2p=std::sqrt(a2p*a2p+b.b*b.b);
    f32 dLp=b.L-a.L, dCp=C2p-C1p;
    f32 h1=std::atan2(a.b,a1p), h2=std::atan2(b.b,a2p);
    f32 dhp=(std::abs(h1-h2)<=3.14159f)?h2-h1:h2-h1+(h2<h1?2:0)*3.14159f;
    f32 dHp=2*std::sqrt(C1p*C2p)*std::sin(dhp*0.5f);
    f32 Lbp=(a.L+b.L)*0.5f,Cbp=(C1p+C2p)*0.5f;
    f32 t=1-0.17f*std::cos(h1-0.5236f)+0.24f*std::cos(2*h1)+0.32f*std::cos(3*h1+0.1047f)-0.2f*std::cos(4*h1-1.0996f);
    f32 SL=1+0.015f*(Lbp-50)*(Lbp-50)/std::sqrt(20+(Lbp-50)*(Lbp-50));
    f32 SC=1+0.045f*Cbp, SH=1+0.015f*Cbp*t;
    return std::sqrt((dLp/SL)*(dLp/SL)+(dCp/SC)*(dCp/SC)+(dHp/SH)*(dHp/SH));
}

// ─── Oklab (perceptually uniform, Ottosson 2020) ──────────────────────────────
[[nodiscard]] constexpr Oklab rgb_to_oklab(RGB c) noexcept {
    f32 l=0.4122214708f*c.r+0.5363325363f*c.g+0.0514459929f*c.b;
    f32 m=0.2119034982f*c.r+0.6806995451f*c.g+0.1073969566f*c.b;
    f32 s=0.0883024619f*c.r+0.2817188376f*c.g+0.6299787005f*c.b;
    f32 l_=std::cbrt(l),m_=std::cbrt(m),s_=std::cbrt(s);
    return {0.2104542553f*l_+0.7936177850f*m_-0.0040720468f*s_,
            1.9779984951f*l_-2.4285922050f*m_+0.4505937099f*s_,
            0.0259040371f*l_+0.7827717662f*m_-0.8086757660f*s_};
}
[[nodiscard]] constexpr RGB oklab_to_rgb(Oklab c) noexcept {
    f32 l_=c.L+0.3963377774f*c.a+0.2158037573f*c.b;
    f32 m_=c.L-0.1055613458f*c.a-0.0638541728f*c.b;
    f32 s_=c.L-0.0894841775f*c.a-1.2914855480f*c.b;
    f32 l=l_*l_*l_,m=m_*m_*m_,s=s_*s_*s_;
    return {4.0767416621f*l-3.3077115913f*m+0.2309699292f*s,
           -1.2684380046f*l+2.6097574011f*m-0.3413193965f*s,
           -0.0041960863f*l-0.7034186147f*m+1.7076147010f*s};
}

// ─── HSV/HSL conversions ──────────────────────────────────────────────────────
[[nodiscard]] constexpr HSV rgb_to_hsv(RGB c) noexcept {
    f32 mx=std::max({c.r,c.g,c.b}),mn=std::min({c.r,c.g,c.b}),d=mx-mn;
    f32 h=0,s=mx>1e-5f?d/mx:0,v=mx;
    if(d>1e-5f) {
        if(mx==c.r) h=(c.g-c.b)/d+6;
        else if(mx==c.g) h=(c.b-c.r)/d+2;
        else h=(c.r-c.g)/d+4;
        h=std::fmod(h,6.0f)/6.0f;
    }
    return{h,s,v};
}
[[nodiscard]] constexpr RGB hsv_to_rgb(HSV c) noexcept {
    f32 h6=c.h*6,i=std::floor(h6),f=h6-i;
    f32 p=c.v*(1-c.s),q=c.v*(1-c.s*f),t_=c.v*(1-c.s*(1-f));
    int si=int(i)%6;
    if(si==0) return{c.v,t_,p};if(si==1) return{q,c.v,p};
    if(si==2) return{p,c.v,t_};if(si==3) return{p,q,c.v};
    if(si==4) return{t_,p,c.v};return{c.v,p,q};
}

// ─── Chromatic adaptation (Bradford transform, D65→D50) ───────────────────────
[[nodiscard]] constexpr XYZ adapt_d65_to_d50(XYZ c) noexcept {
    return {
        1.0478112f*c.X+0.0228866f*c.Y-0.0501270f*c.Z,
        0.0295424f*c.X+0.9904844f*c.Y-0.0170491f*c.Z,
       -0.0092345f*c.X+0.0150436f*c.Y+0.7521316f*c.Z
    };
}

// ─── Color temperature (Planckian locus, McCamy's approximation) ──────────────
[[nodiscard]] constexpr f32 xy_to_cct(f32 x, f32 y) noexcept {
    f32 n=(x-0.3320f)/(y-0.1858f);
    return -449*n*n*n+3525*n*n-6823.3f*n+5520.33f;
}
// Correlated color temperature to xy
[[nodiscard]] constexpr std::pair<f32,f32> cct_to_xy(f32 T) noexcept {
    f32 T2=T*T,T3=T2*T;
    f32 x=(T<4000)
        ? 0.27475e9f/T3-0.98598e6f/T2+1.17444e3f/T+0.145986f
        :-3.0258e9f/T3+2.1070e6f/T2+0.2226e3f/T+0.240390f;
    f32 y=(T<2222)
        ?-1.1063814f*x*x*x-1.34811020f*x*x+2.18555832f*x-0.20219683f
        :(T<4000)
        ?-0.9549476f*x*x*x-1.37418593f*x*x+2.09137015f*x-0.16748867f
        : 3.0817580f*x*x*x-5.87338670f*x*x+3.75112997f*x-0.37001483f;
    return{x,y};
}

// ─── Spectral power distribution → XYZ (CIE 1931 CMFs, 380-780nm, 5nm steps) ─
// Simplified: use only a few representative wavelengths
[[nodiscard]] constexpr XYZ spd_to_xyz(const f32* spd, int n, f32 lambda0, f32 dlambda) noexcept {
    XYZ xyz{};
    // CIE 1931 CMF table (sampled at key wavelengths)
    for(int i=0;i<n;++i) {
        f32 lambda=lambda0+i*dlambda;
        // Gaussian approximations to CMFs
        f32 t=(lambda-600)/50.0f;
        f32 x_bar=std::exp(-t*t)*1.056f+std::exp(-((lambda-450)/50.0f)*((lambda-450)/50.0f))*0.362f;
        f32 y_bar=std::exp(-((lambda-555)/50.0f)*((lambda-555)/50.0f));
        f32 z_bar=std::exp(-((lambda-450)/50.0f)*((lambda-450)/50.0f))*1.8f;
        xyz.X+=spd[i]*x_bar*dlambda;
        xyz.Y+=spd[i]*y_bar*dlambda;
        xyz.Z+=spd[i]*z_bar*dlambda;
    }
    return xyz;
}
} // namespace qs::math
