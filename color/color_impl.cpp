// qs_math/color/color_ext_impl.cpp (C++20)
#include "include/qs_math_color.hpp"
#include <cmath>
namespace qs::math::color_ext {
// ─── XYZ ↔ Lab ───────────────────────────────────────────────────────────────
static f32 lab_f(f32 t) noexcept {
    constexpr f32 d=6.0f/29,d3=d*d*d,f1=1.0f/(3*d*d),f2=4.0f/29;
    return t>d3?std::cbrt(t):f1*t+f2;
}
static f32 lab_f_inv(f32 t) noexcept {
    constexpr f32 d=6.0f/29;
    return t>d?t*t*t:3*d*d*(t-4.0f/29);
}
Lab xyz_to_lab(XYZ xyz) noexcept {
    const f32 fx=lab_f(xyz.X/0.95047f),fy=lab_f(xyz.Y),fz=lab_f(xyz.Z/1.08883f);
    return {116*fy-16, 500*(fx-fy), 200*(fy-fz)};
}
XYZ lab_to_xyz(Lab lab) noexcept {
    const f32 fy=(lab.L+16)/116;
    return {0.95047f*lab_f_inv(fy+lab.a/500),lab_f_inv(fy),1.08883f*lab_f_inv(fy-lab.b/200)};
}
// ─── Oklab ───────────────────────────────────────────────────────────────────
Oklab linear_srgb_to_oklab(f32 r,f32 g,f32 b) noexcept {
    const f32 l=0.4122214708f*r+0.5363325363f*g+0.0514459929f*b;
    const f32 m=0.2119034982f*r+0.6806995451f*g+0.1073969566f*b;
    const f32 s=0.0883024619f*r+0.2817188376f*g+0.6299787005f*b;
    const f32 lc=std::cbrt(l),mc=std::cbrt(m),sc=std::cbrt(s);
    return {0.2104542553f*lc+0.7936177850f*mc-0.0040720468f*sc,
            1.9779984951f*lc-2.4285922050f*mc+0.4505937099f*sc,
            0.0259040371f*lc+0.7827717662f*mc-0.8086757660f*sc};
}
std::array<f32,3> oklab_to_linear_srgb(Oklab c) noexcept {
    const f32 lc=c.L+0.3963377774f*c.a+0.2158037573f*c.b;
    const f32 mc=c.L-0.1055613458f*c.a-0.0638541728f*c.b;
    const f32 sc=c.L-0.0894841775f*c.a-1.2914855480f*c.b;
    const f32 l=lc*lc*lc,m=mc*mc*mc,s=sc*sc*sc;
    return {{ 4.0767416621f*l-3.3077115913f*m+0.2309699292f*s,
             -1.2684380046f*l+2.6097574011f*m-0.3413193965f*s,
             -0.0041960863f*l-0.7034186147f*m+1.7076147010f*s }};
}
// ─── Delta-E 2000 ─────────────────────────────────────────────────────────────
f32 delta_e2000(Lab l1,Lab l2,f32 kL,f32 kC,f32 kH) noexcept {
    constexpr f32 PI=3.14159265358979f;
    const f32 C1=std::sqrt(l1.a*l1.a+l1.b*l1.b);
    const f32 C2=std::sqrt(l2.a*l2.a+l2.b*l2.b);
    const f32 Cavg=(C1+C2)*0.5f;
    const f32 C7=std::pow(Cavg,7.0f);
    const f32 G=0.5f*(1-std::sqrt(C7/(C7+6103515625.0f))); // 25^7
    const f32 a1p=l1.a*(1+G),a2p=l2.a*(1+G);
    const f32 C1p=std::sqrt(a1p*a1p+l1.b*l1.b);
    const f32 C2p=std::sqrt(a2p*a2p+l2.b*l2.b);
    auto h=([&](f32 a,f32 b)->f32{f32 r=std::atan2(b,a)*180/PI;return r<0?r+360:r;});
    const f32 h1p=h(a1p,l1.b),h2p=h(a2p,l2.b);
    const f32 dLp=l2.L-l1.L,dCp=C2p-C1p;
    f32 dhp=h2p-h1p;
    if(C1p*C2p<1e-7f) dhp=0;
    else if(dhp>180) dhp-=360; else if(dhp<-180) dhp+=360;
    const f32 dHp=2*std::sqrt(C1p*C2p)*std::sin(dhp*PI/360);
    const f32 Lpa=(l1.L+l2.L)*0.5f,Cpa=(C1p+C2p)*0.5f;
    f32 Hpa=(h1p+h2p)*0.5f;
    if(C1p*C2p<1e-7f) Hpa=h1p+h2p;
    else if(std::abs(h2p-h1p)>180) Hpa+=(Hpa<180?180:-180);
    const f32 T=1-0.17f*std::cos((Hpa-30)*PI/180)+0.24f*std::cos(2*Hpa*PI/180)
               +0.32f*std::cos((3*Hpa+6)*PI/180)-0.20f*std::cos((4*Hpa-63)*PI/180);
    const f32 SL=1+0.015f*(Lpa-50)*(Lpa-50)/std::sqrt(20+(Lpa-50)*(Lpa-50));
    const f32 SC=1+0.045f*Cpa, SH=1+0.015f*Cpa*T;
    const f32 C7a=std::pow(Cpa,7.0f);
    const f32 RC=2*std::sqrt(C7a/(C7a+6103515625.0f));
    const f32 dt=30*std::exp(-((Hpa-275)/25)*((Hpa-275)/25));
    const f32 RT=-std::sin(2*dt*PI/180)*RC;
    const f32 term1=dLp/(kL*SL),term2=dCp/(kC*SC),term3=dHp/(kH*SH);
    return std::sqrt(term1*term1+term2*term2+term3*term3+RT*term2*term3);
}
} // namespace qs::math::color_ext
