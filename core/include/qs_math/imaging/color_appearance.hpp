#pragma once
// qs_math/color_appearance.hpp — Chromatic adaptation CAT02/CAT16, hue/brightness/colorfulness (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "color.hpp"
#include "color_vision.hpp"
#include <cmath>
#include <array>

namespace qs {

// ─── Chromatic adaptation transforms ─────────────────────────────────────────

// CAT02 chromatic adaptation (used in CIECAM02)
[[nodiscard]] inline Vec3f cat02_adapt(Vec3f xyz_src, Vec3f white_src, Vec3f white_dst, f32 D=1.f) {
    static constexpr f32 M02[3][3]={{0.7328f,0.4296f,-0.1624f},{-0.7036f,1.6975f,0.0061f},{0.0030f,0.0136f,0.9834f}};
    static constexpr f32 M02I[3][3]={{1.09612f,-0.27887f,0.18275f},{0.45437f,0.47353f,0.07210f},{-0.00963f,-0.00572f,1.01534f}};
    auto fwd=[](const f32 M[3][3],Vec3f v)->Vec3f{return {M[0][0]*v.x+M[0][1]*v.y+M[0][2]*v.z,M[1][0]*v.x+M[1][1]*v.y+M[1][2]*v.z,M[2][0]*v.x+M[2][1]*v.y+M[2][2]*v.z};};
    Vec3f RGB=fwd(M02,xyz_src),RGBws=fwd(M02,white_src),RGBwd=fwd(M02,white_dst);
    Vec3f RGBc={RGB.x*(D*RGBwd.x/RGBws.x+(1-D)),RGB.y*(D*RGBwd.y/RGBws.y+(1-D)),RGB.z*(D*RGBwd.z/RGBws.z+(1-D))};
    return fwd(M02I,RGBc);
}

// CAT16 (successor to CAT02)
[[nodiscard]] inline Vec3f cat16_adapt(Vec3f xyz_src, Vec3f white_src, Vec3f white_dst, f32 D=1.f) {
    static constexpr f32 M16[3][3]={{0.401288f,0.650173f,-0.051461f},{-0.250268f,1.204414f,0.045854f},{-0.002079f,0.048952f,0.953127f}};
    static constexpr f32 M16I[3][3]={{1.862068f,-1.011255f,0.149187f},{0.387526f,0.621447f,-0.008973f},{-0.015841f,-0.034123f,1.049964f}};
    auto fwd=[](const f32 M[3][3],Vec3f v)->Vec3f{return {M[0][0]*v.x+M[0][1]*v.y+M[0][2]*v.z,M[1][0]*v.x+M[1][1]*v.y+M[1][2]*v.z,M[2][0]*v.x+M[2][1]*v.y+M[2][2]*v.z};};
    Vec3f RGB=fwd(M16,xyz_src),ws=fwd(M16,white_src),wd=fwd(M16,white_dst);
    Vec3f RGBc={RGB.x*(D*wd.x/ws.x+1-D),RGB.y*(D*wd.y/ws.y+1-D),RGB.z*(D*wd.z/ws.z+1-D)};
    return fwd(M16I,RGBc);
}

// CMCCAT2000 (camera-based)
[[nodiscard]] inline Vec3f cmccat2000_adapt(Vec3f xyz_src, Vec3f white_src, Vec3f white_dst, f32 D=1.f) {
    static constexpr f32 M[3][3]={{0.7982f,0.3389f,-0.1371f},{-0.5918f,1.5512f,0.0406f},{0.0008f,0.0239f,0.9753f}};
    static constexpr f32 MI[3][3]={{1.0765f,-0.2373f,0.1608f},{0.4107f,0.5547f,0.0346f},{-0.0166f,-0.0157f,1.0322f}};
    auto fwd=[](const f32 Mx[3][3],Vec3f v)->Vec3f{return {Mx[0][0]*v.x+Mx[0][1]*v.y+Mx[0][2]*v.z,Mx[1][0]*v.x+Mx[1][1]*v.y+Mx[1][2]*v.z,Mx[2][0]*v.x+Mx[2][1]*v.y+Mx[2][2]*v.z};};
    Vec3f RGB=fwd(M,xyz_src),ws=fwd(M,white_src),wd=fwd(M,white_dst);
    Vec3f RGBc={RGB.x*(D*wd.x/ws.x+1-D),RGB.y*(D*wd.y/ws.y+1-D),RGB.z*(D*wd.z/ws.z+1-D)};
    return fwd(MI,RGBc);
}

// ─── Colour appearance correlates ─────────────────────────────────────────────
// Hue quadrature H (CAM02/16) to unique hues
[[nodiscard]] inline std::string hue_name(f32 H) {
    if(H<100) return "Red";
    if(H<200) return "Yellow";
    if(H<300) return "Green";
    if(H<400) return "Blue";
    return "Red";
}
[[nodiscard]] inline std::pair<std::string,f32> hue_composition(f32 H) {
    // Returns primary hue and amount of secondary hue
    static const char* hues[]={"Red","Yellow","Green","Blue"};
    u32 i=u32(H/100)%4;
    f32 frac=std::fmod(H,100)/100.f;
    return {hues[i],frac};
}

// Hunt-Pointer-Estevez colour space conversions
[[nodiscard]] inline Vec3f xyz_to_hpe(Vec3f xyz) {
    return {0.38971f*xyz.x+0.68898f*xyz.y-0.07868f*xyz.z,
           -0.22981f*xyz.x+1.18340f*xyz.y+0.04641f*xyz.z,
            xyz.z};
}

// Saturation from CAM02 — higher s = more chromatic relative to white
[[nodiscard]] inline f32 cam02_saturation(f32 C, f32 Q) {
    return 50*std::sqrt(0.69f*C/(Q+1e-10f))*100;
}

// Colourfulness from CAM02 appearance
[[nodiscard]] inline f32 colourfulness_from_chroma(f32 C, f32 F_L) {
    return C*std::pow(F_L,0.25f);
}

// ─── ICtCp (perceptual colour space for HDR, Dolby) ──────────────────────────
[[nodiscard]] inline Vec3f rgb_to_ictcp(Vec3f rgb_linear, f32 L_p=100.f) {
    // Step 1: scale to absolute luminance and apply PQ EOTF
    Vec3f abs={rgb_linear.x*L_p/10000.f,rgb_linear.y*L_p/10000.f,rgb_linear.z*L_p/10000.f};
    auto pq=[](f32 E)->f32{
        static constexpr f32 m1=0.1593017578125f,m2=78.84375f,c1=0.8359375f,c2=18.8515625f,c3=18.6875f;
        f32 Et=std::pow(std::max(0.f,E),m1);
        return std::pow((c1+c2*Et)/(1+c3*Et),m2);
    };
    Vec3f pqd={pq(abs.x),pq(abs.y),pq(abs.z)};
    // Step 2: LMS
    Vec3f lms={0.4002f*pqd.x+0.7076f*pqd.y-0.0808f*pqd.z,
              -0.2263f*pqd.x+1.1653f*pqd.y+0.0457f*pqd.z,
               0.f*pqd.x+0.f*pqd.y+0.9182f*pqd.z};
    // Step 3: ICtCp
    return {0.5f*lms.x+0.5f*lms.y,
            1.613769531f*lms.x-3.323486328f*lms.y+1.709716797f*lms.z,
            4.378173828f*lms.x-4.245605469f*lms.y-0.132568359f*lms.z};
}

// ─── Helmholtz-Kohlrausch effect (luminance appearance boost for saturated colors) ──
// Saturated colours appear brighter than grey of same luminance
[[nodiscard]] inline f32 helmholtz_kohlrausch(f32 Y, f32 chroma_C, f32 hue_h_deg) {
    // Simplified model: brightness increase proportional to chroma and hue-dependent factor
    f32 h_rad=hue_h_deg*f32(std::numbers::pi)/180;
    f32 K=0.1551f+0.1218f*std::cos(h_rad)-0.1380f*std::sin(h_rad)+0.0750f*std::cos(2*h_rad)+0.0258f*std::sin(2*h_rad);
    return Y*(1+K*chroma_C/100.f);
}

// ─── OKLCH / OKLAB (perceptual, Björn Ottosson 2020) ─────────────────────────
// (Already in color.hpp as oklab/oklch; here we add appearance-model extensions)

// OKLCH hue interpolation (shortest path)
[[nodiscard]] inline Vec3f oklch_lerp(Vec3f lch1, Vec3f lch2, f32 t) {
    f32 h1=lch1.z,h2=lch2.z;
    f32 dh=h2-h1;
    if(dh>180) h2-=360; if(dh<-180) h2+=360;
    return {lch1.x+(lch2.x-lch1.x)*t,lch1.y+(lch2.y-lch1.y)*t,h1+(h2-h1)*t};
}

// Perceptual contrast (APCA-like, simplified)
[[nodiscard]] inline f32 perceptual_contrast(Vec3f text_rgb, Vec3f bg_rgb) {
    auto luma=[](Vec3f c)->f32{
        auto lin=[](f32 x)->f32{return x<=0.04045f?x/12.92f:std::pow((x+0.055f)/1.055f,2.4f);};
        return 0.2126f*lin(c.x)+0.7152f*lin(c.y)+0.0722f*lin(c.z);
    };
    f32 Yt=luma(text_rgb), Yb=luma(bg_rgb);
    f32 Sapc=std::pow(Yb,0.56f)-std::pow(Yt,0.57f);
    f32 Rapc=Sapc>=0?1.14f*Sapc:-1.14f*Sapc;
    return Rapc*100;
}

// ─── Adaptation degree estimator (field luminance → D) ───────────────────────
[[nodiscard]] inline f32 adaptation_degree(f32 L_A, f32 Y_w=100.f) {
    // Full adaptation when L_A is high; partial in dark
    return std::clamp(1.f-std::exp(-(L_A+42)/92.f)/3.6f,0.f,1.f);
}

} // namespace qs
