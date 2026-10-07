#pragma once
// qs_math/color_vision.hpp — Color deficiency simulation, CAM16, CIECAM02 (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "color.hpp"
#include "color_science.hpp"
#include <cmath>
#include <array>
#include <algorithm>

namespace qs {

// ─── Colour deficiency (CVD) simulation ──────────────────────────────────────
// Brettel 1997 / Viénot 1999 models for protan, deutan, tritan

// LMS cone fundamentals matrix (Hunt-Pointer-Estevez, D65-adapted)
static constexpr f32 RGB_TO_LMS[3][3] = {
    { 0.4002f, 0.7076f, -0.0808f},
    {-0.2263f, 1.1653f,  0.0457f},
    { 0.0000f, 0.0000f,  0.9182f}
};
static constexpr f32 LMS_TO_RGB[3][3] = {
    { 1.8599f,-1.1295f,  0.2199f},
    { 0.3612f, 0.6388f, -0.0000f},
    { 0.0000f, 0.0000f,  1.0889f}
};

[[nodiscard]] inline Vec3f rgb_to_lms(Vec3f rgb) {
    return {RGB_TO_LMS[0][0]*rgb.x+RGB_TO_LMS[0][1]*rgb.y+RGB_TO_LMS[0][2]*rgb.z,
            RGB_TO_LMS[1][0]*rgb.x+RGB_TO_LMS[1][1]*rgb.y+RGB_TO_LMS[1][2]*rgb.z,
            RGB_TO_LMS[2][0]*rgb.x+RGB_TO_LMS[2][1]*rgb.y+RGB_TO_LMS[2][2]*rgb.z};
}
[[nodiscard]] inline Vec3f lms_to_rgb(Vec3f lms) {
    return {LMS_TO_RGB[0][0]*lms.x+LMS_TO_RGB[0][1]*lms.y+LMS_TO_RGB[0][2]*lms.z,
            LMS_TO_RGB[1][0]*lms.x+LMS_TO_RGB[1][1]*lms.y+LMS_TO_RGB[1][2]*lms.z,
            LMS_TO_RGB[2][0]*lms.x+LMS_TO_RGB[2][1]*lms.y+LMS_TO_RGB[2][2]*lms.z};
}

// CVD simulation in LMS space
// severity: 0.0 = normal, 1.0 = dichromat
[[nodiscard]] inline Vec3f simulate_protanopia(Vec3f rgb, f32 severity=1.f) {
    auto lms=rgb_to_lms(rgb);
    // Protan: L cone absent → L derived from M and S
    f32 L_sim=(2.02344f*lms.y-2.52581f*lms.z);
    f32 L_new=lms.x*(1-severity)+L_sim*severity;
    return lms_to_rgb({L_new,lms.y,lms.z});
}

[[nodiscard]] inline Vec3f simulate_deuteranopia(Vec3f rgb, f32 severity=1.f) {
    auto lms=rgb_to_lms(rgb);
    // Deutan: M cone absent → M derived from L and S
    f32 M_sim=(0.49421f*lms.x+1.24827f*lms.z);
    f32 M_new=lms.y*(1-severity)+M_sim*severity;
    return lms_to_rgb({lms.x,M_new,lms.z});
}

[[nodiscard]] inline Vec3f simulate_tritanopia(Vec3f rgb, f32 severity=1.f) {
    auto lms=rgb_to_lms(rgb);
    // Tritan: S cone absent → S derived from L and M
    f32 S_sim=(-0.01224f*lms.x+0.07203f*lms.y);
    f32 S_new=lms.z*(1-severity)+S_sim*severity;
    return lms_to_rgb({lms.x,lms.y,S_new});
}

// Achromatopsia (rod monochromacy — greyscale)
[[nodiscard]] inline Vec3f simulate_achromatopsia(Vec3f rgb, f32 severity=1.f) {
    f32 luma=0.2126f*rgb.x+0.7152f*rgb.y+0.0722f*rgb.z;
    return rgb+(Vec3f{luma,luma,luma}-rgb)*severity;
}

// Daltonize: recolour image to improve CVD accessibility
[[nodiscard]] inline Vec3f daltonize_protan(Vec3f rgb) {
    auto sim=simulate_protanopia(rgb);
    Vec3f err=rgb-sim;
    // Shift error from red channel to green/blue
    return {rgb.x, rgb.y+0.7f*err.x, rgb.z+0.7f*err.x};
}
[[nodiscard]] inline Vec3f daltonize_deutan(Vec3f rgb) {
    auto sim=simulate_deuteranopia(rgb);
    Vec3f err=rgb-sim;
    return {rgb.x+1.0f*err.y, rgb.y, rgb.z+0.1f*err.y};
}

// ─── CIECAM02 colour appearance model ────────────────────────────────────────
struct CIECAM02Env {
    f32 XYZ_w[3]{95.05f,100.f,108.88f}; // white point XYZ
    f32 L_A{64.f};    // luminance of adapting field (cd/m²)
    f32 Y_b{20.f};    // luminance of background (% of Y_w)
    f32 c{0.69f};     // surround factor (average=0.69,dim=0.59,dark=0.525)
    f32 N_c{1.0f};    // chromatic induction (1.0=average)
    f32 F{1.0f};      // adaptation factor (1.0=full)

    // Derived viewing condition parameters
    [[nodiscard]] f32 k()  const { return 1/(5*L_A+1); }
    [[nodiscard]] f32 F_L()const { f32 kv=k(); return 0.2f*std::pow(kv,4)*(5*L_A)+0.1f*std::pow(1-std::pow(kv,4),2)*std::pow(5*L_A,1.f/3); }
    [[nodiscard]] f32 n()  const { return Y_b/XYZ_w[1]; }
    [[nodiscard]] f32 z()  const { return 1.48f+std::sqrt(50*n()); }
    [[nodiscard]] f32 N_bb()const{ return 0.725f*std::pow(1/n(),0.2f); }
    [[nodiscard]] f32 N_cb()const{ return N_bb(); }
};

struct CAM02Appearance {
    f32 J{0};   // lightness
    f32 C{0};   // chroma
    f32 h{0};   // hue angle (degrees)
    f32 s{0};   // saturation
    f32 Q{0};   // brightness
    f32 M{0};   // colourfulness
    f32 H{0};   // hue quadrature
};

[[nodiscard]] inline CAM02Appearance ciecam02_forward(Vec3f xyz, const CIECAM02Env& env) {
    // Step 1: CAT02 chromatic adaptation
    static constexpr f32 M_CAT02[3][3]={{0.7328f,0.4296f,-0.1624f},{-0.7036f,1.6975f,0.0061f},{0.0030f,0.0136f,0.9834f}};
    auto cat=[&](f32 m[3][3],f32 x,f32 y,f32 z)->Vec3f{ return {m[0][0]*x+m[0][1]*y+m[0][2]*z,m[1][0]*x+m[1][1]*y+m[1][2]*z,m[2][0]*x+m[2][1]*y+m[2][2]*z}; };
    auto RGB=cat(const_cast<f32(*)[3]>(M_CAT02),xyz.x,xyz.y,xyz.z);
    auto RGB_w=cat(const_cast<f32(*)[3]>(M_CAT02),env.XYZ_w[0],env.XYZ_w[1],env.XYZ_w[2]);
    f32 D=std::clamp(env.F*(1-1.f/3.6f*std::exp((-env.L_A-42)/92.f)),0.f,1.f);
    auto adapt=[&](f32 c_,f32 cw)->f32{ return D*env.XYZ_w[1]*c_/cw+(1-D)*c_; };
    Vec3f RGBc={adapt(RGB.x,RGB_w.x),adapt(RGB.y,RGB_w.y),adapt(RGB.z,RGB_w.z)};
    // Step 2: HPE matrix
    static constexpr f32 M_HPE[3][3]={{0.38971f,0.68898f,-0.07868f},{-0.22981f,1.18340f,0.04641f},{0.f,0.f,1.f}};
    static constexpr f32 M_CAT02_INV[3][3]={{1.09612f,-0.27887f,0.18275f},{0.45437f,0.47353f,0.07210f},{-0.00963f,-0.00572f,1.01534f}};
    auto xyz_c=cat(const_cast<f32(*)[3]>(M_CAT02_INV),RGBc.x,RGBc.y,RGBc.z);
    auto RGB_hpe=cat(const_cast<f32(*)[3]>(M_HPE),xyz_c.x,xyz_c.y,xyz_c.z);
    // Step 3: Non-linear adaptation
    f32 FL=env.F_L();
    auto adapt_nl=[&](f32 c_)->f32{
        f32 t=std::pow(FL*std::abs(c_)/100.f,0.42f);
        return std::copysign(400.f*t/(t+27.13f)+0.1f,c_);
    };
    Vec3f RGBp={adapt_nl(RGB_hpe.x),adapt_nl(RGB_hpe.y),adapt_nl(RGB_hpe.z)};
    // Step 4: Opponent colour dimensions
    f32 a=RGBp.x-12.f*RGBp.y/11.f+RGBp.z/11.f;
    f32 b=(RGBp.x+RGBp.y-2.f*RGBp.z)/9.f;
    f32 h_=std::atan2(b,a)*180.f/f32(std::numbers::pi);
    if(h_<0) h_+=360;
    // Step 5: Achromatic signal
    f32 A=(2.f*RGBp.x+RGBp.y+RGBp.z/20.f-0.305f)*env.N_bb();
    // Step 6: Lightness J
    f32 Aw=(2.f*adapt_nl(RGB_w.x)+adapt_nl(RGB_w.y)+adapt_nl(RGB_w.z)/20.f-0.305f)*env.N_bb();
    f32 J=100.f*std::pow(A/Aw,env.c*env.z());
    // Step 7: Hue quadrature
    static constexpr f32 h_i[5]={20.14f,90.f,164.25f,237.53f,380.14f};
    static constexpr f32 e_i[4]={0.8f,0.7f,1.0f,1.2f};
    static constexpr f32 H_i[4]={0,100,200,300};
    u32 q=0; while(q<3&&h_>h_i[q+1]) ++q;
    f32 t_h=(h_-h_i[q])/(e_i[q]); f32 tb=(h_i[q+1]-h_)/(e_i[q]);
    f32 H_=H_i[q]+100*t_h/(t_h+tb);
    // Step 8: Chroma, colourfulness, saturation
    f32 t=50000.f/13.f*env.N_c*env.N_cb*(e_i[q]*(std::sqrt(a*a+b*b)/(RGBp.x+RGBp.y+21.f/20*RGBp.z+1e-10f)));
    f32 C=std::pow(t,0.9f)*std::sqrt(J/100.f)*std::pow(1.64f-std::pow(0.29f,env.n()),0.73f);
    f32 n_=env.n(); f32 FL_n=env.F_L();
    f32 Q=4/env.c*(std::sqrt(J/100.f))*(Aw+4)*std::pow(FL_n,0.25f);
    f32 M=C*std::pow(FL_n,0.25f);
    f32 s=100*std::sqrt(M/Q+1e-10f);
    return {J,C,h_,s,Q,M,H_};
}

// ─── CAM16 (simplified, updated CIECAM02) ────────────────────────────────────
struct CAM16 {
    f32 J{0},C{0},h{0},M{0},s{0},Q{0};
};

[[nodiscard]] inline CAM16 cam16_forward(Vec3f xyz, Vec3f white_xyz={95.05f,100.f,108.88f},
    f32 L_A=64.f, f32 Y_b=20.f)
{
    // CAM16 uses same structure as CIECAM02 with updated M_16 matrix
    static constexpr f32 M16[3][3]={{0.401288f,0.650173f,-0.051461f},{-0.250268f,1.204414f,0.045854f},{-0.002079f,0.048952f,0.953127f}};
    auto m=[](const f32 M[3][3],Vec3f v)->Vec3f{ return {M[0][0]*v.x+M[0][1]*v.y+M[0][2]*v.z,M[1][0]*v.x+M[1][1]*v.y+M[1][2]*v.z,M[2][0]*v.x+M[2][1]*v.y+M[2][2]*v.z}; };
    Vec3f w={white_xyz.x,white_xyz.y,white_xyz.z};
    auto RGB  =m(M16,xyz);
    auto RGB_w=m(M16,w);
    f32 k=1/(5*L_A+1);
    f32 F_L=0.2f*std::pow(k,4)*5*L_A+0.1f*std::pow(1-std::pow(k,4),2)*std::pow(5*L_A,1.f/3);
    f32 D=std::clamp(1.f-1.f/3.6f*std::exp((-L_A-42)/92.f),0.f,1.f);
    auto adpt=[&](f32 c_,f32 cw)->f32{ return c_*D*w.y/cw+(1-D)*c_; };
    Vec3f RGBa={adpt(RGB.x,RGB_w.x),adpt(RGB.y,RGB_w.y),adpt(RGB.z,RGB_w.z)};
    auto nl=[&](f32 x)->f32{ f32 t=std::pow(F_L*std::abs(x)/100.f,0.42f); return std::copysign(400*t/(t+27.13f)+0.1f,x); };
    Vec3f p={nl(RGBa.x),nl(RGBa.y),nl(RGBa.z)};
    f32 a=p.x-12*p.y/11.f+p.z/11.f, b=(p.x+p.y-2*p.z)/9.f;
    f32 hue=std::atan2(b,a)*180.f/f32(std::numbers::pi); if(hue<0) hue+=360;
    f32 n=Y_b/w.y, z=1.48f+std::sqrt(50*n), N_bb=0.725f*std::pow(1/n,0.2f);
    f32 Aw=(2*nl(RGB_w.x)+nl(RGB_w.y)+nl(RGB_w.z)/20-0.305f)*N_bb;
    f32 A=(2*p.x+p.y+p.z/20-0.305f)*N_bb;
    f32 J=100*std::pow(A/Aw,0.69f*z);
    f32 t=50000.f/13*N_bb*(std::sqrt(a*a+b*b)/(p.x+p.y+21.f/20*p.z+1e-10f));
    f32 C=std::pow(t,0.9f)*std::sqrt(J/100)*std::pow(1.64f-std::pow(0.29f,n),0.73f);
    f32 M_cam=C*std::pow(F_L,0.25f);
    f32 Q=4/0.69f*std::sqrt(J/100)*(Aw+4)*std::pow(F_L,0.25f);
    f32 s=50*std::sqrt(0.69f*C/Q+1e-10f)*100;
    return {J,C,hue,M_cam,s,Q};
}

// ─── Colour vision deficiency detection helpers ────────────────────────────────
// Ishihara-style confusion line endpoints for CVD types
struct ConfusionLine { Vec3f protan_axis,deutan_axis,tritan_axis; };
static const ConfusionLine CONFUSION_AXES{
    {0.9f,0.09f,0.f},    // protan (red-green)
    {0.1f,0.9f,0.f},     // deutan (red-green)
    {0.f,0.1f,0.9f}      // tritan (blue-yellow)
};

// Check if two colours are likely to be confused by a CVD observer
[[nodiscard]] inline bool likely_confused_protan(Vec3f c1, Vec3f c2, f32 threshold=0.1f) {
    auto s1=simulate_protanopia(c1), s2=simulate_protanopia(c2);
    return dist(s1,s2)<threshold;
}
[[nodiscard]] inline bool likely_confused_deutan(Vec3f c1, Vec3f c2, f32 threshold=0.1f) {
    auto s1=simulate_deuteranopia(c1), s2=simulate_deuteranopia(c2);
    return dist(s1,s2)<threshold;
}

} // namespace qs
