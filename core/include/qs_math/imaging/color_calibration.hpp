#pragma once
// qs_math/color_calibration.hpp — White balance math, Delta-E, gamut mapping, ICC profile math (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "color.hpp"
#include "color_science.hpp"
#include "linear_algebra.hpp"
#include <cmath>
#include <array>
#include <vector>
#include <optional>

namespace qs {

// ─── White balance ────────────────────────────────────────────────────────────
// Grey World algorithm
[[nodiscard]] inline Vec3f grey_world_white_balance(const std::vector<Vec3f>& pixels) {
    Vec3f sum{}; for(const auto& p:pixels) sum=sum+p;
    f32 avg=(sum.x+sum.y+sum.z)/(3*pixels.size()+1e-10f);
    Vec3f scale={avg/sum.x*pixels.size(),avg/sum.y*pixels.size(),avg/sum.z*pixels.size()};
    return scale;
}

// Manual white balance from a known neutral point
[[nodiscard]] inline Vec3f white_balance_from_neutral(Vec3f neutral_rgb) {
    f32 g=neutral_rgb.y;
    return {g/(neutral_rgb.x+1e-10f),1.f,g/(neutral_rgb.z+1e-10f)};
}

// Correlated colour temperature → D-illuminant XYZ (McCamy 1992)
[[nodiscard]] inline Vec3f cct_to_d_illuminant_xy(f32 T) {
    f64 t=T; f64 x,y;
    if(t<=7000){
        x=-4.607e9/(t*t*t)+2.9678e6/(t*t)+0.09911e3/t+0.244063;
    } else {
        x=-2.0064e9/(t*t*t)+1.9018e6/(t*t)+0.24748e3/t+0.237040;
    }
    y=-3*x*x+2.87*x-0.275;
    // xy to XYZ (Y=1)
    return {f32(x/y),1.f,f32((1-x-y)/y)};
}

// Von Kries chromatic adaptation transform (diagonal in LMS space)
[[nodiscard]] inline Vec3f von_kries_adapt(Vec3f rgb, Vec3f src_white_xyz, Vec3f dst_white_xyz) {
    // Convert whites to LMS
    static constexpr f32 RGB2LMS[3][3]={{0.4002f,0.7076f,-0.0808f},{-0.2263f,1.1653f,0.0457f},{0.f,0.f,0.9182f}};
    static constexpr f32 LMS2RGB[3][3]={{1.8599f,-1.1295f,0.2199f},{0.3612f,0.6388f,0.f},{0.f,0.f,1.0889f}};
    auto m=[](const f32 M[3][3],Vec3f v)->Vec3f{return {M[0][0]*v.x+M[0][1]*v.y+M[0][2]*v.z,M[1][0]*v.x+M[1][1]*v.y+M[1][2]*v.z,M[2][0]*v.x+M[2][1]*v.y+M[2][2]*v.z};};
    Vec3f lms_src=m(RGB2LMS,src_white_xyz), lms_dst=m(RGB2LMS,dst_white_xyz);
    Vec3f lms_rgb=m(RGB2LMS,rgb);
    Vec3f lms_adapted={lms_rgb.x*lms_dst.x/(lms_src.x+1e-10f),lms_rgb.y*lms_dst.y/(lms_src.y+1e-10f),lms_rgb.z*lms_dst.z/(lms_src.z+1e-10f)};
    return m(LMS2RGB,lms_adapted);
}

// Bradford CAT (more perceptually uniform than Von Kries)
[[nodiscard]] inline Vec3f bradford_adapt(Vec3f xyz, Vec3f src_white, Vec3f dst_white) {
    static constexpr f32 M_BRAD[3][3]={{ 0.8951f, 0.2664f,-0.1614f},{-0.7502f, 1.7135f, 0.0367f},{ 0.0389f,-0.0685f, 1.0296f}};
    static constexpr f32 M_BRAD_INV[3][3]={{0.9869929f,-0.1470543f,0.1599627f},{0.4323053f,0.5183603f,0.0492912f},{-0.0085287f,0.0400428f,0.9684867f}};
    auto m=[](const f32 M[3][3],Vec3f v)->Vec3f{return {M[0][0]*v.x+M[0][1]*v.y+M[0][2]*v.z,M[1][0]*v.x+M[1][1]*v.y+M[1][2]*v.z,M[2][0]*v.x+M[2][1]*v.y+M[2][2]*v.z};};
    Vec3f ρ_s=m(M_BRAD,src_white),ρ_d=m(M_BRAD,dst_white),ρ_xyz=m(M_BRAD,xyz);
    Vec3f ρ_adapted={ρ_xyz.x*ρ_d.x/ρ_s.x,ρ_xyz.y*ρ_d.y/ρ_s.y,ρ_xyz.z*ρ_d.z/ρ_s.z};
    return m(M_BRAD_INV,ρ_adapted);
}

// ─── Delta-E colour difference formulas ──────────────────────────────────────

// Delta-E 1976 (CIE76): simple Euclidean in LAB
[[nodiscard]] inline f32 delta_e_76(Vec3f lab1, Vec3f lab2) {
    Vec3f d=lab1-lab2; return std::sqrt(dot(d,d));
}

// Delta-E 1994
[[nodiscard]] inline f32 delta_e_94(Vec3f lab1, Vec3f lab2,
    f32 k_L=1,f32 k_C=1,f32 k_H=1,bool graphic=false)
{
    f32 K1=graphic?0.048f:0.045f, K2=graphic?0.014f:0.015f;
    f32 C1=std::sqrt(lab1.y*lab1.y+lab1.z*lab1.z);
    f32 C2=std::sqrt(lab2.y*lab2.y+lab2.z*lab2.z);
    f32 dL=lab1.x-lab2.x, dC=C1-C2;
    f32 da=lab1.y-lab2.y, db=lab1.z-lab2.z;
    f32 dH2=da*da+db*db-dC*dC;
    f32 dH=std::sqrt(std::max(0.f,dH2));
    f32 SL=1,SC=1+K1*C1,SH=1+K2*C1;
    f32 eL=dL/(k_L*SL),eC=dC/(k_C*SC),eH=dH/(k_H*SH);
    return std::sqrt(eL*eL+eC*eC+eH*eH);
}

// Delta-E 2000 (CIEDE2000): most perceptually accurate
[[nodiscard]] inline f32 delta_e_2000(Vec3f lab1, Vec3f lab2,
    f32 k_L=1,f32 k_C=1,f32 k_H=1)
{
    auto deg=[](f32 r)->f32{ return r*180.f/f32(std::numbers::pi); };
    auto rad=[](f32 d)->f32{ return d*f32(std::numbers::pi)/180.f; };
    f32 L1=lab1.x,a1=lab1.y,b1=lab1.z;
    f32 L2=lab2.x,a2=lab2.y,b2=lab2.z;
    f32 C1=std::sqrt(a1*a1+b1*b1), C2=std::sqrt(a2*a2+b2*b2);
    f32 Cab=0.5f*(C1+C2), Cab7=std::pow(Cab,7.f);
    f32 G=0.5f*(1-std::sqrt(Cab7/(Cab7+6103515625.f)));
    f32 a1p=a1*(1+G),a2p=a2*(1+G);
    f32 C1p=std::sqrt(a1p*a1p+b1*b1),C2p=std::sqrt(a2p*a2p+b2*b2);
    auto hprime=[&](f32 bp,f32 ap)->f32{
        if(std::abs(ap)<1e-7f&&std::abs(bp)<1e-7f) return 0;
        f32 h=deg(std::atan2(bp,ap)); return h<0?h+360:h;
    };
    f32 h1p=hprime(b1,a1p),h2p=hprime(b2,a2p);
    f32 dLp=L2-L1,dCp=C2p-C1p;
    f32 dhp; f32 diff=h2p-h1p;
    if(C1p*C2p<1e-7f) dhp=0;
    else if(std::abs(diff)<=180) dhp=diff;
    else if(diff>180) dhp=diff-360;
    else dhp=diff+360;
    f32 dHp=2*std::sqrt(C1p*C2p)*std::sin(rad(dhp*0.5f));
    f32 Lpm=0.5f*(L1+L2),Cpm=0.5f*(C1p+C2p);
    f32 hpm; f32 absdiff=std::abs(h1p-h2p);
    if(C1p*C2p<1e-7f) hpm=h1p+h2p;
    else if(absdiff<=180) hpm=0.5f*(h1p+h2p);
    else if(h1p+h2p<360) hpm=0.5f*(h1p+h2p)+180;
    else hpm=0.5f*(h1p+h2p)-180;
    f32 T=1-0.17f*std::cos(rad(hpm-30))+0.24f*std::cos(rad(2*hpm))+0.32f*std::cos(rad(3*hpm+6))-0.20f*std::cos(rad(4*hpm-63));
    f32 SL=1+0.015f*(Lpm-50)*(Lpm-50)/std::sqrt(20+(Lpm-50)*(Lpm-50));
    f32 SC=1+0.045f*Cpm, SH=1+0.015f*Cpm*T;
    f32 Cpm7=std::pow(Cpm,7.f);
    f32 RC=2*std::sqrt(Cpm7/(Cpm7+6103515625.f));
    f32 dt=30*std::exp(-((hpm-275)/25)*((hpm-275)/25));
    f32 RT=-std::sin(rad(2*dt))*RC;
    f32 eL=dLp/(k_L*SL),eC=dCp/(k_C*SC),eH=dHp/(k_H*SH);
    return std::sqrt(eL*eL+eC*eC+eH*eH+RT*eC*eH);
}

// ─── Gamut mapping ────────────────────────────────────────────────────────────
// Gamut boundary description (simplified convex hull in LCH)
struct GamutBoundary {
    std::vector<f32> max_chroma; // max chroma at each (L,h) pair
    u32 L_steps{16},h_steps{36};

    [[nodiscard]] f32 max_chroma_at(f32 L,f32 h) const {
        u32 li=u32(std::clamp(L,0.f,100.f)/100*(L_steps-1));
        u32 hi=u32(std::fmod(h+360,360)/360*h_steps)%h_steps;
        return max_chroma[li*h_steps+hi];
    }
};

// Clip gamut mapping (simple: clamp RGB to [0,1])
[[nodiscard]] inline Vec3f gamut_clip(Vec3f rgb) {
    return {std::clamp(rgb.x,0.f,1.f),std::clamp(rgb.y,0.f,1.f),std::clamp(rgb.z,0.f,1.f)};
}

// Chroma-reduction gamut mapping (reduce chroma until in-gamut)
[[nodiscard]] inline Vec3f gamut_map_chroma(Vec3f lab, u32 max_iter=50) {
    Vec3f lch=lab_to_lch(lab);
    for(u32 i=0;i<max_iter;++i) {
        Vec3f rgb=xyz_to_srgb(lab_to_xyz(lch_to_lab(lch)));
        if(rgb.x>=0&&rgb.x<=1&&rgb.y>=0&&rgb.y<=1&&rgb.z>=0&&rgb.z<=1) break;
        lch.y*=0.9f; // reduce chroma by 10%
    }
    return lch_to_lab(lch);
}

// Hue-preserving gamut mapping with lightness adjustment
[[nodiscard]] inline Vec3f gamut_map_hue_preserving(Vec3f lab) {
    // 1) Try chroma reduction
    auto mapped=gamut_map_chroma(lab);
    Vec3f rgb=xyz_to_srgb(lab_to_xyz(mapped));
    if(rgb.x>=0&&rgb.x<=1&&rgb.y>=0&&rgb.y<=1&&rgb.z>=0&&rgb.z<=1) return mapped;
    // 2) Adjust lightness toward neutral
    Vec3f lch=lab_to_lch(mapped);
    f32 target_L=lch.z<180?100:0; // move toward white for hues 0-180, black otherwise
    for(u32 i=0;i<30;++i) {
        lch.x=lch.x*0.9f+target_L*0.1f; lch.y*=0.9f;
        Vec3f r=xyz_to_srgb(lab_to_xyz(lch_to_lab(lch)));
        if(r.x>=0&&r.x<=1&&r.y>=0&&r.y<=1&&r.z>=0&&r.z<=1) return lch_to_lab(lch);
    }
    return {};
}

// ─── ICC profile math ─────────────────────────────────────────────────────────
// ICC v4 signature constants
static constexpr u32 ICC_MAGIC = 0x61637370; // 'acsp'
static constexpr u32 ICC_RGB   = 0x52474220; // 'RGB '
static constexpr u32 ICC_XYZ   = 0x58595A20; // 'XYZ '
static constexpr u32 ICC_LAB   = 0x4C616220; // 'Lab '

// Fixed-point ICC S15Fixed16Number encoding
[[nodiscard]] inline i32 icc_s15f16_encode(f64 v) { return i32(v*65536.0); }
[[nodiscard]] inline f64 icc_s15f16_decode(i32 v) { return v/65536.0; }

// ICC u8Fixed8Number
[[nodiscard]] inline u16 icc_u8f8_encode(f64 v) { return u16(v*256.0); }
[[nodiscard]] inline f64 icc_u8f8_decode(u16 v) { return v/256.0; }

// Chromatic adaptation matrix for ICC (PCS=D50)
static constexpr Vec3f ICC_D50{0.9642f,1.0f,0.8251f};
[[nodiscard]] inline Vec3f adapt_to_icc_pcs(Vec3f xyz_d65) {
    return bradford_adapt(xyz_d65,{0.9505f,1.f,1.0888f},ICC_D50);
}

// TRC (Tone Reproduction Curve) evaluation: parametric type 3 (sRGB-like)
[[nodiscard]] inline f32 icc_trc_type3(f32 x, f32 g,f32 a,f32 b,f32 c,f32 d,f32 e,f32 f) {
    // (aX+b)^g + e  if X >= d, else cX+f
    return x>=d?std::pow(a*x+b,g)+e:c*x+f;
}
[[nodiscard]] inline f32 srgb_trc(f32 x) {
    return icc_trc_type3(x,2.4f,1.f/1.055f,0.055f/1.055f,1.f/12.92f,0.04045f,0.f,0.f);
}

// 3×3 colour matrix tag (used in ICC profiles)
struct ICCMatrix3x3 {
    f32 m[3][3];
    [[nodiscard]] Vec3f apply(Vec3f v) const {
        return {m[0][0]*v.x+m[0][1]*v.y+m[0][2]*v.z,
                m[1][0]*v.x+m[1][1]*v.y+m[1][2]*v.z,
                m[2][0]*v.x+m[2][1]*v.y+m[2][2]*v.z};
    }
};

// sRGB ICC profile matrices (D65)
static const ICCMatrix3x3 SRGB_TO_XYZ_D65={{{0.4124564f,0.3575761f,0.1804375f},{0.2126729f,0.7151522f,0.0721750f},{0.0193339f,0.1191920f,0.9503041f}}};
static const ICCMatrix3x3 XYZ_D65_TO_SRGB={{{3.2404542f,-1.5371385f,-0.4985314f},{-0.9692660f,1.8760108f,0.0415560f},{0.0556434f,-0.2040259f,1.0572252f}}};

// Display-P3 to XYZ D65
static const ICCMatrix3x3 P3_TO_XYZ_D65={{{0.4865709f,0.2656677f,0.1982173f},{0.2289946f,0.6917385f,0.0792669f},{0.0000000f,0.0451134f,1.0439444f}}};

// BT.2020 to XYZ D65
static const ICCMatrix3x3 BT2020_TO_XYZ_D65={{{0.6369580f,0.1446169f,0.1688810f},{0.2627002f,0.6779981f,0.0593017f},{0.0000000f,0.0280727f,1.0609851f}}};

// ─── Profiling: compute camera/scanner colour matrix from test chart ──────────
// Given measured LAB values and chart reference LAB values, compute 3x3 correction matrix
[[nodiscard]] inline std::optional<ICCMatrix3x3> fit_colour_matrix(
    const std::vector<Vec3f>& measured_lab,
    const std::vector<Vec3f>& reference_lab)
{
    u32 n=u32(std::min(measured_lab.size(),reference_lab.size()));
    // Least-squares: find M such that M * measured ≈ reference
    // Build system for each output channel
    ICCMatrix3x3 M{};
    for(u32 row=0;row<3;++row) {
        DenseMatrix<f64> A(n,3);
        std::vector<f64> b(n);
        for(u32 i=0;i<n;++i) {
            A.at(i,0)=f64(row==0?measured_lab[i].x:row==1?measured_lab[i].y:measured_lab[i].z);
            // Actually build proper 3x3 system
            A.at(i,0)=measured_lab[i].x; A.at(i,1)=measured_lab[i].y; A.at(i,2)=measured_lab[i].z;
            b[i]=row==0?reference_lab[i].x:row==1?reference_lab[i].y:reference_lab[i].z;
        }
        auto sol=least_squares_normal(A,b);
        if(!sol) return std::nullopt;
        M.m[row][0]=f32((*sol)[0]); M.m[row][1]=f32((*sol)[1]); M.m[row][2]=f32((*sol)[2]);
    }
    return M;
}

} // namespace qs
