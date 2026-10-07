#pragma once
// qs_math/rendering/pbr_math.hpp — Physically-based rendering math:
//   Cook-Torrance BRDF, GGX/Trowbridge-Reitz NDF, Smith G2, Fresnel,
//   energy compensation, importance sampling, LTC, sheen, iridescence (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <cmath>

namespace qs::math::render {

// ─── GGX (Trowbridge-Reitz) NDF ───────────────────────────────────────────
// D(m) = alpha^2 / (pi * (NdotM^2 * (alpha^2 - 1) + 1)^2)
[[nodiscard]] inline f32 ggx_ndf(f32 NdotH, f32 alpha) noexcept {
    f32 a2=alpha*alpha;
    f32 denom=NdotH*NdotH*(a2-1.f)+1.f;
    return a2/(f32(PI)*denom*denom);
}

// Anisotropic GGX NDF (Burley 2012)
[[nodiscard]] inline f32 ggx_ndf_aniso(f32 NdotH, f32 TdotH, f32 BdotH,
                                        f32 ax, f32 ay) noexcept
{
    f32 d=(TdotH/ax)*(TdotH/ax)+(BdotH/ay)*(BdotH/ay)+NdotH*NdotH;
    return 1.f/(f32(PI)*ax*ay*d*d);
}

// ─── Smith height-correlated G2 (Heitz 2014) ──────────────────────────────
[[nodiscard]] inline f32 smith_g1_ggx(f32 NdotV, f32 alpha) noexcept {
    f32 a2=alpha*alpha;
    return 2.f*NdotV/(NdotV+std::sqrt(a2+(1.f-a2)*NdotV*NdotV));
}

[[nodiscard]] inline f32 smith_g2_ggx(f32 NdotV, f32 NdotL, f32 alpha) noexcept {
    f32 a2=alpha*alpha;
    f32 gv=NdotL*std::sqrt(a2+(1.f-a2)*NdotV*NdotV);
    f32 gl=NdotV*std::sqrt(a2+(1.f-a2)*NdotL*NdotL);
    return 0.5f/(gv+gl);
}

// Anisotropic Smith G2
[[nodiscard]] inline f32 smith_g2_aniso(f32 NdotV, f32 NdotL,
                                         f32 TdotV, f32 BdotV,
                                         f32 TdotL, f32 BdotL,
                                         f32 ax, f32 ay) noexcept
{
    auto lambda=[&](f32 NdotX, f32 TdotX, f32 BdotX){
        f32 t=(ax*TdotX)*(ax*TdotX)+(ay*BdotX)*(ay*BdotX);
        return 0.5f*(-1.f+std::sqrt(1.f+t/(NdotX*NdotX)));
    };
    return 1.f/(1.f+lambda(NdotV,TdotV,BdotV)+lambda(NdotL,TdotL,BdotL));
}

// ─── Fresnel ──────────────────────────────────────────────────────────────
// Schlick approximation
[[nodiscard]] inline Vec3f fresnel_schlick(f32 VdotH, Vec3f F0) noexcept {
    f32 x=1.f-VdotH, x5=x*x*x*x*x;
    return F0 + (Vec3f{1,1,1}-F0)*x5;
}

// Schlick with roughness (IBL)
[[nodiscard]] inline Vec3f fresnel_schlick_roughness(f32 VdotN, Vec3f F0, f32 roughness) noexcept {
    f32 x=1.f-VdotN, x5=x*x*x*x*x;
    Vec3f f_rough={max(1.f-roughness,F0.x), max(1.f-roughness,F0.y), max(1.f-roughness,F0.z)};
    return F0 + (f_rough-F0)*x5;
}

// Full dielectric Fresnel (exact, no approximation)
[[nodiscard]] inline f32 fresnel_dielectric(f32 cos_theta_i, f32 eta) noexcept {
    f32 sin2_t=eta*eta*(1.f-cos_theta_i*cos_theta_i);
    if(sin2_t>=1.f) return 1.f; // TIR
    f32 cos_t=std::sqrt(1.f-sin2_t);
    f32 rs=(cos_theta_i-eta*cos_t)/(cos_theta_i+eta*cos_t);
    f32 rp=(eta*cos_theta_i-cos_t)/(eta*cos_theta_i+cos_t);
    return 0.5f*(rs*rs+rp*rp);
}

// Conductor Fresnel (complex IOR)
[[nodiscard]] inline f32 fresnel_conductor(f32 cos_theta, f32 eta, f32 k) noexcept {
    f32 cos2=cos_theta*cos_theta, sin2=1.f-cos2;
    f32 eta2=eta*eta, k2=k*k;
    f32 a2pb2=std::sqrt((eta2-k2-sin2)*(eta2-k2-sin2)+4.f*eta2*k2);
    f32 a=std::sqrt(0.5f*(a2pb2+(eta2-k2-sin2)));
    f32 Rs=(a2pb2-2.f*a*cos_theta+cos2)/(a2pb2+2.f*a*cos_theta+cos2);
    f32 Rp=Rs*(cos2*a2pb2-2.f*a*cos_theta*sin2+sin2*sin2)
              /(cos2*a2pb2+2.f*a*cos_theta*sin2+sin2*sin2);
    return 0.5f*(Rs+Rp);
}

// ─── Cook-Torrance BRDF ───────────────────────────────────────────────────
struct PBRParams {
    Vec3f base_color    = {1,1,1};
    f32   metallic      = 0.f;
    f32   roughness     = 0.5f;
    f32   ao            = 1.f;
    Vec3f emissive      = {0,0,0};
};

struct ShadingContext {
    Vec3f N,V,L,H;       // unit vectors
    f32   NdotV,NdotL,NdotH,VdotH;
};

[[nodiscard]] inline ShadingContext make_shading_ctx(Vec3f N, Vec3f V, Vec3f L) noexcept {
    Vec3f H=(V+L).normalized();
    return {N,V,L,H,
        max(0.f,dot(N,V)),max(0.f,dot(N,L)),
        max(0.f,dot(N,H)),max(0.f,dot(V,H))};
}

[[nodiscard]] inline Vec3f cook_torrance_brdf(
    const ShadingContext& ctx, const PBRParams& mat) noexcept
{
    if(ctx.NdotL<1e-5f||ctx.NdotV<1e-5f) return {0,0,0};
    // Remap roughness
    f32 alpha=mat.roughness*mat.roughness;
    // F0: dielectric 0.04, metal = base_color
    Vec3f F0=lerp(Vec3f{0.04f,0.04f,0.04f}, mat.base_color, mat.metallic);
    Vec3f F=fresnel_schlick(ctx.VdotH, F0);
    f32   D=ggx_ndf(ctx.NdotH, alpha);
    f32   G=smith_g2_ggx(ctx.NdotV, ctx.NdotL, alpha);
    // Specular
    Vec3f spec=F*(D*G); // already divided by 4*NdotV*NdotL in smith_g2
    // Diffuse (Lambertian, energy-conserving)
    Vec3f kd=(Vec3f{1,1,1}-F)*(1.f-mat.metallic);
    Vec3f diff=kd*mat.base_color*f32(INV_PI);
    return (diff+spec)*ctx.NdotL;
}

// ─── Energy compensation (Kulla-Conty) ────────────────────────────────────
// Precomputed E_avg table lookup (simplified)
[[nodiscard]] inline f32 ggx_e_avg(f32 alpha) noexcept {
    // Polynomial fit to precomputed integral
    f32 a=alpha;
    return 1.f-0.28f*a-0.29f*a*a; // approximation
}

[[nodiscard]] inline Vec3f kulla_conty_compensation(
    Vec3f F_avg, f32 E_o, f32 E_i, f32 E_avg) noexcept
{
    f32 denom=1.f-F_avg.x*(1.f-E_avg);
    Vec3f f_ms=F_avg*F_avg*E_avg/Vec3f{denom,denom,denom};
    f32   c=(1.f-E_o)*(1.f-E_i)/(f32(PI)*(1.f-E_avg)+1e-6f);
    return f_ms*c;
}

// ─── GGX importance sampling ──────────────────────────────────────────────
// Sample half vector H from GGX distribution given uniform (u1,u2)
[[nodiscard]] inline Vec3f ggx_sample_h(f32 u1, f32 u2, f32 alpha) noexcept {
    f32 phi=f32(TWO_PI)*u1;
    f32 cos_theta=std::sqrt((1.f-u2)/(1.f+(alpha*alpha-1.f)*u2));
    f32 sin_theta=std::sqrt(max(0.f,1.f-cos_theta*cos_theta));
    return {std::cos(phi)*sin_theta, std::sin(phi)*sin_theta, cos_theta};
}

[[nodiscard]] inline f32 ggx_pdf(f32 NdotH, f32 VdotH, f32 alpha) noexcept {
    return ggx_ndf(NdotH,alpha)*NdotH/(4.f*VdotH+1e-6f);
}

// VNDF (Visible Normal Distribution Function) sampling — Heitz 2018
[[nodiscard]] inline Vec3f ggx_vndf_sample(Vec3f Ve, f32 alpha, f32 u1, f32 u2) noexcept {
    // Stretch view
    Vec3f Vh={alpha*Ve.x, alpha*Ve.y, Ve.z};
    f32 len=std::sqrt(Vh.x*Vh.x+Vh.z*Vh.z);
    Vec3f T1 = len>1e-6f ? Vec3f{-Vh.z,0,Vh.x}*(1.f/len) : Vec3f{0,0,1};
    Vec3f T2={T1.y*Vh.z-T1.z*Vh.y, T1.z*Vh.x-T1.x*Vh.z, T1.x*Vh.y-T1.y*Vh.x};
    Vh=Vh*(1.f/Vh.length());
    // Sample projected disk
    f32 r=std::sqrt(u1), phi=f32(TWO_PI)*u2;
    f32 t1=r*std::cos(phi), t2=r*std::sin(phi);
    f32 s=0.5f*(1.f+Vh.z); t2=lerp(std::sqrt(1.f-t1*t1), t2, s);
    Vec3f Nh=T1*t1+T2*t2+Vh*std::sqrt(max(0.f,1.f-t1*t1-t2*t2));
    // Unstretch
    return Vec3f{alpha*Nh.x, alpha*Nh.y, max(0.f,Nh.z)}.normalized();
}

// ─── Sheen BRDF (fabric/cloth) ────────────────────────────────────────────
[[nodiscard]] inline f32 sheen_ndf(f32 NdotH, f32 alpha) noexcept {
    // Ashikhmin velvet NDF
    f32 inv_a2=1.f/(alpha*alpha);
    f32 cos2=NdotH*NdotH, sin2=1.f-cos2;
    return (2.f+inv_a2)*std::pow(sin2,0.5f*inv_a2)/f32(TWO_PI);
}

[[nodiscard]] inline f32 sheen_l(f32 x, f32 alpha) noexcept {
    float a=alpha*alpha;
    float b=x/(1.f-x);
    return (x<0.5f)?
        std::exp(std::log(a)*b*b/a):
        std::exp(-std::log(a)/b/b*a);
}

[[nodiscard]] inline Vec3f sheen_brdf(Vec3f sheen_color, f32 NdotH, f32 NdotV, f32 NdotL, f32 roughness) noexcept {
    f32 D=sheen_ndf(NdotH,roughness);
    f32 G=1.f/(4.f*(NdotL+NdotV-NdotL*NdotV));
    return sheen_color*(D*G);
}

// ─── Iridescence (thin-film interference) ─────────────────────────────────
[[nodiscard]] inline Vec3f iridescence_fresnel(f32 cos_theta, f32 eta1, f32 eta2, f32 eta3,
                                                f32 thickness_nm) noexcept
{
    // Single-layer thin film (400nm-700nm range sampled)
    f32 cos2=std::sqrt(max(0.f,1.f-(eta1/eta2)*(eta1/eta2)*(1.f-cos_theta*cos_theta)));
    f32 OPD=2.f*eta2*thickness_nm*cos2;
    // Sample at R,G,B wavelengths
    Vec3f rgb={0,0,0};
    f32 lams[3]={630.f,532.f,467.f};
    for(int i=0;i<3;++i){
        f32 phi=f32(TWO_PI)*OPD/lams[i];
        // 2-interface Fabry-Perot reflection
        f32 r12=fresnel_dielectric(cos_theta,eta2/eta1);
        f32 r23=fresnel_dielectric(cos2,eta3/eta2);
        f32 T12=1.f-r12;
        f32 denom=1.f+r12*r23*r12*r23-2.f*r12*r23*std::cos(phi);
        f32 R=r12+(T12*T12*r23)/(denom);
        (&rgb.x)[i]=clamp01(R);
    }
    return rgb;
}

// ─── Subsurface scattering (diffuse approx) ───────────────────────────────
// Christensen-Burley fit to dipole SSS
[[nodiscard]] inline f32 sss_profile(f32 r, f32 d) noexcept {
    return (std::exp(-r/d)+std::exp(-r/(3.f*d)))/(8.f*f32(PI)*d*r+1e-6f);
}

// ─── Transmission / refraction ────────────────────────────────────────────
[[nodiscard]] inline Vec3f transmit_brdf(Vec3f base_color, f32 NdotL, f32 ior) noexcept {
    f32 F=fresnel_dielectric(std::abs(NdotL),ior);
    return base_color*(1.f-F)*std::abs(NdotL);
}

// ─── Clearcoat layer ──────────────────────────────────────────────────────
[[nodiscard]] inline Vec3f clearcoat_brdf(const ShadingContext& ctx, f32 roughness) noexcept {
    // F0=0.04 (IOR=1.5 glass), separate GGX lobe
    f32 alpha=max(0.001f,roughness*roughness);
    f32 D=ggx_ndf(ctx.NdotH,alpha);
    f32 G=smith_g2_ggx(ctx.NdotV,ctx.NdotL,alpha);
    Vec3f F=fresnel_schlick(ctx.VdotH,Vec3f{0.04f,0.04f,0.04f});
    return F*(D*G)*ctx.NdotL;
}

// ─── Environment IBL split-sum ────────────────────────────────────────────
// Evaluate split-sum approximation given precomputed LUT
// f_brdf(NdotV,roughness) -> Vec2f{A,B} such that specular = F0*A + B
[[nodiscard]] inline Vec2f brdf_integration_lut(f32 NdotV, f32 roughness) noexcept {
    // Polynomial fit (Brian Karis 2014)
    Vec4f c0={-1,-0.0275f,-0.572f,0.022f};
    Vec4f c1={1,0.0425f,1.04f,-0.04f};
    Vec4f r=c0*(NdotV*(-1.04f)+roughness)+c1;
    return {r.x*min(r.y,std::exp2(-9.28f*NdotV))+r.z, r.w};
}

// ─── Microfacet model utility ─────────────────────────────────────────────
// Roughness remapping: perceptual -> linear
[[nodiscard]] constexpr f32 roughness_to_alpha(f32 roughness) noexcept {
    return roughness*roughness;
}
[[nodiscard]] constexpr f32 alpha_to_roughness(f32 alpha) noexcept {
    return std::sqrt(alpha);
}
// IOR to F0
[[nodiscard]] constexpr f32 ior_to_f0(f32 ior) noexcept {
    f32 t=(ior-1.f)/(ior+1.f); return t*t;
}
// F0 to IOR
[[nodiscard]] inline f32 f0_to_ior(f32 f0) noexcept {
    f32 sq=std::sqrt(f0); return (1.f+sq)/(1.f-sq);
}

} // namespace qs::math::render
