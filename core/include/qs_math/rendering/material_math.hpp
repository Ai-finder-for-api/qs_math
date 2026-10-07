#pragma once
// material_math.hpp — PBR material math: BRDF, Fresnel, GGX, Cook-Torrance,
//                     subsurface scattering helpers, sheen, clearcoat,
//                     iridescence, anisotropy, material layering
#include "vec3.hpp"
#include "vec4.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "constants.hpp"

namespace qs::math::material {

// ---------------------------------------------------------------------------
// PBR Material parameters
// ---------------------------------------------------------------------------
struct PBRMaterial {
    Vec3f albedo{0.8f,0.8f,0.8f};
    f32   metallic{0.f};
    f32   roughness{0.5f};
    f32   ao{1.f};
    Vec3f emission{0.f};
    f32   alpha{1.f};
    f32   ior{1.5f};          // index of refraction
    f32   subsurface{0.f};
    f32   sheen{0.f};
    f32   sheen_tint{0.5f};
    f32   clearcoat{0.f};
    f32   clearcoat_gloss{0.5f};
    f32   anisotropy{0.f};
    f32   anisotropy_rotation{0.f};
    f32   iridescence{0.f};
    f32   transmission{0.f};
};

// ---------------------------------------------------------------------------
// Fresnel
// ---------------------------------------------------------------------------
// Schlick approximation
[[nodiscard]] constexpr Vec3f fresnel_schlick(f32 cos_theta, Vec3f F0) noexcept {
    f32 p=pow(clamp(1.f-cos_theta,0.f,1.f),5.f);
    return F0+(Vec3f{1.f}-F0)*p;
}

// Schlick with roughness (IBL)
[[nodiscard]] constexpr Vec3f fresnel_schlick_roughness(f32 cos_theta, Vec3f F0, f32 roughness) noexcept {
    f32 p=pow(clamp(1.f-cos_theta,0.f,1.f),5.f);
    return F0+(max(Vec3f{1.f-roughness},F0)-F0)*p;
}

// Full Fresnel (Sellmeier / exact dielectric)
[[nodiscard]] inline f32 fresnel_dielectric(f32 cos_i, f32 eta_i, f32 eta_t) noexcept {
    f32 sin_t=eta_i/eta_t*sqrt(max(0.f,1.f-cos_i*cos_i));
    if(sin_t>=1.f) return 1.f; // total internal reflection
    f32 cos_t=sqrt(max(0.f,1.f-sin_t*sin_t));
    f32 rs=(eta_i*cos_i-eta_t*cos_t)/(eta_i*cos_i+eta_t*cos_t);
    f32 rp=(eta_t*cos_i-eta_i*cos_t)/(eta_t*cos_i+eta_i*cos_t);
    return (rs*rs+rp*rp)*0.5f;
}

// F0 from IOR
[[nodiscard]] constexpr f32 ior_to_f0(f32 ior) noexcept {
    f32 t=(ior-1.f)/(ior+1.f);
    return t*t;
}
[[nodiscard]] constexpr Vec3f ior_to_f0_rgb(f32 ior) noexcept {
    f32 f=ior_to_f0(ior);
    return {f,f,f};
}
[[nodiscard]] constexpr Vec3f compute_f0(Vec3f albedo, f32 metallic, f32 ior=1.5f) noexcept {
    Vec3f dielectric{ior_to_f0(ior)};
    return lerp(dielectric,albedo,metallic);
}

// ---------------------------------------------------------------------------
// Normal Distribution Functions (NDF)
// ---------------------------------------------------------------------------
// GGX / Trowbridge-Reitz
[[nodiscard]] constexpr f32 ndf_ggx(f32 n_dot_h, f32 roughness) noexcept {
    f32 a=roughness*roughness, a2=a*a;
    f32 d=n_dot_h*n_dot_h*(a2-1.f)+1.f;
    return a2/(PI*d*d+EPS_F32);
}

// Anisotropic GGX
[[nodiscard]] constexpr f32 ndf_ggx_aniso(
    f32 n_dot_h, f32 h_dot_t, f32 h_dot_b,
    f32 alpha_t, f32 alpha_b) noexcept
{
    f32 a=h_dot_t/alpha_t, b=h_dot_b/alpha_b;
    f32 d=a*a+b*b+n_dot_h*n_dot_h;
    return 1.f/(PI*alpha_t*alpha_b*d*d+EPS_F32);
}

// Beckmann
[[nodiscard]] inline f32 ndf_beckmann(f32 n_dot_h, f32 roughness) noexcept {
    f32 a2=roughness*roughness;
    f32 cos2=n_dot_h*n_dot_h;
    return exp((cos2-1.f)/(a2*cos2))/(PI*a2*cos2*cos2+EPS_F32);
}

// GTR (Generalized Trowbridge-Reitz) — Disney
[[nodiscard]] constexpr f32 ndf_gtr(f32 n_dot_h, f32 roughness, f32 gamma=2.f) noexcept {
    f32 a2=roughness*roughness;
    f32 c=1.f/(PI*(a2-1.f))*(1.f-pow(a2,1.f-gamma/2.f));
    return c/pow(1.f+(a2-1.f)*n_dot_h*n_dot_h,gamma);
}

// ---------------------------------------------------------------------------
// Geometry / Masking-Shadowing Functions
// ---------------------------------------------------------------------------
// Smith-GGX geometry term
[[nodiscard]] constexpr f32 g_schlick_ggx(f32 n_dot_v, f32 roughness) noexcept {
    f32 r=(roughness+1.f), k=r*r/8.f;
    return n_dot_v/(n_dot_v*(1.f-k)+k+EPS_F32);
}

[[nodiscard]] constexpr f32 g_smith(f32 n_dot_v, f32 n_dot_l, f32 roughness) noexcept {
    return g_schlick_ggx(n_dot_v,roughness)*g_schlick_ggx(n_dot_l,roughness);
}

// Height-correlated Smith (more accurate)
[[nodiscard]] constexpr f32 g_smith_correlated(f32 n_dot_v, f32 n_dot_l, f32 roughness) noexcept {
    f32 a2=roughness*roughness;
    f32 lv=n_dot_l*sqrt(n_dot_v*n_dot_v*(1.f-a2)+a2+EPS_F32);
    f32 ll=n_dot_v*sqrt(n_dot_l*n_dot_l*(1.f-a2)+a2+EPS_F32);
    return 0.5f/(lv+ll+EPS_F32);
}

// ---------------------------------------------------------------------------
// Full Cook-Torrance BRDF
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f brdf_cook_torrance(
    Vec3f N, Vec3f V, Vec3f L, PBRMaterial mat) noexcept
{
    Vec3f H=(V+L).normalized();
    f32 n_dot_v=max(dot(N,V),EPS_F32);
    f32 n_dot_l=max(dot(N,L),EPS_F32);
    f32 n_dot_h=max(dot(N,H),0.f);
    f32 v_dot_h=max(dot(V,H),0.f);

    Vec3f F0=compute_f0(mat.albedo,mat.metallic,mat.ior);
    Vec3f F=fresnel_schlick(v_dot_h,F0);
    f32   D=ndf_ggx(n_dot_h,mat.roughness);
    f32   G=g_smith_correlated(n_dot_v,n_dot_l,mat.roughness);

    Vec3f specular=F*D*G/(4.f*n_dot_v*n_dot_l+EPS_F32);
    Vec3f kd=(Vec3f{1.f}-F)*(1.f-mat.metallic);
    Vec3f diffuse=kd*mat.albedo*INV_PI;
    return (diffuse+specular)*n_dot_l;
}

// ---------------------------------------------------------------------------
// Sheen BRDF (fabric, velvet)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 sheen_d(f32 n_dot_h, f32 roughness) noexcept {
    f32 inv_a2=1.f/(roughness*roughness);
    return (2.f+inv_a2)*pow(1.f-n_dot_h*n_dot_h,inv_a2*0.5f)/(TWO_PI);
}

[[nodiscard]] inline Vec3f brdf_sheen(
    Vec3f N, Vec3f V, Vec3f L, Vec3f H,
    Vec3f albedo, f32 roughness, f32 tint) noexcept
{
    f32 n_dot_l=max(dot(N,L),0.f);
    f32 n_dot_h=max(dot(N,H),0.f);
    Vec3f sheen_color=lerp(Vec3f{1.f},albedo,tint);
    f32 d=sheen_d(n_dot_h,roughness);
    return sheen_color*d*n_dot_l;
}

// ---------------------------------------------------------------------------
// Subsurface scattering helpers
// ---------------------------------------------------------------------------
// Normalized diffusion profile (Burley 2015)
[[nodiscard]] inline f32 burley_diffusion(f32 r, f32 d) noexcept {
    return exp(-r/d)/(4.f*PI*d*r+EPS_F32)
         + exp(-r/(3.f*d))/(4.f*PI*(3.f*d)*r+EPS_F32);
}

// Dipole approximation for SSS
[[nodiscard]] inline f32 dipole_sss(f32 r, f32 sigma_tr) noexcept {
    return exp(-sigma_tr*r)/(4.f*PI*r+EPS_F32);
}

// Pre-integrated SSS lookup (Jimenez 2010) — curve value
[[nodiscard]] inline f32 preintegrated_sss(f32 n_dot_l, f32 curvature) noexcept {
    // Approximate using a 2D Gaussian
    f32 u=n_dot_l*0.5f+0.5f;
    f32 v=curvature;
    return lerp(smoothstep(-0.2f,0.2f,n_dot_l),
                clamp(n_dot_l*2.f,0.f,1.f),
                v);
}

// ---------------------------------------------------------------------------
// Clearcoat BRDF
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 brdf_clearcoat_lobe(
    Vec3f N, Vec3f V, Vec3f L) noexcept
{
    Vec3f H=(V+L).normalized();
    f32 n_dot_h=max(dot(N,H),0.f);
    f32 n_dot_v=max(dot(N,V),EPS_F32);
    f32 n_dot_l=max(dot(N,L),EPS_F32);
    f32 v_dot_h=max(dot(V,H),0.f);
    // Clearcoat uses fixed roughness=0.25 and IOR=1.5
    f32 D=ndf_ggx(n_dot_h,0.25f);
    f32 F=lerp(0.04f,1.f,pow(1.f-v_dot_h,5.f));
    f32 G=g_smith(n_dot_v,n_dot_l,0.25f);
    return D*F*G/(4.f*n_dot_v*n_dot_l+EPS_F32)*n_dot_l;
}

// ---------------------------------------------------------------------------
// Iridescence (thin-film interference)
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f iridescence_fresnel(f32 cos_theta, f32 thickness_nm, f32 ior=1.5f) noexcept {
    // Airy function approximation
    f32 phi=4.f*PI*ior*thickness_nm*cos_theta/550.f; // green channel
    Vec3f phi_rgb{4.f*PI*ior*thickness_nm*cos_theta/700.f,
                  4.f*PI*ior*thickness_nm*cos_theta/550.f,
                  4.f*PI*ior*thickness_nm*cos_theta/450.f};
    Vec3f F0{ior_to_f0(ior)};
    Vec3f F=fresnel_schlick(cos_theta,F0);
    // Modulate by interference pattern
    return Vec3f{F.x*(0.5f+0.5f*cos(phi_rgb.x)),
                 F.y*(0.5f+0.5f*cos(phi_rgb.y)),
                 F.z*(0.5f+0.5f*cos(phi_rgb.z))};
}

// ---------------------------------------------------------------------------
// Environment mapping / IBL
// ---------------------------------------------------------------------------
// Split-sum approximation look-up (analytical fit)
[[nodiscard]] constexpr Vec2f brdf_lut_approx(f32 n_dot_v, f32 roughness) noexcept {
    // Karis 2013 analytical approximation
    Vec4f c0{-1.f,-0.0275f,-0.572f, 0.022f};
    Vec4f c1{ 1.f, 0.0425f, 1.04f, -0.04f};
    Vec4f r=c0*roughness+c1;
    f32 a004=min(r.x*r.x,exp2(-9.28f*n_dot_v))*r.x+r.y;
    return {-1.04f*a004+0.986f, a004*r.z+r.w};
}

// ---------------------------------------------------------------------------
// Emissive bloom
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec3f compute_emissive(
    Vec3f emission_color, f32 intensity, f32 exposure) noexcept
{
    return emission_color*intensity*ev_to_exposure(exposure);
}
[[nodiscard]] constexpr f32 ev_to_exposure(f32 ev) noexcept { return exp2(-ev); }

// ---------------------------------------------------------------------------
// Material blending (layered materials)
// ---------------------------------------------------------------------------
[[nodiscard]] inline PBRMaterial blend_materials(
    PBRMaterial a, PBRMaterial b, f32 t) noexcept
{
    PBRMaterial r;
    r.albedo=lerp(a.albedo,b.albedo,t);
    r.metallic=lerp(a.metallic,b.metallic,t);
    r.roughness=lerp(a.roughness,b.roughness,t);
    r.ao=lerp(a.ao,b.ao,t);
    r.emission=lerp(a.emission,b.emission,t);
    r.ior=lerp(a.ior,b.ior,t);
    r.subsurface=lerp(a.subsurface,b.subsurface,t);
    r.clearcoat=lerp(a.clearcoat,b.clearcoat,t);
    r.sheen=lerp(a.sheen,b.sheen,t);
    return r;
}

} // namespace qs::math::material
