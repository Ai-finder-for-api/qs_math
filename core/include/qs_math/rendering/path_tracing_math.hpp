#pragma once
// qs_math/path_tracing_math.hpp — BSDF sampling, MIS, NEE, volume transmittance (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat3.hpp"
#include "random.hpp"
#include "ray_tracing_math.hpp"
#include <cmath>
#include <algorithm>
#include <optional>

namespace qs {

// ─── Sampling utilities ───────────────────────────────────────────────────────

[[nodiscard]] inline Vec3f sample_cosine_hemisphere(f32 u1, f32 u2) {
    f32 r=std::sqrt(u1);
    f32 phi=k_tau*u2;
    return {r*std::cos(phi), std::sqrt(std::max(0.f,1.f-u1)), r*std::sin(phi)};
}

[[nodiscard]] inline f32 pdf_cosine_hemisphere(f32 cos_theta) {
    return std::max(0.f,cos_theta)/k_pi;
}

[[nodiscard]] inline Vec3f sample_uniform_hemisphere(f32 u1, f32 u2) {
    f32 sinTheta=std::sqrt(std::max(0.f,1.f-u1*u1));
    f32 phi=k_tau*u2;
    return {sinTheta*std::cos(phi), u1, sinTheta*std::sin(phi)};
}

[[nodiscard]] inline f32 pdf_uniform_hemisphere() { return 1.f/(2.f*k_pi); }

[[nodiscard]] inline Vec3f sample_uniform_sphere(f32 u1, f32 u2) {
    f32 z=1.f-2.f*u1;
    f32 r=std::sqrt(std::max(0.f,1.f-z*z));
    f32 phi=k_tau*u2;
    return {r*std::cos(phi),z,r*std::sin(phi)};
}

// Power heuristic for MIS
[[nodiscard]] inline f32 mis_power_heuristic(f32 pdf_a, f32 pdf_b, u32 beta=2) {
    f32 a=std::pow(pdf_a,f32(beta));
    f32 b=std::pow(pdf_b,f32(beta));
    return a/(a+b+1e-10f);
}

// Balance heuristic
[[nodiscard]] inline f32 mis_balance_heuristic(f32 pdf_a, f32 pdf_b) {
    return pdf_a/(pdf_a+pdf_b+1e-10f);
}

// ─── TBN frame for BSDF evaluation ───────────────────────────────────────────

struct TBNFrame {
    Vec3f T{1,0,0}, B{0,0,1}, N{0,1,0};

    [[nodiscard]] static TBNFrame from_normal(const Vec3f& n) {
        TBNFrame f; f.N=n;
        Vec3f up=(std::abs(n.y)<0.9f)?Vec3f{0,1,0}:Vec3f{1,0,0};
        f.T=normalize(cross(up,n));
        f.B=cross(n,f.T);
        return f;
    }

    [[nodiscard]] Vec3f to_world(const Vec3f& v) const {
        return T*v.x+N*v.y+B*v.z;
    }

    [[nodiscard]] Vec3f to_local(const Vec3f& v) const {
        return {dot(v,T),dot(v,N),dot(v,B)};
    }
};

// ─── BSDF evaluation ─────────────────────────────────────────────────────────

// Lambertian diffuse
[[nodiscard]] inline Vec3f bsdf_lambertian(const Vec3f& albedo) {
    return albedo/k_pi;
}

[[nodiscard]] inline Vec3f bsdf_lambertian_sample(f32 u1, f32 u2) {
    return sample_cosine_hemisphere(u1,u2);
}

// GGX microfacet distribution
[[nodiscard]] inline f32 ggx_D(const Vec3f& wh, f32 alpha) {
    f32 cos_theta=wh.y;
    if(cos_theta<=0.f) return 0.f;
    f32 a2=alpha*alpha;
    f32 cos2=cos_theta*cos_theta;
    f32 tan2=(1.f-cos2)/cos2;
    f32 denom=k_pi*cos2*cos2*(a2+tan2)*(a2+tan2);
    return a2/denom;
}

// Smith shadowing-masking
[[nodiscard]] inline f32 ggx_G1(const Vec3f& w, f32 alpha) {
    f32 cos_theta=w.y;
    if(cos_theta<=0.f) return 0.f;
    f32 tan2=(1.f-cos_theta*cos_theta)/(cos_theta*cos_theta);
    return 2.f/(1.f+std::sqrt(1.f+alpha*alpha*tan2));
}

[[nodiscard]] inline f32 ggx_G(const Vec3f& wi, const Vec3f& wo, f32 alpha) {
    return ggx_G1(wi,alpha)*ggx_G1(wo,alpha);
}

// GGX sample half-vector in local space
[[nodiscard]] inline Vec3f ggx_sample_wh(f32 alpha, f32 u1, f32 u2) {
    f32 phi=k_tau*u1;
    f32 tan2=alpha*alpha*u2/(1.f-u2);
    f32 cos_theta=1.f/std::sqrt(1.f+tan2);
    f32 sin_theta=std::sqrt(std::max(0.f,1.f-cos_theta*cos_theta));
    return {sin_theta*std::cos(phi), cos_theta, sin_theta*std::sin(phi)};
}

[[nodiscard]] inline f32 ggx_pdf_wh(const Vec3f& wh, f32 alpha) {
    return ggx_D(wh,alpha)*std::abs(wh.y);
}

// Schlick Fresnel
[[nodiscard]] inline f32 schlick_fresnel(f32 cos_theta, f32 F0) {
    f32 t=1.f-std::abs(cos_theta);
    return F0+(1.f-F0)*t*t*t*t*t;
}

[[nodiscard]] inline Vec3f schlick_fresnel_v(f32 cos_theta, const Vec3f& F0) {
    f32 t=1.f-std::abs(cos_theta); f32 t5=t*t*t*t*t;
    return F0+(Vec3f{1,1,1}-F0)*t5;
}

// Full GGX BRDF evaluation: wo, wi in local space (y = normal)
struct GGXResult { Vec3f value{}; f32 pdf{}; };

[[nodiscard]] inline GGXResult ggx_eval(
    const Vec3f& wi, const Vec3f& wo, f32 alpha, const Vec3f& F0)
{
    if(wi.y<=0.f||wo.y<=0.f) return {};
    Vec3f wh=normalize(wi+wo);
    f32 D=ggx_D(wh,alpha);
    f32 G=ggx_G(wi,wo,alpha);
    Vec3f F=schlick_fresnel_v(dot(wi,wh),F0);
    Vec3f bsdf=F*(D*G/(4.f*wi.y*wo.y));
    f32 pdf=ggx_pdf_wh(wh,alpha)/(4.f*dot(wi,wh));
    return {bsdf,pdf};
}

[[nodiscard]] inline std::pair<Vec3f,f32> ggx_sample(
    const Vec3f& wo, f32 alpha, const Vec3f& F0, f32 u1, f32 u2)
{
    Vec3f wh=ggx_sample_wh(alpha,u1,u2);
    Vec3f wi=reflect(-wo,wh); // reflect wo around wh
    if(wi.y<=0.f) return {{},0.f};
    f32 pdf=ggx_pdf_wh(wh,alpha)/(4.f*std::abs(dot(wo,wh)));
    return {wi,pdf};
}

// ─── Volume rendering ─────────────────────────────────────────────────────────

struct HGPhase {
    f32 g{0.f}; // asymmetry (-1=back, 0=iso, 1=forward)

    [[nodiscard]] f32 eval(f32 cos_theta) const {
        f32 denom=1.f+g*g+2.f*g*cos_theta;
        return (1.f-g*g)/(4.f*k_pi*denom*std::sqrt(denom));
    }

    [[nodiscard]] Vec3f sample(f32 u1, f32 u2) const {
        f32 cos_theta;
        if(std::abs(g)<1e-4f){
            cos_theta=1.f-2.f*u1;
        } else {
            f32 s=(1.f-g*g)/(1.f-g+2.f*g*u1);
            cos_theta=(1.f+g*g-s*s)/(2.f*g);
        }
        f32 sin_theta=std::sqrt(std::max(0.f,1.f-cos_theta*cos_theta));
        f32 phi=k_tau*u2;
        return {sin_theta*std::cos(phi),cos_theta,sin_theta*std::sin(phi)};
    }

    [[nodiscard]] f32 pdf(f32 cos_theta) const { return eval(cos_theta); }
};

// Delta tracking for heterogeneous media
[[nodiscard]] inline f32 delta_track(
    const Vec3f& origin, const Vec3f& dir, f32 max_t,
    f32 sigma_maj, // majorant extinction coefficient
    auto density_fn,  // density_fn(Vec3f) -> f32 in [0,1]
    Pcg32& rng)
{
    f32 t=0.f;
    while(t<max_t){
        t-=std::log(std::max(1e-10f,rng.next_f32()))/sigma_maj;
        if(t>=max_t) break;
        f32 sigma_t=density_fn(origin+dir*t)*sigma_maj;
        if(rng.next_f32()<sigma_t/sigma_maj) return t;
    }
    return max_t;
}

// Beer-Lambert transmittance
[[nodiscard]] inline f32 transmittance(f32 sigma_t, f32 distance) {
    return std::exp(-sigma_t*distance);
}

// Ratio tracking estimator of transmittance
[[nodiscard]] inline f32 ratio_track_transmittance(
    const Vec3f& a, const Vec3f& b, f32 sigma_maj,
    auto density_fn, Pcg32& rng)
{
    Vec3f dir=b-a; f32 max_t=length(dir);
    dir=dir/max_t;
    f32 Tr=1.f, t=0.f;
    while(t<max_t){
        t-=std::log(rng.next_f32()+1e-10f)/sigma_maj;
        if(t>=max_t) break;
        f32 sigma_t=density_fn(a+dir*t)*sigma_maj;
        Tr*=1.f-sigma_t/sigma_maj;
    }
    return Tr;
}

// ─── Path throughput russian roulette ────────────────────────────────────────

[[nodiscard]] inline f32 russian_roulette_prob(const Vec3f& throughput, u32 bounce) {
    if(bounce<3) return 1.f;
    return std::min(1.f,std::max({throughput.x,throughput.y,throughput.z}));
}

// ─── Light sampling ───────────────────────────────────────────────────────────

struct AreaLight {
    Vec3f center{};
    Vec3f u_vec{1,0,0}; // half-extents
    Vec3f v_vec{0,0,1};
    Vec3f normal{0,1,0};
    Vec3f emission{10,10,10};

    [[nodiscard]] f32 area() const { return length(u_vec)*length(v_vec)*4.f; }

    [[nodiscard]] Vec3f sample_point(f32 u1, f32 u2) const {
        return center+u_vec*(u1*2.f-1.f)+v_vec*(u2*2.f-1.f);
    }

    [[nodiscard]] f32 pdf_solid_angle(const Vec3f& surface_point, const Vec3f& light_point) const {
        Vec3f dir=light_point-surface_point;
        f32 dist2=dot(dir,dir);
        f32 cos_l=std::abs(dot(normalize(dir),normal));
        return dist2/(cos_l*area()+1e-10f);
    }
};

} // namespace qs
