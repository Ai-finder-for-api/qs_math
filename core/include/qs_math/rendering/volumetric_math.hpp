#pragma once
// qs_math/volumetric_math.hpp — Ray marching, phase functions, transmittance (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <cmath>
#include <numbers>
#include <vector>
#include <functional>

namespace qs {

// ─── Phase functions ──────────────────────────────────────────────────────────

// Isotropic phase function
[[nodiscard]] inline f32 phase_isotropic() { return 1.f/(4*f32(std::numbers::pi)); }

// Henyey-Greenstein phase function p(cos θ) for cos_theta = dot(ω_in, ω_out)
[[nodiscard]] inline f32 phase_hg(f32 cos_theta, f32 g) {
    f32 g2=g*g;
    f32 denom=1+g2-2*g*cos_theta;
    return (1-g2)/(4*f32(std::numbers::pi)*denom*std::sqrt(denom));
}

// HG importance sample: given uniform u ∈ [0,1), sample cos_theta
[[nodiscard]] inline f32 hg_sample_cos_theta(f32 u, f32 g) {
    if(std::abs(g)<1e-4f) return 1-2*u; // isotropic
    f32 s=(1-g*g)/(1-g+2*g*u);
    return (1+g*g-s*s)/(2*g);
}

// Schlick's approximation of HG (cheaper)
[[nodiscard]] inline f32 phase_schlick(f32 cos_theta, f32 k) {
    f32 d=1-k*cos_theta;
    return (1-k*k)/(4*f32(std::numbers::pi)*d*d);
}

// Mie phase function (Cornette-Shanks approximation)
[[nodiscard]] inline f32 phase_mie_cs(f32 cos_theta, f32 g) {
    f32 g2=g*g;
    return 3*(1-g2)*(1+cos_theta*cos_theta)/(2*(2+g2)*std::pow(1+g2-2*g*cos_theta,1.5f));
}

// Rayleigh phase function (small particles, symmetric)
[[nodiscard]] inline f32 phase_rayleigh(f32 cos_theta) {
    return 3.f/(16*f32(std::numbers::pi))*(1+cos_theta*cos_theta);
}

// ─── Beer-Lambert transmittance ───────────────────────────────────────────────
// T = exp(-σ_t * d)  where σ_t = σ_a + σ_s (extinction coefficient)
[[nodiscard]] inline f32 transmittance_beer(f32 sigma_t, f32 dist) {
    return std::exp(-sigma_t*dist);
}
[[nodiscard]] inline Vec3f transmittance_beer_rgb(Vec3f sigma_t, f32 dist) {
    return {std::exp(-sigma_t.x*dist),std::exp(-sigma_t.y*dist),std::exp(-sigma_t.z*dist)};
}

// Free-flight distance sampling: d = -log(1-u)/σ_t
[[nodiscard]] inline f32 sample_free_flight(f32 u, f32 sigma_t) {
    return -std::log(std::max(1e-10f,1-u))/sigma_t;
}

// ─── Woodcock tracking (null-scattering / delta tracking) ─────────────────────
// Handles heterogeneous media by treating max density as reference
[[nodiscard]] inline std::optional<f32> delta_tracking(
    const std::function<f32(Vec3f)>& sigma_t_fn,
    Vec3f ray_origin, Vec3f ray_dir,
    f32 t_min, f32 t_max,
    f32 sigma_t_max, PCG32& rng)
{
    f32 t=t_min;
    while(true) {
        f32 u=rng.next_float();
        t+=sample_free_flight(u,sigma_t_max);
        if(t>=t_max) return std::nullopt;
        Vec3f p=ray_origin+ray_dir*t;
        f32 sigma_t=sigma_t_fn(p);
        // Accept/reject: probability sigma_t/sigma_t_max
        if(rng.next_float()<sigma_t/sigma_t_max) return t;
    }
}

// Ratio tracking for unbiased transmittance estimation
[[nodiscard]] inline f32 ratio_tracking(
    const std::function<f32(Vec3f)>& sigma_t_fn,
    Vec3f ray_origin, Vec3f ray_dir,
    f32 t_min, f32 t_max,
    f32 sigma_t_max, PCG32& rng)
{
    f32 T=1, t=t_min;
    while(true) {
        t+=sample_free_flight(rng.next_float(),sigma_t_max);
        if(t>=t_max) break;
        Vec3f p=ray_origin+ray_dir*t;
        T*=1-sigma_t_fn(p)/sigma_t_max;
    }
    return T;
}

// ─── Ray marching ─────────────────────────────────────────────────────────────
struct VolumeHit {
    Vec3f transmittance{1,1,1};
    Vec3f in_scattering{};
    f32   optical_depth{0};
};

// Fixed-step ray marching through a volume
[[nodiscard]] inline VolumeHit ray_march(
    const std::function<f32(Vec3f)>& density_fn,
    const std::function<Vec3f(Vec3f)>& emission_fn,
    Vec3f ro, Vec3f rd,
    f32 t_near, f32 t_far,
    u32 steps, f32 sigma_a=0.1f, f32 sigma_s=0.5f)
{
    f32 step_size=(t_far-t_near)/steps;
    VolumeHit hit;
    Vec3f T{1,1,1};
    for(u32 i=0;i<steps;++i) {
        f32 t=t_near+(i+0.5f)*step_size;
        Vec3f p=ro+rd*t;
        f32 rho=density_fn(p);
        f32 sigma_t=(sigma_a+sigma_s)*rho;
        hit.optical_depth+=sigma_t*step_size;
        Vec3f dT{std::exp(-sigma_t*step_size),std::exp(-sigma_t*step_size),std::exp(-sigma_t*step_size)};
        // In-scattering contribution
        Vec3f emission=emission_fn(p);
        hit.in_scattering=hit.in_scattering+T*emission*sigma_s*rho*step_size;
        T=T*dT;
    }
    hit.transmittance=T;
    return hit;
}

// Adaptive ray marching (smaller steps where density is higher)
[[nodiscard]] inline VolumeHit adaptive_ray_march(
    const std::function<f32(Vec3f)>& density_fn,
    const std::function<Vec3f(Vec3f)>& emission_fn,
    Vec3f ro, Vec3f rd,
    f32 t_near, f32 t_far,
    f32 step_min=0.001f, f32 step_max=0.1f,
    f32 sigma_a=0.1f, f32 sigma_s=0.5f)
{
    VolumeHit hit;
    Vec3f T{1,1,1};
    f32 t=t_near;
    while(t<t_far) {
        Vec3f p=ro+rd*t;
        f32 rho=density_fn(p);
        f32 step=std::clamp(step_max/(rho+0.1f),step_min,step_max);
        step=std::min(step,t_far-t);
        f32 sigma_t=(sigma_a+sigma_s)*rho;
        Vec3f dT{std::exp(-sigma_t*step),std::exp(-sigma_t*step),std::exp(-sigma_t*step)};
        hit.in_scattering=hit.in_scattering+T*emission_fn(p)*sigma_s*rho*step;
        T=T*dT;
        hit.optical_depth+=sigma_t*step;
        t+=step;
    }
    hit.transmittance=T;
    return hit;
}

// ─── Atmosphere model (Rayleigh + Mie scattering) ─────────────────────────────
struct AtmosphereModel {
    f32 R_earth{6371e3f};
    f32 R_atmo{6471e3f};
    Vec3f beta_R{5.8e-6f,13.5e-6f,33.1e-6f}; // Rayleigh scattering coeffs
    f32  beta_M{21e-6f};   // Mie scattering
    f32  H_R{8000.f};      // Rayleigh scale height
    f32  H_M{1200.f};      // Mie scale height
    f32  g_M{0.758f};      // Mie asymmetry

    [[nodiscard]] f32 rayleigh_density(f32 h) const { return std::exp(-h/H_R); }
    [[nodiscard]] f32 mie_density(f32 h) const { return std::exp(-h/H_M); }

    // Optical depth along a ray from point to top of atmosphere
    [[nodiscard]] std::pair<Vec3f,f32> optical_depth(Vec3f pos, Vec3f dir, u32 steps=25) const {
        // Find intersection with top of atmosphere
        f32 tmax=0;
        Vec3f center{};
        // Quadratic: find t where |pos+t*dir| = R_atmo
        f32 b=dot(pos,dir), c=dot(pos,pos)-R_atmo*R_atmo;
        f32 disc=b*b-c;
        if(disc<0) return {{},0};
        tmax=-b+std::sqrt(disc);
        f32 step=tmax/steps;
        Vec3f odR{};
        f32 odM=0;
        for(u32 i=0;i<steps;++i) {
            Vec3f p=pos+dir*(i+0.5f)*step;
            f32 h=length(p)-R_earth;
            h=std::max(0.f,h);
            odR=odR+beta_R*(rayleigh_density(h)*step);
            odM+=beta_M*mie_density(h)*step;
        }
        return {odR,odM};
    }

    // Sky colour for a given sun direction and view direction
    [[nodiscard]] Vec3f sky_colour(Vec3f view_dir, Vec3f sun_dir, u32 steps=16) const {
        Vec3f pos{0,R_earth+100.f,0};
        // Find atmosphere exit
        f32 b=dot(pos,view_dir), c=dot(pos,pos)-R_atmo*R_atmo;
        f32 disc=b*b-c; if(disc<0) return {};
        f32 tmax=-b+std::sqrt(disc);
        f32 step=tmax/steps;
        Vec3f colour{};
        Vec3f T{1,1,1}; // transmittance so far
        f32 cos_theta=dot(view_dir,sun_dir);
        f32 phase_R=phase_rayleigh(cos_theta);
        f32 phase_M_=phase_hg(cos_theta,g_M);
        for(u32 i=0;i<steps;++i) {
            Vec3f p=pos+view_dir*(i+0.5f)*step;
            f32 h=std::max(0.f,length(p)-R_earth);
            f32 rhoR=rayleigh_density(h), rhoM=mie_density(h);
            Vec3f sigma_t_R=beta_R*rhoR; f32 sigma_t_M=beta_M*rhoM;
            // Sun optical depth from p
            auto [od_sun_R,od_sun_M]=optical_depth(p,sun_dir,8);
            Vec3f T_sun{std::exp(-(od_sun_R.x+sigma_t_M*od_sun_M)),
                        std::exp(-(od_sun_R.y+sigma_t_M*od_sun_M)),
                        std::exp(-(od_sun_R.z+sigma_t_M*od_sun_M))};
            colour=colour+T*T_sun*(sigma_t_R*phase_R+Vec3f{1,1,1}*(sigma_t_M*phase_M_))*step;
            Vec3f dT{std::exp(-(sigma_t_R.x+sigma_t_M)*step),
                     std::exp(-(sigma_t_R.y+sigma_t_M)*step),
                     std::exp(-(sigma_t_R.z+sigma_t_M)*step)};
            T=T*dT;
        }
        return colour;
    }
};

// ─── Subsurface scattering (diffusion approximation) ─────────────────────────

// Diffusion profile Rd(r) for dipole approximation (Jensen 2001)
[[nodiscard]] inline f32 sss_dipole(f32 r, f32 sigma_a, f32 sigma_s_prime) {
    f32 sigma_tr=std::sqrt(3*sigma_a*(sigma_a+sigma_s_prime));
    f32 D=1.f/(3*(sigma_a+sigma_s_prime));
    f32 zr=1.f/(sigma_a+sigma_s_prime);
    f32 zv=zr*(1+4.f/3*0.440f); // approx A
    f32 dr=std::sqrt(r*r+zr*zr), dv=std::sqrt(r*r+zv*zv);
    f32 Rd=zr*(1+sigma_tr*dr)*std::exp(-sigma_tr*dr)/(dr*dr*dr);
    Rd+=zv*(1+sigma_tr*dv)*std::exp(-sigma_tr*dv)/(dv*dv*dv);
    return Rd/(4*f32(std::numbers::pi));
}

// Normalized diffusion (Burley 2015) — used in production renderers
[[nodiscard]] inline f32 sss_normalized_diffusion(f32 r, f32 d) {
    return (std::exp(-r/d)+std::exp(-r/(3*d)))/(8*f32(std::numbers::pi)*d*r);
}

// Effective albedo → diffusion parameter s (Burley fit)
[[nodiscard]] inline f32 sss_burley_s(f32 albedo) {
    return 1.9f-albedo+3.5f*(albedo-0.8f)*(albedo-0.8f);
}

} // namespace qs
