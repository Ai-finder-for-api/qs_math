#pragma once
// qs_math/volumetric_lighting.hpp — Fog, god rays, participating media, phase functions (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "ray_tracing_math.hpp"
#include "random.hpp"
#include <cmath>
#include <vector>
#include <span>
#include <array>
#include <algorithm>

namespace qs {

// ─── Participating media ──────────────────────────────────────────────────────

struct MediaParams {
    f32 sigma_a{0.01f};   // absorption coefficient (1/m)
    f32 sigma_s{0.1f};    // scattering coefficient (1/m)
    f32 g{0.0f};          // Henyey-Greenstein asymmetry
    Vec3f albedo{1,1,1};  // single-scattering albedo per channel

    [[nodiscard]] f32 sigma_t() const { return sigma_a+sigma_s; }
    [[nodiscard]] f32 ss_albedo() const { return sigma_s/(sigma_t()+1e-8f); }
};

// ─── Phase functions ──────────────────────────────────────────────────────────

// Henyey-Greenstein
[[nodiscard]] inline f32 hg_phase(f32 cos_theta, f32 g) {
    f32 g2=g*g;
    return (1.f-g2)/(4.f*k_pi*std::pow(1.f+g2-2.f*g*cos_theta,1.5f));
}

// Rayleigh scattering
[[nodiscard]] inline f32 rayleigh_phase(f32 cos_theta) {
    return 3.f/(16.f*k_pi)*(1.f+cos_theta*cos_theta);
}

// Mie (approximate, Cornette-Shanks)
[[nodiscard]] inline f32 mie_phase_cs(f32 cos_theta, f32 g) {
    f32 g2=g*g;
    f32 num=3.f*(1.f-g2)*(1.f+cos_theta*cos_theta);
    f32 den=8.f*k_pi*(2.f+g2)*std::pow(1.f+g2-2.f*g*cos_theta,1.5f);
    return num/den;
}

// Double HG (skin/material)
[[nodiscard]] inline f32 double_hg_phase(f32 cos_theta, f32 g1, f32 g2, f32 w=0.5f) {
    return w*hg_phase(cos_theta,g1)+(1.f-w)*hg_phase(cos_theta,g2);
}

// ─── Homogeneous fog ──────────────────────────────────────────────────────────

struct FogParams {
    Vec3f color{0.8f,0.85f,0.9f};
    f32   density{0.05f};          // sigma_t
    f32   start_dist{0.f};
    f32   end_dist{100.f};
    f32   height_falloff{0.f};     // 0=uniform, >0=exponential height fog
    Vec3f wind{0.1f,0,0.05f};      // for animated fog
    f32   anisotropy{0.2f};        // g for scattering

    [[nodiscard]] f32 density_at(const Vec3f& p) const {
        if(height_falloff<=0.f) return density;
        return density*std::exp(-height_falloff*std::max(0.f,p.y));
    }
};

// Exponential distance fog factor
[[nodiscard]] inline f32 fog_factor_exp(f32 dist, f32 density) {
    return std::exp(-density*dist);
}

[[nodiscard]] inline f32 fog_factor_exp2(f32 dist, f32 density) {
    f32 d=density*dist;
    return std::exp(-d*d);
}

[[nodiscard]] inline f32 fog_factor_linear(f32 dist, f32 start, f32 end) {
    return 1.f-std::clamp((dist-start)/(end-start+1e-8f),0.f,1.f);
}

// Apply fog to a color
[[nodiscard]] inline Vec3f apply_fog(
    const Vec3f& color, const Vec3f& fog_color, f32 factor)
{
    return color*factor+fog_color*(1.f-factor);
}

// ─── Height fog ───────────────────────────────────────────────────────────────

// Analytical integral of exp(-h*y) along a ray segment
[[nodiscard]] inline f32 height_fog_integral(
    const Vec3f& ray_o, const Vec3f& ray_d,
    f32 t_start, f32 t_end,
    f32 density, f32 falloff)
{
    if(std::abs(falloff)<1e-6f) return density*(t_end-t_start);
    f32 y0=ray_o.y+ray_d.y*t_start;
    f32 y1=ray_o.y+ray_d.y*t_end;
    if(std::abs(ray_d.y)<1e-6f)
        return density*std::exp(-falloff*y0)*(t_end-t_start);
    return density*(std::exp(-falloff*y0)-std::exp(-falloff*y1))/(falloff*std::abs(ray_d.y));
}

// ─── Volumetric shadow (shadow map integration) ───────────────────────────────

struct VolumetricShadowParams {
    u32  num_samples{64};
    f32  max_dist{50.f};
    f32  jitter{1.f};
    f32  scattering_coeff{0.1f};
};

// Ray march through fog toward light, accumulate in-scattering
[[nodiscard]] inline Vec3f volumetric_inscattering(
    const Vec3f& ray_origin, const Vec3f& ray_dir, f32 ray_dist,
    const Vec3f& light_pos, const Vec3f& light_color,
    const FogParams& fog, const VolumetricShadowParams& p,
    Pcg32& rng,
    auto shadow_fn) // shadow_fn(Vec3f world_pos) -> f32 [0,1]
{
    Vec3f acc{0,0,0};
    f32 step_size=std::min(p.max_dist,ray_dist)/f32(p.num_samples);
    f32 transmittance=1.f;

    for(u32 i=0;i<p.num_samples;++i){
        f32 t=(f32(i)+rng.next_f32()*p.jitter)*step_size;
        if(t>ray_dist) break;

        Vec3f pos=ray_origin+ray_dir*t;
        f32 d=fog.density_at(pos);
        f32 sigma_t=d;
        f32 step_transmittance=std::exp(-sigma_t*step_size);
        transmittance*=step_transmittance;

        // In-scattering from light
        Vec3f to_light=light_pos-pos;
        f32 light_dist=length(to_light);
        Vec3f light_dir=to_light/light_dist;
        f32 cos_theta=dot(ray_dir,light_dir);
        f32 phase=hg_phase(cos_theta,fog.anisotropy);
        f32 shadow=shadow_fn(pos);
        f32 light_atten=1.f/(light_dist*light_dist+1.f);

        Vec3f scatter=fog.color*d*p.scattering_coeff*phase*shadow*light_atten;
        acc=acc+(scatter*transmittance)*step_size;
        acc.x*=light_color.x; acc.y*=light_color.y; acc.z*=light_color.z;
    }
    return acc;
}

// ─── God rays (screen-space radial blur) ─────────────────────────────────────

struct GodRayParams {
    Vec2f light_pos_ss{0.5f,0.5f}; // light position in [0,1] screen space
    f32   density{0.5f};
    f32   weight{0.01f};
    f32   decay{0.96f};
    f32   exposure{0.2f};
    u32   num_samples{100};
};

// Screen-space god ray accumulation for a single pixel
[[nodiscard]] inline Vec3f god_ray_sample(
    Vec2f uv, const GodRayParams& p,
    auto sample_fn) // sample_fn(Vec2f uv) -> Vec3f (scene color / occlusion)
{
    Vec2f delta=(uv-p.light_pos_ss)*(1.f/(f32(p.num_samples)*p.density));
    Vec2f cur=uv;
    Vec3f acc{0,0,0};
    f32 illum_decay=1.f;

    for(u32 i=0;i<p.num_samples;++i){
        cur=cur-delta;
        Vec3f s=sample_fn(cur)*p.weight*illum_decay;
        acc=acc+s;
        illum_decay*=p.decay;
    }
    return acc*p.exposure;
}

// ─── Volumetric clouds ────────────────────────────────────────────────────────

struct CloudParams {
    Vec3f bounds_min{-500,800,-500};
    Vec3f bounds_max{ 500,1200, 500};
    f32   density_scale{0.3f};
    f32   cloud_coverage{0.5f};
    f32   cloud_type{0.5f};      // 0=stratus, 1=cumulus
    f32   absorption{0.75f};
    Vec3f sun_dir{0.3f,-1.f,0.5f};
};

// Sample cloud density at world position (using simple FBM noise)
[[nodiscard]] inline f32 cloud_density(const Vec3f& p, const CloudParams& c) {
    // Normalize height within cloud layer
    f32 h_frac=(p.y-c.bounds_min.y)/(c.bounds_max.y-c.bounds_min.y);
    h_frac=std::clamp(h_frac,0.f,1.f);

    // Height gradient (round-top clouds)
    f32 height_shape=h_frac<0.1f?h_frac/0.1f:1.f-std::pow((h_frac-0.1f)/0.9f,2.f);

    // Simple noise (would use real 3D noise in practice)
    f32 base=std::sin(p.x*0.01f)*std::cos(p.z*0.01f)*0.5f+0.5f;
    base=base-c.cloud_coverage;
    if(base<0.f) return 0.f;

    return base*height_shape*c.density_scale;
}

// Ray march through cloud layer
[[nodiscard]] inline std::pair<f32,Vec3f> cloud_march(
    const Vec3f& ray_o, const Vec3f& ray_d,
    const CloudParams& c, u32 steps=64, Pcg32& rng={})
{
    // Intersect with cloud AABB
    Vec3f inv_d{1.f/ray_d.x,1.f/ray_d.y,1.f/ray_d.z};
    Vec3f t_lo=(c.bounds_min-ray_o)*inv_d;
    Vec3f t_hi=(c.bounds_max-ray_o)*inv_d;
    f32 t_enter=std::max({std::min(t_lo.x,t_hi.x),std::min(t_lo.y,t_hi.y),std::min(t_lo.z,t_hi.z)});
    f32 t_exit =std::min({std::max(t_lo.x,t_hi.x),std::max(t_lo.y,t_hi.y),std::max(t_lo.z,t_hi.z)});
    if(t_enter>t_exit||t_exit<0.f) return {1.f,{0,0,0}};

    t_enter=std::max(0.f,t_enter);
    f32 step=(t_exit-t_enter)/f32(steps);
    f32 transmittance=1.f;
    Vec3f scattering{0,0,0};

    for(u32 i=0;i<steps;++i){
        f32 t=t_enter+(f32(i)+rng.next_f32())*step;
        Vec3f pos=ray_o+ray_d*t;
        f32 d=cloud_density(pos,c);
        if(d<=0.f) continue;
        f32 sigma_t=d*c.absorption;
        f32 step_tr=std::exp(-sigma_t*step);
        // Approximate light contribution
        f32 light_d=d*std::max(0.f,-dot(normalize(c.sun_dir),ray_d));
        scattering=scattering+Vec3f{1.f,0.98f,0.95f}*(light_d*transmittance*(1.f-step_tr)/sigma_t);
        transmittance*=step_tr;
        if(transmittance<0.01f) break;
    }
    return {transmittance,scattering};
}

// ─── Atmospheric scattering ───────────────────────────────────────────────────

struct AtmosphericScattering {
    f32 R_e{6371000.f};     // Earth radius (m)
    f32 H_r{8000.f};        // Rayleigh scale height (m)
    f32 H_m{1200.f};        // Mie scale height (m)
    Vec3f beta_r{5.8e-6f,13.5e-6f,33.1e-6f}; // Rayleigh scattering coeffs
    f32 beta_m{2.1e-5f};    // Mie scattering coeff
    f32 g_m{0.76f};         // Mie asymmetry

    [[nodiscard]] f32 density_rayleigh(f32 altitude) const {
        return std::exp(-altitude/H_r);
    }
    [[nodiscard]] f32 density_mie(f32 altitude) const {
        return std::exp(-altitude/H_m);
    }

    // Precomputed sky color at view direction
    [[nodiscard]] Vec3f sky_color(const Vec3f& view_dir, const Vec3f& sun_dir) const {
        f32 cos_theta=dot(view_dir,sun_dir);
        f32 rayleigh=rayleigh_phase(cos_theta);
        f32 mie=mie_phase_cs(cos_theta,g_m);
        // Simplified (no actual integration): use analytical approximation
        f32 altitude_view=std::max(0.f,view_dir.y*10000.f);
        f32 dr=density_rayleigh(altitude_view);
        f32 dm=density_mie(altitude_view);
        Vec3f sky_r={beta_r.x*dr*rayleigh, beta_r.y*dr*rayleigh, beta_r.z*dr*rayleigh};
        Vec3f sky_m={beta_m*dm*mie, beta_m*dm*mie, beta_m*dm*mie};
        return (sky_r+sky_m)*std::max(0.f,sun_dir.y)*20.f;
    }
};

} // namespace qs
