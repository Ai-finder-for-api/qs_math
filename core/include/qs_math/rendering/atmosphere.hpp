#pragma once
// qs_math/rendering/atmosphere.hpp — Atmospheric scattering: Rayleigh & Mie,
//   transmittance LUT, single/multi-scatter, aerial perspective,
//   ozone absorption, phase functions, sky model (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <cmath>
#include <vector>

namespace qs::math::render {

// ─── Atmosphere parameters ────────────────────────────────────────────────
struct AtmosphereParams {
    f32 planet_radius    = 6360e3f;  // m
    f32 atm_height       = 100e3f;   // m (atmosphere thickness)
    f32 H_rayleigh       = 8000.f;   // scale height Rayleigh (m)
    f32 H_mie           = 1200.f;   // scale height Mie (m)
    Vec3f beta_rayleigh  = {5.802e-6f, 13.558e-6f, 33.1e-6f}; // 1/m at 680,550,440nm
    f32 beta_mie         = 3.996e-6f; // 1/m
    f32 beta_mie_extinction_ratio = 1.11f; // extinction/scattering ratio
    Vec3f beta_ozone     = {0.650e-6f, 1.881e-6f, 0.085e-6f}; // 1/m
    f32 mie_g            = 0.8f;      // Mie asymmetry
    Vec3f ground_albedo  = {0.3f,0.3f,0.3f};
    Vec3f sun_irradiance = {1.474f,1.850f,1.911f}; // W/m^2/nm (approx)
};

// ─── Density profiles ─────────────────────────────────────────────────────
[[nodiscard]] inline f32 rayleigh_density(f32 altitude, const AtmosphereParams& a) noexcept {
    return std::exp(-altitude/a.H_rayleigh);
}
[[nodiscard]] inline f32 mie_density(f32 altitude, const AtmosphereParams& a) noexcept {
    return std::exp(-altitude/a.H_mie);
}
// Ozone: tent function peaking at 25km
[[nodiscard]] inline f32 ozone_density(f32 altitude) noexcept {
    if(altitude<10e3f) return 0.f;
    if(altitude<25e3f) return (altitude-10e3f)/15e3f;
    if(altitude<40e3f) return 1.f-(altitude-25e3f)/15e3f;
    return 0.f;
}

// ─── Altitude from point ──────────────────────────────────────────────────
[[nodiscard]] inline f32 altitude_at(Vec3f pos, const AtmosphereParams& a) noexcept {
    return pos.length()-a.planet_radius;
}

// ─── Ray-sphere intersection ──────────────────────────────────────────────
struct SphereIsect { bool hit; f32 t0,t1; };
[[nodiscard]] inline SphereIsect ray_sphere(Vec3f o, Vec3f d, f32 r) noexcept {
    f32 b=dot(o,d), c=dot(o,o)-r*r;
    f32 disc=b*b-c;
    if(disc<0) return {false,0,0};
    f32 sq=std::sqrt(disc);
    return {true,-b-sq,-b+sq};
}

// ─── Optical depth along ray segment ─────────────────────────────────────
struct OpticalDepth { f32 rayleigh, mie, ozone; };

[[nodiscard]] inline OpticalDepth optical_depth_segment(
    Vec3f p, Vec3f q, const AtmosphereParams& a, int steps=32) noexcept
{
    OpticalDepth od{0,0,0};
    Vec3f step_v=(q-p)*(1.f/steps);
    f32 step_len=step_v.length();
    for(int i=0;i<steps;++i){
        Vec3f sample=p+step_v*(i+0.5f);
        f32 alt=altitude_at(sample,a);
        if(alt<0||alt>a.atm_height) continue;
        od.rayleigh+=rayleigh_density(alt,a)*step_len;
        od.mie     +=mie_density(alt,a)*step_len;
        od.ozone   +=ozone_density(alt)*step_len;
    }
    return od;
}

// ─── Transmittance ────────────────────────────────────────────────────────
[[nodiscard]] inline Vec3f transmittance(
    const OpticalDepth& od, const AtmosphereParams& a) noexcept
{
    Vec3f beta_ext_mie=a.beta_mie*a.beta_mie_extinction_ratio;
    return {
        std::exp(-(a.beta_rayleigh.x*od.rayleigh+beta_ext_mie*od.mie+a.beta_ozone.x*od.ozone)),
        std::exp(-(a.beta_rayleigh.y*od.rayleigh+beta_ext_mie*od.mie+a.beta_ozone.y*od.ozone)),
        std::exp(-(a.beta_rayleigh.z*od.rayleigh+beta_ext_mie*od.mie+a.beta_ozone.z*od.ozone))
    };
}

[[nodiscard]] inline Vec3f transmittance_ray(
    Vec3f p, Vec3f d, f32 t_max, const AtmosphereParams& a, int steps=32) noexcept
{
    Vec3f q=p+d*t_max;
    auto od=optical_depth_segment(p,q,a,steps);
    return transmittance(od,a);
}

// ─── Phase functions ──────────────────────────────────────────────────────
[[nodiscard]] inline f32 rayleigh_phase(f32 cos_theta) noexcept {
    return 3.f/(16.f*f32(PI))*(1+cos_theta*cos_theta);
}
[[nodiscard]] inline f32 mie_phase_hg(f32 cos_theta, f32 g) noexcept {
    f32 g2=g*g;
    f32 denom=1+g2-2*g*cos_theta;
    return (1-g2)/(4.f*f32(PI)*denom*std::sqrt(denom));
}
// Cornette-Shanks (more physically accurate Mie)
[[nodiscard]] inline f32 mie_phase_cs(f32 cos_theta, f32 g) noexcept {
    f32 g2=g*g;
    f32 denom=std::pow(1+g2-2*g*cos_theta,1.5f);
    return 3.f/(8.f*f32(PI))*(1-g2)*(1+cos_theta*cos_theta)/((2+g2)*denom);
}

// ─── Single-scatter sky radiance ──────────────────────────────────────────
[[nodiscard]] inline Vec3f sky_radiance_single(
    Vec3f ray_origin, Vec3f ray_dir, Vec3f sun_dir,
    const AtmosphereParams& a,
    int view_steps=32, int sun_steps=8) noexcept
{
    // Find ray end at atmosphere top
    auto isect=ray_sphere(ray_origin,ray_dir,a.planet_radius+a.atm_height);
    if(!isect.hit||isect.t1<0) return {0,0,0};
    f32 t_max=max(0.f,isect.t0)<0?isect.t1:isect.t0;
    // Check planet intersection
    auto planet=ray_sphere(ray_origin,ray_dir,a.planet_radius);
    if(planet.hit&&planet.t0>0) t_max=min(t_max,planet.t0);

    Vec3f L{0,0,0};
    f32 step_len=t_max/view_steps;
    f32 cos_theta=dot(ray_dir,sun_dir);
    f32 pr=rayleigh_phase(cos_theta);
    f32 pm=mie_phase_cs(cos_theta,a.mie_g);

    for(int i=0;i<view_steps;++i){
        Vec3f p=ray_origin+ray_dir*(step_len*(i+0.5f));
        f32 alt=altitude_at(p,a);
        if(alt<0||alt>a.atm_height) continue;
        f32 rho_r=rayleigh_density(alt,a);
        f32 rho_m=mie_density(alt,a);
        // Transmittance from camera to sample
        auto od_cam=optical_depth_segment(ray_origin,p,a,view_steps/2);
        Vec3f T_cam=transmittance(od_cam,a);
        // Transmittance from sample to sun
        auto sun_isect=ray_sphere(p,sun_dir,a.planet_radius+a.atm_height);
        Vec3f T_sun{1,1,1};
        if(sun_isect.hit&&sun_isect.t1>0){
            Vec3f sun_end=p+sun_dir*sun_isect.t1;
            auto od_sun=optical_depth_segment(p,sun_end,a,sun_steps);
            T_sun=transmittance(od_sun,a);
        }
        // Scattering contribution
        Vec3f scatter_r=a.beta_rayleigh*(rho_r*pr);
        Vec3f scatter_m=Vec3f{a.beta_mie,a.beta_mie,a.beta_mie}*(rho_m*pm);
        Vec3f in_scatter={
            (scatter_r.x+scatter_m.x)*T_cam.x*T_sun.x,
            (scatter_r.y+scatter_m.y)*T_cam.y*T_sun.y,
            (scatter_r.z+scatter_m.z)*T_cam.z*T_sun.z
        };
        L=L+in_scatter*step_len;
    }
    return {L.x*a.sun_irradiance.x, L.y*a.sun_irradiance.y, L.z*a.sun_irradiance.z};
}

// ─── Aerial perspective ───────────────────────────────────────────────────
// Computes in-scattering and transmittance for objects in the scene
struct AerialResult { Vec3f in_scatter; Vec3f transmittance_val; };

[[nodiscard]] inline AerialResult aerial_perspective(
    Vec3f camera, Vec3f frag_pos, Vec3f sun_dir,
    const AtmosphereParams& a, int steps=16) noexcept
{
    Vec3f ray_dir=(frag_pos-camera);
    f32 dist=ray_dir.length();
    if(dist<1e-3f) return {{0,0,0},{1,1,1}};
    ray_dir=ray_dir*(1.f/dist);

    f32 cos_theta=dot(ray_dir,sun_dir);
    f32 pr=rayleigh_phase(cos_theta), pm=mie_phase_cs(cos_theta,a.mie_g);

    Vec3f L{0,0,0};
    f32 step_len=dist/steps;
    for(int i=0;i<steps;++i){
        Vec3f p=camera+ray_dir*(step_len*(i+0.5f));
        f32 alt=altitude_at(p,a);
        if(alt<0) continue;
        f32 rho_r=rayleigh_density(alt,a), rho_m=mie_density(alt,a);
        auto od_cam=optical_depth_segment(camera,p,a,steps/2);
        Vec3f T_cam=transmittance(od_cam,a);
        auto sun_isect=ray_sphere(p,sun_dir,a.planet_radius+a.atm_height);
        Vec3f T_sun{1,1,1};
        if(sun_isect.hit&&sun_isect.t1>0){
            Vec3f se=p+sun_dir*sun_isect.t1;
            T_sun=transmittance(optical_depth_segment(p,se,a,4),a);
        }
        Vec3f sc_r=a.beta_rayleigh*(rho_r*pr), sc_m=Vec3f{a.beta_mie,a.beta_mie,a.beta_mie}*(rho_m*pm);
        L=L+Vec3f{(sc_r.x+sc_m.x)*T_cam.x*T_sun.x,(sc_r.y+sc_m.y)*T_cam.y*T_sun.y,(sc_r.z+sc_m.z)*T_cam.z*T_sun.z}*step_len;
    }
    L={L.x*a.sun_irradiance.x,L.y*a.sun_irradiance.y,L.z*a.sun_irradiance.z};
    auto od_full=optical_depth_segment(camera,frag_pos,a,steps);
    return {L,transmittance(od_full,a)};
}

// ─── LUT baking helpers ───────────────────────────────────────────────────
// Bake transmittance LUT: indexed by (altitude, cos_zenith)
[[nodiscard]] inline std::vector<Vec3f> bake_transmittance_lut(
    const AtmosphereParams& a, int res_alt=64, int res_cos=64) noexcept
{
    std::vector<Vec3f> lut(res_alt*res_cos);
    for(int ia=0;ia<res_alt;++ia){
        f32 alt=a.atm_height*f32(ia)/(res_alt-1);
        Vec3f p{0,a.planet_radius+alt,0};
        for(int ic=0;ic<res_cos;++ic){
            f32 cos_z=2.f*ic/(res_cos-1)-1.f;
            f32 sin_z=std::sqrt(max(0.f,1-cos_z*cos_z));
            Vec3f d{sin_z,cos_z,0};
            auto isect=ray_sphere(p,d,a.planet_radius+a.atm_height);
            f32 t_max=isect.hit?max(0.f,isect.t1):0.f;
            lut[ia*res_cos+ic]=transmittance_ray(p,d,t_max,a,64);
        }
    }
    return lut;
}

// ─── Sun disk ─────────────────────────────────────────────────────────────
[[nodiscard]] inline Vec3f sun_disk(Vec3f view_dir, Vec3f sun_dir, f32 sun_angle=0.00872665f) noexcept {
    f32 cos_a=dot(view_dir.normalized(),sun_dir);
    f32 cos_min=std::cos(sun_angle);
    if(cos_a<cos_min) return {0,0,0};
    f32 t=(cos_a-cos_min)/(1.f-cos_min);
    return Vec3f{1,1,1}*(t*t);
}

// ─── Preetham sky model (simpler analytic) ────────────────────────────────
struct PreethamCoeffs {
    std::array<f32,5> Ax,Bx,Cx,Dx,Ex;
    std::array<f32,5> Ay,By,Cy,Dy,Ey;
};

[[nodiscard]] inline f32 preetham_F(const std::array<f32,5>& ABCDE, f32 theta, f32 gamma) noexcept {
    return (1+ABCDE[0]*std::exp(ABCDE[1]/std::cos(theta)))
          *(1+ABCDE[2]*std::exp(ABCDE[3]*gamma)+ABCDE[4]*std::cos(gamma)*std::cos(gamma));
}

// Perez luminance for sun elevation theta_s (radians)
[[nodiscard]] inline Vec3f preetham_sky(
    f32 theta_v, f32 gamma, f32 theta_s, f32 turbidity=2.0f) noexcept
{
    // Simplified coefficient set for turbidity 2
    f32 t=turbidity;
    std::array<f32,5> AY={0.1787f*t-1.4630f, -0.3554f*t+0.4275f,
                           -0.0227f*t+5.3251f, 0.1206f*t-2.5771f, -0.0670f*t+0.3703f};
    std::array<f32,5> Ax={-0.0193f*t-0.2592f,-0.0665f*t+0.0008f,
                           -0.0004f*t+0.2125f,-0.0641f*t-0.8989f,-0.0033f*t+0.0452f};
    std::array<f32,5> Ay={-0.0167f*t-0.2608f,-0.0950f*t+0.0092f,
                           -0.0079f*t+0.2102f,-0.0441f*t-1.6537f,-0.0109f*t+0.0529f};
    f32 Fvs=preetham_F(AY,theta_v,gamma);
    f32 Fzs=preetham_F(AY,0.f,theta_s);
    f32 Ys=(Fzs>1e-6f)?Fvs/Fzs*0.1f:0.f;
    return Vec3f{Ys,Ys,Ys}; // simplified monochrome for now
}

} // namespace qs::math::render
