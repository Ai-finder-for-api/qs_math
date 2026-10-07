#pragma once
// lighting_math.hpp — Lighting models: punctual lights, area lights,
//                     spherical harmonics, light probes, shadow math,
//                     sky model (Preetham/Hosek), photometric units
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat3.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "constants.hpp"
#include "material_math.hpp"

namespace qs::math::lighting {

// ---------------------------------------------------------------------------
// Light types
// ---------------------------------------------------------------------------
struct DirectionalLight {
    Vec3f direction{0.f,-1.f,0.f}; // normalized, points FROM light
    Vec3f color{1.f};
    f32   intensity{1.f};
};

struct PointLight {
    Vec3f position{0.f};
    Vec3f color{1.f};
    f32   intensity{1.f};
    f32   radius{10.f};      // influence radius
    f32   falloff{2.f};      // exponent

    [[nodiscard]] f32 attenuation(f32 dist) const noexcept {
        f32 d=dist/radius;
        return max(0.f,1.f-d*d)/(dist*dist+EPS_F32);
    }
};

struct SpotLight {
    Vec3f position{0.f};
    Vec3f direction{0.f,-1.f,0.f};
    Vec3f color{1.f};
    f32   intensity{1.f};
    f32   inner_cone_cos{0.866f}; // cos(30°)
    f32   outer_cone_cos{0.707f}; // cos(45°)
    f32   radius{20.f};

    [[nodiscard]] f32 angular_attenuation(Vec3f to_light_normalized) const noexcept {
        f32 cd=dot(to_light_normalized,-direction);
        return smoothstep(outer_cone_cos,inner_cone_cos,cd);
    }
};

struct AreaLight {
    Vec3f position{0.f};
    Vec3f normal{0.f,1.f,0.f};
    Vec3f tangent{1.f,0.f,0.f};
    Vec3f color{1.f};
    f32   intensity{1.f};
    Vec2f size{1.f,1.f};
    bool  two_sided{false};
};

struct CapsuleLight {
    Vec3f point_a{0.f,-1.f,0.f};
    Vec3f point_b{0.f, 1.f,0.f};
    f32   radius{0.05f};
    Vec3f color{1.f};
    f32   intensity{1.f};
};

// ---------------------------------------------------------------------------
// Punctual light evaluation
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f evaluate_directional(
    DirectionalLight light, Vec3f N, Vec3f V,
    material::PBRMaterial mat) noexcept
{
    Vec3f L=-light.direction.normalized();
    return material::brdf_cook_torrance(N,V,L,mat)*light.color*light.intensity;
}

[[nodiscard]] inline Vec3f evaluate_point(
    PointLight light, Vec3f P, Vec3f N, Vec3f V,
    material::PBRMaterial mat) noexcept
{
    Vec3f LP=light.position-P;
    f32 dist=LP.length();
    if(dist<EPS_F32) return Vec3f{0.f};
    Vec3f L=LP*(1.f/dist);
    f32 att=light.attenuation(dist);
    return material::brdf_cook_torrance(N,V,L,mat)*light.color*light.intensity*att;
}

// ---------------------------------------------------------------------------
// Spherical Harmonics (order 2 / L2)
// ---------------------------------------------------------------------------
static constexpr i32 SH_COEFF_COUNT = 9; // L0+L1+L2

struct SH9 { Vec3f c[SH_COEFF_COUNT]; };

// SH basis functions
[[nodiscard]] inline void sh_evaluate_l2(Vec3f dir, f32 out[SH_COEFF_COUNT]) noexcept {
    f32 x=dir.x,y=dir.y,z=dir.z;
    // L0
    out[0]=0.282095f;
    // L1
    out[1]=0.488603f*y;
    out[2]=0.488603f*z;
    out[3]=0.488603f*x;
    // L2
    out[4]=1.092548f*x*y;
    out[5]=1.092548f*y*z;
    out[6]=0.315392f*(3.f*z*z-1.f);
    out[7]=1.092548f*x*z;
    out[8]=0.546274f*(x*x-y*y);
}

// Add radiance sample to SH
inline void sh_add_sample(SH9& sh, Vec3f dir, Vec3f radiance, f32 weight) noexcept {
    f32 basis[SH_COEFF_COUNT];
    sh_evaluate_l2(dir,basis);
    for(i32 i=0;i<SH_COEFF_COUNT;++i)
        sh.c[i]=sh.c[i]+radiance*(basis[i]*weight);
}

// Evaluate SH irradiance at a surface normal
[[nodiscard]] inline Vec3f sh_irradiance(const SH9& sh, Vec3f N) noexcept {
    f32 basis[SH_COEFF_COUNT];
    sh_evaluate_l2(N,basis);
    // Apply cosine lobe convolution coefficients
    static constexpr f32 A[3]={1.f,2.f/3.f,0.25f};
    static constexpr i32 BAND[9]={0,1,1,1,2,2,2,2,2};
    Vec3f result{0.f};
    for(i32 i=0;i<SH_COEFF_COUNT;++i)
        result=result+sh.c[i]*(basis[i]*A[BAND[i]]);
    return result*PI;
}

// Project environment map to SH (sampling-based)
inline void sh_project_envmap_sample(SH9& sh, Vec3f dir, Vec3f color) noexcept {
    f32 b[SH_COEFF_COUNT]; sh_evaluate_l2(dir,b);
    for(i32 i=0;i<SH_COEFF_COUNT;++i) sh.c[i]=sh.c[i]+color*(b[i]*(4.f*PI/SH_COEFF_COUNT));
}

// ---------------------------------------------------------------------------
// Preetham sky model
// ---------------------------------------------------------------------------
struct PreethamParams {
    f32 turbidity{2.f};
    Vec3f sun_dir{0.f,1.f,0.f}; // normalized, pointing toward sun
};

[[nodiscard]] inline Vec3f preetham_sky(Vec3f view_dir, PreethamParams p) noexcept {
    // Simplified Preetham (compute xy chromaticity then convert to XYZ)
    f32 cos_gamma=clamp(dot(view_dir,p.sun_dir),-1.f,1.f);
    f32 cos_theta=max(0.f,view_dir.y);
    f32 gamma=safe_acos(cos_gamma);
    f32 theta_s=safe_acos(p.sun_dir.y);
    f32 T=p.turbidity;

    auto perez=[&](f32 theta, f32 g, f32 A, f32 B, f32 C, f32 D, f32 E)->f32{
        return (1.f+A*exp(B/max(cos_theta,0.01f)))*(1.f+C*exp(D*g)+E*cos_gamma*cos_gamma);
    };

    // Y luminance
    f32 aY=-0.0193f*T-0.2592f, bY=-0.0665f*T+0.0008f;
    f32 cY=-0.0004f*T+0.2125f, dY=-0.0641f*T-0.8989f, eY=-0.0033f*T+0.0452f;
    f32 Y=perez(cos_theta,gamma,aY,bY,cY,dY,eY);

    // Approximate RGB sky color
    Vec3f sky_color{Y*0.06f,Y*0.1f,Y*0.2f};
    sky_color=sky_color+Vec3f{1.f,0.9f,0.6f}*max(0.f,pow(cos_gamma,128.f))*Y*10.f;
    return sky_color;
}

// ---------------------------------------------------------------------------
// Shadow mapping helpers
// ---------------------------------------------------------------------------
// PCF shadow (percentage closer filtering)
[[nodiscard]] inline f32 pcf_shadow(Vec3f shadow_coord, f32 bias=0.005f) noexcept {
    // Assumes shadow_coord.z is depth, shadow_coord.xy is UV
    // Actual texture sampling done in shader — here we just compute the bias
    return shadow_coord.z - bias;
}

// Variance shadow map moment generation
[[nodiscard]] constexpr Vec2f vsm_moments(f32 depth) noexcept {
    return {depth, depth*depth};
}
[[nodiscard]] constexpr f32 vsm_shadow(Vec2f moments, f32 depth, f32 min_variance=0.0001f) noexcept {
    f32 p=depth<=moments.x?1.f:0.f;
    f32 variance=moments.y-moments.x*moments.x;
    variance=max(variance,min_variance);
    f32 d=depth-moments.x;
    f32 p_max=variance/(variance+d*d);
    return max(p,p_max);
}

// Exponential shadow map
[[nodiscard]] constexpr f32 esm_shadow(f32 esm_depth, f32 depth, f32 c=80.f) noexcept {
    return saturate(esm_depth*exp(-c*depth));
}

// ---------------------------------------------------------------------------
// Photometric units
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 luminous_flux_to_intensity(f32 lumens) noexcept { return lumens/(4.f*PI); }
[[nodiscard]] constexpr f32 lux_to_ev(f32 lux, f32 iso=100.f) noexcept {
    return log2(lux*iso/250.f);
}
[[nodiscard]] constexpr f32 candela_to_lux(f32 cd, f32 dist) noexcept { return cd/(dist*dist); }

// Photometric to radiometric
[[nodiscard]] constexpr f32 lumens_to_watts(f32 lumens, f32 efficacy=683.f) noexcept {
    return lumens/efficacy;
}

} // namespace qs::math::lighting
