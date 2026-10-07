
#pragma once
// qs_math/rendering/include/qs_math_rendering.hpp (C++20)
// Rendering math: PBR BRDFs, importance sampling, tone mapping,
// ray utilities, color spaces, halftoning, dithering, sampling.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/vec4.hpp>
#include <cmath>
#include <array>
namespace qs::math::render {
using namespace qs::math;
constexpr f32 PI_F = 3.14159265358979323846f;
constexpr f32 INV_PI = 1.0f / PI_F;
// ─── PBR BRDFs ───────────────────────────────────────────────────────────────
// GGX Normal Distribution Function
[[nodiscard]] inline f32 ggx_ndf(Vec3<f32> N, Vec3<f32> H, f32 roughness) noexcept {
    const f32 a = roughness*roughness;
    const f32 a2 = a*a;
    const f32 NdH = std::max(0.0f, N.dot(H));
    const f32 NdH2 = NdH*NdH;
    const f32 d = NdH2*(a2-1.0f)+1.0f;
    return a2/(PI_F*d*d);
}
// Smith geometry function (Schlick-GGX)
[[nodiscard]] inline f32 schlick_ggx(f32 NdV, f32 roughness) noexcept {
    const f32 r = roughness+1.0f;
    const f32 k = r*r/8.0f;
    return NdV/(NdV*(1-k)+k);
}
[[nodiscard]] inline f32 smith_g(f32 NdV, f32 NdL, f32 roughness) noexcept {
    return schlick_ggx(std::max(NdV,0.0f),roughness)*schlick_ggx(std::max(NdL,0.0f),roughness);
}
// Fresnel-Schlick
[[nodiscard]] inline Vec3<f32> fresnel_schlick(f32 cos_theta, Vec3<f32> F0) noexcept {
    const f32 p = std::pow(std::max(0.0f,1.0f-cos_theta),5.0f);
    return F0 + (Vec3<f32>{1,1,1}-F0)*p;
}
[[nodiscard]] inline Vec3<f32> fresnel_schlick_rough(f32 cos_theta, Vec3<f32> F0, f32 r) noexcept {
    const f32 p = std::pow(std::max(0.0f,1.0f-cos_theta),5.0f);
    return F0 + (Vec3<f32>{1,1,1}-F0)*p*(1.0f-r);
}
// Cook-Torrance specular BRDF
[[nodiscard]] Vec3<f32> cook_torrance(Vec3<f32> N, Vec3<f32> V, Vec3<f32> L,
    Vec3<f32> F0, f32 roughness) noexcept;
// Lambertian diffuse BRDF
[[nodiscard]] inline Vec3<f32> lambertian(Vec3<f32> albedo) noexcept {
    return albedo * INV_PI;
}
// Disney diffuse (Burley)
[[nodiscard]] f32 disney_diffuse(f32 NdV, f32 NdL, f32 LdH, f32 roughness) noexcept;
// Oren-Nayar diffuse
[[nodiscard]] f32 oren_nayar(Vec3<f32> N, Vec3<f32> V, Vec3<f32> L, f32 sigma) noexcept;
// ─── Importance sampling ─────────────────────────────────────────────────────
[[nodiscard]] Vec3<f32> sample_ggx(Vec3<f32> N, f32 roughness, f32 u1, f32 u2) noexcept;
[[nodiscard]] f32 pdf_ggx(Vec3<f32> N, Vec3<f32> H, Vec3<f32> V, f32 roughness) noexcept;
// ─── Tone mapping operators ───────────────────────────────────────────────────
[[nodiscard]] inline Vec3<f32> tone_aces(Vec3<f32> x) noexcept {
    const f32 a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
    return (x*(x*a+Vec3<f32>{b,b,b})) / (x*(x*c+Vec3<f32>{d,d,d})+Vec3<f32>{e,e,e});
}
[[nodiscard]] inline Vec3<f32> tone_reinhard(Vec3<f32> x) noexcept {
    return x/(Vec3<f32>{1,1,1}+x);
}
[[nodiscard]] inline Vec3<f32> tone_reinhard_ext(Vec3<f32> x, f32 max_l) noexcept {
    return x*(Vec3<f32>{1,1,1}+x*(1.0f/(max_l*max_l)))/(Vec3<f32>{1,1,1}+x);
}
[[nodiscard]] inline Vec3<f32> tone_exposure(Vec3<f32> x, f32 ev) noexcept {
    return Vec3<f32>{1,1,1}-Vec3<f32>{std::exp(-x.x*ev),std::exp(-x.y*ev),std::exp(-x.z*ev)};
}
[[nodiscard]] Vec3<f32> tone_agx(Vec3<f32> x) noexcept;
// ─── Color space conversions ──────────────────────────────────────────────────
[[nodiscard]] Vec3<f32> linear_to_srgb(Vec3<f32> c) noexcept;
[[nodiscard]] Vec3<f32> srgb_to_linear(Vec3<f32> c) noexcept;
[[nodiscard]] Vec3<f32> rgb_to_hsv(Vec3<f32> rgb) noexcept;
[[nodiscard]] Vec3<f32> hsv_to_rgb(Vec3<f32> hsv) noexcept;
[[nodiscard]] Vec3<f32> rgb_to_hsl(Vec3<f32> rgb) noexcept;
[[nodiscard]] Vec3<f32> hsl_to_rgb(Vec3<f32> hsl) noexcept;
[[nodiscard]] Vec3<f32> rgb_to_ycbcr(Vec3<f32> rgb) noexcept;
[[nodiscard]] Vec3<f32> ycbcr_to_rgb(Vec3<f32> y) noexcept;
[[nodiscard]] Vec3<f32> rgb_to_xyz(Vec3<f32> rgb) noexcept;
[[nodiscard]] Vec3<f32> xyz_to_rgb(Vec3<f32> xyz) noexcept;
// ─── Dithering ────────────────────────────────────────────────────────────────
[[nodiscard]] f32 bayer_dither(i32 x, i32 y, f32 value) noexcept;
[[nodiscard]] Vec3<f32> ordered_dither_rgb(Vec3<f32> c, i32 x, i32 y) noexcept;
// ─── Environment mapping ─────────────────────────────────────────────────────
[[nodiscard]] Vec2<f32> dir_to_equirect(Vec3<f32> dir) noexcept;
[[nodiscard]] Vec3<f32> equirect_to_dir(Vec2<f32> uv) noexcept;
[[nodiscard]] Vec2<f32> dir_to_octahedron(Vec3<f32> dir) noexcept;
[[nodiscard]] Vec3<f32> octahedron_to_dir(Vec2<f32> uv) noexcept;
// ─── Shadow utilities ─────────────────────────────────────────────────────────
[[nodiscard]] f32 pcf_shadow(f32 depth, f32 shadow_depth, f32 bias=0.005f) noexcept;
[[nodiscard]] f32 vsm_shadow(Vec2<f32> moments, f32 depth) noexcept;
} // namespace qs::math::render
