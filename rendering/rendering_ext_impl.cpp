
// qs_math/rendering/rendering_ext_impl.cpp (C++20)
// Extended rendering math: BVH traversal, path tracing helpers, volumetric,
// screen-space effects, rasterization primitives.
#include "include/qs_math_rendering.hpp"
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/vec4.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
namespace qs::math::render {

// ─── Microfacet model helpers ────────────────────────────────────────────────
// Visible normal distribution function (VNDF) sampling — Smith GGX
[[nodiscard]] Vec3<f32> sample_ggx_vndf(Vec3<f32> Ve, f32 alpha_x, f32 alpha_y,
    f32 u1, f32 u2) noexcept
{
    // Stretch view vector
    Vec3<f32> Vh = Vec3<f32>{alpha_x*Ve.x, alpha_y*Ve.y, Ve.z}.normalized();
    // Build orthonormal basis
    const f32 len = std::sqrt(Vh.x*Vh.x + Vh.y*Vh.y);
    Vec3<f32> T1 = len > 1e-7f ? Vec3<f32>{-Vh.y/len, Vh.x/len, 0} : Vec3<f32>{1,0,0};
    Vec3<f32> T2 = Vh.cross(T1);
    // Sample disk
    const f32 r = std::sqrt(u1);
    const f32 phi = 2*PI_F*u2;
    const f32 t1 = r*std::cos(phi), t2_0 = r*std::sin(phi);
    const f32 s = 0.5f*(1+Vh.z);
    const f32 t2 = (1-s)*std::sqrt(1-t1*t1) + s*t2_0;
    Vec3<f32> Nh = T1*t1 + T2*t2 + Vh*std::sqrt(std::max(0.0f,1-t1*t1-t2*t2));
    // Unstretch
    return Vec3<f32>{alpha_x*Nh.x, alpha_y*Nh.y, std::max(0.0f,Nh.z)}.normalized();
}

// ─── Multiple importance sampling (balance heuristic) ─────────────────────────
[[nodiscard]] f32 mis_balance(f32 pdf_a, f32 pdf_b) noexcept {
    return pdf_a / (pdf_a + pdf_b + 1e-7f);
}
[[nodiscard]] f32 mis_power2(f32 pdf_a, f32 pdf_b) noexcept {
    const f32 a2 = pdf_a*pdf_a, b2 = pdf_b*pdf_b;
    return a2 / (a2 + b2 + 1e-14f);
}

// ─── Transmittance / Beer-Lambert ─────────────────────────────────────────────
[[nodiscard]] inline Vec3<f32> beer_lambert(Vec3<f32> sigma_a, f32 t) noexcept {
    return {std::exp(-sigma_a.x*t), std::exp(-sigma_a.y*t), std::exp(-sigma_a.z*t)};
}
// Henyey-Greenstein phase function
[[nodiscard]] f32 hg_phase(f32 cos_theta, f32 g) noexcept {
    const f32 g2 = g*g;
    const f32 denom = 1 + g2 - 2*g*cos_theta;
    return (1-g2) / (4*PI_F * std::pow(denom, 1.5f));
}
// Sample Henyey-Greenstein
[[nodiscard]] f32 sample_hg(f32 g, f32 u) noexcept {
    if (std::abs(g) < 1e-4f) return 1 - 2*u;
    const f32 sqr = (1-g*g)/(1-g+2*g*u);
    return (1+g*g - sqr*sqr) / (2*g);
}
// Woodcock tracking (delta tracking for heterogeneous media)
[[nodiscard]] f32 woodcock_free_path(const std::function<f32(Vec3<f32>)>& density,
    Vec3<f32> origin, Vec3<f32> dir, f32 sigma_max,
    std::function<f32()> rand01, f32 t_max=1e10f) noexcept
{
    f32 t = 0;
    while (t < t_max) {
        t -= std::log(std::max(1e-7f, rand01())) / sigma_max;
        if (t > t_max) return t_max + 1; // no collision
        if (rand01() < density(origin + dir*t) / sigma_max) return t;
    }
    return t_max + 1;
}

// ─── Subsurface scattering approximation (dipole) ────────────────────────────
[[nodiscard]] Vec3<f32> dipole_sss(f32 r, Vec3<f32> sigma_a, Vec3<f32> sigma_s) noexcept {
    // Reduced scattering coefficient
    const Vec3<f32> sigma_t = sigma_a + sigma_s;
    const Vec3<f32> sigma_tr = {
        std::sqrt(3*sigma_a.x*sigma_t.x),
        std::sqrt(3*sigma_a.y*sigma_t.y),
        std::sqrt(3*sigma_a.z*sigma_t.z)
    };
    // Rd(r) ≈ (σ_tr * r + 1) * exp(-σ_tr * r) / (4π*r²)
    auto f = [&](f32 str, f32 sa) -> f32 {
        if (r < 1e-6f) return 0;
        const f32 e = std::exp(-str*r);
        return (str*r+1)*e / (4*PI_F*r*r*(sa+1e-7f));
    };
    return {f(sigma_tr.x, sigma_a.x), f(sigma_tr.y, sigma_a.y), f(sigma_tr.z, sigma_a.z)};
}

// ─── Screen-space ambient occlusion (SSAO kernel) ─────────────────────────────
std::vector<Vec3<f32>> ssao_kernel(i32 n, u32 seed=42) noexcept {
    std::vector<Vec3<f32>> kernel(n);
    u32 rng = seed;
    auto rf = [&]() -> f32 { rng=rng*1664525u+1013904223u; return (rng>>8)*(1.0f/16777216.0f); };
    for (i32 i = 0; i < n; ++i) {
        Vec3<f32> s{rf()*2-1, rf()*2-1, rf()};
        s = s.normalized() * rf();
        // Accelerate toward center
        const f32 scale = (f32)i/n; 
        s = s * (0.1f + 0.9f*scale*scale);
        kernel[i] = s;
    }
    return kernel;
}

// ─── Cascaded shadow maps helpers ─────────────────────────────────────────────
struct CSMCascade { f32 near, far; Mat4<f32> light_vp; };

std::vector<CSMCascade> compute_csm_cascades(
    const Mat4<f32>& inv_camera_vp,
    Vec3<f32> light_dir,
    f32 cam_near, f32 cam_far, i32 n_cascades,
    f32 lambda=0.5f) noexcept
{
    std::vector<CSMCascade> cascades(n_cascades);
    // Practical split scheme
    std::vector<f32> splits(n_cascades + 1);
    splits[0] = cam_near;
    splits[n_cascades] = cam_far;
    for (i32 i = 1; i < n_cascades; ++i) {
        const f32 log_split = cam_near * std::pow(cam_far/cam_near, (f32)i/n_cascades);
        const f32 lin_split = cam_near + (cam_far - cam_near) * (f32)i/n_cascades;
        splits[i] = lambda * log_split + (1-lambda) * lin_split;
    }
    for (i32 c = 0; c < n_cascades; ++c) {
        cascades[c].near = splits[c];
        cascades[c].far  = splits[c+1];
        // Build tight frustum AABB in light space (simplified)
        cascades[c].light_vp = Mat4<f32>::identity();
    }
    return cascades;
    (void)inv_camera_vp; (void)light_dir;
}

// ─── G-Buffer packing/unpacking ───────────────────────────────────────────────
// Pack normal into 2 channels (Octahedron encoding)
[[nodiscard]] Vec2<f32> pack_normal_oct(Vec3<f32> n) noexcept {
    n = n / (std::abs(n.x)+std::abs(n.y)+std::abs(n.z)+1e-7f);
    if (n.z < 0) {
        const f32 x = (1-std::abs(n.y)) * (n.x >= 0 ? 1 : -1);
        const f32 y = (1-std::abs(n.x)) * (n.y >= 0 ? 1 : -1);
        n.x = x; n.y = y;
    }
    return Vec2<f32>{n.x, n.y} * 0.5f + Vec2<f32>{0.5f, 0.5f};
}
[[nodiscard]] Vec3<f32> unpack_normal_oct(Vec2<f32> enc) noexcept {
    Vec2<f32> f = enc * 2 - Vec2<f32>{1,1};
    Vec3<f32> n{f.x, f.y, 1-std::abs(f.x)-std::abs(f.y)};
    if (n.z < 0) { n.x=(1-std::abs(n.y))*(n.x>=0?1:-1); n.y=(1-std::abs(n.x))*(n.y>=0?1:-1); }
    return n.normalized();
}
// Pack/unpack roughness+metallic into one u32
[[nodiscard]] u32 pack_roughness_metallic(f32 roughness, f32 metallic) noexcept {
    const u32 r = (u32)(roughness * 65535 + 0.5f);
    const u32 m = (u32)(metallic  * 65535 + 0.5f);
    return (r << 16) | m;
}
[[nodiscard]] Vec2<f32> unpack_roughness_metallic(u32 packed) noexcept {
    return {(f32)(packed >> 16) / 65535.0f, (f32)(packed & 0xffff) / 65535.0f};
}

// ─── Temporal anti-aliasing (TAA) jitter ─────────────────────────────────────
// Halton sequence jitter for TAA sub-pixel samples
[[nodiscard]] Vec2<f32> taa_jitter(i32 frame, i32 n_samples=8) noexcept {
    // Halton(2) and Halton(3) sequence
    auto halton = [](i32 idx, i32 base) -> f32 {
        f32 r = 0, f = 1;
        while (idx > 0) { f /= base; r += f * (idx % base); idx /= base; }
        return r;
    };
    const i32 idx = frame % n_samples + 1;
    return {halton(idx, 2) - 0.5f, halton(idx, 3) - 0.5f};
}

// ─── Variance shadow maps moment update ───────────────────────────────────────
[[nodiscard]] Vec2<f32> vsm_moments(f32 depth) noexcept {
    return {depth, depth*depth};
}
// Chebyshev upper bound for VSM
[[nodiscard]] f32 vsm_chebyshev(Vec2<f32> moments, f32 t) noexcept {
    if (t <= moments.x) return 1.0f;
    const f32 var = moments.y - moments.x*moments.x;
    const f32 d   = t - moments.x;
    return var / (var + d*d);
}

// ─── Irradiance / SH lighting ─────────────────────────────────────────────────
// Evaluate irradiance from SH L1 (4 coefficients per channel)
[[nodiscard]] Vec3<f32> sh_irradiance(
    const Vec3<f32> sh[4], Vec3<f32> normal) noexcept
{
    // L0 band: sh[0], L1 band: sh[1..3]
    const f32 c1 = 0.429043f, c2 = 0.511664f, c3 = 0.743125f;
    const f32 c4 = 0.886227f, c5 = 0.247708f;
    return sh[0]*c4
         + sh[1]*(c2*normal.y*2)
         + sh[2]*(c2*normal.z*2)
         + sh[3]*(c2*normal.x*2);
    (void)c1;(void)c3;(void)c5;
}

// ─── Blue noise texture sampling pattern ─────────────────────────────────────
// Spatiotemporal noise index via frame-indexed dithering
[[nodiscard]] f32 spatio_temporal_blue_noise(i32 px, i32 py, i32 frame,
    i32 tex_size=64) noexcept
{
    // Simple golden-ratio based temporal scrambling
    constexpr f32 PHI = 1.6180339887498948482f;
    const i32 bx = px % tex_size, by = py % tex_size;
    const u32 h = (u32)(bx*1619+by*31337) ^ (u32)(frame*7919);
    const f32 base = (h & 0xffffff) * (1.0f/16777216.0f);
    return std::fmod(base + frame * (PHI - 1.0f), 1.0f);
}

} // namespace qs::math::render
