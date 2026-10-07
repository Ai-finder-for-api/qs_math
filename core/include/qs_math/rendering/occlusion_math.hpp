#pragma once
// qs_math/rendering/occlusion_math.hpp — SSAO, HBAO, bent normals, voxel AO, horizon-based AO (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../linear_algebra/mat4.hpp"
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/random.hpp"
#include <array>
#include <cmath>

namespace qs::math::render {

// ─── SSAO kernel generation ────────────────────────────────────────────────
// Cosine-weighted hemisphere samples biased toward origin (Hammersley)
template<int N>
struct SSAOKernel {
    std::array<Vec3f, N> samples;

    SSAOKernel() noexcept {
        for (int i = 0; i < N; ++i) {
            f32 fi = f32(i) / f32(N);
            // Hammersley sequence
            u32 bits = u32(i);
            bits = (bits << 16u) | (bits >> 16u);
            bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
            bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
            bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
            bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
            f32 radical = f32(bits) * 2.3283064365386963e-10f;

            f32 phi = TWO_PI * fi;
            f32 cos_theta = std::sqrt(1.f - radical);
            f32 sin_theta = std::sqrt(radical);
            Vec3f s = {std::cos(phi)*sin_theta, std::sin(phi)*sin_theta, cos_theta};
            // Accelerating interpolation — bias toward origin
            f32 scale = fi * fi;
            scale = lerp(0.1f, 1.f, scale);
            samples[i] = s * scale;
        }
    }
};

// ─── SSAO sample rotation noise ───────────────────────────────────────────
// 4x4 tiling rotation vectors (Mittring technique)
struct SSAONoise {
    static constexpr int SIZE = 16;
    std::array<Vec3f, SIZE> vectors;

    SSAONoise(u32 seed = 0x12345678u) noexcept {
        PCG32 rng(seed);
        for (auto& v : vectors) {
            v = Vec3f{rng.next_f32()*2.f-1.f, rng.next_f32()*2.f-1.f, 0.f};
        }
    }

    [[nodiscard]] Vec3f get(int x, int y) const noexcept {
        return vectors[(y % 4) * 4 + (x % 4)];
    }
};

// ─── SSAO per-pixel computation ───────────────────────────────────────────
struct SSAOParams {
    f32 radius       = 0.5f;   // world-space sample radius
    f32 bias         = 0.025f; // depth bias to prevent self-occlusion
    f32 power        = 2.0f;   // occlusion contrast
    int num_samples  = 64;
    f32 range_check  = 1.0f;   // falloff at radius edge
};

// Compute SSAO occlusion factor for one fragment.
// position_vs: fragment position in view space
// normal_vs:   fragment normal in view space (unit)
// sample_kernel: array of hemisphere samples
// depth_at:   lambda (Vec2f uv) -> f32 depth in view space
// proj:       projection matrix
template<int N, typename DepthSampler>
[[nodiscard]] f32 ssao_occlusion(
    Vec3f position_vs, Vec3f normal_vs,
    const SSAOKernel<N>& kernel,
    const SSAONoise& noise,
    const Mat4f& proj,
    DepthSampler&& depth_at,
    const SSAOParams& p,
    int px, int py) noexcept
{
    Vec3f rvec = noise.get(px, py);
    // Gram-Schmidt TBN
    Vec3f tangent   = (rvec - normal_vs * dot(rvec, normal_vs)).normalized();
    Vec3f bitangent = cross(normal_vs, tangent);

    f32 occlusion = 0.f;
    for (int i = 0; i < N; ++i) {
        Vec3f s = kernel.samples[i];
        Vec3f sample_vs = {
            tangent.x*s.x + bitangent.x*s.y + normal_vs.x*s.z,
            tangent.y*s.x + bitangent.y*s.y + normal_vs.y*s.z,
            tangent.z*s.x + bitangent.z*s.y + normal_vs.z*s.z
        };
        sample_vs = position_vs + sample_vs * p.radius;

        // Project to UV
        Vec4f offset = proj * Vec4f{sample_vs.x, sample_vs.y, sample_vs.z, 1.f};
        offset.x /= offset.w; offset.y /= offset.w;
        Vec2f uv = {offset.x * 0.5f + 0.5f, offset.y * 0.5f + 0.5f};

        f32 sample_depth = depth_at(uv);
        f32 range = smooth_step(0.f, 1.f, p.radius / abs(position_vs.z - sample_depth));
        occlusion += (sample_depth >= sample_vs.z + p.bias ? 1.f : 0.f) * range;
    }
    occlusion = 1.f - occlusion / f32(N);
    return std::pow(occlusion, p.power);
}

// ─── Bent normals ─────────────────────────────────────────────────────────
// Accumulate unoccluded direction for bent normal computation.
// bent_normal = average of unoccluded sample directions
template<int N, typename DepthSampler>
[[nodiscard]] Vec3f compute_bent_normal(
    Vec3f position_vs, Vec3f normal_vs,
    const SSAOKernel<N>& kernel,
    const SSAONoise& noise,
    const Mat4f& proj,
    DepthSampler&& depth_at,
    const SSAOParams& p,
    int px, int py) noexcept
{
    Vec3f rvec = noise.get(px, py);
    Vec3f tangent   = (rvec - normal_vs * dot(rvec, normal_vs)).normalized();
    Vec3f bitangent = cross(normal_vs, tangent);

    Vec3f bent{0,0,0};
    for (int i = 0; i < N; ++i) {
        Vec3f s = kernel.samples[i];
        Vec3f dir_vs = {
            tangent.x*s.x + bitangent.x*s.y + normal_vs.x*s.z,
            tangent.y*s.x + bitangent.y*s.y + normal_vs.y*s.z,
            tangent.z*s.x + bitangent.z*s.y + normal_vs.z*s.z
        };
        Vec3f sample_vs = position_vs + dir_vs * p.radius;
        Vec4f offset = proj * Vec4f{sample_vs.x, sample_vs.y, sample_vs.z, 1.f};
        offset.x /= offset.w; offset.y /= offset.w;
        Vec2f uv = {offset.x*0.5f+0.5f, offset.y*0.5f+0.5f};
        f32 sample_depth = depth_at(uv);
        if (sample_depth < sample_vs.z + p.bias)
            bent = bent + dir_vs; // unoccluded
    }
    f32 len = bent.length();
    return len > EPS_LENGTH ? bent * (1.f / len) : normal_vs;
}

// ─── HBAO (Horizon-Based AO) ──────────────────────────────────────────────
struct HBAOParams {
    f32 radius        = 1.0f;   // world-space max radius
    f32 bias          = 0.05f;  // angle bias (radians)
    int num_directions= 8;      // azimuthal steps
    int num_steps     = 4;      // radial steps per direction
    f32 power         = 1.5f;
    f32 strength      = 1.0f;
};

// Linearize depth from [0,1] clip to view-space Z (negative depth convention)
[[nodiscard]] inline f32 linearize_depth(f32 depth_ndc, f32 near, f32 far) noexcept {
    return (2.f * near * far) / (far + near - depth_ndc * (far - near));
}

// Reconstruct view-space position from UV + linear depth
[[nodiscard]] inline Vec3f reconstruct_position(Vec2f uv, f32 depth_linear,
    f32 tan_half_fov_x, f32 tan_half_fov_y) noexcept
{
    Vec2f ndc = {uv.x * 2.f - 1.f, uv.y * 2.f - 1.f};
    return {ndc.x * tan_half_fov_x * depth_linear,
            ndc.y * tan_half_fov_y * depth_linear,
            -depth_linear};
}

// Compute HBAO for one pixel.
// horizon_angle accumulation per direction, then integrate.
template<typename DepthSampler, typename NormalSampler>
[[nodiscard]] f32 hbao_occlusion(
    Vec2f uv, Vec2f texel_size,
    f32 tan_half_fov_x, f32 tan_half_fov_y,
    DepthSampler&& depth_at,
    NormalSampler&& normal_at,
    const HBAOParams& p,
    u32 random_angle_bits = 0x3F800000u) noexcept
{
    f32 depth      = depth_at(uv);
    Vec3f pos      = reconstruct_position(uv, depth, tan_half_fov_x, tan_half_fov_y);
    Vec3f norm     = normal_at(uv);
    f32 rand_angle = f32(random_angle_bits & 0x00FFFFFFu) * (TWO_PI / f32(0x01000000u));

    f32 ao = 0.f;
    f32 dir_step = TWO_PI / f32(p.num_directions);

    for (int d = 0; d < p.num_directions; ++d) {
        f32 phi = rand_angle + f32(d) * dir_step;
        Vec2f dir = {std::cos(phi), std::sin(phi)};
        f32 max_horizon = p.bias;

        for (int s = 1; s <= p.num_steps; ++s) {
            f32 t = f32(s) / f32(p.num_steps);
            Vec2f sample_uv = uv + dir * (texel_size * t * p.radius);
            f32 sd = depth_at(sample_uv);
            Vec3f sp = reconstruct_position(sample_uv, sd, tan_half_fov_x, tan_half_fov_y);
            Vec3f diff = sp - pos;
            f32 dist = diff.length();
            if (dist < EPS_LENGTH) continue;
            f32 horizon = dot(diff, norm) / dist;
            f32 falloff = 1.f - t * t; // quadratic falloff
            max_horizon = max(max_horizon, horizon * falloff);
        }
        ao += clamp01(1.f - max_horizon);
    }
    ao /= f32(p.num_directions);
    return std::pow(ao, p.power) * p.strength;
}

// ─── Voxel AO ─────────────────────────────────────────────────────────────
// Computes ambient occlusion from 3D voxel occupancy grid.
// Samples 8 hemisphere directions using precomputed offsets.

// Returns AO factor [0,1] for a surface point given its normal.
// voxel_solid(ix,iy,iz) -> bool
template<typename VoxelQuery>
[[nodiscard]] f32 voxel_ao(
    Vec3f pos, Vec3f normal,
    VoxelQuery&& voxel_solid,
    f32 voxel_size = 1.f,
    int max_dist = 4) noexcept
{
    // 16-direction hemisphere samples
    static constexpr Vec3f dirs[16] = {
        { 0.f,  1.f,  0.f}, { 0.707f, 0.707f, 0.f},
        { 1.f,  0.f,  0.f}, { 0.707f, 0.f, 0.707f},
        { 0.f,  0.f,  1.f}, {-0.707f, 0.f, 0.707f},
        {-1.f,  0.f,  0.f}, {-0.707f, 0.707f, 0.f},
        { 0.577f, 0.577f, 0.577f}, {-0.577f, 0.577f, 0.577f},
        { 0.577f, 0.577f,-0.577f}, {-0.577f, 0.577f,-0.577f},
        { 0.707f,-0.707f, 0.f},  {-0.707f,-0.707f, 0.f},
        { 0.f, -0.707f, 0.707f}, { 0.f, -0.707f,-0.707f}
    };
    f32 blocked = 0.f, weight_sum = 0.f;
    for (auto& dir : dirs) {
        f32 w = max(0.f, dot(dir, normal));
        if (w < 0.01f) continue;
        weight_sum += w;
        // March ray through voxels
        for (int step = 1; step <= max_dist; ++step) {
            Vec3f p = pos + dir * (f32(step) * voxel_size);
            i32 ix = i32(std::floor(p.x / voxel_size));
            i32 iy = i32(std::floor(p.y / voxel_size));
            i32 iz = i32(std::floor(p.z / voxel_size));
            if (voxel_solid(ix, iy, iz)) {
                f32 falloff = 1.f - f32(step-1) / f32(max_dist);
                blocked += w * falloff;
                break;
            }
        }
    }
    return weight_sum > EPS_WEIGHT ? clamp01(1.f - blocked / weight_sum) : 1.f;
}

// ─── Capsule AO (analytic) ────────────────────────────────────────────────
// Analytic AO contribution from a single capsule occluder.
// Returns [0,1] darkening factor at point p with normal n.
[[nodiscard]] inline f32 capsule_ao(
    Vec3f p, Vec3f n,
    Vec3f cap_a, Vec3f cap_b, f32 cap_r) noexcept
{
    Vec3f ab = cap_b - cap_a;
    Vec3f ao = cap_a - p;
    f32 len = ab.length();
    if (len < EPS_LENGTH) {
        // Degenerate: sphere
        Vec3f d = p - cap_a;
        f32 dist = d.length();
        if (dist < cap_r) return 0.f;
        f32 horizon = cap_r / dist;
        return 1.f - clamp01(horizon * horizon * max(0.f, dot(d.normalized(), n)));
    }
    Vec3f ab_n = ab * (1.f / len);
    f32 t = clamp01(dot(-ao, ab_n) / len);
    Vec3f closest = cap_a + ab_n * (t * len);
    Vec3f d = p - closest;
    f32 dist = d.length();
    if (dist < cap_r) return 0.f;
    f32 sin_alpha = cap_r / dist;
    f32 cos_alpha = std::sqrt(max(0.f, 1.f - sin_alpha*sin_alpha));
    f32 n_dot_d = dot(n, d.normalized());
    // Integrate solid angle of visible cap against cosine-weighted hemisphere
    return 1.f - clamp01((1.f - cos_alpha) * max(0.f, n_dot_d));
}

// ─── Horizon AO blur (separable box) ──────────────────────────────────────
// Simple separable 5-tap blur for AO buffer
template<typename AOSampler>
[[nodiscard]] f32 ao_blur_h(AOSampler&& ao_at, Vec2f uv, f32 texel_w) noexcept {
    f32 sum = 0.f;
    static constexpr f32 w[5] = {0.0625f,0.25f,0.375f,0.25f,0.0625f};
    for (int i=-2; i<=2; ++i)
        sum += w[i+2] * ao_at(Vec2f{uv.x + f32(i)*texel_w, uv.y});
    return sum;
}
template<typename AOSampler>
[[nodiscard]] f32 ao_blur_v(AOSampler&& ao_at, Vec2f uv, f32 texel_h) noexcept {
    f32 sum = 0.f;
    static constexpr f32 w[5] = {0.0625f,0.25f,0.375f,0.25f,0.0625f};
    for (int i=-2; i<=2; ++i)
        sum += w[i+2] * ao_at(Vec2f{uv.x, uv.y + f32(i)*texel_h});
    return sum;
}

// ─── Bilateral AO upscale ─────────────────────────────────────────────────
// Bilateral filter to upscale half-res AO to full-res.
template<typename AOSampler, typename DepthSampler>
[[nodiscard]] f32 ao_bilateral_upsample(
    AOSampler&& ao_at, DepthSampler&& depth_at,
    Vec2f uv, f32 center_depth,
    f32 half_texel_x, f32 half_texel_y,
    f32 depth_sigma = 0.5f) noexcept
{
    f32 sum = 0.f, weight_sum = 0.f;
    static constexpr Vec2f offsets[4] = {{-1,-1},{1,-1},{-1,1},{1,1}};
    for (auto& off : offsets) {
        Vec2f s_uv = {uv.x + off.x*half_texel_x, uv.y + off.y*half_texel_y};
        f32 s_depth = depth_at(s_uv);
        f32 diff = abs(s_depth - center_depth);
        f32 w = std::exp(-diff * diff / (2.f*depth_sigma*depth_sigma));
        sum += ao_at(s_uv) * w;
        weight_sum += w;
    }
    return weight_sum > EPS_WEIGHT ? sum / weight_sum : ao_at(uv);
}

} // namespace qs::math::render
