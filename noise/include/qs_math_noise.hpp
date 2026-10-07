
#pragma once
// qs_math/noise/include/qs_math_noise.hpp (C++20)
// Extended noise: domain warping, cellular, value, curl, blue noise.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <array>
#include <vector>
namespace qs::math::noise {
using namespace qs::math;
// ─── Value noise ──────────────────────────────────────────────────────────────
[[nodiscard]] f32 value_noise_2d(f32 x, f32 y, u32 seed=0) noexcept;
[[nodiscard]] f32 value_noise_3d(f32 x, f32 y, f32 z, u32 seed=0) noexcept;
// ─── Cellular (Worley) noise ──────────────────────────────────────────────────
struct CellResult { f32 F1,F2; Vec2<f32> cell_center; };
[[nodiscard]] CellResult worley_2d(f32 x, f32 y, u32 seed=0) noexcept;
[[nodiscard]] f32 worley_3d_f1(f32 x, f32 y, f32 z, u32 seed=0) noexcept;
// ─── Domain warped Perlin (Iq-style) ─────────────────────────────────────────
[[nodiscard]] f32 fbm_warped(f32 x, f32 y, i32 octaves=6, f32 warp=1.0f) noexcept;
// ─── Curl noise (divergence-free) ─────────────────────────────────────────────
[[nodiscard]] Vec2<f32> curl_noise_2d(f32 x, f32 y, f32 eps=0.001f) noexcept;
[[nodiscard]] Vec3<f32> curl_noise_3d(f32 x, f32 y, f32 z, f32 eps=0.001f) noexcept;
// ─── Blue noise (Mitchell's best-candidate) ───────────────────────────────────
std::vector<Vec2<f32>> blue_noise_2d(i32 n, f32 width, f32 height, u32 seed=42) noexcept;
// ─── Spectral synthesis (random phase + 1/f^β spectrum) ───────────────────────
std::vector<f32> spectral_1d(i32 n, f32 beta=1.0f, u32 seed=0) noexcept;
// ─── FBM with analytical derivative ──────────────────────────────────────────
struct FBMResult { f32 value; Vec2<f32> derivative; };
[[nodiscard]] FBMResult fbm_deriv_2d(f32 x, f32 y,
    i32 octaves=6, f32 lacunarity=2.0f, f32 gain=0.5f) noexcept;
} // namespace qs::math::noise
