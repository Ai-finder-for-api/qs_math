
#pragma once
// qs_math/random/include/qs_math_random.hpp (C++20)
// Extended random: LCG, MT19937 wrapper, scrambled Sobol, low-discrepancy.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <functional>
namespace qs::math::rng {
using namespace qs::math;
// ─── LCG (Linear congruential generator) ────────────────────────────────────
struct LCG {
    u64 state;
    explicit LCG(u64 seed=12345) noexcept : state(seed) {}
    [[nodiscard]] u32 next() noexcept { state=state*6364136223846793005ULL+1442695040888963407ULL; return (u32)(state>>32); }
    [[nodiscard]] f32 next_f32() noexcept { return (f32)(next()>>8)*(1.0f/16777216.0f); }
    [[nodiscard]] f64 next_f64() noexcept { return ((u64)next()<<32|next())*(1.0/18446744073709551616.0); }
    [[nodiscard]] i32 next_range(i32 lo,i32 hi) noexcept { return lo+(i32)(next()%(u32)(hi-lo)); }
};
// ─── Stratified sampling ──────────────────────────────────────────────────────
void stratified_1d(i32 n, f32* samples, LCG& rng) noexcept;
void stratified_2d(i32 nx, i32 ny, Vec2<f32>* samples, LCG& rng) noexcept;
// ─── Latin hypercube sampling ─────────────────────────────────────────────────
void latin_hypercube(i32 n, i32 dim, f64* out, LCG& rng) noexcept;
// ─── Hammersley sequence ──────────────────────────────────────────────────────
[[nodiscard]] Vec2<f32> hammersley(u32 i, u32 n) noexcept;
void hammersley_set(u32 n, Vec2<f32>* out) noexcept;
// ─── Fibonacci hemisphere sampling ───────────────────────────────────────────
[[nodiscard]] Vec3<f32> fibonacci_sphere(u32 i, u32 n) noexcept;
void fibonacci_sphere_set(u32 n, Vec3<f32>* out) noexcept;
// ─── Rejection sampling ───────────────────────────────────────────────────────
template<typename T>
std::vector<T> rejection_sample(std::function<T(LCG&)> propose,
    std::function<bool(T)> accept, i32 n, LCG& rng) noexcept
{
    std::vector<T> result; result.reserve(n);
    while((i32)result.size()<n){auto v=propose(rng);if(accept(v))result.push_back(v);}
    return result;
}
// ─── Importance sampling ──────────────────────────────────────────────────────
[[nodiscard]] Vec3<f32> sample_cosine_hemisphere(f32 u1, f32 u2) noexcept;
[[nodiscard]] Vec3<f32> sample_uniform_sphere(f32 u1, f32 u2) noexcept;
[[nodiscard]] Vec3<f32> sample_uniform_hemisphere(f32 u1, f32 u2) noexcept;
[[nodiscard]] f32 pdf_cosine_hemisphere(Vec3<f32> dir) noexcept;
// ─── Alias method (O(1) discrete sampling) ────────────────────────────────────
struct AliasTable {
    std::vector<f32> prob;
    std::vector<i32> alias;
    void build(const f32* weights, i32 n) noexcept;
    [[nodiscard]] i32 sample(LCG& rng) const noexcept;
};
// ─── Sobol sequence (first 4 dimensions) ─────────────────────────────────────
struct SobolSeq {
    u32 n{0};
    [[nodiscard]] Vec2<f32> next_2d() noexcept;
    [[nodiscard]] std::array<f32,4> next_4d() noexcept;
private:
    u32 scramble[4]{0x5a5a5a5a,0xa5a5a5a5,0x3c3c3c3c,0xc3c3c3c3};
};
} // namespace qs::math::rng
