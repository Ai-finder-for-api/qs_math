#pragma once
// qs_math/scalar_math.hpp — constexpr scalar math utilities
#include "scalar_types.hpp"
#include "constants.hpp"
#include <cmath>
#include <bit>

namespace qs::math {

// abs
template<Scalar T> [[nodiscard]] constexpr T abs(T v) noexcept { return v < T(0) ? -v : v; }

// sign
template<Scalar T> [[nodiscard]] constexpr T sign(T v) noexcept {
    return v > T(0) ? T(1) : (v < T(0) ? T(-1) : T(0));
}

// clamp
template<Scalar T> [[nodiscard]] constexpr T clamp(T v, T lo, T hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}
template<FloatScalar T> [[nodiscard]] constexpr T saturate(T v) noexcept { return clamp(v, T(0), T(1)); }

// min / max
template<Scalar T> [[nodiscard]] constexpr T min(T a, T b) noexcept { return a < b ? a : b; }
template<Scalar T> [[nodiscard]] constexpr T max(T a, T b) noexcept { return a > b ? a : b; }
template<Scalar T> [[nodiscard]] constexpr T min3(T a,T b,T c) noexcept { return min(min(a,b),c); }
template<Scalar T> [[nodiscard]] constexpr T max3(T a,T b,T c) noexcept { return max(max(a,b),c); }

// lerp
template<FloatScalar T> [[nodiscard]] constexpr T lerp(T a, T b, T t) noexcept { return a + (b-a)*t; }
template<FloatScalar T> [[nodiscard]] constexpr T lerp_clamped(T a,T b,T t) noexcept { return lerp(a,b,saturate(t)); }
template<FloatScalar T> [[nodiscard]] constexpr T inverse_lerp(T a,T b,T v) noexcept { return (v-a)/(b-a); }
template<FloatScalar T> [[nodiscard]] constexpr T remap(T in_lo,T in_hi,T out_lo,T out_hi,T v) noexcept {
    return lerp(out_lo, out_hi, inverse_lerp(in_lo, in_hi, v));
}

// step / smoothstep / smootherstep
template<FloatScalar T> [[nodiscard]] constexpr T step(T edge, T v) noexcept { return v < edge ? T(0) : T(1); }
template<FloatScalar T> [[nodiscard]] constexpr T smoothstep(T lo, T hi, T v) noexcept {
    T t = saturate((v-lo)/(hi-lo));
    return t*t*(T(3)-T(2)*t);
}
template<FloatScalar T> [[nodiscard]] constexpr T smootherstep(T lo, T hi, T v) noexcept {
    T t = saturate((v-lo)/(hi-lo));
    return t*t*t*(t*(t*T(6)-T(15))+T(10));
}

// wrap / repeat
template<FloatScalar T> [[nodiscard]] constexpr T wrap(T v, T lo, T hi) noexcept {
    T range = hi - lo;
    if (range <= T(0)) return lo;
    T result = v - range * static_cast<T>(static_cast<i64>((v-lo)/range));
    if (result < lo) result += range;
    return result;
}
template<IntScalar T> [[nodiscard]] constexpr T wrap(T v, T lo, T hi) noexcept {
    T range = hi - lo;
    return lo + ((v-lo) % range + range) % range;
}

// floor / ceil / round / frac — delegate to std but wrap for discoverability
template<FloatScalar T> [[nodiscard]] inline T floor(T v) noexcept  { return std::floor(v); }
template<FloatScalar T> [[nodiscard]] inline T ceil(T v)  noexcept  { return std::ceil(v); }
template<FloatScalar T> [[nodiscard]] inline T round(T v) noexcept  { return std::round(v); }
template<FloatScalar T> [[nodiscard]] inline T trunc(T v) noexcept  { return std::trunc(v); }
template<FloatScalar T> [[nodiscard]] inline T frac(T v)  noexcept  { return v - std::floor(v); }

// sqrt / rsqrt / cbrt
template<FloatScalar T> [[nodiscard]] inline T sqrt(T v)   noexcept { return std::sqrt(v); }
template<FloatScalar T> [[nodiscard]] inline T rsqrt(T v)  noexcept { return T(1)/std::sqrt(v); }
template<FloatScalar T> [[nodiscard]] inline T cbrt(T v)   noexcept { return std::cbrt(v); }

// Fast inverse sqrt (Q_rsqrt) — f32 only
[[nodiscard]] inline f32 fast_rsqrt(f32 v) noexcept {
    f32 half = 0.5f * v;
    u32 i; __builtin_memcpy(&i, &v, 4);
    i = 0x5f3759df - (i >> 1);
    __builtin_memcpy(&v, &i, 4);
    return v * (1.5f - half * v * v);
}

// pow / log / exp
template<FloatScalar T> [[nodiscard]] inline T pow(T b, T e)  noexcept { return std::pow(b,e); }
template<FloatScalar T> [[nodiscard]] inline T log(T v)       noexcept { return std::log(v); }
template<FloatScalar T> [[nodiscard]] inline T log2(T v)      noexcept { return std::log2(v); }
template<FloatScalar T> [[nodiscard]] inline T log10(T v)     noexcept { return std::log10(v); }
template<FloatScalar T> [[nodiscard]] inline T exp(T v)       noexcept { return std::exp(v); }
template<FloatScalar T> [[nodiscard]] inline T exp2(T v)      noexcept { return std::exp2(v); }

// mod
template<FloatScalar T> [[nodiscard]] inline T mod(T a, T b)  noexcept { return std::fmod(a,b); }
template<IntScalar T>   [[nodiscard]] constexpr T mod(T a,T b) noexcept { return a % b; }

// Float comparison
template<FloatScalar T>
[[nodiscard]] constexpr bool nearly_equal(T a, T b, T eps=EPS_F32) noexcept {
    T diff = abs(a-b);
    if (diff <= eps) return true;
    return diff <= eps * max(abs(a), abs(b));
}

// ULP comparison
[[nodiscard]] inline bool ulp_equal(f32 a, f32 b, i32 max_ulp=4) noexcept {
    u32 ai, bi; __builtin_memcpy(&ai,&a,4); __builtin_memcpy(&bi,&b,4);
    if ((ai>>31)!=(bi>>31)) return a==b;
    i32 diff = static_cast<i32>(ai) - static_cast<i32>(bi);
    return (diff < 0 ? -diff : diff) <= max_ulp;
}

// is_nan / is_inf / is_finite
template<FloatScalar T> [[nodiscard]] inline bool is_nan(T v)    noexcept { return std::isnan(v); }
template<FloatScalar T> [[nodiscard]] inline bool is_inf(T v)    noexcept { return std::isinf(v); }
template<FloatScalar T> [[nodiscard]] inline bool is_finite(T v) noexcept { return std::isfinite(v); }

// Integer helpers
[[nodiscard]] constexpr bool is_power_of_two(u64 v) noexcept { return v && !(v & (v-1)); }
[[nodiscard]] constexpr u64  next_power_of_two(u64 v) noexcept {
    if (v == 0) return 1;
    --v; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; v|=v>>32; return v+1;
}
[[nodiscard]] constexpr u64  align_up(u64 v, u64 align) noexcept { return (v+align-1) & ~(align-1); }
[[nodiscard]] constexpr u64  align_down(u64 v, u64 align) noexcept { return v & ~(align-1); }
[[nodiscard]] constexpr i32  popcount(u64 v) noexcept { return std::popcount(v); }
[[nodiscard]] constexpr i32  count_leading_zeros(u64 v) noexcept { return v ? std::countl_zero(v) : 64; }
[[nodiscard]] constexpr i32  count_trailing_zeros(u64 v) noexcept { return v ? std::countr_zero(v) : 64; }

} // namespace qs::math
