#pragma once
// qs_math/scalar_types.hpp — Scalar aliases, numeric limits, type traits, concepts
#include "../../../../include/qs_types.h"
#include <limits>
#include <type_traits>
#include <concepts>

namespace qs::math {

using f32 = qs_f32; using f64 = qs_f64;
using i8  = qs_i8;  using i16 = qs_i16; using i32 = qs_i32; using i64 = qs_i64;
using u8  = qs_u8;  using u16 = qs_u16; using u32 = qs_u32; using u64 = qs_u64;

// Concepts
template<typename T> concept Scalar      = std::is_arithmetic_v<T>;
template<typename T> concept FloatScalar = std::is_floating_point_v<T>;
template<typename T> concept IntScalar   = std::is_integral_v<T>;
template<typename T> concept SignedScalar= std::is_signed_v<T>;
template<typename T> concept MathElement = FloatScalar<T> || IntScalar<T>;

template<typename T> using promote_float_t  = std::conditional_t<std::is_same_v<T,f64>,f64,f32>;
template<typename A, typename B> using common_scalar_t = std::common_type_t<A,B>;

// Numeric limits
template<typename T> struct NumericLimits {
    static constexpr T min_val  = std::numeric_limits<T>::lowest();
    static constexpr T max_val  = std::numeric_limits<T>::max();
    static constexpr T eps      = std::numeric_limits<T>::epsilon();
    static constexpr T inf      = std::numeric_limits<T>::infinity();
    static constexpr bool is_fp = std::is_floating_point_v<T>;
    static constexpr bool is_int= std::is_integral_v<T>;
    static constexpr int  dig   = std::numeric_limits<T>::digits10;
};

// Math traits
template<typename T> struct MathTraits {
    static constexpr T zero=T(0), one=T(1), two=T(2), half=T(0.5), neg_one=T(-1);
    static constexpr T eps = std::numeric_limits<T>::epsilon();
    static constexpr bool is_fp = std::is_floating_point_v<T>;
};

// Epsilon constants
inline constexpr f32 EPS_F32         = 1.0e-6f;
inline constexpr f32 EPS_F32_LOOSE   = 1.0e-4f;
inline constexpr f32 EPS_F32_TIGHT   = 1.0e-7f;
inline constexpr f64 EPS_F64         = 1.0e-12;
inline constexpr f64 EPS_F64_LOOSE   = 1.0e-9;
inline constexpr f32 EPS_LENGTH      = 1.0e-8f;
inline constexpr f32 EPS_NORMAL      = 1.0e-6f;
inline constexpr f32 EPS_ANGLE       = 1.0e-5f;
inline constexpr f32 EPS_DOT         = 1.0e-6f;
inline constexpr f32 EPS_DETERMINANT = 1.0e-9f;
inline constexpr f32 EPS_INTERSECT   = 1.0e-8f;
inline constexpr f32 EPS_WEIGHT      = 1.0e-5f;
inline constexpr f32 EPS_ULP         = 1.192093e-7f; // 2^-23


// ─── Type-safe index wrapper ───────────────────────────────────────────────
template<typename Tag, typename T = u32>
struct TypedIndex {
    T value{};
    constexpr TypedIndex() noexcept = default;
    constexpr explicit TypedIndex(T v) noexcept : value(v) {}
    [[nodiscard]] constexpr bool valid() const noexcept { return value != T(-1); }
    static constexpr TypedIndex invalid() noexcept { return TypedIndex{T(-1)}; }
    constexpr bool operator==(TypedIndex o) const noexcept { return value==o.value; }
    constexpr bool operator!=(TypedIndex o) const noexcept { return value!=o.value; }
    constexpr bool operator< (TypedIndex o) const noexcept { return value< o.value; }
};

// ─── Half-open range ──────────────────────────────────────────────────────
template<typename T>
struct Range {
    T lo{}, hi{};
    constexpr Range() noexcept = default;
    constexpr Range(T l, T h) noexcept : lo(l), hi(h) {}
    [[nodiscard]] constexpr bool   contains(T v)    const noexcept { return v>=lo && v<hi; }
    [[nodiscard]] constexpr T      length()          const noexcept { return hi-lo; }
    [[nodiscard]] constexpr T      clamp(T v)        const noexcept { return v<lo?lo:v>=hi?hi:v; }
    [[nodiscard]] constexpr T      lerp(T t)         const noexcept { return lo+t*(hi-lo); }
    [[nodiscard]] constexpr T      remap(T v)        const noexcept { return (v-lo)/(hi-lo); }
    [[nodiscard]] constexpr bool   empty()           const noexcept { return lo>=hi; }
    [[nodiscard]] constexpr Range  intersect(Range o) const noexcept {
        return {lo>o.lo?lo:o.lo, hi<o.hi?hi:o.hi};
    }
};
using Rangef32 = Range<f32>;
using Rangef64 = Range<f64>;
using Rangei32 = Range<i32>;

// ─── Signed/unsigned cast helpers ─────────────────────────────────────────
[[nodiscard]] constexpr i32 as_signed  (u32 v) noexcept { return static_cast<i32>(v); }
[[nodiscard]] constexpr u32 as_unsigned(i32 v) noexcept { return static_cast<u32>(v); }
[[nodiscard]] constexpr i64 as_signed  (u64 v) noexcept { return static_cast<i64>(v); }
[[nodiscard]] constexpr u64 as_unsigned(i64 v) noexcept { return static_cast<u64>(v); }

// ─── Quantize: map float to N uniform levels ──────────────────────────────
[[nodiscard]] constexpr u32 quantize_u8 (f32 v) noexcept { return u32(clamp01(v)*255.0f+0.5f) & 0xFF; }
[[nodiscard]] constexpr u32 quantize_u16(f32 v) noexcept { return u32(clamp01(v)*65535.0f+0.5f) & 0xFFFF; }
[[nodiscard]] constexpr f32 dequantize_u8 (u32 v) noexcept { return f32(v&0xFF)/255.0f; }
[[nodiscard]] constexpr f32 dequantize_u16(u32 v) noexcept { return f32(v&0xFFFF)/65535.0f; }

} // namespace qs::math
