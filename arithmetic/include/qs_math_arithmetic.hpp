#pragma once
// qs_math/arithmetic/include/qs_math_arithmetic.hpp (C++20)
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cstdint>
#include <cmath>
#include <array>
#include <cassert>
namespace qs::math::arithmetic {
using namespace qs::math;
// ─── Fixed-point Q16.16 ───────────────────────────────────────────────────────
struct Fixed16 {
    i32 raw{0};
    static constexpr i32 FRAC = 16;
    static constexpr i32 SCALE = 1 << FRAC;
    [[nodiscard]] static Fixed16 from_f32(f32 v) noexcept { return {(i32)(v*SCALE)}; }
    [[nodiscard]] f32 to_f32() const noexcept { return (f32)raw / SCALE; }
    [[nodiscard]] Fixed16 operator+(Fixed16 o) const noexcept { return {raw+o.raw}; }
    [[nodiscard]] Fixed16 operator-(Fixed16 o) const noexcept { return {raw-o.raw}; }
    [[nodiscard]] Fixed16 operator*(Fixed16 o) const noexcept { return {(i32)(((i64)raw*o.raw)>>FRAC)}; }
    [[nodiscard]] Fixed16 operator/(Fixed16 o) const noexcept { return {(i32)(((i64)raw<<FRAC)/o.raw)}; }
    [[nodiscard]] bool operator<(Fixed16 o) const noexcept { return raw<o.raw; }
    [[nodiscard]] bool operator==(Fixed16 o) const noexcept { return raw==o.raw; }
};
// ─── Q8.24 (higher precision) ─────────────────────────────────────────────────
struct Fixed24 {
    i32 raw{0};
    static constexpr i32 FRAC=24, SCALE=1<<24;
    [[nodiscard]] static Fixed24 from_f64(f64 v) noexcept { return {(i32)(v*SCALE)}; }
    [[nodiscard]] f64 to_f64() const noexcept { return (f64)raw/SCALE; }
    [[nodiscard]] Fixed24 operator*(Fixed24 o) const noexcept { return {(i32)(((i64)raw*o.raw)>>FRAC)}; }
};
// ─── Saturating integer arithmetic ────────────────────────────────────────────
[[nodiscard]] inline i32 sat_add_i32(i32 a, i32 b) noexcept {
    const i64 r=(i64)a+b;
    if (r>INT32_MAX) return INT32_MAX;
    if (r<INT32_MIN) return INT32_MIN;
    return (i32)r;
}
[[nodiscard]] inline i32 sat_mul_i32(i32 a, i32 b) noexcept {
    const i64 r=(i64)a*b;
    if (r>INT32_MAX) return INT32_MAX;
    if (r<INT32_MIN) return INT32_MIN;
    return (i32)r;
}
// ─── Bitwise utilities ────────────────────────────────────────────────────────
[[nodiscard]] inline u32 popcount(u32 v) noexcept { return (u32)__builtin_popcount(v); }
[[nodiscard]] inline u32 clz(u32 v) noexcept { return v?__builtin_clz(v):32; }
[[nodiscard]] inline u32 ctz(u32 v) noexcept { return v?__builtin_ctz(v):32; }
[[nodiscard]] inline u32 next_pow2(u32 v) noexcept { --v;v|=v>>1;v|=v>>2;v|=v>>4;v|=v>>8;v|=v>>16;return v+1; }
[[nodiscard]] inline bool is_pow2(u32 v) noexcept { return v&&!(v&(v-1)); }
[[nodiscard]] inline u32 bswap32(u32 v) noexcept { return __builtin_bswap32(v); }
[[nodiscard]] inline u64 bswap64(u64 v) noexcept { return __builtin_bswap64(v); }
[[nodiscard]] inline u32 bit_reverse(u32 v, i32 bits) noexcept {
    u32 r=0; for(i32 i=0;i<bits;++i){r=(r<<1)|(v&1);v>>=1;} return r;
}
// ─── Big integer (128-bit via compiler) ───────────────────────────────────────
#ifdef __SIZEOF_INT128__
using u128 = unsigned __int128;
using i128 = __int128;
[[nodiscard]] inline u128 mul128(u64 a, u64 b) noexcept { return (u128)a*b; }
[[nodiscard]] inline u64  mul128_hi(u64 a, u64 b) noexcept { return (u64)((u128)a*b>>64); }
#endif
// ─── Floating-point utilities ─────────────────────────────────────────────────
[[nodiscard]] inline i32 float_sign_bit(f32 v) noexcept {
    u32 bits; __builtin_memcpy(&bits,&v,4); return (i32)(bits>>31);
}
[[nodiscard]] inline i32 float_exponent(f32 v) noexcept {
    u32 bits; __builtin_memcpy(&bits,&v,4); return (i32)((bits>>23)&0xff)-127;
}
[[nodiscard]] inline u32 float_mantissa(f32 v) noexcept {
    u32 bits; __builtin_memcpy(&bits,&v,4); return bits&0x7fffff;
}
[[nodiscard]] inline bool is_finite(f32 v) noexcept { return std::isfinite(v); }
[[nodiscard]] inline bool is_nan(f32 v) noexcept { return std::isnan(v); }
// Ulp distance between two floats
[[nodiscard]] inline i32 ulp_dist(f32 a, f32 b) noexcept {
    i32 ia,ib; __builtin_memcpy(&ia,&a,4); __builtin_memcpy(&ib,&b,4);
    if (ia<0) ia=0x80000000-ia; if (ib<0) ib=0x80000000-ib;
    return std::abs(ia-ib);
}
// ─── Multi-precision float (double-double) ────────────────────────────────────
struct DDFloat { f64 hi{0}, lo{0};
    [[nodiscard]] static DDFloat from_f64(f64 v) noexcept { return {v,0}; }
    [[nodiscard]] DDFloat operator+(DDFloat o) const noexcept {
        f64 s=hi+o.hi, e=(std::abs(hi)>=std::abs(o.hi))?(hi-s+o.hi):(o.hi-s+hi);
        e+=lo+o.lo; const f64 r=s+e; return {r,e-(r-s)};
    }
    [[nodiscard]] DDFloat operator*(DDFloat o) const noexcept;
    [[nodiscard]] f64 to_f64() const noexcept { return hi+lo; }
};
} // namespace qs::math::arithmetic
