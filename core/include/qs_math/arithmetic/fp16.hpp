#pragma once
// qs_math/fp16.hpp — IEEE 754 half-float (binary16) with full math ops (C++20)
// No hardware dependency; software implementation with optional HW path via __fp16.

#include "scalar_types.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <compare>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// Internal bit-level conversion helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace detail_fp16 {

[[nodiscard]] inline u16 f32_to_f16_bits(f32 v) noexcept {
    u32 bits; std::memcpy(&bits, &v, 4);
    u32 sign     = (bits >> 31) & 0x1u;
    i32 exp32    = i32((bits >> 23) & 0xFFu) - 127;
    u32 mant32   = bits & 0x7FFFFFu;

    if (exp32 == 128) { // inf or nan
        return u16((sign << 15) | 0x7C00u | (mant32 ? 0x200u : 0u));
    }
    if (exp32 < -24) return u16(sign << 15); // underflow → ±0
    if (exp32 < -14) {
        // Subnormal half
        u32 mant16 = (mant32 | 0x800000u) >> u32(-14 - exp32 + 1);
        return u16((sign << 15) | mant16);
    }
    if (exp32 > 15) return u16((sign << 15) | 0x7C00u); // overflow → ±inf
    u32 exp16  = u32(exp32 + 15);
    u32 mant16 = mant32 >> 13;
    // Round to nearest even
    u32 round  = (mant32 >> 12) & 1u;
    u32 sticky = mant32 & 0xFFFu;
    if (round && (sticky || (mant16 & 1u))) ++mant16;
    if (mant16 == 0x400u) { ++exp16; mant16 = 0; }
    return u16((sign << 15) | (exp16 << 10) | (mant16 & 0x3FFu));
}

[[nodiscard]] inline f32 f16_bits_to_f32(u16 h) noexcept {
    u32 sign  = u32((h >> 15) & 0x1u);
    u32 exp16 = u32((h >> 10) & 0x1Fu);
    u32 mant16= u32(h & 0x3FFu);
    u32 bits;
    if (exp16 == 0x1Fu) { // inf/nan
        bits = (sign << 31) | 0x7F800000u | (mant16 << 13);
    } else if (exp16 == 0u) { // subnormal/zero
        if (mant16 == 0u) { bits = sign << 31; }
        else {
            u32 m = mant16; int e = -14;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            bits = (sign << 31) | (u32(e + 127) << 23) | ((m & 0x3FFu) << 13);
        }
    } else {
        bits = (sign << 31) | ((exp16 - 15u + 127u) << 23) | (mant16 << 13);
    }
    f32 r; std::memcpy(&r, &bits, 4); return r;
}

} // namespace detail_fp16

// ─────────────────────────────────────────────────────────────────────────────
// f16 — IEEE 754 binary16 half-float type
// ─────────────────────────────────────────────────────────────────────────────
struct f16 {
    u16 bits{0};

    constexpr f16() noexcept = default;
    constexpr explicit f16(u16 raw, bool) noexcept : bits(raw) {} // raw-bits ctor
    f16(f32 v) noexcept : bits(detail_fp16::f32_to_f16_bits(v)) {}
    f16(f64 v) noexcept : f16(f32(v)) {}
    explicit f16(i32 v) noexcept : f16(f32(v)) {}

    [[nodiscard]] f32 to_f32() const noexcept { return detail_fp16::f16_bits_to_f32(bits); }
    [[nodiscard]] f64 to_f64() const noexcept { return f64(to_f32()); }
    explicit operator f32() const noexcept { return to_f32(); }
    explicit operator f64() const noexcept { return to_f64(); }

    // Component accessors
    [[nodiscard]] constexpr u16 sign_bit()  const noexcept { return (bits >> 15) & 0x1u; }
    [[nodiscard]] constexpr u16 exponent()  const noexcept { return (bits >> 10) & 0x1Fu; }
    [[nodiscard]] constexpr u16 mantissa()  const noexcept { return bits & 0x3FFu; }

    // Predicates
    [[nodiscard]] bool is_nan()      const noexcept { return exponent() == 0x1Fu && mantissa() != 0; }
    [[nodiscard]] bool is_inf()      const noexcept { return exponent() == 0x1Fu && mantissa() == 0; }
    [[nodiscard]] bool is_finite()   const noexcept { return exponent() != 0x1Fu; }
    [[nodiscard]] bool is_zero()     const noexcept { return (bits & 0x7FFFu) == 0; }
    [[nodiscard]] bool is_subnormal()const noexcept { return exponent() == 0 && mantissa() != 0; }
    [[nodiscard]] bool is_negative() const noexcept { return sign_bit() != 0; }

    // Unary
    [[nodiscard]] f16 operator-()    const noexcept { return f16(u16(bits ^ 0x8000u), true); }
    [[nodiscard]] f16 operator+()    const noexcept { return *this; }

    // Comparison (via f32)
    [[nodiscard]] bool operator==(const f16& o) const noexcept {
        if (is_nan() || o.is_nan()) return false;
        if (is_zero() && o.is_zero()) return true;
        return bits == o.bits;
    }
    [[nodiscard]] std::partial_ordering operator<=>(const f16& o) const noexcept {
        if (is_nan() || o.is_nan()) return std::partial_ordering::unordered;
        return to_f32() <=> o.to_f32();
    }

    // Arithmetic (promote to f32, demote back)
    [[nodiscard]] f16 operator+(const f16& b) const noexcept { return f16(to_f32() + b.to_f32()); }
    [[nodiscard]] f16 operator-(const f16& b) const noexcept { return f16(to_f32() - b.to_f32()); }
    [[nodiscard]] f16 operator*(const f16& b) const noexcept { return f16(to_f32() * b.to_f32()); }
    [[nodiscard]] f16 operator/(const f16& b) const noexcept { return f16(to_f32() / b.to_f32()); }
    f16& operator+=(const f16& b) noexcept { *this = *this + b; return *this; }
    f16& operator-=(const f16& b) noexcept { *this = *this - b; return *this; }
    f16& operator*=(const f16& b) noexcept { *this = *this * b; return *this; }
    f16& operator/=(const f16& b) noexcept { *this = *this / b; return *this; }

    // Mixed f32
    [[nodiscard]] f32 operator+(f32 b) const noexcept { return to_f32() + b; }
    [[nodiscard]] f32 operator-(f32 b) const noexcept { return to_f32() - b; }
    [[nodiscard]] f32 operator*(f32 b) const noexcept { return to_f32() * b; }
    [[nodiscard]] f32 operator/(f32 b) const noexcept { return to_f32() / b; }

    // Constants
    [[nodiscard]] static constexpr f16 zero()     noexcept { return f16(u16(0x0000u), true); }
    [[nodiscard]] static constexpr f16 one()      noexcept { return f16(u16(0x3C00u), true); }
    [[nodiscard]] static constexpr f16 inf()      noexcept { return f16(u16(0x7C00u), true); }
    [[nodiscard]] static constexpr f16 nan()      noexcept { return f16(u16(0x7E00u), true); }
    [[nodiscard]] static constexpr f16 max_val()  noexcept { return f16(u16(0x7BFFu), true); } // 65504
    [[nodiscard]] static constexpr f16 min_normal()noexcept{ return f16(u16(0x0400u), true); } // ~6.1e-5
    [[nodiscard]] static constexpr f16 epsilon()  noexcept { return f16(u16(0x1400u), true); } // ~9.77e-4
    [[nodiscard]] static constexpr f16 pi()       noexcept { return f16(u16(0x4248u), true); } // ~3.140625
};

// ─────────────────────────────────────────────────────────────────────────────
// Math functions for f16 (promote to f32)
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] inline f16 sqrt(f16 v)  noexcept { return f16(std::sqrt(v.to_f32())); }
[[nodiscard]] inline f16 abs(f16 v)   noexcept { return f16(u16(v.bits & 0x7FFFu), true); }
[[nodiscard]] inline f16 floor(f16 v) noexcept { return f16(std::floor(v.to_f32())); }
[[nodiscard]] inline f16 ceil(f16 v)  noexcept { return f16(std::ceil(v.to_f32())); }
[[nodiscard]] inline f16 round(f16 v) noexcept { return f16(std::round(v.to_f32())); }
[[nodiscard]] inline f16 fma(f16 a, f16 b, f16 c) noexcept { return f16(std::fma(a.to_f32(), b.to_f32(), c.to_f32())); }
[[nodiscard]] inline f16 sin(f16 v)   noexcept { return f16(std::sin(v.to_f32())); }
[[nodiscard]] inline f16 cos(f16 v)   noexcept { return f16(std::cos(v.to_f32())); }
[[nodiscard]] inline f16 tan(f16 v)   noexcept { return f16(std::tan(v.to_f32())); }
[[nodiscard]] inline f16 exp(f16 v)   noexcept { return f16(std::exp(v.to_f32())); }
[[nodiscard]] inline f16 log(f16 v)   noexcept { return f16(std::log(v.to_f32())); }
[[nodiscard]] inline f16 pow(f16 a, f16 b) noexcept { return f16(std::pow(a.to_f32(), b.to_f32())); }
[[nodiscard]] inline f16 clamp(f16 v, f16 lo, f16 hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}
[[nodiscard]] inline f16 lerp(f16 a, f16 b, f16 t) noexcept {
    return f16(a.to_f32() * (1.f - t.to_f32()) + b.to_f32() * t.to_f32());
}

// Bulk convert arrays f32 → f16 and back (useful for texture uploads)
inline void f32_to_f16_bulk(const f32* src, f16* dst, u32 count) noexcept {
    for (u32 i = 0; i < count; ++i) dst[i] = f16(src[i]);
}
inline void f16_to_f32_bulk(const f16* src, f32* dst, u32 count) noexcept {
    for (u32 i = 0; i < count; ++i) dst[i] = src[i].to_f32();
}

// ULP distance between two f16 values
[[nodiscard]] inline u32 f16_ulp_distance(f16 a, f16 b) noexcept {
    if (a.is_nan() || b.is_nan()) return 0xFFFFu;
    i32 ai = i32(a.bits), bi = i32(b.bits);
    if (ai < 0) ai = i32(0x8000) - ai;
    if (bi < 0) bi = i32(0x8000) - bi;
    return u32(ai > bi ? ai - bi : bi - ai);
}

// UDLs
namespace fp16_literals {
    [[nodiscard]] inline f16 operator""_h(long double v) noexcept { return f16(f32(v)); }
}

} // namespace qs
