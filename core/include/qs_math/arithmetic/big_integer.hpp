#pragma once
// qs_math/big_integer.hpp — Arbitrary-precision integer helpers (C++20)
// Part of qs_math/core — no allocations beyond std::vector internals

#include "scalar_types.hpp"
#include <vector>
#include <string>
#include <string_view>
#include <compare>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <cstring>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// BigUInt — unsigned arbitrary-precision integer (base 2^32 limbs, little-endian)
// ─────────────────────────────────────────────────────────────────────────────
struct BigUInt {
    using Limb  = u32;
    using DLimb = u64;
    static constexpr u32 BASE_BITS = 32u;

    std::vector<Limb> d; // d[0] = least significant limb

    BigUInt()                    { d.push_back(0); }
    explicit BigUInt(u64 v)      { d.push_back(Limb(v)); Limb hi = Limb(v >> 32); if (hi) d.push_back(hi); }
    explicit BigUInt(std::string_view s, int base = 10);

    // Normalization: remove leading zero limbs (keep at least one)
    void trim() { while (d.size() > 1 && d.back() == 0) d.pop_back(); }

    [[nodiscard]] bool is_zero() const { return d.size() == 1 && d[0] == 0; }
    [[nodiscard]] u32  bit_length() const;
    [[nodiscard]] bool bit(u32 i)  const { return (d[i / 32] >> (i % 32)) & 1u; }

    // Comparison
    [[nodiscard]] std::strong_ordering operator<=>(const BigUInt& o) const;
    [[nodiscard]] bool operator==(const BigUInt& o)  const { return d == o.d; }

    // Arithmetic
    [[nodiscard]] BigUInt operator+(const BigUInt& b) const;
    [[nodiscard]] BigUInt operator-(const BigUInt& b) const; // assumes *this >= b
    [[nodiscard]] BigUInt operator*(const BigUInt& b) const;
    [[nodiscard]] BigUInt operator/(const BigUInt& b) const;
    [[nodiscard]] BigUInt operator%(const BigUInt& b) const;
    BigUInt& operator+=(const BigUInt& b) { *this = *this + b; return *this; }
    BigUInt& operator-=(const BigUInt& b) { *this = *this - b; return *this; }
    BigUInt& operator*=(const BigUInt& b) { *this = *this * b; return *this; }
    BigUInt& operator/=(const BigUInt& b) { *this = *this / b; return *this; }
    BigUInt& operator%=(const BigUInt& b) { *this = *this % b; return *this; }

    // Shifts
    [[nodiscard]] BigUInt operator<<(u32 n) const;
    [[nodiscard]] BigUInt operator>>(u32 n) const;
    BigUInt& operator<<=(u32 n) { *this = *this << n; return *this; }
    BigUInt& operator>>=(u32 n) { *this = *this >> n; return *this; }

    // Bitwise
    [[nodiscard]] BigUInt operator&(const BigUInt& b) const;
    [[nodiscard]] BigUInt operator|(const BigUInt& b) const;
    [[nodiscard]] BigUInt operator^(const BigUInt& b) const;

    // Scalar ops (fast path)
    [[nodiscard]] BigUInt mul_limb(Limb v) const;
    [[nodiscard]] Limb    div_limb(Limb v); // divides in-place, returns remainder

    // Conversion
    [[nodiscard]] u64         to_u64()    const { return u64(d[0]) | (d.size() > 1 ? u64(d[1]) << 32 : 0); }
    [[nodiscard]] std::string to_string(int base = 10) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// BigInt — signed arbitrary-precision integer
// ─────────────────────────────────────────────────────────────────────────────
struct BigInt {
    BigUInt mag;
    bool    neg{false};

    BigInt() = default;
    explicit BigInt(i64 v) : mag(u64(v < 0 ? -v : v)), neg(v < 0) {}
    explicit BigInt(BigUInt m, bool n = false) : mag(std::move(m)), neg(n) {}
    explicit BigInt(std::string_view s);

    void normalize() { if (mag.is_zero()) neg = false; }

    [[nodiscard]] bool is_zero()    const { return mag.is_zero(); }
    [[nodiscard]] bool is_negative()const { return neg; }
    [[nodiscard]] BigInt abs()       const { return BigInt{mag, false}; }
    [[nodiscard]] BigInt negate()    const { BigInt r = *this; r.neg = !r.neg; r.normalize(); return r; }

    // Comparison
    [[nodiscard]] std::strong_ordering operator<=>(const BigInt& o) const;
    [[nodiscard]] bool operator==(const BigInt& o) const { return neg == o.neg && mag == o.mag; }

    // Arithmetic
    [[nodiscard]] BigInt operator+(const BigInt& b) const;
    [[nodiscard]] BigInt operator-(const BigInt& b) const;
    [[nodiscard]] BigInt operator*(const BigInt& b) const;
    [[nodiscard]] BigInt operator/(const BigInt& b) const;
    [[nodiscard]] BigInt operator%(const BigInt& b) const;
    [[nodiscard]] BigInt operator-()                const { return negate(); }
    BigInt& operator+=(const BigInt& b) { *this = *this + b; return *this; }
    BigInt& operator-=(const BigInt& b) { *this = *this - b; return *this; }
    BigInt& operator*=(const BigInt& b) { *this = *this * b; return *this; }
    BigInt& operator/=(const BigInt& b) { *this = *this / b; return *this; }

    [[nodiscard]] std::string to_string(int base = 10) const;
    [[nodiscard]] i64         to_i64()  const { return neg ? -i64(mag.to_u64()) : i64(mag.to_u64()); }

    // Shifts (arithmetic)
    [[nodiscard]] BigInt operator<<(u32 n) const { return BigInt{mag << n, neg}; }
    [[nodiscard]] BigInt operator>>(u32 n) const { return BigInt{mag >> n, neg}; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Number-theoretic helpers on BigUInt
// ─────────────────────────────────────────────────────────────────────────────

/// Greatest common divisor (binary GCD)
[[nodiscard]] inline BigUInt gcd(BigUInt a, BigUInt b) {
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;
    u32 shift = 0;
    while (((a.d[0] | b.d[0]) & 1u) == 0) { a >>= 1; b >>= 1; ++shift; }
    while ((a.d[0] & 1u) == 0) a >>= 1;
    do {
        while ((b.d[0] & 1u) == 0) b >>= 1;
        if ((a <=> b) == std::strong_ordering::greater) std::swap(a, b);
        b -= a;
    } while (!b.is_zero());
    return a << shift;
}

/// Least common multiple
[[nodiscard]] inline BigUInt lcm(const BigUInt& a, const BigUInt& b) {
    return (a / gcd(a, b)) * b;
}

/// Modular exponentiation: base^exp mod m (square-and-multiply)
[[nodiscard]] inline BigUInt pow_mod(BigUInt base, BigUInt exp, const BigUInt& m) {
    BigUInt result{1u};
    base %= m;
    while (!exp.is_zero()) {
        if (exp.d[0] & 1u) result = (result * base) % m;
        exp >>= 1;
        base = (base * base) % m;
    }
    return result;
}

/// Integer square root (Newton's method)
[[nodiscard]] inline BigUInt isqrt(const BigUInt& n) {
    if (n.is_zero()) return BigUInt{0u};
    u32 bits = n.bit_length();
    BigUInt x = n >> ((bits - 1) / 2);
    while (true) {
        BigUInt x1 = (x + n / x) >> 1;
        if (!((x1 <=> x) == std::strong_ordering::less)) break;
        x = x1;
    }
    return x;
}

/// Factorial n!
[[nodiscard]] inline BigUInt factorial(u32 n) {
    BigUInt r{1u};
    for (u32 i = 2; i <= n; ++i) r = r.mul_limb(i);
    return r;
}

/// Binomial coefficient C(n,k)
[[nodiscard]] inline BigUInt binomial(u32 n, u32 k) {
    if (k > n) return BigUInt{0u};
    if (k > n - k) k = n - k;
    BigUInt r{1u};
    for (u32 i = 0; i < k; ++i) {
        r *= BigUInt{u64(n - i)};
        r /= BigUInt{u64(i + 1)};
    }
    return r;
}

/// Fibonacci number (fast doubling)
[[nodiscard]] inline BigUInt fibonacci(u32 n) {
    BigUInt a{0u}, b{1u};
    for (i32 i = i32(__builtin_clz(1) - __builtin_clz(n < 1 ? 1 : n)); i >= 0; --i) {
        BigUInt c = a * (b * BigUInt{2u} - a);
        BigUInt d = a * a + b * b;
        if ((n >> u32(i)) & 1u) { a = d; b = c + d; }
        else                    { a = c; b = d; }
    }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementations (inline for header-only)
// ─────────────────────────────────────────────────────────────────────────────

inline u32 BigUInt::bit_length() const {
    if (is_zero()) return 0u;
    u32 top = u32((d.size() - 1) * BASE_BITS);
    Limb hi = d.back();
    top += 32u - u32(__builtin_clz(hi));
    return top;
}

inline std::strong_ordering BigUInt::operator<=>(const BigUInt& o) const {
    if (d.size() != o.d.size()) return d.size() <=> o.d.size();
    for (i32 i = i32(d.size()) - 1; i >= 0; --i) {
        if (d[u32(i)] != o.d[u32(i)]) return d[u32(i)] <=> o.d[u32(i)];
    }
    return std::strong_ordering::equal;
}

inline BigUInt BigUInt::operator+(const BigUInt& b) const {
    BigUInt r;
    r.d.resize(std::max(d.size(), b.d.size()) + 1, 0);
    DLimb carry = 0;
    for (u32 i = 0; i < u32(r.d.size() - 1); ++i) {
        DLimb av = i < d.size()   ? d[i]   : 0;
        DLimb bv = i < b.d.size() ? b.d[i] : 0;
        DLimb s  = av + bv + carry;
        r.d[i]   = Limb(s);
        carry    = s >> 32;
    }
    r.d.back() = Limb(carry);
    r.trim();
    return r;
}

inline BigUInt BigUInt::operator-(const BigUInt& b) const {
    assert((*this <=> b) != std::strong_ordering::less);
    BigUInt r;
    r.d.resize(d.size(), 0);
    i64 borrow = 0;
    for (u32 i = 0; i < u32(d.size()); ++i) {
        i64 bv = i < b.d.size() ? i64(b.d[i]) : 0;
        i64 s  = i64(d[i]) - bv - borrow;
        if (s < 0) { s += (i64(1) << 32); borrow = 1; } else borrow = 0;
        r.d[i] = Limb(s);
    }
    r.trim();
    return r;
}

inline BigUInt BigUInt::operator*(const BigUInt& b) const {
    BigUInt r;
    r.d.assign(d.size() + b.d.size(), 0);
    for (u32 i = 0; i < u32(d.size()); ++i) {
        DLimb carry = 0;
        for (u32 j = 0; j < u32(b.d.size()); ++j) {
            DLimb cur = DLimb(d[i]) * b.d[j] + r.d[i+j] + carry;
            r.d[i+j] = Limb(cur);
            carry    = cur >> 32;
        }
        r.d[i + b.d.size()] += Limb(carry);
    }
    r.trim();
    return r;
}

// Simple long division — sets *this to quotient, returns remainder
inline void big_divmod(const BigUInt& num, const BigUInt& den, BigUInt& quot, BigUInt& rem) {
    if (den.is_zero()) throw std::domain_error("BigUInt: division by zero");
    quot = BigUInt{0u}; rem = BigUInt{0u};
    for (i32 i = i32(num.bit_length()) - 1; i >= 0; --i) {
        rem <<= 1;
        if (num.bit(u32(i))) rem.d[0] |= 1u;
        if (!((rem <=> den) == std::strong_ordering::less)) {
            rem -= den;
            // set bit i in quot
            while (quot.d.size() <= u32(i)/32) quot.d.push_back(0);
            quot.d[u32(i)/32] |= 1u << (u32(i)%32);
        }
    }
    quot.trim(); rem.trim();
}

inline BigUInt BigUInt::operator/(const BigUInt& b) const { BigUInt q,r; big_divmod(*this,b,q,r); return q; }
inline BigUInt BigUInt::operator%(const BigUInt& b) const { BigUInt q,r; big_divmod(*this,b,q,r); return r; }

inline BigUInt BigUInt::operator<<(u32 n) const {
    if (is_zero() || n == 0) return *this;
    u32 limb_shift = n / 32, bit_shift = n % 32;
    BigUInt r; r.d.assign(d.size() + limb_shift + 1, 0);
    for (u32 i = 0; i < u32(d.size()); ++i) {
        r.d[i + limb_shift] |= d[i] << bit_shift;
        if (bit_shift && i + limb_shift + 1 < u32(r.d.size()))
            r.d[i + limb_shift + 1] |= d[i] >> (32 - bit_shift);
    }
    r.trim(); return r;
}

inline BigUInt BigUInt::operator>>(u32 n) const {
    u32 ls = n/32, bs = n%32;
    if (ls >= u32(d.size())) return BigUInt{0u};
    BigUInt r; r.d.resize(d.size() - ls, 0);
    for (u32 i = 0; i < u32(r.d.size()); ++i) {
        r.d[i] = d[i + ls] >> bs;
        if (bs && i + ls + 1 < u32(d.size()))
            r.d[i] |= d[i + ls + 1] << (32 - bs);
    }
    r.trim(); return r;
}

inline BigUInt BigUInt::operator&(const BigUInt& b) const {
    BigUInt r; r.d.resize(std::min(d.size(), b.d.size()));
    for (u32 i = 0; i < u32(r.d.size()); ++i) r.d[i] = d[i] & b.d[i];
    r.trim(); return r;
}
inline BigUInt BigUInt::operator|(const BigUInt& b) const {
    BigUInt r; r.d.resize(std::max(d.size(), b.d.size()), 0);
    for (u32 i = 0; i < u32(r.d.size()); ++i) {
        r.d[i] = (i < d.size() ? d[i] : 0u) | (i < b.d.size() ? b.d[i] : 0u);
    }
    r.trim(); return r;
}
inline BigUInt BigUInt::operator^(const BigUInt& b) const {
    BigUInt r; r.d.resize(std::max(d.size(), b.d.size()), 0);
    for (u32 i = 0; i < u32(r.d.size()); ++i) {
        r.d[i] = (i < d.size() ? d[i] : 0u) ^ (i < b.d.size() ? b.d[i] : 0u);
    }
    r.trim(); return r;
}

inline BigUInt BigUInt::mul_limb(Limb v) const {
    BigUInt r; r.d.resize(d.size() + 1, 0);
    DLimb carry = 0;
    for (u32 i = 0; i < u32(d.size()); ++i) {
        DLimb cur = DLimb(d[i]) * v + carry;
        r.d[i] = Limb(cur); carry = cur >> 32;
    }
    r.d[d.size()] = Limb(carry);
    r.trim(); return r;
}

inline BigUInt::Limb BigUInt::div_limb(Limb v) {
    DLimb rem = 0;
    for (i32 i = i32(d.size()) - 1; i >= 0; --i) {
        DLimb cur = (rem << 32) | d[u32(i)];
        d[u32(i)] = Limb(cur / v);
        rem       = cur % v;
    }
    trim();
    return Limb(rem);
}

inline BigUInt::BigUInt(std::string_view s, int base) {
    d.push_back(0);
    for (char c : s) {
        int digit = 0;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
        else continue;
        *this = mul_limb(Limb(base));
        *this += BigUInt{u64(digit)};
    }
}

inline std::string BigUInt::to_string(int base) const {
    if (is_zero()) return "0";
    BigUInt tmp = *this;
    std::string out;
    while (!tmp.is_zero()) {
        Limb rem = tmp.div_limb(Limb(base));
        out += char(rem < 10 ? '0' + rem : 'a' + rem - 10);
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// BigInt arithmetic
inline BigInt::BigInt(std::string_view s) {
    if (!s.empty() && s[0] == '-') { neg = true; mag = BigUInt{s.substr(1)}; }
    else                           { neg = false; mag = BigUInt{s}; }
}

inline std::strong_ordering BigInt::operator<=>(const BigInt& o) const {
    if (neg != o.neg) return neg ? std::strong_ordering::less : std::strong_ordering::greater;
    auto cmp = mag <=> o.mag;
    return neg ? (cmp == std::strong_ordering::less ? std::strong_ordering::greater :
                  cmp == std::strong_ordering::greater ? std::strong_ordering::less : cmp) : cmp;
}

inline BigInt BigInt::operator+(const BigInt& b) const {
    if (neg == b.neg) return BigInt{mag + b.mag, neg};
    if ((mag <=> b.mag) == std::strong_ordering::greater) return BigInt{mag - b.mag, neg};
    BigInt r{b.mag - mag, b.neg}; r.normalize(); return r;
}
inline BigInt BigInt::operator-(const BigInt& b) const { return *this + b.negate(); }
inline BigInt BigInt::operator*(const BigInt& b) const {
    BigInt r{mag * b.mag, neg ^ b.neg}; r.normalize(); return r;
}
inline BigInt BigInt::operator/(const BigInt& b) const {
    BigInt r{mag / b.mag, neg ^ b.neg}; r.normalize(); return r;
}
inline BigInt BigInt::operator%(const BigInt& b) const {
    BigInt r{mag % b.mag, neg}; r.normalize(); return r;
}
inline std::string BigInt::to_string(int base) const {
    return (neg ? "-" : "") + mag.to_string(base);
}

} // namespace qs
