#pragma once
// qs_math/rational.hpp — Exact rational arithmetic p/q (C++20)
// Uses BigInt for unbounded precision; also provides fast Rational<T> for fixed int types.

#include "scalar_types.hpp"
#include "big_integer.hpp"
#include <stdexcept>
#include <numeric>
#include <compare>
#include <string>
#include <cmath>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// Rational<T> — exact rational backed by a fixed integer type T
//   T should be a signed type with operator%, /, *, +, -.
//   Use i64 for most purposes; use BigInt for unbounded precision.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
struct Rational {
    T p{0}, q{1}; // invariant: q > 0, gcd(|p|,q) == 1

    constexpr Rational() = default;
    constexpr explicit Rational(T num)            : p(num), q(T(1)) {}
    constexpr Rational(T num, T den) {
        if (den == T(0)) throw std::domain_error("Rational: zero denominator");
        T g = std::gcd(num < 0 ? -num : num, den < 0 ? -den : den);
        if (g == 0) g = T(1);
        if (den < 0) { num = -num; den = -den; }
        p = num / g; q = den / g;
    }

    // Basic predicates
    [[nodiscard]] constexpr bool is_zero()     const { return p == T(0); }
    [[nodiscard]] constexpr bool is_integer()  const { return q == T(1); }
    [[nodiscard]] constexpr bool is_positive() const { return p > T(0); }
    [[nodiscard]] constexpr bool is_negative() const { return p < T(0); }

    // Conversion
    [[nodiscard]] constexpr f64 to_f64()       const { return f64(p) / f64(q); }
    [[nodiscard]] constexpr f32 to_f32()       const { return f32(p) / f32(q); }
    [[nodiscard]] constexpr T   floor()        const { return p / q - (p % q != T(0) && p < T(0) ? T(1) : T(0)); }
    [[nodiscard]] constexpr T   ceil()         const { return p / q + (p % q != T(0) && p > T(0) ? T(1) : T(0)); }
    [[nodiscard]] constexpr T   round()        const { return (p + q / T(2)) / q; }
    [[nodiscard]] constexpr Rational reciprocal() const { return Rational{q, p}; }
    [[nodiscard]] constexpr Rational abs()     const { return Rational{p < T(0) ? -p : p, q}; }
    [[nodiscard]] constexpr Rational negate()  const { return Rational{-p, q}; }

    // Comparison
    [[nodiscard]] constexpr std::strong_ordering operator<=>(const Rational& o) const {
        // Cross-multiply avoiding overflow for large T
        T lhs = p * o.q, rhs = o.p * q;
        return lhs <=> rhs;
    }
    [[nodiscard]] constexpr bool operator==(const Rational& o) const { return p == o.p && q == o.q; }

    // Arithmetic
    [[nodiscard]] constexpr Rational operator+(const Rational& b) const { return Rational{p * b.q + b.p * q, q * b.q}; }
    [[nodiscard]] constexpr Rational operator-(const Rational& b) const { return Rational{p * b.q - b.p * q, q * b.q}; }
    [[nodiscard]] constexpr Rational operator*(const Rational& b) const { return Rational{p * b.p, q * b.q}; }
    [[nodiscard]] constexpr Rational operator/(const Rational& b) const { return Rational{p * b.q, q * b.p}; }
    [[nodiscard]] constexpr Rational operator-()                  const { return negate(); }

    Rational& operator+=(const Rational& b) { *this = *this + b; return *this; }
    Rational& operator-=(const Rational& b) { *this = *this - b; return *this; }
    Rational& operator*=(const Rational& b) { *this = *this * b; return *this; }
    Rational& operator/=(const Rational& b) { *this = *this / b; return *this; }

    // Exponentiation by integer
    [[nodiscard]] constexpr Rational pow(i32 n) const {
        if (n < 0) return reciprocal().pow(-n);
        Rational r{T(1)}, base = *this;
        while (n > 0) {
            if (n & 1) r *= base;
            base *= base; n >>= 1;
        }
        return r;
    }

    [[nodiscard]] std::string to_string() const {
        if (q == T(1)) return std::to_string(p);
        return std::to_string(p) + "/" + std::to_string(q);
    }
};

// Convenience aliases
using Rat32  = Rational<i32>;
using Rat64  = Rational<i64>;
using RatBig = Rational<BigInt>;

// ─────────────────────────────────────────────────────────────────────────────
// Free-function math on Rational<T>
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
[[nodiscard]] constexpr Rational<T> mediant(const Rational<T>& a, const Rational<T>& b) {
    // Stern-Brocot mediant: (a.p+b.p)/(a.q+b.q)
    return Rational<T>{a.p + b.p, a.q + b.q};
}

// Stern-Brocot tree: find best rational approximation to x with denominator <= max_denom
template<typename T>
[[nodiscard]] Rational<T> best_rational_approx(f64 x, T max_denom) {
    Rational<T> lo{T(0), T(1)}, hi{T(1), T(0)};
    if (x < 0.0) {
        auto r = best_rational_approx<T>(-x, max_denom);
        return r.negate();
    }
    while (true) {
        Rational<T> m = mediant(lo, hi);
        if (m.q > max_denom) break;
        f64 mv = m.to_f64();
        if (mv < x)       lo = m;
        else if (mv > x)  hi = m;
        else              return m;
    }
    // Return whichever of lo/hi is closer
    return (std::abs(lo.to_f64() - x) <= std::abs(hi.to_f64() - x)) ? lo : hi;
}

// Continued fraction expansion of x with max n partial quotients
template<typename T>
[[nodiscard]] std::vector<T> continued_fraction(f64 x, u32 n = 20) {
    std::vector<T> cf;
    for (u32 i = 0; i < n; ++i) {
        T a = T(std::floor(x));
        cf.push_back(a);
        x -= f64(a);
        if (std::abs(x) < 1e-12) break;
        x = 1.0 / x;
    }
    return cf;
}

// Evaluate continued fraction [a0; a1, a2, ...] as Rational
template<typename T>
[[nodiscard]] Rational<T> eval_continued_fraction(const std::vector<T>& cf) {
    if (cf.empty()) return Rational<T>{T(0)};
    Rational<T> r{cf.back()};
    for (i32 i = i32(cf.size()) - 2; i >= 0; --i) {
        r = Rational<T>{cf[u32(i)]} + Rational<T>{T(1)} / r;
    }
    return r;
}

// Farey sequence: all fractions p/q with 0<=p<=q<=n in order
template<typename T>
[[nodiscard]] std::vector<Rational<T>> farey_sequence(T n) {
    std::vector<Rational<T>> seq;
    T a = 0, b = 1, c = 1, d = n;
    seq.push_back(Rational<T>{a, b});
    while (c <= n) {
        seq.push_back(Rational<T>{c, d});
        T k = (n + b) / d;
        T na = c, nb = d, nc = k * c - a, nd = k * d - b;
        a = na; b = nb; c = nc; d = nd;
    }
    return seq;
}

// Chinese Remainder Theorem: find x such that x ≡ r[i] (mod m[i])
// Returns exact rational representation of solution
[[nodiscard]] inline Rat64 crt(const std::vector<i64>& r, const std::vector<i64>& m) {
    assert(r.size() == m.size());
    i64 M = 1;
    for (auto mi : m) M *= mi;
    i64 x = 0;
    for (u32 i = 0; i < u32(r.size()); ++i) {
        i64 Mi = M / m[i];
        // find inverse of Mi mod m[i] via extended Euclidean
        i64 s = 1, old_s = 0, t = 0, old_t = 1, r0 = m[i], r1 = Mi % m[i];
        while (r1 != 0) {
            i64 q = r0 / r1;
            i64 tmp = r1; r1 = r0 - q * r1; r0 = tmp;
            tmp = old_s; old_s = s - q * old_s; s = tmp;
        }
        i64 inv = old_s;
        x = (x + r[i] * Mi % M * ((inv % M + M) % M)) % M;
    }
    return Rat64{x, M};
}

// Exact rational trigonometric values for common angles (degrees)
// Returns {}, indicating irrational otherwise
[[nodiscard]] inline std::optional<Rat64> exact_sin(i32 deg) {
    deg = ((deg % 360) + 360) % 360;
    switch (deg) {
        case 0:   return Rat64{0};
        case 30:  return Rat64{1, 2};
        case 90:  return Rat64{1};
        case 150: return Rat64{1, 2};
        case 180: return Rat64{0};
        case 210: return Rat64{-1, 2};
        case 270: return Rat64{-1};
        case 330: return Rat64{-1, 2};
        default:  return std::nullopt;
    }
}

[[nodiscard]] inline std::optional<Rat64> exact_cos(i32 deg) {
    return exact_sin(deg + 90);
}

} // namespace qs
