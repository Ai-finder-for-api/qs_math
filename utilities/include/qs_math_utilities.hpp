
#pragma once
// qs_math/utilities/include/qs_math_utilities.hpp (C++20)
// Math utilities: bit manipulation, fixed-point helpers, lookup tables,
// numeric limits, string formatting, PRNG utilities, compile-time math.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <functional>
#include <limits>
#include <bit>
#include <concepts>
namespace qs::math::util {
using namespace qs::math;
// ─── Bit manipulation ────────────────────────────────────────────────────────
[[nodiscard]] constexpr u32 rotl32(u32 x, i32 k) noexcept { return (x<<k)|(x>>(32-k)); }
[[nodiscard]] constexpr u32 rotr32(u32 x, i32 k) noexcept { return (x>>k)|(x<<(32-k)); }
[[nodiscard]] constexpr u64 rotl64(u64 x, i32 k) noexcept { return (x<<k)|(x>>(64-k)); }
[[nodiscard]] constexpr u32 spread_bits(u16 v) noexcept {
    u32 x=v;
    x=(x|(x<<8))&0x00ff00ff; x=(x|(x<<4))&0x0f0f0f0f;
    x=(x|(x<<2))&0x33333333; x=(x|(x<<1))&0x55555555;
    return x;
}
[[nodiscard]] constexpr u32 morton_encode(u16 x, u16 y) noexcept {
    return spread_bits(x)|(spread_bits(y)<<1);
}
[[nodiscard]] constexpr u32 morton_encode3(u16 x,u16 y,u16 z) noexcept {
    return spread_bits(x)|(spread_bits(y)<<1)|(spread_bits(z)<<2);
}
[[nodiscard]] constexpr bool is_pow2(u64 n) noexcept { return n&&!(n&(n-1)); }
[[nodiscard]] constexpr u64 next_pow2_u64(u64 n) noexcept {
    --n;n|=n>>1;n|=n>>2;n|=n>>4;n|=n>>8;n|=n>>16;n|=n>>32;return n+1;
}
[[nodiscard]] constexpr i32 ilog2(u64 n) noexcept { return n?63-__builtin_clzll(n):-1; }
[[nodiscard]] constexpr i32 popcount32(u32 n) noexcept { return __builtin_popcount(n); }
// ─── Float bit tricks ─────────────────────────────────────────────────────────
[[nodiscard]] inline f32 rsqrt_fast(f32 x) noexcept {
    // Quake III / Carmack fast inverse square root
    union { f32 f; u32 i; } u;
    u.f=x; u.i=0x5f3759df-(u.i>>1);
    return u.f*(1.5f-x*0.5f*u.f*u.f);
}
[[nodiscard]] inline bool nearly_equal(f32 a, f32 b, f32 eps=1e-6f) noexcept {
    return std::abs(a-b)<=eps*std::max(1.0f,std::max(std::abs(a),std::abs(b)));
}
[[nodiscard]] inline bool nearly_equal_f64(f64 a, f64 b, f64 eps=1e-12) noexcept {
    return std::abs(a-b)<=eps*std::max(1.0,std::max(std::abs(a),std::abs(b)));
}
// ─── Numeric utilities ────────────────────────────────────────────────────────
[[nodiscard]] constexpr f32 deg2rad(f32 d) noexcept { return d*0.01745329251994329577f; }
[[nodiscard]] constexpr f32 rad2deg(f32 r) noexcept { return r*57.29577951308232087680f; }
[[nodiscard]] constexpr f64 deg2rad_f64(f64 d) noexcept { return d*0.01745329251994329577; }
[[nodiscard]] constexpr f64 rad2deg_f64(f64 r) noexcept { return r*57.29577951308232087680; }
template<typename T> [[nodiscard]] constexpr T clamp(T v,T lo,T hi) noexcept { return v<lo?lo:(v>hi?hi:v); }
template<typename T> [[nodiscard]] constexpr T saturate(T v) noexcept { return clamp(v,T(0),T(1)); }
template<typename T> [[nodiscard]] constexpr T remap(T v,T s0,T s1,T d0,T d1) noexcept {
    return d0+(v-s0)*(d1-d0)/(s1-s0);
}
// ─── Lookup table helpers ─────────────────────────────────────────────────────
template<i32 N>
struct SinTable {
    std::array<f32,N> data{};
    constexpr SinTable() noexcept {
        for(i32 i=0;i<N;++i) data[i]=static_cast<f32>(std::sin(2*3.14159265358979323846*i/N));
    }
    [[nodiscard]] f32 operator[](i32 i) const noexcept { return data[((i%N)+N)%N]; }
    [[nodiscard]] f32 lerp(f32 phase_norm) const noexcept { // phase in [0,1)
        const f32 t=phase_norm*N; const i32 i=(i32)t; const f32 frac=t-i;
        return data[i%N]+(data[(i+1)%N]-data[i%N])*frac;
    }
};
// ─── Prefix sum / scan ───────────────────────────────────────────────────────
void prefix_sum(const f32* in, f32* out, i32 n) noexcept;
void exclusive_scan(const f32* in, f32* out, i32 n, f32 init=0) noexcept;
// ─── Running statistics ───────────────────────────────────────────────────────
struct RunningStats {
    i64 count{0}; f64 mean{0}, M2{0}, min_v{1e300}, max_v{-1e300};
    void push(f64 x) noexcept;
    [[nodiscard]] f64 variance() const noexcept { return count<2?0:M2/(count-1); }
    [[nodiscard]] f64 stddev() const noexcept { return std::sqrt(variance()); }
};
// ─── Compile-time factorial / binomial ───────────────────────────────────────
template<u64 N> struct Factorial { static constexpr u64 value = N*Factorial<N-1>::value; };
template<> struct Factorial<0> { static constexpr u64 value = 1; };
template<u64 N,u64 K> struct Binomial {
    static constexpr u64 value = Factorial<N>::value/(Factorial<K>::value*Factorial<N-K>::value);
};
// ─── Error accumulation helpers ───────────────────────────────────────────────
struct KahanSum { f64 sum{0}, c{0};
    void add(f64 v) noexcept { const f64 y=v-c; const f64 t=sum+y; c=(t-sum)-y; sum=t; }
    [[nodiscard]] f64 get() const noexcept { return sum; }
};
// ─── Piecewise linear function ────────────────────────────────────────────────
struct PiecewiseLinear {
    std::vector<f64> xs, ys;
    void push(f64 x, f64 y) noexcept { xs.push_back(x); ys.push_back(y); }
    [[nodiscard]] f64 eval(f64 x) const noexcept;
    [[nodiscard]] f64 inv_eval(f64 y) const noexcept; // inverse lookup
};
} // namespace qs::math::util
