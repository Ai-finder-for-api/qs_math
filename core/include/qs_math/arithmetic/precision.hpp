#pragma once
// qs_math/precision.hpp — Float comparison, ULP, epsilon utilities, double comparison
#include "scalar_types.hpp"
#include "scalar_math.hpp"
#include <cstring>

namespace qs::math {

// ---------------------------------------------------------------------------
// Float bit manipulation
// ---------------------------------------------------------------------------
[[nodiscard]] inline u32 float_to_bits(f32 v) noexcept { u32 b; __builtin_memcpy(&b,&v,4); return b; }
[[nodiscard]] inline f32 bits_to_float(u32 b) noexcept { f32 v; __builtin_memcpy(&v,&b,4); return v; }
[[nodiscard]] inline u64 double_to_bits(f64 v) noexcept { u64 b; __builtin_memcpy(&b,&v,8); return b; }
[[nodiscard]] inline f64 bits_to_double(u64 b) noexcept { f64 v; __builtin_memcpy(&v,&b,8); return v; }

// ULP distance
[[nodiscard]] inline i32 ulp_distance(f32 a, f32 b) noexcept {
    u32 ai=float_to_bits(a), bi=float_to_bits(b);
    if((ai>>31)!=(bi>>31)) return (a==b)?0:i32(0x7FFFFFFF);
    i32 diff=i32(ai)-i32(bi);
    return diff<0?-diff:diff;
}
[[nodiscard]] inline i64 ulp_distance(f64 a, f64 b) noexcept {
    u64 ai=double_to_bits(a), bi=double_to_bits(b);
    if((ai>>63)!=(bi>>63)) return (a==b)?0:i64(0x7FFFFFFFFFFFFFFFLL);
    i64 diff=i64(ai)-i64(bi);
    return diff<0?-diff:diff;
}

// ULP-based equality
[[nodiscard]] inline bool equal_ulp(f32 a, f32 b, i32 max_ulps=4) noexcept { return ulp_distance(a,b)<=max_ulps; }
[[nodiscard]] inline bool equal_ulp(f64 a, f64 b, i64 max_ulps=4) noexcept { return ulp_distance(a,b)<=max_ulps; }

// Relative epsilon comparison
[[nodiscard]] constexpr bool equal_rel(f32 a, f32 b, f32 rel_eps=EPS_F32) noexcept {
    f32 diff=abs(a-b);
    f32 scale=max(abs(a),abs(b));
    return diff<=rel_eps*scale || diff<=EPS_F32;
}
[[nodiscard]] constexpr bool equal_rel(f64 a, f64 b, f64 rel_eps=EPS_F64) noexcept {
    f64 diff=abs(a-b);
    f64 scale=max(abs(a),abs(b));
    return diff<=rel_eps*scale || diff<=EPS_F64;
}

// Absolute epsilon
[[nodiscard]] constexpr bool equal_abs(f32 a, f32 b, f32 eps=EPS_F32) noexcept { return abs(a-b)<=eps; }
[[nodiscard]] constexpr bool equal_abs(f64 a, f64 b, f64 eps=EPS_F64) noexcept { return abs(a-b)<=eps; }

// Combined absolute+relative (Knuth)
[[nodiscard]] constexpr bool equal_mixed(f32 a, f32 b, f32 abs_eps=EPS_F32, f32 rel_eps=EPS_F32) noexcept {
    return equal_abs(a,b,abs_eps) || equal_rel(a,b,rel_eps);
}

// Is NaN / Inf / Subnormal
[[nodiscard]] inline bool is_subnormal(f32 v) noexcept { return (float_to_bits(v)&0x7F800000)==0; }
[[nodiscard]] inline bool is_normal_fp(f32 v) noexcept { u32 e=float_to_bits(v)&0x7F800000; return e!=0&&e!=0x7F800000; }

// Next float in given direction
[[nodiscard]] inline f32 next_float(f32 v, bool up=true) noexcept {
    u32 b=float_to_bits(v);
    if(up) return (v>=0.f)?bits_to_float(b+1):bits_to_float(b-1);
    return (v>0.f)?bits_to_float(b-1):bits_to_float(b+1);
}

// Robust interval arithmetic helper
struct Interval {
    f32 lo, hi;
    constexpr Interval(f32 v) noexcept : lo(v), hi(v) {}
    constexpr Interval(f32 l, f32 h) noexcept : lo(l), hi(h) {}
    constexpr Interval operator+(Interval o) const noexcept { return {lo+o.lo-EPS_F32,hi+o.hi+EPS_F32}; }
    constexpr Interval operator*(Interval o) const noexcept {
        f32 v0=lo*o.lo,v1=lo*o.hi,v2=hi*o.lo,v3=hi*o.hi;
        return {min(min(v0,v1),min(v2,v3))-EPS_F32, max(max(v0,v1),max(v2,v3))+EPS_F32};
    }
    [[nodiscard]] constexpr bool contains(f32 v) const noexcept { return v>=lo&&v<=hi; }
    [[nodiscard]] constexpr bool overlaps(Interval o) const noexcept { return lo<=o.hi&&hi>=o.lo; }
    [[nodiscard]] constexpr f32 width() const noexcept { return hi-lo; }
    [[nodiscard]] constexpr f32 mid()   const noexcept { return (lo+hi)*0.5f; }
};

} // namespace qs::math

// ─── Compensated summation ─────────────────────────────────────────────────
// Kahan summation for accurate f32 accumulation
struct KahanSum32 {
    f32 sum{0}, c{0};
    void add(f32 v) noexcept { f32 y=v-c, t=sum+y; c=(t-sum)-y; sum=t; }
    [[nodiscard]] f32 result() const noexcept { return sum; }
    void reset() noexcept { sum=c=0.f; }
};

// Neumaier summation (handles large cancellations better)
struct NeumaierSum32 {
    f32 sum{0}, c{0};
    void add(f32 v) noexcept {
        f32 t=sum+v;
        c += (abs(sum)>=abs(v)) ? ((sum-t)+v) : ((v-t)+sum);
        sum=t;
    }
    [[nodiscard]] f32 result() const noexcept { return sum+c; }
};

// ─── Two-sum / Fast-two-sum (error-free transforms) ───────────────────────
struct TwoSum { f32 s, e; }; // s+e = a+b exactly
[[nodiscard]] inline TwoSum two_sum(f32 a, f32 b) noexcept {
    f32 s=a+b, v=s-a, e=(a-(s-v))+(b-v); return {s,e};
}
[[nodiscard]] inline TwoSum fast_two_sum(f32 a, f32 b) noexcept { // |a|>=|b|
    f32 s=a+b; return {s, b-(s-a)};
}

// ─── Fused multiply-add helpers ────────────────────────────────────────────
[[nodiscard]] inline f32 fma_f32(f32 a, f32 b, f32 c) noexcept { return std::fma(a,b,c); }
[[nodiscard]] inline f64 fma_f64(f64 a, f64 b, f64 c) noexcept { return std::fma(a,b,c); }

// Polynomial evaluation via Horner (numerically stable)
template<int N>
[[nodiscard]] inline f32 horner(f32 x, const f32(&c)[N]) noexcept {
    f32 r=c[N-1];
    for(int i=N-2;i>=0;--i) r=std::fma(r,x,c[i]);
    return r;
}
template<int N>
[[nodiscard]] inline f64 horner(f64 x, const f64(&c)[N]) noexcept {
    f64 r=c[N-1];
    for(int i=N-2;i>=0;--i) r=std::fma(r,x,c[i]);
    return r;
}

// ─── Float class inspection ────────────────────────────────────────────────
enum class FloatClass : u8 {
    NegInf, NegNormal, NegSubnormal, NegZero,
    PosZero, PosSubnormal, PosNormal, PosInf, NaN
};
[[nodiscard]] inline FloatClass classify(f32 v) noexcept {
    switch(std::fpclassify(v)) {
        case FP_INFINITE:  return v<0?FloatClass::NegInf:FloatClass::PosInf;
        case FP_NAN:       return FloatClass::NaN;
        case FP_ZERO:      return std::signbit(v)?FloatClass::NegZero:FloatClass::PosZero;
        case FP_SUBNORMAL: return std::signbit(v)?FloatClass::NegSubnormal:FloatClass::PosSubnormal;
        default:           return std::signbit(v)?FloatClass::NegNormal:FloatClass::PosNormal;
    }
}
[[nodiscard]] inline bool is_finite_normal(f32 v) noexcept {
    auto c=classify(v); return c==FloatClass::NegNormal||c==FloatClass::PosNormal;
}
