#pragma once
// qs_math/arithmetic/fixed_point.hpp — Q-format fixed point, saturating, CORDIC
#include "scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Q-format fixed point: QN.M where N+M = total bits ───────────────────────
template<int TOTAL_BITS, int FRAC_BITS>
struct Fixed {
    static_assert(FRAC_BITS>=0&&FRAC_BITS<TOTAL_BITS);
    using Raw = std::conditional_t<TOTAL_BITS<=16, i32,
                std::conditional_t<TOTAL_BITS<=32, i32, i64>>;
    Raw raw{0};
    static constexpr Raw k_one  = Raw(1)<<FRAC_BITS;
    static constexpr Raw k_max  = (Raw(1)<<(TOTAL_BITS-1))-1;
    static constexpr Raw k_min  = -(Raw(1)<<(TOTAL_BITS-1));
    static constexpr f64 k_scale= f64(Raw(1)<<FRAC_BITS);

    constexpr Fixed() noexcept = default;
    constexpr Fixed(f64 v) noexcept : raw(Raw(v*k_scale)) {}
    explicit constexpr Fixed(Raw r) noexcept : raw(r) {}

    [[nodiscard]] constexpr f64 to_f64() const noexcept { return f64(raw)/k_scale; }
    [[nodiscard]] constexpr f32 to_f32() const noexcept { return f32(raw)/f32(k_scale); }

    [[nodiscard]] constexpr Fixed operator+(Fixed o) const noexcept { return Fixed{Raw(raw+o.raw)}; }
    [[nodiscard]] constexpr Fixed operator-(Fixed o) const noexcept { return Fixed{Raw(raw-o.raw)}; }
    [[nodiscard]] constexpr Fixed operator-()        const noexcept { return Fixed{Raw(-raw)}; }
    [[nodiscard]] constexpr Fixed operator*(Fixed o) const noexcept {
        // Widened multiply, shift back
        i64 prod=i64(raw)*i64(o.raw);
        return Fixed{Raw(prod>>FRAC_BITS)};
    }
    [[nodiscard]] constexpr Fixed operator/(Fixed o) const noexcept {
        if(!o.raw) return Fixed{};
        i64 div=(i64(raw)<<FRAC_BITS)/o.raw;
        return Fixed{Raw(div)};
    }
    [[nodiscard]] constexpr bool operator<(Fixed o)  const noexcept { return raw<o.raw; }
    [[nodiscard]] constexpr bool operator==(Fixed o) const noexcept { return raw==o.raw; }
    [[nodiscard]] constexpr Fixed abs() const noexcept { return raw<0?Fixed{Raw(-raw)}:*this; }

    // Saturating add
    [[nodiscard]] constexpr Fixed sat_add(Fixed o) const noexcept {
        i64 s=i64(raw)+i64(o.raw);
        if(s>k_max) return Fixed{k_max};
        if(s<k_min) return Fixed{k_min};
        return Fixed{Raw(s)};
    }
    // Saturating multiply
    [[nodiscard]] constexpr Fixed sat_mul(Fixed o) const noexcept {
        i64 prod=i64(raw)*i64(o.raw)>>FRAC_BITS;
        if(prod>k_max) return Fixed{k_max};
        if(prod<k_min) return Fixed{k_min};
        return Fixed{Raw(prod)};
    }
};

// Common formats
using Q8_8   = Fixed<16,8>;
using Q16_16 = Fixed<32,16>;
using Q1_15  = Fixed<16,15>;  // Audio DSP
using Q1_31  = Fixed<32,31>;  // High precision audio

// ─── CORDIC algorithm ─────────────────────────────────────────────────────────
// Compute sin/cos/atan/magnitude via shift-add iterations
// Mode: 0=circular (sin/cos), 1=linear (multiply), -1=hyperbolic
template<int ITERS=16>
struct CORDIC {
    static constexpr f64 k_gain = 0.6072529350088812561694; // product of cos(atan(2^-i))

    // Rotation mode: compute (cos θ, sin θ) given angle in radians
    [[nodiscard]] constexpr std::pair<f64,f64> sincos(f64 theta) const noexcept {
        // Precomputed atan table
        constexpr std::array<f64,ITERS> atan_table=[](){
            std::array<f64,ITERS> t{};
            for(int i=0;i<ITERS;++i) t[i]=std::atan(std::pow(2.0,-i));
            return t;
        }();
        f64 x=k_gain,y=0,z=theta;
        // Reduce to [-π/2,π/2]
        while(z>3.14159265358979323846/2){z-=3.14159265358979323846;x=-x;y=-y;}
        while(z<-3.14159265358979323846/2){z+=3.14159265358979323846;x=-x;y=-y;}
        f64 po=1;
        for(int i=0;i<ITERS;++i) {
            int d=z>=0?1:-1;
            f64 xn=x-d*y*po, yn=y+d*x*po;
            z-=d*atan_table[i];
            x=xn;y=yn;po*=0.5;
        }
        return {x,y};
    }

    // Vectoring mode: compute magnitude and angle from (x,y)
    [[nodiscard]] constexpr std::pair<f64,f64> polar(f64 x, f64 y) const noexcept {
        constexpr std::array<f64,ITERS> atan_table=[](){
            std::array<f64,ITERS> t{};
            for(int i=0;i<ITERS;++i) t[i]=std::atan(std::pow(2.0,-i));
            return t;
        }();
        f64 z=0,po=1;
        for(int i=0;i<ITERS;++i) {
            int d=y<0?1:-1;
            f64 xn=x-d*y*po, yn=y+d*x*po;
            z-=d*atan_table[i];
            x=xn;y=yn;po*=0.5;
        }
        return {x*k_gain,z};
    }

    // Atan2 via vectoring
    [[nodiscard]] constexpr f64 atan2(f64 y, f64 x) const noexcept {
        if(x>=0) return polar(x,y).second;
        return y>=0?polar(-x,-y).second+3.14159265358979323846
                   :polar(-x,-y).second-3.14159265358979323846;
    }

    // Square root via modified CORDIC (hyperbolic mode)
    [[nodiscard]] constexpr f64 sqrt(f64 v) const noexcept {
        // sqrt(v) = sqrt((v+0.25)² - (v-0.25)²)/1... use Newton instead
        f64 x=v>0?v:1,xp;
        for(int i=0;i<20;++i){xp=0.5*(x+v/x);if(std::abs(xp-x)<1e-14)break;x=xp;}
        return x;
    }
};

// ─── Rounding modes ───────────────────────────────────────────────────────────
enum class RoundMode { Truncate, Round, Ceil, Floor, ConvergentRound };

template<typename Fixed_t>
[[nodiscard]] constexpr Fixed_t round_to(Fixed_t x, RoundMode mode=RoundMode::Round) noexcept {
    using Raw=typename Fixed_t::Raw;
    constexpr Raw half=Raw(1)<<(Fixed_t::FRAC_BITS-1);
    switch(mode) {
        case RoundMode::Truncate: return Fixed_t{Raw(x.raw>>Fixed_t::FRAC_BITS<<Fixed_t::FRAC_BITS)};
        case RoundMode::Round:    return Fixed_t{Raw((x.raw+half)>>Fixed_t::FRAC_BITS<<Fixed_t::FRAC_BITS)};
        case RoundMode::Ceil:     return Fixed_t{Raw((x.raw+Fixed_t::k_one-1)>>Fixed_t::FRAC_BITS<<Fixed_t::FRAC_BITS)};
        case RoundMode::Floor:    return Fixed_t{Raw(x.raw>>Fixed_t::FRAC_BITS<<Fixed_t::FRAC_BITS)};
        case RoundMode::ConvergentRound: {
            // Round to even (banker's rounding)
            Raw frac=x.raw&(Fixed_t::k_one-1);
            Raw trunc=x.raw>>Fixed_t::FRAC_BITS<<Fixed_t::FRAC_BITS;
            if(frac<half) return Fixed_t{trunc};
            if(frac>half) return Fixed_t{Raw(trunc+Fixed_t::k_one)};
            // Exactly 0.5: round to even
            return Fixed_t{Raw((trunc>>Fixed_t::FRAC_BITS)%2==0?trunc:trunc+Fixed_t::k_one)};
        }
    }
    return x;
}

// ─── Overflow detection ───────────────────────────────────────────────────────
template<typename Fixed_t>
[[nodiscard]] constexpr bool will_overflow_add(Fixed_t a, Fixed_t b) noexcept {
    using Raw=typename Fixed_t::Raw;
    i64 s=i64(a.raw)+i64(b.raw);
    return s>Fixed_t::k_max||s<Fixed_t::k_min;
}
template<typename Fixed_t>
[[nodiscard]] constexpr bool will_overflow_mul(Fixed_t a, Fixed_t b) noexcept {
    i64 p=i64(a.raw)*i64(b.raw)>>Fixed_t::FRAC_BITS;
    return p>Fixed_t::k_max||p<Fixed_t::k_min;
}

// ─── Fixed-point Newton-Raphson reciprocal (1/x) ─────────────────────────────
[[nodiscard]] constexpr Q16_16 fp_reciprocal(Q16_16 x, int iters=4) noexcept {
    if(!x.raw) return Q16_16{Q16_16::k_max};
    // Initial estimate via bit manipulation
    Q16_16 r{Q16_16::Raw(0x60000000/(x.raw>0?x.raw:-x.raw))};
    for(int i=0;i<iters;++i) r=r.sat_mul(Q16_16{2.0}-x.sat_mul(r));
    return x.raw<0?Q16_16{Q16_16::Raw(-r.raw)}:r;
}
} // namespace qs::math
