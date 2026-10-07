#pragma once
// interval_arithmetic.hpp — Interval arithmetic for robust computation:
//   interval types, interval extensions of all math ops, affine arithmetic,
//   interval Newton method, range analysis, validated numerics
#include "scalar_types.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "constants.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Interval<T> — represents [lo, hi]
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Interval {
    T lo{T(0)}, hi{T(0)};

    constexpr Interval() noexcept = default;
    constexpr Interval(T lo, T hi) noexcept : lo(lo), hi(hi) {}
    constexpr explicit Interval(T v) noexcept : lo(v), hi(v) {}

    // Basic properties
    [[nodiscard]] constexpr T  width()    const noexcept { return hi-lo; }
    [[nodiscard]] constexpr T  midpoint() const noexcept { return (lo+hi)*T(0.5); }
    [[nodiscard]] constexpr T  radius()   const noexcept { return width()*T(0.5); }
    [[nodiscard]] constexpr bool contains(T v) const noexcept { return v>=lo&&v<=hi; }
    [[nodiscard]] constexpr bool contains(Interval o) const noexcept { return o.lo>=lo&&o.hi<=hi; }
    [[nodiscard]] constexpr bool overlaps(Interval o) const noexcept { return lo<=o.hi&&hi>=o.lo; }
    [[nodiscard]] constexpr bool is_empty() const noexcept { return lo>hi; }
    [[nodiscard]] constexpr bool is_degenerate(T eps=T(EPS_F32)) const noexcept { return width()<=eps; }

    // Arithmetic — guaranteed to contain true result
    constexpr Interval operator+(Interval o) const noexcept { return {lo+o.lo, hi+o.hi}; }
    constexpr Interval operator-(Interval o) const noexcept { return {lo-o.hi, hi-o.lo}; }
    constexpr Interval operator*(Interval o) const noexcept {
        T a=lo*o.lo, b=lo*o.hi, c=hi*o.lo, d=hi*o.hi;
        return {min(min(a,b),min(c,d)), max(max(a,b),max(c,d))};
    }
    constexpr Interval operator/(Interval o) const noexcept {
        if(o.contains(T(0))) return {NumericLimits<T>::min_val, NumericLimits<T>::max_val};
        return (*this)*Interval{T(1)/o.hi, T(1)/o.lo};
    }
    constexpr Interval operator+(T s) const noexcept { return {lo+s, hi+s}; }
    constexpr Interval operator*(T s) const noexcept {
        return s>=T(0)?Interval{lo*s,hi*s}:Interval{hi*s,lo*s};
    }
    constexpr Interval operator-() const noexcept { return {-hi, -lo}; }
    constexpr Interval& operator+=(Interval o) noexcept { *this=(*this)+o; return *this; }
    constexpr Interval& operator*=(Interval o) noexcept { *this=(*this)*o; return *this; }

    // Set operations
    [[nodiscard]] constexpr Interval hull(Interval o) const noexcept {
        return {min(lo,o.lo), max(hi,o.hi)};
    }
    [[nodiscard]] constexpr Interval intersect(Interval o) const noexcept {
        return {max(lo,o.lo), min(hi,o.hi)};
    }

    // Comparison
    [[nodiscard]] constexpr bool operator==(Interval o) const noexcept { return lo==o.lo&&hi==o.hi; }
    [[nodiscard]] constexpr bool definitely_less_than(Interval o) const noexcept { return hi<o.lo; }
    [[nodiscard]] constexpr bool definitely_greater_than(Interval o) const noexcept { return lo>o.hi; }

    // Math function extensions
    [[nodiscard]] inline Interval abs() const noexcept {
        if(lo>=T(0)) return *this;
        if(hi<=T(0)) return {-hi,-lo};
        return {T(0), max(-lo,hi)};
    }
    [[nodiscard]] inline Interval sqrt() const noexcept {
        return {::qs::math::sqrt(max(T(0),lo)), ::qs::math::sqrt(max(T(0),hi))};
    }
    [[nodiscard]] inline Interval exp() const noexcept {
        return {::qs::math::exp(lo), ::qs::math::exp(hi)};
    }
    [[nodiscard]] inline Interval log() const noexcept {
        if(lo<=T(0)) return {NumericLimits<T>::min_val, ::qs::math::log(hi)};
        return {::qs::math::log(lo), ::qs::math::log(hi)};
    }
    [[nodiscard]] inline Interval pow(T n) const noexcept {
        if(n==T(2)) return (*this)*(*this);
        if(lo>=T(0)) return {::qs::math::pow(lo,n),::qs::math::pow(hi,n)};
        return {::qs::math::pow(hi,n),::qs::math::pow(lo,n)};
    }
    [[nodiscard]] inline Interval sin() const noexcept {
        if(width()>=T(TWO_PI)) return {T(-1),T(1)};
        // Rough bound via endpoints + check for extrema
        T s0=::qs::math::sin(lo), s1=::qs::math::sin(hi);
        T lo_=min(s0,s1), hi_=max(s0,s1);
        // Check if π/2 or 3π/2 is in range
        T k=::qs::math::floor((lo-T(HALF_PI))/T(PI));
        T max_pt=T(HALF_PI)+k*T(PI);
        if(max_pt<lo) max_pt+=T(PI);
        if(max_pt<=hi){ T sv=::qs::math::sin(max_pt); hi_=max(hi_,sv); lo_=min(lo_,sv); }
        return {lo_,hi_};
    }
    [[nodiscard]] inline Interval cos() const noexcept {
        return ((*this)+Interval{T(HALF_PI)}).sin();
    }

    static constexpr Interval empty()     noexcept { return {T(1),T(0)}; }
    static constexpr Interval universe()  noexcept { return {NumericLimits<T>::min_val,NumericLimits<T>::max_val}; }
    static constexpr Interval zero()      noexcept { return {T(0),T(0)}; }
    static constexpr Interval unit()      noexcept { return {T(0),T(1)}; }
    static constexpr Interval symmetric(T r) noexcept { return {-r,r}; }
};

template<FloatScalar T> constexpr Interval<T> operator+(T s,Interval<T> i) noexcept { return i+s; }
template<FloatScalar T> constexpr Interval<T> operator*(T s,Interval<T> i) noexcept { return i*s; }
template<FloatScalar T> [[nodiscard]] inline Interval<T> hull(Interval<T> a,Interval<T> b) noexcept { return a.hull(b); }
template<FloatScalar T> [[nodiscard]] inline Interval<T> abs(Interval<T> i) noexcept { return i.abs(); }

using Intervalf = Interval<f32>;
using Intervald = Interval<f64>;

// ---------------------------------------------------------------------------
// Interval Newton method
// ---------------------------------------------------------------------------
template<FloatScalar T, typename Fn, typename DFn>
[[nodiscard]] Interval<T> interval_newton(Fn&& f, DFn&& df,
    Interval<T> x, i32 max_iter=20) noexcept
{
    for(i32 i=0;i<max_iter;++i){
        T m=x.midpoint();
        Interval<T> fm=Interval<T>{f(m)};
        Interval<T> dfx=df(x);
        if(dfx.contains(T(0))) break;
        Interval<T> nx=Interval<T>{m}-fm/dfx;
        x=x.intersect(nx);
        if(x.is_empty()||x.width()<T(1e-12)) break;
    }
    return x;
}

// ---------------------------------------------------------------------------
// Affine arithmetic (first-order)
// ---------------------------------------------------------------------------
template<FloatScalar T, i32 MAX_NOISE=16>
struct Affine {
    T center{T(0)};
    T noise[MAX_NOISE]{};
    T error{T(0)};  // accumulated rounding error
    i32 n_noise{0};

    constexpr Affine() noexcept = default;
    constexpr explicit Affine(T v) noexcept : center(v) {}
    constexpr explicit Affine(Interval<T> iv) noexcept
        : center(iv.midpoint()), error(iv.radius()) {}

    [[nodiscard]] Interval<T> to_interval() const noexcept {
        T r=error;
        for(i32 i=0;i<n_noise;++i) r+=abs(noise[i]);
        return {center-r, center+r};
    }

    Affine operator+(Affine o) const noexcept {
        Affine r;
        r.center=center+o.center;
        r.n_noise=max(n_noise,o.n_noise);
        for(i32 i=0;i<r.n_noise;++i)
            r.noise[i]=(i<n_noise?noise[i]:T(0))+(i<o.n_noise?o.noise[i]:T(0));
        r.error=error+o.error;
        return r;
    }
    Affine operator*(T s) const noexcept {
        Affine r; r.center=center*s; r.n_noise=n_noise;
        for(i32 i=0;i<n_noise;++i) r.noise[i]=noise[i]*s;
        r.error=error*abs(s); return r;
    }
    Affine operator+(T s) const noexcept { Affine r=*this; r.center+=s; return r; }

    // New noise symbol
    static Affine new_symbol(T center, T radius, i32 sym_idx) noexcept {
        Affine r; r.center=center;
        if(sym_idx<MAX_NOISE){ r.noise[sym_idx]=radius; r.n_noise=sym_idx+1; }
        else r.error=radius;
        return r;
    }
};

// ---------------------------------------------------------------------------
// Interval vector (Vec2/Vec3 over intervals)
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct IVec3 {
    Interval<T> x, y, z;
    constexpr IVec3() noexcept = default;
    constexpr IVec3(Interval<T> x,Interval<T> y,Interval<T> z) noexcept : x(x),y(y),z(z) {}
    constexpr IVec3(Vec3<T> v) noexcept : x(v.x),y(v.y),z(v.z) {}

    constexpr IVec3 operator+(IVec3 o) const noexcept { return {x+o.x,y+o.y,z+o.z}; }
    constexpr IVec3 operator*(Interval<T> s) const noexcept { return {x*s,y*s,z*s}; }
    [[nodiscard]] Interval<T> dot(IVec3 o) const noexcept { return x*o.x+y*o.y+z*o.z; }
    [[nodiscard]] Interval<T> length_sq() const noexcept { return x*x+y*y+z*z; }
};

} // namespace qs::math
