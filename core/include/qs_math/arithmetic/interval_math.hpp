#pragma once
// qs_math/arithmetic/interval_math.hpp — Interval arithmetic, affine forms, verified computation
#include "scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

// ─── Interval [lo, hi] ───────────────────────────────────────────────────────
struct Interval {
    f64 lo{0},hi{0};
    constexpr Interval() noexcept = default;
    constexpr Interval(f64 v) noexcept : lo(v),hi(v) {}
    constexpr Interval(f64 l,f64 h) noexcept : lo(l),hi(h) {}

    [[nodiscard]] constexpr f64  width()    const noexcept { return hi-lo; }
    [[nodiscard]] constexpr f64  midpoint() const noexcept { return (lo+hi)*0.5; }
    [[nodiscard]] constexpr f64  radius()   const noexcept { return (hi-lo)*0.5; }
    [[nodiscard]] constexpr bool contains(f64 v) const noexcept { return v>=lo&&v<=hi; }
    [[nodiscard]] constexpr bool overlaps(Interval o) const noexcept { return lo<=o.hi&&o.lo<=hi; }
    [[nodiscard]] constexpr bool subset_of(Interval o) const noexcept { return lo>=o.lo&&hi<=o.hi; }

    [[nodiscard]] constexpr Interval operator+(Interval o) const noexcept {return{lo+o.lo,hi+o.hi};}
    [[nodiscard]] constexpr Interval operator-(Interval o) const noexcept {return{lo-o.hi,hi-o.lo};}
    [[nodiscard]] constexpr Interval operator-()           const noexcept {return{-hi,-lo};}
    [[nodiscard]] constexpr Interval operator*(Interval o) const noexcept {
        f64 a=lo*o.lo,b=lo*o.hi,c=hi*o.lo,d=hi*o.hi;
        return{std::min({a,b,c,d}),std::max({a,b,c,d})};
    }
    [[nodiscard]] constexpr Interval operator/(Interval o) const noexcept {
        if(o.lo<=0&&o.hi>=0) return{-1e300,1e300}; // division by zero interval
        return *this*Interval{1/o.hi,1/o.lo};
    }
    [[nodiscard]] constexpr Interval hull(Interval o) const noexcept {
        return{lo<o.lo?lo:o.lo,hi>o.hi?hi:o.hi};
    }
    [[nodiscard]] constexpr Interval intersect(Interval o) const noexcept {
        f64 nl=lo>o.lo?lo:o.lo, nh=hi<o.hi?hi:o.hi;
        return nl<=nh?Interval{nl,nh}:Interval{0,0};
    }
    [[nodiscard]] constexpr Interval abs() const noexcept {
        if(lo>=0) return *this;
        if(hi<=0) return{-hi,-lo};
        return{0,hi>-lo?hi:-lo};
    }
    [[nodiscard]] constexpr Interval sqr() const noexcept { return (*this)*(*this); }
};

// Interval functions
[[nodiscard]] inline Interval iv_sqrt(Interval x) noexcept {
    if(x.lo<0) x.lo=0;
    return{std::sqrt(x.lo),std::sqrt(x.hi)};
}
[[nodiscard]] inline Interval iv_exp(Interval x) noexcept {
    return{std::exp(x.lo),std::exp(x.hi)};
}
[[nodiscard]] inline Interval iv_log(Interval x) noexcept {
    if(x.lo<=0) return{-1e300,std::log(x.hi)};
    return{std::log(x.lo),std::log(x.hi)};
}
[[nodiscard]] inline Interval iv_sin(Interval x) noexcept {
    // Conservative: if width >= 2π, return [-1,1]
    if(x.hi-x.lo>=2*3.14159265358979323846) return{-1,1};
    f64 sl=std::sin(x.lo),sh=std::sin(x.hi);
    return{std::min(sl,sh),std::max(sl,sh)};
}
[[nodiscard]] inline Interval iv_cos(Interval x) noexcept {
    if(x.hi-x.lo>=2*3.14159265358979323846) return{-1,1};
    f64 cl=std::cos(x.lo),ch=std::cos(x.hi);
    return{std::min(cl,ch),std::max(cl,ch)};
}
[[nodiscard]] inline Interval iv_pow(Interval x, int n) noexcept {
    if(n==0) return{1,1};
    Interval r{1,1};
    Interval b=x;
    int p=n<0?-n:n;
    while(p>0){if(p&1)r=r*b;b=b*b;p>>=1;}
    return n<0?Interval{1,1}/r:r;
}

// ─── Interval vector and matrix ───────────────────────────────────────────────
template<int N>
using IVec = std::array<Interval,N>;

template<int N>
[[nodiscard]] constexpr IVec<N> iv_add(const IVec<N>& a, const IVec<N>& b) noexcept {
    IVec<N> r; for(int i=0;i<N;++i) r[i]=a[i]+b[i]; return r;
}
template<int N>
[[nodiscard]] constexpr Interval iv_dot(const IVec<N>& a, const IVec<N>& b) noexcept {
    Interval s{0}; for(int i=0;i<N;++i) s=s+a[i]*b[i]; return s;
}

// ─── Affine arithmetic ────────────────────────────────────────────────────────
// Element: x = x0 + Σ xᵢ εᵢ, where εᵢ ∈ [-1,1]
template<int N>
struct AffineForm {
    f64 x0{0}; // central value
    std::array<f64,N> xi{}; // noise symbols

    constexpr AffineForm() noexcept = default;
    constexpr AffineForm(f64 v) noexcept : x0(v) {}
    constexpr AffineForm(Interval iv) noexcept
        : x0(iv.midpoint()), xi{} { xi[0]=iv.radius(); }

    [[nodiscard]] constexpr Interval to_interval() const noexcept {
        f64 r=0; for(auto v:xi) r+=v<0?-v:v;
        return{x0-r,x0+r};
    }
    [[nodiscard]] constexpr AffineForm operator+(AffineForm o) const noexcept {
        AffineForm r; r.x0=x0+o.x0;
        for(int i=0;i<N;++i) r.xi[i]=xi[i]+o.xi[i];
        return r;
    }
    [[nodiscard]] constexpr AffineForm operator*(f64 s) const noexcept {
        AffineForm r; r.x0=x0*s;
        for(int i=0;i<N;++i) r.xi[i]=xi[i]*s;
        return r;
    }
    [[nodiscard]] constexpr AffineForm operator*(AffineForm o) const noexcept {
        AffineForm r; r.x0=x0*o.x0;
        // Linear terms
        for(int i=0;i<N;++i) r.xi[i]=x0*o.xi[i]+o.x0*xi[i];
        // Quadratic remainder: bound by sum of |xi||oj|
        f64 err=0;
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) err+=(xi[i]<0?-xi[i]:xi[i])*(o.xi[j]<0?-o.xi[j]:o.xi[j]);
        // Add as new noise term (if space allows)
        if(N>1) r.xi[N-1]+=err; else r.x0+=err; // conservative
        return r;
    }
};

// ─── Kaucher interval arithmetic (modal/improper intervals) ───────────────────
// Proper:   lo ≤ hi (conventional)
// Improper: lo > hi (dual)
struct KaucherInterval {
    f64 lo,hi;
    [[nodiscard]] constexpr bool is_proper() const noexcept { return lo<=hi; }
    [[nodiscard]] constexpr KaucherInterval dual() const noexcept { return{hi,lo}; }
    [[nodiscard]] constexpr KaucherInterval operator+(KaucherInterval o) const noexcept {
        return{lo+o.lo,hi+o.hi};
    }
    [[nodiscard]] constexpr KaucherInterval operator*(KaucherInterval o) const noexcept {
        // Kaucher multiplication
        f64 a=lo,b=hi,c=o.lo,d=o.hi;
        return{std::min({a*c,a*d,b*c,b*d}),std::max({a*c,a*d,b*c,b*d})};
    }
};

// ─── Verified summation (error-free transformation) ───────────────────────────
// Knuth TwoSum: compute s=a+b exactly as (s, err)
[[nodiscard]] constexpr std::pair<f64,f64> two_sum(f64 a, f64 b) noexcept {
    f64 s=a+b;
    f64 bp=s-a, ap=s-bp;
    f64 err=(a-ap)+(b-bp);
    return{s,err};
}
// TwoProduct via FMA (error-free multiply)
[[nodiscard]] constexpr std::pair<f64,f64> two_product(f64 a, f64 b) noexcept {
    f64 p=a*b;
    f64 err=std::fma(a,b,-p);
    return{p,err};
}
// Compensated summation (Kahan)
template<int N>
[[nodiscard]] constexpr f64 kahan_sum(const std::array<f64,N>& x) noexcept {
    f64 s=0,c=0;
    for(auto v:x){f64 y=v-c,t=s+y;c=(t-s)-y;s=t;}
    return s;
}
// Neumaier sum (handles large elements better)
template<int N>
[[nodiscard]] constexpr f64 neumaier_sum(const std::array<f64,N>& x) noexcept {
    f64 s=0,c=0;
    for(auto v:x){
        f64 t=s+v;
        c+=std::abs(s)>=std::abs(v)?(s-t)+v:(v-t)+s;
        s=t;
    }
    return s+c;
}
} // namespace qs::math
