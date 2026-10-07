#pragma once
// qs_math/arithmetic/multiprecision.hpp — 128/256-bit integers, compensated float, extended precision
#include "scalar_types.hpp"
#include <array>
#include <cmath>
#include <cstring>

namespace qs::math {

// ─── 128-bit unsigned integer ─────────────────────────────────────────────────
struct U128 {
    u64 lo{0},hi{0};
    constexpr U128() noexcept=default;
    constexpr U128(u64 v) noexcept:lo(v){}
    constexpr U128(u64 hi,u64 lo) noexcept:lo(lo),hi(hi){}

    [[nodiscard]] constexpr U128 operator+(U128 b) const noexcept {
        u64 nlo=lo+b.lo; return{hi+b.hi+(nlo<lo?1:0),nlo};
    }
    [[nodiscard]] constexpr U128 operator-(U128 b) const noexcept {
        u64 nlo=lo-b.lo; return{hi-b.hi-(nlo>lo?1:0),nlo};
    }
    [[nodiscard]] constexpr U128 operator<<(int n) const noexcept {
        if(n>=64) return{lo<<(n-64),0};
        return{(hi<<n)|(lo>>(64-n)),lo<<n};
    }
    [[nodiscard]] constexpr U128 operator>>(int n) const noexcept {
        if(n>=64) return{0,hi>>(n-64)};
        return{hi>>n,(lo>>n)|(hi<<(64-n))};
    }
    [[nodiscard]] constexpr U128 operator|(U128 b) const noexcept {return{hi|b.hi,lo|b.lo};}
    [[nodiscard]] constexpr U128 operator&(U128 b) const noexcept {return{hi&b.hi,lo&b.lo};}
    [[nodiscard]] constexpr U128 operator^(U128 b) const noexcept {return{hi^b.hi,lo^b.lo};}
    [[nodiscard]] constexpr bool operator==(U128 b) const noexcept {return hi==b.hi&&lo==b.lo;}
    [[nodiscard]] constexpr bool operator<(U128 b)  const noexcept {return hi<b.hi||(hi==b.hi&&lo<b.lo);}
    [[nodiscard]] constexpr bool operator>(U128 b)  const noexcept {return b<*this;}
    [[nodiscard]] constexpr bool is_zero() const noexcept {return !hi&&!lo;}
    [[nodiscard]] constexpr U128 operator*(U128 b) const noexcept {
        // 128x128→128 (low 128 bits of result)
        U128 r;
        // lo*lo
        __uint128_t ll=(__uint128_t)lo*b.lo;
        r.lo=u64(ll); r.hi=u64(ll>>64);
        // hi*lo and lo*hi contribute to upper bits
        r.hi+=hi*b.lo+lo*b.hi;
        return r;
    }
    // Division (long division)
    [[nodiscard]] constexpr std::pair<U128,U128> divmod(U128 d) const noexcept {
        if(d.is_zero()) return{{},*this};
        U128 q{},r{};
        for(int i=127;i>=0;--i) {
            r=r<<1;
            // Set bit i of *this into bit 0 of r
            bool bit=(i>=64)?((hi>>(i-64))&1):((lo>>i)&1);
            r.lo|=bit?1:0;
            if(!(r<d)){r=r-d;if(i>=64)q.hi|=(1ULL<<(i-64));else q.lo|=(1ULL<<i);}
        }
        return{q,r};
    }
    [[nodiscard]] constexpr U128 operator/(U128 d) const noexcept {return divmod(d).first;}
    [[nodiscard]] constexpr U128 operator%(U128 d) const noexcept {return divmod(d).second;}
    [[nodiscard]] constexpr f64 to_f64() const noexcept {return f64(hi)*1.8446744073709552e19+f64(lo);}
};

// ─── 256-bit integer (4×64) ───────────────────────────────────────────────────
struct U256 {
    std::array<u64,4> d{};  // little-endian limbs
    [[nodiscard]] constexpr U256 operator+(U256 b) const noexcept {
        U256 r; u64 carry=0;
        for(int i=0;i<4;++i){
            __uint128_t s=(__uint128_t)d[i]+b.d[i]+carry;
            r.d[i]=u64(s); carry=u64(s>>64);}
        return r;
    }
    [[nodiscard]] constexpr U256 operator-(U256 b) const noexcept {
        U256 r; u64 borrow=0;
        for(int i=0;i<4;++i){
            __uint128_t s=(__uint128_t)d[i]-b.d[i]-borrow;
            r.d[i]=u64(s); borrow=u64(s>>64)?1:0;}
        return r;
    }
    [[nodiscard]] constexpr bool operator<(const U256& b) const noexcept {
        for(int i=3;i>=0;--i){if(d[i]<b.d[i])return true;if(d[i]>b.d[i])return false;}
        return false;
    }
    [[nodiscard]] constexpr bool is_zero() const noexcept {
        for(auto v:d) if(v) return false; return true;
    }
    [[nodiscard]] constexpr f64 to_f64() const noexcept {
        f64 r=0,s=1;
        for(auto v:d){r+=f64(v)*s;s*=1.8446744073709552e19;}
        return r;
    }
};

// ─── Extended precision double-double ─────────────────────────────────────────
// dd = hi + lo (|lo| ≤ 0.5 ulp(hi))
struct DD {
    f64 hi{0},lo{0};
    constexpr DD() noexcept=default;
    constexpr DD(f64 v) noexcept:hi(v){}
    constexpr DD(f64 h,f64 l) noexcept:hi(h),lo(l){}

    [[nodiscard]] constexpr DD operator+(DD b) const noexcept {
        auto [s,e]=two_sum(hi,b.hi);
        e+=lo+b.lo;
        auto [h,l]=two_sum(s,e);
        return{h,l};
    }
    [[nodiscard]] constexpr DD operator-(DD b) const noexcept {
        return *this+DD{-b.hi,-b.lo};
    }
    [[nodiscard]] constexpr DD operator*(DD b) const noexcept {
        auto [p,e]=two_product(hi,b.hi);
        e+=hi*b.lo+lo*b.hi;
        auto [h,l]=two_sum(p,e);
        return{h,l};
    }
    [[nodiscard]] constexpr DD operator/(DD b) const noexcept {
        f64 q=hi/b.hi;
        DD r=*this-DD{q}*b;
        return{q+r.hi/b.hi,0};
    }
    [[nodiscard]] constexpr f64 to_f64() const noexcept {return hi+lo;}
    [[nodiscard]] constexpr bool operator<(DD b)  const noexcept {return hi<b.hi||(hi==b.hi&&lo<b.lo);}
    [[nodiscard]] constexpr bool operator==(DD b) const noexcept {return hi==b.hi&&lo==b.lo;}
};

// DD math functions
[[nodiscard]] constexpr DD dd_abs(DD a) noexcept {return a.hi<0?DD{-a.hi,-a.lo}:a;}
[[nodiscard]] constexpr DD dd_from_int(i64 n) noexcept {return DD{f64(n)};}
[[nodiscard]] constexpr DD dd_sqrt(DD a) noexcept {
    f64 x=std::sqrt(a.hi);
    // Newton refinement
    DD xdd{x};
    xdd=xdd+(a-xdd*xdd)*(DD{1.0}/(xdd+xdd));
    return xdd;
}
[[nodiscard]] constexpr DD dd_exp(DD a, int terms=20) noexcept {
    // exp(a) via series
    DD r{1},term{1};
    for(int n=1;n<=terms;++n){term=term*a*(DD{1.0/n});r=r+term;}
    return r;
}
[[nodiscard]] constexpr DD dd_pi() noexcept {
    // Machin formula: π = 16*arctan(1/5) - 4*arctan(1/239)
    return DD{3.141592653589793,1.2246467991473532e-16};
}

// ─── Long double arithmetic helpers ──────────────────────────────────────────
// Compensated dot product (Ogita-Rump-Oishi)
template<int N>
[[nodiscard]] constexpr f64 compensated_dot(
    const std::array<f64,N>& a,
    const std::array<f64,N>& b) noexcept
{
    f64 s=0,c=0;
    for(int i=0;i<N;++i){
        auto[p,e]=two_product(a[i],b[i]);
        auto[si,ei]=two_sum(s,p);
        c+=e+ei; s=si;
    }
    return s+c;
}

// Accurate sum (Rump & Ogita)
template<int N>
[[nodiscard]] constexpr f64 accurate_sum(const std::array<f64,N>& a) noexcept {
    return kahan_sum(a);
}

// ─── Fixed-precision rational number ─────────────────────────────────────────
struct Rational {
    i64 num{0},den{1};
    constexpr Rational(i64 n=0,i64 d=1) noexcept : num(n),den(d>0?d:-d) {
        if(d<0) num=-n;
        i64 g=num<0?-num:num,b=den;
        while(b){i64 t=b;b=g%b;g=t;}
        if(g>1){num/=g;den/=g;}
    }
    [[nodiscard]] constexpr f64 to_f64() const noexcept {return f64(num)/f64(den);}
    [[nodiscard]] constexpr Rational operator+(Rational b) const noexcept {return{num*b.den+b.num*den,den*b.den};}
    [[nodiscard]] constexpr Rational operator-(Rational b) const noexcept {return{num*b.den-b.num*den,den*b.den};}
    [[nodiscard]] constexpr Rational operator*(Rational b) const noexcept {return{num*b.num,den*b.den};}
    [[nodiscard]] constexpr Rational operator/(Rational b) const noexcept {return{num*b.den,den*b.num};}
    [[nodiscard]] constexpr bool operator==(Rational b) const noexcept {return num*b.den==b.num*den;}
    [[nodiscard]] constexpr bool operator<(Rational b)  const noexcept {return num*b.den<b.num*den;}
};
} // namespace qs::math
