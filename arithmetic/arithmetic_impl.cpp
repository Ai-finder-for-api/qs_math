// qs_math/arithmetic/arithmetic_impl.cpp (C++20)
#include "include/qs_math_arithmetic.hpp"
#include <cmath>
namespace qs::math::arithmetic {
// Veltkamp splitting for double-double multiply
DDFloat DDFloat::operator*(DDFloat o) const noexcept {
    // Split hi into two 26-bit parts
    auto split = [](f64 a) -> std::pair<f64,f64> {
        const f64 c=134217729.0*a; const f64 x=c-(c-a); return {x,a-x};
    };
    auto [ahi,alo]=split(hi); auto [ohi,olo]=split(o.hi);
    const f64 p=hi*o.hi;
    const f64 e=((ahi*ohi-p)+ahi*olo+alo*ohi+alo*olo)+hi*o.lo+lo*o.hi;
    const f64 r=p+e; return {r,e-(r-p)};
}
// ─── Explicit instantiations ──────────────────────────────────────────────────
template struct std::array<Fixed16,4>;
template struct std::array<Fixed24,4>;
// ─── Fixed-point sqrt via Newton-Raphson ─────────────────────────────────────
[[nodiscard]] Fixed16 fixed16_sqrt(Fixed16 x) noexcept {
    if (x.raw <= 0) return {0};
    // Initial estimate: halve the exponent
    i32 r = (i32)std::sqrt((f32)x.raw * Fixed16::SCALE);
    // Two Newton-Raphson iterations: r = (r + x/r) / 2
    for (i32 i=0;i<3;++i) {
        if (!r) break;
        r = (r + (i32)(((i64)x.raw<<Fixed16::FRAC)/r)) >> 1;
    }
    return {r};
}
// ─── Fixed-point trig (CORDIC — vectoring mode, 16 iterations) ───────────────
static const i32 CORDIC_ANGLES16[16] = {
    26980,15986,8466,4270,2139,1070,535,267,134,67,33,17,8,4,2,1
}; // atan(2^-i) * (1<<16) / (π/2)  scaled
[[nodiscard]] std::pair<Fixed16,Fixed16> cordic_sincos(i32 theta_q16) noexcept {
    // theta_q16: angle in units of (1<<16) = 2π
    // Returns {sin, cos} as Fixed16
    i32 x=19898,y=0,z=theta_q16; // K=0.6072...*2^16 = 39797; x=K/2 scaled
    for (i32 i=0;i<16;++i) {
        const i32 d=z<0?-1:1;
        const i32 nx=x-d*(y>>i), ny=y+d*(x>>i);
        z-=d*CORDIC_ANGLES16[i];
        x=nx; y=ny;
    }
    return {Fixed16{y},Fixed16{x}};
}
// ─── Modular exponentiation ────────────────────────────────────────────────────
[[nodiscard]] u64 mod_pow(u64 base, u64 exp, u64 mod) noexcept {
    u64 r=1; base%=mod;
    for (;exp>0;exp>>=1){ if(exp&1) r=(__uint128_t)r*base%mod; base=(__uint128_t)base*base%mod; }
    return r;
}
// Miller-Rabin primality test (deterministic for n < 3,317,044,064,679,887,385,961,981)
[[nodiscard]] bool is_prime(u64 n) noexcept {
    if (n<2) return false; if (n==2||n==3||n==5||n==7) return true;
    if (!(n&1)||n%3==0) return false;
    u64 d=n-1; i32 r=0; while(!(d&1)){d>>=1;++r;}
    for (u64 a:{2,3,5,7,11,13,17,19,23,29,31,37}) {
        if (a>=n) continue;
        u64 x=mod_pow(a,d,n);
        if (x==1||x==n-1) continue;
        bool composite=true;
        for (i32 i=0;i<r-1;++i){ x=(__uint128_t)x*x%n; if(x==n-1){composite=false;break;} }
        if (composite) return false;
    }
    return true;
}
// Pollard's rho factorization
[[nodiscard]] u64 pollard_rho(u64 n) noexcept {
    if (!(n&1)) return 2;
    u64 x=2,y=2,c=1,d=1;
    auto f=[&](u64 v){return ((__uint128_t)v*v+c)%n;};
    while(d==1){x=f(x);y=f(f(y));u64 a=x>y?x-y:y-x;d=std::__gcd(a,n);}
    return d==n?n:d; // n means failed (retry with different c)
}
} // namespace qs::math::arithmetic
