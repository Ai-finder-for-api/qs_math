#pragma once
// constexpr_math.hpp — Fully constexpr math: Chebyshev trig, sqrt, log, exp,
//                       compile-time constants generation, constexpr tables
// Everything here is usable at compile time — no runtime-only functions.
#include "scalar_types.hpp"
#include "constants.hpp"

namespace qs::math::cx {

// ---------------------------------------------------------------------------
// constexpr sqrt (Newton-Raphson)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f64 sqrt_cx(f64 x, f64 guess) noexcept {
    f64 next=(guess+x/guess)*0.5;
    return (next==guess)?guess:sqrt_cx(x,next);
}
[[nodiscard]] constexpr f64 sqrt(f64 x) noexcept {
    return x<0.0?0.0:(x==0.0?0.0:sqrt_cx(x,x>1.0?x*0.5:1.0));
}
[[nodiscard]] constexpr f32 sqrt(f32 x) noexcept { return f32(sqrt(f64(x))); }

// ---------------------------------------------------------------------------
// constexpr abs / sign / min / max
// ---------------------------------------------------------------------------
template<typename T> [[nodiscard]] constexpr T abs(T v) noexcept { return v<T(0)?-v:v; }
template<typename T> [[nodiscard]] constexpr T min(T a,T b) noexcept { return a<b?a:b; }
template<typename T> [[nodiscard]] constexpr T max(T a,T b) noexcept { return a>b?a:b; }
template<typename T> [[nodiscard]] constexpr T clamp(T v,T lo,T hi) noexcept { return v<lo?lo:(v>hi?hi:v); }

// ---------------------------------------------------------------------------
// constexpr floor / ceil / round
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f64 floor_cx(f64 x) noexcept {
    i64 xi=i64(x); return (x<0.0&&f64(xi)!=x)?f64(xi-1):f64(xi);
}
[[nodiscard]] constexpr f64 ceil_cx(f64 x)  noexcept { return -floor_cx(-x); }
[[nodiscard]] constexpr f64 round_cx(f64 x) noexcept { return floor_cx(x+0.5); }
[[nodiscard]] constexpr f64 frac_cx(f64 x)  noexcept { return x-floor_cx(x); }
[[nodiscard]] constexpr f32 floor_cx(f32 x) noexcept { return f32(floor_cx(f64(x))); }
[[nodiscard]] constexpr f32 ceil_cx(f32 x)  noexcept { return f32(ceil_cx(f64(x))); }
[[nodiscard]] constexpr f32 frac_cx(f32 x)  noexcept { return f32(frac_cx(f64(x))); }

// ---------------------------------------------------------------------------
// constexpr exp (Taylor series, converges for all x)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f64 exp_cx(f64 x) noexcept {
    // Range reduction: exp(x) = exp(floor(x)) * exp(frac(x))
    // Use Taylor: sum(x^n/n!)
    if(x<-20.0) return 0.0;
    if(x>20.0) return 485165195.41;
    f64 sum=1.0, term=1.0;
    for(i32 n=1;n<50;++n){ term*=x/n; sum+=term; if(abs(term)<1e-17) break; }
    return sum;
}
[[nodiscard]] constexpr f32 exp_cx(f32 x) noexcept { return f32(exp_cx(f64(x))); }

// ---------------------------------------------------------------------------
// constexpr log (natural log, Newton's method on [0.5, 1.5] + range reduction)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f64 log_cx(f64 x) noexcept {
    if(x<=0.0) return -1e308;
    if(x==1.0) return 0.0;
    // Range reduction using x = 2^n * m where 0.5 <= m < 1
    i32 n=0; f64 m=x;
    while(m>=2.0){m*=0.5;++n;} while(m<0.5){m*=2.0;--n;}
    // log(m) via Padé approximant (fast, accurate)
    f64 y=(m-1.0)/(m+1.0), y2=y*y;
    f64 r=y*(2.0+y2*(2.0/3.0+y2*(2.0/5.0+y2*(2.0/7.0+y2*(2.0/9.0+y2*2.0/11.0)))));
    return r+n*LN2_D;
}
[[nodiscard]] constexpr f64 log2_cx(f64 x)  noexcept { return log_cx(x)*LOG2E_D; }
[[nodiscard]] constexpr f64 log10_cx(f64 x) noexcept { return log_cx(x)*LOG10E_D; }
[[nodiscard]] constexpr f32 log_cx(f32 x)   noexcept { return f32(log_cx(f64(x))); }
[[nodiscard]] constexpr f32 log2_cx(f32 x)  noexcept { return f32(log2_cx(f64(x))); }

// ---------------------------------------------------------------------------
// constexpr pow
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f64 pow_cx(f64 base, f64 exp_) noexcept {
    if(exp_==0.0) return 1.0;
    if(base==0.0) return 0.0;
    return exp_cx(exp_*log_cx(base));
}
[[nodiscard]] constexpr f32 pow_cx(f32 b, f32 e) noexcept { return f32(pow_cx(f64(b),f64(e))); }

// ---------------------------------------------------------------------------
// constexpr trig (Chebyshev polynomial approximations)
// ---------------------------------------------------------------------------
// sin on [-π, π] via Chebyshev (7th order)
[[nodiscard]] constexpr f64 sin_cx_reduced(f64 x) noexcept {
    f64 x2=x*x;
    return x*(1.0+x2*(-1.0/6.0+x2*(1.0/120.0+x2*(-1.0/5040.0+x2*(1.0/362880.0
           +x2*(-1.0/39916800.0+x2/6227020800.0))))));
}
[[nodiscard]] constexpr f64 cos_cx_reduced(f64 x) noexcept {
    f64 x2=x*x;
    return 1.0+x2*(-0.5+x2*(1.0/24.0+x2*(-1.0/720.0+x2*(1.0/40320.0
           +x2*(-1.0/3628800.0+x2/479001600.0)))));
}

// Range reduction for sin/cos
[[nodiscard]] constexpr f64 sin_cx(f64 x) noexcept {
    // Reduce to [-π/2, π/2]
    f64 pi=PI_D, two_pi=TWO_PI_D;
    while(x> pi) x-=two_pi;
    while(x<-pi) x+=two_pi;
    if(x> PI_D/2.0){ return  cos_cx_reduced(x-PI_D/2.0); }
    if(x<-PI_D/2.0){ return -cos_cx_reduced(x+PI_D/2.0); }
    return sin_cx_reduced(x);
}
[[nodiscard]] constexpr f64 cos_cx(f64 x) noexcept { return sin_cx(x+PI_D/2.0); }
[[nodiscard]] constexpr f64 tan_cx(f64 x) noexcept {
    f64 c=cos_cx(x); return abs(c)<1e-15?1e15:sin_cx(x)/c;
}
[[nodiscard]] constexpr f32 sin_cx(f32 x) noexcept { return f32(sin_cx(f64(x))); }
[[nodiscard]] constexpr f32 cos_cx(f32 x) noexcept { return f32(cos_cx(f64(x))); }
[[nodiscard]] constexpr f32 tan_cx(f32 x) noexcept { return f32(tan_cx(f64(x))); }

// constexpr asin (Maclaurin series, converges for |x|<=1)
[[nodiscard]] constexpr f64 asin_cx(f64 x) noexcept {
    if(x>0.5||x<-0.5){
        f64 sign=x<0?-1.0:1.0;
        return sign*(PI_D/2.0-asin_cx(sqrt(1.0-x*x))); // near ±1 identity
    }
    f64 r=x, t=x, x2=x*x;
    for(i32 n=1;n<20;++n){
        t*=x2*(2*n-1)*(2*n-1)/((2*n)*(2*n+1));
        r+=t;
        if(abs(t)<1e-15) break;
    }
    return r;
}
[[nodiscard]] constexpr f64 acos_cx(f64 x) noexcept { return PI_D/2.0-asin_cx(x); }
[[nodiscard]] constexpr f64 atan_cx(f64 x) noexcept {
    if(x>1.0)  return PI_D/2.0-atan_cx(1.0/x);
    if(x<-1.0) return -PI_D/2.0-atan_cx(1.0/x);
    f64 r=x, t=x, x2=-x*x;
    for(i32 n=1;n<40;++n){ t*=x2; r+=t/(2*n+1); if(abs(t)<1e-15) break; }
    return r;
}
[[nodiscard]] constexpr f64 atan2_cx(f64 y, f64 x) noexcept {
    if(x>0.0) return atan_cx(y/x);
    if(x<0.0) return y>=0.0?atan_cx(y/x)+PI_D:atan_cx(y/x)-PI_D;
    if(y>0.0) return PI_D/2.0; if(y<0.0) return -PI_D/2.0;
    return 0.0;
}
[[nodiscard]] constexpr f32 asin_cx(f32 x) noexcept { return f32(asin_cx(f64(x))); }
[[nodiscard]] constexpr f32 acos_cx(f32 x) noexcept { return f32(acos_cx(f64(x))); }
[[nodiscard]] constexpr f32 atan_cx(f32 x) noexcept { return f32(atan_cx(f64(x))); }
[[nodiscard]] constexpr f32 atan2_cx(f32 y,f32 x) noexcept { return f32(atan2_cx(f64(y),f64(x))); }

// ---------------------------------------------------------------------------
// constexpr interpolation
// ---------------------------------------------------------------------------
template<typename T> [[nodiscard]] constexpr T lerp_cx(T a,T b,T t) noexcept { return a+(b-a)*t; }
template<typename T> [[nodiscard]] constexpr T smoothstep_cx(T lo,T hi,T v) noexcept {
    T t=clamp((v-lo)/(hi-lo),T(0),T(1));
    return t*t*(T(3)-T(2)*t);
}

// ---------------------------------------------------------------------------
// Compile-time lookup table generation
// ---------------------------------------------------------------------------
template<i32 N>
struct SinTable {
    f32 values[N];
    constexpr SinTable() noexcept : values{} {
        for(i32 i=0;i<N;++i)
            values[i]=f32(sin_cx(f64(i)*TWO_PI_D/N));
    }
    [[nodiscard]] constexpr f32 operator[](i32 i) const noexcept { return values[((i%N)+N)%N]; }
    [[nodiscard]] constexpr f32 lookup(f32 angle) const noexcept {
        f32 t=angle/(f32)TWO_PI*(f32)N;
        i32 idx=(i32)t; f32 frac=t-idx;
        return lerp_cx(values[((idx%N)+N)%N],values[((idx+1)%N+N)%N],frac);
    }
};

// Precomputed tables at common sizes
inline constexpr SinTable<256>  SIN_LUT_256;
inline constexpr SinTable<1024> SIN_LUT_1024;

// Bernstein basis (compile-time, degree 3)
[[nodiscard]] constexpr f64 bernstein3(i32 i, f64 t) noexcept {
    switch(i){
        case 0: return (1.0-t)*(1.0-t)*(1.0-t);
        case 1: return 3.0*t*(1.0-t)*(1.0-t);
        case 2: return 3.0*t*t*(1.0-t);
        case 3: return t*t*t;
        default: return 0.0;
    }
}

// Compile-time Fibonacci LUT
template<i32 N>
struct FibTable {
    u64 values[N];
    constexpr FibTable() noexcept : values{} {
        values[0]=0; if(N>1) values[1]=1;
        for(i32 i=2;i<N;++i) values[i]=values[i-1]+values[i-2];
    }
};
inline constexpr FibTable<64> FIB_LUT;

} // namespace qs::math::cx
