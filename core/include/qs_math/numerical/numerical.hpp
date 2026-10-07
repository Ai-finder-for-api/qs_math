#pragma once
// qs_math/numerical.hpp — Root solving, Newton, Bisection, Gradient, Finite differences, Integration
#include "scalar_math.hpp"
#include <functional>

namespace qs::math::numerical {

// ---------------------------------------------------------------------------
// Root finding
// ---------------------------------------------------------------------------
struct SolverResult {
    f64 root{0.0};
    i32 iterations{0};
    bool converged{false};
};

// Bisection — guaranteed convergence, slow
template<typename Fn>
[[nodiscard]] SolverResult bisection(Fn&& f, f64 a, f64 b, f64 tol=1e-10, i32 max_iter=100) noexcept {
    SolverResult r;
    f64 fa=f(a);
    for(i32 i=0;i<max_iter;++i){
        f64 c=(a+b)*0.5;
        f64 fc=f(c);
        ++r.iterations;
        if(abs(fc)<tol||abs(b-a)<tol){ r.root=c; r.converged=true; return r; }
        if(fa*fc<0.0){b=c;} else {a=c;fa=fc;}
    }
    r.root=(a+b)*0.5; return r;
}

// Newton-Raphson — fast near root, may diverge
template<typename Fn, typename DFn>
[[nodiscard]] SolverResult newton(Fn&& f, DFn&& df, f64 x0, f64 tol=1e-10, i32 max_iter=50) noexcept {
    SolverResult r; r.root=x0;
    for(i32 i=0;i<max_iter;++i){
        f64 fx=f(r.root), dfx=df(r.root);
        ++r.iterations;
        if(abs(dfx)<1e-14) break;
        r.root-=fx/dfx;
        if(abs(fx)<tol){ r.converged=true; return r; }
    }
    return r;
}

// Secant method — Newton without analytical derivative
template<typename Fn>
[[nodiscard]] SolverResult secant(Fn&& f, f64 x0, f64 x1, f64 tol=1e-10, i32 max_iter=50) noexcept {
    SolverResult r; r.root=x1;
    f64 f0=f(x0),f1=f(x1);
    for(i32 i=0;i<max_iter;++i){
        ++r.iterations;
        if(abs(f1-f0)<1e-14) break;
        f64 x2=x1-f1*(x1-x0)/(f1-f0);
        x0=x1; f0=f1; x1=x2; f1=f(x1); r.root=x1;
        if(abs(f1)<tol){ r.converged=true; return r; }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Numerical differentiation
// ---------------------------------------------------------------------------
template<typename Fn>
[[nodiscard]] inline f64 derivative(Fn&& f, f64 x, f64 h=1e-5) noexcept {
    return (f(x+h)-f(x-h))/(2.0*h);
}

template<typename Fn>
[[nodiscard]] inline f64 derivative2(Fn&& f, f64 x, f64 h=1e-5) noexcept {
    return (f(x+h)-2.0*f(x)+f(x-h))/(h*h);
}

// Gradient of f:R^n -> R by finite differences
template<typename Fn>
inline void gradient(Fn&& f, const f64* x, f64* grad, i32 n, f64 h=1e-5) noexcept {
    // Allocate on stack (n <= 32 assumed)
    f64 xp[32], xm[32];
    for(i32 i=0;i<n&&i<32;++i){ xp[i]=xm[i]=x[i]; }
    for(i32 i=0;i<n&&i<32;++i){
        xp[i]+=h; xm[i]-=h;
        grad[i]=(f(xp)-f(xm))/(2.0*h);
        xp[i]=xm[i]=x[i];
    }
}

// Jacobian (n inputs, m outputs)
template<typename Fn>
inline void jacobian(Fn&& f, const f64* x, f64* jac, i32 n, i32 m, f64 h=1e-5) noexcept {
    f64 xp[32]{}, xm[32]{}, fp[32]{}, fm[32]{};
    for(i32 i=0;i<n&&i<32;++i){ xp[i]=xm[i]=x[i]; }
    for(i32 j=0;j<n&&j<32;++j){
        xp[j]+=h; xm[j]-=h;
        f(xp,fp); f(xm,fm);
        for(i32 i=0;i<m&&i<32;++i) jac[i*n+j]=(fp[i]-fm[i])/(2.0*h);
        xp[j]=xm[j]=x[j];
    }
}

// ---------------------------------------------------------------------------
// Numerical integration
// ---------------------------------------------------------------------------
// Trapezoidal rule
template<typename Fn>
[[nodiscard]] inline f64 integrate_trapezoid(Fn&& f, f64 a, f64 b, i32 n=100) noexcept {
    f64 h=(b-a)/n, sum=f(a)+f(b);
    for(i32 i=1;i<n;++i) sum+=2.0*f(a+i*h);
    return sum*h*0.5;
}

// Simpson's rule (n must be even)
template<typename Fn>
[[nodiscard]] inline f64 integrate_simpson(Fn&& f, f64 a, f64 b, i32 n=100) noexcept {
    if(n%2!=0) ++n;
    f64 h=(b-a)/n, sum=f(a)+f(b);
    for(i32 i=1;i<n;++i) sum+=(i%2==0?2.0:4.0)*f(a+i*h);
    return sum*h/3.0;
}

// Gaussian quadrature (5-point Gauss-Legendre)
template<typename Fn>
[[nodiscard]] inline f64 integrate_gauss5(Fn&& f, f64 a, f64 b) noexcept {
    static constexpr f64 w[]={0.2369268851,0.4786286705,0.5688888889,0.4786286705,0.2369268851};
    static constexpr f64 x[]={-0.9061798459,-0.5384693101,0.0,0.5384693101,0.9061798459};
    f64 sum=0.0, mid=(a+b)*0.5, half=(b-a)*0.5;
    for(i32 i=0;i<5;++i) sum+=w[i]*f(mid+half*x[i]);
    return half*sum;
}

// Monte Carlo integration
template<typename Fn>
[[nodiscard]] f64 integrate_monte_carlo(Fn&& f, f64 a, f64 b, i32 samples, u64 seed=42) noexcept {
    u64 s=seed; f64 sum=0.0;
    for(i32 i=0;i<samples;++i){
        s^=s<<13;s^=s>>7;s^=s<<17;
        f64 x=a+(b-a)*(f64(s>>11)*0x1.0p-53);
        sum+=f(x);
    }
    return (b-a)*sum/samples;
}

} // namespace qs::math::numerical
