
// qs_math/numerical/numerical_module_impl.cpp (C++20)
#include "include/qs_math_numerical.hpp"
#include <cmath>
#include <cassert>
namespace qs::math::numerical {

// ─── Log Gamma (Lanczos approximation, g=7) ──────────────────────────────────
f64 gamma_ln(f64 x) noexcept {
    static constexpr f64 c[]={0.99999999999980993,676.5203681218851,-1259.1392167224028,
        771.32342877765313,-176.61502916214059,12.507343278686905,
        -0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7};
    if(x<0.5) return std::log(M_PI/std::sin(M_PI*x))-gamma_ln(1-x);
    x-=1; f64 a=c[0]; f64 t=x+7.5;
    for(i32 i=1;i<9;++i) a+=c[i]/(x+i);
    return 0.5*std::log(2*M_PI)+(x+0.5)*std::log(t)-t+std::log(a);
}
f64 gamma_fn(f64 x) noexcept { return std::exp(gamma_ln(x)); }
f64 beta_fn(f64 a, f64 b) noexcept { return std::exp(gamma_ln(a)+gamma_ln(b)-gamma_ln(a+b)); }

// ─── Erf (Horner polynomial approx, max error 1.2e-7) ────────────────────────
f64 erf_approx(f64 x) noexcept {
    const f64 t=1.0/(1.0+0.3275911*std::abs(x));
    const f64 p=t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))));
    const f64 r=1.0-p*std::exp(-x*x);
    return x<0?-r:r;
}
f64 erfc_approx(f64 x) noexcept { return 1.0-erf_approx(x); }

// ─── Digamma (psi function) ────────────────────────────────────────────────────
f64 digamma(f64 x) noexcept {
    if(x<0.5) return digamma(1-x)-M_PI/std::tan(M_PI*x);
    f64 r=0;
    while(x<7){r-=1.0/x;x+=1;}
    r+=std::log(x)-0.5/x;
    const f64 x2=1.0/(x*x);
    r-=x2*(1.0/12-x2*(1.0/120-x2/252));
    return r;
}

// ─── Bessel functions ─────────────────────────────────────────────────────────
f64 bessel_j0(f64 x) noexcept {
    if(std::abs(x)<8){
        const f64 y=x*x;
        const f64 p1=57568490574.0,p2=-13362590354.0,p3=651619640.7,p4=-11214424.18,p5=77392.33017,p6=-184.9052456;
        const f64 q1=57568490411.0,q2=1029532985.0,q3=9494680.718,q4=59272.64853,q5=267.8532712;
        return (p1+y*(p2+y*(p3+y*(p4+y*(p5+y*p6)))))/(q1+y*(q2+y*(q3+y*(q4+y*(q5+y)))));
    }
    const f64 z=8.0/x,y=z*z,xx=x-0.785398164;
    const f64 p1=1.0,p2=-0.1098628627e-2,p3=0.2734510407e-4,p4=-0.2073370639e-5,p5=0.2093887211e-6;
    const f64 q1=-0.1562499995e-1,q2=0.1430488765e-3,q3=-0.6911147651e-5,q4=0.7621095161e-6,q5=-0.934945152e-7;
    return std::sqrt(0.636619772/x)*(std::cos(xx)*(p1+y*(p2+y*(p3+y*(p4+y*p5))))-z*std::sin(xx)*(q1+y*(q2+y*(q3+y*(q4+y*q5)))));
}
f64 bessel_j1(f64 x) noexcept {
    if(std::abs(x)<8){
        const f64 y=x*x;
        const f64 p1=72362614232.0,p2=-7895059235.0,p3=242396853.1,p4=-2972611.439,p5=15704.4826,p6=-30.16036606;
        const f64 q1=144725228442.0,q2=2300535178.0,q3=18583304.74,q4=99447.43394,q5=376.9991397;
        return x*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*p6)))))/(q1+y*(q2+y*(q3+y*(q4+y*(q5+y)))));
    }
    const f64 z=8.0/x,y=z*z,xx=x-2.356194491;
    const f64 p1=1.0,p2=0.183105e-2,p3=-0.3516396496e-4,p4=0.2457520174e-5,p5=-0.240337019e-6;
    const f64 q1=0.04687499995,q2=-0.2002690873e-3,q3=0.8449199096e-5,q4=-0.88228987e-6,q5=0.105787412e-6;
    return std::sqrt(0.636619772/x)*(std::cos(xx)*(p1+y*(p2+y*(p3+y*(p4+y*p5))))-z*std::sin(xx)*(q1+y*(q2+y*(q3+y*(q4+y*q5)))));
}
f64 bessel_jn(i32 n, f64 x) noexcept {
    if(n==0) return bessel_j0(x); if(n==1) return bessel_j1(x);
    if(x==0) return 0;
    // Miller's backward recurrence
    const i32 NMAX=n+std::max(20,(i32)std::sqrt(40.0*n));
    f64 prev2=0,prev1=1e-30,cur=0;
    f64 sum=0;
    for(i32 j=NMAX;j>=1;--j){
        cur=2*j/x*prev1-prev2;
        prev2=prev1; prev1=cur;
        if(j==n) sum=cur;
    }
    // Normalise using J0
    return sum/prev1*bessel_j0(x);
}

// ─── Orthogonal polynomials ──────────────────────────────────────────────────
f64 legendre_p(i32 n, f64 x) noexcept {
    if(n==0)return 1; if(n==1)return x;
    f64 p0=1,p1=x,p2=0;
    for(i32 k=1;k<n;++k){p2=((2*k+1)*x*p1-k*p0)/(k+1);p0=p1;p1=p2;}
    return p2;
}
f64 chebyshev_t(i32 n, f64 x) noexcept {
    if(n==0)return 1; if(n==1)return x;
    f64 t0=1,t1=x,t2=0;
    for(i32 k=1;k<n;++k){t2=2*x*t1-t0;t0=t1;t1=t2;}
    return t2;
}
f64 laguerre(i32 n, f64 x) noexcept {
    if(n==0)return 1; if(n==1)return 1-x;
    f64 l0=1,l1=1-x,l2=0;
    for(i32 k=1;k<n;++k){l2=((2*k+1-x)*l1-k*l0)/(k+1);l0=l1;l1=l2;}
    return l2;
}
f64 hermite(i32 n, f64 x) noexcept {
    if(n==0)return 1; if(n==1)return 2*x;
    f64 h0=1,h1=2*x,h2=0;
    for(i32 k=1;k<n;++k){h2=2*x*h1-2*k*h0;h0=h1;h1=h2;}
    return h2;
}

// ─── Incomplete beta (continued fraction) ────────────────────────────────────
f64 inc_beta(f64 a, f64 b, f64 x) noexcept {
    if(x<0||x>1)return 0;
    if(x==0)return 0; if(x==1)return 1;
    const f64 lbeta=gamma_ln(a)+gamma_ln(b)-gamma_ln(a+b);
    const f64 front=std::exp(std::log(x)*a+std::log(1-x)*b-lbeta)/a;
    // Lentz continued fraction
    f64 h=1,C=h,D=0,delta;
    for(i32 m=0;m<200;++m){
        f64 Am=(m==0)?1:-((a+m)*(a+b+m)*x)/((a+2*m)*(a+2*m+1));
        D=1+Am*D; if(std::abs(D)<1e-30)D=1e-30; D=1/D;
        C=1+Am/C; if(std::abs(C)<1e-30)C=1e-30;
        delta=C*D; h*=delta;
        if(std::abs(delta-1)<1e-10) break;
    }
    return front*h;
}

f64 inc_gamma_lower(f64 a, f64 x) noexcept {
    if(x<0)return 0;
    // Series: gamma(a,x) = e^-x * x^a * sum_{n=0}^inf x^n / Gamma(a+n+1)
    f64 sum=1.0/a,term=1.0/a;
    for(i32 n=1;n<200;++n){term*=x/(a+n);sum+=term;if(std::abs(term)<1e-12*sum)break;}
    return sum*std::exp(-x+a*std::log(x))*std::exp(-gamma_ln(a+1));
}

f64 riemann_zeta(f64 s) noexcept {
    if(s<=1)return 1e100;
    // Euler-Maclaurin summation with 50 terms + correction
    f64 sum=0; const i32 N=50;
    for(i32 n=1;n<=N;++n) sum+=std::pow((f64)n,-s);
    // Asymptotic tail: integral + correction terms
    const f64 tail=std::pow((f64)N,1-s)/(s-1)+0.5*std::pow((f64)N,-s);
    return sum+tail;
}
f64 eta_fn(f64 s) noexcept { return (1-std::pow(2.0,1-s))*riemann_zeta(s); }

u64 fibonacci(i32 n) noexcept {
    if(n<=0)return 0; if(n==1)return 1;
    u64 a=0,b=1;
    for(i32 i=2;i<=n;++i){u64 c=a+b;a=b;b=c;}
    return b;
}
u64 catalan(i32 n) noexcept {
    u64 r=1;
    for(i32 i=0;i<n;++i){r=r*2*(2*i+1)/((i+2)*(i<1?1:i));}
    return binomial_coeff(2*n,n)/(n+1);
}
u64 binomial_coeff(i32 n, i32 k) noexcept {
    if(k<0||k>n)return 0; if(k==0||k==n)return 1;
    k=std::min(k,n-k); u64 r=1;
    for(i32 i=0;i<k;++i){r=r*(n-i)/(i+1);}
    return r;
}
f64 stirling_approx(i32 n) noexcept {
    return std::sqrt(2*M_PI*n)*std::pow((f64)n/M_E,(f64)n);
}
f64 continued_fraction(const f64* a, const f64* b, i32 n, f64 tol) noexcept {
    f64 h=b[0],C=h,D=0;
    for(i32 i=1;i<n;++i){
        D=b[i]+a[i]*D; if(std::abs(D)<1e-30)D=1e-30; D=1/D;
        C=b[i]+a[i]/C; if(std::abs(C)<1e-30)C=1e-30;
        const f64 delta=C*D; h*=delta;
        if(std::abs(delta-1)<tol)break;
    }
    return h;
}
} // namespace qs::math::numerical
