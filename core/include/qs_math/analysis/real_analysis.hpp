#pragma once
// qs_math/analysis/real_analysis.hpp — Sequences, series, limits, continuity, convergence (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <optional>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <array>

namespace qs {

// ─── Sequences ────────────────────────────────────────────────────────────────

// Generate first n terms of a sequence a(n)
[[nodiscard]] inline std::vector<f64> sequence(std::function<f64(u32)> a, u32 n) {
    std::vector<f64> v(n);
    for(u32 i=0;i<n;++i) v[i]=a(i);
    return v;
}

// Limit of sequence (numerical, using Cauchy criterion)
[[nodiscard]] inline std::optional<f64> sequence_limit(
    std::function<f64(u32)> a, u32 max_terms=10000, f64 eps=1e-12)
{
    f64 prev=a(100);
    for(u32 n=101;n<max_terms;++n){
        f64 cur=a(n);
        if(std::abs(cur-prev)<eps) return cur;
        prev=cur;
    }
    return std::nullopt; // did not converge
}

// Cauchy sequence check: |a_m - a_n| < eps for all m,n > N
[[nodiscard]] inline bool is_cauchy(std::function<f64(u32)> a, u32 N=100, f64 eps=1e-8) {
    f64 aN=a(N);
    for(u32 m=N+1;m<N+50;++m){
        f64 am=a(m);
        if(std::abs(am-aN)>eps) return false;
    }
    return true;
}

// Monotone sequence detection
[[nodiscard]] inline bool is_increasing(const std::vector<f64>& a) {
    for(u32 i=0;i+1<a.size();++i) if(a[i]>a[i+1]) return false;
    return true;
}
[[nodiscard]] inline bool is_decreasing(const std::vector<f64>& a) {
    for(u32 i=0;i+1<a.size();++i) if(a[i]<a[i+1]) return false;
    return true;
}
[[nodiscard]] inline bool is_bounded(const std::vector<f64>& a, f64 M) {
    for(f64 v:a) if(std::abs(v)>M) return false;
    return true;
}

// Subsequence at given indices
[[nodiscard]] inline std::vector<f64> subsequence(const std::vector<f64>& a, const std::vector<u32>& idx) {
    std::vector<f64> sub;
    for(u32 i:idx) if(i<a.size()) sub.push_back(a[i]);
    return sub;
}

// ─── Series ───────────────────────────────────────────────────────────────────

// Partial sums S_n = sum_{k=0}^{n} a(k)
[[nodiscard]] inline std::vector<f64> partial_sums(std::function<f64(u32)> a, u32 n) {
    std::vector<f64> S(n+1);
    f64 s=0.;
    for(u32 k=0;k<=n;++k){ s+=a(k); S[k]=s; }
    return S;
}

// Sum of series (numerical, with convergence check)
[[nodiscard]] inline std::optional<f64> series_sum(
    std::function<f64(u32)> a, u32 max_terms=100000, f64 eps=1e-15)
{
    f64 sum=0., prev_sum=std::numeric_limits<f64>::max();
    for(u32 k=0;k<max_terms;++k){
        f64 term=a(k);
        sum+=term;
        if(std::abs(sum-prev_sum)<eps&&k>10) return sum;
        prev_sum=sum;
    }
    return std::nullopt;
}

// Convergence tests

// Ratio test: L = lim |a_{n+1}/a_n|; L<1 => converges
[[nodiscard]] inline std::optional<bool> ratio_test(std::function<f64(u32)> a, u32 n=1000) {
    f64 prev=std::abs(a(n));
    if(prev<1e-300) return true;
    f64 L=std::abs(a(n+1))/prev;
    if(L<1.-1e-6) return true;
    if(L>1.+1e-6) return false;
    return std::nullopt; // inconclusive
}

// Root test: L = lim |a_n|^(1/n)
[[nodiscard]] inline std::optional<bool> root_test(std::function<f64(u32)> a, u32 n=1000) {
    f64 L=std::pow(std::abs(a(n)),1./f64(n));
    if(L<1.-1e-6) return true;
    if(L>1.+1e-6) return false;
    return std::nullopt;
}

// Comparison test: |a_n| <= b_n and sum b_n converges => sum a_n converges
[[nodiscard]] inline bool comparison_test(
    std::function<f64(u32)> a, std::function<f64(u32)> b, u32 n_terms=1000)
{
    for(u32 k=0;k<n_terms;++k) if(std::abs(a(k))>b(k)) return false;
    return true;
}

// Alternating series (Leibniz): terms decreasing to 0 => converges
[[nodiscard]] inline bool alternating_series_test(std::function<f64(u32)> b, u32 n=100) {
    // b must be positive, decreasing, limit 0
    for(u32 k=0;k<n;++k) if(b(k+1)>=b(k)||b(k)<0.) return false;
    return std::abs(b(n))<1e-8;
}

// ─── Famous series ────────────────────────────────────────────────────────────

// Basel problem: sum 1/n^2 = pi^2/6
[[nodiscard]] inline f64 basel_series_partial(u32 n) {
    f64 s=0.; for(u32 k=1;k<=n;++k) s+=1./f64(k*k); return s;
}

// Harmonic series (diverges)
[[nodiscard]] inline f64 harmonic_partial(u32 n) {
    f64 s=0.; for(u32 k=1;k<=n;++k) s+=1./f64(k); return s;
}

// Euler-Maclaurin formula for sum approximation
[[nodiscard]] inline f64 euler_maclaurin_correction(
    std::function<f64(f64)> f, u32 a, u32 b, u32 p=4)
{
    // Sum_{n=a}^{b} f(n) ≈ integral + (f(a)+f(b))/2 + Bernoulli corrections
    // Bernoulli numbers B_2=1/6, B_4=-1/30, B_6=1/42
    static constexpr f64 B[5]={1.,-.5,1./6.,0.,-1./30.};
    f64 corr=(f(f64(a))+f(f64(b)))*0.5;
    f64 h=1.;
    // First correction
    if(p>=2){
        auto fd=[&](f64 x,u32 d)->f64{
            f64 ep=1e-5;
            if(d==1) return (f(x+ep)-f(x-ep))/(2.*ep);
            return (f(x+ep)-2.*f(x)+f(x-ep))/(ep*ep);
        };
        corr+=B[2]/2.*(fd(f64(b),1)-fd(f64(a),1));
    }
    return corr;
}

// ─── Limits ───────────────────────────────────────────────────────────────────

// One-sided limit from the right: lim_{x→c+} f(x)
[[nodiscard]] inline std::optional<f64> limit_right(
    std::function<f64(f64)> f, f64 c, f64 eps=1e-10)
{
    f64 prev=f(c+1.);
    for(u32 k=1;k<=50;++k){
        f64 h=std::pow(0.1,f64(k));
        f64 cur=f(c+h);
        if(std::isnan(cur)||std::isinf(cur)) return std::nullopt;
        if(std::abs(cur-prev)<eps) return cur;
        prev=cur;
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<f64> limit_left(
    std::function<f64(f64)> f, f64 c, f64 eps=1e-10)
{
    f64 prev=f(c-1.);
    for(u32 k=1;k<=50;++k){
        f64 h=std::pow(0.1,f64(k));
        f64 cur=f(c-h);
        if(std::isnan(cur)||std::isinf(cur)) return std::nullopt;
        if(std::abs(cur-prev)<eps) return cur;
        prev=cur;
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<f64> limit(std::function<f64(f64)> f, f64 c, f64 eps=1e-10) {
    auto L=limit_right(f,c,eps);
    auto R=limit_left(f,c,eps);
    if(!L||!R) return std::nullopt;
    if(std::abs(*L-*R)<eps*100.) return (*L+*R)*0.5;
    return std::nullopt; // different one-sided limits
}

// Limit at infinity
[[nodiscard]] inline std::optional<f64> limit_at_infinity(
    std::function<f64(f64)> f, f64 eps=1e-10, bool positive=true)
{
    f64 sign=positive?1.:-1.;
    f64 prev=f(sign*100.);
    for(u32 k=3;k<=20;++k){
        f64 cur=f(sign*std::pow(10.,f64(k)));
        if(std::isnan(cur)||std::isinf(cur)) return std::nullopt;
        if(std::abs(cur-prev)<eps) return cur;
        prev=cur;
    }
    return std::nullopt;
}

// L'Hôpital's rule (numerical): applies when f(c)/g(c) = 0/0 or ∞/∞
[[nodiscard]] inline std::optional<f64> lhopital(
    std::function<f64(f64)> f, std::function<f64(f64)> g, f64 c, u32 max_iter=5)
{
    auto deriv=[](std::function<f64(f64)> h, f64 x)->f64{
        f64 ep=1e-7; return (h(x+ep)-h(x-ep))/(2.*ep);};
    std::function<f64(f64)> fn=f, gn=g;
    for(u32 i=0;i<max_iter;++i){
        auto fn_copy=fn, gn_copy=gn;
        f64 gval=gn_copy(c);
        if(std::abs(gval)>1e-10) return fn_copy(c)/gval;
        fn=[fn_copy,deriv,&c](f64 x){return deriv(fn_copy,x);};
        gn=[gn_copy,deriv,&c](f64 x){return deriv(gn_copy,x);};
    }
    return std::nullopt;
}

// ─── Continuity ───────────────────────────────────────────────────────────────

// Check continuity at point c
[[nodiscard]] inline bool is_continuous_at(
    std::function<f64(f64)> f, f64 c, f64 eps=1e-8)
{
    auto L=limit(f,c,1e-10);
    if(!L) return false;
    f64 fc=f(c);
    if(std::isnan(fc)) return false;
    return std::abs(*L-fc)<eps;
}

// Lipschitz constant estimation on [a,b]
[[nodiscard]] inline f64 lipschitz_constant(
    std::function<f64(f64)> f, f64 a, f64 b, u32 samples=1000)
{
    f64 L=0.;
    f64 h=(b-a)/f64(samples);
    for(u32 i=0;i<samples;++i){
        f64 x=a+f64(i)*h, y=x+h;
        f64 lip=std::abs(f(y)-f(x))/h;
        L=std::max(L,lip);
    }
    return L;
}

// Uniform continuity check on [a,b]: for all eps>0 exists delta>0 such that |x-y|<delta => |f(x)-f(y)|<eps
[[nodiscard]] inline bool is_uniformly_continuous(
    std::function<f64(f64)> f, f64 a, f64 b, f64 eps=1e-6, u32 samples=500)
{
    f64 h=(b-a)/f64(samples);
    for(u32 i=0;i<samples;++i)
    for(u32 j=i+1;j<samples&&j<i+10;++j){
        f64 x=a+f64(i)*h, y=a+f64(j)*h;
        if(std::abs(f(x)-f(y))>eps&&std::abs(x-y)<eps*0.01) return false;
    }
    return true;
}

// Intermediate Value Theorem: find c in [a,b] where f(c)=0
[[nodiscard]] inline std::optional<f64> ivt_zero(
    std::function<f64(f64)> f, f64 a, f64 b, f64 tol=1e-12)
{
    f64 fa=f(a), fb=f(b);
    if(fa*fb>0.) return std::nullopt;
    for(u32 i=0;i<100;++i){
        f64 c=(a+b)*0.5, fc=f(c);
        if(std::abs(fc)<tol||std::abs(b-a)<tol) return c;
        if(fa*fc<0.){b=c;fb=fc;} else {a=c;fa=fc;}
    }
    return (a+b)*0.5;
}

// ─── Compactness (Heine-Borel on R) ──────────────────────────────────────────

// [a,b] is compact: closed and bounded
[[nodiscard]] inline bool is_compact_interval(f64 a, f64 b) {
    return std::isfinite(a)&&std::isfinite(b)&&a<=b;
}

// Bolzano-Weierstrass: every bounded sequence has a convergent subsequence
[[nodiscard]] inline std::optional<f64> bolzano_weierstrass(
    const std::vector<f64>& seq, f64 lo, f64 hi)
{
    // Find a cluster point via bisection
    for(u32 iter=0;iter<60;++iter){
        f64 mid=(lo+hi)*0.5;
        u32 cnt_lo=0, cnt_hi=0;
        for(f64 v:seq){ if(v>=lo&&v<mid)++cnt_lo; if(v>=mid&&v<=hi)++cnt_hi; }
        if(cnt_lo>=cnt_hi) hi=mid; else lo=mid;
        if(hi-lo<1e-12) return (lo+hi)*0.5;
    }
    return std::nullopt;
}

// ─── Power series ─────────────────────────────────────────────────────────────

struct PowerSeries {
    std::vector<f64> coeffs; // coeffs[n] for x^n
    f64 center{0.};

    [[nodiscard]] f64 eval(f64 x, u32 max_terms=0) const {
        u32 n=max_terms?std::min(max_terms,u32(coeffs.size())):u32(coeffs.size());
        f64 s=0., xpow=1., xc=x-center;
        for(u32 k=0;k<n;++k){ s+=coeffs[k]*xpow; xpow*=xc; }
        return s;
    }

    // Radius of convergence via Cauchy-Hadamard: 1/R = limsup |a_n|^(1/n)
    [[nodiscard]] f64 radius_of_convergence() const {
        f64 limsup=0.;
        u32 n=u32(coeffs.size());
        for(u32 k=n/2;k<n;++k)
            limsup=std::max(limsup,std::pow(std::abs(coeffs[k]),1./f64(k+1)));
        return limsup>0.?1./limsup:std::numeric_limits<f64>::infinity();
    }

    // Derivative: d/dx sum a_n x^n = sum n*a_n x^(n-1)
    [[nodiscard]] PowerSeries derivative() const {
        if(coeffs.size()<=1) return {{0.},center};
        PowerSeries d; d.center=center; d.coeffs.resize(coeffs.size()-1);
        for(u32 n=1;n<coeffs.size();++n) d.coeffs[n-1]=f64(n)*coeffs[n];
        return d;
    }

    // Antiderivative: integral of sum a_n x^n = C + sum a_n/(n+1) x^(n+1)
    [[nodiscard]] PowerSeries antiderivative(f64 C=0.) const {
        PowerSeries a; a.center=center; a.coeffs.resize(coeffs.size()+1);
        a.coeffs[0]=C;
        for(u32 n=0;n<coeffs.size();++n) a.coeffs[n+1]=coeffs[n]/f64(n+1);
        return a;
    }

    // Taylor series for common functions
    [[nodiscard]] static PowerSeries exp_series(u32 terms=20) {
        PowerSeries p; p.coeffs.resize(terms);
        f64 fact=1.;
        for(u32 n=0;n<terms;++n){ if(n>0)fact*=f64(n); p.coeffs[n]=1./fact; }
        return p;
    }

    [[nodiscard]] static PowerSeries sin_series(u32 terms=20) {
        PowerSeries p; p.coeffs.resize(2*terms,0.);
        f64 fact=1.,sign=1.;
        for(u32 n=0;n<terms;++n){
            if(n>0){ fact*=f64(2*n-1)*f64(2*n); sign=-sign; }
            p.coeffs[2*n+1]=sign/fact;
        }
        return p;
    }

    [[nodiscard]] static PowerSeries cos_series(u32 terms=20) {
        PowerSeries p; p.coeffs.resize(2*terms,0.);
        f64 fact=1.,sign=1.;
        for(u32 n=0;n<terms;++n){
            if(n>0){ fact*=f64(2*n-1)*f64(2*n); sign=-sign; }
            p.coeffs[2*n]=sign/fact;
        }
        return p;
    }

    [[nodiscard]] static PowerSeries ln_series(u32 terms=30) {
        // ln(1+x) = x - x^2/2 + x^3/3 - ...
        PowerSeries p; p.coeffs.resize(terms+1,0.);
        for(u32 n=1;n<=terms;++n) p.coeffs[n]=(n%2==0?-1.:1.)/f64(n);
        return p;
    }

    [[nodiscard]] static PowerSeries geometric_series() {
        // 1/(1-x) = 1 + x + x^2 + ...
        // All coefficients = 1, R=1
        PowerSeries p; p.coeffs.assign(100,1.); return p;
    }
};

// ─── Uniform convergence ─────────────────────────────────────────────────────

// Check if sequence of functions converges uniformly on [a,b]
[[nodiscard]] inline bool converges_uniformly(
    std::function<f64(u32,f64)> fn, // fn(n, x) = f_n(x)
    f64 limit_val_fn,               // approximate limit
    f64 a, f64 b, u32 N=100, f64 eps=1e-6, u32 x_samples=50)
{
    f64 h=(b-a)/f64(x_samples);
    for(u32 xi=0;xi<=x_samples;++xi){
        f64 x=a+f64(xi)*h;
        if(std::abs(fn(N,x)-limit_val_fn)>eps) return false;
    }
    return true;
}

} // namespace qs
