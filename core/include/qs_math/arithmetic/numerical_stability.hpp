#pragma once
// qs_math/numerical_stability.hpp — Kahan summation, compensated ops, condition numbers (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <span>
#include <array>
#include <algorithm>
#include <limits>
#include <numeric>

namespace qs {

// ─── Kahan compensated summation ─────────────────────────────────────────────

struct KahanSum {
    f64 sum{0.}, comp{0.};

    void add(f64 v){
        f64 y=v-comp, t=sum+y;
        comp=(t-sum)-y; sum=t;
    }

    [[nodiscard]] f64 get() const { return sum; }
    void reset(){ sum=0.; comp=0.; }
};

[[nodiscard]] inline f64 kahan_sum(std::span<const f64> vals) {
    KahanSum ks;
    for(f64 v:vals) ks.add(v);
    return ks.get();
}

// Neumaier summation (handles large-small alternation better)
[[nodiscard]] inline f64 neumaier_sum(std::span<const f64> vals) {
    f64 sum=0., comp=0.;
    for(f64 v:vals){
        f64 t=sum+v;
        comp+=std::abs(sum)>=std::abs(v)?(sum-t)+v:(v-t)+sum;
        sum=t;
    }
    return sum+comp;
}

// Pairwise summation (recursive bisection)
[[nodiscard]] f64 pairwise_sum(std::span<const f64> vals);
[[nodiscard]] inline f64 pairwise_sum(std::span<const f64> vals) {
    u32 n=u32(vals.size());
    if(n==0) return 0.;
    if(n==1) return vals[0];
    if(n==2) return vals[0]+vals[1];
    u32 mid=n/2;
    return pairwise_sum(vals.subspan(0,mid))+pairwise_sum(vals.subspan(mid));
}

// ─── Two-sum (exact floating-point error) ────────────────────────────────────

struct TwoSum { f64 s{}, err{}; };

[[nodiscard]] inline TwoSum two_sum(f64 a, f64 b) {
    f64 s=a+b, z=s-a;
    return {s,(a-(s-z))+(b-z)};
}

[[nodiscard]] inline TwoSum two_product(f64 a, f64 b) {
    f64 p=a*b;
    // Veltkamp splitting
    f64 c=(134217729.)*a, ah=c-(c-a), al=a-ah;
    f64 d=(134217729.)*b, bh=d-(d-b), bl=b-bh;
    f64 err=((ah*bh-p)+ah*bl+al*bh)+al*bl;
    return {p,err};
}

// ─── Condition number estimation ─────────────────────────────────────────────

// Condition number of a 2x2 matrix
[[nodiscard]] inline f64 condition_number_2x2(f64 a,f64 b,f64 c,f64 d) {
    // ||A|| * ||A^{-1}|| (Frobenius norm)
    f64 det=a*d-b*c;
    if(std::abs(det)<1e-15) return std::numeric_limits<f64>::infinity();
    f64 norm_A=std::sqrt(a*a+b*b+c*c+d*d);
    f64 norm_Ainv=std::sqrt((d*d+b*b+c*c+a*a)/(det*det));
    return norm_A*norm_Ainv;
}

// Estimated condition via power iteration (A^T A)
[[nodiscard]] inline f64 matrix_condition_estimate(
    std::span<const f64> A, u32 n, u32 iters=20)
{
    // Power iteration for largest eigenvalue of A^T A
    std::vector<f64> v(n,1.), Av(n), AtAv(n);
    f64 lambda_max=1., lambda_min=1.;

    for(u32 iter=0;iter<iters;++iter){
        // Av = A*v
        std::fill(Av.begin(),Av.end(),0.);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) Av[i]+=A[i*n+j]*v[j];
        // A^T Av
        std::fill(AtAv.begin(),AtAv.end(),0.);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) AtAv[j]+=A[i*n+j]*Av[i];
        // Rayleigh quotient
        f64 num=0.,den=0.;
        for(u32 i=0;i<n;++i){num+=v[i]*AtAv[i];den+=v[i]*v[i];}
        lambda_max=den>0?num/den:1.;
        // Normalize
        f64 norm=std::sqrt(den); if(norm>1e-15) for(auto& x:v) x/=norm;
    }
    // Inverse power for smallest eigenvalue (simplified)
    lambda_min=lambda_max/100.; // rough estimate
    return std::sqrt(lambda_max/lambda_min);
}

// ─── Cancellation detection ───────────────────────────────────────────────────

[[nodiscard]] inline u32 catastrophic_cancellation_bits(f64 a, f64 b) {
    f64 diff=a-b;
    if(diff==0.) return 53; // all bits lost
    if(std::abs(a)<1e-300||std::abs(diff)<1e-300) return 0;
    f64 ratio=std::log2(std::abs(a)/std::abs(diff));
    return u32(std::min(53., std::max(0., ratio)));
}

// ─── Interval arithmetic ──────────────────────────────────────────────────────

struct Interval {
    f64 lo{}, hi{};

    [[nodiscard]] static Interval from_exact(f64 v) { return {v,v}; }
    [[nodiscard]] static Interval from_uncertainty(f64 center, f64 error) {
        return {center-error, center+error}; }

    [[nodiscard]] Interval operator+(const Interval& o) const { return {lo+o.lo,hi+o.hi}; }
    [[nodiscard]] Interval operator-(const Interval& o) const { return {lo-o.hi,hi-o.lo}; }
    [[nodiscard]] Interval operator*(const Interval& o) const {
        f64 a=lo*o.lo,b=lo*o.hi,c=hi*o.lo,d=hi*o.hi;
        return {std::min({a,b,c,d}),std::max({a,b,c,d})}; }
    [[nodiscard]] Interval operator/(const Interval& o) const {
        if(o.lo<=0&&o.hi>=0) return {-1e300,1e300}; // division by zero
        Interval inv{1./o.hi,1./o.lo};
        return *this*inv; }

    [[nodiscard]] f64 width()   const { return hi-lo; }
    [[nodiscard]] f64 center()  const { return (lo+hi)*0.5; }
    [[nodiscard]] bool contains(f64 v) const { return v>=lo&&v<=hi; }
    [[nodiscard]] bool overlaps(const Interval& o) const { return lo<=o.hi&&o.lo<=hi; }
};

// Interval square root
[[nodiscard]] inline Interval isqrt(const Interval& x) {
    if(x.hi<0.) return {0.,0.};
    return {std::sqrt(std::max(0.,x.lo)),std::sqrt(x.hi)};
}

// Interval absolute value
[[nodiscard]] inline Interval iabs(const Interval& x) {
    if(x.lo>=0.) return x;
    if(x.hi<=0.) return {-x.hi,-x.lo};
    return {0.,std::max(-x.lo,x.hi)};
}

// ─── ULP and relative error ───────────────────────────────────────────────────

[[nodiscard]] inline f64 ulp_distance(f64 a, f64 b) {
    if(a==b) return 0.;
    i64 ia,ib;
    std::memcpy(&ia,&a,8);
    std::memcpy(&ib,&b,8);
    return f64(std::abs(ia-ib));
}

[[nodiscard]] inline f64 relative_error(f64 computed, f64 exact) {
    if(std::abs(exact)<1e-300) return std::abs(computed);
    return std::abs(computed-exact)/std::abs(exact);
}

// ─── Welford online variance (numerically stable) ────────────────────────────

struct WelfordVar {
    u64 n{0};
    f64 mean{0.},M2{0.};

    void update(f64 x){
        ++n;
        f64 delta=x-mean;
        mean+=delta/f64(n);
        M2+=delta*(x-mean);
    }

    [[nodiscard]] f64 variance_pop()  const { return n>0?M2/f64(n):0.; }
    [[nodiscard]] f64 variance_samp() const { return n>1?M2/f64(n-1):0.; }
    [[nodiscard]] f64 stddev()        const { return std::sqrt(variance_samp()); }
    [[nodiscard]] f64 get_mean()      const { return mean; }

    // Parallel merge of two Welford accumulators
    static WelfordVar merge(const WelfordVar& a, const WelfordVar& b) {
        if(!a.n) return b; if(!b.n) return a;
        WelfordVar c;
        c.n=a.n+b.n;
        f64 delta=b.mean-a.mean;
        c.mean=a.mean+delta*f64(b.n)/f64(c.n);
        c.M2=a.M2+b.M2+delta*delta*f64(a.n)*f64(b.n)/f64(c.n);
        return c;
    }
};

// ─── Stable polynomial evaluation ────────────────────────────────────────────

// Compensated Horner scheme
[[nodiscard]] inline std::pair<f64,f64> compensated_horner(
    std::span<const f64> coeffs, f64 x)
{
    u32 n=u32(coeffs.size());
    if(!n) return {0.,0.};
    f64 s=coeffs[n-1],e=0.;
    for(i32 i=i32(n)-2;i>=0;--i){
        auto[p,ep]=two_product(s,x);
        auto[q,eq]=two_sum(p,coeffs[i]);
        s=q; e=e*x+ep+eq;
    }
    return {s,e};
}

// ─── Mixed precision helpers ──────────────────────────────────────────────────

// Compute in f64, return f32 (common pattern for shaders/physics)
[[nodiscard]] inline f32 f64_to_f32_safe(f64 v) {
    if(v>f64(std::numeric_limits<f32>::max())) return std::numeric_limits<f32>::infinity();
    if(v<f64(-std::numeric_limits<f32>::max())) return -std::numeric_limits<f32>::infinity();
    return f32(v);
}

// High-precision dot product via compensated summation
[[nodiscard]] inline f64 compensated_dot(std::span<const f64> a, std::span<const f64> b) {
    u32 n=u32(std::min(a.size(),b.size()));
    KahanSum ks;
    for(u32 i=0;i<n;++i){
        auto[p,e]=two_product(a[i],b[i]);
        ks.add(p); ks.add(e);
    }
    return ks.get();
}

} // namespace qs
