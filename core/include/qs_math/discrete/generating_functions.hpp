#pragma once
// qs_math/discrete/generating_functions.hpp — OGF/EGF/Dirichlet series arithmetic
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cstring>

namespace qs::math {

// ─── Formal power series mod x^N ─────────────────────────────────────────────
template<int N>
struct FPS {
    std::array<f64,N> a{};
    [[nodiscard]] constexpr f64  operator[](int i) const noexcept { return a[i]; }
    [[nodiscard]] constexpr f64& operator[](int i)       noexcept { return a[i]; }
};

template<int N>
[[nodiscard]] constexpr FPS<N> fps_add(const FPS<N>& f, const FPS<N>& g) noexcept {
    FPS<N> r; for(int i=0;i<N;++i) r[i]=f[i]+g[i]; return r;
}

template<int N>
[[nodiscard]] constexpr FPS<N> fps_sub(const FPS<N>& f, const FPS<N>& g) noexcept {
    FPS<N> r; for(int i=0;i<N;++i) r[i]=f[i]-g[i]; return r;
}

template<int N>
[[nodiscard]] constexpr FPS<N> fps_mul(const FPS<N>& f, const FPS<N>& g) noexcept {
    FPS<N> r{};
    for(int i=0;i<N;++i)
        for(int j=0;j+i<N;++j)
            r[i+j]+=f[i]*g[j];
    return r;
}

// Multiplicative inverse mod x^N: result[0] must be nonzero
template<int N>
[[nodiscard]] constexpr FPS<N> fps_inv(const FPS<N>& f) noexcept {
    FPS<N> r{};
    r[0] = 1.0/f[0];
    for(int i=1;i<N;++i) {
        f64 s=0;
        for(int j=1;j<=i;++j) s+=f[j]*r[i-j];
        r[i]=-s*r[0];
    }
    return r;
}

// Composition g(f(x)) mod x^N, f[0]=0 required
template<int N>
[[nodiscard]] constexpr FPS<N> fps_compose(const FPS<N>& g, const FPS<N>& f) noexcept {
    FPS<N> result{}, f_pow{}; f_pow[0]=1.0;
    for(int k=0;k<N;++k) {
        for(int i=0;i<N;++i) result[i]+=g[k]*f_pow[i];
        f_pow=fps_mul(f_pow,f);
    }
    return result;
}

// Formal derivative
template<int N>
[[nodiscard]] constexpr FPS<N> fps_deriv(const FPS<N>& f) noexcept {
    FPS<N> r{};
    for(int i=1;i<N;++i) r[i-1]=f64(i)*f[i];
    return r;
}

// Formal integral (constant=0)
template<int N>
[[nodiscard]] constexpr FPS<N> fps_integral(const FPS<N>& f) noexcept {
    FPS<N> r{};
    for(int i=0;i<N-1;++i) r[i+1]=f[i]/f64(i+1);
    return r;
}

// Formal log: log(f) = integral(f'/f), requires f[0]=1
template<int N>
[[nodiscard]] constexpr FPS<N> fps_log(const FPS<N>& f) noexcept {
    return fps_integral(fps_mul(fps_deriv(f), fps_inv(f)));
}

// Formal exp: via Newton iteration exp(g)=h where log(h)-g=0
template<int N>
[[nodiscard]] constexpr FPS<N> fps_exp(const FPS<N>& g) noexcept {
    // Requires g[0]=0
    FPS<N> h{}; h[0]=1.0;
    // Newton: h_{k+1} = h_k*(1 - log(h_k) + g)
    for(int iter=0;iter<12;++iter) {
        FPS<N> lh = fps_log(h);
        FPS<N> delta = fps_sub(g, lh); delta[0]+=1.0;
        h = fps_mul(h, delta);
    }
    return h;
}

// FPS power: f^r using exp(r*log(f)), f[0]=1
template<int N>
[[nodiscard]] constexpr FPS<N> fps_pow(const FPS<N>& f, f64 r) noexcept {
    FPS<N> lf = fps_log(f);
    for(int i=0;i<N;++i) lf[i]*=r;
    return fps_exp(lf);
}

// ─── OGF: partition function p(n) ────────────────────────────────────────────
// p(n) = number of integer partitions of n
// Generating function: prod_{k>=1} 1/(1-x^k)
template<int N>
[[nodiscard]] constexpr FPS<N> partition_ogf() noexcept {
    FPS<N> f{}; f[0]=1.0;
    for(int k=1;k<N;++k) {
        FPS<N> factor{}; factor[0]=1.0;
        // 1/(1-x^k) = 1 + x^k + x^{2k} + ...
        FPS<N> geom{}; geom[0]=1.0;
        for(int j=k;j<N;j+=k) geom[j]=-1.0; // (1-x^k) as denominator
        // multiply f by 1/(1-x^k)
        // Use geometric series accumulation instead
        for(int j=k;j<N;++j) f[j]+=f[j-k];
    }
    return f;
}

// ─── EGF: exponential generating function for permutations/derangements ───────
// Derangements: D(n) = n! * [x^n] e^{-x}/(1-x)
[[nodiscard]] constexpr i64 derangements(int n) noexcept {
    if(n==0) return 1; if(n==1) return 0;
    i64 d0=1,d1=0;
    for(int i=2;i<=n;++i){i64 t=(i-1)*(d0+d1);d0=d1;d1=t;}
    return d1;
}

// Bell numbers via EGF: B(n) = n! [x^n] exp(e^x - 1)
[[nodiscard]] constexpr i64 bell_number(int n) noexcept {
    // Via Bell triangle
    std::array<i64,64> row{};
    row[0]=1;
    for(int i=1;i<=n;++i) {
        i64 prev=row[i-1];
        row[0]=prev;
        for(int j=1;j<=i;++j) row[j]=row[j-1]+row[j-1]; // corrected below
        // Reset and do properly
        i64 r0=row[0];
        for(int j=1;j<=i;++j){i64 t=row[j-1]+r0;r0=row[j];row[j]=t;}
        row[0]=r0;
    }
    return row[0];
}

// ─── Dirichlet series (multiplicative functions) ──────────────────────────────
// Dirichlet convolution of arithmetic functions f,g: (f*g)(n)=sum_{d|n}f(d)g(n/d)
template<int N>
[[nodiscard]] constexpr std::array<i64,N> dirichlet_conv(
    const std::array<i64,N>& f, const std::array<i64,N>& g) noexcept
{
    std::array<i64,N> h{};
    for(int n=1;n<N;++n)
        for(int d=1;d*d<=n||d<=n;++d) if(n%d==0)
            h[n]+=f[d]*g[n/d];
    return h;
}

// Möbius function mu(n)
[[nodiscard]] constexpr std::array<i64,256> compute_mobius() noexcept {
    std::array<i64,256> mu{};
    std::array<int,256> min_fac{};
    mu[1]=1;
    for(int i=2;i<256;++i) {
        if(!min_fac[i]) { // prime
            for(int j=i;j<256;j+=i) if(!min_fac[j]) min_fac[j]=i;
        }
        int p=min_fac[i], prev=i/p;
        if(prev%p==0) mu[i]=0;
        else mu[i]=-mu[prev];
    }
    return mu;
}
inline constexpr auto k_mobius = compute_mobius();

// Euler totient phi(n)
[[nodiscard]] constexpr i64 euler_totient(i64 n) noexcept {
    i64 result=n;
    for(i64 p=2;p*p<=n;++p)
        if(n%p==0){while(n%p==0)n/=p;result-=result/p;}
    if(n>1) result-=result/n;
    return result;
}

// Sum of divisors sigma_k(n)
[[nodiscard]] constexpr i64 sigma_k(i64 n, int k) noexcept {
    i64 s=0;
    for(i64 d=1;d*d<=n;++d) if(n%d==0) {
        i64 pk=1; for(int j=0;j<k;++j) pk*=d;
        s+=pk;
        if(d!=n/d){i64 qk=1;for(int j=0;j<k;++j)qk*=n/d;s+=qk;}
    }
    return s;
}

} // namespace qs::math
