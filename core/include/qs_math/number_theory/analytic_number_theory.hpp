#pragma once
// qs_math/number_theory/analytic_number_theory.hpp — PNT, L-functions, sieve, circle method
#include "../arithmetic/scalar_types.hpp"
#include "../analysis/complex_analysis.hpp"
#include <array>
#include <cmath>
#include <numbers>

namespace qs::math {

// ─── Chebyshev functions ──────────────────────────────────────────────────────
// θ(x) = Σ log(p), p≤x prime
[[nodiscard]] constexpr f64 chebyshev_theta(f64 x) noexcept {
    f64 s=0;
    for(i64 n=2;f64(n)<=x;++n) {
        bool prime=true;
        for(i64 d=2;d*d<=n&&prime;++d) if(n%d==0) prime=false;
        if(prime) s+=std::log(f64(n));
    }
    return s;
}

// ψ(x) = Σ Λ(n), n≤x (von Mangoldt sum)
[[nodiscard]] constexpr f64 chebyshev_psi(f64 x) noexcept {
    f64 s=0;
    for(i64 n=2;f64(n)<=x;++n) {
        i64 m=n;
        for(i64 p=2;p*p<=m;++p) {
            if(m%p==0) {
                while(m%p==0) m/=p;
                s+=std::log(f64(p));
                if(m==1) break;
            }
        }
        if(m>1) s+=std::log(f64(m)); // n is prime
    }
    return s;
}

// ─── Prime counting function π(x) ────────────────────────────────────────────
[[nodiscard]] constexpr i64 prime_pi(i64 x) noexcept {
    if(x<2) return 0;
    i64 count=0;
    for(i64 n=2;n<=x;++n) {
        bool prime=true;
        for(i64 d=2;d*d<=n&&prime;++d) if(n%d==0) prime=false;
        if(prime) ++count;
    }
    return count;
}

// Prime number theorem approximation: π(x) ≈ x/log(x)
[[nodiscard]] constexpr f64 pnt_approx(f64 x) noexcept { return x/std::log(x); }

// Logarithmic integral Li(x) = PV ∫_0^x dt/log(t)
[[nodiscard]] constexpr f64 li(f64 x, int terms=100) noexcept {
    if(x<=1.0) return 0;
    f64 s=0,ln_x=std::log(x),factorial=1;
    f64 power=1;
    for(int k=1;k<=terms;++k) {
        power*=ln_x; factorial*=k;
        s+=power/factorial;
    }
    return 0.5772156649015328+std::log(ln_x)+s; // Ei(log x)
}

// ─── Sieve of Eratosthenes ────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<bool,N+1> sieve_eratosthenes() noexcept {
    std::array<bool,N+1> is_prime{};
    is_prime.fill(true);
    is_prime[0]=is_prime[1]=false;
    for(int p=2;p*p<=N;++p) if(is_prime[p])
        for(int m=p*p;m<=N;m+=p) is_prime[m]=false;
    return is_prime;
}

// Segmented sieve for large ranges [lo, hi)
template<int SEGMENT=32768>
[[nodiscard]] constexpr i64 count_primes_range(i64 lo, i64 hi) noexcept {
    if(hi<=2) return 0;
    if(lo<2) lo=2;
    // Use small primes up to sqrt(hi)
    constexpr int SQMAX=200;
    auto small=sieve_eratosthenes<SQMAX>();
    std::array<bool,SEGMENT> seg{};
    i64 count=0;
    for(i64 base=lo;base<hi;base+=SEGMENT) {
        i64 top=std::min(base+SEGMENT,hi);
        seg.fill(true);
        for(int p=2;p<SQMAX;++p) if(small[p]) {
            i64 start=((base+p-1)/p)*p;
            if(start==p) start+=p;
            for(i64 j=start;j<top;j+=p) seg[j-base]=false;
        }
        for(i64 j=base;j<top;++j) if(seg[j-base]) ++count;
    }
    return count;
}

// ─── Dirichlet L-functions ────────────────────────────────────────────────────
// Dirichlet character χ mod q: complete multiplicative, periodic
// For principal character χ_0: χ_0(n)=1 if gcd(n,q)=1, else 0
[[nodiscard]] constexpr i64 principal_character(i64 n, i64 q) noexcept {
    i64 a=n%q,b=q; while(b){i64 t=b;b=a%b;a=t;} return a==1?1:0;
}

// L(s,χ_0) partial sum (real s>1)
[[nodiscard]] constexpr f64 dirichlet_l_principal(f64 s, i64 q, int terms=1000) noexcept {
    f64 sum=0;
    for(i64 n=1;n<=terms;++n)
        if(principal_character(n,q)) sum+=1.0/std::pow(f64(n),s);
    return sum;
}

// Riemann hypothesis: all nontrivial zeros have Re(s)=1/2
// Critical strip: 0 < Re(s) < 1
[[nodiscard]] constexpr bool in_critical_strip(Cx s) noexcept {
    return s.re>0 && s.re<1;
}
[[nodiscard]] constexpr bool on_critical_line(Cx s, f64 tol=1e-9) noexcept {
    return std::abs(s.re-0.5)<tol;
}

// ─── Hardy-Ramanujan formula for partitions ────────────────────────────────────
// p(n) ≈ (1/(4n√3)) * exp(π√(2n/3))
[[nodiscard]] constexpr f64 partition_approx(i64 n) noexcept {
    if(n<=0) return 1.0;
    f64 t=std::numbers::pi_v<f64>*std::sqrt(2.0*n/3.0);
    return std::exp(t)/(4.0*n*std::numbers::sqrt3_v<f64>);
}

// ─── Dirichlet's theorem on primes in arithmetic progressions ─────────────────
// Counts primes p≤x with p≡a (mod q), gcd(a,q)=1
[[nodiscard]] constexpr i64 primes_in_progression(i64 x, i64 a, i64 q) noexcept {
    i64 g=a,b=q; while(b){i64 t=b;b=g%b;g=t;}
    if(g!=1) return 0; // gcd(a,q)!=1 means only finitely many
    i64 count=0;
    for(i64 p=2;p<=x;++p) {
        if(p%q!=a%q) continue;
        bool prime=true;
        for(i64 d=2;d*d<=p&&prime;++d) if(p%d==0) prime=false;
        if(prime) ++count;
    }
    return count;
}

// By PNT for APs: π(x;q,a) ≈ Li(x)/φ(q)
[[nodiscard]] constexpr f64 primes_in_progression_approx(f64 x, i64 q) noexcept {
    return li(x)/f64(euler_totient(q));
}

// ─── Goldbach-like: express n as sum of k primes (verification) ──────────────
[[nodiscard]] constexpr bool is_prime_fast(i64 n) noexcept {
    if(n<2) return false;
    if(n==2||n==3) return true;
    if(n%2==0||n%3==0) return false;
    for(i64 p=5;p*p<=n;p+=6) if(n%p==0||n%(p+2)==0) return false;
    return true;
}

[[nodiscard]] constexpr bool goldbach_verify(i64 n) noexcept {
    // Every even n>2 is sum of 2 primes
    if(n%2!=0||n<=2) return false;
    for(i64 p=2;p<=n/2;++p)
        if(is_prime_fast(p)&&is_prime_fast(n-p)) return true;
    return false;
}

// ─── Möbius function and its applications ────────────────────────────────────
// Already have k_mobius table; define here for runtime values
[[nodiscard]] constexpr i64 mobius(i64 n) noexcept {
    if(n<=0) return 0;
    if(n<256) return k_mobius[n];
    // Compute directly
    i64 m=n;
    int primes=0;
    for(i64 p=2;p*p<=m;++p) {
        if(m%p==0){++primes;m/=p;if(m%p==0) return 0;}
    }
    if(m>1) ++primes;
    return (primes%2==0)?1:-1;
}

// Mertens function M(n) = Σ_{k=1}^{n} μ(k)
[[nodiscard]] constexpr i64 mertens(i64 n) noexcept {
    i64 s=0; for(i64 k=1;k<=n;++k) s+=mobius(k); return s;
}

} // namespace qs::math
