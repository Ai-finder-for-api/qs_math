#pragma once
// qs_math/number_theory/primes.hpp — Primality, sieves, factorisation, prime gaps (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <optional>
#include <array>
#include <cassert>

namespace qs {

// ─── Trial division primality ─────────────────────────────────────────────────

[[nodiscard]] inline bool is_prime_trial(u64 n) {
    if(n<2) return false;
    if(n==2||n==3) return true;
    if(n%2==0||n%3==0) return false;
    for(u64 i=5;i*i<=n;i+=6)
        if(n%i==0||n%(i+2)==0) return false;
    return true;
}

// ─── Miller-Rabin (deterministic for n < 3.3×10^24) ─────────────────────────

[[nodiscard]] inline u64 mulmod64(u64 a, u64 b, u64 m) {
    return u64((__uint128_t(a)*b)%m);
}

[[nodiscard]] inline u64 powmod64(u64 base, u64 exp, u64 mod) {
    u64 r=1; base%=mod;
    while(exp>0){if(exp&1)r=mulmod64(r,base,mod);base=mulmod64(base,base,mod);exp>>=1;}
    return r;
}

[[nodiscard]] inline bool miller_rabin_witness(u64 n, u64 a) {
    if(n%a==0) return n==a;
    u64 d=n-1; u32 r=0;
    while(d%2==0){d>>=1;++r;}
    u64 x=powmod64(a,d,n);
    if(x==1||x==n-1) return true;
    for(u32 i=0;i<r-1;++i){x=mulmod64(x,x,n);if(x==n-1)return true;}
    return false;
}

[[nodiscard]] inline bool is_prime(u64 n) {
    if(n<2) return false;
    for(u64 p:{2,3,5,7,11,13,17,19,23,29,31,37})
        if(n==p) return true;
        else if(n%p==0) return false;
    // Deterministic witnesses for n < 3,317,044,064,679,887,385,961,981
    for(u64 a:{2,3,5,7,11,13,17,19,23,29,31,37,41})
        if(!miller_rabin_witness(n,a)) return false;
    return true;
}

// ─── Sieve of Eratosthenes ────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<u32> sieve_eratosthenes(u32 limit) {
    std::vector<bool> is_p(limit+1,true);
    is_p[0]=is_p[1]=false;
    for(u32 i=2;u64(i)*i<=limit;++i)
        if(is_p[i]) for(u32 j=i*i;j<=limit;j+=i) is_p[j]=false;
    std::vector<u32> primes;
    for(u32 i=2;i<=limit;++i) if(is_p[i]) primes.push_back(i);
    return primes;
}

// Segmented sieve for primes in [lo, hi]
[[nodiscard]] inline std::vector<u64> sieve_segment(u64 lo, u64 hi) {
    u64 root=u64(std::sqrt(f64(hi)))+1;
    auto small_primes=sieve_eratosthenes(u32(root));
    std::vector<bool> is_p(hi-lo+1,true);
    if(lo==0) is_p[0]=false;
    if(lo<=1&&lo+1<=hi) is_p[1-lo]=false;
    for(u32 p:small_primes){
        u64 start=std::max(u64(p)*p,(lo+p-1)/p*p);
        for(u64 j=start;j<=hi;j+=p) is_p[j-lo]=false;
    }
    std::vector<u64> primes;
    for(u64 i=lo;i<=hi;++i) if(is_p[i-lo]) primes.push_back(i);
    return primes;
}

// Linear sieve (Euler's sieve) — O(n)
[[nodiscard]] inline std::vector<u32> sieve_linear(u32 n) {
    std::vector<u32> primes, spf(n+1,0); // spf = smallest prime factor
    for(u32 i=2;i<=n;++i){
        if(!spf[i]){spf[i]=i;primes.push_back(i);}
        for(u32 p:primes){
            if(p>spf[i]||u64(i)*p>n) break;
            spf[i*p]=p;
        }
    }
    return primes;
}

// ─── Factorisation ────────────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<u64> prime_factors(u64 n) {
    std::vector<u64> factors;
    for(u64 p=2;p*p<=n;++p)
        while(n%p==0){factors.push_back(p);n/=p;}
    if(n>1) factors.push_back(n);
    return factors;
}

// Pollard's rho factorisation (for large composites)
[[nodiscard]] inline u64 pollard_rho(u64 n) {
    if(n%2==0) return 2;
    u64 x=2,y=2,d=1,c=1;
    auto f=[&](u64 x){return (mulmod64(x,x,n)+c)%n;};
    while(d==1){x=f(x);y=f(f(y));u64 diff=x>y?x-y:y-x;d=std::gcd(diff,n);}
    return d;
}

[[nodiscard]] inline std::vector<u64> factorise(u64 n) {
    if(n<=1) return {};
    if(is_prime(n)) return {n};
    u64 d=n;
    while(d==n){
        d=pollard_rho(n);
        // Change constant if stuck
        static u64 c=1; ++c;
    }
    auto f1=factorise(d), f2=factorise(n/d);
    for(auto x:f2) f1.push_back(x);
    std::sort(f1.begin(),f1.end());
    return f1;
}

// Distinct prime factors
[[nodiscard]] inline std::vector<u64> distinct_prime_factors(u64 n) {
    auto f=prime_factors(n);
    f.erase(std::unique(f.begin(),f.end()),f.end());
    return f;
}

// ─── Euler's totient function ─────────────────────────────────────────────────

[[nodiscard]] inline u64 euler_totient(u64 n) {
    u64 result=n;
    for(u64 p=2;p*p<=n;++p){
        if(n%p==0){
            while(n%p==0) n/=p;
            result-=result/p;
        }
    }
    if(n>1) result-=result/n;
    return result;
}

// Totient sieve for all values up to n
[[nodiscard]] inline std::vector<u64> totient_sieve(u32 n) {
    std::vector<u64> phi(n+1);
    std::iota(phi.begin(),phi.end(),0ull);
    for(u32 i=2;i<=n;++i)
        if(phi[i]==i) // i is prime
            for(u32 j=i;j<=n;j+=i) phi[j]-=phi[j]/i;
    return phi;
}

// ─── Möbius function ─────────────────────────────────────────────────────────

[[nodiscard]] inline i32 mobius(u64 n) {
    auto f=prime_factors(n);
    // Check squarefree
    for(u32 i=0;i+1<f.size();++i) if(f[i]==f[i+1]) return 0;
    return (f.size()%2==0)?1:-1;
}

// Möbius sieve
[[nodiscard]] inline std::vector<i8> mobius_sieve(u32 n) {
    std::vector<i8> mu(n+1,0);
    std::vector<u32> primes;
    std::vector<bool> is_p(n+1,true);
    mu[1]=1;
    for(u32 i=2;i<=n;++i){
        if(is_p[i]){primes.push_back(i);mu[i]=-1;}
        for(u32 p:primes){
            if(u64(i)*p>n) break;
            is_p[i*p]=false;
            if(i%p==0){mu[i*p]=0;break;}
            mu[i*p]=-mu[i];
        }
    }
    return mu;
}

// ─── Number-theoretic functions ───────────────────────────────────────────────

// Sum of divisors: σ_k(n) = sum d^k for d|n
[[nodiscard]] inline u64 divisor_sum(u64 n, u32 k=1) {
    u64 s=0;
    for(u64 d=1;d*d<=n;++d){
        if(n%d==0){
            u64 dk=1; for(u32 i=0;i<k;++i) dk*=d;
            s+=dk;
            if(d!=n/d){u64 dk2=1;u64 d2=n/d;for(u32 i=0;i<k;++i)dk2*=d2;s+=dk2;}
        }
    }
    return s;
}

// Number of divisors: τ(n) = σ_0(n)
[[nodiscard]] inline u64 num_divisors(u64 n) { return divisor_sum(n,0); }

// All divisors of n
[[nodiscard]] inline std::vector<u64> divisors(u64 n) {
    std::vector<u64> divs;
    for(u64 d=1;d*d<=n;++d){
        if(n%d==0){divs.push_back(d);if(d!=n/d)divs.push_back(n/d);}
    }
    std::sort(divs.begin(),divs.end());
    return divs;
}

// Liouville function: λ(n) = (-1)^Ω(n) where Ω = total prime factor count
[[nodiscard]] inline i32 liouville(u64 n) {
    auto f=prime_factors(n);
    return (f.size()%2==0)?1:-1;
}

// Mertens function: M(n) = sum_{k=1}^{n} μ(k)
[[nodiscard]] inline i64 mertens(u32 n) {
    auto mu=mobius_sieve(n);
    i64 M=0; for(u32 i=1;i<=n;++i) M+=mu[i];
    return M;
}

// ─── Primes of special forms ──────────────────────────────────────────────────

// Mersenne prime: 2^p - 1 where p prime
[[nodiscard]] inline bool is_mersenne_prime_exponent(u32 p) {
    if(!is_prime(p)) return false;
    if(p==2) return true;
    // Lucas-Lehmer test
    u64 M=(u64(1)<<p)-1;  // only works for p<=63
    if(p>=64) return false; // need bignum for larger
    u64 s=4;
    for(u32 i=0;i<p-2;++i) s=(u64((__uint128_t(s)*s-2))%M);
    return s==0;
}

// Twin primes: (p, p+2) both prime
[[nodiscard]] inline std::vector<std::pair<u32,u32>> twin_primes(u32 limit) {
    auto primes=sieve_eratosthenes(limit);
    std::vector<std::pair<u32,u32>> twins;
    for(u32 i=0;i+1<primes.size();++i)
        if(primes[i+1]-primes[i]==2) twins.push_back({primes[i],primes[i+1]});
    return twins;
}

// Goldbach: even n = p + q (Goldbach conjecture, unproven)
[[nodiscard]] inline std::optional<std::pair<u32,u32>> goldbach(u32 n) {
    if(n<=2||n%2!=0) return std::nullopt;
    auto primes=sieve_eratosthenes(n);
    for(u32 p:primes){
        u32 q=n-p;
        if(is_prime_trial(q)) return std::make_pair(p,q);
    }
    return std::nullopt;
}

// ─── Primitive roots ──────────────────────────────────────────────────────────

[[nodiscard]] inline bool is_primitive_root(u64 g, u64 p) {
    if(!is_prime(p)) return false;
    u64 phi=p-1;
    auto factors=distinct_prime_factors(phi);
    for(u64 f:factors)
        if(powmod64(g,phi/f,p)==1) return false;
    return true;
}

[[nodiscard]] inline std::optional<u64> find_primitive_root(u64 p) {
    for(u64 g=2;g<p;++g)
        if(is_primitive_root(g,p)) return g;
    return std::nullopt;
}

// ─── Prime counting function π(x) (approx via Li(x)) ─────────────────────────

[[nodiscard]] inline f64 prime_counting_approx(f64 x) {
    if(x<2.) return 0.;
    // Li(x) via series: Ei(ln x) - asymptotic
    f64 ln_x=std::log(x);
    f64 li=0.;
    f64 term=1.;
    for(u32 k=1;k<50;++k){
        term*=ln_x/f64(k);
        li+=term/f64(k);
    }
    return li+0.5772156649f; // Euler-Mascheroni correction
}

// Prime gaps
[[nodiscard]] inline std::vector<u32> prime_gaps(u32 limit) {
    auto primes=sieve_eratosthenes(limit);
    std::vector<u32> gaps;
    for(u32 i=0;i+1<primes.size();++i) gaps.push_back(primes[i+1]-primes[i]);
    return gaps;
}

} // namespace qs
