#pragma once
// qs_math/algebra/number_theory.hpp — Number theory: primes, modular arithmetic,
//   Chinese Remainder Theorem, Euler totient, Mobius, Diophantine, GCD extensions,
//   discrete logarithm, quadratic residues, Miller-Rabin primality (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <optional>

namespace qs::math::algebra {

// ─── GCD / LCM ───────────────────────────────────────────────────────────
[[nodiscard]] constexpr u64 gcd(u64 a, u64 b) noexcept { return b?gcd(b,a%b):a; }
[[nodiscard]] constexpr u64 lcm(u64 a, u64 b) noexcept { return a/gcd(a,b)*b; }

// Extended Euclidean: returns gcd(a,b), sets x,y s.t. ax+by=gcd
[[nodiscard]] inline i64 ext_gcd(i64 a, i64 b, i64& x, i64& y) noexcept {
    if(!b){x=1;y=0;return a;}
    i64 x1,y1; i64 g=ext_gcd(b,a%b,x1,y1);
    x=y1; y=x1-(a/b)*y1; return g;
}

// Modular inverse (a must be coprime to m)
[[nodiscard]] inline std::optional<i64> mod_inverse(i64 a, i64 m) noexcept {
    i64 x,y; i64 g=ext_gcd(a,m,x,y);
    if(g!=1) return std::nullopt;
    return ((x%m)+m)%m;
}

// ─── Modular arithmetic ───────────────────────────────────────────────────
[[nodiscard]] inline u64 mod_add(u64 a, u64 b, u64 m) noexcept { return (a+b)%m; }
[[nodiscard]] inline u64 mod_sub(u64 a, u64 b, u64 m) noexcept { return (a-b%m+m)%m; }
[[nodiscard]] inline u64 mod_mul(u64 a, u64 b, u64 m) noexcept {
    return (__uint128_t)a*b%m;
}

[[nodiscard]] inline u64 mod_pow(u64 base, u64 exp, u64 mod) noexcept {
    u64 result=1; base%=mod;
    while(exp>0){if(exp&1) result=mod_mul(result,base,mod);base=mod_mul(base,base,mod);exp>>=1;}
    return result;
}

// ─── Chinese Remainder Theorem ────────────────────────────────────────────
// Solve x ≡ r_i (mod m_i) for pairwise coprime m_i
[[nodiscard]] inline std::optional<u64> chinese_remainder(
    const std::vector<u64>& remainders,
    const std::vector<u64>& moduli) noexcept
{
    u64 M=1; for(auto m:moduli) M=lcm(M,m);
    u64 x=0;
    for(int i=0;i<int(moduli.size());++i){
        u64 Mi=M/moduli[i];
        auto inv=mod_inverse(i64(Mi),i64(moduli[i]));
        if(!inv) return std::nullopt;
        x=(x+mod_mul(mod_mul(remainders[i],u64(*inv),M),Mi,M))%M;
    }
    return x;
}

// ─── Sieve of Eratosthenes ────────────────────────────────────────────────
[[nodiscard]] inline std::vector<bool> prime_sieve(int n) noexcept {
    std::vector<bool> is_prime(n+1,true);
    is_prime[0]=is_prime[1]=false;
    for(int i=2;i*i<=n;++i)
        if(is_prime[i])
            for(int j=i*i;j<=n;j+=i) is_prime[j]=false;
    return is_prime;
}

[[nodiscard]] inline std::vector<int> primes_up_to(int n) noexcept {
    auto sieve=prime_sieve(n);
    std::vector<int> ps;
    for(int i=2;i<=n;++i) if(sieve[i]) ps.push_back(i);
    return ps;
}

// Segmented sieve for large ranges
[[nodiscard]] inline std::vector<u64> segmented_sieve(u64 lo, u64 hi) noexcept {
    int sqrt_hi=int(std::sqrt(f64(hi)));
    auto small=primes_up_to(sqrt_hi+1);
    std::vector<bool> is_prime(hi-lo+1,true);
    if(lo==1) is_prime[0]=false;
    for(int p:small){
        u64 start=std::max(u64(p*p),(lo+p-1)/p*p);
        for(u64 j=start;j<=hi;j+=p) is_prime[j-lo]=false;
    }
    std::vector<u64> ps;
    for(u64 i=lo;i<=hi;++i) if(is_prime[i-lo]&&i>=2) ps.push_back(i);
    return ps;
}

// ─── Miller-Rabin primality test ──────────────────────────────────────────
[[nodiscard]] inline bool miller_rabin_test(u64 n, u64 a) noexcept {
    if(n%a==0) return n==a;
    u64 d=n-1; int r=0;
    while(d%2==0){d/=2;++r;}
    u64 x=mod_pow(a,d,n);
    if(x==1||x==n-1) return true;
    for(int i=0;i<r-1;++i){x=mod_mul(x,x,n);if(x==n-1) return true;}
    return false;
}

[[nodiscard]] inline bool is_prime(u64 n) noexcept {
    if(n<2) return false;
    if(n==2||n==3||n==5||n==7) return true;
    if(n%2==0||n%3==0) return false;
    // Deterministic for n < 3,317,044,064,679,887,385,961,981
    for(u64 a:{2,3,5,7,11,13,17,19,23,29,31,37})
        if(n==a) return true;
        else if(!miller_rabin_test(n,a)) return false;
    return true;
}

// ─── Pollard's rho factorization ──────────────────────────────────────────
[[nodiscard]] inline u64 pollard_rho(u64 n) noexcept {
    if(n%2==0) return 2;
    u64 x=2,y=2,c=1,d=1;
    while(d==1){
        x=mod_mul(x,x,n)+c;
        y=mod_mul(y,y,n)+c;
        y=mod_mul(y,y,n)+c;
        d=gcd(x>y?x-y:y-x,n);
    }
    return d==n?pollard_rho(n):d; // retry
}

[[nodiscard]] inline std::vector<u64> factorize(u64 n) noexcept {
    if(n<=1) return {};
    if(is_prime(n)) return {n};
    u64 d=pollard_rho(n);
    auto f1=factorize(d), f2=factorize(n/d);
    f1.insert(f1.end(),f2.begin(),f2.end());
    std::sort(f1.begin(),f1.end());
    return f1;
}

// ─── Euler's totient function ─────────────────────────────────────────────
[[nodiscard]] inline u64 euler_totient(u64 n) noexcept {
    u64 result=n;
    auto factors=factorize(n);
    u64 last=0;
    for(u64 p:factors) if(p!=last){result-=result/p;last=p;}
    return result;
}

// Totient sieve up to n
[[nodiscard]] inline std::vector<u64> totient_sieve(int n) noexcept {
    std::vector<u64> phi(n+1);
    std::iota(phi.begin(),phi.end(),0);
    for(int i=2;i<=n;++i)
        if(phi[i]==(u64)i) // i is prime
            for(int j=i;j<=n;j+=i) phi[j]-=phi[j]/i;
    return phi;
}

// ─── Mobius function ─────────────────────────────────────────────────────
[[nodiscard]] inline int mobius(u64 n) noexcept {
    if(n==1) return 1;
    auto f=factorize(n);
    // Check for square factor
    for(int i=1;i<int(f.size());++i) if(f[i]==f[i-1]) return 0;
    return (f.size()%2==0)?1:-1;
}

// Mobius sieve
[[nodiscard]] inline std::vector<int> mobius_sieve(int n) noexcept {
    std::vector<int> mu(n+1,1);
    std::vector<bool> is_composite(n+1,false);
    std::vector<int> primes;
    for(int i=2;i<=n;++i){
        if(!is_composite[i]){primes.push_back(i);mu[i]=-1;}
        for(int j=0;j<int(primes.size())&&i*primes[j]<=n;++j){
            is_composite[i*primes[j]]=true;
            if(i%primes[j]==0){mu[i*primes[j]]=0;break;}
            mu[i*primes[j]]=-mu[i];
        }
    }
    return mu;
}

// ─── Discrete logarithm (Baby-step Giant-step) ────────────────────────────
// Solve g^x ≡ h (mod p), returns x or -1
[[nodiscard]] inline i64 discrete_log(u64 g, u64 h, u64 p) noexcept {
    u64 m=u64(std::ceil(std::sqrt(f64(p))));
    // Baby steps: build table gj -> j
    std::vector<std::pair<u64,u64>> baby;
    u64 gj=1;
    for(u64 j=0;j<m;++j){
        baby.push_back({gj,j});
        gj=mod_mul(gj,g,p);
    }
    std::sort(baby.begin(),baby.end());
    // Giant step factor: g^{-m} mod p
    u64 gm=mod_pow(g,m*(p-2)%(p-1),p); // g^(-m) = g^(m*(p-2)) mod p (Fermat)
    u64 cur=h;
    for(u64 i=0;i<m;++i){
        auto it=std::lower_bound(baby.begin(),baby.end(),std::make_pair(cur,u64(0)));
        if(it!=baby.end()&&it->first==cur) return i64(i*m+it->second);
        cur=mod_mul(cur,gm,p);
    }
    return -1;
}

// ─── Quadratic residues ───────────────────────────────────────────────────
// Legendre symbol (a/p) where p is odd prime
[[nodiscard]] inline int legendre(i64 a, i64 p) noexcept {
    i64 r=i64(mod_pow(u64((a%p+p)%p),u64(p-1)/2,u64(p)));
    return r==p-1?-1:int(r);
}

// Jacobi symbol (a/n) — generalization of Legendre
[[nodiscard]] inline int jacobi(i64 a, i64 n) noexcept {
    if(n<=0||n%2==0) return 0;
    a=((a%n)+n)%n;
    int result=1;
    while(a){
        while(a%2==0){a/=2;if(n%8==3||n%8==5) result=-result;}
        std::swap(a,n);
        if(a%4==3&&n%4==3) result=-result;
        a%=n;
    }
    return n==1?result:0;
}

// Tonelli-Shanks square root mod prime
[[nodiscard]] inline std::optional<u64> sqrt_mod(u64 n, u64 p) noexcept {
    if(p==2) return n%2;
    if(legendre(i64(n),i64(p))!=1) return std::nullopt;
    u64 Q=p-1, S=0;
    while(Q%2==0){Q/=2;++S;}
    u64 z=2;
    while(legendre(i64(z),i64(p))!=-1) ++z;
    u64 M=S, c=mod_pow(z,Q,p), t=mod_pow(n,Q,p), R=mod_pow(n,(Q+1)/2,p);
    while(true){
        if(t==0) return 0; if(t==1) return R;
        u64 i=1, tmp=mod_mul(t,t,p);
        while(tmp!=1){tmp=mod_mul(tmp,tmp,p);++i;}
        u64 b=mod_pow(c,mod_pow(2,M-i-1,p-1),p);
        M=i; c=mod_mul(b,b,p); t=mod_mul(t,c,p); R=mod_mul(R,b,p);
    }
}

// ─── Number-theoretic transform (NTT) ────────────────────────────────────
// NTT mod a prime p with primitive root g
inline void ntt(std::vector<u64>& a, bool inv, u64 p=998244353ULL, u64 g=3ULL) noexcept {
    int n=int(a.size());
    for(int i=1,j=0;i<n;++i){
        int bit=n>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit; if(i<j) std::swap(a[i],a[j]);
    }
    for(int len=2;len<=n;len<<=1){
        u64 w=inv?mod_pow(g,(p-1)-(p-1)/len,p):mod_pow(g,(p-1)/len,p);
        for(int i=0;i<n;i+=len){
            u64 wn=1;
            for(int j=0;j<len/2;++j){
                u64 u=a[i+j],v=mod_mul(a[i+j+len/2],wn,p);
                a[i+j]=(u+v)%p; a[i+j+len/2]=(u-v+p)%p;
                wn=mod_mul(wn,w,p);
            }
        }
    }
    if(inv){
        u64 inv_n=mod_pow(n,p-2,p);
        for(auto& x:a) x=mod_mul(x,inv_n,p);
    }
}

// Polynomial multiplication via NTT
[[nodiscard]] inline std::vector<u64> ntt_multiply(
    std::vector<u64> a, std::vector<u64> b, u64 p=998244353ULL) noexcept
{
    int result_len=int(a.size()+b.size())-1;
    int n=1; while(n<result_len) n<<=1;
    a.resize(n); b.resize(n);
    ntt(a,false,p); ntt(b,false,p);
    for(int i=0;i<n;++i) a[i]=mod_mul(a[i],b[i],p);
    ntt(a,true,p);
    a.resize(result_len);
    return a;
}

// ─── Continued fractions ─────────────────────────────────────────────────
// Rational approximation via continued fraction expansion
[[nodiscard]] inline std::vector<i64> continued_fraction(f64 x, int max_terms=20) noexcept {
    std::vector<i64> cf;
    for(int i=0;i<max_terms;++i){
        cf.push_back(i64(x));
        f64 frac=x-cf.back();
        if(std::abs(frac)<1e-10) break;
        x=1.0/frac;
    }
    return cf;
}

struct Rational { i64 num, den; };
[[nodiscard]] inline Rational cf_convergent(std::span<const i64> cf, int k) noexcept {
    if(k<0) return {0,1};
    i64 h_prev=1,h=cf[0],k_prev=0,kk=1;
    for(int i=1;i<=k&&i<int(cf.size());++i){
        i64 h_new=cf[i]*h+h_prev, k_new=cf[i]*kk+k_prev;
        h_prev=h; h=h_new; k_prev=kk; kk=k_new;
    }
    return {h,kk};
}

} // namespace qs::math::algebra
