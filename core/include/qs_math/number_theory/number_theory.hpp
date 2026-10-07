#pragma once
// number_theory.hpp — Number theory: primes, factorization, Euler totient,
//                     Chinese Remainder Theorem, quadratic residues,
//                     discrete log, primality tests, Dirichlet series helpers
#include "scalar_types.hpp"
#include "integer_math.hpp"
#include <array>

namespace qs::math::number {

// ---------------------------------------------------------------------------
// Primality
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr bool is_prime(u64 n) noexcept {
    if(n<2) return false; if(n<4) return true;
    if(n%2==0||n%3==0) return false;
    for(u64 i=5;i*i<=n;i+=6) if(n%i==0||n%(i+2)==0) return false;
    return true;
}

// Miller-Rabin (deterministic for n < 3,317,044,064,679,887,385,961,981)
[[nodiscard]] inline bool miller_rabin(u64 n) noexcept {
    if(n<2) return false; if(n<4) return true;
    if(n%2==0) return false;
    // Write n-1 = 2^r * d
    u64 d=n-1; i32 r=0;
    while(d%2==0){ d>>=1; ++r; }
    // Witnesses covering all n < 3.3*10^24
    static constexpr u64 witnesses[]={2,3,5,7,11,13,17,19,23,29,31,37};
    for(u64 a:witnesses){
        if(a>=n) continue;
        u64 x=mod_pow(a,d,n);
        if(x==1||x==n-1) continue;
        bool composite=true;
        for(i32 i=0;i<r-1;++i){
            x=mod_mul(x,x,n);
            if(x==n-1){composite=false;break;}
        }
        if(composite) return false;
    }
    return true;
}

// Sieve of Eratosthenes
template<i32 LIMIT>
struct PrimeSieve {
    bool is_prime_[LIMIT+1]{};
    i32  primes[LIMIT/5+100];
    i32  count{0};

    constexpr PrimeSieve() noexcept {
        for(i32 i=2;i<=LIMIT;++i) is_prime_[i]=true;
        for(i32 i=2;i*i<=LIMIT;++i)
            if(is_prime_[i]) for(i32 j=i*i;j<=LIMIT;j+=i) is_prime_[j]=false;
        for(i32 i=2;i<=LIMIT;++i) if(is_prime_[i]) primes[count++]=i;
    }
    [[nodiscard]] constexpr bool test(i32 n) const noexcept {
        return n>=2&&n<=LIMIT&&is_prime_[n];
    }
    [[nodiscard]] constexpr i32 nth(i32 n) const noexcept {
        return n<count?primes[n]:-1;
    }
};

inline constexpr PrimeSieve<10000> SIEVE_10K;

// Prime factorization
inline i32 factorize(u64 n, u64* factors) noexcept {
    i32 count=0;
    for(u64 d=2;d*d<=n;++d) while(n%d==0){ factors[count++]=d; n/=d; }
    if(n>1) factors[count++]=n;
    return count;
}

// ---------------------------------------------------------------------------
// Euler's totient function φ(n)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 euler_totient(u64 n) noexcept {
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

// Möbius function μ(n)
[[nodiscard]] constexpr i32 mobius(u64 n) noexcept {
    if(n==1) return 1;
    u64 factors[64]; i32 nf=factorize(n,factors);
    // Check for square factor
    for(i32 i=0;i<nf-1;++i) if(factors[i]==factors[i+1]) return 0;
    return nf%2==0?1:-1;
}

// Jordan's totient function J_k(n)
[[nodiscard]] constexpr u64 jordan_totient(u64 n, i32 k) noexcept {
    u64 result=ipow(n,k);
    u64 tmp=n;
    for(u64 p=2;p*p<=tmp;++p){
        if(tmp%p==0){
            while(tmp%p==0) tmp/=p;
            result-=result/ipow(p,k);
        }
    }
    if(tmp>1) result-=result/ipow(tmp,k);
    return result;
}

// ---------------------------------------------------------------------------
// Chinese Remainder Theorem
// ---------------------------------------------------------------------------
// Solve x ≡ r[i] (mod m[i]) for all i (m[i] must be pairwise coprime)
[[nodiscard]] inline i64 crt(const i64* r, const i64* m, i32 n) noexcept {
    i64 M=1; for(i32 i=0;i<n;++i) M*=m[i];
    i64 x=0;
    for(i32 i=0;i<n;++i){
        i64 Mi=M/m[i];
        i64 yi=0,dummy=0;
        gcd_extended(Mi,m[i],yi,dummy);
        x=(x+r[i]*Mi*yi)%M;
    }
    return (x%M+M)%M;
}

// ---------------------------------------------------------------------------
// Quadratic residues
// ---------------------------------------------------------------------------
// Legendre symbol (p/q) — assumes p prime
[[nodiscard]] constexpr i32 legendre(i64 a, i64 p) noexcept {
    if(a%p==0) return 0;
    i64 r=mod_pow(u64(a%p+p),u64(p-1)/2,u64(p));
    return r==1?1:-1;
}

// Jacobi symbol generalizes Legendre to odd composite
[[nodiscard]] constexpr i32 jacobi(i64 a, i64 n) noexcept {
    if(n<=0||n%2==0) return 0;
    a%=n; if(a<0) a+=n;
    i32 result=1;
    while(a!=0){
        while(a%2==0){
            a/=2; i64 n8=n%8;
            if(n8==3||n8==5) result=-result;
        }
        i64 tmp=a; a=n; n=tmp;
        if(a%4==3&&n%4==3) result=-result;
        a%=n;
    }
    return n==1?result:0;
}

// Tonelli-Shanks algorithm: find sqrt of n mod p
[[nodiscard]] inline u64 sqrt_mod_p(u64 n, u64 p) noexcept {
    if(legendre(n,p)!=1) return 0;
    if(p%4==3) return mod_pow(n,(p+1)/4,p);
    // Factor p-1 = Q * 2^S
    u64 Q=p-1; i32 S=0;
    while(Q%2==0){Q>>=1;++S;}
    // Find quadratic non-residue
    u64 z=2;
    while(legendre(z,p)!=-1) ++z;
    u64 M=S, c=mod_pow(z,Q,p), t=mod_pow(n,Q,p), R=mod_pow(n,(Q+1)/2,p);
    while(true){
        if(t==0) return 0; if(t==1) return R;
        u64 i=1, tmp=mod_mul(t,t,p);
        while(tmp!=1){tmp=mod_mul(tmp,tmp,p);++i;}
        u64 b=mod_pow(c,u64(1)<<(M-i-1),p);
        M=i; c=mod_mul(b,b,p); t=mod_mul(t,c,p); R=mod_mul(R,b,p);
    }
}

// ---------------------------------------------------------------------------
// Discrete logarithm (Baby-step Giant-step)
// ---------------------------------------------------------------------------
[[nodiscard]] inline i64 discrete_log(u64 g, u64 h, u64 p) noexcept {
    // Find x such that g^x ≡ h (mod p)
    u64 m=u64(sqrt(f64(p)))+1;
    // Baby steps: table[g^j mod p] = j for j=0..m
    struct Entry { u64 val; u64 j; };
    constexpr i32 MAX_M=10000;
    if(m>MAX_M) return -1;
    Entry table[MAX_M];
    u64 gj=1;
    for(u64 j=0;j<m;++j){ table[j]={gj,j}; gj=mod_mul(gj,g,p); }
    // Sort by val
    for(u64 i=1;i<m;++i){ Entry k=table[i]; u64 j=i;
        while(j>0&&table[j-1].val>k.val){table[j]=table[j-1];--j;} table[j]=k; }
    // Giant steps: g^(-m) mod p
    u64 gm=mod_pow(mod_pow(g,m,p),p-2,p);
    u64 gamma=h;
    for(u64 i=0;i<=m;++i){
        // Binary search for gamma in table
        i64 lo=0,hi=i64(m)-1;
        while(lo<=hi){ i64 mid=(lo+hi)/2;
            if(table[mid].val==gamma) return i64(i)*i64(m)+i64(table[mid].j);
            if(table[mid].val<gamma) lo=mid+1; else hi=mid-1; }
        gamma=mod_mul(gamma,gm,p);
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Number-theoretic transform (NTT) — discrete Fourier over Z_p
// ---------------------------------------------------------------------------
inline void ntt(u64* a, i32 n, bool inverse, u64 MOD=998244353ULL, u64 g_root=3) noexcept {
    for(i32 i=1,j=0;i<n;++i){
        i32 bit=n>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit; if(i<j){ u64 t=a[i];a[i]=a[j];a[j]=t; }
    }
    for(i32 len=2;len<=n;len<<=1){
        u64 w=inverse?mod_pow(g_root,(MOD-1)-(MOD-1)/len,MOD):mod_pow(g_root,(MOD-1)/len,MOD);
        for(i32 i=0;i<n;i+=len){
            u64 wn=1;
            for(i32 j=0;j<len/2;++j){
                u64 u=a[i+j], v=mod_mul(a[i+j+len/2],wn,MOD);
                a[i+j]=(u+v)%MOD; a[i+j+len/2]=(u+MOD-v)%MOD;
                wn=mod_mul(wn,w,MOD);
            }
        }
    }
    if(inverse){ u64 inv_n=mod_pow(n,MOD-2,MOD); for(i32 i=0;i<n;++i) a[i]=mod_mul(a[i],inv_n,MOD); }
}

// Polynomial multiplication mod p
inline void poly_mul_ntt(const u64* a, i32 na, const u64* b, i32 nb, u64* c, u64 MOD=998244353ULL) noexcept {
    i32 n=1; while(n<na+nb) n<<=1;
    static u64 fa[1<<20],fb[1<<20];
    for(i32 i=0;i<n;++i) fa[i]=i<na?a[i]%MOD:0;
    for(i32 i=0;i<n;++i) fb[i]=i<nb?b[i]%MOD:0;
    ntt(fa,n,false,MOD); ntt(fb,n,false,MOD);
    for(i32 i=0;i<n;++i) fa[i]=mod_mul(fa[i],fb[i],MOD);
    ntt(fa,n,true,MOD);
    for(i32 i=0;i<na+nb-1;++i) c[i]=fa[i];
}

// ---------------------------------------------------------------------------
// Continued fractions
// ---------------------------------------------------------------------------
// Compute continued fraction expansion of x
inline i32 continued_fraction(f64 x, i64* a, i32 max_terms, f64 eps=1e-10) noexcept {
    i32 n=0;
    while(n<max_terms){
        a[n]=i64(x); x-=a[n]; ++n;
        if(abs(x)<eps) break;
        x=1.0/x;
    }
    return n;
}

// Convergents of a continued fraction
inline void cf_convergents(const i64* a, i32 n, i64* p, i64* q) noexcept {
    p[0]=a[0];q[0]=1;
    if(n==1) return;
    p[1]=a[1]*a[0]+1; q[1]=a[1];
    for(i32 i=2;i<n;++i){ p[i]=a[i]*p[i-1]+p[i-2]; q[i]=a[i]*q[i-1]+q[i-2]; }
}

// Best rational approximation
[[nodiscard]] inline void best_rational(f64 x, i64& num, i64& den, i32 max_denom=1000) noexcept {
    i64 a[64]; i32 n=continued_fraction(x,a,64);
    i64 p[64],q[64]; cf_convergents(a,n,p,q);
    num=1; den=1;
    for(i32 i=0;i<n;++i){ if(q[i]>max_denom) break; num=p[i]; den=q[i]; }
}

} // namespace qs::math::number
