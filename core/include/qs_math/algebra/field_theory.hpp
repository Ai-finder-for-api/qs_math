#pragma once
// qs_math/algebra/field_theory.hpp — Fields, algebraic closures, valuations, p-adic fields
#include "../arithmetic/scalar_types.hpp"
#include "ring_theory.hpp"
#include <array>
#include <optional>
#include <cmath>

namespace qs::math {

// ─── Field axiom verification (for finite structures) ─────────────────────────
template<i64 P>
[[nodiscard]] constexpr bool verify_field_axioms_Zp() noexcept {
    if(!quotient_is_field(P)) return false;
    // Closure, associativity — structural; check inverses exist for all nonzero
    for(i64 a=1;a<P;++a) {
        auto inv=Zn<P>{a}.inverse();
        if(!inv) return false;
        if((Zn<P>{a}*(*inv)).v!=1) return false;
    }
    return true;
}

// ─── Finite field GF(p) — prime field ────────────────────────────────────────
template<i64 P>
using GFp = Zn<P>;

// Primitive root modulo prime p (generator of multiplicative group)
[[nodiscard]] constexpr i64 primitive_root(i64 p) noexcept {
    if(p==2) return 1;
    // Factor p-1 and check order
    auto fac=factorize(p-1);
    for(i64 g=2;g<p;++g) {
        bool ok=true;
        for(int i=0;i<fac.count&&ok;++i) {
            i64 exp=(p-1)/fac.primes[i];
            // g^exp mod p != 1
            i64 r=1,base=g%p,e=exp;
            while(e>0){if(e&1)r=r*base%p;base=base*base%p;e>>=1;}
            if(r==1) ok=false;
        }
        if(ok) return g;
    }
    return -1; // not found (shouldn't happen for prime p)
}

// Discrete logarithm: find x such that g^x = h (mod p) via baby-step giant-step
[[nodiscard]] constexpr std::optional<i64> discrete_log(i64 g, i64 h, i64 p) noexcept {
    i64 m=(i64)std::ceil(std::sqrt((f64)(p-1)))+1;
    // Baby steps: store g^j for j=0..m-1
    std::array<std::pair<i64,i64>,256> table{};
    int table_sz=0;
    i64 gj=1;
    for(i64 j=0;j<m&&j<256;++j) {
        table[table_sz++]={gj,j};
        gj=gj*g%p;
    }
    // Sort by first element (insertion sort for small table)
    for(int i=1;i<table_sz;++i)
        for(int k=i;k>0&&table[k].first<table[k-1].first;--k)
            std::swap(table[k],table[k-1]);
    // Giant steps: compute g^{-m} mod p
    i64 gm_inv=1,base=gj; // gj = g^m now; compute g^{-m} = (g^m)^{p-2} mod p
    i64 e=p-2;
    while(e>0){if(e&1)gm_inv=gm_inv*base%p;base=base*base%p;e>>=1;}
    i64 cur=h;
    for(i64 i=0;i<m;++i) {
        // Binary search for cur in table
        int lo=0,hi=table_sz-1;
        while(lo<=hi){
            int mid=(lo+hi)/2;
            if(table[mid].first==cur) return i*m+table[mid].second;
            if(table[mid].first<cur) lo=mid+1; else hi=mid-1;
        }
        cur=cur*gm_inv%p;
    }
    return std::nullopt;
}

// ─── Quadratic residues and Legendre symbol ───────────────────────────────────
[[nodiscard]] constexpr i32 legendre_symbol(i64 a, i64 p) noexcept {
    // (a/p) = a^{(p-1)/2} mod p
    if(a%p==0) return 0;
    i64 r=1,base=((a%p)+p)%p,e=(p-1)/2;
    while(e>0){if(e&1)r=r*base%p;base=base*base%p;e>>=1;}
    return (r==1)?1:-1;
}

[[nodiscard]] constexpr bool is_quadratic_residue(i64 a, i64 p) noexcept {
    return legendre_symbol(a,p)==1;
}

// Tonelli-Shanks: square root modulo prime p
[[nodiscard]] constexpr std::optional<i64> sqrt_mod_p(i64 a, i64 p) noexcept {
    a=((a%p)+p)%p;
    if(a==0) return 0;
    if(legendre_symbol(a,p)!=1) return std::nullopt;
    if(p%4==3) {
        // Simple case: r = a^{(p+1)/4} mod p
        i64 r=1,base=a,e=(p+1)/4;
        while(e>0){if(e&1)r=r*base%p;base=base*base%p;e>>=1;}
        return r;
    }
    // General Tonelli-Shanks
    i64 Q=p-1,S=0;
    while(Q%2==0){Q/=2;++S;}
    i64 z=2;
    while(legendre_symbol(z,p)!=-1) ++z;
    i64 M=S, c=1,R=1,t=1;
    {i64 e=Q;i64 b=z;while(e>0){if(e&1)c=c*b%p;b=b*b%p;e>>=1;}}
    {i64 e=(Q+1)/2;i64 b=a;while(e>0){if(e&1)R=R*b%p;b=b*b%p;e>>=1;}}
    {i64 e=Q;i64 b=a;while(e>0){if(e&1)t=t*b%p;b=b*b%p;e>>=1;}}
    for(int iter=0;iter<64;++iter) {
        if(t==1) return R;
        i64 i=1,tmp=t*t%p;
        while(tmp!=1){tmp=tmp*tmp%p;++i;}
        i64 b=c; for(int j=0;j<M-i-1;++j) b=b*b%p;
        M=i; c=b*b%p; R=R*b%p; t=t*c%p;
    }
    return std::nullopt;
}

// ─── p-adic valuation v_p(n) ─────────────────────────────────────────────────
[[nodiscard]] constexpr i64 p_adic_valuation(i64 n, i64 p) noexcept {
    if(n==0) return INT64_MAX; // infinity
    n=n<0?-n:n;
    i64 v=0;
    while(n%p==0){++v;n/=p;}
    return v;
}

// p-adic norm: |n|_p = p^{-v_p(n)}
[[nodiscard]] constexpr f64 p_adic_norm(i64 n, i64 p) noexcept {
    if(n==0) return 0;
    i64 v=p_adic_valuation(n,p);
    return std::pow(f64(p),-f64(v));
}

// p-adic distance
[[nodiscard]] constexpr f64 p_adic_distance(i64 a, i64 b, i64 p) noexcept {
    return p_adic_norm(a-b,p);
}

// ─── Hensel's lemma lifting (root lifting from Z/p to Z/p^k) ─────────────────
// Given f(r)≡0 (mod p) and f'(r)≢0 (mod p), lift to mod p^k
template<typename Poly, typename DPoly>
[[nodiscard]] constexpr i64 hensel_lift(Poly&& f, DPoly&& df,
                                          i64 r0, i64 p, int k) noexcept {
    i64 r=r0, pk=p;
    for(int i=1;i<k;++i) {
        pk*=p;
        i64 fr=f(r), dfr=df(r);
        // r_{i+1} = r_i - f(r_i)/f'(r_i) mod p^{i+1}
        i64 inv=1,base=dfr%pk,e=pk/p-1; // Fermat on Z/p part
        // Find inverse of dfr mod p^{i+1} via extended Euclidean
        i64 a=((dfr%pk)+pk)%pk,b=pk,x=1,x1=0;
        while(b){i64 q=a/b;i64 t=b;b=a-q*b;a=t;t=x1;x1=x-q*x1;x=t;}
        inv=((x%pk)+pk)%pk;
        r=(r-fr*inv%pk+pk)%pk;
    }
    return r;
}

// ─── Field extensions as vector spaces ────────────────────────────────────────
// [Q(sqrt(d)):Q] = 2 for non-square d
// Elements: a + b*sqrt(d)
template<i64 D>
struct QuadraticField {
    i64 a{0},b{0}; // a + b*sqrt(D)

    [[nodiscard]] constexpr QuadraticField operator+(QuadraticField o) const noexcept {return{a+o.a,b+o.b};}
    [[nodiscard]] constexpr QuadraticField operator-(QuadraticField o) const noexcept {return{a-o.a,b-o.b};}
    [[nodiscard]] constexpr QuadraticField operator*(QuadraticField o) const noexcept {return{a*o.a+D*b*o.b,a*o.b+b*o.a};}
    [[nodiscard]] constexpr QuadraticField conj() const noexcept {return{a,-b};}
    [[nodiscard]] constexpr i64 norm() const noexcept {return a*a-D*b*b;} // N(α)=αᾱ
    [[nodiscard]] constexpr i64 trace() const noexcept {return 2*a;} // Tr(α)=α+ᾱ
    [[nodiscard]] constexpr bool is_unit() const noexcept {
        i64 n=norm(); return n==1||n==-1;
    }
};

// ─── Archimedean and non-Archimedean absolute values ─────────────────────────
// Ostrowski's theorem: every non-trivial absolute value on Q is | |_p or | |_inf
[[nodiscard]] constexpr f64 archimedean_abs(i64 n) noexcept { return n<0?-f64(n):f64(n); }

// Check ultrametric inequality: |a+b|_p <= max(|a|_p, |b|_p)
[[nodiscard]] constexpr bool verify_ultrametric(i64 a, i64 b, i64 p) noexcept {
    f64 pa=p_adic_norm(a,p), pb=p_adic_norm(b,p), pab=p_adic_norm(a+b,p);
    return pab<=std::max(pa,pb)+1e-12;
}

} // namespace qs::math
