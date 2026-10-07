#pragma once
// qs_math/algebra/ring_theory.hpp — Rings, ideals, quotient rings, PIDs, UFDs
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <optional>

namespace qs::math {

// ─── Ring element concept (compile-time duck typing) ──────────────────────────
// A ring R must provide: operator+, operator*, unary operator-, zero(), one()
// We provide concrete rings below

// ─── Z/nZ (integers mod n) ───────────────────────────────────────────────────
template<i64 N>
struct Zn {
    i64 v{0};
    static constexpr i64 mod=N;

    constexpr Zn(i64 x=0) noexcept : v(((x%N)+N)%N) {}
    [[nodiscard]] static constexpr Zn zero() noexcept { return {0}; }
    [[nodiscard]] static constexpr Zn one()  noexcept { return {1}; }
    [[nodiscard]] constexpr Zn operator+(Zn o) const noexcept { return {v+o.v}; }
    [[nodiscard]] constexpr Zn operator-(Zn o) const noexcept { return {v-o.v+N}; }
    [[nodiscard]] constexpr Zn operator*(Zn o) const noexcept { return {v*o.v}; }
    [[nodiscard]] constexpr Zn operator-()     const noexcept { return {N-v}; }
    [[nodiscard]] constexpr bool operator==(Zn o) const noexcept { return v==o.v; }
    [[nodiscard]] constexpr bool is_unit() const noexcept {
        // gcd(v,N)==1
        i64 a=v,b=N; while(b){i64 t=b;b=a%b;a=t;} return a==1;
    }
    [[nodiscard]] constexpr std::optional<Zn> inverse() const noexcept {
        if(!is_unit()) return std::nullopt;
        // Extended Euclidean
        i64 a=v,b=N,x0=1,x1=0;
        while(b){i64 q=a/b;i64 t=b;b=a-q*b;a=t;t=x1;x1=x0-q*x1;x0=t;}
        return Zn{x0};
    }
};

// ─── Ideal membership ─────────────────────────────────────────────────────────
// Principal ideal (a) in Z: {na : n∈Z}
[[nodiscard]] constexpr bool in_principal_ideal(i64 a, i64 x) noexcept {
    return a!=0 && x%a==0;
}

// Sum of ideals (a)+(b) = (gcd(a,b))
[[nodiscard]] constexpr i64 ideal_sum(i64 a, i64 b) noexcept {
    while(b){i64 t=b;b=a%b;a=t;} return a;
}

// Product of ideals (a)*(b) = (ab)
[[nodiscard]] constexpr i64 ideal_product(i64 a, i64 b) noexcept { return a*b; }

// Intersection (a)∩(b) = (lcm(a,b))
[[nodiscard]] constexpr i64 ideal_intersection(i64 a, i64 b) noexcept {
    i64 g=ideal_sum(a,b); return (g==0)?0:a/g*b;
}

// ─── Quotient ring Z/nZ properties ───────────────────────────────────────────
[[nodiscard]] constexpr bool quotient_is_field(i64 n) noexcept {
    // Z/nZ is a field iff n is prime
    if(n<=1) return false;
    for(i64 p=2;p*p<=n;++p) if(n%p==0) return false;
    return true;
}

[[nodiscard]] constexpr bool quotient_is_integral_domain(i64 n) noexcept {
    return quotient_is_field(n); // Z/nZ is integral domain iff prime
}

// ─── Polynomial ring R[x] over Z (simple, bounded degree) ─────────────────────
template<int MAXDEG>
struct PolyZ {
    std::array<i64,MAXDEG+1> c{};

    [[nodiscard]] constexpr int degree() const noexcept {
        for(int i=MAXDEG;i>=0;--i) if(c[i]) return i; return 0;
    }
    [[nodiscard]] constexpr PolyZ operator+(const PolyZ& o) const noexcept {
        PolyZ r; for(int i=0;i<=MAXDEG;++i) r.c[i]=c[i]+o.c[i]; return r;
    }
    [[nodiscard]] constexpr PolyZ operator-(const PolyZ& o) const noexcept {
        PolyZ r; for(int i=0;i<=MAXDEG;++i) r.c[i]=c[i]-o.c[i]; return r;
    }
    [[nodiscard]] constexpr PolyZ operator*(const PolyZ& o) const noexcept {
        PolyZ r{};
        for(int i=0;i<=MAXDEG;++i)
            for(int j=0;j+i<=MAXDEG;++j) r.c[i+j]+=c[i]*o.c[j];
        return r;
    }
    [[nodiscard]] constexpr i64 eval(i64 x) const noexcept {
        i64 r=0,xn=1; for(int i=0;i<=MAXDEG;++i){r+=c[i]*xn;xn*=x;} return r;
    }
    [[nodiscard]] constexpr bool is_zero() const noexcept {
        for(auto v:c) if(v) return false; return true;
    }
    // GCD via subresultant (Euclidean on polynomials)
    [[nodiscard]] constexpr PolyZ gcd(PolyZ b) const noexcept {
        PolyZ a=*this;
        while(!b.is_zero()) {
            // Pseudo-division: a = q*b + r
            int da=a.degree(),db=b.degree();
            if(da<db){PolyZ t=a;a=b;b=t;continue;}
            // Multiply a by lead(b)^(da-db+1) for exact division
            i64 lb=b.c[b.degree()];
            int steps=da-db+1;
            PolyZ r=a;
            i64 scale=1; for(int i=0;i<steps;++i) scale*=lb;
            for(int i=0;i<=MAXDEG;++i) r.c[i]*=scale;
            // Long division
            for(int i=da;i>=db;--i) {
                i64 q=r.c[i]/b.c[db];
                for(int j=0;j<=db;++j) if(i-db+j<=MAXDEG) r.c[i-db+j]-=q*b.c[j];
            }
            a=b; b=r;
        }
        // Remove content
        i64 g=0; for(auto v:a.c){if(v<0)v=-v;while(v){i64 t=g%v;g=v;v=t;}};
        if(g>1) for(auto& v:a.c) v/=g;
        return a;
    }
};

// ─── Factorization in Z (trial division) ─────────────────────────────────────
template<int MAXFACTORS>
struct Factorization {
    std::array<i64,MAXFACTORS> primes{};
    std::array<int,MAXFACTORS> exponents{};
    int count{0};
    i64 sign{1};

    constexpr void add(i64 p, int e) noexcept {
        for(int i=0;i<count;++i) if(primes[i]==p){exponents[i]+=e;return;}
        if(count<MAXFACTORS){primes[count]=p;exponents[count++]=e;}
    }
};

template<int MF=16>
[[nodiscard]] constexpr Factorization<MF> factorize(i64 n) noexcept {
    Factorization<MF> f;
    if(n<0){f.sign=-1;n=-n;}
    for(i64 p=2;p*p<=n;++p) {
        int e=0;
        while(n%p==0){++e;n/=p;}
        if(e) f.add(p,e);
    }
    if(n>1) f.add(n,1);
    return f;
}

// ─── Noetherian ring property: every ideal finitely generated ─────────────────
// Z is Noetherian: every ideal is principal (gcd generator)
// Z[x] is Noetherian (Hilbert basis theorem)
// Check if n-generated ideal equals (g) where g = gcd of generators
template<int K>
[[nodiscard]] constexpr i64 ideal_generator(const std::array<i64,K>& gens) noexcept {
    i64 g=0;
    for(auto v:gens){v=v<0?-v:v;while(v){i64 t=g%v;g=v;v=t;}}
    return g;
}

// ─── Krull dimension (for Z/nZ and polynomial rings) ─────────────────────────
// dim(Z) = 1, dim(field) = 0, dim(Z[x]) = 2
[[nodiscard]] constexpr int krull_dimension_Zn(i64 n) noexcept {
    return quotient_is_field(n)?0:1;
}

// ─── Primary decomposition check (for Z/nZ) ───────────────────────────────────
// Z/nZ has primary decomposition corresponding to prime factorization of n
template<int MF=8>
[[nodiscard]] constexpr std::array<i64,MF> primary_components(i64 n) noexcept {
    auto f=factorize<MF>(n);
    std::array<i64,MF> components{};
    for(int i=0;i<f.count;++i) {
        i64 pk=f.primes[i];
        for(int e=1;e<f.exponents[i];++e) pk*=f.primes[i];
        components[i]=pk;
    }
    return components;
}

} // namespace qs::math
