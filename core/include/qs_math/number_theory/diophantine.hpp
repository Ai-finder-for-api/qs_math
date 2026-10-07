#pragma once
// qs_math/number_theory/diophantine.hpp — Linear Diophantine eqs, Pell, Thue, lattice methods
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <optional>
#include <cmath>

namespace qs::math {

// ─── Extended GCD ─────────────────────────────────────────────────────────────
struct ExtGcd { i64 g,x,y; }; // g=gcd, g=ax+by

[[nodiscard]] constexpr ExtGcd ext_gcd(i64 a, i64 b) noexcept {
    if(b==0) return {a,1,0};
    auto [g,x1,y1]=ext_gcd(b,a%b);
    return {g,y1,x1-(a/b)*y1};
}

// ─── Linear Diophantine equation ax+by=c ──────────────────────────────────────
// Returns (x0,y0) as particular solution; general: x=x0+bt/g, y=y0-at/g
struct DiophantineSol { i64 x0,y0; bool has_solution; };

[[nodiscard]] constexpr DiophantineSol linear_diophantine(i64 a, i64 b, i64 c) noexcept {
    auto [g,x,y]=ext_gcd(std::abs(a),std::abs(b));
    if(c%g!=0) return {0,0,false};
    i64 scale=c/g;
    return {x*(a<0?-scale:scale), y*(b<0?-scale:scale), true};
}

// ─── Chinese Remainder Theorem ────────────────────────────────────────────────
// Solve x ≡ r[i] (mod m[i]) for pairwise coprime m[i]
template<int K>
[[nodiscard]] constexpr std::optional<i64> crt(
    const std::array<i64,K>& r, const std::array<i64,K>& m) noexcept
{
    i64 M=1; for(auto v:m) M*=v;
    i64 x=0;
    for(int i=0;i<K;++i) {
        i64 Mi=M/m[i];
        auto [g,inv,_]=ext_gcd(Mi%m[i],m[i]);
        if(g!=1) return std::nullopt;
        inv=((inv%m[i])+m[i])%m[i];
        x=(x+r[i]*Mi%M*inv)%M;
    }
    return (x+M)%M;
}

// ─── Pell equation: x^2 - D*y^2 = 1 ─────────────────────────────────────────
// Fundamental solution via continued fraction expansion of sqrt(D)
struct PellSol { i64 x,y; };

[[nodiscard]] constexpr PellSol pell_fundamental(i64 D) noexcept {
    i64 a0=(i64)std::sqrt((f64)D);
    if(a0*a0==D) return {0,0}; // D is perfect square, no solution
    // Continued fraction: sqrt(D) = a0 + 1/(a1 + 1/(a2+...))
    i64 m=0,d=1,a=a0;
    i64 p0=1,p1=a0,q0=0,q1=1;
    for(int iter=0;iter<1000;++iter) {
        m=d*a-m; d=(D-m*m)/d; a=(a0+m)/d;
        i64 p2=a*p1+p0, q2=a*q1+q0;
        if(p2*p2-D*q2*q2==1) return {p2,q2};
        p0=p1;p1=p2;q0=q1;q1=q2;
    }
    return {-1,-1}; // not found
}

// Next Pell solution from fundamental: (x_n,y_n) from (x1,y1)
[[nodiscard]] constexpr PellSol pell_next(PellSol curr, PellSol fund, i64 D) noexcept {
    return {curr.x*fund.x+D*curr.y*fund.y, curr.x*fund.y+curr.y*fund.x};
}

// Pell equation x^2 - D*y^2 = -1 (solvable when period of CF is odd)
[[nodiscard]] constexpr std::optional<PellSol> pell_neg1(i64 D) noexcept {
    i64 a0=(i64)std::sqrt((f64)D);
    if(a0*a0==D) return std::nullopt;
    i64 m=0,d=1,a=a0;
    i64 p0=1,p1=a0,q0=0,q1=1;
    int period=0;
    for(int iter=0;iter<2000;++iter) {
        m=d*a-m;d=(D-m*m)/d;a=(a0+m)/d;++period;
        i64 p2=a*p1+p0,q2=a*q1+q0;
        if(p2*p2-D*q2*q2==-1) return PellSol{p2,q2};
        if(a==2*a0) break; // period ends
        p0=p1;p1=p2;q0=q1;q1=q2;
    }
    return std::nullopt;
}

// ─── Pythagorean triples ──────────────────────────────────────────────────────
struct PythTriple { i64 a,b,c; };

// Primitive Pythagorean triple from m>n>0, gcd(m,n)=1, m-n odd:
// a=m^2-n^2, b=2mn, c=m^2+n^2
[[nodiscard]] constexpr PythTriple pythagorean_from_mn(i64 m, i64 n) noexcept {
    return {m*m-n*n, 2*m*n, m*m+n*n};
}

// Count Pythagorean triples with hypotenuse ≤ N
[[nodiscard]] constexpr i64 count_pythagorean_triples(i64 N) noexcept {
    i64 count=0;
    for(i64 a=3;a<N;++a) for(i64 b=a+1;b<N;++b) {
        i64 c2=a*a+b*b;
        i64 c=(i64)std::sqrt((f64)c2);
        if(c*c==c2&&c<=N) ++count;
    }
    return count;
}

// ─── Lattice reduction (LLL sketch) ───────────────────────────────────────────
// Two-dimensional Gauss lattice reduction (finds short basis)
template<int DIM>
struct LatticeBasis {
    std::array<std::array<f64,DIM>,DIM> rows{};

    [[nodiscard]] constexpr f64 dot(int i, int j) const noexcept {
        f64 s=0; for(int k=0;k<DIM;++k) s+=rows[i][k]*rows[j][k]; return s;
    }
    [[nodiscard]] constexpr f64 norm2(int i) const noexcept { return dot(i,i); }
};

// Gram-Schmidt coefficient μ_{ij}
template<int DIM>
[[nodiscard]] constexpr f64 gs_mu(const LatticeBasis<DIM>& B,
                                    const LatticeBasis<DIM>& Bstar,
                                    int i, int j) noexcept {
    f64 d=Bstar.norm2(j); if(d<1e-30) return 0;
    return B.dot_approx(i,j)/d; // would use Bstar.rows[j] dot B.rows[i]
    // Simplified: return B.dot(i,j)/d; // approximation without full orthogonalization
}

// LLL reduction (2D case, exact)
template<>
[[nodiscard]] constexpr LatticeBasis<2> lll_reduce<2>(LatticeBasis<2> B) noexcept {
    for(int iter=0;iter<100;++iter) {
        // Gram-Schmidt
        f64 mu=B.dot(1,0)/B.norm2(0);
        i64 q=(i64)std::round(mu);
        for(int k=0;k<2;++k) B.rows[1][k]-=q*B.rows[0][k];
        // Lovász condition: ||b*_{i+1}||^2 >= (3/4-mu^2)||b*_i||^2
        f64 mu2=B.dot(1,0)/B.norm2(0);
        if(B.norm2(1)>=(0.75-mu2*mu2)*B.norm2(0)) break;
        std::swap(B.rows[0],B.rows[1]);
    }
    return B;
}

// ─── Sum of two squares representation n = a^2 + b^2 ─────────────────────────
[[nodiscard]] constexpr std::optional<std::pair<i64,i64>> sum_of_two_squares(i64 n) noexcept {
    for(i64 a=0;a*a<=n/2;++a) {
        i64 b2=n-a*a;
        i64 b=(i64)std::sqrt((f64)b2);
        if(b*b==b2) return {{a,b}};
    }
    return std::nullopt;
}

// Lagrange 4-square: every n is sum of 4 squares
[[nodiscard]] constexpr std::array<i64,4> sum_of_four_squares(i64 n) noexcept {
    for(i64 a=0;a*a<=n;++a) for(i64 b=a;a*a+b*b<=n;++b)
        for(i64 c=b;a*a+b*b+c*c<=n;++c) {
            i64 r=n-a*a-b*b-c*c;
            i64 d=(i64)std::sqrt((f64)r);
            if(d*d==r) return {a,b,c,d};
        }
    return {0,0,0,n}; // fallback (n=n=0^2+0^2+0^2+sqrt(n)^2 if perfect square)
}

} // namespace qs::math
