#pragma once
// qs_math/algebra/galois_theory.hpp — Galois groups, field extensions, splitting fields
#include "../arithmetic/scalar_types.hpp"
#include "../number_theory/primes.hpp"
#include <array>
#include <optional>

namespace qs::math {

// ─── Polynomial over Z_p ──────────────────────────────────────────────────────
template<int DEG, i64 P>
struct PolyZp {
    std::array<i64,DEG+1> c{}; // c[0]=constant term

    [[nodiscard]] constexpr i64 eval(i64 x) const noexcept {
        i64 r=0, xn=1;
        for(int i=0;i<=DEG;++i){r=(r+c[i]*xn)%P;xn=xn*x%P;}
        return (r+P)%P;
    }

    [[nodiscard]] constexpr bool is_zero() const noexcept {
        for(auto v:c) if(v%P!=0) return false; return true;
    }

    [[nodiscard]] constexpr PolyZp operator+(const PolyZp& o) const noexcept {
        PolyZp r; for(int i=0;i<=DEG;++i) r.c[i]=(c[i]+o.c[i])%P; return r;
    }
    [[nodiscard]] constexpr PolyZp operator-(const PolyZp& o) const noexcept {
        PolyZp r; for(int i=0;i<=DEG;++i) r.c[i]=(c[i]-o.c[i]+P)%P; return r;
    }
};

// ─── GF(p^n) arithmetic (extension field via irreducible polynomial) ──────────
// Elements of GF(p^n) represented as polynomials mod irred. poly of degree n over Z_p
template<int N, i64 P>
struct GFpn {
    std::array<i64,N> coeff{}; // element = sum coeff[i]*alpha^i

    [[nodiscard]] constexpr bool is_zero() const noexcept {
        for(auto v:coeff) if(v) return false; return true;
    }
    [[nodiscard]] constexpr bool operator==(const GFpn& o) const noexcept {
        for(int i=0;i<N;++i) if(coeff[i]!=o.coeff[i]) return false; return true;
    }
    [[nodiscard]] constexpr GFpn operator+(const GFpn& o) const noexcept {
        GFpn r; for(int i=0;i<N;++i) r.coeff[i]=(coeff[i]+o.coeff[i])%P; return r;
    }
    [[nodiscard]] constexpr GFpn operator-(const GFpn& o) const noexcept {
        GFpn r; for(int i=0;i<N;++i) r.coeff[i]=(coeff[i]-o.coeff[i]+P)%P; return r;
    }
    // Multiply with reduction mod irred poly (provided as array of N+1 coefficients)
    [[nodiscard]] constexpr GFpn mul_mod(const GFpn& o,
                                          const std::array<i64,N+1>& irred) const noexcept {
        std::array<i64,2*N-1> prod{};
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            prod[i+j]=(prod[i+j]+coeff[i]*o.coeff[j])%P;
        // Reduce mod irred
        for(int i=2*N-2;i>=N;--i) {
            i64 lead=prod[i]; if(!lead) continue;
            for(int j=0;j<N;++j)
                prod[i-N+j]=(prod[i-N+j]-lead*irred[j]%P+P)%P;
            prod[i]=0;
        }
        GFpn r;
        for(int i=0;i<N;++i) r.coeff[i]=prod[i];
        return r;
    }
};

// ─── Irreducibility test (polynomial over Z_p) ─────────────────────────────────
// A poly f of degree n over Z_p is irreducible iff:
// 1) f | (x^{p^n} - x)
// 2) gcd(f, x^{p^i} - x) = 1 for all i | n, i < n
// We check roots (sufficient for low degree)
template<int DEG, i64 P>
[[nodiscard]] constexpr bool is_irreducible_over_fp(const PolyZp<DEG,P>& f) noexcept {
    // No roots in Z_p and squarefree → irreducible for DEG≤3
    for(i64 x=0;x<P;++x) if(f.eval(x)==0) return false;
    return true; // for DEG<=3; general case needs x^{p^k} mod f computation
}

// ─── Galois group order (degree of splitting field extension) ─────────────────
// For irreducible poly of degree n over Q: |Gal| divides n!
// Discriminant-based classification for quadratic and cubic
[[nodiscard]] constexpr int galois_group_order_quadratic(i64 disc) noexcept {
    // Quadratic: Gal = Z/2 if disc not a perfect square, trivial otherwise
    if(disc<0) return 2;
    i64 s=(i64)std::sqrt((f64)disc);
    return (s*s==disc)?1:2;
}

// Cubic: Gal(f/Q) based on discriminant Δ
// Δ>0 and sqrt(Δ)∈Q → A3≅Z/3 (order 3)
// Δ>0 and sqrt(Δ)∉Q → S3 (order 6)
// Δ<0 → S3 (order 6)
[[nodiscard]] constexpr int galois_group_order_cubic(i64 a, i64 b, i64 c, i64 d) noexcept {
    // Δ = 18abcd - 4b^3d + b^2c^2 - 4ac^3 - 27a^2d^2
    i64 disc = 18*a*b*c*d - 4*b*b*b*d + b*b*c*c - 4*a*c*c*c - 27*a*a*d*d;
    if(disc<0) return 6;
    i64 s=(i64)std::sqrt((f64)disc);
    return (s*s==disc)?3:6;
}

// ─── Field extension [F:K] degree ────────────────────────────────────────────
// Minimal polynomial degree of alpha over base field
template<int MAXDEG>
[[nodiscard]] constexpr int extension_degree_via_min_poly(
    const std::array<f64,MAXDEG>& min_poly_coeffs) noexcept
{
    // degree = index of leading nonzero coefficient
    for(int i=MAXDEG-1;i>=0;--i)
        if(std::abs(min_poly_coeffs[i])>1e-12) return i;
    return 0;
}

// ─── Frobenius endomorphism φ_p: x → x^p in GF(p^n) ─────────────────────────
template<int N, i64 P>
[[nodiscard]] constexpr GFpn<N,P> frobenius(const GFpn<N,P>& x,
                                              const std::array<i64,N+1>& irred) noexcept {
    // x^p mod irred (repeated squaring)
    GFpn<N,P> result{}; result.coeff[0]=1;
    GFpn<N,P> base=x;
    i64 exp=P;
    while(exp>0) {
        if(exp&1) result=result.mul_mod(base,irred);
        base=base.mul_mod(base,irred);
        exp>>=1;
    }
    return result;
}

// ─── Normal basis (all conjugates form a basis) ───────────────────────────────
// Check if alpha generates a normal basis: {alpha, alpha^p, alpha^{p^2}, ..., alpha^{p^{n-1}}}
template<int N, i64 P>
[[nodiscard]] constexpr bool is_normal_basis_generator(
    const GFpn<N,P>& alpha, const std::array<i64,N+1>& irred) noexcept
{
    // Check that the n conjugates are linearly independent (nonzero det of matrix)
    std::array<GFpn<N,P>,N> conjugates{};
    conjugates[0]=alpha;
    for(int i=1;i<N;++i) conjugates[i]=frobenius(conjugates[i-1],irred);
    // Build matrix and check det != 0 (mod p)
    std::array<std::array<i64,N>,N> mat{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) mat[i][j]=conjugates[i].coeff[j];
    // Gaussian elimination mod p
    i64 det=1;
    for(int col=0;col<N;++col) {
        int pivot=-1;
        for(int row=col;row<N;++row) if(mat[row][col]%P!=0){pivot=row;break;}
        if(pivot<0) return false;
        if(pivot!=col){std::swap(mat[pivot],mat[col]);det=(P-det)%P;}
        i64 inv=1,base=mat[col][col]%P,e=P-2;
        while(e>0){if(e&1)inv=inv*base%P;base=base*base%P;e>>=1;}
        det=det*mat[col][col]%P;
        for(int row=0;row<N;++row) if(row!=col) {
            i64 f=mat[row][col]*inv%P;
            for(int j=col;j<N;++j) mat[row][j]=(mat[row][j]-f*mat[col][j]%P+P)%P;
        }
    }
    return det%P!=0;
}

// ─── Solvability by radicals ──────────────────────────────────────────────────
// A polynomial is solvable by radicals iff its Galois group is solvable
// Solvable groups: trivial, cyclic, dihedral, A4, S4 — NOT S5 and above
[[nodiscard]] constexpr bool galois_group_solvable(int order, bool is_simple) noexcept {
    // Simple groups of order > 1 are not solvable (except Z/p)
    if(is_simple && order > 1) {
        // Z/p is simple and solvable; A5 (order 60) is simple and not solvable
        return order < 60; // prime cyclic groups
    }
    // Solvable: order is a product of primes (nilpotent) or order < 60
    return order < 60;
}

} // namespace qs::math
