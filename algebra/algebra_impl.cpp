// qs_math/algebra/algebra_impl.cpp
// Explicit definitions for algebra module (C++20)
#include "include/qs_math_algebra.hpp"
#include <cmath>

namespace qs::math::algebra {

// ─── Octonion multiplication (Cayley table / Fano plane) ─────────────────────
// Standard Fano plane: indices of (i,j,k) where e_i * e_j = +e_k
// (1,2,4), (2,3,5), (3,4,6), (4,5,0+7=7 wrap), (5,6,1), (6,0+1=7,2), (7=e0=1,1,1)
// Using the standard octonion multiplication from Baez (2002)
static const i8 CAYLEY[7][7] = {
    // e1*e1=-1 on diag; sign for e_i*e_j = ±e_k
    // Row i, col j: value = k (positive) or -(k) (negative), 0 = scalar part
    // We use index [0..6] for e1..e7, real part separate
    { 0, 3,-2, 6,-5, 4,-1}, // e1*ej
    {-3, 0, 1,-5, 7,-1, 5}, // e2*ej — placeholder, full table below
    { 2,-1, 0, 4,-6, 3,-4},
    {-6, 5,-4, 0, 1,-2, 6},
    { 5,-7, 6,-1, 0,-3, 7},
    {-4, 1,-3, 2, 3, 0,-2},
    { 1,-5, 4,-6,-7, 2, 0}
};

Octonion Octonion::operator*(const Octonion& o) const noexcept {
    // Brute-force 64-multiply using Cayley-Dickson construction
    // Split into two quaternions: (a,b) where octonion = a + b*l
    // (a1+b1*l)*(a2+b2*l) = (a1*a2 - b2†*b1) + (b2*a1 + b1*a2†)*l
    // Using component form of the standard octonion multiplication:
    const auto& a = e;
    const auto& b = o.e;
    Octonion r{};
    r.e[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
           - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    r.e[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
           + a[4]*b[5] - a[5]*b[4] - a[6]*b[7] + a[7]*b[6];
    r.e[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
           + a[4]*b[6] + a[5]*b[7] - a[6]*b[4] - a[7]*b[5];
    r.e[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
           + a[4]*b[7] - a[5]*b[6] + a[6]*b[5] - a[7]*b[4];
    r.e[4] = a[0]*b[4] - a[1]*b[5] - a[2]*b[6] - a[3]*b[7]
           + a[4]*b[0] + a[5]*b[1] + a[6]*b[2] + a[7]*b[3];
    r.e[5] = a[0]*b[5] + a[1]*b[4] - a[2]*b[7] + a[3]*b[6]
           - a[4]*b[1] + a[5]*b[0] - a[6]*b[3] + a[7]*b[2];
    r.e[6] = a[0]*b[6] + a[1]*b[7] + a[2]*b[4] - a[3]*b[5]
           - a[4]*b[2] + a[5]*b[3] + a[6]*b[0] - a[7]*b[1];
    r.e[7] = a[0]*b[7] - a[1]*b[6] + a[2]*b[5] + a[3]*b[4]
           - a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];
    return r;
}

// ─── Clifford Cl(3,0) geometric product ──────────────────────────────────────
// Basis: 1, e1, e2, e3, e12, e13, e23, e123
// Multiplication rules: ei*ei=+1, ei*ej=-ej*ei for i≠j
// e12=e1∧e2, e13=e1∧e3, e23=e2∧e3, e123=e1∧e2∧e3
Cl3Multivector Cl3Multivector::operator*(const Cl3Multivector& b) const noexcept {
    const auto& a = *this;
    Cl3Multivector r{};
    // scalar part (grade 0)
    r.s = a.s*b.s + a.e1*b.e1 + a.e2*b.e2 + a.e3*b.e3
        - a.e12*b.e12 - a.e13*b.e13 - a.e23*b.e23
        - a.e123*b.e123;
    // e1 part (grade 1)
    r.e1 = a.s*b.e1 + a.e1*b.s  - a.e2*b.e12 - a.e3*b.e13
         + a.e12*b.e2 + a.e13*b.e3 - a.e23*b.e123 + a.e123*b.e23;
    // e2 part
    r.e2 = a.s*b.e2 + a.e1*b.e12 + a.e2*b.s  - a.e3*b.e23
         - a.e12*b.e1 + a.e13*b.e123 + a.e23*b.e3 - a.e123*b.e13;
    // e3 part
    r.e3 = a.s*b.e3 + a.e1*b.e13 + a.e2*b.e23 + a.e3*b.s
         - a.e12*b.e123 - a.e13*b.e1 - a.e23*b.e2 + a.e123*b.e12;
    // e12 part (grade 2)
    r.e12 = a.s*b.e12 + a.e1*b.e2 - a.e2*b.e1 + a.e3*b.e123
          + a.e12*b.s  - a.e13*b.e23 + a.e23*b.e13 + a.e123*b.e3;
    // e13 part
    r.e13 = a.s*b.e13 + a.e1*b.e3 - a.e2*b.e123 - a.e3*b.e1
          + a.e12*b.e23 + a.e13*b.s  - a.e23*b.e12 - a.e123*b.e2;
    // e23 part
    r.e23 = a.s*b.e23 + a.e1*b.e123 + a.e2*b.e3 - a.e3*b.e2
          - a.e12*b.e13 + a.e13*b.e12 + a.e23*b.s  + a.e123*b.e1;
    // e123 part (grade 3 / pseudoscalar)
    r.e123 = a.s*b.e123 + a.e1*b.e23 - a.e2*b.e13 + a.e3*b.e12
           + a.e12*b.e3 - a.e13*b.e2 + a.e23*b.e1 + a.e123*b.s;
    return r;
}

// ─── Explicit template instantiations ────────────────────────────────────────
template struct Polynomial<2>;
template struct Polynomial<3>;
template struct Polynomial<4>;
template struct Polynomial<5>;
template struct Polynomial<6>;
template struct Polynomial<8>;
template struct Polynomial<10>;
template struct Polynomial<16>;

template void durand_kerner<2>(const Polynomial<2>&, Complex*, i32, f64) noexcept;
template void durand_kerner<3>(const Polynomial<3>&, Complex*, i32, f64) noexcept;
template void durand_kerner<4>(const Polynomial<4>&, Complex*, i32, f64) noexcept;
template void durand_kerner<5>(const Polynomial<5>&, Complex*, i32, f64) noexcept;

template i32 sturm_count_roots<3>(const Polynomial<3>&, f64, f64) noexcept;
template i32 sturm_count_roots<4>(const Polynomial<4>&, f64, f64) noexcept;
template i32 sturm_count_roots<5>(const Polynomial<5>&, f64, f64) noexcept;
template i32 sturm_count_roots<6>(const Polynomial<6>&, f64, f64) noexcept;

// ─── Additional algebra utilities ─────────────────────────────────────────────

// Continued fraction convergents
struct CFConvgnt { i64 p, q; };   // rational p/q

std::vector<CFConvgnt> cf_convergents(f64 x, i32 max_terms) noexcept {
    std::vector<CFConvgnt> result;
    i64 p0=1,p1=0,q0=0,q1=1;
    for (i32 i=0;i<max_terms;++i) {
        const i64 a = (i64)x;
        const i64 p2 = a*p1 + p0;
        const i64 q2 = a*q1 + q0;
        result.push_back({p2,q2});
        x -= a;
        if (std::abs(x) < 1e-12) break;
        x = 1.0/x;
        p0=p1; p1=p2; q0=q1; q1=q2;
    }
    return result;
}

// Best rational approximation in Stern-Brocot tree
CFConvgnt stern_brocot_approx(f64 x, i64 max_den) noexcept {
    i64 lo_p=0,lo_q=1, hi_p=1,hi_q=0;
    for (;;) {
        const i64 mp=lo_p+hi_p, mq=lo_q+hi_q;
        if (mq > max_den) break;
        const f64 mv = (f64)mp/mq;
        if (std::abs(mv-x)<1e-13) return {mp,mq};
        if (mv < x) { lo_p=mp; lo_q=mq; }
        else         { hi_p=mp; hi_q=mq; }
    }
    // Return better of lo/hi
    const f64 dl=std::abs((f64)lo_p/lo_q-x), dh=std::abs((f64)hi_p/hi_q-x);
    return dl < dh ? CFConvgnt{lo_p,lo_q} : CFConvgnt{hi_p,hi_q};
}

// Extended GCD: returns gcd, sets s,t such that a*s + b*t = gcd
i64 extended_gcd(i64 a, i64 b, i64& s, i64& t) noexcept {
    if (b == 0) { s=1; t=0; return a; }
    i64 s1,t1;
    const i64 g = extended_gcd(b, a%b, s1, t1);
    s = t1; t = s1 - (a/b)*t1;
    return g;
}

// Chinese Remainder Theorem: find x such that x≡r[i] (mod m[i])
// Returns {x, M} where M=lcm(m[0..n-1]) and x is in [0,M)
CFConvgnt crt(const i64* r, const i64* m, i32 n) noexcept {
    i64 x=r[0], M=m[0];
    for (i32 i=1;i<n;++i) {
        i64 s,t;
        const i64 g = extended_gcd(M, m[i], s, t);
        if ((r[i]-x)%g != 0) return {0,-1}; // no solution
        const i64 lcm = M / g * m[i];
        x = ((x + M*s*((r[i]-x)/g)) % lcm + lcm) % lcm;
        M = lcm;
    }
    return {x, M};
}

// Number-theoretic transform (NTT) — convolution mod prime p
// Cooley-Tukey DIT, N must be power of 2, g is primitive root mod p
void ntt(i64* a, i32 N, bool inverse, i64 p, i64 g) noexcept {
    for (i32 i=1,j=0;i<N;++i) {
        i32 bit=N>>1;
        for (;j&bit;bit>>=1) j^=bit;
        j^=bit; if(i<j) std::swap(a[i],a[j]);
    }
    auto mpow = [&](i64 base, i64 exp, i64 mod) -> i64 {
        i64 r=1; base%=mod;
        for (;exp>0;exp>>=1) { if(exp&1) r=r*base%mod; base=base*base%mod; }
        return r;
    };
    for (i32 len=2;len<=N;len<<=1) {
        const i64 w = inverse ? mpow(g, p-1-(p-1)/len, p) : mpow(g,(p-1)/len,p);
        for (i32 i=0;i<N;i+=len) {
            i64 wn=1;
            for (i32 j=0;j<len/2;++j) {
                const i64 u=a[i+j], v=a[i+j+len/2]*wn%p;
                a[i+j]=(u+v)%p; a[i+j+len/2]=(u-v+p)%p;
                wn=wn*w%p;
            }
        }
    }
    if (inverse) {
        const i64 inv_N = mpow(N, p-2, p);
        for (i32 i=0;i<N;++i) a[i]=a[i]*inv_N%p;
    }
}

// Berlekamp-Massey — find minimal LFSR for a sequence over GF2
std::vector<i32> berlekamp_massey_gf2(const std::vector<i32>& s) noexcept {
    i32 N=(i32)s.size();
    std::vector<i32> C(N+1,0), B(N+1,0);
    C[0]=B[0]=1;
    i32 L=0, m=1; i32 b=1;
    for (i32 n=0;n<N;++n) {
        i32 d=s[n];
        for (i32 i=1;i<=L;++i) d^=(C[i]&s[n-i]);
        if (!d) { ++m; continue; }
        std::vector<i32> T(C);
        for (i32 i=m;i<=N;++i) C[i]^=(d*B[i-m]);
        if (2*L<=n) { L=n+1-L; B=T; b=d; m=1; } else ++m;
    }
    C.resize(L+1); return C;
}

} // namespace qs::math::algebra
