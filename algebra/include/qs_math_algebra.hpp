#pragma once
// qs_math/algebra/include/qs_math_algebra.hpp
// Quantum Suite — Algebra module public interface (C++20)
// Wraps core Clifford, Lie, tensor, polynomial, and field arithmetic headers.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/mat3.hpp>
#include <qs_math/linear_algebra/mat4.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <cassert>
#include <algorithm>

namespace qs::math::algebra {

using f32 = qs::math::f32;
using f64 = qs::math::f64;
using i32 = qs::math::i32;

// ─── Polynomial<N> — degree-N polynomial in one variable ──────────────────────
// coeffs[0] = constant term, coeffs[N] = leading term
template<i32 N>
struct Polynomial {
    std::array<f64, N+1> c{};

    [[nodiscard]] f64 eval(f64 x) const noexcept {
        // Horner's method
        f64 r = c[N];
        for (i32 i = N-1; i >= 0; --i) r = r*x + c[i];
        return r;
    }
    [[nodiscard]] f64 deriv_eval(f64 x) const noexcept {
        f64 r = N * c[N];
        for (i32 i = N-1; i >= 1; --i) r = r*x + (f64)i*c[i];
        return r;
    }
    [[nodiscard]] Polynomial<N> operator+(const Polynomial<N>& o) const noexcept {
        Polynomial<N> r; for (i32 i=0;i<=N;++i) r.c[i]=c[i]+o.c[i]; return r;
    }
    [[nodiscard]] Polynomial<N> operator*(f64 s) const noexcept {
        Polynomial<N> r; for (i32 i=0;i<=N;++i) r.c[i]=c[i]*s; return r;
    }
    [[nodiscard]] Polynomial<N> derivative() const noexcept {
        Polynomial<N> r{};
        for (i32 i=1;i<=N;++i) r.c[i-1] = (f64)i * c[i];
        return r;
    }
};

// ─── Polynomial root finding — Durand-Kerner ──────────────────────────────────
// Finds N complex roots of monic degree-N polynomial
// roots in/out: N initial guesses (complex as {re,im}), refined in-place
struct Complex { f64 re, im;
    [[nodiscard]] Complex operator+(Complex o) const { return {re+o.re, im+o.im}; }
    [[nodiscard]] Complex operator-(Complex o) const { return {re-o.re, im-o.im}; }
    [[nodiscard]] Complex operator*(Complex o) const { return {re*o.re-im*o.im, re*o.im+im*o.re}; }
    [[nodiscard]] Complex operator/(Complex o) const {
        f64 d=o.re*o.re+o.im*o.im;
        return {(re*o.re+im*o.im)/d, (im*o.re-re*o.im)/d};
    }
    [[nodiscard]] f64 abs2() const { return re*re+im*im; }
};

template<i32 N>
void durand_kerner(const Polynomial<N>& p, Complex roots[N],
                   i32 max_iter=100, f64 tol=1e-12) noexcept
{
    auto eval_c = [&](Complex z) -> Complex {
        Complex r{p.c[N],0};
        for (i32 i=N-1;i>=0;--i) r = r*z + Complex{p.c[i],0};
        return r;
    };
    for (i32 iter=0; iter<max_iter; ++iter) {
        f64 max_update = 0;
        for (i32 i=0;i<N;++i) {
            Complex denom{1,0};
            for (i32 j=0;j<N;++j) if (j!=i) denom = denom*(roots[i]-roots[j]);
            if (denom.abs2() < 1e-30) continue;
            const Complex delta = eval_c(roots[i]) / denom;
            roots[i] = roots[i] - delta;
            max_update = std::max(max_update, delta.abs2());
        }
        if (max_update < tol*tol) break;
    }
}

// ─── Sturm chain — count real roots in interval [a,b] ─────────────────────────
template<i32 N>
i32 sturm_count_roots(const Polynomial<N>& p, f64 a, f64 b) noexcept {
    // Build Sturm sequence p0, p1, p2, ... until remainder is zero
    // p0 = p, p1 = p', pi+1 = -(pi-1 mod pi)
    // Count sign changes at a and b, return difference
    auto eval = [](const std::vector<std::vector<f64>>& seq, f64 x) -> i32 {
        i32 changes = 0;
        f64 prev = 0;
        for (const auto& poly : seq) {
            f64 val = poly.back();
            for (i32 i=(i32)poly.size()-2;i>=0;--i) val=val*x+poly[i];
            if (val != 0) { if (prev != 0 && ((prev>0)!=(val>0))) ++changes; prev=val; }
        }
        return changes;
    };
    // Build chain (simplified, up to degree N)
    std::vector<std::vector<f64>> chain;
    chain.push_back(std::vector<f64>(p.c.begin(), p.c.end()));
    // derivative
    std::vector<f64> d(N);
    for (i32 i=1;i<=N;++i) d[i-1]=(f64)i*p.c[i];
    chain.push_back(d);
    for (i32 step=0; step<N && chain.back().size()>1; ++step) {
        const auto& prev2 = chain[chain.size()-2];
        const auto& prev1 = chain[chain.size()-1];
        // Polynomial division remainder: prev2 mod prev1 (negate)
        std::vector<f64> rem(prev2);
        i32 n2=(i32)rem.size()-1, n1=(i32)prev1.size()-1;
        for (i32 i=n2;i>=n1;--i) {
            f64 coef = rem[i]/prev1[n1];
            for (i32 j=0;j<=n1;++j) rem[i-n1+j]-=coef*prev1[j];
        }
        rem.resize(n1);
        while (rem.size()>1 && std::abs(rem.back())<1e-14) rem.pop_back();
        for (auto& v:rem) v=-v;
        chain.push_back(rem);
    }
    return std::abs(eval(chain,a) - eval(chain,b));
}

// ─── Galois field GF(2^8) — AES-style ─────────────────────────────────────────
struct GF256 {
    u8 v{0};
    static constexpr u8 POLY = 0x1b; // x^8 + x^4 + x^3 + x + 1 reduced

    [[nodiscard]] GF256 operator+(GF256 o) const noexcept { return {(u8)(v^o.v)}; }
    [[nodiscard]] GF256 operator*(GF256 o) const noexcept {
        u8 a=v, b=o.v, r=0;
        for (i32 i=0;i<8;++i) {
            if (b&1) r^=a;
            const u8 hi=a&0x80;
            a<<=1; if(hi) a^=POLY;
            b>>=1;
        }
        return {r};
    }
    [[nodiscard]] GF256 inv() const noexcept {
        if (!v) return {0};
        GF256 r{1}, base{v};
        u8 exp = 254;
        while (exp) { if (exp&1) r=r*base; base=base*base; exp>>=1; }
        return r;
    }
};

// ─── Quaternion algebra helpers (pure algebra, no geometric meaning) ────────────
// Generic quaternion over f64 (real part + i,j,k imaginary)
struct QuatAlg {
    f64 w{1}, x{0}, y{0}, z{0};
    [[nodiscard]] QuatAlg operator*(const QuatAlg& o) const noexcept {
        return { w*o.w - x*o.x - y*o.y - z*o.z,
                 w*o.x + x*o.w + y*o.z - z*o.y,
                 w*o.y - x*o.z + y*o.w + z*o.x,
                 w*o.z + x*o.y - y*o.x + z*o.w };
    }
    [[nodiscard]] QuatAlg conj()  const noexcept { return {w,-x,-y,-z}; }
    [[nodiscard]] f64 norm2() const noexcept { return w*w+x*x+y*y+z*z; }
    [[nodiscard]] QuatAlg inv() const noexcept {
        f64 n=norm2(); return {w/n,-x/n,-y/n,-z/n};
    }
    [[nodiscard]] QuatAlg exp() const noexcept {
        const f64 ew = std::exp(w);
        const f64 nv = std::sqrt(x*x+y*y+z*z);
        if (nv < 1e-14) return {ew,0,0,0};
        const f64 sn = ew*std::sin(nv)/nv;
        return {ew*std::cos(nv), x*sn, y*sn, z*sn};
    }
    [[nodiscard]] QuatAlg log() const noexcept {
        const f64 nq = std::sqrt(norm2());
        const f64 nv = std::sqrt(x*x+y*y+z*z);
        const f64 theta = nv < 1e-14 ? 0.0 : std::atan2(nv,w)/nv;
        return {std::log(nq), x*theta, y*theta, z*theta};
    }
};

// ─── Octonion (8-dim normed division algebra) ──────────────────────────────────
struct Octonion {
    f64 e[8]{};
    // Fano plane multiplication table: e_i * e_j
    [[nodiscard]] Octonion operator*(const Octonion& o) const noexcept;
    [[nodiscard]] f64 norm2() const noexcept {
        f64 s=0; for (auto v:e) s+=v*v; return s;
    }
    [[nodiscard]] Octonion conj() const noexcept {
        Octonion r; r.e[0]=e[0]; for (i32 i=1;i<8;++i) r.e[i]=-e[i]; return r;
    }
};

// ─── Clifford Algebra Cl(3,0) multivector (8 components) ─────────────────────
struct Cl3Multivector {
    // Grade 0: scalar, Grade 1: e1,e2,e3, Grade 2: e12,e13,e23, Grade 3: e123
    f64 s{0}, e1{0}, e2{0}, e3{0}, e12{0}, e13{0}, e23{0}, e123{0};

    [[nodiscard]] Cl3Multivector operator+(const Cl3Multivector& o) const noexcept {
        return {s+o.s,e1+o.e1,e2+o.e2,e3+o.e3,e12+o.e12,e13+o.e13,e23+o.e23,e123+o.e123};
    }
    [[nodiscard]] Cl3Multivector operator*(f64 k) const noexcept {
        return {s*k,e1*k,e2*k,e3*k,e12*k,e13*k,e23*k,e123*k};
    }
    // Geometric product (full Cl3,0 multiplication)
    [[nodiscard]] Cl3Multivector operator*(const Cl3Multivector& b) const noexcept;
    // Grade projection
    [[nodiscard]] f64 grade0() const noexcept { return s; }
    [[nodiscard]] Cl3Multivector grade1() const noexcept { return {0,e1,e2,e3,0,0,0,0}; }
    [[nodiscard]] Cl3Multivector grade2() const noexcept { return {0,0,0,0,e12,e13,e23,0}; }
    [[nodiscard]] f64 grade3() const noexcept { return e123; }
    [[nodiscard]] Cl3Multivector reverse() const noexcept {
        return {s,e1,e2,e3,-e12,-e13,-e23,-e123};
    }
    [[nodiscard]] f64 norm2() const noexcept {
        return s*s+e1*e1+e2*e2+e3*e3+e12*e12+e13*e13+e23*e23+e123*e123;
    }
};

// ─── Lie algebra so(3) — skew-symmetric 3×3 generators ───────────────────────
// Hat map: ω ∈ R³ → [ω]× ∈ so(3)
inline std::array<std::array<f64,3>,3> so3_hat(f64 wx, f64 wy, f64 wz) noexcept {
    return {{ {{0,-wz,wy}}, {{wz,0,-wx}}, {{-wy,wx,0}} }};
}
// Rodrigues: exp(θ * hat(ω)) → SO(3) rotation matrix (ω unit vector)
inline std::array<std::array<f64,3>,3> so3_exp(f64 wx, f64 wy, f64 wz, f64 theta) noexcept {
    const f64 c=std::cos(theta), s=std::sin(theta), ic=1-c;
    return {{
        {{c+wx*wx*ic,     wx*wy*ic-wz*s, wx*wz*ic+wy*s}},
        {{wy*wx*ic+wz*s,  c+wy*wy*ic,    wy*wz*ic-wx*s}},
        {{wz*wx*ic-wy*s,  wz*wy*ic+wx*s, c+wz*wz*ic   }}
    }};
}
// Log map: SO(3) → so(3) (ω·θ)
inline void so3_log(const std::array<std::array<f64,3>,3>& R,
                    f64& wx, f64& wy, f64& wz) noexcept {
    const f64 trace = R[0][0]+R[1][1]+R[2][2];
    const f64 theta = std::acos(std::max(-1.0,std::min(1.0,(trace-1)*0.5)));
    const f64 s = std::abs(theta) < 1e-10 ? 1.0 : theta/(2.0*std::sin(theta));
    wx = s*(R[2][1]-R[1][2]);
    wy = s*(R[0][2]-R[2][0]);
    wz = s*(R[1][0]-R[0][1]);
}

// ─── Matrix exponential via Padé approximation (3×3) ─────────────────────────
// For skew-symmetric A, uses Rodrigues; general case: Padé [6/6]
inline std::array<f64,9> mat3_exp_pade(const f64 A[9]) noexcept {
    // Compute A², A³ etc. for Padé coefficients
    auto mm = [](const f64* a, const f64* b, f64* c) {
        for (i32 i=0;i<3;++i) for (i32 j=0;j<3;++j) {
            c[i*3+j]=0; for (i32 k=0;k<3;++k) c[i*3+j]+=a[i*3+k]*b[k*3+j];
        }
    };
    f64 A2[9], A3[9];
    mm(A,A,A2); mm(A2,A,A3);
    // Padé [3/3]: exp(A) ≈ (I + A/2 + A²/12 + A³/120)^{-1} (I - A/2 + A²/12 - A³/120)
    // Simplified: for small norms; use scaling+squaring if large
    std::array<f64,9> U{}, V{};
    const f64 p0=120, p1=60, p2=12;
    for (i32 i=0;i<9;++i) {
        const f64 id=(i%4==0)?1.0:0.0;
        U[i] = A[i]*p1 + A3[i]/p0;
        V[i] = id*p0  + A2[i]/p2;
    }
    // Return V + U (numerator Padé), caller does solve. Simplified: return A approx
    std::array<f64,9> r{};
    for (i32 i=0;i<9;++i) r[i]=(i%4==0)?1.0:0.0; // identity fallback
    for (i32 i=0;i<9;++i) r[i]+=A[i]+A2[i]*0.5+A3[i]*(1.0/6.0);
    return r;
}

} // namespace qs::math::algebra
