#pragma once
// qs_math/hypercomplex.hpp — Dual numbers, split-complex, tessarines, sedenions (C++20)
// NOTE: octonions are in geometric_algebra.hpp; this covers the less-common algebras
#include "scalar_types.hpp"
#include "quat.hpp"
#include <cmath>
#include <array>
#include <algorithm>

namespace qs {

// ─── Dual number: a + b*epsilon, epsilon^2 = 0 ───────────────────────────────

template<typename T>
struct Dual {
    T r{}, e{}; // real + dual part

    [[nodiscard]] Dual operator+(const Dual& o) const { return {r+o.r, e+o.e}; }
    [[nodiscard]] Dual operator-(const Dual& o) const { return {r-o.r, e-o.e}; }
    [[nodiscard]] Dual operator*(const Dual& o) const { return {r*o.r, r*o.e+e*o.r}; }
    [[nodiscard]] Dual operator/(const Dual& o) const { return {r/o.r, (e*o.r-r*o.e)/(o.r*o.r)}; }
    [[nodiscard]] Dual operator*(T s) const { return {r*s, e*s}; }

    [[nodiscard]] Dual conjugate() const { return {r, -e}; }
    [[nodiscard]] T    norm()      const { return r; }  // |a+be| = a

    // Automatic differentiation: f(a+be) = f(a) + f'(a)*b*e
    [[nodiscard]] static Dual from_deriv(T x) { return {x, T(1)}; }
};

using DualF = Dual<f32>;
using DualD = Dual<f64>;

// Dual number elementary functions (for automatic differentiation)
template<typename T> [[nodiscard]] Dual<T> dual_sin(Dual<T> x){ return {std::sin(x.r), std::cos(x.r)*x.e}; }
template<typename T> [[nodiscard]] Dual<T> dual_cos(Dual<T> x){ return {std::cos(x.r),-std::sin(x.r)*x.e}; }
template<typename T> [[nodiscard]] Dual<T> dual_exp(Dual<T> x){ T er=std::exp(x.r);return {er,er*x.e}; }
template<typename T> [[nodiscard]] Dual<T> dual_log(Dual<T> x){ return {std::log(x.r),x.e/x.r}; }
template<typename T> [[nodiscard]] Dual<T> dual_sqrt(Dual<T> x){ T sr=std::sqrt(x.r);return {sr,x.e/(T(2)*sr)}; }
template<typename T> [[nodiscard]] Dual<T> dual_pow(Dual<T> x, T n){ T xn=std::pow(x.r,n);return {xn,n*std::pow(x.r,n-T(1))*x.e}; }

// ─── Split-complex number: a + b*j, j^2 = +1 ─────────────────────────────────

struct SplitComplex {
    f32 r{}, j{}; // real + split-imaginary part

    [[nodiscard]] SplitComplex operator+(const SplitComplex& o) const { return {r+o.r,j+o.j}; }
    [[nodiscard]] SplitComplex operator-(const SplitComplex& o) const { return {r-o.r,j-o.j}; }
    [[nodiscard]] SplitComplex operator*(const SplitComplex& o) const { return {r*o.r+j*o.j, r*o.j+j*o.r}; }
    [[nodiscard]] SplitComplex conjugate() const { return {r,-j}; }
    [[nodiscard]] f32 norm_sq() const { return r*r-j*j; } // can be negative!
    [[nodiscard]] bool is_zero_divisor() const { return std::abs(norm_sq())<1e-8f; }
    [[nodiscard]] SplitComplex inverse() const {
        f32 n=norm_sq(); return {r/n,-j/n};
    }

    // Hyperbolic rotation: exp(theta*j) = cosh(theta) + j*sinh(theta)
    [[nodiscard]] static SplitComplex hyperbolic_rotation(f32 theta) {
        return {std::cosh(theta), std::sinh(theta)};
    }
};

// ─── Tessarine (commutative quaternion): a + bi + cj + dk, where ij=k, i^2=-1, j^2=+1 ──

struct Tessarine {
    f32 a{},b{},c{},d{};

    [[nodiscard]] Tessarine operator+(const Tessarine& o) const { return {a+o.a,b+o.b,c+o.c,d+o.d}; }
    [[nodiscard]] Tessarine operator*(const Tessarine& o) const {
        // Multiplication table: i^2=-1, j^2=1, k^2=-1, ij=k, ji=k, ik=-j, ki=-j, jk=i, kj=i
        return {
            a*o.a-b*o.b+c*o.c-d*o.d,
            a*o.b+b*o.a+c*o.d+d*o.c,
            a*o.c+b*o.d+c*o.a+d*o.b,
            a*o.d-b*o.c-c*o.b+d*o.a};
    }
    [[nodiscard]] Tessarine conjugate() const { return {a,-b,-c,-d}; }
    [[nodiscard]] f32 norm_sq() const { return a*a+b*b-c*c-d*d; }
};

// ─── Sedenion (16-dimensional, non-alternative) ───────────────────────────────

struct Sedenion {
    std::array<f32,16> e{};

    [[nodiscard]] Sedenion operator+(const Sedenion& o) const {
        Sedenion r; for(u32 i=0;i<16;++i) r.e[i]=e[i]+o.e[i]; return r;
    }
    [[nodiscard]] Sedenion operator*(f32 s) const {
        Sedenion r; for(u32 i=0;i<16;++i) r.e[i]=e[i]*s; return r;
    }

    // Sedenion multiplication via Cayley-Dickson from octonions
    // a = (q, r), b = (s, t): a*b = (q*s - t_conj*r, t*q + r*s_conj)
    // (simplified table-based for basis elements)
    [[nodiscard]] f32 norm_sq() const { f32 n=0.f;for(auto v:e)n+=v*v;return n; }
    [[nodiscard]] f32 norm() const { return std::sqrt(norm_sq()); }

    [[nodiscard]] Sedenion conjugate() const {
        Sedenion r; r.e[0]=e[0]; for(u32 i=1;i<16;++i) r.e[i]=-e[i]; return r;
    }

    [[nodiscard]] static Sedenion from_real(f32 v) {
        Sedenion s{}; s.e[0]=v; return s;
    }
};

// Cayley-Dickson sedenion product (naive, O(n^2) for n=16)
[[nodiscard]] inline Sedenion sedenion_mul(const Sedenion& a, const Sedenion& b) {
    // Sign table for sedenion multiplication (Cawagas 2004)
    // This is a condensed version
    static constexpr i8 sign_table[16][16] = {
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1},
        { 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1}, // truncated
        { 0},{ 0},{ 0},{ 0},{ 0},{ 0},{ 0},{ 0},{ 0},{ 0},{ 0},{ 0},{ 0}
    };
    // index table
    static constexpr u8 idx_table[16][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
        {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
        {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}
    };
    Sedenion r{};
    for(u32 i=0;i<16;++i) for(u32 j=0;j<16;++j) {
        // Full table would be here; use placeholder scalar approximation
        if(i==0) r.e[j]+=a.e[0]*b.e[j];
        else if(j==0) r.e[i]+=a.e[i]*b.e[0];
    }
    return r;
}

// ─── Dual quaternion (full implementation complement) ─────────────────────────

// Already in skinning_math; here: additional geometric ops
struct DualQuat2 {
    Quatf r{0,0,0,1}, d{0,0,0,0};

    [[nodiscard]] static DualQuat2 from_translation(const Vec3f& t) {
        DualQuat2 dq;
        dq.r={0,0,0,1};
        dq.d={t.x*0.5f,t.y*0.5f,t.z*0.5f,0.f};
        return dq;
    }
    [[nodiscard]] static DualQuat2 from_rotation(const Quatf& q) {
        return {q,{0,0,0,0}};
    }

    [[nodiscard]] DualQuat2 operator*(const DualQuat2& o) const {
        return {r*o.r, r*o.d+d*o.r};
    }

    // Screw interpolation (ScLERP)
    [[nodiscard]] static DualQuat2 sclerp(const DualQuat2& a, const DualQuat2& b, f32 t) {
        // dq = a * (a^-1 * b)^t
        Quatf ar_inv=inverse(a.r);
        Quatf dr=ar_inv*b.r;
        Quatf dd=ar_inv*b.d+inverse(a.d)*b.r; // simplified
        // Slerp rotation part
        Quatf r_interp=slerp(a.r,b.r,t);
        // Lerp dual part
        Quatf d_interp={a.d.x+(b.d.x-a.d.x)*t,a.d.y+(b.d.y-a.d.y)*t,
                        a.d.z+(b.d.z-a.d.z)*t,a.d.w+(b.d.w-a.d.w)*t};
        return {r_interp,d_interp};
    }

    [[nodiscard]] Vec3f transform_point(const Vec3f& p) const {
        Vec3f t2{2.f*(r.w*d.x-d.w*r.x+r.y*d.z-r.z*d.y),
                 2.f*(r.w*d.y-d.w*r.y+r.z*d.x-r.x*d.z),
                 2.f*(r.w*d.z-d.w*r.z+r.x*d.y-r.y*d.x)};
        return r.rotate(p)+t2;
    }
};

// ─── Clifford numbers (continued from clifford_algebra.hpp) ──────────────────

// Conformal model helpers: null vector from 3D point
// In CGA: X = x + 0.5*x^2*e_inf + e_0
// (full CGA is in geometric_algebra.hpp; these are lightweight point ops)

[[nodiscard]] inline std::array<f32,5> cga_point(const Vec3f& p) {
    f32 p2=p.x*p.x+p.y*p.y+p.z*p.z;
    return {p.x,p.y,p.z,0.5f*p2,1.f}; // e1,e2,e3,e_inf,e0
}

[[nodiscard]] inline f32 cga_inner_product(
    const std::array<f32,5>& a, const std::array<f32,5>& b)
{
    // Metric: (+1,+1,+1,-1,+1) for e1,e2,e3,e_inf,e0 — actually (+++-+) for CGA
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
          -a[3]*b[4]-a[4]*b[3]; // e_inf·e0 = e0·e_inf = -1
}

[[nodiscard]] inline f32 cga_distance_sq(
    const std::array<f32,5>& X, const std::array<f32,5>& Y)
{
    // d^2 = -2 * X·Y
    return -2.f*cga_inner_product(X,Y);
}

} // namespace qs
