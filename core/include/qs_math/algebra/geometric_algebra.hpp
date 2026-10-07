#pragma once
// qs_math/geometric_algebra.hpp — Multivectors, bivectors, rotors (GA in 3D/4D, C++20)
// Implements Geometric Algebra Cl(3,0) and Cl(4,0) for 3D/4D geometric computing.
// Rotors replace quaternions; bivectors replace cross products.

#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "quat.hpp"
#include <array>
#include <cmath>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// Cl(3,0) — 3D Geometric Algebra
// Basis: {1, e1, e2, e3, e12, e13, e23, e123}
//  Index: [0,  1,  2,  3,   4,   5,   6,   7]
//  Grade: [0,  1,  1,  1,   2,   2,   2,   3]
// ─────────────────────────────────────────────────────────────────────────────

// Grade masks for Cl(3,0)
static constexpr u8 GA3_GRADE[8] = {0,1,1,1,2,2,2,3};

template<typename T = f32>
struct Multivector3 {
    // e[0]=1, e[1]=e1, e[2]=e2, e[3]=e3, e[4]=e12, e[5]=e13, e[6]=e23, e[7]=e123
    std::array<T, 8> c{};

    constexpr Multivector3() { c.fill(T(0)); }
    static constexpr Multivector3 scalar(T s) { Multivector3 m; m.c[0]=s; return m; }
    static constexpr Multivector3 vector(T x,T y,T z) { Multivector3 m; m.c[1]=x;m.c[2]=y;m.c[3]=z; return m; }
    static constexpr Multivector3 bivector(T e12,T e13,T e23) { Multivector3 m; m.c[4]=e12;m.c[5]=e13;m.c[6]=e23; return m; }
    static constexpr Multivector3 pseudoscalar(T v) { Multivector3 m; m.c[7]=v; return m; }

    [[nodiscard]] constexpr T  grade0()  const { return c[0]; }
    [[nodiscard]] constexpr Vec3<T> grade1() const { return {c[1],c[2],c[3]}; }
    [[nodiscard]] constexpr Vec3<T> grade2() const { return {c[4],c[5],c[6]}; }
    [[nodiscard]] constexpr T  grade3()  const { return c[7]; }

    // Extract grade
    [[nodiscard]] constexpr Multivector3 grade(int g) const {
        Multivector3 r;
        for (int i=0;i<8;++i) if (GA3_GRADE[i]==u8(g)) r.c[u32(i)]=c[u32(i)];
        return r;
    }

    // Addition / subtraction
    [[nodiscard]] constexpr Multivector3 operator+(const Multivector3& b) const {
        Multivector3 r; for(int i=0;i<8;++i) r.c[u32(i)]=c[u32(i)]+b.c[u32(i)]; return r;
    }
    [[nodiscard]] constexpr Multivector3 operator-(const Multivector3& b) const {
        Multivector3 r; for(int i=0;i<8;++i) r.c[u32(i)]=c[u32(i)]-b.c[u32(i)]; return r;
    }
    [[nodiscard]] constexpr Multivector3 operator*(T s) const {
        Multivector3 r; for(int i=0;i<8;++i) r.c[u32(i)]=c[u32(i)]*s; return r;
    }
    [[nodiscard]] constexpr Multivector3 operator-() const {
        Multivector3 r; for(int i=0;i<8;++i) r.c[u32(i)]=-c[u32(i)]; return r;
    }

    // Reverse ~M: reverse the order of basis vectors in each blade
    // Grade 0,1: unchanged; Grade 2: flip sign; Grade 3: unchanged ((-1)^(k(k-1)/2))
    [[nodiscard]] constexpr Multivector3 reverse() const {
        Multivector3 r = *this;
        r.c[4]=-r.c[4]; r.c[5]=-r.c[5]; r.c[6]=-r.c[6]; // grade 2: negate
        return r;
    }

    // Grade involution: negate all odd grades
    [[nodiscard]] constexpr Multivector3 grade_involution() const {
        Multivector3 r = *this;
        r.c[1]=-r.c[1];r.c[2]=-r.c[2];r.c[3]=-r.c[3]; // grade 1
        r.c[7]=-r.c[7]; // grade 3
        return r;
    }

    // Clifford conjugate = grade_involution then reverse
    [[nodiscard]] constexpr Multivector3 clifford_conj() const { return grade_involution().reverse(); }

    // Squared norm: M * ~M (scalar part)
    [[nodiscard]] constexpr T norm_sq() const { return (*this * reverse()).c[0]; }
    [[nodiscard]] T norm() const { return std::sqrt(std::abs(norm_sq())); }

    // Dual: M* = M · I^{-1}, pseudoscalar I = e123, I^{-1} = e123 in Cl(3,0)
    [[nodiscard]] constexpr Multivector3 dual() const {
        // Right complement: multiply by e123 on right
        // e1*e123 = e23, e2*e123 = -e13, e3*e123 = e12, etc.
        Multivector3 r;
        r.c[0] =  c[7];  r.c[7] =  c[0];
        r.c[1] =  c[6];  r.c[6] =  c[1];
        r.c[2] = -c[5];  r.c[5] = -c[2];
        r.c[3] =  c[4];  r.c[4] =  c[3];
        return r;
    }

    // Geometric product (full 8×8 table for Cl(3,0))
    [[nodiscard]] constexpr Multivector3 operator*(const Multivector3& b) const {
        const auto& a = c; const auto& B = b.c;
        Multivector3 r;
        auto& o = r.c;
        // Generated from Cl(3,0) multiplication table
        o[0] = a[0]*B[0]+a[1]*B[1]+a[2]*B[2]+a[3]*B[3]-a[4]*B[4]-a[5]*B[5]-a[6]*B[6]-a[7]*B[7];
        o[1] = a[0]*B[1]+a[1]*B[0]-a[2]*B[4]+a[3]*B[5]+a[4]*B[2]-a[5]*B[3]+a[6]*B[7]+a[7]*B[6];
        o[2] = a[0]*B[2]+a[1]*B[4]+a[2]*B[0]-a[3]*B[6]-a[4]*B[1]+a[5]*B[7]+a[6]*B[3]-a[7]*B[5];
        o[3] = a[0]*B[3]-a[1]*B[5]+a[2]*B[6]+a[3]*B[0]-a[4]*B[7]-a[5]*B[1]-a[6]*B[2]+a[7]*B[4];
        o[4] = a[0]*B[4]+a[1]*B[2]-a[2]*B[1]+a[3]*B[7]+a[4]*B[0]-a[5]*B[6]+a[6]*B[5]-a[7]*B[3];
        o[5] = a[0]*B[5]-a[1]*B[3]+a[2]*B[7]+a[3]*B[1]+a[4]*B[6]+a[5]*B[0]-a[6]*B[4]+a[7]*B[2];
        o[6] = a[0]*B[6]+a[1]*B[7]+a[2]*B[3]-a[3]*B[2]-a[4]*B[5]+a[5]*B[4]+a[6]*B[0]-a[7]*B[1];
        o[7] = a[0]*B[7]+a[1]*B[6]-a[2]*B[5]+a[3]*B[4]+a[4]*B[3]-a[5]*B[2]+a[6]*B[1]+a[7]*B[0];
        return r;
    }

    // Outer (wedge) product: grade(a∧b) = grade(a)+grade(b), zero if blades share a factor
    [[nodiscard]] constexpr Multivector3 outer(const Multivector3& b) const {
        // outer = geometric - inner for vectors; full version via grade extraction
        const auto& a = c; const auto& B = b.c;
        Multivector3 r;
        r.c[0] = a[0]*B[0];
        r.c[1] = a[0]*B[1]+a[1]*B[0];
        r.c[2] = a[0]*B[2]+a[2]*B[0];
        r.c[3] = a[0]*B[3]+a[3]*B[0];
        r.c[4] = a[0]*B[4]+a[1]*B[2]-a[2]*B[1]+a[4]*B[0];
        r.c[5] = a[0]*B[5]-a[1]*B[3]+a[3]*B[1]+a[5]*B[0];
        r.c[6] = a[0]*B[6]+a[2]*B[3]-a[3]*B[2]+a[6]*B[0];
        r.c[7] = a[0]*B[7]+a[1]*B[6]-a[2]*B[5]+a[3]*B[4]+a[4]*B[3]-a[5]*B[2]+a[6]*B[1]+a[7]*B[0];
        return r;
    }

    // Inner (left contraction) product
    [[nodiscard]] constexpr Multivector3 inner(const Multivector3& b) const {
        // lc(A,B): grade(lc(Ar,Bs)) = s-r if s>=r, else 0
        return (*this * b + b.reverse() * reverse()).grade_involution() * T(0.5);
        // Simpler: use geometric - outer for vectors
    }
};

// ─── Rotor in Cl(3,0) — replaces quaternions ─────────────────────────────────
// R = cos(θ/2) - sin(θ/2)·B  where B is a unit bivector
template<typename T = f32>
struct Rotor3 {
    T s{1};           // scalar part
    T b12{0}, b13{0}, b23{0}; // bivector components e12, e13, e23

    constexpr Rotor3() = default;
    constexpr Rotor3(T s, T b12, T b13, T b23) : s(s), b12(b12), b13(b13), b23(b23) {}

    // Construct from axis-angle
    static Rotor3 from_axis_angle(Vec3<T> axis, T angle_rad) {
        axis = normalize(axis);
        T half = angle_rad * T(0.5);
        T sh = std::sin(half), ch = std::cos(half);
        // B = x·e23 - y·e13 + z·e12 (bivector dual to axis)
        return {ch, -sh*axis.z, sh*axis.y, -sh*axis.x};
    }

    // Construct from two unit vectors: R = (1 + ba) / |1 + ba|
    static Rotor3 from_vectors(Vec3<T> from, Vec3<T> to) {
        from = normalize(from); to = normalize(to);
        T half_cos = T(1) + dot(from, to);
        if (std::abs(half_cos) < T(1e-6)) {
            // 180° rotation — find perpendicular
            Vec3<T> perp = std::abs(from.x) < T(0.9) ? Vec3<T>{T(1),T(0),T(0)} : Vec3<T>{T(0),T(1),T(0)};
            Vec3<T> ax = normalize(cross(from, perp));
            return from_axis_angle(ax, T(3.14159265358979));
        }
        T n = std::sqrt(T(2) * half_cos);
        Vec3<T> cv = cross(from, to);
        return {half_cos / n, -cv.z/n, cv.y/n, -cv.x/n};
    }

    // Geometric product of two rotors
    [[nodiscard]] constexpr Rotor3 operator*(const Rotor3& o) const {
        return {
            s*o.s  - b12*o.b12 - b13*o.b13 - b23*o.b23,
            s*o.b12 + b12*o.s  + b23*o.b13 - b13*o.b23,
            s*o.b13 + b13*o.s  - b23*o.b12 + b12*o.b23,
            s*o.b23 + b23*o.s  + b13*o.b12 - b12*o.b13
        };
    }

    // Reverse (conjugate) of a rotor: ~R = s - B
    [[nodiscard]] constexpr Rotor3 reverse() const { return {s, -b12, -b13, -b23}; }

    // Norm squared (should be 1 for unit rotor)
    [[nodiscard]] constexpr T norm_sq() const { return s*s + b12*b12 + b13*b13 + b23*b23; }
    [[nodiscard]] T norm()       const { return std::sqrt(norm_sq()); }
    [[nodiscard]] Rotor3 normalized() const { T n = norm(); return {s/n, b12/n, b13/n, b23/n}; }

    // Apply rotor to vector: v' = R v ~R
    [[nodiscard]] Vec3<T> rotate(const Vec3<T>& v) const {
        // Sandwich product with grade-1 extraction
        // Optimised form (same as quaternion rotation)
        T tx = T(2) * (b13*v.z - b23*v.y);
        T ty = T(2) * (b23*v.x - b12*v.z);
        T tz = T(2) * (b12*v.y - b13*v.x);
        return {
            v.x + s*tx + b23*tz - b13*ty, // ...simplified expansion
            v.y + s*ty + b12*tx - b23*tz, // this is equivalent to qvq*
            v.z + s*tz + b13*ty - b12*tx
        };
    }

    // Convert to/from quaternion (structural equivalence)
    [[nodiscard]] Quat<T> to_quat() const { return Quat<T>{b23, -b13, b12, s}; }
    static Rotor3 from_quat(const Quat<T>& q) { return {q.w, q.z, -q.y, q.x}; }

    // Exponential: exp(B/2) where B is a bivector
    [[nodiscard]] static Rotor3 exp_bivector(T b12, T b13, T b23) {
        T half_angle = std::sqrt(b12*b12 + b13*b13 + b23*b23);
        if (half_angle < T(1e-7)) return {T(1), b12, b13, b23};
        T s = std::sin(half_angle) / half_angle;
        return {std::cos(half_angle), s*b12, s*b13, s*b23};
    }

    // Logarithm: log(R) = bivector = atan2(|B|, s) * B/|B|
    [[nodiscard]] std::tuple<T,T,T> log_bivector() const {
        T bv = std::sqrt(b12*b12 + b13*b13 + b23*b23);
        if (bv < T(1e-7)) return {T(0), T(0), T(0)};
        T t = std::atan2(bv, s) / bv;
        return {t*b12, t*b13, t*b23};
    }

    // SLERP
    [[nodiscard]] static Rotor3 slerp(const Rotor3& a, const Rotor3& b, T t) {
        // a * exp(t * log(~a * b))
        Rotor3 rel = a.reverse() * b;
        auto [lb12, lb13, lb23] = rel.log_bivector();
        Rotor3 dt = exp_bivector(lb12*t, lb13*t, lb23*t);
        return (a * dt).normalized();
    }
};

// ─── Bivector in 3D ───────────────────────────────────────────────────────────
template<typename T = f32>
struct Bivector3 {
    T e12{0}, e13{0}, e23{0};

    constexpr Bivector3() = default;
    constexpr Bivector3(T e12, T e13, T e23) : e12(e12), e13(e13), e23(e23) {}

    // Bivector from two vectors: a ∧ b
    static constexpr Bivector3 outer(const Vec3<T>& a, const Vec3<T>& b) {
        return {a.x*b.y - a.y*b.x, a.x*b.z - a.z*b.x, a.y*b.z - a.z*b.y};
    }

    // Dual vector (hodge dual in 3D): *B = cross product equivalent
    [[nodiscard]] constexpr Vec3<T> dual() const { return {e23, -e13, e12}; }

    // Norm
    [[nodiscard]] constexpr T norm_sq() const { return e12*e12 + e13*e13 + e23*e23; }
    [[nodiscard]] T norm() const { return std::sqrt(norm_sq()); }

    [[nodiscard]] constexpr Bivector3 operator+(const Bivector3& b) const { return {e12+b.e12, e13+b.e13, e23+b.e23}; }
    [[nodiscard]] constexpr Bivector3 operator*(T s) const { return {e12*s, e13*s, e23*s}; }
    [[nodiscard]] constexpr Bivector3 operator-() const { return {-e12,-e13,-e23}; }
};

// ─── Point, Line, Plane in PGA (Projective Geometric Algebra Cl(3,0,1)) ──────
// Just the construction helpers; full PGA is a dedicated module
template<typename T = f32>
struct PGAPoint {
    T e032{0}, e013{0}, e021{0}, e123{1}; // (x:y:z:1) in homogeneous
    static PGAPoint from_xyz(T x, T y, T z) { return {x, y, z, T(1)}; }
};

// Convenience aliases
using MV3f    = Multivector3<f32>;
using Rotor3f = Rotor3<f32>;
using Biv3f   = Bivector3<f32>;

// ─────────────────────────────────────────────────────────────────────────────
// Free-function GA utilities
// ─────────────────────────────────────────────────────────────────────────────

// Outer product of two vectors (shorthand)
template<typename T>
[[nodiscard]] constexpr Bivector3<T> wedge(const Vec3<T>& a, const Vec3<T>& b) {
    return Bivector3<T>::outer(a, b);
}

// Scalar part of geometric product of two vectors = dot product
template<typename T>
[[nodiscard]] constexpr T ga_dot(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Regressive product (meet) of two bivectors in 3D = dual of outer of duals
template<typename T>
[[nodiscard]] constexpr Vec3<T> regressive(const Bivector3<T>& a, const Bivector3<T>& b) {
    return cross(a.dual(), b.dual());
}

} // namespace qs
