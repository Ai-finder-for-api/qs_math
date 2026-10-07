#pragma once
// qs_math/lie_algebra.hpp — sl(2), su(2), so(3) generators and structure constants (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "quat.hpp"
#include <array>
#include <cmath>

namespace qs {

// ─── Generic Lie algebra element (3D): linear combination of generators ───────
template<typename T = f32>
struct LieElement3 {
    T a{0}, b{0}, c{0}; // coefficients for 3 generators

    constexpr LieElement3() = default;
    constexpr LieElement3(T a, T b, T c) : a(a), b(b), c(c) {}

    [[nodiscard]] constexpr LieElement3 operator+(const LieElement3& o) const { return {a+o.a,b+o.b,c+o.c}; }
    [[nodiscard]] constexpr LieElement3 operator*(T s) const { return {a*s,b*s,c*s}; }
    [[nodiscard]] constexpr LieElement3 operator-() const { return {-a,-b,-c}; }
    [[nodiscard]] constexpr T norm_sq() const { return a*a+b*b+c*c; }
    [[nodiscard]] T norm() const { return std::sqrt(norm_sq()); }
};

// ─── so(3) — Lie algebra of SO(3) ─────────────────────────────────────────────
// Generators: L1, L2, L3 (skew-symmetric 3×3 matrices)
// [Li, Lj] = ε_ijk Lk
namespace so3 {
    // Generator matrices: L_k = d/dθ of rotation about axis k at θ=0
    template<typename T=f32> [[nodiscard]] constexpr Mat3<T> L1() {
        return {T(0),T(0),T(0), T(0),T(0),T(-1), T(0),T(1),T(0)};
    }
    template<typename T=f32> [[nodiscard]] constexpr Mat3<T> L2() {
        return {T(0),T(0),T(1), T(0),T(0),T(0), T(-1),T(0),T(0)};
    }
    template<typename T=f32> [[nodiscard]] constexpr Mat3<T> L3() {
        return {T(0),T(-1),T(0), T(1),T(0),T(0), T(0),T(0),T(0)};
    }

    // Structure constants: [Li, Lj] = f^k_ij * Lk
    // For so(3): f^k_ij = ε_ijk (Levi-Civita)
    template<typename T=f32>
    [[nodiscard]] constexpr T structure_constant(int i, int j, int k) {
        // ε_ijk
        if (i==0&&j==1&&k==2) return T( 1);
        if (i==1&&j==2&&k==0) return T( 1);
        if (i==2&&j==0&&k==1) return T( 1);
        if (i==0&&j==2&&k==1) return T(-1);
        if (i==2&&j==1&&k==0) return T(-1);
        if (i==1&&j==0&&k==2) return T(-1);
        return T(0);
    }

    // Lie bracket [X,Y] = XY - YX in so(3) representation
    template<typename T=f32>
    [[nodiscard]] LieElement3<T> bracket(const LieElement3<T>& X, const LieElement3<T>& Y) {
        // [X,Y]_k = ε_ijk X_i Y_j (cross product of coefficients)
        return {
            X.b*Y.c - X.c*Y.b,
            X.c*Y.a - X.a*Y.c,
            X.a*Y.b - X.b*Y.a
        };
    }

    // Exponential map: exp(θ·n̂) → rotation matrix (Rodrigues)
    template<typename T=f32>
    [[nodiscard]] Mat3<T> exp_map(const LieElement3<T>& X) {
        T angle = X.norm();
        if (angle < T(1e-7)) return Mat3<T>::identity();
        Vec3<T> axis{X.a/angle, X.b/angle, X.c/angle};
        T c = std::cos(angle), s = std::sin(angle), t = T(1)-c;
        return {
            t*axis.x*axis.x+c,       t*axis.x*axis.y-s*axis.z, t*axis.x*axis.z+s*axis.y,
            t*axis.x*axis.y+s*axis.z, t*axis.y*axis.y+c,       t*axis.y*axis.z-s*axis.x,
            t*axis.x*axis.z-s*axis.y, t*axis.y*axis.z+s*axis.x, t*axis.z*axis.z+c
        };
    }

    // Logarithm map: rotation matrix → so(3) element
    template<typename T=f32>
    [[nodiscard]] LieElement3<T> log_map(const Mat3<T>& R) {
        T trace = R.m[0][0]+R.m[1][1]+R.m[2][2];
        T cos_theta = (trace - T(1)) * T(0.5);
        cos_theta = std::clamp(cos_theta, T(-1), T(1));
        T theta = std::acos(cos_theta);
        if (std::abs(theta) < T(1e-7)) return {};
        T s = theta / (T(2) * std::sin(theta));
        return {
            s*(R.m[2][1]-R.m[1][2]),
            s*(R.m[0][2]-R.m[2][0]),
            s*(R.m[1][0]-R.m[0][1])
        };
    }

    // Adjoint representation: Ad_R(X) = R X R^T in vector form
    template<typename T=f32>
    [[nodiscard]] LieElement3<T> adjoint(const Mat3<T>& R, const LieElement3<T>& X) {
        Vec3<T> v = R * Vec3<T>{X.a, X.b, X.c};
        return {v.x, v.y, v.z};
    }

    // hat operator: vector → skew-symmetric matrix (so(3) element)
    template<typename T=f32>
    [[nodiscard]] constexpr Mat3<T> hat(const Vec3<T>& v) {
        return {T(0),-v.z,v.y, v.z,T(0),-v.x, -v.y,v.x,T(0)};
    }
    // vee operator: skew-symmetric matrix → vector
    template<typename T=f32>
    [[nodiscard]] constexpr Vec3<T> vee(const Mat3<T>& M) {
        return {M.m[2][1], M.m[0][2], M.m[1][0]};
    }
} // namespace so3

// ─── su(2) — Lie algebra of SU(2), double cover of SO(3) ─────────────────────
// Generators: σ_k/2i where σ_k are Pauli matrices (here stored as 2×2 complex reps)
namespace su2 {
    // Pauli matrix σ_1 components: [[0,1],[1,0]]
    // We represent su(2) elements as quaternion imaginary parts (isomorphic to so(3))
    template<typename T=f32>
    [[nodiscard]] constexpr LieElement3<T> bracket(const LieElement3<T>& X, const LieElement3<T>& Y) {
        return so3::bracket(X, Y); // su(2) ≅ so(3) as Lie algebras
    }

    // Exponential map: su(2) → SU(2) (unit quaternion)
    template<typename T=f32>
    [[nodiscard]] Quat<T> exp_map(const LieElement3<T>& X) {
        T half_angle = X.norm();
        if (half_angle < T(1e-7)) return Quat<T>::identity();
        T s = std::sin(half_angle) / half_angle;
        return {X.a*s, X.b*s, X.c*s, std::cos(half_angle)};
    }

    // Log map: unit quaternion → su(2)
    template<typename T=f32>
    [[nodiscard]] LieElement3<T> log_map(const Quat<T>& q) {
        T vn = std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
        if (vn < T(1e-7)) return {};
        T s = std::atan2(vn, q.w) / vn;
        return {q.x*s, q.y*s, q.z*s};
    }
} // namespace su2

// ─── sl(2,R) — Lie algebra of SL(2,R) ────────────────────────────────────────
// Generators: H = [[1,0],[0,-1]], E = [[0,1],[0,0]], F = [[0,0],[1,0]]
// [H,E]=2E, [H,F]=-2F, [E,F]=H  (sl2 triple)
namespace sl2 {
    // Coefficients (h,e,f) for aH + bE + cF
    template<typename T=f32>
    struct Element { T h{0}, e{0}, f{0}; };

    template<typename T=f32>
    [[nodiscard]] constexpr Element<T> bracket(const Element<T>& X, const Element<T>& Y) {
        // [aH+bE+cF, dH+eE+fF]
        // = a*d[H,H]+a*e[H,E]+a*f[H,F]+b*d[E,H]+b*e[E,E]+b*f[E,F]+c*d[F,H]+c*e[F,E]+c*f[F,F]
        // = 2ae*E - 2af*F - 2bd*E + bf*H + 2cd*F - ce*H
        return {
            X.e*Y.f - X.f*Y.e,             // H coeff
            T(2)*(X.h*Y.e - X.e*Y.h),       // E coeff
            T(2)*(X.f*Y.h - X.h*Y.f)        // F coeff
        };
    }

    // Killing form B(X,Y) = tr(ad_X ∘ ad_Y)
    // For sl(2): B(H,H)=8, B(E,F)=B(F,E)=4, others 0
    template<typename T=f32>
    [[nodiscard]] constexpr T killing_form(const Element<T>& X, const Element<T>& Y) {
        return T(8)*X.h*Y.h + T(4)*(X.e*Y.f + X.f*Y.e);
    }
} // namespace sl2

// ─── SE(3) — Lie algebra se(3) of rigid body motions ─────────────────────────
namespace se3 {
    // se(3) element: (ω, v) — angular velocity + linear velocity
    template<typename T=f32>
    struct Twist { Vec3<T> omega, v; };

    template<typename T=f32>
    [[nodiscard]] constexpr Twist<T> bracket(const Twist<T>& X, const Twist<T>& Y) {
        return {
            cross(X.omega, Y.omega),
            cross(X.omega, Y.v) - cross(Y.omega, X.v)
        };
    }

    // 4×4 hat matrix for se(3) element
    template<typename T=f32>
    struct SE3Hat {
        T m[4][4]{};
    };
    template<typename T=f32>
    [[nodiscard]] SE3Hat<T> hat(const Twist<T>& t) {
        SE3Hat<T> M{};
        M.m[0][1]=-t.omega.z; M.m[0][2]= t.omega.y; M.m[0][3]=t.v.x;
        M.m[1][0]= t.omega.z; M.m[1][2]=-t.omega.x; M.m[1][3]=t.v.y;
        M.m[2][0]=-t.omega.y; M.m[2][1]= t.omega.x; M.m[2][3]=t.v.z;
        return M;
    }
} // namespace se3

// Convenience aliases
using LieElem3f = LieElement3<f32>;
using Twistf    = se3::Twist<f32>;

} // namespace qs
