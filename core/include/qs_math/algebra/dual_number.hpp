#pragma once
// dual_number.hpp — Dual numbers for automatic differentiation,
//                   dual quaternions review, dual number arithmetic,
//                   forward-mode AD, hyperdual numbers for 2nd derivatives
#include "scalar_types.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "constants.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Dual<T> — dual number: a + bε where ε²=0
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Dual {
    T real{T(0)}, dual_{T(0)};

    constexpr Dual() noexcept = default;
    constexpr Dual(T real, T dual_=T(0)) noexcept : real(real), dual_(dual_) {}

    // Arithmetic
    constexpr Dual operator+(Dual o) const noexcept { return {real+o.real, dual_+o.dual_}; }
    constexpr Dual operator-(Dual o) const noexcept { return {real-o.real, dual_-o.dual_}; }
    constexpr Dual operator*(Dual o) const noexcept {
        return {real*o.real, real*o.dual_+dual_*o.real}; // product rule
    }
    constexpr Dual operator/(Dual o) const noexcept {
        return {real/o.real, (dual_*o.real-real*o.dual_)/(o.real*o.real)};
    }
    constexpr Dual operator*(T s) const noexcept { return {real*s, dual_*s}; }
    constexpr Dual operator+(T s) const noexcept { return {real+s, dual_}; }
    constexpr Dual operator-()    const noexcept { return {-real, -dual_}; }
    constexpr bool operator==(Dual o) const noexcept { return real==o.real&&dual_==o.dual_; }

    // Derivative extraction
    [[nodiscard]] constexpr T value()      const noexcept { return real; }
    [[nodiscard]] constexpr T derivative() const noexcept { return dual_; }

    // Math functions — chain rule applied
    [[nodiscard]] inline Dual sqrt() const noexcept {
        T r=::qs::math::sqrt(real);
        return {r, dual_/(T(2)*r)};
    }
    [[nodiscard]] inline Dual exp() const noexcept {
        T e=::qs::math::exp(real);
        return {e, dual_*e};
    }
    [[nodiscard]] inline Dual log() const noexcept {
        return {::qs::math::log(real), dual_/real};
    }
    [[nodiscard]] inline Dual pow(T n) const noexcept {
        T r=::qs::math::pow(real,n);
        return {r, dual_*n*::qs::math::pow(real,n-T(1))};
    }
    [[nodiscard]] inline Dual sin() const noexcept {
        return {::qs::math::sin(real), dual_*::qs::math::cos(real)};
    }
    [[nodiscard]] inline Dual cos() const noexcept {
        return {::qs::math::cos(real), -dual_*::qs::math::sin(real)};
    }
    [[nodiscard]] inline Dual tan() const noexcept {
        T c=::qs::math::cos(real);
        return {::qs::math::tan(real), dual_/(c*c)};
    }
    [[nodiscard]] inline Dual asin() const noexcept {
        return {::qs::math::asin(real), dual_/::qs::math::sqrt(T(1)-real*real)};
    }
    [[nodiscard]] inline Dual acos() const noexcept {
        return {::qs::math::acos(real), -dual_/::qs::math::sqrt(T(1)-real*real)};
    }
    [[nodiscard]] inline Dual atan() const noexcept {
        return {::qs::math::atan(real), dual_/(T(1)+real*real)};
    }
    [[nodiscard]] inline Dual sinh() const noexcept {
        return {std::sinh(real), dual_*std::cosh(real)};
    }
    [[nodiscard]] inline Dual cosh() const noexcept {
        return {std::cosh(real), dual_*std::sinh(real)};
    }
    [[nodiscard]] constexpr Dual abs() const noexcept {
        return {::qs::math::abs(real), real>=T(0)?dual_:-dual_};
    }
    [[nodiscard]] constexpr Dual max(Dual o) const noexcept {
        return real>=o.real?*this:o;
    }
    [[nodiscard]] constexpr Dual min(Dual o) const noexcept {
        return real<=o.real?*this:o;
    }

    static constexpr Dual variable(T x) noexcept { return {x, T(1)}; }
    static constexpr Dual constant(T c) noexcept { return {c, T(0)}; }
};

template<FloatScalar T> constexpr Dual<T> operator+(T s,Dual<T> d) noexcept { return {s+d.real,d.dual_}; }
template<FloatScalar T> constexpr Dual<T> operator*(T s,Dual<T> d) noexcept { return {s*d.real,s*d.dual_}; }

using Dualf = Dual<f32>;
using Duald = Dual<f64>;

// ---------------------------------------------------------------------------
// Automatic differentiation helpers
// ---------------------------------------------------------------------------
// Differentiate f(x) at x=x0 using dual numbers
template<FloatScalar T, typename Fn>
[[nodiscard]] inline T auto_diff(Fn&& f, T x0) noexcept {
    Dual<T> result=f(Dual<T>::variable(x0));
    return result.derivative();
}

// Gradient of f:R^n -> R using dual numbers (n evaluations)
template<FloatScalar T, typename Fn>
inline void auto_grad(Fn&& f, const T* x, T* grad, i32 n) noexcept {
    // n separate forward passes
    static constexpr i32 MAX_N=64;
    Dual<T> xd[MAX_N];
    for(i32 i=0;i<n&&i<MAX_N;++i) xd[i]=Dual<T>::constant(x[i]);
    for(i32 i=0;i<n&&i<MAX_N;++i){
        xd[i]=Dual<T>::variable(x[i]);
        Dual<T> r=f(xd);
        grad[i]=r.derivative();
        xd[i]=Dual<T>::constant(x[i]);
    }
}

// Jacobian column by column
template<FloatScalar T, typename Fn>
inline void auto_jacobian(Fn&& f, const T* x, T* jac, i32 n_in, i32 n_out) noexcept {
    static constexpr i32 MAX_N=64;
    Dual<T> xd[MAX_N];
    for(i32 i=0;i<n_in&&i<MAX_N;++i) xd[i]=Dual<T>::constant(x[i]);
    Dual<T> out[MAX_N];
    for(i32 j=0;j<n_in&&j<MAX_N;++j){
        xd[j]=Dual<T>::variable(x[j]);
        f(xd,out,n_out);
        for(i32 i=0;i<n_out&&i<MAX_N;++i) jac[i*n_in+j]=out[i].derivative();
        xd[j]=Dual<T>::constant(x[j]);
    }
}

// ---------------------------------------------------------------------------
// Hyperdual number — a + bε₁ + cε₂ + dε₁ε₂, captures 2nd derivatives
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct HyperDual {
    T f0{T(0)}, f1{T(0)}, f2{T(0)}, f12{T(0)};

    constexpr HyperDual() noexcept = default;
    constexpr HyperDual(T f0,T f1=T(0),T f2=T(0),T f12=T(0)) noexcept
        : f0(f0),f1(f1),f2(f2),f12(f12) {}

    constexpr HyperDual operator+(HyperDual o) const noexcept {
        return {f0+o.f0,f1+o.f1,f2+o.f2,f12+o.f12};
    }
    constexpr HyperDual operator*(HyperDual o) const noexcept {
        return {f0*o.f0, f0*o.f1+f1*o.f0, f0*o.f2+f2*o.f0,
                f0*o.f12+f1*o.f2+f2*o.f1+f12*o.f0};
    }
    constexpr HyperDual operator*(T s) const noexcept { return {f0*s,f1*s,f2*s,f12*s}; }
    constexpr HyperDual operator+(T s) const noexcept { return {f0+s,f1,f2,f12}; }
    constexpr HyperDual operator-()    const noexcept { return {-f0,-f1,-f2,-f12}; }

    [[nodiscard]] constexpr T value()       const noexcept { return f0; }
    [[nodiscard]] constexpr T grad1()       const noexcept { return f1; }
    [[nodiscard]] constexpr T grad2()       const noexcept { return f2; }
    [[nodiscard]] constexpr T hessian()     const noexcept { return f12; }

    [[nodiscard]] inline HyperDual sin() const noexcept {
        T s=::qs::math::sin(f0), c=::qs::math::cos(f0);
        return {s, c*f1, c*f2, c*f12-s*f1*f2};
    }
    [[nodiscard]] inline HyperDual cos() const noexcept {
        T s=::qs::math::sin(f0), c=::qs::math::cos(f0);
        return {c, -s*f1, -s*f2, -s*f12-c*f1*f2};
    }
    [[nodiscard]] inline HyperDual exp() const noexcept {
        T e=::qs::math::exp(f0);
        return {e, e*f1, e*f2, e*(f12+f1*f2)};
    }
    [[nodiscard]] inline HyperDual log() const noexcept {
        return {::qs::math::log(f0), f1/f0, f2/f0, f12/f0-f1*f2/(f0*f0)};
    }
    [[nodiscard]] inline HyperDual sqrt() const noexcept {
        T r=::qs::math::sqrt(f0);
        return {r, f1/(T(2)*r), f2/(T(2)*r), f12/(T(2)*r)-f1*f2/(T(4)*f0*r)};
    }

    static constexpr HyperDual variable1(T x) noexcept { return {x,T(1),T(0),T(0)}; }
    static constexpr HyperDual variable2(T x) noexcept { return {x,T(0),T(1),T(0)}; }
    static constexpr HyperDual constant(T c)  noexcept { return {c,T(0),T(0),T(0)}; }
};

// 2nd derivative via hyperdual
template<FloatScalar T, typename Fn>
[[nodiscard]] inline T auto_diff2(Fn&& f, T x0) noexcept {
    HyperDual<T> xd{x0,T(1),T(1),T(0)};
    return f(xd).hessian();
}

} // namespace qs::math
