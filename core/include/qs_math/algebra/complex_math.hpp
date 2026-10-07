#pragma once
// complex_math.hpp — Complex number mathematics: arithmetic, polar form,
//                    complex functions, roots of unity, DFT math, conformal maps
#include "scalar_types.hpp"
#include "constants.hpp"
#include "trig.hpp"
#include "scalar_math.hpp"
#include "vec2.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Complex<T>
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Complex {
    T re{T(0)}, im{T(0)};

    constexpr Complex() noexcept = default;
    constexpr Complex(T re, T im=T(0)) noexcept : re(re), im(im) {}
    constexpr explicit Complex(Vec2<T> v) noexcept : re(v.x), im(v.y) {}

    // Arithmetic
    constexpr Complex operator+(Complex o) const noexcept { return {re+o.re, im+o.im}; }
    constexpr Complex operator-(Complex o) const noexcept { return {re-o.re, im-o.im}; }
    constexpr Complex operator*(Complex o) const noexcept {
        return {re*o.re-im*o.im, re*o.im+im*o.re};
    }
    constexpr Complex operator/(Complex o) const noexcept {
        T d=o.re*o.re+o.im*o.im;
        return {(re*o.re+im*o.im)/d, (im*o.re-re*o.im)/d};
    }
    constexpr Complex operator+(T s) const noexcept { return {re+s, im}; }
    constexpr Complex operator*(T s) const noexcept { return {re*s, im*s}; }
    constexpr Complex operator-() const noexcept { return {-re,-im}; }
    constexpr Complex& operator+=(Complex o) noexcept { re+=o.re; im+=o.im; return *this; }
    constexpr Complex& operator*=(Complex o) noexcept { *this=(*this)*o; return *this; }
    constexpr bool operator==(Complex o) const noexcept { return re==o.re&&im==o.im; }

    // Properties
    [[nodiscard]] constexpr T  norm()    const noexcept { return re*re+im*im; }
    [[nodiscard]] inline     T abs()     const noexcept { return sqrt(norm()); }
    [[nodiscard]] inline     T arg()     const noexcept { return atan2(im,re); }
    [[nodiscard]] constexpr Complex conj() const noexcept { return {re,-im}; }
    [[nodiscard]] inline     Complex inv()  const noexcept { T n=norm(); return {re/n,-im/n}; }
    [[nodiscard]] inline     Complex normalized() const noexcept { T a=abs(); return {re/a,im/a}; }
    [[nodiscard]] constexpr Vec2<T> to_vec2() const noexcept { return {re,im}; }

    // Polar form
    [[nodiscard]] inline T modulus() const noexcept { return abs(); }
    [[nodiscard]] inline T argument()const noexcept { return arg(); }

    static inline Complex from_polar(T r, T theta) noexcept {
        return {r*cos(theta), r*sin(theta)};
    }

    // Powers
    [[nodiscard]] inline Complex pow(T n) const noexcept {
        T r=abs(), theta=arg();
        return from_polar(::qs::math::pow(r,n), theta*n);
    }
    [[nodiscard]] inline Complex pow(Complex n) const noexcept {
        // z^w = exp(w*log(z))
        T r=abs(), theta=arg();
        T log_r=log(r);
        T re_exp=n.re*log_r-n.im*theta;
        T im_exp=n.re*theta+n.im*log_r;
        return from_polar(exp(re_exp),im_exp);
    }
    [[nodiscard]] inline Complex sqrt() const noexcept {
        T r=abs();
        return {::qs::math::sqrt((r+re)*T(0.5)), (im>=T(0)?T(1):T(-1))*::qs::math::sqrt((r-re)*T(0.5))};
    }

    // Complex exponential / log
    [[nodiscard]] inline Complex exp() const noexcept {
        return from_polar(::qs::math::exp(re),im);
    }
    [[nodiscard]] inline Complex log() const noexcept {
        return {::qs::math::log(abs()), arg()};
    }
    [[nodiscard]] inline Complex log(Complex base) const noexcept {
        return log()/base.log();
    }

    // Trig
    [[nodiscard]] inline Complex sin() const noexcept {
        return {::qs::math::sin(re)*cosh(im), ::qs::math::cos(re)*sinh(im)};
    }
    [[nodiscard]] inline Complex cos() const noexcept {
        return {::qs::math::cos(re)*cosh(im), -::qs::math::sin(re)*sinh(im)};
    }
    [[nodiscard]] inline Complex tan() const noexcept { return sin()/cos(); }
    [[nodiscard]] inline Complex sinh() const noexcept {
        return {::qs::math::sinh(re)*::qs::math::cos(im), ::qs::math::cosh(re)*::qs::math::sin(im)};
    }
    [[nodiscard]] inline Complex cosh() const noexcept {
        return {::qs::math::cosh(re)*::qs::math::cos(im), ::qs::math::sinh(re)*::qs::math::sin(im)};
    }

    static constexpr Complex zero()     noexcept { return {T(0),T(0)}; }
    static constexpr Complex one()      noexcept { return {T(1),T(0)}; }
    static constexpr Complex i()        noexcept { return {T(0),T(1)}; }
    static constexpr Complex neg_one()  noexcept { return {T(-1),T(0)}; }
    static constexpr Complex neg_i()    noexcept { return {T(0),T(-1)}; }
};

template<FloatScalar T> constexpr Complex<T> operator+(T s,Complex<T> c) noexcept { return {s+c.re,c.im}; }
template<FloatScalar T> constexpr Complex<T> operator*(T s,Complex<T> c) noexcept { return {s*c.re,s*c.im}; }
template<FloatScalar T> [[nodiscard]] constexpr T  abs(Complex<T> c) noexcept { return c.abs(); }
template<FloatScalar T> [[nodiscard]] constexpr Complex<T> conj(Complex<T> c) noexcept { return c.conj(); }
template<FloatScalar T> [[nodiscard]] constexpr T  norm(Complex<T> c) noexcept { return c.norm(); }
template<FloatScalar T> [[nodiscard]] constexpr bool nearly_equal(Complex<T> a,Complex<T> b,T eps=EPS_F32) noexcept {
    return nearly_equal(a.re,b.re,eps)&&nearly_equal(a.im,b.im,eps);
}

using Complexf = Complex<f32>;
using Complexd = Complex<f64>;

// ---------------------------------------------------------------------------
// Roots of unity
// ---------------------------------------------------------------------------
template<FloatScalar T>
inline void roots_of_unity(i32 n, Complex<T>* out) noexcept {
    for(i32 k=0;k<n;++k)
        out[k]=Complex<T>::from_polar(T(1), T(TWO_PI_D)*k/n);
}

// nth root
template<FloatScalar T>
inline void nth_roots(Complex<T> z, i32 n, Complex<T>* out) noexcept {
    T r=pow(z.abs(),T(1)/n);
    T theta=z.arg();
    for(i32 k=0;k<n;++k)
        out[k]=Complex<T>::from_polar(r,(theta+T(TWO_PI_D)*k)/n);
}

// ---------------------------------------------------------------------------
// Möbius / conformal transformations
// ---------------------------------------------------------------------------
// Möbius transform: f(z) = (az+b)/(cz+d)
template<FloatScalar T>
[[nodiscard]] inline Complex<T> mobius(Complex<T> z,
    Complex<T> a, Complex<T> b,
    Complex<T> c, Complex<T> d) noexcept
{
    Complex<T> num=a*z+b, den=c*z+d;
    if(den.norm()<T(EPS_F32)) return Complex<T>{T(1e30)};
    return num/den;
}

// Joukowski transform (airfoil)
template<FloatScalar T>
[[nodiscard]] inline Complex<T> joukowski(Complex<T> z, T c=T(1)) noexcept {
    Complex<T> cc{c,T(0)};
    return z+cc*cc/z;
}

// Schwarz-Christoffel (stub — full implementation requires numerical integration)
template<FloatScalar T>
[[nodiscard]] inline Complex<T> schwarz_christoffel_step(Complex<T> z, Complex<T> dz) noexcept {
    return z+dz;
}

// ---------------------------------------------------------------------------
// Complex polynomial evaluation (Horner)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] inline Complex<T> poly_eval(const Complex<T>* coeffs, i32 n, Complex<T> z) noexcept {
    Complex<T> r{T(0)};
    for(i32 i=n-1;i>=0;--i) r=r*z+coeffs[i];
    return r;
}

// ---------------------------------------------------------------------------
// Complex DFT twiddle factor
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] inline Complex<T> twiddle(i32 k, i32 n, bool inverse=false) noexcept {
    T angle=T(TWO_PI_D)*k/n*(inverse?T(1):T(-1));
    return {cos(angle),sin(angle)};
}

// Mandelbrot iteration count (returns escape iteration, -1=inside)
[[nodiscard]] inline i32 mandelbrot(Complexf c, i32 max_iter=256, f32 bailout=4.f) noexcept {
    Complexf z{0.f};
    for(i32 i=0;i<max_iter;++i){
        z=z*z+c;
        if(z.norm()>bailout) return i;
    }
    return -1;
}

// Smooth mandelbrot (continuous escape)
[[nodiscard]] inline f32 mandelbrot_smooth(Complexf c, i32 max_iter=256) noexcept {
    Complexf z{0.f};
    for(i32 i=0;i<max_iter;++i){
        z=z*z+c;
        if(z.norm()>4.f) return f32(i+1)-log2(log2(z.abs()));
    }
    return f32(max_iter);
}

// Julia set
[[nodiscard]] inline i32 julia(Complexf z, Complexf c, i32 max_iter=256) noexcept {
    for(i32 i=0;i<max_iter;++i){
        z=z*z+c;
        if(z.norm()>4.f) return i;
    }
    return -1;
}

} // namespace qs::math
