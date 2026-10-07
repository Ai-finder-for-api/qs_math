#pragma once
// qs_math/vec2.hpp — 2D vector, all operations, constexpr throughout
#include "scalar_math.hpp"
#include "trig.hpp"
#include <compare>

namespace qs::math {

template<MathElement T>
struct Vec2 {
    T x{T(0)}, y{T(0)};

    // Construction
    constexpr Vec2() noexcept = default;
    constexpr Vec2(T x, T y)  noexcept : x(x), y(y) {}
    constexpr explicit Vec2(T s) noexcept : x(s), y(s) {}
    template<MathElement U> constexpr explicit Vec2(Vec2<U> v) noexcept : x(T(v.x)), y(T(v.y)) {}

    // Array access
    [[nodiscard]] constexpr T  operator[](i32 i) const noexcept { return (&x)[i]; }
    [[nodiscard]] constexpr T& operator[](i32 i)       noexcept { return (&x)[i]; }

    // Arithmetic — vector
    constexpr Vec2  operator+(Vec2 o) const noexcept { return {x+o.x, y+o.y}; }
    constexpr Vec2  operator-(Vec2 o) const noexcept { return {x-o.x, y-o.y}; }
    constexpr Vec2  operator*(Vec2 o) const noexcept { return {x*o.x, y*o.y}; }
    constexpr Vec2  operator/(Vec2 o) const noexcept { return {x/o.x, y/o.y}; }
    constexpr Vec2  operator-()       const noexcept { return {-x, -y}; }
    constexpr Vec2& operator+=(Vec2 o) noexcept { x+=o.x; y+=o.y; return *this; }
    constexpr Vec2& operator-=(Vec2 o) noexcept { x-=o.x; y-=o.y; return *this; }
    constexpr Vec2& operator*=(Vec2 o) noexcept { x*=o.x; y*=o.y; return *this; }
    constexpr Vec2& operator/=(Vec2 o) noexcept { x/=o.x; y/=o.y; return *this; }

    // Arithmetic — scalar
    constexpr Vec2  operator+(T s) const noexcept { return {x+s, y+s}; }
    constexpr Vec2  operator-(T s) const noexcept { return {x-s, y-s}; }
    constexpr Vec2  operator*(T s) const noexcept { return {x*s, y*s}; }
    constexpr Vec2  operator/(T s) const noexcept { return {x/s, y/s}; }
    constexpr Vec2& operator+=(T s) noexcept { x+=s; y+=s; return *this; }
    constexpr Vec2& operator-=(T s) noexcept { x-=s; y-=s; return *this; }
    constexpr Vec2& operator*=(T s) noexcept { x*=s; y*=s; return *this; }
    constexpr Vec2& operator/=(T s) noexcept { x/=s; y/=s; return *this; }

    // Comparison
    constexpr bool operator==(Vec2 o) const noexcept { return x==o.x && y==o.y; }
    constexpr bool operator!=(Vec2 o) const noexcept { return !(*this==o); }

    // Length
    [[nodiscard]] constexpr T length_sq()  const noexcept { return x*x + y*y; }
    [[nodiscard]] inline     T length()    const noexcept { return sqrt(length_sq()); }
    [[nodiscard]] inline     T inv_length()const noexcept { return T(1)/length(); }

    // Normalize
    [[nodiscard]] inline Vec2 normalized() const noexcept {
        T len = length();
        if (len < T(EPS_LENGTH)) return *this;
        return {x/len, y/len};
    }
    [[nodiscard]] inline bool is_normalized(T eps=T(EPS_NORMAL)) const noexcept {
        return abs(length_sq()-T(1)) <= eps;
    }

    // Dot
    [[nodiscard]] constexpr T dot(Vec2 o) const noexcept { return x*o.x + y*o.y; }

    // Cross (returns scalar — the z-component of the 3D cross)
    [[nodiscard]] constexpr T cross(Vec2 o) const noexcept { return x*o.y - y*o.x; }

    // Perpendicular (rotate 90 deg CCW)
    [[nodiscard]] constexpr Vec2 perp() const noexcept { return {-y, x}; }

    // Distance
    [[nodiscard]] inline T distance(Vec2 o)    const noexcept { return (*this-o).length(); }
    [[nodiscard]] constexpr T distance_sq(Vec2 o) const noexcept { return (*this-o).length_sq(); }

    // Projection of this onto other
    [[nodiscard]] constexpr Vec2 project_onto(Vec2 o) const noexcept {
        T d = o.dot(o);
        if (d < T(EPS_LENGTH)) return Vec2{T(0)};
        return o * (dot(o)/d);
    }

    // Reflection: reflect this around normal n (n must be unit length)
    [[nodiscard]] constexpr Vec2 reflect(Vec2 n) const noexcept {
        return *this - n * (T(2)*dot(n));
    }

    // Refraction
    [[nodiscard]] inline Vec2 refract(Vec2 n, T eta) const noexcept {
        T d = dot(n);
        T k = T(1) - eta*eta*(T(1)-d*d);
        if (k < T(0)) return Vec2{T(0)};
        return *this*eta - n*(eta*d + sqrt(k));
    }

    // Min / Max component-wise
    [[nodiscard]] constexpr Vec2 min(Vec2 o) const noexcept { return {qs::math::min(x,o.x), qs::math::min(y,o.y)}; }
    [[nodiscard]] constexpr Vec2 max(Vec2 o) const noexcept { return {qs::math::max(x,o.x), qs::math::max(y,o.y)}; }
    [[nodiscard]] constexpr Vec2 clamp(Vec2 lo, Vec2 hi) const noexcept { return min(hi).max(lo); }
    [[nodiscard]] constexpr Vec2 abs()  const noexcept { return {qs::math::abs(x), qs::math::abs(y)}; }
    [[nodiscard]] constexpr T min_component() const noexcept { return qs::math::min(x,y); }
    [[nodiscard]] constexpr T max_component() const noexcept { return qs::math::max(x,y); }

    // Swizzles (common)
    [[nodiscard]] constexpr Vec2 xx() const noexcept { return {x,x}; }
    [[nodiscard]] constexpr Vec2 xy() const noexcept { return {x,y}; }
    [[nodiscard]] constexpr Vec2 yx() const noexcept { return {y,x}; }
    [[nodiscard]] constexpr Vec2 yy() const noexcept { return {y,y}; }

    // Angle of this vector from +X axis
    [[nodiscard]] inline T angle() const noexcept { return atan2(y,x); }

    // Rotate by angle (radians)
    [[nodiscard]] inline Vec2 rotated(T angle_rad) const noexcept {
        T c=cos(angle_rad), s=sin(angle_rad);
        return {x*c-y*s, x*s+y*c};
    }
    [[nodiscard]] inline Vec2 rotated(Radians<T> a) const noexcept { return rotated(a.value); }

    // Lerp
    [[nodiscard]] constexpr Vec2 lerp(Vec2 o, T t) const noexcept {
        return {qs::math::lerp(x,o.x,t), qs::math::lerp(y,o.y,t)};
    }

    // Predicates
    [[nodiscard]] constexpr bool is_zero(T eps=T(EPS_LENGTH)) const noexcept { return length_sq() <= eps*eps; }
    [[nodiscard]] constexpr bool is_finite() const noexcept requires FloatScalar<T> {
        return qs::math::is_finite(x) && qs::math::is_finite(y);
    }

    // Static factories
    [[nodiscard]] static constexpr Vec2 zero()     noexcept { return {T(0),T(0)}; }
    [[nodiscard]] static constexpr Vec2 one()      noexcept { return {T(1),T(1)}; }
    [[nodiscard]] static constexpr Vec2 unit_x()   noexcept { return {T(1),T(0)}; }
    [[nodiscard]] static constexpr Vec2 unit_y()   noexcept { return {T(0),T(1)}; }
    [[nodiscard]] static constexpr Vec2 neg_unit_x()noexcept{ return {T(-1),T(0)}; }
    [[nodiscard]] static constexpr Vec2 neg_unit_y()noexcept{ return {T(0),T(-1)}; }
    [[nodiscard]] static Vec2 from_angle(T rad)    noexcept { return {cos(rad),sin(rad)}; }
};

// Scalar-first operators
template<MathElement T> constexpr Vec2<T> operator+(T s, Vec2<T> v) noexcept { return v+s; }
template<MathElement T> constexpr Vec2<T> operator-(T s, Vec2<T> v) noexcept { return Vec2<T>{s-v.x, s-v.y}; }
template<MathElement T> constexpr Vec2<T> operator*(T s, Vec2<T> v) noexcept { return v*s; }

// Free functions
template<MathElement T> [[nodiscard]] constexpr T    dot(Vec2<T> a, Vec2<T> b)      noexcept { return a.dot(b); }
template<MathElement T> [[nodiscard]] constexpr T    cross(Vec2<T> a, Vec2<T> b)    noexcept { return a.cross(b); }
template<MathElement T> [[nodiscard]] inline     T    length(Vec2<T> v)              noexcept { return v.length(); }
template<MathElement T> [[nodiscard]] inline     Vec2<T> normalize(Vec2<T> v)        noexcept { return v.normalized(); }
template<MathElement T> [[nodiscard]] inline     T    distance(Vec2<T> a, Vec2<T> b) noexcept { return a.distance(b); }
template<MathElement T> [[nodiscard]] constexpr Vec2<T> reflect(Vec2<T> v, Vec2<T> n) noexcept { return v.reflect(n); }
template<MathElement T> [[nodiscard]] constexpr Vec2<T> lerp(Vec2<T> a, Vec2<T> b, T t) noexcept { return a.lerp(b,t); }
template<MathElement T> [[nodiscard]] constexpr Vec2<T> min(Vec2<T> a, Vec2<T> b)   noexcept { return a.min(b); }
template<MathElement T> [[nodiscard]] constexpr Vec2<T> max(Vec2<T> a, Vec2<T> b)   noexcept { return a.max(b); }
template<MathElement T> [[nodiscard]] constexpr Vec2<T> abs(Vec2<T> v)              noexcept { return v.abs(); }
template<FloatScalar T> [[nodiscard]] constexpr bool nearly_equal(Vec2<T> a, Vec2<T> b, T eps=EPS_F32) noexcept {
    return nearly_equal(a.x,b.x,eps) && nearly_equal(a.y,b.y,eps);
}

// Common aliases
using Vec2f  = Vec2<f32>;
using Vec2d  = Vec2<f64>;
using Vec2i  = Vec2<i32>;
using Vec2u  = Vec2<u32>;
using Vec2i64= Vec2<i64>;
using Vec2u64= Vec2<u64>;

} // namespace qs::math
