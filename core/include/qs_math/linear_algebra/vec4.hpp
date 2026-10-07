#pragma once
// qs_math/vec4.hpp — 4D vector
#include "vec3.hpp"

namespace qs::math {

template<MathElement T>
struct Vec4 {
    T x{T(0)}, y{T(0)}, z{T(0)}, w{T(0)};

    constexpr Vec4() noexcept = default;
    constexpr Vec4(T x,T y,T z,T w) noexcept : x(x),y(y),z(z),w(w) {}
    constexpr explicit Vec4(T s) noexcept : x(s),y(s),z(s),w(s) {}
    constexpr Vec4(Vec3<T> v, T w) noexcept : x(v.x),y(v.y),z(v.z),w(w) {}
    constexpr Vec4(Vec2<T> a, Vec2<T> b) noexcept : x(a.x),y(a.y),z(b.x),w(b.y) {}
    template<MathElement U> constexpr explicit Vec4(Vec4<U> v) noexcept : x(T(v.x)),y(T(v.y)),z(T(v.z)),w(T(v.w)) {}

    [[nodiscard]] constexpr T  operator[](i32 i) const noexcept { return (&x)[i]; }
    [[nodiscard]] constexpr T& operator[](i32 i)       noexcept { return (&x)[i]; }

    constexpr Vec4  operator+(Vec4 o) const noexcept { return {x+o.x,y+o.y,z+o.z,w+o.w}; }
    constexpr Vec4  operator-(Vec4 o) const noexcept { return {x-o.x,y-o.y,z-o.z,w-o.w}; }
    constexpr Vec4  operator*(Vec4 o) const noexcept { return {x*o.x,y*o.y,z*o.z,w*o.w}; }
    constexpr Vec4  operator/(Vec4 o) const noexcept { return {x/o.x,y/o.y,z/o.z,w/o.w}; }
    constexpr Vec4  operator-()       const noexcept { return {-x,-y,-z,-w}; }
    constexpr Vec4& operator+=(Vec4 o) noexcept { x+=o.x;y+=o.y;z+=o.z;w+=o.w; return *this; }
    constexpr Vec4& operator-=(Vec4 o) noexcept { x-=o.x;y-=o.y;z-=o.z;w-=o.w; return *this; }
    constexpr Vec4& operator*=(Vec4 o) noexcept { x*=o.x;y*=o.y;z*=o.z;w*=o.w; return *this; }
    constexpr Vec4& operator/=(Vec4 o) noexcept { x/=o.x;y/=o.y;z/=o.z;w/=o.w; return *this; }

    constexpr Vec4  operator*(T s) const noexcept { return {x*s,y*s,z*s,w*s}; }
    constexpr Vec4  operator/(T s) const noexcept { return {x/s,y/s,z/s,w/s}; }
    constexpr Vec4& operator*=(T s) noexcept { x*=s;y*=s;z*=s;w*=s; return *this; }
    constexpr Vec4& operator/=(T s) noexcept { x/=s;y/=s;z/=s;w/=s; return *this; }

    constexpr bool operator==(Vec4 o) const noexcept { return x==o.x&&y==o.y&&z==o.z&&w==o.w; }
    constexpr bool operator!=(Vec4 o) const noexcept { return !(*this==o); }

    [[nodiscard]] constexpr T length_sq() const noexcept { return x*x+y*y+z*z+w*w; }
    [[nodiscard]] inline     T length()   const noexcept { return sqrt(length_sq()); }

    [[nodiscard]] inline Vec4 normalized() const noexcept {
        T len=length(); if(len<T(EPS_LENGTH)) return *this; return {x/len,y/len,z/len,w/len};
    }

    [[nodiscard]] constexpr T dot(Vec4 o) const noexcept { return x*o.x+y*o.y+z*o.z+w*o.w; }

    // Swizzles
    [[nodiscard]] constexpr Vec2<T> xy()  const noexcept { return {x,y}; }
    [[nodiscard]] constexpr Vec2<T> zw()  const noexcept { return {z,w}; }
    [[nodiscard]] constexpr Vec3<T> xyz() const noexcept { return {x,y,z}; }
    [[nodiscard]] constexpr Vec3<T> xyw() const noexcept { return {x,y,w}; }
    [[nodiscard]] constexpr Vec3<T> xzw() const noexcept { return {x,z,w}; }
    [[nodiscard]] constexpr Vec3<T> yzw() const noexcept { return {y,z,w}; }
    [[nodiscard]] constexpr Vec4    wzyx()const noexcept { return {w,z,y,x}; }
    [[nodiscard]] constexpr Vec4    xxxx()const noexcept { return {x,x,x,x}; }
    [[nodiscard]] constexpr Vec4    yyyy()const noexcept { return {y,y,y,y}; }
    [[nodiscard]] constexpr Vec4    zzzz()const noexcept { return {z,z,z,z}; }
    [[nodiscard]] constexpr Vec4    wwww()const noexcept { return {w,w,w,w}; }

    // Perspective divide
    [[nodiscard]] constexpr Vec3<T> perspective_divide() const noexcept {
        if(qs::math::abs(w) < T(EPS_LENGTH)) return xyz();
        T inv=T(1)/w; return {x*inv,y*inv,z*inv};
    }

    [[nodiscard]] constexpr Vec4 lerp(Vec4 o, T t) const noexcept {
        return {qs::math::lerp(x,o.x,t),qs::math::lerp(y,o.y,t),
                qs::math::lerp(z,o.z,t),qs::math::lerp(w,o.w,t)};
    }

    [[nodiscard]] constexpr Vec4 min(Vec4 o) const noexcept {
        return {qs::math::min(x,o.x),qs::math::min(y,o.y),
                qs::math::min(z,o.z),qs::math::min(w,o.w)};
    }
    [[nodiscard]] constexpr Vec4 max(Vec4 o) const noexcept {
        return {qs::math::max(x,o.x),qs::math::max(y,o.y),
                qs::math::max(z,o.z),qs::math::max(w,o.w)};
    }
    [[nodiscard]] constexpr Vec4 abs() const noexcept {
        return {qs::math::abs(x),qs::math::abs(y),qs::math::abs(z),qs::math::abs(w)};
    }
    [[nodiscard]] constexpr T sum()     const noexcept { return x+y+z+w; }
    [[nodiscard]] constexpr T product() const noexcept { return x*y*z*w; }

    [[nodiscard]] constexpr bool is_finite() const noexcept requires FloatScalar<T> {
        return qs::math::is_finite(x)&&qs::math::is_finite(y)&&qs::math::is_finite(z)&&qs::math::is_finite(w);
    }

    static constexpr Vec4 zero()     noexcept { return {T(0),T(0),T(0),T(0)}; }
    static constexpr Vec4 one()      noexcept { return {T(1),T(1),T(1),T(1)}; }
    static constexpr Vec4 unit_x()   noexcept { return {T(1),T(0),T(0),T(0)}; }
    static constexpr Vec4 unit_y()   noexcept { return {T(0),T(1),T(0),T(0)}; }
    static constexpr Vec4 unit_z()   noexcept { return {T(0),T(0),T(1),T(0)}; }
    static constexpr Vec4 unit_w()   noexcept { return {T(0),T(0),T(0),T(1)}; }
};

template<MathElement T> constexpr Vec4<T> operator*(T s, Vec4<T> v) noexcept { return v*s; }
template<MathElement T> [[nodiscard]] constexpr T     dot(Vec4<T> a, Vec4<T> b) noexcept { return a.dot(b); }
template<MathElement T> [[nodiscard]] inline     Vec4<T> normalize(Vec4<T> v)   noexcept { return v.normalized(); }
template<MathElement T> [[nodiscard]] constexpr Vec4<T> lerp(Vec4<T> a, Vec4<T> b, T t) noexcept { return a.lerp(b,t); }
template<FloatScalar T> [[nodiscard]] constexpr bool nearly_equal(Vec4<T> a, Vec4<T> b, T eps=EPS_F32) noexcept {
    return nearly_equal(a.x,b.x,eps)&&nearly_equal(a.y,b.y,eps)&&nearly_equal(a.z,b.z,eps)&&nearly_equal(a.w,b.w,eps);
}

using Vec4f  = Vec4<f32>; using Vec4d  = Vec4<f64>;
using Vec4i  = Vec4<i32>; using Vec4u  = Vec4<u32>;
using Vec4i64= Vec4<i64>; using Vec4u64= Vec4<u64>;

} // namespace qs::math
