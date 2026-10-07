#pragma once
// qs_math/vec3.hpp — 3D vector, full feature set
#include "vec2.hpp"

namespace qs::math {

template<MathElement T>
struct Vec3 {
    T x{T(0)}, y{T(0)}, z{T(0)};

    constexpr Vec3() noexcept = default;
    constexpr Vec3(T x, T y, T z)  noexcept : x(x), y(y), z(z) {}
    constexpr explicit Vec3(T s)    noexcept : x(s), y(s), z(s) {}
    constexpr Vec3(Vec2<T> v, T z)  noexcept : x(v.x), y(v.y), z(z) {}
    template<MathElement U> constexpr explicit Vec3(Vec3<U> v) noexcept : x(T(v.x)), y(T(v.y)), z(T(v.z)) {}

    [[nodiscard]] constexpr T  operator[](i32 i) const noexcept { return (&x)[i]; }
    [[nodiscard]] constexpr T& operator[](i32 i)       noexcept { return (&x)[i]; }

    constexpr Vec3  operator+(Vec3 o) const noexcept { return {x+o.x, y+o.y, z+o.z}; }
    constexpr Vec3  operator-(Vec3 o) const noexcept { return {x-o.x, y-o.y, z-o.z}; }
    constexpr Vec3  operator*(Vec3 o) const noexcept { return {x*o.x, y*o.y, z*o.z}; }
    constexpr Vec3  operator/(Vec3 o) const noexcept { return {x/o.x, y/o.y, z/o.z}; }
    constexpr Vec3  operator-()       const noexcept { return {-x,-y,-z}; }
    constexpr Vec3& operator+=(Vec3 o) noexcept { x+=o.x; y+=o.y; z+=o.z; return *this; }
    constexpr Vec3& operator-=(Vec3 o) noexcept { x-=o.x; y-=o.y; z-=o.z; return *this; }
    constexpr Vec3& operator*=(Vec3 o) noexcept { x*=o.x; y*=o.y; z*=o.z; return *this; }
    constexpr Vec3& operator/=(Vec3 o) noexcept { x/=o.x; y/=o.y; z/=o.z; return *this; }

    constexpr Vec3  operator+(T s) const noexcept { return {x+s, y+s, z+s}; }
    constexpr Vec3  operator-(T s) const noexcept { return {x-s, y-s, z-s}; }
    constexpr Vec3  operator*(T s) const noexcept { return {x*s, y*s, z*s}; }
    constexpr Vec3  operator/(T s) const noexcept { return {x/s, y/s, z/s}; }
    constexpr Vec3& operator+=(T s) noexcept { x+=s; y+=s; z+=s; return *this; }
    constexpr Vec3& operator-=(T s) noexcept { x-=s; y-=s; z-=s; return *this; }
    constexpr Vec3& operator*=(T s) noexcept { x*=s; y*=s; z*=s; return *this; }
    constexpr Vec3& operator/=(T s) noexcept { x/=s; y/=s; z/=s; return *this; }

    constexpr bool operator==(Vec3 o) const noexcept { return x==o.x && y==o.y && z==o.z; }
    constexpr bool operator!=(Vec3 o) const noexcept { return !(*this==o); }

    [[nodiscard]] constexpr T length_sq()   const noexcept { return x*x+y*y+z*z; }
    [[nodiscard]] inline     T length()     const noexcept { return sqrt(length_sq()); }
    [[nodiscard]] inline     T inv_length() const noexcept { return T(1)/length(); }

    [[nodiscard]] inline Vec3 normalized() const noexcept {
        T len=length(); if(len<T(EPS_LENGTH)) return *this; return {x/len,y/len,z/len};
    }
    [[nodiscard]] inline bool is_normalized(T eps=T(EPS_NORMAL)) const noexcept {
        return abs(length_sq()-T(1))<=eps;
    }

    [[nodiscard]] constexpr T dot(Vec3 o) const noexcept { return x*o.x+y*o.y+z*o.z; }

    [[nodiscard]] constexpr Vec3 cross(Vec3 o) const noexcept {
        return {y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x};
    }

    [[nodiscard]] inline T distance(Vec3 o)       const noexcept { return (*this-o).length(); }
    [[nodiscard]] constexpr T distance_sq(Vec3 o) const noexcept { return (*this-o).length_sq(); }

    [[nodiscard]] constexpr Vec3 project_onto(Vec3 o) const noexcept {
        T d=o.dot(o); if(d<T(EPS_LENGTH)) return Vec3{T(0)}; return o*(dot(o)/d);
    }
    [[nodiscard]] constexpr Vec3 reject_from(Vec3 o) const noexcept { return *this-project_onto(o); }
    [[nodiscard]] constexpr Vec3 reflect(Vec3 n)     const noexcept { return *this-n*(T(2)*dot(n)); }
    [[nodiscard]] inline    Vec3 refract(Vec3 n, T eta) const noexcept {
        T d=dot(n); T k=T(1)-eta*eta*(T(1)-d*d);
        if(k<T(0)) return Vec3{T(0)};
        return *this*eta - n*(eta*d+sqrt(k));
    }

    [[nodiscard]] constexpr Vec3 min(Vec3 o) const noexcept {
        return {qs::math::min(x,o.x),qs::math::min(y,o.y),qs::math::min(z,o.z)};
    }
    [[nodiscard]] constexpr Vec3 max(Vec3 o) const noexcept {
        return {qs::math::max(x,o.x),qs::math::max(y,o.y),qs::math::max(z,o.z)};
    }
    [[nodiscard]] constexpr Vec3 clamp(Vec3 lo,Vec3 hi) const noexcept { return min(hi).max(lo); }
    [[nodiscard]] constexpr Vec3 abs() const noexcept { return {qs::math::abs(x),qs::math::abs(y),qs::math::abs(z)}; }
    [[nodiscard]] constexpr T min_component() const noexcept { return qs::math::min3(x,y,z); }
    [[nodiscard]] constexpr T max_component() const noexcept { return qs::math::max3(x,y,z); }

    // Swizzles — common ones (full swizzle set in swizzle.hpp)
    [[nodiscard]] constexpr Vec2<T> xy() const noexcept { return {x,y}; }
    [[nodiscard]] constexpr Vec2<T> xz() const noexcept { return {x,z}; }
    [[nodiscard]] constexpr Vec2<T> yz() const noexcept { return {y,z}; }
    [[nodiscard]] constexpr Vec3    xxx()const noexcept { return {x,x,x}; }
    [[nodiscard]] constexpr Vec3    yyy()const noexcept { return {y,y,y}; }
    [[nodiscard]] constexpr Vec3    zzz()const noexcept { return {z,z,z}; }
    [[nodiscard]] constexpr Vec3    xzy()const noexcept { return {x,z,y}; }
    [[nodiscard]] constexpr Vec3    yxz()const noexcept { return {y,x,z}; }
    [[nodiscard]] constexpr Vec3    yzx()const noexcept { return {y,z,x}; }
    [[nodiscard]] constexpr Vec3    zxy()const noexcept { return {z,x,y}; }
    [[nodiscard]] constexpr Vec3    zyx()const noexcept { return {z,y,x}; }

    [[nodiscard]] constexpr Vec3 lerp(Vec3 o, T t) const noexcept {
        return {qs::math::lerp(x,o.x,t),qs::math::lerp(y,o.y,t),qs::math::lerp(z,o.z,t)};
    }

    [[nodiscard]] constexpr bool is_zero(T eps=T(EPS_LENGTH))  const noexcept { return length_sq()<=eps*eps; }
    [[nodiscard]] constexpr bool is_finite() const noexcept requires FloatScalar<T> {
        return qs::math::is_finite(x) && qs::math::is_finite(y) && qs::math::is_finite(z);
    }

    // Orthonormal basis from this (must be unit)
    void make_basis(Vec3& out_tangent, Vec3& out_bitangent) const noexcept {
        Vec3 n = *this;
        Vec3 t = (qs::math::abs(n.x) > T(0.9f)) ? Vec3{T(0),T(1),T(0)} : Vec3{T(1),T(0),T(0)};
        out_bitangent = n.cross(t).normalized();
        out_tangent   = out_bitangent.cross(n);
    }

    // Component sum / product
    [[nodiscard]] constexpr T sum()     const noexcept { return x+y+z; }
    [[nodiscard]] constexpr T product() const noexcept { return x*y*z; }

    static constexpr Vec3 zero()    noexcept { return {T(0),T(0),T(0)}; }
    static constexpr Vec3 one()     noexcept { return {T(1),T(1),T(1)}; }
    static constexpr Vec3 unit_x()  noexcept { return {T(1),T(0),T(0)}; }
    static constexpr Vec3 unit_y()  noexcept { return {T(0),T(1),T(0)}; }
    static constexpr Vec3 unit_z()  noexcept { return {T(0),T(0),T(1)}; }
    static constexpr Vec3 right()   noexcept { return unit_x(); }
    static constexpr Vec3 up()      noexcept { return unit_y(); }
    static constexpr Vec3 forward() noexcept { return {T(0),T(0),T(-1)}; } // RH convention
};

template<MathElement T> constexpr Vec3<T> operator+(T s, Vec3<T> v) noexcept { return v+s; }
template<MathElement T> constexpr Vec3<T> operator*(T s, Vec3<T> v) noexcept { return v*s; }
template<MathElement T> [[nodiscard]] constexpr T     dot(Vec3<T> a, Vec3<T> b) noexcept { return a.dot(b); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> cross(Vec3<T> a, Vec3<T> b) noexcept { return a.cross(b); }
template<MathElement T> [[nodiscard]] inline     T     length(Vec3<T> v) noexcept { return v.length(); }
template<MathElement T> [[nodiscard]] inline     Vec3<T> normalize(Vec3<T> v) noexcept { return v.normalized(); }
template<MathElement T> [[nodiscard]] inline     T     distance(Vec3<T> a, Vec3<T> b) noexcept { return a.distance(b); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> reflect(Vec3<T> v, Vec3<T> n) noexcept { return v.reflect(n); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> lerp(Vec3<T> a, Vec3<T> b, T t) noexcept { return a.lerp(b,t); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> min(Vec3<T> a, Vec3<T> b) noexcept { return a.min(b); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> max(Vec3<T> a, Vec3<T> b) noexcept { return a.max(b); }
template<MathElement T> [[nodiscard]] constexpr Vec3<T> abs(Vec3<T> v) noexcept { return v.abs(); }
template<FloatScalar T> [[nodiscard]] constexpr bool nearly_equal(Vec3<T> a, Vec3<T> b, T eps=EPS_F32) noexcept {
    return nearly_equal(a.x,b.x,eps) && nearly_equal(a.y,b.y,eps) && nearly_equal(a.z,b.z,eps);
}

using Vec3f  = Vec3<f32>; using Vec3d  = Vec3<f64>;
using Vec3i  = Vec3<i32>; using Vec3u  = Vec3<u32>;
using Vec3i64= Vec3<i64>; using Vec3u64= Vec3<u64>;

} // namespace qs::math
