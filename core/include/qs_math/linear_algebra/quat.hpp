#pragma once
// qs_math/quat.hpp — Quaternion: construction, ops, SLERP, NLERP, matrix conversion
#include "mat3.hpp"
#include "mat4.hpp"
#include "vec3.hpp"

namespace qs::math {

template<FloatScalar T>
struct Quat {
    T x{T(0)}, y{T(0)}, z{T(0)}, w{T(1)};

    constexpr Quat() noexcept = default;
    constexpr Quat(T x,T y,T z,T w) noexcept : x(x),y(y),z(z),w(w) {}
    constexpr explicit Quat(Vec4<T> v) noexcept : x(v.x),y(v.y),z(v.z),w(v.w) {}

    // From axis + angle
    static inline Quat from_axis_angle(Vec3<T> axis, T angle_rad) noexcept {
        T half=angle_rad*T(0.5), s=qs::math::sin(half), c=qs::math::cos(half);
        Vec3<T> a=axis.normalized();
        return {a.x*s, a.y*s, a.z*s, c};
    }
    static inline Quat from_axis_angle(Vec3<T> axis, Radians<T> angle) noexcept {
        return from_axis_angle(axis, angle.value);
    }

    // From Euler (XYZ order)
    static inline Quat from_euler(T pitch, T yaw, T roll) noexcept {
        T hp=pitch*T(0.5), hy=yaw*T(0.5), hr=roll*T(0.5);
        T sp=qs::math::sin(hp),cp=qs::math::cos(hp);
        T sy=qs::math::sin(hy),cy=qs::math::cos(hy);
        T sr=qs::math::sin(hr),cr=qs::math::cos(hr);
        return {
            sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy,
            cr*cp*cy+sr*sp*sy
        };
    }
    static inline Quat from_euler(Vec3<T> euler_rad) noexcept {
        return from_euler(euler_rad.x, euler_rad.y, euler_rad.z);
    }

    // From rotation matrix
    static inline Quat from_mat3(Mat3<T> m) noexcept {
        T trace=m[0].x+m[1].y+m[2].z;
        Quat q;
        if(trace>T(0)){
            T s=qs::math::sqrt(trace+T(1))*T(2); T inv=T(1)/s;
            q.w=s*T(0.25); q.x=(m[2].y-m[1].z)*inv; q.y=(m[0].z-m[2].x)*inv; q.z=(m[1].x-m[0].y)*inv;
        } else if(m[0].x>m[1].y&&m[0].x>m[2].z){
            T s=qs::math::sqrt(T(1)+m[0].x-m[1].y-m[2].z)*T(2); T inv=T(1)/s;
            q.w=(m[2].y-m[1].z)*inv; q.x=s*T(0.25); q.y=(m[0].y+m[1].x)*inv; q.z=(m[0].z+m[2].x)*inv;
        } else if(m[1].y>m[2].z){
            T s=qs::math::sqrt(T(1)+m[1].y-m[0].x-m[2].z)*T(2); T inv=T(1)/s;
            q.w=(m[0].z-m[2].x)*inv; q.x=(m[0].y+m[1].x)*inv; q.y=s*T(0.25); q.z=(m[1].z+m[2].y)*inv;
        } else {
            T s=qs::math::sqrt(T(1)+m[2].z-m[0].x-m[1].y)*T(2); T inv=T(1)/s;
            q.w=(m[1].x-m[0].y)*inv; q.x=(m[0].z+m[2].x)*inv; q.y=(m[1].z+m[2].y)*inv; q.z=s*T(0.25);
        }
        return q;
    }

    // From/to vectors (shortest rotation)
    static inline Quat from_to(Vec3<T> from, Vec3<T> to) noexcept {
        from=from.normalized(); to=to.normalized();
        T d=dot(from,to);
        if(d>T(1)-EPS_NORMAL) return identity();
        if(d<-(T(1)-EPS_NORMAL)) {
            Vec3<T> perp={T(1),T(0),T(0)};
            if(qs::math::abs(from.x)>T(0.9f)) perp={T(0),T(1),T(0)};
            Vec3<T> ax=cross(from,perp).normalized();
            return from_axis_angle(ax, T(PI));
        }
        Vec3<T> ax=cross(from,to);
        return Quat{ax.x,ax.y,ax.z,T(1)+d}.normalized();
    }

    // Arithmetic
    constexpr Quat operator+(Quat o) const noexcept { return {x+o.x,y+o.y,z+o.z,w+o.w}; }
    constexpr Quat operator-(Quat o) const noexcept { return {x-o.x,y-o.y,z-o.z,w-o.w}; }
    constexpr Quat operator*(T s)    const noexcept { return {x*s,y*s,z*s,w*s}; }
    constexpr Quat operator-()       const noexcept { return {-x,-y,-z,-w}; }

    // Hamilton product
    constexpr Quat operator*(Quat o) const noexcept {
        return {
            w*o.x+x*o.w+y*o.z-z*o.y,
            w*o.y-x*o.z+y*o.w+z*o.x,
            w*o.z+x*o.y-y*o.x+z*o.w,
            w*o.w-x*o.x-y*o.y-z*o.z
        };
    }
    constexpr Quat& operator*=(Quat o) noexcept { *this=(*this)*o; return *this; }

    constexpr bool operator==(Quat o) const noexcept { return x==o.x&&y==o.y&&z==o.z&&w==o.w; }

    // Length
    [[nodiscard]] constexpr T length_sq() const noexcept { return x*x+y*y+z*z+w*w; }
    [[nodiscard]] inline     T length()   const noexcept { return qs::math::sqrt(length_sq()); }

    [[nodiscard]] inline Quat normalized() const noexcept {
        T len=length(); if(len<T(EPS_LENGTH)) return identity();
        T inv=T(1)/len; return {x*inv,y*inv,z*inv,w*inv};
    }
    [[nodiscard]] constexpr bool is_normalized(T eps=T(EPS_NORMAL)) const noexcept {
        return qs::math::abs(length_sq()-T(1))<=eps;
    }

    // Conjugate / inverse
    [[nodiscard]] constexpr Quat conjugate() const noexcept { return {-x,-y,-z,w}; }
    [[nodiscard]] inline     Quat inverse()  const noexcept {
        T lsq=length_sq(); if(lsq<T(EPS_LENGTH)) return identity();
        return conjugate()*(T(1)/lsq);
    }

    // Dot
    [[nodiscard]] constexpr T dot(Quat o) const noexcept { return x*o.x+y*o.y+z*o.z+w*o.w; }

    // Rotate vector
    [[nodiscard]] constexpr Vec3<T> rotate(Vec3<T> v) const noexcept {
        Vec3<T> qv{x,y,z};
        Vec3<T> uv=cross(qv,v);
        Vec3<T> uuv=cross(qv,uv);
        return v + (uv*w + uuv)*T(2);
    }

    // To rotation matrix
    [[nodiscard]] constexpr Mat3<T> to_mat3() const noexcept {
        T xx=x*x,yy=y*y,zz=z*z,xy=x*y,xz=x*z,yz=y*z,wx=w*x,wy=w*y,wz=w*z;
        return {
            {T(1)-T(2)*(yy+zz), T(2)*(xy+wz),       T(2)*(xz-wy)},
            {T(2)*(xy-wz),       T(1)-T(2)*(xx+zz), T(2)*(yz+wx)},
            {T(2)*(xz+wy),       T(2)*(yz-wx),       T(1)-T(2)*(xx+yy)}
        };
    }
    [[nodiscard]] constexpr Mat4<T> to_mat4() const noexcept { return Mat4<T>{to_mat3()}; }

    // Euler angles (YXZ convention)
    [[nodiscard]] inline Vec3<T> to_euler() const noexcept {
        T pitch=qs::math::asin(clamp(T(2)*(w*x-y*z),T(-1),T(1)));
        T yaw,roll;
        T sy=T(2)*(w*y+z*x), cy=T(1)-T(2)*(x*x+y*y);
        yaw=qs::math::atan2(sy,cy);
        T sr=T(2)*(w*z+x*y), cr=T(1)-T(2)*(x*x+z*z);
        roll=qs::math::atan2(sr,cr);
        return {pitch,yaw,roll};
    }

    // Axis/angle extraction
    [[nodiscard]] inline void to_axis_angle(Vec3<T>& axis, T& angle) const noexcept {
        Quat q=normalized();
        angle=T(2)*qs::math::acos(clamp(q.w,T(-1),T(1)));
        T s=qs::math::sqrt(T(1)-q.w*q.w);
        if(s<T(EPS_LENGTH)) axis={T(1),T(0),T(0)};
        else axis={q.x/s,q.y/s,q.z/s};
    }

    // SLERP
    [[nodiscard]] static inline Quat slerp(Quat a, Quat b, T t) noexcept {
        T d=a.dot(b);
        if(d<T(0)){b=-b;d=-d;}
        if(d>T(1)-EPS_NORMAL) return (a+(b-a)*t).normalized();
        T theta0=qs::math::acos(d);
        T theta=theta0*t;
        T s0=qs::math::cos(theta)-d*qs::math::sin(theta)/qs::math::sin(theta0);
        T s1=qs::math::sin(theta)/qs::math::sin(theta0);
        return (a*s0+b*s1).normalized();
    }

    // NLERP (cheaper, slight error)
    [[nodiscard]] static constexpr Quat nlerp(Quat a, Quat b, T t) noexcept {
        if(a.dot(b)<T(0)) b=-b;
        return (a+(b-a)*t).normalized();
    }

    // Squad (smooth quaternion interpolation along a path)
    [[nodiscard]] static inline Quat squad(Quat q0,Quat q1,Quat s0,Quat s1,T t) noexcept {
        return slerp(slerp(q0,q1,t),slerp(s0,s1,t),T(2)*t*(T(1)-t));
    }

    [[nodiscard]] constexpr Vec4<T> to_vec4() const noexcept { return {x,y,z,w}; }
    [[nodiscard]] constexpr Vec3<T> vector_part() const noexcept { return {x,y,z}; }

    static constexpr Quat identity()   noexcept { return {T(0),T(0),T(0),T(1)}; }
    static constexpr Quat zero()       noexcept { return {T(0),T(0),T(0),T(0)}; }
};

template<FloatScalar T> [[nodiscard]] inline     Quat<T> normalize(Quat<T> q)  noexcept { return q.normalized(); }
template<FloatScalar T> [[nodiscard]] constexpr  Quat<T> conjugate(Quat<T> q)  noexcept { return q.conjugate(); }
template<FloatScalar T> [[nodiscard]] inline     Quat<T> inverse(Quat<T> q)    noexcept { return q.inverse(); }
template<FloatScalar T> [[nodiscard]] constexpr  T       dot(Quat<T> a,Quat<T> b) noexcept { return a.dot(b); }
template<FloatScalar T> [[nodiscard]] static inline Quat<T> slerp(Quat<T> a, Quat<T> b, T t) noexcept { return Quat<T>::slerp(a,b,t); }
template<FloatScalar T> [[nodiscard]] constexpr bool nearly_equal(Quat<T> a,Quat<T> b,T eps=EPS_F32) noexcept {
    return (nearly_equal(a.x,b.x,eps)&&nearly_equal(a.y,b.y,eps)&&nearly_equal(a.z,b.z,eps)&&nearly_equal(a.w,b.w,eps))
        || (nearly_equal(-a.x,b.x,eps)&&nearly_equal(-a.y,b.y,eps)&&nearly_equal(-a.z,b.z,eps)&&nearly_equal(-a.w,b.w,eps));
}

using Quatf = Quat<f32>;
using Quatd = Quat<f64>;

} // namespace qs::math
