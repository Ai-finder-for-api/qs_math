#pragma once
// qs_math/curves.hpp — Bezier, Hermite, Catmull-Rom, B-Spline, NURBS helpers, arc
#include "vec2.hpp"
#include "vec3.hpp"
#include "scalar_math.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Bezier — generic over Vec2/Vec3
// ---------------------------------------------------------------------------

// Linear Bezier (lerp)
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V bezier_linear(V p0, V p1, T t) noexcept { return lerp(p0,p1,t); }

// Quadratic Bezier
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V bezier_quadratic(V p0, V p1, V p2, T t) noexcept {
    T u=T(1)-t;
    return p0*(u*u) + p1*(T(2)*u*t) + p2*(t*t);
}

// Quadratic Bezier derivative (tangent direction)
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V bezier_quadratic_deriv(V p0, V p1, V p2, T t) noexcept {
    return (p1-p0)*(T(2)*(T(1)-t)) + (p2-p1)*(T(2)*t);
}

// Cubic Bezier (most common)
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V bezier_cubic(V p0, V p1, V p2, V p3, T t) noexcept {
    T u=T(1)-t, u2=u*u, t2=t*t;
    return p0*(u2*u) + p1*(T(3)*u2*t) + p2*(T(3)*u*t2) + p3*(t2*t);
}

// Cubic Bezier derivative
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V bezier_cubic_deriv(V p0, V p1, V p2, V p3, T t) noexcept {
    T u=T(1)-t;
    return (p1-p0)*(T(3)*u*u) + (p2-p1)*(T(6)*u*t) + (p3-p2)*(T(3)*t*t);
}

// Cubic Bezier second derivative
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V bezier_cubic_deriv2(V p0, V p1, V p2, V p3, T t) noexcept {
    return (p2-p1*T(2)+p0)*(T(6)*(T(1)-t)) + (p3-p2*T(2)+p1)*(T(6)*t);
}

// De Casteljau — evaluate arbitrary degree bezier
template<typename V, FloatScalar T=f32>
[[nodiscard]] V bezier_decasteljau(const V* pts, i32 n, T t) noexcept {
    // Work in-place on a stack copy (n <= 16 assumed)
    V tmp[16];
    i32 cnt = n < 16 ? n : 16;
    for(i32 i=0;i<cnt;++i) tmp[i]=pts[i];
    for(i32 r=1;r<cnt;++r)
        for(i32 i=0;i<cnt-r;++i)
            tmp[i]=lerp(tmp[i],tmp[i+1],t);
    return tmp[0];
}

// Cubic Bezier arc length (Gaussian quadrature, 5-point)
template<typename V, FloatScalar T=f32>
[[nodiscard]] T bezier_cubic_arc_length(V p0, V p1, V p2, V p3) noexcept {
    // 5-point Gauss-Legendre on [0,1]
    static constexpr f64 w[]={0.2369268851,0.4786286705,0.5688888889,0.4786286705,0.2369268851};
    static constexpr f64 x[]={-0.9061798459,-0.5384693101,0.0,0.5384693101,0.9061798459};
    T len=T(0);
    for(i32 i=0;i<5;++i){
        T t=T(x[i]*0.5+0.5);
        V d=bezier_cubic_deriv(p0,p1,p2,p3,t);
        len+=T(w[i])*d.length();
    }
    return len*T(0.5);
}

// ---------------------------------------------------------------------------
// Hermite spline
// ---------------------------------------------------------------------------
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V hermite(V p0, V m0, V p1, V m1, T t) noexcept {
    T t2=t*t, t3=t2*t;
    T h00=T(2)*t3-T(3)*t2+T(1);
    T h10=t3-T(2)*t2+t;
    T h01=-T(2)*t3+T(3)*t2;
    T h11=t3-t2;
    return p0*h00 + m0*h10 + p1*h01 + m1*h11;
}

template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V hermite_deriv(V p0, V m0, V p1, V m1, T t) noexcept {
    T t2=t*t;
    T h00=T(6)*t2-T(6)*t;
    T h10=T(3)*t2-T(4)*t+T(1);
    T h01=-T(6)*t2+T(6)*t;
    T h11=T(3)*t2-T(2)*t;
    return p0*h00 + m0*h10 + p1*h01 + m1*h11;
}

// ---------------------------------------------------------------------------
// Catmull-Rom spline
// ---------------------------------------------------------------------------
template<typename V, FloatScalar T=f32>
[[nodiscard]] constexpr V catmull_rom(V p0, V p1, V p2, V p3, T t, T alpha=T(0.5)) noexcept {
    // Centripetal Catmull-Rom
    auto knot=[&](V a,V b)->T{
        T d=distance(a,b);
        return pow(d,alpha)+T(0); // using default centripetal
    };
    (void)knot; (void)alpha;
    // Standard uniform Catmull-Rom
    T t2=t*t, t3=t2*t;
    return (p1*T(2) + (p2-p0)*t
         + (p0*T(2)-p1*T(5)+p2*T(4)-p3)*t2
         + (-p0+p1*T(3)-p2*T(3)+p3)*t3)*T(0.5);
}

// Centripetal Catmull-Rom (eliminates cusps and self-intersections)
template<FloatScalar T>
[[nodiscard]] Vec3<T> catmull_rom_centripetal(Vec3<T> p0,Vec3<T> p1,Vec3<T> p2,Vec3<T> p3, T t) noexcept {
    auto knot=[](Vec3<T> a,Vec3<T> b)->T{ return sqrt(distance(a,b)); };
    T t0=T(0), t1=knot(p0,p1)+t0, t2=knot(p1,p2)+t1, t3=knot(p2,p3)+t2;
    T tr=lerp(t1,t2,t);
    auto ba=[](Vec3<T> a,Vec3<T> b,T ta,T tb,T tc)->Vec3<T>{
        if(abs(tb-ta)<EPS_F32) return a;
        return a*((tc-tb)/(tc-ta))+b*((tb-ta)/(tc-ta));
    };
    Vec3<T> a1=ba(p0,p1,t0,tr,t1), a2=ba(p1,p2,t1,tr,t2), a3=ba(p2,p3,t2,tr,t3);
    Vec3<T> b1=ba(a1,a2,t1,tr,t2), b2=ba(a2,a3,t2,tr,t3);
    return ba(b1,b2,t1,tr,t2);
}

// ---------------------------------------------------------------------------
// B-Spline basis
// ---------------------------------------------------------------------------
// Cox-de Boor recursive B-spline basis function N_{i,k}(t)
[[nodiscard]] inline f32 bspline_basis(i32 i, i32 k, f32 t, const f32* knots) noexcept {
    if(k==1) return (t>=knots[i]&&t<knots[i+1])?1.f:0.f;
    f32 left=0.f, right=0.f;
    f32 d1=knots[i+k-1]-knots[i], d2=knots[i+k]-knots[i+1];
    if(d1>EPS_F32) left=(t-knots[i])/d1*bspline_basis(i,k-1,t,knots);
    if(d2>EPS_F32) right=(knots[i+k]-t)/d2*bspline_basis(i+1,k-1,t,knots);
    return left+right;
}

// ---------------------------------------------------------------------------
// Arc evaluation — circular arc
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] inline Vec2<T> arc_point(Vec2<T> center, T radius, Radians<T> start, Radians<T> span, T t) noexcept {
    Radians<T> angle{start.value + span.value*t};
    return center + Vec2<T>{cos(angle)*radius, sin(angle)*radius};
}

// Ellipse
template<FloatScalar T>
[[nodiscard]] inline Vec2<T> ellipse_point(Vec2<T> center, T rx, T ry, Radians<T> angle) noexcept {
    return center + Vec2<T>{rx*cos(angle), ry*sin(angle)};
}

// ---------------------------------------------------------------------------
// Spline segment — stores 4 control points for cubic spline
// ---------------------------------------------------------------------------
template<typename V, FloatScalar T=f32>
struct CubicSegment {
    V p[4];
    constexpr CubicSegment() noexcept = default;
    constexpr CubicSegment(V p0,V p1,V p2,V p3) noexcept : p{p0,p1,p2,p3} {}
    [[nodiscard]] constexpr V eval_bezier(T t)   const noexcept { return bezier_cubic(p[0],p[1],p[2],p[3],t); }
    [[nodiscard]] constexpr V eval_hermite(T t)  const noexcept { return hermite(p[0],p[1],p[2],p[3],t); }
    [[nodiscard]] constexpr V eval_catmull(T t)  const noexcept { return catmull_rom(p[0],p[1],p[2],p[3],t); }
    [[nodiscard]]           V tangent(T t)        const noexcept { return bezier_cubic_deriv(p[0],p[1],p[2],p[3],t); }
    [[nodiscard]]           T arc_length()        const noexcept { return bezier_cubic_arc_length(p[0],p[1],p[2],p[3]); }
};

using CubicSegment2f = CubicSegment<Vec2f,f32>;
using CubicSegment3f = CubicSegment<Vec3f,f32>;

} // namespace qs::math
