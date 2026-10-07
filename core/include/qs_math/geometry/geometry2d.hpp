#pragma once
// qs_math/geometry2d.hpp — 2D geometric primitives
#include "vec2.hpp"
#include "scalar_math.hpp"
#include <array>

namespace qs::math {

// ---------------------------------------------------------------------------
// Point2 (alias with semantic meaning)
// ---------------------------------------------------------------------------
template<FloatScalar T> using Point2 = Vec2<T>;
using Point2f = Vec2f; using Point2d = Vec2d;

// ---------------------------------------------------------------------------
// Line2 — infinite line defined by point + direction
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Line2 {
    Vec2<T> origin, direction; // direction must be normalized

    constexpr Line2() noexcept = default;
    constexpr Line2(Vec2<T> o, Vec2<T> d) noexcept : origin(o), direction(d.normalized()) {}

    static constexpr Line2 from_two_points(Vec2<T> a, Vec2<T> b) noexcept {
        return {a, (b-a)};
    }

    [[nodiscard]] constexpr Vec2<T> point_at(T t) const noexcept { return origin + direction*t; }

    // Signed distance from point to line (positive = left of dir)
    [[nodiscard]] constexpr T signed_distance(Vec2<T> p) const noexcept {
        return cross(direction, p - origin);
    }
    [[nodiscard]] constexpr T distance(Vec2<T> p) const noexcept { return abs(signed_distance(p)); }

    // Closest point on line to p
    [[nodiscard]] constexpr Vec2<T> closest_point(Vec2<T> p) const noexcept {
        T t = dot(p - origin, direction);
        return point_at(t);
    }

    // Intersection with another line; returns false if parallel
    [[nodiscard]] bool intersect(Line2 other, Vec2<T>& out) const noexcept {
        T d = cross(direction, other.direction);
        if (abs(d) < T(EPS_INTERSECT)) return false;
        T t = cross(other.origin - origin, other.direction) / d;
        out = point_at(t);
        return true;
    }
};

// ---------------------------------------------------------------------------
// Segment2 — finite line segment
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Segment2 {
    Vec2<T> a, b;

    constexpr Segment2() noexcept = default;
    constexpr Segment2(Vec2<T> a, Vec2<T> b) noexcept : a(a), b(b) {}

    [[nodiscard]] constexpr Vec2<T> direction()   const noexcept { return (b-a); }
    [[nodiscard]] inline     T      length()       const noexcept { return (b-a).length(); }
    [[nodiscard]] constexpr T      length_sq()    const noexcept { return (b-a).length_sq(); }
    [[nodiscard]] constexpr Vec2<T> midpoint()    const noexcept { return (a+b)*T(0.5); }
    [[nodiscard]] constexpr Vec2<T> point_at(T t) const noexcept { return a + (b-a)*t; }

    [[nodiscard]] constexpr Vec2<T> closest_point(Vec2<T> p) const noexcept {
        T lsq = length_sq();
        if (lsq < T(EPS_LENGTH)) return a;
        T t = saturate(dot(p-a, b-a) / lsq);
        return point_at(t);
    }
    [[nodiscard]] inline T distance(Vec2<T> p) const noexcept { return (p - closest_point(p)).length(); }

    [[nodiscard]] bool intersect(Segment2 other, Vec2<T>& out) const noexcept {
        Vec2<T> r=b-a, s=other.b-other.a;
        T denom=cross(r,s);
        if (abs(denom)<T(EPS_INTERSECT)) return false;
        Vec2<T> qp=other.a-a;
        T t=cross(qp,s)/denom, u=cross(qp,r)/denom;
        if (t<T(0)||t>T(1)||u<T(0)||u>T(1)) return false;
        out=a+r*t; return true;
    }
};

// ---------------------------------------------------------------------------
// Ray2
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Ray2 {
    Vec2<T> origin, direction;
    constexpr Ray2() noexcept = default;
    constexpr Ray2(Vec2<T> o, Vec2<T> d) noexcept : origin(o), direction(d.normalized()) {}
    [[nodiscard]] constexpr Vec2<T> at(T t) const noexcept { return origin + direction*t; }
    [[nodiscard]] constexpr Vec2<T> closest_point(Vec2<T> p) const noexcept {
        T t = max(T(0), dot(p-origin, direction));
        return at(t);
    }
};

// ---------------------------------------------------------------------------
// Circle
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Circle {
    Vec2<T> center;
    T radius{T(0)};

    constexpr Circle() noexcept = default;
    constexpr Circle(Vec2<T> c, T r) noexcept : center(c), radius(r) {}

    [[nodiscard]] constexpr T area()        const noexcept { return T(PI)*radius*radius; }
    [[nodiscard]] constexpr T circumference()const noexcept{ return T(TWO_PI)*radius; }
    [[nodiscard]] constexpr bool contains(Vec2<T> p) const noexcept {
        return (p-center).length_sq() <= radius*radius;
    }
    [[nodiscard]] constexpr bool intersects(Circle o) const noexcept {
        T rd=radius+o.radius; return (center-o.center).length_sq()<=rd*rd;
    }

    // Bounding circle from 3 points
    static Circle from_three_points(Vec2<T> a, Vec2<T> b, Vec2<T> c) noexcept;
};

// ---------------------------------------------------------------------------
// Rect2 (axis-aligned)
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Rect2 {
    Vec2<T> min_pt, max_pt;

    constexpr Rect2() noexcept = default;
    constexpr Rect2(Vec2<T> mn, Vec2<T> mx) noexcept : min_pt(mn), max_pt(mx) {}
    static constexpr Rect2 from_center_size(Vec2<T> c, Vec2<T> sz) noexcept {
        Vec2<T> h=sz*T(0.5); return {c-h,c+h};
    }
    static constexpr Rect2 from_min_size(Vec2<T> mn, Vec2<T> sz) noexcept { return {mn,mn+sz}; }

    [[nodiscard]] constexpr Vec2<T> size()    const noexcept { return max_pt-min_pt; }
    [[nodiscard]] constexpr Vec2<T> center()  const noexcept { return (min_pt+max_pt)*T(0.5); }
    [[nodiscard]] constexpr T       area()    const noexcept { Vec2<T> s=size(); return s.x*s.y; }
    [[nodiscard]] constexpr T       perimeter()const noexcept{ Vec2<T> s=size(); return T(2)*(s.x+s.y); }
    [[nodiscard]] constexpr bool contains(Vec2<T> p) const noexcept {
        return p.x>=min_pt.x&&p.x<=max_pt.x&&p.y>=min_pt.y&&p.y<=max_pt.y;
    }
    [[nodiscard]] constexpr bool intersects(Rect2 o) const noexcept {
        return min_pt.x<=o.max_pt.x&&max_pt.x>=o.min_pt.x&&min_pt.y<=o.max_pt.y&&max_pt.y>=o.min_pt.y;
    }
    [[nodiscard]] constexpr Rect2 intersection(Rect2 o) const noexcept {
        return {min_pt.max(o.min_pt), max_pt.min(o.max_pt)};
    }
    [[nodiscard]] constexpr Rect2 merge(Rect2 o) const noexcept {
        return {min_pt.min(o.min_pt), max_pt.max(o.max_pt)};
    }
    [[nodiscard]] constexpr Rect2 expanded(T amount) const noexcept {
        Vec2<T> a{amount,amount}; return {min_pt-a,max_pt+a};
    }
    [[nodiscard]] constexpr Vec2<T> closest_point(Vec2<T> p) const noexcept {
        return p.clamp(min_pt, max_pt);
    }
};

// ---------------------------------------------------------------------------
// Triangle2
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Triangle2 {
    Vec2<T> a, b, c;

    constexpr Triangle2() noexcept = default;
    constexpr Triangle2(Vec2<T> a, Vec2<T> b, Vec2<T> c) noexcept : a(a),b(b),c(c) {}

    [[nodiscard]] constexpr T area() const noexcept {
        return abs(cross(b-a, c-a))*T(0.5);
    }
    [[nodiscard]] constexpr T signed_area() const noexcept { return cross(b-a,c-a)*T(0.5); }
    [[nodiscard]] constexpr bool is_ccw() const noexcept { return signed_area()>T(0); }
    [[nodiscard]] constexpr Vec2<T> centroid() const noexcept { return (a+b+c)*ONE_THIRD; }
    [[nodiscard]] constexpr bool contains(Vec2<T> p) const noexcept {
        T d0=cross(b-a,p-a), d1=cross(c-b,p-b), d2=cross(a-c,p-c);
        bool n=(d0<T(0))||(d1<T(0))||(d2<T(0));
        bool pos=(d0>T(0))||(d1>T(0))||(d2>T(0));
        return !(n&&pos);
    }

    [[nodiscard]] constexpr Vec3<T> barycentric(Vec2<T> p) const noexcept {
        Vec2<T> v0=b-a,v1=c-a,v2=p-a;
        T d00=dot(v0,v0),d01=dot(v0,v1),d11=dot(v1,v1),d20=dot(v2,v0),d21=dot(v2,v1);
        T inv=T(1)/(d00*d11-d01*d01);
        T v=(d11*d20-d01*d21)*inv, w=(d00*d21-d01*d20)*inv;
        return {T(1)-v-w,v,w};
    }
};

using Line2f = Line2<f32>;     using Segment2f = Segment2<f32>;
using Ray2f  = Ray2<f32>;      using Circle2f  = Circle<f32>;
using Rect2f = Rect2<f32>;     using Triangle2f= Triangle2<f32>;

} // namespace qs::math
