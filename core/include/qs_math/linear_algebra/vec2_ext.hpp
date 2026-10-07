#pragma once
// vec2_ext.hpp — Extended Vec2 operations: polar, complex multiply,
//               perpendicular, winding, signed angle, closest segment point,
//               triangle tests, convex hull helpers, min bounding circle
#include "vec2.hpp"
#include "trig.hpp"
#include "scalar_math.hpp"
#include <array>

namespace qs::math {

// Complex multiplication (treat Vec2 as complex number)
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> complex_mul(Vec2<T> a, Vec2<T> b) noexcept {
    return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> complex_div(Vec2<T> a, Vec2<T> b) noexcept {
    T denom = b.x*b.x + b.y*b.y;
    return {(a.x*b.x+a.y*b.y)/denom, (a.y*b.x-a.x*b.y)/denom};
}
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> complex_conj(Vec2<T> v) noexcept { return {v.x,-v.y}; }
template<FloatScalar T>
[[nodiscard]] inline Vec2<T> complex_exp(Vec2<T> v) noexcept {
    T e=exp(v.x); return {e*cos(v.y), e*sin(v.y)};
}

// Rotate 90/180/270
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> rot90_ccw(Vec2<T> v)  noexcept { return {-v.y, v.x}; }
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> rot90_cw(Vec2<T> v)   noexcept { return {v.y, -v.x}; }
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> rot180(Vec2<T> v)     noexcept { return {-v.x,-v.y}; }

// Winding order of triangle (positive = CCW)
template<FloatScalar T>
[[nodiscard]] constexpr T signed_area_2d(Vec2<T> a, Vec2<T> b, Vec2<T> c) noexcept {
    return (b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y);
}
template<FloatScalar T>
[[nodiscard]] constexpr bool is_ccw(Vec2<T> a, Vec2<T> b, Vec2<T> c) noexcept {
    return signed_area_2d(a,b,c) > T(0);
}

// Point in convex polygon (n vertices, CCW wound)
template<FloatScalar T>
[[nodiscard]] constexpr bool point_in_convex_polygon(Vec2<T> p, const Vec2<T>* verts, i32 n) noexcept {
    for(i32 i=0;i<n;++i){
        Vec2<T> a=verts[i], b=verts[(i+1)%n];
        if((b-a).cross(p-a) < T(0)) return false;
    }
    return true;
}

// Closest point on line segment [a,b] to p
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> closest_point_segment(Vec2<T> a, Vec2<T> b, Vec2<T> p) noexcept {
    Vec2<T> ab=b-a;
    T t=saturate(dot(p-a,ab)/max(ab.length_sq(),T(EPS_LENGTH)));
    return a+ab*t;
}

// Minimum bounding circle of 3 points (circumscribed)
template<FloatScalar T>
[[nodiscard]] bool circumscribed_circle(Vec2<T> a, Vec2<T> b, Vec2<T> c,
                                         Vec2<T>& center, T& radius) noexcept {
    T ax=b.x-a.x, ay=b.y-a.y;
    T bx=c.x-a.x, by=c.y-a.y;
    T D=T(2)*(ax*by-ay*bx);
    if(abs(D)<T(EPS_INTERSECT)) return false;
    T ux=((ax*ax+ay*ay)*by-(bx*bx+by*by)*ay)/D;
    T uy=((ax*ax+ay*ay)*bx-(bx*bx+by*by)*ax)/D; // was -ux fix
    uy = ((bx*bx+by*by)*ax-(ax*ax+ay*ay)*bx)/D;
    center={a.x+ux,a.y+uy};
    radius=(center-a).length();
    return true;
}

// Graham scan convex hull (modifies input — must be >= 3 points)
// Returns hull size; hull written to out (must be >= n)
template<FloatScalar T>
[[nodiscard]] i32 convex_hull_2d(Vec2<T>* pts, i32 n, Vec2<T>* hull) noexcept {
    if(n<3) return n;
    // Find lowest-y (then leftmost) point
    i32 pivot=0;
    for(i32 i=1;i<n;++i)
        if(pts[i].y<pts[pivot].y||(pts[i].y==pts[pivot].y&&pts[i].x<pts[pivot].x)) pivot=i;
    Vec2<T> tmp=pts[0]; pts[0]=pts[pivot]; pts[pivot]=tmp;
    Vec2<T> origin=pts[0];
    // Sort by polar angle (simple insertion sort for correctness)
    for(i32 i=2;i<n;++i){
        Vec2<T> key=pts[i]; i32 j=i-1;
        while(j>=1){
            T cross=(pts[j].x-origin.x)*(key.y-origin.y)-(pts[j].y-origin.y)*(key.x-origin.x);
            if(cross<T(0)) { pts[j+1]=pts[j]; --j; } else break;
        }
        pts[j+1]=key;
    }
    // Build hull
    i32 sz=0;
    for(i32 i=0;i<n;++i){
        while(sz>=2&&signed_area_2d(hull[sz-2],hull[sz-1],pts[i])<=T(0)) --sz;
        hull[sz++]=pts[i];
    }
    return sz;
}

// Catmull-Rom tangent for smooth 2D paths
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> catmull_tangent(Vec2<T> prev, Vec2<T> next) noexcept {
    return (next-prev)*T(0.5);
}

// Point reflection across line through origin with direction d
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> reflect_across_line(Vec2<T> p, Vec2<T> d) noexcept {
    Vec2<T> dn=d.normalized();
    return p - dn*(T(2)*dot(p,dn));
}

// Bilinear coordinates in quad (p0,p1,p2,p3)
template<FloatScalar T>
[[nodiscard]] bool bilinear_coords_quad(Vec2<T> p0,Vec2<T> p1,Vec2<T> p2,Vec2<T> p3,
                                         Vec2<T> p, T& u, T& v) noexcept {
    // Solve iteratively
    u=v=T(0.5);
    for(i32 iter=0;iter<8;++iter){
        Vec2<T> q=lerp(lerp(p0,p1,u),lerp(p3,p2,u),v);
        Vec2<T> err=q-p;
        if(err.length_sq()<T(EPS_INTERSECT*EPS_INTERSECT)) return true;
        Vec2<T> du=lerp(p1-p0,p2-p3,v), dv=lerp(p3-p0,p2-p1,u);
        T det=cross(du,dv);
        if(abs(det)<T(EPS_INTERSECT)) return false;
        u-=cross(err,dv)/det;
        v-=cross(du,err)/det;
    }
    return true;
}

} // namespace qs::math
