#pragma once
// distance_queries.hpp — Distance between geometric primitives:
//   point-segment, point-triangle, segment-segment, line-line,
//   point-AABB, point-OBB, point-plane, sphere-sphere, etc.
#include "geometry3d.hpp"
#include "geometry2d.hpp"

namespace qs::math {

struct DistResult {
    f32  distance{0.f};
    Vec3f closest_a, closest_b;  // closest points on each primitive
};

// Point to line segment (3D)
[[nodiscard]] inline DistResult dist_point_segment3(Vec3f p, Vec3f a, Vec3f b) noexcept {
    Vec3f ab=b-a, ap=p-a;
    f32 t=saturate(dot(ap,ab)/max(ab.length_sq(),EPS_LENGTH));
    Vec3f closest=a+ab*t;
    return {(p-closest).length(), p, closest};
}

// Point to triangle (3D)
[[nodiscard]] inline DistResult dist_point_triangle3(Vec3f p, Vec3f a, Vec3f b, Vec3f c) noexcept {
    // Voronoi region test
    Vec3f ab=b-a, ac=c-a, ap=p-a;
    f32 d1=dot(ab,ap), d2=dot(ac,ap);
    if(d1<=0&&d2<=0){ return {(p-a).length(),p,a}; }
    Vec3f bp=p-b; f32 d3=dot(ab,bp), d4=dot(ac,bp);
    if(d3>=0&&d4<=d3){ return {(p-b).length(),p,b}; }
    Vec3f cp=p-c; f32 d5=dot(ab,cp), d6=dot(ac,cp);
    if(d6>=0&&d5<=d6){ return {(p-c).length(),p,c}; }
    f32 vc=d1*d4-d3*d2;
    if(vc<=0&&d1>=0&&d3<=0){ f32 v=d1/(d1-d3); Vec3f cl=a+ab*v; return {(p-cl).length(),p,cl}; }
    f32 vb=d5*d2-d1*d6;
    if(vb<=0&&d2>=0&&d6<=0){ f32 w=d2/(d2-d6); Vec3f cl=a+ac*w; return {(p-cl).length(),p,cl}; }
    f32 va=d3*d6-d5*d4;
    if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){
        f32 w=(d4-d3)/((d4-d3)+(d5-d6)); Vec3f cl=b+(c-b)*w; return {(p-cl).length(),p,cl};
    }
    f32 denom=1.f/(va+vb+vc), v=vb*denom, w=vc*denom;
    Vec3f cl=a+ab*v+ac*w;
    return {(p-cl).length(),p,cl};
}

// Segment to segment (3D) — closest points
[[nodiscard]] inline DistResult dist_segment_segment3(Vec3f p1,Vec3f p2,Vec3f p3,Vec3f p4) noexcept {
    Vec3f d1=p2-p1, d2=p4-p3, r=p1-p3;
    f32 a=dot(d1,d1), e=dot(d2,d2), f=dot(d2,r);
    f32 s,t;
    if(a<=EPS_LENGTH&&e<=EPS_LENGTH){s=t=0.f;}
    else if(a<=EPS_LENGTH){t=saturate(f/e);s=0.f;}
    else {
        f32 c=dot(d1,r);
        if(e<=EPS_LENGTH){s=saturate(-c/a);t=0.f;}
        else {
            f32 b=dot(d1,d2),denom=a*e-b*b;
            if(abs(denom)>EPS_LENGTH) s=saturate((b*f-c*e)/denom); else s=0.f;
            t=saturate((b*s+f)/e);
            s=saturate((b*t-c)/a);
        }
    }
    Vec3f ca=p1+d1*s, cb=p3+d2*t;
    return {(ca-cb).length(),ca,cb};
}

// Point to AABB
[[nodiscard]] inline DistResult dist_point_aabb(Vec3f p, AABBf box) noexcept {
    Vec3f cl=p.clamp(box.min_pt,box.max_pt);
    return {(p-cl).length(),p,cl};
}

// Point to OBB
[[nodiscard]] inline DistResult dist_point_obb(Vec3f p, OBBf obb) noexcept {
    Vec3f lp=obb.local(p);
    Vec3f cl_l=lp.clamp(-obb.half_extents,obb.half_extents);
    Vec3f cl=obb.center+obb.axes[0]*cl_l.x+obb.axes[1]*cl_l.y+obb.axes[2]*cl_l.z;
    return {(p-cl).length(),p,cl};
}

// Sphere to sphere
[[nodiscard]] inline f32 dist_sphere_sphere(Spheref a, Spheref b) noexcept {
    return max(0.f,(a.center-b.center).length()-a.radius-b.radius);
}

// Sphere to AABB
[[nodiscard]] inline f32 dist_sphere_aabb(Spheref s, AABBf box) noexcept {
    return max(0.f,dist_point_aabb(s.center,box).distance-s.radius);
}

// Point to plane (signed)
[[nodiscard]] constexpr f32 dist_point_plane(Vec3f p, Planef pl) noexcept {
    return pl.signed_distance(p);
}

// AABB to AABB (separation distance, negative = overlap)
[[nodiscard]] constexpr f32 dist_aabb_aabb(AABBf a, AABBf b) noexcept {
    Vec3f d=max(a.min_pt-b.max_pt,b.min_pt-a.max_pt);
    d=d.max(Vec3f{0.f});
    return d.length();
}

// Capsule to capsule distance
[[nodiscard]] inline f32 dist_capsule_capsule(Capsulef a, Capsulef b) noexcept {
    auto r=dist_segment_segment3(a.a,a.b,b.a,b.b);
    return max(0.f,r.distance-a.radius-b.radius);
}

// GJK support function for a convex shape defined by point cloud
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> support_point_cloud(const Vec3<T>* pts, i32 n, Vec3<T> dir) noexcept {
    T best=T(-1e30); Vec3<T> best_pt=pts[0];
    for(i32 i=0;i<n;++i){
        T d=dot(pts[i],dir);
        if(d>best){best=d;best_pt=pts[i];}
    }
    return best_pt;
}

// Support function for AABB
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> support_aabb(AABBf box, Vec3<T> dir) noexcept {
    return {dir.x>T(0)?box.max_pt.x:box.min_pt.x,
            dir.y>T(0)?box.max_pt.y:box.min_pt.y,
            dir.z>T(0)?box.max_pt.z:box.min_pt.z};
}

// Support function for sphere
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> support_sphere(Sphere<T> s, Vec3<T> dir) noexcept {
    return s.center+dir.normalized()*s.radius;
}

} // namespace qs::math
