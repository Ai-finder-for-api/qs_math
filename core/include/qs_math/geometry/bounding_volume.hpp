#pragma once
// qs_math/bounding_volume.hpp — Bounding volume types, hierarchy helpers
#include "geometry3d.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Bounding Sphere fitting
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] Sphere<T> bounding_sphere_naive(const Vec3<T>* pts, i32 count) noexcept {
    if(count<=0) return {Vec3<T>{T(0)},T(0)};
    Vec3<T> center{T(0)};
    for(i32 i=0;i<count;++i) center+=pts[i];
    center=center*(T(1)/T(count));
    T r=T(0);
    for(i32 i=0;i<count;++i) r=max(r,(pts[i]-center).length());
    return {center,r};
}

// Ritter's bounding sphere (fast, slightly loose)
template<FloatScalar T>
[[nodiscard]] Sphere<T> bounding_sphere_ritter(const Vec3<T>* pts, i32 count) noexcept {
    if(count<=0) return {};
    // Pick initial pair
    Vec3<T> min_x=pts[0],max_x=pts[0],min_y=pts[0],max_y=pts[0],min_z=pts[0],max_z=pts[0];
    for(i32 i=1;i<count;++i){
        if(pts[i].x<min_x.x) min_x=pts[i]; if(pts[i].x>max_x.x) max_x=pts[i];
        if(pts[i].y<min_y.y) min_y=pts[i]; if(pts[i].y>max_y.y) max_y=pts[i];
        if(pts[i].z<min_z.z) min_z=pts[i]; if(pts[i].z>max_z.z) max_z=pts[i];
    }
    T dx=(max_x-min_x).length_sq(), dy=(max_y-min_y).length_sq(), dz=(max_z-min_z).length_sq();
    Vec3<T> p1=min_x,p2=max_x;
    if(dy>dx&&dy>dz){p1=min_y;p2=max_y;}
    else if(dz>dx&&dz>dy){p1=min_z;p2=max_z;}
    Vec3<T> center=(p1+p2)*T(0.5);
    T radius=(p2-center).length();
    // Grow sphere to include all points
    for(i32 i=0;i<count;++i){
        Vec3<T> d=pts[i]-center;
        T dist=d.length();
        if(dist>radius){
            T new_r=(radius+dist)*T(0.5);
            center=center+d*((new_r-radius)/dist);
            radius=new_r;
        }
    }
    return {center,radius};
}

// ---------------------------------------------------------------------------
// AABB from points
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr AABB<T> aabb_from_points(const Vec3<T>* pts, i32 count) noexcept {
    if(count<=0) return AABB<T>::empty();
    AABB<T> result{pts[0],pts[0]};
    for(i32 i=1;i<count;++i) result=result.merge(pts[i]);
    return result;
}

// AABB from sphere
template<FloatScalar T>
[[nodiscard]] constexpr AABB<T> aabb_from_sphere(Sphere<T> s) noexcept {
    Vec3<T> r{s.radius}; return {s.center-r, s.center+r};
}

// ---------------------------------------------------------------------------
// BVH Node (for BVH builder helpers)
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct BVHNode {
    AABB<T>  bounds;
    i32      left{-1};    // -1 = leaf
    i32      right{-1};
    i32      prim_start{0};
    i32      prim_count{0};

    [[nodiscard]] constexpr bool is_leaf() const noexcept { return left<0; }
};

// SAH cost for BVH split
template<FloatScalar T>
[[nodiscard]] constexpr T bvh_sah_cost(AABB<T> parent, AABB<T> left, AABB<T> right,
                                         i32 n_left, i32 n_right,
                                         T traversal_cost=T(1), T intersect_cost=T(1)) noexcept {
    T parent_area = parent.surface_area();
    if(parent_area<T(EPS_F32)) return F32_INF;
    return traversal_cost
         + (left.surface_area()/parent_area)*T(n_left)*intersect_cost
         + (right.surface_area()/parent_area)*T(n_right)*intersect_cost;
}

// ---------------------------------------------------------------------------
// Bounding Capsule from AABB
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr Capsule<T> bounding_capsule_from_aabb(AABB<T> box) noexcept {
    Vec3<T> c=box.center();
    Vec3<T> he=box.extents();
    T max_h=max(he.x,max(he.y,he.z));
    T r=max(he.x,he.z);
    T half_h=max(T(0),he.y-r);
    return {c-Vec3<T>{T(0),half_h,T(0)}, c+Vec3<T>{T(0),half_h,T(0)}, r};
}

} // namespace qs::math
