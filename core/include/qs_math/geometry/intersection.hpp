#pragma once
// qs_math/intersection.hpp — All intersection tests
#include "geometry3d.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Ray vs Triangle (Möller–Trumbore)
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct HitResult {
    bool  hit{false};
    T     t{T(0)};
    Vec3<T> point;
    Vec3<T> normal;
    T     u{T(0)}, v{T(0)};
};

template<FloatScalar T>
[[nodiscard]] HitResult<T> ray_triangle(Ray3<T> ray, Triangle3<T> tri, bool cull_back=false) noexcept {
    HitResult<T> result;
    Vec3<T> e1=tri.b-tri.a, e2=tri.c-tri.a;
    Vec3<T> h=cross(ray.direction,e2);
    T a=dot(e1,h);
    if(cull_back&&a<T(EPS_INTERSECT)) return result;
    if(abs(a)<T(EPS_INTERSECT)) return result;
    T f=T(1)/a;
    Vec3<T> s=ray.origin-tri.a;
    T u=f*dot(s,h);
    if(u<T(0)||u>T(1)) return result;
    Vec3<T> q=cross(s,e1);
    T v=f*dot(ray.direction,q);
    if(v<T(0)||u+v>T(1)) return result;
    T t=f*dot(e2,q);
    if(t<T(EPS_INTERSECT)) return result;
    result.hit=true; result.t=t; result.u=u; result.v=v;
    result.point=ray.at(t); result.normal=tri.normal();
    return result;
}

// ---------------------------------------------------------------------------
// Ray vs AABB (slab method)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] HitResult<T> ray_aabb(Ray3<T> ray, AABB<T> box) noexcept {
    HitResult<T> result;
    Vec3<T> inv{T(1)/ray.direction.x, T(1)/ray.direction.y, T(1)/ray.direction.z};
    T t1=(box.min_pt.x-ray.origin.x)*inv.x, t2=(box.max_pt.x-ray.origin.x)*inv.x;
    T t3=(box.min_pt.y-ray.origin.y)*inv.y, t4=(box.max_pt.y-ray.origin.y)*inv.y;
    T t5=(box.min_pt.z-ray.origin.z)*inv.z, t6=(box.max_pt.z-ray.origin.z)*inv.z;
    T tmin=max(max(min(t1,t2),min(t3,t4)),min(t5,t6));
    T tmax=min(min(max(t1,t2),max(t3,t4)),max(t5,t6));
    if(tmax<T(0)||tmin>tmax) return result;
    T t=tmin<T(0)?tmax:tmin;
    result.hit=true; result.t=t; result.point=ray.at(t);
    // Compute face normal
    Vec3<T> c=result.point-box.center();
    Vec3<T> h=box.extents();
    T ax=abs(c.x/h.x), ay=abs(c.y/h.y), az=abs(c.z/h.z);
    if(ax>=ay&&ax>=az)      result.normal={sign(c.x),T(0),T(0)};
    else if(ay>=ax&&ay>=az) result.normal={T(0),sign(c.y),T(0)};
    else                    result.normal={T(0),T(0),sign(c.z)};
    return result;
}

// ---------------------------------------------------------------------------
// Ray vs OBB
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] HitResult<T> ray_obb(Ray3<T> ray, OBB<T> obb) noexcept {
    // Transform ray to OBB local space
    Vec3<T> d=ray.origin-obb.center;
    Vec3<T> lo{dot(d,obb.axes[0]),dot(d,obb.axes[1]),dot(d,obb.axes[2])};
    Vec3<T> ld{dot(ray.direction,obb.axes[0]),dot(ray.direction,obb.axes[1]),dot(ray.direction,obb.axes[2])};
    Ray3<T> local_ray{lo,ld};
    AABB<T> local_box{-obb.half_extents, obb.half_extents};
    HitResult<T> r=ray_aabb(local_ray,local_box);
    if(r.hit){
        r.point=ray.at(r.t);
        r.normal=obb.axes[0]*r.normal.x+obb.axes[1]*r.normal.y+obb.axes[2]*r.normal.z;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Ray vs Sphere
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] HitResult<T> ray_sphere(Ray3<T> ray, Sphere<T> sphere) noexcept {
    HitResult<T> result;
    Vec3<T> oc=ray.origin-sphere.center;
    T a=dot(ray.direction,ray.direction);
    T b=T(2)*dot(oc,ray.direction);
    T c=dot(oc,oc)-sphere.radius*sphere.radius;
    T disc=b*b-T(4)*a*c;
    if(disc<T(0)) return result;
    T sq=sqrt(disc);
    T t=(-b-sq)/(T(2)*a);
    if(t<T(EPS_INTERSECT)) t=(-b+sq)/(T(2)*a);
    if(t<T(EPS_INTERSECT)) return result;
    result.hit=true; result.t=t; result.point=ray.at(t);
    result.normal=(result.point-sphere.center)*(T(1)/sphere.radius);
    return result;
}

// ---------------------------------------------------------------------------
// Ray vs Plane
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] HitResult<T> ray_plane(Ray3<T> ray, Plane<T> plane) noexcept {
    HitResult<T> result;
    T denom=dot(plane.normal,ray.direction);
    if(abs(denom)<T(EPS_INTERSECT)) return result;
    T t=-(dot(plane.normal,ray.origin)+plane.d)/denom;
    if(t<T(0)) return result;
    result.hit=true; result.t=t; result.point=ray.at(t); result.normal=plane.normal;
    return result;
}

// ---------------------------------------------------------------------------
// Ray vs Capsule
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] HitResult<T> ray_capsule(Ray3<T> ray, Capsule<T> cap) noexcept {
    HitResult<T> result;
    Vec3<T> ab=cap.b-cap.a, ao=ray.origin-cap.a;
    T ab2=dot(ab,ab), ab_rd=dot(ab,ray.direction), ab_ao=dot(ab,ao);
    T rd2=dot(ray.direction,ray.direction), rd_ao=dot(ray.direction,ao), ao2=dot(ao,ao);
    T m=ab2*rd2-ab_rd*ab_rd, n=ab2*rd_ao-ab_rd*ab_ao, o=ab2*(ao2-cap.radius*cap.radius)-ab_ao*ab_ao;
    if(abs(m)<T(EPS_INTERSECT)){
        // Ray parallel to capsule axis — check end caps
        // simplified: check against sphere at a and b
        auto ra=ray_sphere(ray, Sphere<T>{cap.a,cap.radius});
        auto rb=ray_sphere(ray, Sphere<T>{cap.b,cap.radius});
        if(!ra.hit&&!rb.hit) return result;
        if(!ra.hit) return rb; if(!rb.hit) return ra;
        return ra.t<rb.t?ra:rb;
    }
    T disc=n*n-m*o;
    if(disc<T(0)) return result;
    T t=(-n-sqrt(disc))/m;
    if(t<T(0)) return result;
    // Check if hit is within cylinder segment
    T y=ab_ao+t*ab_rd;
    if(y>=T(0)&&y<=ab2){
        result.hit=true; result.t=t; result.point=ray.at(t);
        result.normal=((result.point-(cap.a+ab*(y/ab2)))*(T(1)/cap.radius));
    } else {
        // Hit end caps
        auto ra=ray_sphere(ray,Sphere<T>{cap.a,cap.radius});
        auto rb=ray_sphere(ray,Sphere<T>{cap.b,cap.radius});
        if(ra.hit&&(!rb.hit||ra.t<rb.t)) result=ra;
        else if(rb.hit) result=rb;
    }
    return result;
}

// ---------------------------------------------------------------------------
// AABB vs AABB
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr bool aabb_aabb(AABB<T> a, AABB<T> b) noexcept { return a.intersects(b); }

// AABB vs AABB overlap vector
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> aabb_aabb_overlap(AABB<T> a, AABB<T> b) noexcept {
    Vec3<T> overlap{
        min(a.max_pt.x,b.max_pt.x)-max(a.min_pt.x,b.min_pt.x),
        min(a.max_pt.y,b.max_pt.y)-max(a.min_pt.y,b.min_pt.y),
        min(a.max_pt.z,b.max_pt.z)-max(a.min_pt.z,b.min_pt.z)
    };
    return overlap;
}

// ---------------------------------------------------------------------------
// Sphere vs Sphere
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr bool sphere_sphere(Sphere<T> a, Sphere<T> b) noexcept { return a.intersects(b); }

// ---------------------------------------------------------------------------
// Sphere vs AABB
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr bool sphere_aabb(Sphere<T> s, AABB<T> box) noexcept { return box.intersects(s); }

// ---------------------------------------------------------------------------
// Capsule vs Capsule (GJK simplified for capsules)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] bool capsule_capsule(Capsule<T> a, Capsule<T> b) noexcept {
    // Closest point between two segments
    Vec3<T> d1=a.b-a.a, d2=b.b-b.a, r=a.a-b.a;
    T sqlen1=dot(d1,d1), sqlen2=dot(d2,d2), f=dot(d2,r);
    T s,t;
    if(sqlen1<T(EPS_LENGTH)&&sqlen2<T(EPS_LENGTH)){s=t=T(0);}
    else if(sqlen1<T(EPS_LENGTH)){s=T(0);t=saturate(f/sqlen2);}
    else {
        T e=dot(d1,r);
        if(sqlen2<T(EPS_LENGTH)){t=T(0);s=saturate(-e/sqlen1);}
        else {
            T b_=dot(d1,d2),denom=sqlen1*sqlen2-b_*b_;
            if(abs(denom)>T(EPS_LENGTH)) s=saturate((b_*f-e*sqlen2)/denom); else s=T(0);
            t=saturate((b_*s+f)/sqlen2);
            s=saturate((t*b_-e)/sqlen1);
        }
    }
    Vec3<T> cp1=a.a+d1*s, cp2=b.a+d2*t;
    T radsum=a.radius+b.radius;
    return (cp1-cp2).length_sq()<=radsum*radsum;
}

// ---------------------------------------------------------------------------
// Plane vs Plane (always intersect unless parallel)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] bool plane_plane(Plane<T> a, Plane<T> b, Ray3<T>& line_out) noexcept {
    return a.intersect_plane(b, line_out);
}

// ---------------------------------------------------------------------------
// Frustum from ViewProjection matrix
// ---------------------------------------------------------------------------
template<FloatScalar T>
Frustum<T> Frustum<T>::from_mat4(Mat4<T> vp) noexcept {
    Frustum<T> f;
    // Left
    f.planes[0]={Vec3<T>{vp.at(3,0)+vp.at(0,0),vp.at(3,1)+vp.at(0,1),vp.at(3,2)+vp.at(0,2)}.normalized(),
                 vp.at(3,3)+vp.at(0,3)};
    // Right
    f.planes[1]={Vec3<T>{vp.at(3,0)-vp.at(0,0),vp.at(3,1)-vp.at(0,1),vp.at(3,2)-vp.at(0,2)}.normalized(),
                 vp.at(3,3)-vp.at(0,3)};
    // Bottom
    f.planes[2]={Vec3<T>{vp.at(3,0)+vp.at(1,0),vp.at(3,1)+vp.at(1,1),vp.at(3,2)+vp.at(1,2)}.normalized(),
                 vp.at(3,3)+vp.at(1,3)};
    // Top
    f.planes[3]={Vec3<T>{vp.at(3,0)-vp.at(1,0),vp.at(3,1)-vp.at(1,1),vp.at(3,2)-vp.at(1,2)}.normalized(),
                 vp.at(3,3)-vp.at(1,3)};
    // Near
    f.planes[4]={Vec3<T>{vp.at(3,0)+vp.at(2,0),vp.at(3,1)+vp.at(2,1),vp.at(3,2)+vp.at(2,2)}.normalized(),
                 vp.at(3,3)+vp.at(2,3)};
    // Far
    f.planes[5]={Vec3<T>{vp.at(3,0)-vp.at(2,0),vp.at(3,1)-vp.at(2,1),vp.at(3,2)-vp.at(2,2)}.normalized(),
                 vp.at(3,3)-vp.at(2,3)};
    return f;
}

} // namespace qs::math
