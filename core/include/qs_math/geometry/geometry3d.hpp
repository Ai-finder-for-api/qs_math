#pragma once
// qs_math/geometry3d.hpp — 3D geometric primitives
#include "vec3.hpp"
#include "vec4.hpp"
#include "scalar_math.hpp"
#include <array>

namespace qs::math {

// ---------------------------------------------------------------------------
// Ray3
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Ray3 {
    Vec3<T> origin, direction;
    constexpr Ray3() noexcept = default;
    constexpr Ray3(Vec3<T> o, Vec3<T> d) noexcept : origin(o), direction(d.normalized()) {}
    [[nodiscard]] constexpr Vec3<T> at(T t) const noexcept { return origin + direction*t; }
    [[nodiscard]] constexpr Vec3<T> closest_point(Vec3<T> p) const noexcept {
        T t = max(T(0), dot(p-origin, direction));
        return at(t);
    }
};

// ---------------------------------------------------------------------------
// Plane (ax+by+cz+d=0, normal=(a,b,c))
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Plane {
    Vec3<T> normal;
    T d{T(0)};

    constexpr Plane() noexcept = default;
    constexpr Plane(Vec3<T> n, T d) noexcept : normal(n.normalized()), d(d) {}
    constexpr Plane(Vec3<T> n, Vec3<T> p) noexcept : normal(n.normalized()), d(-dot(n.normalized(),p)) {}

    static Plane from_three_points(Vec3<T> a, Vec3<T> b, Vec3<T> c) noexcept {
        Vec3<T> n = cross(b-a, c-a).normalized();
        return {n, -dot(n,a)};
    }

    [[nodiscard]] constexpr T signed_distance(Vec3<T> p) const noexcept { return dot(normal,p)+d; }
    [[nodiscard]] constexpr T distance(Vec3<T> p)        const noexcept { return abs(signed_distance(p)); }
    [[nodiscard]] constexpr Vec3<T> project(Vec3<T> p)   const noexcept { return p - normal*signed_distance(p); }
    [[nodiscard]] constexpr Vec3<T> reflect(Vec3<T> p)   const noexcept { return p - normal*(T(2)*signed_distance(p)); }
    [[nodiscard]] constexpr bool contains(Vec3<T> p, T eps=T(EPS_F32)) const noexcept {
        return abs(signed_distance(p))<=eps;
    }

    // Ray-plane intersection
    [[nodiscard]] bool intersect_ray(Ray3<T> ray, T& t_out) const noexcept {
        T denom=dot(normal, ray.direction);
        if (abs(denom)<T(EPS_INTERSECT)) return false;
        t_out=-(dot(normal,ray.origin)+d)/denom;
        return t_out>=T(0);
    }

    // Plane-plane intersection → line
    [[nodiscard]] bool intersect_plane(Plane other, Ray3<T>& line_out) const noexcept {
        Vec3<T> dir=cross(normal,other.normal);
        T len=dir.length();
        if (len<T(EPS_INTERSECT)) return false;
        dir=dir*(T(1)/len);
        // Find a point on both planes
        Vec3<T> n1=normal, n2=other.normal;
        T d1=d, d2=other.d;
        T denom=T(1)-dot(n1,n2)*dot(n1,n2);
        if (abs(denom)<T(EPS_INTERSECT)) return false;
        T c1=(-d1+dot(n1,n2)*d2)/denom;
        T c2=(-d2+dot(n1,n2)*d1)/denom; // Note: sign depends on convention
        line_out={n1*c1+n2*c2, dir};
        return true;
    }
};

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Sphere {
    Vec3<T> center;
    T radius{T(0)};

    constexpr Sphere() noexcept = default;
    constexpr Sphere(Vec3<T> c, T r) noexcept : center(c), radius(r) {}

    [[nodiscard]] constexpr T volume()      const noexcept { return T(4)*T(ONE_THIRD)*T(PI)*radius*radius*radius; }
    [[nodiscard]] constexpr T surface_area()const noexcept { return T(4)*T(PI)*radius*radius; }
    [[nodiscard]] constexpr bool contains(Vec3<T> p) const noexcept {
        return (p-center).length_sq()<=radius*radius;
    }
    [[nodiscard]] constexpr bool intersects(Sphere o) const noexcept {
        T rd=radius+o.radius; return (center-o.center).length_sq()<=rd*rd;
    }
    [[nodiscard]] constexpr bool intersects_plane(Plane<T> pl) const noexcept {
        return abs(pl.signed_distance(center))<=radius;
    }
    [[nodiscard]] constexpr Sphere merge(Sphere o) const noexcept {
        Vec3<T> dir=o.center-center;
        T dist=dir.length();
        if (dist+o.radius<=radius) return *this;
        if (dist+radius<=o.radius) return o;
        T nr=(dist+radius+o.radius)*T(0.5);
        Vec3<T> nc=(dist>T(EPS_LENGTH))?(center+dir*((nr-radius)/dist)):center;
        return {nc,nr};
    }
    [[nodiscard]] constexpr Sphere expanded(T amount) const noexcept { return {center, radius+amount}; }
    [[nodiscard]] constexpr Vec3<T> closest_point(Vec3<T> p) const noexcept {
        Vec3<T> dir=(p-center);
        T len=dir.length();
        if(len<T(EPS_LENGTH)) return center;
        return center+dir*(radius/len);
    }
};

// ---------------------------------------------------------------------------
// AABB (Axis-Aligned Bounding Box)
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct AABB {
    Vec3<T> min_pt, max_pt;

    constexpr AABB() noexcept : min_pt{T(1e30f)}, max_pt{T(-1e30f)} {}
    constexpr AABB(Vec3<T> mn, Vec3<T> mx) noexcept : min_pt(mn), max_pt(mx) {}

    static constexpr AABB from_center_size(Vec3<T> c, Vec3<T> sz) noexcept {
        Vec3<T> h=sz*T(0.5); return {c-h,c+h};
    }
    static constexpr AABB empty() noexcept { return {}; }
    static constexpr AABB infinite() noexcept {
        return {Vec3<T>{NumericLimits<T>::min_val},Vec3<T>{NumericLimits<T>::max_val}};
    }

    [[nodiscard]] constexpr Vec3<T> size()    const noexcept { return max_pt-min_pt; }
    [[nodiscard]] constexpr Vec3<T> center()  const noexcept { return (min_pt+max_pt)*T(0.5); }
    [[nodiscard]] constexpr Vec3<T> extents() const noexcept { return size()*T(0.5); }
    [[nodiscard]] constexpr T volume()        const noexcept { Vec3<T> s=size(); return s.x*s.y*s.z; }
    [[nodiscard]] constexpr T surface_area()  const noexcept {
        Vec3<T> s=size(); return T(2)*(s.x*s.y+s.y*s.z+s.x*s.z);
    }
    [[nodiscard]] constexpr bool is_valid()   const noexcept { return min_pt.x<=max_pt.x&&min_pt.y<=max_pt.y&&min_pt.z<=max_pt.z; }

    [[nodiscard]] constexpr bool contains(Vec3<T> p) const noexcept {
        return p.x>=min_pt.x&&p.x<=max_pt.x&&p.y>=min_pt.y&&p.y<=max_pt.y&&p.z>=min_pt.z&&p.z<=max_pt.z;
    }
    [[nodiscard]] constexpr bool contains(AABB o) const noexcept {
        return contains(o.min_pt)&&contains(o.max_pt);
    }
    [[nodiscard]] constexpr bool intersects(AABB o) const noexcept {
        return min_pt.x<=o.max_pt.x&&max_pt.x>=o.min_pt.x&&
               min_pt.y<=o.max_pt.y&&max_pt.y>=o.min_pt.y&&
               min_pt.z<=o.max_pt.z&&max_pt.z>=o.min_pt.z;
    }
    [[nodiscard]] constexpr bool intersects(Sphere<T> s) const noexcept {
        Vec3<T> closest=s.center.clamp(min_pt,max_pt);
        return (closest-s.center).length_sq()<=s.radius*s.radius;
    }
    [[nodiscard]] constexpr AABB merge(AABB o) const noexcept { return {min_pt.min(o.min_pt),max_pt.max(o.max_pt)}; }
    [[nodiscard]] constexpr AABB merge(Vec3<T> p) const noexcept { return {min_pt.min(p),max_pt.max(p)}; }
    [[nodiscard]] constexpr AABB intersection(AABB o) const noexcept { return {min_pt.max(o.min_pt),max_pt.min(o.max_pt)}; }
    [[nodiscard]] constexpr AABB expanded(T amount) const noexcept {
        Vec3<T> a{amount}; return {min_pt-a,max_pt+a};
    }
    [[nodiscard]] constexpr Vec3<T> closest_point(Vec3<T> p) const noexcept { return p.clamp(min_pt,max_pt); }
    [[nodiscard]] constexpr T distance_sq(Vec3<T> p) const noexcept {
        Vec3<T> d=max(min_pt-p,Vec3<T>{T(0)}).max(p-max_pt);
        return d.length_sq();
    }

    // 8 corner points
    [[nodiscard]] constexpr std::array<Vec3<T>,8> corners() const noexcept {
        return {
            Vec3<T>{min_pt.x,min_pt.y,min_pt.z}, Vec3<T>{max_pt.x,min_pt.y,min_pt.z},
            Vec3<T>{min_pt.x,max_pt.y,min_pt.z}, Vec3<T>{max_pt.x,max_pt.y,min_pt.z},
            Vec3<T>{min_pt.x,min_pt.y,max_pt.z}, Vec3<T>{max_pt.x,min_pt.y,max_pt.z},
            Vec3<T>{min_pt.x,max_pt.y,max_pt.z}, Vec3<T>{max_pt.x,max_pt.y,max_pt.z}
        };
    }

    // Longest axis: 0=x, 1=y, 2=z
    [[nodiscard]] constexpr i32 longest_axis() const noexcept {
        Vec3<T> s=size();
        if(s.x>=s.y&&s.x>=s.z) return 0;
        if(s.y>=s.z) return 1;
        return 2;
    }
};

// ---------------------------------------------------------------------------
// OBB (Oriented Bounding Box)
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct OBB {
    Vec3<T> center;
    Vec3<T> axes[3]; // right, up, forward (unit)
    Vec3<T> half_extents;

    constexpr OBB() noexcept = default;
    constexpr OBB(Vec3<T> c, Vec3<T> r, Vec3<T> u, Vec3<T> f, Vec3<T> he) noexcept
        : center(c), axes{r,u,f}, half_extents(he) {}

    static OBB from_aabb(AABB<T> aabb) noexcept {
        return {aabb.center(), Vec3<T>::unit_x(), Vec3<T>::unit_y(), Vec3<T>::unit_z(), aabb.extents()};
    }

    [[nodiscard]] constexpr std::array<Vec3<T>,8> corners() const noexcept {
        Vec3<T> ax=axes[0]*half_extents.x, ay=axes[1]*half_extents.y, az=axes[2]*half_extents.z;
        return {
            center-ax-ay-az, center+ax-ay-az, center-ax+ay-az, center+ax+ay-az,
            center-ax-ay+az, center+ax-ay+az, center-ax+ay+az, center+ax+ay+az
        };
    }

    [[nodiscard]] constexpr AABB<T> to_aabb() const noexcept {
        auto c = corners();
        AABB<T> result{c[0],c[0]};
        for(i32 i=1;i<8;++i) result=result.merge(c[i]);
        return result;
    }

    // Project point into local space
    [[nodiscard]] constexpr Vec3<T> local(Vec3<T> p) const noexcept {
        Vec3<T> d=p-center;
        return {dot(d,axes[0]),dot(d,axes[1]),dot(d,axes[2])};
    }

    [[nodiscard]] constexpr bool contains(Vec3<T> p) const noexcept {
        Vec3<T> lp=local(p);
        return abs(lp.x)<=half_extents.x&&abs(lp.y)<=half_extents.y&&abs(lp.z)<=half_extents.z;
    }
};

// ---------------------------------------------------------------------------
// Capsule
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Capsule {
    Vec3<T> a, b; // segment endpoints
    T radius{T(0)};

    constexpr Capsule() noexcept = default;
    constexpr Capsule(Vec3<T> a, Vec3<T> b, T r) noexcept : a(a),b(b),radius(r) {}

    [[nodiscard]] inline T length() const noexcept { return (b-a).length(); }
    [[nodiscard]] constexpr Vec3<T> center() const noexcept { return (a+b)*T(0.5); }
    [[nodiscard]] constexpr T volume() const noexcept {
        T L=length();
        return T(PI)*radius*radius*(L+T(4)*radius*ONE_THIRD);
    }
    [[nodiscard]] constexpr bool contains(Vec3<T> p) const noexcept;
};

// ---------------------------------------------------------------------------
// Cylinder
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Cylinder {
    Vec3<T> base_center;
    Vec3<T> axis_dir; // normalized
    T radius{T(0)}, height{T(0)};

    constexpr Cylinder() noexcept = default;
    constexpr Cylinder(Vec3<T> base, Vec3<T> dir, T r, T h) noexcept
        : base_center(base), axis_dir(dir.normalized()), radius(r), height(h) {}

    [[nodiscard]] constexpr T volume()       const noexcept { return T(PI)*radius*radius*height; }
    [[nodiscard]] constexpr T surface_area() const noexcept { return T(TWO_PI)*radius*(radius+height); }
    [[nodiscard]] constexpr Vec3<T> top_center() const noexcept { return base_center+axis_dir*height; }
};

// ---------------------------------------------------------------------------
// Cone
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Cone {
    Vec3<T> apex;
    Vec3<T> axis_dir; // normalized, points from apex toward base
    T half_angle{T(0)};
    T height{T(0)};

    constexpr Cone() noexcept = default;
    constexpr Cone(Vec3<T> apex, Vec3<T> dir, T half_angle, T h) noexcept
        : apex(apex), axis_dir(dir.normalized()), half_angle(half_angle), height(h) {}

    [[nodiscard]] inline T base_radius() const noexcept { return height*tan(half_angle); }
    [[nodiscard]] inline T volume()      const noexcept { T r=base_radius(); return T(ONE_THIRD)*T(PI)*r*r*height; }

    [[nodiscard]] inline bool contains(Vec3<T> p) const noexcept {
        Vec3<T> dp=p-apex;
        T proj=dot(dp,axis_dir);
        if(proj<T(0)||proj>height) return false;
        T cone_r=proj*tan(half_angle);
        Vec3<T> perp=dp-axis_dir*proj;
        return perp.length_sq()<=cone_r*cone_r;
    }
};

// ---------------------------------------------------------------------------
// Triangle3
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Triangle3 {
    Vec3<T> a, b, c;

    constexpr Triangle3() noexcept = default;
    constexpr Triangle3(Vec3<T> a, Vec3<T> b, Vec3<T> c) noexcept : a(a),b(b),c(c) {}

    [[nodiscard]] constexpr Vec3<T> normal()   const noexcept { return cross(b-a,c-a).normalized(); }
    [[nodiscard]] constexpr Vec3<T> centroid() const noexcept { return (a+b+c)*T(ONE_THIRD); }
    [[nodiscard]] constexpr T area()           const noexcept { return cross(b-a,c-a).length()*T(0.5); }
    [[nodiscard]] constexpr Plane<T> plane()   const noexcept { return Plane<T>::from_three_points(a,b,c); }

    [[nodiscard]] constexpr Vec3<T> barycentric(Vec3<T> p) const noexcept {
        Vec3<T> v0=b-a,v1=c-a,v2=p-a;
        T d00=dot(v0,v0),d01=dot(v0,v1),d11=dot(v1,v1),d20=dot(v2,v0),d21=dot(v2,v1);
        T inv=T(1)/(d00*d11-d01*d01);
        T v=(d11*d20-d01*d21)*inv, w=(d00*d21-d01*d20)*inv;
        return {T(1)-v-w,v,w};
    }

    [[nodiscard]] constexpr bool contains_point(Vec3<T> p, T eps=T(EPS_F32)) const noexcept {
        Vec3<T> bary=barycentric(p);
        return bary.x>=-eps&&bary.y>=-eps&&bary.z>=-eps;
    }

    [[nodiscard]] constexpr Vec3<T> closest_point(Vec3<T> p) const noexcept {
        Vec3<T> bary=barycentric(p);
        if(bary.x>=T(0)&&bary.y>=T(0)&&bary.z>=T(0)) return a*bary.x+b*bary.y+c*bary.z;
        // Clamp to edges
        T u=saturate(bary.y), v=saturate(bary.z), w=saturate(bary.x);
        return a*w+b*u+c*v; // simplified
    }

    [[nodiscard]] constexpr AABB<T> aabb() const noexcept {
        return {a.min(b).min(c), a.max(b).max(c)};
    }
};

// ---------------------------------------------------------------------------
// Frustum (6 planes: left, right, bottom, top, near, far)
// ---------------------------------------------------------------------------
template<FloatScalar T>
struct Frustum {
    Plane<T> planes[6]; // L R B T N F

    static Frustum from_mat4(Mat4<T> vp) noexcept;  // defined in intersection.hpp

    [[nodiscard]] bool contains_point(Vec3<T> p) const noexcept {
        for(i32 i=0;i<6;++i) if(planes[i].signed_distance(p)<T(0)) return false;
        return true;
    }
    [[nodiscard]] bool intersects_sphere(Sphere<T> s) const noexcept {
        for(i32 i=0;i<6;++i) if(planes[i].signed_distance(s.center)<-s.radius) return false;
        return true;
    }
    [[nodiscard]] bool intersects_aabb(AABB<T> aabb) const noexcept {
        for(i32 i=0;i<6;++i){
            Vec3<T> n=planes[i].normal;
            Vec3<T> p=aabb.min_pt;
            if(n.x>=T(0)) p.x=aabb.max_pt.x;
            if(n.y>=T(0)) p.y=aabb.max_pt.y;
            if(n.z>=T(0)) p.z=aabb.max_pt.z;
            if(planes[i].signed_distance(p)<T(0)) return false;
        }
        return true;
    }
};

using Ray3f      = Ray3<f32>;     using Ray3d      = Ray3<f64>;
using Planef     = Plane<f32>;    using Planed     = Plane<f64>;
using Spheref    = Sphere<f32>;   using Sphered    = Sphere<f64>;
using AABBf      = AABB<f32>;     using AABBd      = AABB<f64>;
using OBBf       = OBB<f32>;      using OBBd       = OBB<f64>;
using Capsulef   = Capsule<f32>;
using Triangle3f = Triangle3<f32>;using Triangle3d = Triangle3<f64>;
using Frustumf   = Frustum<f32>;  using Frustumd   = Frustum<f64>;

} // namespace qs::math
