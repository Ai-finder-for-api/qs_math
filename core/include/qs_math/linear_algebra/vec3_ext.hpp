#pragma once
// vec3_ext.hpp — Extended Vec3 ops: orthonormal basis, tensor product,
//               triple product, solid angle, spherical coords, Rodrigues rotation,
//               curve framing (Frenet), winding helpers, volume helpers
#include "vec3.hpp"
#include "trig.hpp"
#include "coordinate.hpp"

namespace qs::math {

// Orthonormal basis from single vector (Frisvad / Duff method)
template<FloatScalar T>
inline void orthonormal_basis(Vec3<T> n, Vec3<T>& t, Vec3<T>& b) noexcept {
    // Duff et al. 2017 — numerically stable
    T sign = n.z >= T(0) ? T(1) : T(-1);
    T a = T(-1) / (sign + n.z);
    T c = n.x * n.y * a;
    t = {T(1) + sign * n.x * n.x * a, sign * c, -sign * n.x};
    b = {c, sign + n.y * n.y * a, -n.y};
}

// Tensor (outer) product of two Vec3 → Mat3
template<FloatScalar T>
[[nodiscard]] constexpr Mat3<T> outer_product(Vec3<T> a, Vec3<T> b) noexcept {
    return {
        Vec3<T>{a.x*b.x, a.y*b.x, a.z*b.x},
        Vec3<T>{a.x*b.y, a.y*b.y, a.z*b.y},
        Vec3<T>{a.x*b.z, a.y*b.z, a.z*b.z}
    };
}

// Rodrigues rotation formula: rotate v around axis by angle
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> rodrigues(Vec3<T> v, Vec3<T> axis, T angle_rad) noexcept {
    Vec3<T> k=axis.normalized();
    T c=cos(angle_rad), s=sin(angle_rad);
    return v*c + cross(k,v)*s + k*dot(k,v)*(T(1)-c);
}

// Solid angle subtended by triangle at origin
template<FloatScalar T>
[[nodiscard]] inline T solid_angle(Vec3<T> a, Vec3<T> b, Vec3<T> c) noexcept {
    a=a.normalized(); b=b.normalized(); c=c.normalized();
    T num=abs(dot(a,cross(b,c)));
    T denom=T(1)+dot(a,b)+dot(b,c)+dot(a,c);
    return T(2)*atan2(num,denom);
}

// Spherical to Cartesian and back
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> spherical_to_cartesian(T theta, T phi) noexcept {
    T st=sin(theta);
    return {st*cos(phi), st*sin(phi), cos(theta)};
}
template<FloatScalar T>
[[nodiscard]] inline void cartesian_to_spherical(Vec3<T> v, T& theta, T& phi) noexcept {
    v=v.normalized();
    theta=safe_acos(v.z);
    phi=atan2(v.y,v.x);
}

// Frenet–Serret frame along curve (T=tangent, N=normal, B=binormal)
template<FloatScalar T>
struct FrenetFrame {
    Vec3<T> T_,N,B;
};
template<FloatScalar T>
[[nodiscard]] inline FrenetFrame<T> frenet_frame(Vec3<T> tangent, Vec3<T> dtangent) noexcept {
    Vec3<T> t=tangent.normalized();
    Vec3<T> n=(dtangent-t*dot(dtangent,t)).normalized();
    if(n.is_zero()) {
        Vec3<T> perp{T(1),T(0),T(0)};
        if(abs(t.x)>T(0.9)) perp={T(0),T(1),T(0)};
        n=cross(t,perp).normalized();
    }
    Vec3<T> b=cross(t,n);
    return {t,n,b};
}

// Curvature of a curve: |T' × T| / |T|^3
template<FloatScalar T>
[[nodiscard]] inline T curvature(Vec3<T> tangent, Vec3<T> dtangent) noexcept {
    T len=tangent.length();
    if(len<T(EPS_LENGTH)) return T(0);
    return cross(tangent,dtangent).length()/(len*len*len);
}

// Torsion of a curve
template<FloatScalar T>
[[nodiscard]] inline T torsion(Vec3<T> tangent, Vec3<T> dtangent, Vec3<T> d2tangent) noexcept {
    Vec3<T> cp=cross(tangent,dtangent);
    T cpl=cp.length_sq();
    if(cpl<T(EPS_LENGTH)) return T(0);
    return dot(cp,d2tangent)/cpl;
}

// Lambertian (cosine-weighted) direction from surface normal
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> cosine_hemisphere_sample(Vec3<T> n, T u1, T u2) noexcept {
    T phi=T(TWO_PI)*u2;
    T r=sqrt(u1);
    T x=r*cos(phi), y=r*sin(phi), z=sqrt(max(T(0),T(1)-u1));
    Vec3<T> t,b; orthonormal_basis(n,t,b);
    return t*x+b*y+n*z;
}

// GGX microfacet half-vector sample
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> ggx_sample(Vec3<T> n, T roughness, T u1, T u2) noexcept {
    T a=roughness*roughness;
    T phi=T(TWO_PI)*u2;
    T cos_theta=sqrt((T(1)-u1)/(T(1)+(a*a-T(1))*u1));
    T sin_theta=sqrt(max(T(0),T(1)-cos_theta*cos_theta));
    Vec3<T> h{sin_theta*cos(phi), sin_theta*sin(phi), cos_theta};
    Vec3<T> t,b; orthonormal_basis(n,t,b);
    return (t*h.x+b*h.y+n*h.z).normalized();
}

// Gram-Schmidt orthogonalize v against n
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> orthogonalize(Vec3<T> v, Vec3<T> n) noexcept {
    return (v - n*dot(v,n)).normalized();
}

// Transform normal by inverse-transpose of matrix
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> transform_normal(Vec3<T> normal, Mat3<T> inv_transpose) noexcept {
    return (inv_transpose * normal).normalized();
}

// Volume of tetrahedron (a,b,c,d)
template<FloatScalar T>
[[nodiscard]] constexpr T tet_volume(Vec3<T> a,Vec3<T> b,Vec3<T> c,Vec3<T> d) noexcept {
    return abs(dot(b-a,cross(c-a,d-a)))/T(6);
}

// Centroid of tetrahedron
template<FloatScalar T>
[[nodiscard]] constexpr Vec3<T> tet_centroid(Vec3<T> a,Vec3<T> b,Vec3<T> c,Vec3<T> d) noexcept {
    return (a+b+c+d)*T(0.25);
}

} // namespace qs::math
