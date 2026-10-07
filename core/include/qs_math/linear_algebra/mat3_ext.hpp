#pragma once
// mat3_ext.hpp — Mat3 decomposition, Euler extraction, normal matrix, SVD3 stub,
//               skew-symmetric, orthogonalize, exponential map
#include "mat3.hpp"
#include "quat.hpp"

namespace qs::math {

// Extract Euler angles from rotation matrix (XYZ convention)
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> mat3_to_euler_xyz(Mat3<T> m) noexcept {
    // R = Rx * Ry * Rz
    T sy=m[2].x;
    if(abs(sy)>=T(1)-T(EPS_ANGLE)){
        // Gimbal lock
        T x=atan2(-m[1].z, m[1].y);
        T y=sy>T(0)?T(HALF_PI):-T(HALF_PI);
        return {x,y,T(0)};
    }
    T y=safe_asin(-sy);
    T cy=cos(y);
    T x=atan2(m[2].y/cy, m[2].z/cy);
    T z=atan2(m[1].x/cy, m[0].x/cy);
    return {x,y,z};
}

// Extract Euler (ZYX / Yaw-Pitch-Roll)
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> mat3_to_euler_zyx(Mat3<T> m) noexcept {
    T roll=atan2(m[0].y,m[0].x);
    T pitch=atan2(-m[0].z, sqrt(m[0].y*m[0].y+m[0].x*m[0].x));
    T yaw=atan2(m[1].z,m[2].z);
    return {pitch,yaw,roll};
}

// Skew-symmetric matrix from vector (used for cross product as matrix)
template<FloatScalar T>
[[nodiscard]] constexpr Mat3<T> skew_symmetric(Vec3<T> v) noexcept {
    return {
        {T(0), v.z,-v.y},
        {-v.z,T(0), v.x},
        { v.y,-v.x,T(0)}
    };
}

// Normal matrix (inverse transpose of upper-left 3x3)
template<FloatScalar T>
[[nodiscard]] constexpr Mat3<T> normal_matrix(Mat3<T> model) noexcept {
    return model.inversed().transposed();
}

// Gram-Schmidt orthogonalization of columns
template<FloatScalar T>
[[nodiscard]] inline Mat3<T> orthogonalize_gs(Mat3<T> m) noexcept {
    Vec3<T> c0=m[0].normalized();
    Vec3<T> c1=(m[1]-c0*dot(m[1],c0)).normalized();
    Vec3<T> c2=cross(c0,c1);
    return {c0,c1,c2};
}

// Matrix exponential (via Rodrigues for rotation matrices)
template<FloatScalar T>
[[nodiscard]] inline Mat3<T> mat3_exp(Mat3<T> A) noexcept {
    // Assume A is skew-symmetric → exp(A) is rotation
    Vec3<T> omega{A[1].z-A[2].y, A[2].x-A[0].z, A[0].y-A[1].x};
    omega=omega*T(0.5);
    T angle=omega.length();
    if(angle<T(EPS_ANGLE)) return Mat3<T>::identity();
    Quat<T> q=Quat<T>::from_axis_angle(omega*(T(1)/angle),angle);
    return q.to_mat3();
}

// Matrix logarithm (rotation matrix → skew-symmetric)
template<FloatScalar T>
[[nodiscard]] inline Mat3<T> mat3_log(Mat3<T> R) noexcept {
    T trace=R[0].x+R[1].y+R[2].z;
    T angle=safe_acos(clamp((trace-T(1))*T(0.5),T(-1),T(1)));
    if(abs(angle)<T(EPS_ANGLE)) return Mat3<T>::zero();
    Mat3<T> S=(R-R.transposed())*(angle/(T(2)*sin(angle)));
    return S;
}

// Kabsch algorithm: find optimal rotation between two point sets
template<FloatScalar T>
[[nodiscard]] inline Mat3<T> kabsch_rotation(
    const Vec3<T>* src, const Vec3<T>* dst, i32 n) noexcept
{
    // Compute covariance H = src^T * dst
    Mat3<T> H=Mat3<T>::zero();
    for(i32 i=0;i<n;++i){
        H[0].x+=src[i].x*dst[i].x; H[0].y+=src[i].y*dst[i].x; H[0].z+=src[i].z*dst[i].x;
        H[1].x+=src[i].x*dst[i].y; H[1].y+=src[i].y*dst[i].y; H[1].z+=src[i].z*dst[i].y;
        H[2].x+=src[i].x*dst[i].z; H[2].y+=src[i].y*dst[i].z; H[2].z+=src[i].z*dst[i].z;
    }
    // Approximate SVD via Gram-Schmidt on H^T * H (simplified)
    // Full SVD would require Jacobi iterations — use orthogonalize as approximation
    Mat3<T> U=orthogonalize_gs(H);
    Mat3<T> V=orthogonalize_gs(H.transposed());
    T det=determinant(V*U.transposed());
    Mat3<T> diag=Mat3<T>::scale({T(1),T(1),det>T(0)?T(1):T(-1)});
    return V*diag*U.transposed();
}

} // namespace qs::math
