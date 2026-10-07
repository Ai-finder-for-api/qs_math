#pragma once
// qs_math/mat3.hpp — 3x3 column-major matrix
#include "vec3.hpp"
#include "angle.hpp"

namespace qs::math {

template<FloatScalar T>
struct Mat4;  // forward

template<FloatScalar T>
struct Mat3 {
    Vec3<T> col[3];

    constexpr Mat3() noexcept : col{{T(1),T(0),T(0)},{T(0),T(1),T(0)},{T(0),T(0),T(1)}} {}
    constexpr Mat3(Vec3<T> c0, Vec3<T> c1, Vec3<T> c2) noexcept : col{c0,c1,c2} {}
    constexpr Mat3(T m00,T m10,T m20, T m01,T m11,T m21, T m02,T m12,T m22) noexcept
        : col{{m00,m10,m20},{m01,m11,m21},{m02,m12,m22}} {}

    [[nodiscard]] constexpr Vec3<T>  operator[](i32 i) const noexcept { return col[i]; }
    [[nodiscard]] constexpr Vec3<T>& operator[](i32 i)       noexcept { return col[i]; }
    [[nodiscard]] constexpr T  at(i32 r, i32 c) const noexcept { return col[c][r]; }
    [[nodiscard]] constexpr T& at(i32 r, i32 c)       noexcept { return col[c][r]; }

    constexpr Vec3<T> row(i32 r) const noexcept { return {col[0][r],col[1][r],col[2][r]}; }

    constexpr Mat3 operator+(Mat3 o) const noexcept { return {col[0]+o.col[0],col[1]+o.col[1],col[2]+o.col[2]}; }
    constexpr Mat3 operator-(Mat3 o) const noexcept { return {col[0]-o.col[0],col[1]-o.col[1],col[2]-o.col[2]}; }
    constexpr Mat3 operator*(T s)    const noexcept { return {col[0]*s,col[1]*s,col[2]*s}; }

    constexpr Vec3<T> operator*(Vec3<T> v) const noexcept {
        return col[0]*v.x + col[1]*v.y + col[2]*v.z;
    }

    constexpr Mat3 operator*(Mat3 b) const noexcept {
        return {(*this)*b.col[0], (*this)*b.col[1], (*this)*b.col[2]};
    }

    [[nodiscard]] constexpr T determinant() const noexcept {
        return col[0].x*(col[1].y*col[2].z-col[2].y*col[1].z)
              -col[1].x*(col[0].y*col[2].z-col[2].y*col[0].z)
              +col[2].x*(col[0].y*col[1].z-col[1].y*col[0].z);
    }

    [[nodiscard]] constexpr Mat3 transposed() const noexcept {
        return {{col[0].x,col[1].x,col[2].x},
                {col[0].y,col[1].y,col[2].y},
                {col[0].z,col[1].z,col[2].z}};
    }

    [[nodiscard]] constexpr Mat3 cofactor() const noexcept {
        return {
            { col[1].y*col[2].z-col[2].y*col[1].z, -(col[0].y*col[2].z-col[2].y*col[0].z),  col[0].y*col[1].z-col[1].y*col[0].z },
            {-(col[1].x*col[2].z-col[2].x*col[1].z), col[0].x*col[2].z-col[2].x*col[0].z, -(col[0].x*col[1].z-col[1].x*col[0].z)},
            { col[1].x*col[2].y-col[2].x*col[1].y, -(col[0].x*col[2].y-col[2].x*col[0].y),  col[0].x*col[1].y-col[1].x*col[0].y }
        };
    }

    [[nodiscard]] constexpr Mat3 adjoint() const noexcept { return cofactor().transposed(); }

    [[nodiscard]] constexpr Mat3 inversed() const noexcept {
        T det = determinant();
        if (qs::math::abs(det) < T(EPS_DETERMINANT)) return identity();
        return adjoint() * (T(1)/det);
    }

    // Extract scale
    [[nodiscard]] constexpr Vec3<T> scale() const noexcept {
        return {col[0].length(), col[1].length(), col[2].length()};
    }

    static constexpr Mat3 identity() noexcept { return {}; }
    static constexpr Mat3 zero()     noexcept { return {{T(0),T(0),T(0)},{T(0),T(0),T(0)},{T(0),T(0),T(0)}}; }

    static constexpr Mat3 scale(T sx,T sy,T sz) noexcept {
        Mat3 m; m.col[0].x=sx; m.col[1].y=sy; m.col[2].z=sz; return m;
    }
    static constexpr Mat3 scale(Vec3<T> s) noexcept { return scale(s.x,s.y,s.z); }

    static inline Mat3 rotation_x(T a) noexcept {
        T c=qs::math::cos(a),s=qs::math::sin(a);
        return {{T(1),T(0),T(0)},{T(0),c,s},{T(0),-s,c}};
    }
    static inline Mat3 rotation_y(T a) noexcept {
        T c=qs::math::cos(a),s=qs::math::sin(a);
        return {{c,T(0),-s},{T(0),T(1),T(0)},{s,T(0),c}};
    }
    static inline Mat3 rotation_z(T a) noexcept {
        T c=qs::math::cos(a),s=qs::math::sin(a);
        return {{c,s,T(0)},{-s,c,T(0)},{T(0),T(0),T(1)}};
    }
    static inline Mat3 rotation_axis_angle(Vec3<T> axis, T angle) noexcept {
        axis = axis.normalized();
        T c=qs::math::cos(angle), s=qs::math::sin(angle), t=T(1)-c;
        T x=axis.x, y=axis.y, z=axis.z;
        return {{t*x*x+c, t*x*y+s*z, t*x*z-s*y},
                {t*x*y-s*z, t*y*y+c, t*y*z+s*x},
                {t*x*z+s*y, t*y*z-s*x, t*z*z+c}};
    }
};

template<FloatScalar T> [[nodiscard]] constexpr T      determinant(Mat3<T> m) noexcept { return m.determinant(); }
template<FloatScalar T> [[nodiscard]] constexpr Mat3<T> transpose(Mat3<T> m)  noexcept { return m.transposed(); }
template<FloatScalar T> [[nodiscard]] constexpr Mat3<T> inverse(Mat3<T> m)    noexcept { return m.inversed(); }
template<FloatScalar T> [[nodiscard]] constexpr bool nearly_equal(Mat3<T> a, Mat3<T> b, T eps=EPS_F32) noexcept {
    for(i32 i=0;i<3;++i) if(!nearly_equal(a.col[i],b.col[i],eps)) return false; return true;
}

using Mat3f = Mat3<f32>;
using Mat3d = Mat3<f64>;

} // namespace qs::math
