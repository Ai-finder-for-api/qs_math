#pragma once
// qs_math/mat2.hpp — 2x2 column-major matrix
#include "vec2.hpp"

namespace qs::math {

template<FloatScalar T>
struct Mat2 {
    // Column-major: col[0]=right, col[1]=up
    Vec2<T> col[2];

    constexpr Mat2() noexcept : col{{T(1),T(0)},{T(0),T(1)}} {}
    constexpr Mat2(Vec2<T> c0, Vec2<T> c1) noexcept : col{c0,c1} {}
    constexpr Mat2(T m00,T m10,T m01,T m11) noexcept : col{{m00,m10},{m01,m11}} {}

    [[nodiscard]] constexpr Vec2<T>  operator[](i32 i) const noexcept { return col[i]; }
    [[nodiscard]] constexpr Vec2<T>& operator[](i32 i)       noexcept { return col[i]; }
    [[nodiscard]] constexpr T at(i32 r, i32 c) const noexcept { return col[c][r]; }
    [[nodiscard]] constexpr T& at(i32 r, i32 c)      noexcept { return col[c][r]; }

    constexpr Mat2 operator+(Mat2 o) const noexcept { return {col[0]+o.col[0], col[1]+o.col[1]}; }
    constexpr Mat2 operator-(Mat2 o) const noexcept { return {col[0]-o.col[0], col[1]-o.col[1]}; }
    constexpr Mat2 operator*(T s)    const noexcept { return {col[0]*s, col[1]*s}; }

    constexpr Mat2 operator*(Mat2 o) const noexcept {
        return {
            {col[0].x*o.col[0].x+col[1].x*o.col[0].y, col[0].y*o.col[0].x+col[1].y*o.col[0].y},
            {col[0].x*o.col[1].x+col[1].x*o.col[1].y, col[0].y*o.col[1].x+col[1].y*o.col[1].y}
        };
    }

    constexpr Vec2<T> operator*(Vec2<T> v) const noexcept {
        return {col[0].x*v.x+col[1].x*v.y, col[0].y*v.x+col[1].y*v.y};
    }

    [[nodiscard]] constexpr T determinant() const noexcept {
        return col[0].x*col[1].y - col[1].x*col[0].y;
    }

    [[nodiscard]] constexpr Mat2 transposed() const noexcept {
        return {{col[0].x,col[1].x},{col[0].y,col[1].y}};
    }

    [[nodiscard]] constexpr Mat2 inversed() const noexcept {
        T det = determinant();
        T inv = (det < T(EPS_DETERMINANT) && det > -T(EPS_DETERMINANT)) ? T(0) : T(1)/det;
        return {{ col[1].y*inv, -col[0].y*inv},
                {-col[1].x*inv,  col[0].x*inv}};
    }

    [[nodiscard]] constexpr bool is_identity(T eps=T(EPS_F32)) const noexcept {
        return nearly_equal(col[0].x,T(1),eps)&&nearly_equal(col[0].y,T(0),eps)&&
               nearly_equal(col[1].x,T(0),eps)&&nearly_equal(col[1].y,T(1),eps);
    }

    static constexpr Mat2 identity()           noexcept { return {}; }
    static constexpr Mat2 zero()               noexcept { return {{T(0),T(0)},{T(0),T(0)}}; }
    static constexpr Mat2 scale(T sx, T sy)    noexcept { return {{sx,T(0)},{T(0),sy}}; }
    static inline    Mat2 rotation(T angle_rad) noexcept {
        T c=qs::math::cos(angle_rad), s=qs::math::sin(angle_rad);
        return {{c,s},{-s,c}};
    }
};

template<FloatScalar T> [[nodiscard]] constexpr T   determinant(Mat2<T> m) noexcept { return m.determinant(); }
template<FloatScalar T> [[nodiscard]] constexpr Mat2<T> transpose(Mat2<T> m) noexcept { return m.transposed(); }
template<FloatScalar T> [[nodiscard]] constexpr Mat2<T> inverse (Mat2<T> m) noexcept { return m.inversed(); }

using Mat2f = Mat2<f32>;
using Mat2d = Mat2<f64>;

} // namespace qs::math
