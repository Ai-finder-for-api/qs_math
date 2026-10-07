#pragma once
// mat2_ext.hpp — Mat2 decomposition, polar decomposition, 2D transform helpers
#include "mat2.hpp"

namespace qs::math {

// 2D affine transform helpers using Mat3 (augmented 2D)
// [a b tx]
// [c d ty]
// [0 0  1]
template<FloatScalar T>
struct Affine2D {
    T a{T(1)},b{T(0)},c{T(0)},d{T(1)},tx{T(0)},ty{T(0)};
    constexpr Affine2D() noexcept = default;
    constexpr Affine2D(T a,T b,T c,T d,T tx,T ty) noexcept : a(a),b(b),c(c),d(d),tx(tx),ty(ty) {}

    [[nodiscard]] constexpr Vec2<T> transform(Vec2<T> p) const noexcept {
        return {a*p.x+b*p.y+tx, c*p.x+d*p.y+ty};
    }
    [[nodiscard]] constexpr Affine2D operator*(Affine2D o) const noexcept {
        return {a*o.a+b*o.c, a*o.b+b*o.d, c*o.a+d*o.c, c*o.b+d*o.d,
                a*o.tx+b*o.ty+tx, c*o.tx+d*o.ty+ty};
    }
    [[nodiscard]] constexpr T det() const noexcept { return a*d-b*c; }
    [[nodiscard]] constexpr Affine2D inversed() const noexcept {
        T inv=T(1)/det();
        return {d*inv,-b*inv,-c*inv,a*inv, (b*ty-d*tx)*inv,(c*tx-a*ty)*inv};
    }
    static constexpr Affine2D identity() noexcept { return {}; }
    static inline Affine2D rotation(T a) noexcept {
        T c=qs::math::cos(a),s=qs::math::sin(a); return {c,s,-s,c,T(0),T(0)};
    }
    static constexpr Affine2D translation(T tx,T ty) noexcept { return {T(1),T(0),T(0),T(1),tx,ty}; }
    static constexpr Affine2D scale(T sx,T sy) noexcept { return {sx,T(0),T(0),sy,T(0),T(0)}; }
};

using Affine2Df = Affine2D<f32>;
using Affine2Dd = Affine2D<f64>;

// SVD of 2x2 matrix (approximate, useful for decomposition)
template<FloatScalar T>
struct SVD2 { Mat2<T> U, V; Vec2<T> S; };

template<FloatScalar T>
[[nodiscard]] inline SVD2<T> svd2(Mat2<T> m) noexcept {
    // Jacobi iteration for 2x2
    T a=m[0].x,b=m[1].x,c=m[0].y,d=m[1].y;
    T theta=T(0.5)*atan2(T(2)*(a*c+b*d),(a*a+b*b-c*c-d*d));
    Mat2<T> U=Mat2<T>::rotation(theta);
    Mat2<T> tmp=m.transposed()*U;
    T phi=T(0.5)*atan2(T(2)*tmp[0].x*tmp[1].x,(tmp[0].x*tmp[0].x-tmp[1].x*tmp[1].x));
    Mat2<T> V=Mat2<T>::rotation(-phi);
    Mat2<T> S_mat=U.transposed()*m*V;
    return {U,V,{S_mat[0].x,S_mat[1].y}};
}

// Polar decomposition: M = R * S (rotation * symmetric)
template<FloatScalar T>
[[nodiscard]] inline Mat2<T> polar_rotation(Mat2<T> m) noexcept {
    // Iterative polar decomposition
    Mat2<T> R=m;
    for(i32 i=0;i<8;++i){
        Mat2<T> Rinv=R.inversed();
        R=(R+Rinv.transposed())*T(0.5);
    }
    return R;
}

} // namespace qs::math
