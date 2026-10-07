#pragma once
// qs_math/mat4.hpp — 4x4 column-major matrix, full projection/view set
#include "mat3.hpp"
#include "vec4.hpp"

namespace qs::math {

template<FloatScalar T>
struct Mat4 {
    Vec4<T> col[4];

    constexpr Mat4() noexcept
        : col{{T(1),T(0),T(0),T(0)},{T(0),T(1),T(0),T(0)},
              {T(0),T(0),T(1),T(0)},{T(0),T(0),T(0),T(1)}} {}
    constexpr Mat4(Vec4<T> c0,Vec4<T> c1,Vec4<T> c2,Vec4<T> c3) noexcept : col{c0,c1,c2,c3} {}
    explicit constexpr Mat4(Mat3<T> m) noexcept
        : col{{m[0].x,m[0].y,m[0].z,T(0)},{m[1].x,m[1].y,m[1].z,T(0)},
              {m[2].x,m[2].y,m[2].z,T(0)},{T(0),T(0),T(0),T(1)}} {}

    [[nodiscard]] constexpr Vec4<T>  operator[](i32 i) const noexcept { return col[i]; }
    [[nodiscard]] constexpr Vec4<T>& operator[](i32 i)       noexcept { return col[i]; }
    [[nodiscard]] constexpr T  at(i32 r, i32 c) const noexcept { return col[c][r]; }
    [[nodiscard]] constexpr T& at(i32 r, i32 c)       noexcept { return col[c][r]; }
    [[nodiscard]] constexpr Vec4<T> row(i32 r) const noexcept { return {col[0][r],col[1][r],col[2][r],col[3][r]}; }

    constexpr Vec4<T> operator*(Vec4<T> v) const noexcept {
        return col[0]*v.x + col[1]*v.y + col[2]*v.z + col[3]*v.w;
    }
    constexpr Vec3<T> transform_point(Vec3<T> p) const noexcept {
        Vec4<T> r = (*this) * Vec4<T>{p.x,p.y,p.z,T(1)};
        if (qs::math::abs(r.w) > T(EPS_LENGTH)) return {r.x/r.w,r.y/r.w,r.z/r.w};
        return {r.x,r.y,r.z};
    }
    constexpr Vec3<T> transform_dir(Vec3<T> d) const noexcept {
        Vec4<T> r = (*this) * Vec4<T>{d.x,d.y,d.z,T(0)};
        return {r.x,r.y,r.z};
    }
    constexpr Mat4 operator*(Mat4 b) const noexcept {
        return {(*this)*b.col[0], (*this)*b.col[1], (*this)*b.col[2], (*this)*b.col[3]};
    }
    constexpr Mat4 operator+(Mat4 o) const noexcept {
        return {col[0]+o.col[0],col[1]+o.col[1],col[2]+o.col[2],col[3]+o.col[3]};
    }
    constexpr Mat4 operator*(T s) const noexcept { return {col[0]*s,col[1]*s,col[2]*s,col[3]*s}; }

    [[nodiscard]] constexpr Mat4 transposed() const noexcept {
        return {
            {col[0].x,col[1].x,col[2].x,col[3].x},
            {col[0].y,col[1].y,col[2].y,col[3].y},
            {col[0].z,col[1].z,col[2].z,col[3].z},
            {col[0].w,col[1].w,col[2].w,col[3].w}
        };
    }

    [[nodiscard]] constexpr T determinant() const noexcept {
        T s0=col[0].x*col[1].y-col[1].x*col[0].y; T s1=col[0].x*col[2].y-col[2].x*col[0].y;
        T s2=col[0].x*col[3].y-col[3].x*col[0].y; T s3=col[1].x*col[2].y-col[2].x*col[1].y;
        T s4=col[1].x*col[3].y-col[3].x*col[1].y; T s5=col[2].x*col[3].y-col[3].x*col[2].y;
        T c0=col[0].z*col[1].w-col[1].z*col[0].w; T c1=col[0].z*col[2].w-col[2].z*col[0].w;
        T c2=col[0].z*col[3].w-col[3].z*col[0].w; T c3=col[1].z*col[2].w-col[2].z*col[1].w;
        T c4=col[1].z*col[3].w-col[3].z*col[1].w; T c5=col[2].z*col[3].w-col[3].z*col[2].w;
        return s0*c5-s1*c4+s2*c3+s3*c2-s4*c1+s5*c0;
    }

    [[nodiscard]] Mat4 inversed() const noexcept;  // defined in mat4_ops.hpp to keep size down

    [[nodiscard]] constexpr Mat3<T> upper3x3() const noexcept {
        return {col[0].xyz(), col[1].xyz(), col[2].xyz()};
    }
    [[nodiscard]] constexpr Vec3<T> translation() const noexcept { return col[3].xyz(); }
    [[nodiscard]] constexpr Vec3<T> scale_factors() const noexcept {
        return {col[0].xyz().length(), col[1].xyz().length(), col[2].xyz().length()};
    }

    static constexpr Mat4 identity() noexcept { return {}; }
    static constexpr Mat4 zero()     noexcept {
        return {{T(0),T(0),T(0),T(0)},{T(0),T(0),T(0),T(0)},{T(0),T(0),T(0),T(0)},{T(0),T(0),T(0),T(0)}};
    }

    static constexpr Mat4 from_translation(Vec3<T> t) noexcept {
        Mat4 m; m.col[3]={t.x,t.y,t.z,T(1)}; return m;
    }
    static constexpr Mat4 from_scale(Vec3<T> s) noexcept {
        Mat4 m; m.col[0].x=s.x; m.col[1].y=s.y; m.col[2].z=s.z; return m;
    }
    static inline Mat4 from_rotation_x(T a) noexcept { return Mat4{Mat3<T>::rotation_x(a)}; }
    static inline Mat4 from_rotation_y(T a) noexcept { return Mat4{Mat3<T>::rotation_y(a)}; }
    static inline Mat4 from_rotation_z(T a) noexcept { return Mat4{Mat3<T>::rotation_z(a)}; }
    static inline Mat4 from_axis_angle(Vec3<T> ax, T a) noexcept { return Mat4{Mat3<T>::rotation_axis_angle(ax,a)}; }

    // TRS composition
    static Mat4 from_trs(Vec3<T> t, Mat3<T> r, Vec3<T> s) noexcept {
        Mat4 m{r};
        m.col[0]*=s.x; m.col[1]*=s.y; m.col[2]*=s.z;
        m.col[3]={t.x,t.y,t.z,T(1)}; return m;
    }

    // View matrix (right-handed, -Z forward)
    static Mat4 look_at(Vec3<T> eye, Vec3<T> target, Vec3<T> up) noexcept {
        Vec3<T> f=(target-eye).normalized();
        Vec3<T> r=cross(f,up).normalized();
        Vec3<T> u=cross(r,f);
        return {
            {r.x,u.x,-f.x,T(0)},
            {r.y,u.y,-f.y,T(0)},
            {r.z,u.z,-f.z,T(0)},
            {-dot(r,eye),-dot(u,eye),dot(f,eye),T(1)}
        };
    }

    // Perspective (reversed-Z, infinite far, right-handed)
    static Mat4 perspective(T fov_y_rad, T aspect, T near_z) noexcept {
        T f=T(1)/qs::math::tan(fov_y_rad*T(0.5));
        Mat4 m=zero();
        m.col[0].x=f/aspect; m.col[1].y=f;
        m.col[2].z=T(0); m.col[2].w=T(-1);
        m.col[3].z=near_z; return m;
    }

    // Perspective standard (finite far)
    static Mat4 perspective(T fov_y_rad, T aspect, T near_z, T far_z) noexcept {
        T f=T(1)/qs::math::tan(fov_y_rad*T(0.5));
        T rng=T(1)/(near_z-far_z);
        Mat4 m=zero();
        m.col[0].x=f/aspect; m.col[1].y=f;
        m.col[2].z=(near_z+far_z)*rng; m.col[2].w=T(-1);
        m.col[3].z=T(2)*near_z*far_z*rng; return m;
    }

    // Orthographic (right-handed, NDC [-1,1])
    static constexpr Mat4 orthographic(T l,T r,T b,T t,T n,T f) noexcept {
        Mat4 m;
        m.col[0].x=T(2)/(r-l); m.col[1].y=T(2)/(t-b); m.col[2].z=T(-2)/(f-n);
        m.col[3].x=-(r+l)/(r-l); m.col[3].y=-(t+b)/(t-b); m.col[3].z=-(f+n)/(f-n);
        return m;
    }

    bool operator==(Mat4 o) const noexcept {
        for(i32 i=0;i<4;++i) if(col[i]!=o.col[i]) return false; return true;
    }
};

template<FloatScalar T>
[[nodiscard]] Mat4<T> Mat4<T>::inversed() const noexcept {
    // Cofactor expansion (Laplace)
    T s0=at(0,0)*at(1,1)-at(1,0)*at(0,1); T s1=at(0,0)*at(2,1)-at(2,0)*at(0,1);
    T s2=at(0,0)*at(3,1)-at(3,0)*at(0,1); T s3=at(1,0)*at(2,1)-at(2,0)*at(1,1);
    T s4=at(1,0)*at(3,1)-at(3,0)*at(1,1); T s5=at(2,0)*at(3,1)-at(3,0)*at(2,1);
    T c0=at(0,2)*at(1,3)-at(1,2)*at(0,3); T c1=at(0,2)*at(2,3)-at(2,2)*at(0,3);
    T c2=at(0,2)*at(3,3)-at(3,2)*at(0,3); T c3=at(1,2)*at(2,3)-at(2,2)*at(1,3);
    T c4=at(1,2)*at(3,3)-at(3,2)*at(1,3); T c5=at(2,2)*at(3,3)-at(3,2)*at(2,3);
    T det=s0*c5-s1*c4+s2*c3+s3*c2-s4*c1+s5*c0;
    if(qs::math::abs(det)<T(EPS_DETERMINANT)) return identity();
    T inv=T(1)/det;
    Mat4 r;
    r.at(0,0)=( at(1,1)*c5-at(2,1)*c4+at(3,1)*c3)*inv;
    r.at(0,1)=(-at(0,1)*c5+at(2,1)*c2-at(3,1)*c1)*inv;
    r.at(0,2)=( at(0,1)*c4-at(1,1)*c2+at(3,1)*c0)*inv;
    r.at(0,3)=(-at(0,1)*c3+at(1,1)*c1-at(2,1)*c0)*inv;
    r.at(1,0)=(-at(1,0)*c5+at(2,0)*c4-at(3,0)*c3)*inv;
    r.at(1,1)=( at(0,0)*c5-at(2,0)*c2+at(3,0)*c1)*inv;
    r.at(1,2)=(-at(0,0)*c4+at(1,0)*c2-at(3,0)*c0)*inv;
    r.at(1,3)=( at(0,0)*c3-at(1,0)*c1+at(2,0)*c0)*inv;
    r.at(2,0)=( at(1,0)*s5-at(2,0)*s4+at(3,0)*s3)*inv;
    r.at(2,1)=(-at(0,0)*s5+at(2,0)*s2-at(3,0)*s1)*inv;
    r.at(2,2)=( at(0,0)*s4-at(1,0)*s2+at(3,0)*s0)*inv;
    r.at(2,3)=(-at(0,0)*s3+at(1,0)*s1-at(2,0)*s0)*inv;
    r.at(3,0)=(-at(1,0)*s5+at(2,0)*s4-at(3,0)*s3)*inv; // col3
    r.at(3,1)=( at(0,0)*s5-at(2,0)*s2+at(3,0)*s1)*inv;
    r.at(3,2)=(-at(0,0)*s4+at(1,0)*s2-at(3,0)*s0)*inv;
    r.at(3,3)=( at(0,0)*s3-at(1,0)*s1+at(2,0)*s0)*inv;
    return r;
}

template<FloatScalar T> [[nodiscard]] constexpr Mat4<T> transpose(Mat4<T> m)  noexcept { return m.transposed(); }
template<FloatScalar T> [[nodiscard]]          Mat4<T>  inverse(Mat4<T> m)    noexcept { return m.inversed(); }
template<FloatScalar T> [[nodiscard]] constexpr T       determinant(Mat4<T> m) noexcept { return m.determinant(); }
template<FloatScalar T> [[nodiscard]] constexpr bool nearly_equal(Mat4<T> a, Mat4<T> b, T eps=EPS_F32) noexcept {
    for(i32 i=0;i<4;++i) if(!nearly_equal(a.col[i],b.col[i],eps)) return false; return true;
}

using Mat4f = Mat4<f32>;
using Mat4d = Mat4<f64>;

} // namespace qs::math
