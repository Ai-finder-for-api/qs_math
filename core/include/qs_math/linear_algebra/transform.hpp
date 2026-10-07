#pragma once
// qs_math/transform.hpp — TRS Transform, local/world, hierarchical composition
#include "quat.hpp"
#include "mat4.hpp"
#include "vec3.hpp"

namespace qs::math {

template<FloatScalar T = f32>
struct Transform {
    Vec3<T> translation{T(0),T(0),T(0)};
    Quat<T> rotation{Quat<T>::identity()};
    Vec3<T> scale{T(1),T(1),T(1)};

    constexpr Transform() noexcept = default;
    constexpr Transform(Vec3<T> t, Quat<T> r, Vec3<T> s) noexcept : translation(t),rotation(r),scale(s) {}

    // Matrix form
    [[nodiscard]] Mat4<T> to_mat4() const noexcept {
        return Mat4<T>::from_trs(translation, rotation.to_mat3(), scale);
    }
    [[nodiscard]] Mat4<T> to_inverse_mat4() const noexcept {
        Vec3<T> inv_s{T(1)/scale.x, T(1)/scale.y, T(1)/scale.z};
        Quat<T> inv_r=rotation.conjugate();
        Vec3<T> inv_t=inv_r.rotate(-translation)*inv_s;
        return Mat4<T>::from_trs(inv_t, inv_r.to_mat3(), inv_s);
    }

    // Transform point (applies scale, rotation, translation)
    [[nodiscard]] constexpr Vec3<T> transform_point(Vec3<T> p) const noexcept {
        return rotation.rotate(p * scale) + translation;
    }
    // Transform direction (no translation, with scale)
    [[nodiscard]] constexpr Vec3<T> transform_dir(Vec3<T> d) const noexcept {
        return rotation.rotate(d * scale);
    }
    // Transform normal (inverse-transpose)
    [[nodiscard]] constexpr Vec3<T> transform_normal(Vec3<T> n) const noexcept {
        Vec3<T> inv_s{T(1)/scale.x, T(1)/scale.y, T(1)/scale.z};
        return rotation.rotate(n * inv_s).normalized();
    }

    // Inverse transform
    [[nodiscard]] constexpr Vec3<T> inverse_transform_point(Vec3<T> p) const noexcept {
        Vec3<T> inv_s{T(1)/scale.x, T(1)/scale.y, T(1)/scale.z};
        return rotation.conjugate().rotate(p - translation) * inv_s;
    }
    [[nodiscard]] constexpr Vec3<T> inverse_transform_dir(Vec3<T> d) const noexcept {
        Vec3<T> inv_s{T(1)/scale.x, T(1)/scale.y, T(1)/scale.z};
        return rotation.conjugate().rotate(d) * inv_s;
    }

    // Compose: parent * child gives world transform
    [[nodiscard]] Transform operator*(Transform child) const noexcept {
        return {
            transform_point(child.translation),
            rotation * child.rotation,
            scale * child.scale
        };
    }

    // Inverse transform
    [[nodiscard]] Transform inversed() const noexcept {
        Quat<T> inv_r=rotation.conjugate();
        Vec3<T> inv_s{T(1)/scale.x, T(1)/scale.y, T(1)/scale.z};
        Vec3<T> inv_t=inv_r.rotate(-translation)*inv_s;
        return {inv_t, inv_r, inv_s};
    }

    // Interpolation
    [[nodiscard]] static Transform lerp(Transform a, Transform b, T t) noexcept {
        return {
            qs::math::lerp(a.translation, b.translation, t),
            Quat<T>::slerp(a.rotation, b.rotation, t),
            qs::math::lerp(a.scale, b.scale, t)
        };
    }

    // Decompose Mat4 into TRS (approximate, assumes no shear)
    static Transform from_mat4(Mat4<T> m) noexcept {
        Vec3<T> t=m.translation();
        Vec3<T> s=m.scale_factors();
        Mat3<T> rot=m.upper3x3();
        if(s.x>T(EPS_LENGTH)) { rot[0]=rot[0]*(T(1)/s.x); }
        if(s.y>T(EPS_LENGTH)) { rot[1]=rot[1]*(T(1)/s.y); }
        if(s.z>T(EPS_LENGTH)) { rot[2]=rot[2]*(T(1)/s.z); }
        Quat<T> r=Quat<T>::from_mat3(rot);
        return {t,r,s};
    }

    // Direction helpers
    [[nodiscard]] Vec3<T> forward() const noexcept { return rotation.rotate(Vec3<T>::forward()); }
    [[nodiscard]] Vec3<T> right()   const noexcept { return rotation.rotate(Vec3<T>::right()); }
    [[nodiscard]] Vec3<T> up()      const noexcept { return rotation.rotate(Vec3<T>::up()); }

    // Look at target
    static Transform look_at(Vec3<T> position, Vec3<T> target, Vec3<T> up=Vec3<T>::up()) noexcept {
        Vec3<T> fwd=(target-position).normalized();
        Quat<T> rot=Quat<T>::from_to(Vec3<T>::forward(), fwd);
        return {position, rot, Vec3<T>{T(1)}};
    }

    static constexpr Transform identity() noexcept { return {}; }

    bool operator==(Transform o) const noexcept {
        return nearly_equal(translation,o.translation)&&
               nearly_equal(rotation,o.rotation)&&
               nearly_equal(scale,o.scale);
    }
};

using Transformf = Transform<f32>;
using Transformd = Transform<f64>;

// Hierarchical chain: world = root * child[0] * child[1] * ...
template<FloatScalar T>
[[nodiscard]] Transform<T> compose_chain(const Transform<T>* transforms, i32 count) noexcept {
    Transform<T> result = Transform<T>::identity();
    for(i32 i=0; i<count; ++i) result = result * transforms[i];
    return result;
}

} // namespace qs::math
