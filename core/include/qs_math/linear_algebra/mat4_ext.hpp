#pragma once
// mat4_ext.hpp — Mat4 decompose, TRS extraction, billboard, skybox, shadow map,
//               jitter, texture coord transform, depth remap, clip-space utilities
#include "mat4.hpp"
#include "quat.hpp"
#include "transform.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Full TRS decompose (returns false if degenerate)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] bool decompose_trs(Mat4<T> m,
    Vec3<T>& translation, Quat<T>& rotation, Vec3<T>& scale_out) noexcept
{
    translation = m.translation();
    scale_out.x = m.col[0].xyz().length();
    scale_out.y = m.col[1].xyz().length();
    scale_out.z = m.col[2].xyz().length();
    if(scale_out.x<T(EPS_LENGTH)||scale_out.y<T(EPS_LENGTH)||scale_out.z<T(EPS_LENGTH))
        return false;
    // Handle reflection: det < 0 → negative scale
    if(determinant(m)<T(0)) { scale_out.x=-scale_out.x; }
    Mat3<T> rot_m{
        m.col[0].xyz()*(T(1)/scale_out.x),
        m.col[1].xyz()*(T(1)/scale_out.y),
        m.col[2].xyz()*(T(1)/scale_out.z)
    };
    rotation = Quat<T>::from_mat3(rot_m).normalized();
    return true;
}

// ---------------------------------------------------------------------------
// Billboard matrices (always face camera)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] inline Mat4<T> billboard_spherical(Vec3<T> pos, Vec3<T> camera_pos, Vec3<T> up) noexcept {
    Vec3<T> look=(camera_pos-pos).normalized();
    Vec3<T> right=cross(up,look).normalized();
    Vec3<T> u=cross(look,right);
    return {
        {right.x,right.y,right.z,T(0)},
        {u.x,u.y,u.z,T(0)},
        {look.x,look.y,look.z,T(0)},
        {pos.x,pos.y,pos.z,T(1)}
    };
}
template<FloatScalar T>
[[nodiscard]] inline Mat4<T> billboard_cylindrical(Vec3<T> pos, Vec3<T> camera_pos) noexcept {
    Vec3<T> look{camera_pos.x-pos.x, T(0), camera_pos.z-pos.z};
    look=look.normalized();
    Vec3<T> right=cross(Vec3<T>::up(),look).normalized();
    Vec3<T> up=cross(look,right);
    return {
        {right.x,right.y,right.z,T(0)},
        {up.x,up.y,up.z,T(0)},
        {look.x,look.y,look.z,T(0)},
        {pos.x,pos.y,pos.z,T(1)}
    };
}

// ---------------------------------------------------------------------------
// Shadow map bias matrix (NDC→texture space)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr Mat4<T> shadow_bias_matrix() noexcept {
    // Maps NDC [-1,1] to [0,1] for texture sampling
    return {
        {T(0.5),T(0),T(0),T(0)},
        {T(0),T(0.5),T(0),T(0)},
        {T(0),T(0),T(0.5),T(0)},
        {T(0.5),T(0.5),T(0.5),T(1)}
    };
}

// ---------------------------------------------------------------------------
// Jitter matrix (temporal anti-aliasing sub-pixel jitter)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr Mat4<T> taa_jitter(T jx, T jy, T viewport_w, T viewport_h) noexcept {
    Mat4<T> m=Mat4<T>::identity();
    m.col[3].x=jx*T(2)/viewport_w;
    m.col[3].y=jy*T(2)/viewport_h;
    return m;
}

// ---------------------------------------------------------------------------
// Infinite reverse-Z projection (best precision, no far clip)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] inline Mat4<T> perspective_reverse_z(T fov_y, T aspect, T near_z) noexcept {
    T f=T(1)/qs::math::tan(fov_y*T(0.5));
    Mat4<T> m=Mat4<T>::zero();
    m.col[0].x=f/aspect;
    m.col[1].y=f;
    m.col[2].z=T(0);  m.col[2].w=T(-1);
    m.col[3].z=near_z;
    return m;
}

// ---------------------------------------------------------------------------
// UV→world position from depth
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> unproject(Vec2<T> uv, T depth, Mat4<T> inv_proj, Mat4<T> inv_view) noexcept {
    Vec4<T> clip{uv.x*T(2)-T(1), uv.y*T(2)-T(1), depth*T(2)-T(1), T(1)};
    Vec4<T> view=inv_proj*clip;
    view=view*(T(1)/view.w);
    Vec4<T> world=inv_view*view;
    return {world.x,world.y,world.z};
}

// ---------------------------------------------------------------------------
// Clip-space utilities
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr bool ndc_in_unit_cube(Vec3<T> ndc) noexcept {
    return ndc.x>=-T(1)&&ndc.x<=T(1)&&ndc.y>=-T(1)&&ndc.y<=T(1)&&ndc.z>=-T(1)&&ndc.z<=T(1);
}

// Depth buffer value from linear depth
template<FloatScalar T>
[[nodiscard]] constexpr T linear_to_ndc_depth(T linear_z, T near, T far) noexcept {
    return (far*(linear_z-near))/(linear_z*(far-near));
}

// Halton sequence for TAA jitter
[[nodiscard]] constexpr f32 halton(i32 index, i32 base) noexcept {
    f32 result=0.f; f32 f=1.f;
    while(index>0){ f/=base; result+=f*(index%base); index/=base; }
    return result;
}

// Generate TAA jitter pattern (2D Halton 2,3)
[[nodiscard]] constexpr Vec2f halton_jitter(i32 frame_index, i32 sequence_length=16) noexcept {
    i32 idx=(frame_index%sequence_length)+1;
    return {halton(idx,2)-0.5f, halton(idx,3)-0.5f};
}

} // namespace qs::math
