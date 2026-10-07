#pragma once
// qs_math/editor/include/qs_math_editor.hpp (C++20)
// Math helpers for editor tools: snapping, grid, gizmo hit tests, screen-space ops
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/mat4.hpp>
#include <cmath>
namespace qs::math::editor {
using namespace qs::math;
// ─── Grid snapping ────────────────────────────────────────────────────────────
[[nodiscard]] inline f32 snap(f32 v, f32 step) noexcept {
    return step<1e-7f?v:std::round(v/step)*step;
}
[[nodiscard]] inline Vec3<f32> snap3(Vec3<f32> v, f32 step) noexcept {
    return {snap(v.x,step),snap(v.y,step),snap(v.z,step)};
}
[[nodiscard]] inline f32 snap_angle_deg(f32 deg, f32 step_deg) noexcept {
    return snap(deg,step_deg);
}
// ─── Screen-space to world-space ray ─────────────────────────────────────────
struct Ray { Vec3<f32> origin, dir; };
[[nodiscard]] Ray screen_to_ray(f32 sx, f32 sy, f32 screen_w, f32 screen_h,
    const Mat4<f32>& inv_view_proj) noexcept;
// ─── World to screen (returns {x,y,depth} where depth in [-1,1]) ─────────────
[[nodiscard]] Vec3<f32> world_to_screen(Vec3<f32> world,
    const Mat4<f32>& view_proj, f32 screen_w, f32 screen_h) noexcept;
// ─── Axis handle hit test (translate gizmo) ───────────────────────────────────
[[nodiscard]] f32 axis_handle_hit(Ray ray, Vec3<f32> origin,
    Vec3<f32> axis_dir, f32 handle_len, f32 handle_radius) noexcept;
// ─── Arc-ball rotation ────────────────────────────────────────────────────────
[[nodiscard]] Vec4<f32> arcball_delta(Vec2<f32> p0, Vec2<f32> p1,
    f32 radius, Vec3<f32> sphere_center) noexcept;
// ─── AABB screen-space extent ────────────────────────────────────────────────
[[nodiscard]] std::array<f32,4> aabb_screen_rect(Vec3<f32> mn, Vec3<f32> mx,
    const Mat4<f32>& view_proj, f32 w, f32 h) noexcept;
} // namespace qs::math::editor
