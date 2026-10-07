
#pragma once
// qs_math/uv/include/qs_math_uv.hpp (C++20)
// UV mapping math: unwrapping, LSCM, ABF, texture-space operations.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <array>
namespace qs::math::uv {
using namespace qs::math;
// ─── UV chart ────────────────────────────────────────────────────────────────
struct UVChart {
    std::vector<Vec2<f32>> uvs;   // per-vertex UVs
    std::vector<i32> vertex_ids;  // original mesh vertex ids
    f32 surface_area_3d{0};
    f32 surface_area_uv{0};
    [[nodiscard]] f32 texel_density() const noexcept {
        return surface_area_uv>1e-7f?surface_area_3d/surface_area_uv:0;
    }
};
// ─── Planar projection ────────────────────────────────────────────────────────
[[nodiscard]] Vec2<f32> project_planar_xy(Vec3<f32> p) noexcept;
[[nodiscard]] Vec2<f32> project_planar_xz(Vec3<f32> p) noexcept;
[[nodiscard]] Vec2<f32> project_cylindrical(Vec3<f32> p, Vec3<f32> axis) noexcept;
[[nodiscard]] Vec2<f32> project_spherical(Vec3<f32> dir) noexcept;
[[nodiscard]] Vec2<f32> project_cube(Vec3<f32> dir, i32& face_out) noexcept;
// ─── Triangle UV → barycentric → 3D ─────────────────────────────────────────
[[nodiscard]] Vec3<f32> uv_to_bary(Vec2<f32> uv,
    Vec2<f32> uv0, Vec2<f32> uv1, Vec2<f32> uv2) noexcept;
[[nodiscard]] Vec2<f32> bary_to_uv(Vec3<f32> bary,
    Vec2<f32> uv0, Vec2<f32> uv1, Vec2<f32> uv2) noexcept;
// ─── UV distortion metrics ────────────────────────────────────────────────────
[[nodiscard]] f32 uv_area_distortion(Vec3<f32> p0, Vec3<f32> p1, Vec3<f32> p2,
    Vec2<f32> uv0, Vec2<f32> uv1, Vec2<f32> uv2) noexcept;
[[nodiscard]] f32 uv_angle_distortion(Vec3<f32> p0, Vec3<f32> p1, Vec3<f32> p2,
    Vec2<f32> uv0, Vec2<f32> uv1, Vec2<f32> uv2) noexcept;
// ─── LSCM-style least-squares conformal UV ────────────────────────────────────
UVChart lscm_unwrap(const Vec3<f32>* verts, i32 nv,
    const i32* faces, i32 nf) noexcept;
// ─── UV packing (bin packing into [0,1]^2) ───────────────────────────────────
struct UVIsland {
    Vec2<f32> min{0,0}, max{1,1};
    std::vector<Vec2<f32>>* uvs{nullptr};
};
void pack_islands(std::vector<UVIsland>& islands, f32 padding=0.01f) noexcept;
// ─── Seam relaxation ─────────────────────────────────────────────────────────
void relax_uvs(Vec2<f32>* uvs, i32 nv,
    const i32* faces, i32 nf, i32 iters=10) noexcept;
// ─── Texture-space tangent frame (for normal mapping) ─────────────────────────
struct TangentFrame {
    Vec3<f32> T, B, N;
    [[nodiscard]] Vec3<f32> tangent_to_world(Vec3<f32> ts_normal) const noexcept {
        return (T*ts_normal.x + B*ts_normal.y + N*ts_normal.z).normalized();
    }
};
[[nodiscard]] TangentFrame compute_tangent_frame(
    Vec3<f32> p0, Vec3<f32> p1, Vec3<f32> p2,
    Vec2<f32> uv0, Vec2<f32> uv1, Vec2<f32> uv2) noexcept;
// ─── Mip level computation ────────────────────────────────────────────────────
[[nodiscard]] inline f32 mip_level(Vec2<f32> duvdx, Vec2<f32> duvdy,
    f32 tex_w, f32 tex_h) noexcept
{
    const f32 lx=(duvdx.x*tex_w)*(duvdx.x*tex_w)+(duvdx.y*tex_h)*(duvdx.y*tex_h);
    const f32 ly=(duvdy.x*tex_w)*(duvdy.x*tex_w)+(duvdy.y*tex_h)*(duvdy.y*tex_h);
    return 0.5f*std::log2(std::max(lx,ly));
}
} // namespace qs::math::uv
