
#pragma once
// qs_math/sculpting/include/qs_math_sculpting.hpp (C++20)
// Sculpting/mesh-editing math: brush falloffs, displacement, remeshing helpers.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <functional>
namespace qs::math::sculpt {
using namespace qs::math;
// ─── Brush falloff functions ──────────────────────────────────────────────────
enum class Falloff { Smooth, Constant, Linear, Sharp, Sphere, Root };
[[nodiscard]] f32 brush_falloff(f32 t, Falloff f=Falloff::Smooth) noexcept;
// ─── Brush operations (return displacement weight) ───────────────────────────
[[nodiscard]] f32 brush_draw(Vec3<f32> p, Vec3<f32> center, Vec3<f32> normal,
    f32 radius, f32 strength, Falloff f=Falloff::Smooth) noexcept;
[[nodiscard]] Vec3<f32> brush_smooth(const Vec3<f32>* verts, const i32* neighbors,
    i32 n_neighbors, f32 strength) noexcept;
[[nodiscard]] Vec3<f32> brush_flatten(Vec3<f32> p, Vec3<f32> plane_pt,
    Vec3<f32> plane_n, f32 strength) noexcept;
[[nodiscard]] Vec3<f32> brush_pinch(Vec3<f32> p, Vec3<f32> center,
    f32 strength) noexcept;
[[nodiscard]] Vec3<f32> brush_crease(Vec3<f32> p, Vec3<f32> center,
    Vec3<f32> normal, f32 strength, f32 radius) noexcept;
// ─── Subdivision (Loop scheme, one level) ─────────────────────────────────────
struct SubdivResult {
    std::vector<Vec3<f32>> verts;
    std::vector<std::array<i32,3>> faces;
};
[[nodiscard]] SubdivResult loop_subdivide(const Vec3<f32>* verts, i32 nv,
    const i32* faces, i32 nf) noexcept;
// ─── Dynamic topology: edge collapse ─────────────────────────────────────────
struct HalfEdgeMesh {
    struct Edge { i32 v0,v1,face,next,twin; f32 collapse_cost; };
    std::vector<Vec3<f32>> verts;
    std::vector<Edge> edges;
    std::vector<i32> face_edges; // first half-edge per face
    void build(const Vec3<f32>* v,i32 nv,const i32* f,i32 nf) noexcept;
    void compute_collapse_costs() noexcept;
    void collapse_edge(i32 eid) noexcept;
    [[nodiscard]] i32 cheapest_edge() const noexcept;
};
// ─── Voxel remeshing helper ───────────────────────────────────────────────────
struct VoxelGrid {
    Vec3<f32> origin{};
    f32 voxel_size{0.01f};
    i32 nx{0},ny{0},nz{0};
    std::vector<f32> sdf; // signed distance field
    void compute_sdf(const Vec3<f32>* verts, i32 nv,
        const i32* faces, i32 nf) noexcept;
    [[nodiscard]] Vec3<f32> gradient_at(i32 x,i32 y,i32 z) const noexcept;
    [[nodiscard]] f32 sample(Vec3<f32> p) const noexcept;
};
// ─── Subdivision surface limit point ─────────────────────────────────────────
[[nodiscard]] Vec3<f32> catmull_clark_limit(Vec3<f32> v,
    const Vec3<f32>* ring, i32 n) noexcept;
[[nodiscard]] Vec3<f32> catmull_clark_limit_normal(Vec3<f32> v,
    const Vec3<f32>* ring, i32 n) noexcept;
} // namespace qs::math::sculpt
