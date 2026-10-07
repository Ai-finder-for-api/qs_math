#pragma once
// qs_math/geometry/include/qs_math_geometry.hpp — Extended geometry (C++20)
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <array>
namespace qs::math::geometry {
using namespace qs::math;
// ─── 2D convex hull (Graham scan) ────────────────────────────────────────────
std::vector<Vec2<f32>> convex_hull_2d(std::vector<Vec2<f32>> pts) noexcept;
// ─── Polygon area (signed, shoelace) ─────────────────────────────────────────
[[nodiscard]] inline f32 polygon_area(const Vec2<f32>* pts, i32 n) noexcept {
    f32 s=0;
    for(i32 i=0,j=n-1;i<n;j=i++) s+=pts[j].x*pts[i].y-pts[i].x*pts[j].y;
    return s*0.5f;
}
[[nodiscard]] inline bool polygon_is_ccw(const Vec2<f32>* pts, i32 n) noexcept { return polygon_area(pts,n)>0; }
// ─── Polygon centroid ─────────────────────────────────────────────────────────
[[nodiscard]] Vec2<f32> polygon_centroid(const Vec2<f32>* pts, i32 n) noexcept;
// ─── Sutherland-Hodgman polygon clipping ─────────────────────────────────────
std::vector<Vec2<f32>> clip_polygon_half_plane(const std::vector<Vec2<f32>>& poly,
    Vec2<f32> plane_pt, Vec2<f32> plane_normal) noexcept;
// ─── Minkowski sum of two convex polygons ────────────────────────────────────
std::vector<Vec2<f32>> minkowski_sum(const std::vector<Vec2<f32>>& A,
    const std::vector<Vec2<f32>>& B) noexcept;
// ─── 3D convex hull via incremental algorithm ─────────────────────────────────
struct ConvexHull3 {
    std::vector<Vec3<f32>> verts;
    std::vector<std::array<i32,3>> faces;
};
ConvexHull3 convex_hull_3d(const std::vector<Vec3<f32>>& pts) noexcept;
// ─── Plane from 3 points ──────────────────────────────────────────────────────
struct Plane { Vec3<f32> normal; f32 d;
    [[nodiscard]] f32 signed_dist(Vec3<f32> p) const noexcept { return normal.dot(p)+d; }
    [[nodiscard]] Vec3<f32> project(Vec3<f32> p) const noexcept { return p-normal*signed_dist(p); }
};
[[nodiscard]] inline Plane plane_from_3pts(Vec3<f32> a, Vec3<f32> b, Vec3<f32> c) noexcept {
    Vec3<f32> n=(b-a).cross(c-a).normalized();
    return {n,-n.dot(a)};
}
// ─── Polygon triangulation (ear-clipping) ────────────────────────────────────
std::vector<std::array<i32,3>> triangulate_polygon(const Vec2<f32>* pts, i32 n) noexcept;
// ─── Voronoi diagram (Fortune's algorithm, 2D) ───────────────────────────────
struct VoronoiEdge { Vec2<f32> a, b; i32 left_site, right_site; };
std::vector<VoronoiEdge> voronoi_2d(const Vec2<f32>* sites, i32 n,
    Vec2<f32> bbox_min, Vec2<f32> bbox_max) noexcept;
// ─── Delaunay triangulation (Bowyer-Watson) ───────────────────────────────────
struct DelaunayTri { std::array<i32,3> v; };
std::vector<DelaunayTri> delaunay_2d(const Vec2<f32>* pts, i32 n) noexcept;
// ─── Point in convex polygon (O(log n)) ──────────────────────────────────────
[[nodiscard]] bool point_in_convex_polygon(Vec2<f32> p,
    const Vec2<f32>* hull, i32 n) noexcept;
// ─── SAT: polygon vs polygon ─────────────────────────────────────────────────
[[nodiscard]] bool sat_overlap(const Vec2<f32>* A, i32 na,
    const Vec2<f32>* B, i32 nb,
    Vec2<f32>& mtv_out) noexcept;
} // namespace qs::math::geometry
