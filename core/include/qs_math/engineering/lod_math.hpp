#pragma once
// qs_math/lod_math.hpp — LOD selection, geomorphing, error metrics, imposter math (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>

namespace qs {

// ─── LOD levels ───────────────────────────────────────────────────────────────

struct LODLevel {
    f32 screen_size_threshold{};  // switch when screen size < this (pixels)
    f32 distance_threshold{};     // or when distance > this
    u32 mesh_index{};
    f32 fade_start{};             // begin cross-fade at this distance
    f32 fade_end{};
};

struct LODGroup {
    std::vector<LODLevel> levels;
    Vec3f center{};
    f32   radius{1.f};

    void add_level(f32 screen_thresh, f32 dist_thresh, u32 mesh_idx, f32 fade=0.1f) {
        LODLevel l; l.screen_size_threshold=screen_thresh;
        l.distance_threshold=dist_thresh; l.mesh_index=mesh_idx;
        l.fade_start=dist_thresh; l.fade_end=dist_thresh*(1.f+fade);
        levels.push_back(l);
        std::sort(levels.begin(),levels.end(),[](auto& a,auto& b){
            return a.distance_threshold<b.distance_threshold;});
    }

    [[nodiscard]] u32 select_lod(f32 dist) const {
        for (u32 i=0;i<u32(levels.size());++i)
            if (dist<=levels[i].distance_threshold) return levels[i].mesh_index;
        return levels.empty() ? 0 : levels.back().mesh_index;
    }

    // Cross-fade weight between LOD i and i+1
    [[nodiscard]] f32 fade_weight(u32 lod_idx, f32 dist) const {
        if (lod_idx >= levels.size()) return 1.f;
        const auto& l = levels[lod_idx];
        if (dist <= l.fade_start) return 1.f;
        if (dist >= l.fade_end)   return 0.f;
        return 1.f - (dist-l.fade_start)/(l.fade_end-l.fade_start);
    }
};

// ─── Screen-space size computation ───────────────────────────────────────────

[[nodiscard]] inline f32 projected_screen_size(
    const Vec3f& obj_center, f32 obj_radius,
    const Vec3f& cam_pos, f32 fov_y_rad,
    f32 screen_height_px)
{
    f32 dist = std::max(length(obj_center - cam_pos), 1e-4f);
    f32 proj = (obj_radius / dist) * (screen_height_px / (2.f*std::tan(fov_y_rad*0.5f)));
    return proj * 2.f; // diameter
}

// ─── Geomorphing ──────────────────────────────────────────────────────────────

// Blend vertex position between LOD levels
[[nodiscard]] inline Vec3f geomorph_vertex(
    const Vec3f& lod_hi, const Vec3f& lod_lo, f32 t)
{
    return lod_hi*(1.f-t) + lod_lo*t;
}

// Quadratic error metric for geomorph threshold
[[nodiscard]] inline f32 geomorph_error(
    const Vec3f& v_hi, const Vec3f& v_lo,
    const Vec3f& cam_pos, f32 screen_height, f32 fov_y)
{
    f32 geometric_err = length(v_hi - v_lo);
    f32 dist = length((v_hi+v_lo)*0.5f - cam_pos);
    f32 proj_err = geometric_err / dist * (screen_height / (2.f*std::tan(fov_y*0.5f)));
    return proj_err;
}

// ─── Quadric Error Metric (QEM) for mesh simplification ──────────────────────

struct Quadric {
    f32 A[4][4]{}; // Symmetric 4x4 error quadric

    void add_plane(const Vec3f& n, f32 d) {
        // Outer product [n, d]^T [n, d]
        f32 v[4] = {n.x,n.y,n.z,d};
        for (u32 i=0;i<4;++i)
        for (u32 j=0;j<4;++j)
            A[i][j] += v[i]*v[j];
    }

    Quadric operator+(const Quadric& o) const {
        Quadric q;
        for (u32 i=0;i<4;++i)
        for (u32 j=0;j<4;++j) q.A[i][j]=A[i][j]+o.A[i][j];
        return q;
    }

    [[nodiscard]] f32 eval(const Vec3f& v) const {
        f32 x[4]={v.x,v.y,v.z,1.f};
        f32 e=0.f;
        for (u32 i=0;i<4;++i)
        for (u32 j=0;j<4;++j) e+=x[i]*A[i][j]*x[j];
        return e;
    }
};

struct CollapseEdge {
    u32 v0{}, v1{};
    f32 cost{};
    Vec3f optimal{};
};

[[nodiscard]] inline CollapseEdge compute_collapse_cost(
    u32 v0, u32 v1,
    const Vec3f& p0, const Vec3f& p1,
    const Quadric& q0, const Quadric& q1)
{
    Quadric q = q0 + q1;
    Vec3f mid = (p0+p1)*0.5f;
    return {v0, v1, q.eval(mid), mid};
}

// ─── Imposters ────────────────────────────────────────────────────────────────

struct ImposterAtlas {
    u32 grid_x{8}, grid_y{8}; // cells in atlas (e.g. 8x8 = 64 angles)
    f32 fov_y{k_pi * 0.5f};

    // Map view direction to atlas UV
    [[nodiscard]] Vec2f view_to_uv(const Vec3f& view_dir) const {
        // Octahedral encoding
        Vec3f d = view_dir / (std::abs(view_dir.x)+std::abs(view_dir.y)+std::abs(view_dir.z));
        Vec2f oct;
        if (d.y >= 0.f) {
            oct = {d.x, d.z};
        } else {
            oct = {(1.f-std::abs(d.z))*(d.x>=0.f?1.f:-1.f),
                   (1.f-std::abs(d.x))*(d.z>=0.f?1.f:-1.f)};
        }
        // Map [-1,1] -> [0,1]
        oct.x = oct.x*0.5f+0.5f;
        oct.y = oct.y*0.5f+0.5f;
        // Snap to atlas cell
        u32 cx = u32(std::clamp(oct.x*f32(grid_x), 0.f, f32(grid_x-1)));
        u32 cy = u32(std::clamp(oct.y*f32(grid_y), 0.f, f32(grid_y-1)));
        return {(f32(cx)+0.5f)/f32(grid_x), (f32(cy)+0.5f)/f32(grid_y)};
    }
};

// ─── Cluster-based LOD (nanite-style conceptual) ─────────────────────────────

struct MeshCluster {
    Vec3f center{};
    f32   radius{};
    f32   parent_error{};   // max error in parent cluster
    f32   cluster_error{};  // max error in this cluster
    u32   lod_level{};
};

[[nodiscard]] inline bool cluster_should_render(
    const MeshCluster& c, const Vec3f& cam_pos,
    f32 screen_height, f32 fov_y, f32 error_threshold_px)
{
    f32 dist = std::max(length(c.center - cam_pos), 1e-4f);
    auto proj_err = [&](f32 e) {
        return e / dist * (screen_height / (2.f*std::tan(fov_y*0.5f)));
    };
    return proj_err(c.cluster_error) <= error_threshold_px
        && proj_err(c.parent_error) > error_threshold_px;
}

// ─── Distance-based alpha dithering for LOD transitions ──────────────────────

[[nodiscard]] inline f32 lod_dither_alpha(
    f32 fade_t, Vec2f screen_pos, u32 frame = 0)
{
    // Bayer 4x4 ordered dither
    static constexpr f32 bayer4[4][4] = {
        { 0/16.f,  8/16.f,  2/16.f, 10/16.f},
        {12/16.f,  4/16.f, 14/16.f,  6/16.f},
        { 3/16.f, 11/16.f,  1/16.f,  9/16.f},
        {15/16.f,  7/16.f, 13/16.f,  5/16.f}
    };
    u32 px = u32(screen_pos.x+f32(frame)*1.618f) & 3;
    u32 py = u32(screen_pos.y) & 3;
    return fade_t - bayer4[py][px];
}

} // namespace qs
