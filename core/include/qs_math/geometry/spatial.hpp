#pragma once
// qs_math/spatial.hpp — Morton encoding, QuadTree/Octree index helpers, Grid, KD-tree helpers
#include "hashing.hpp"
#include "geometry3d.hpp"

namespace qs::math::spatial {

// ---------------------------------------------------------------------------
// Grid indexing
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr i32 grid_index_2d(i32 x, i32 y, i32 width) noexcept { return y*width+x; }
[[nodiscard]] constexpr Vec2i grid_coord_2d(i32 idx, i32 width) noexcept { return {idx%width, idx/width}; }
[[nodiscard]] constexpr i32 grid_index_3d(i32 x,i32 y,i32 z,i32 w,i32 h) noexcept { return z*w*h+y*w+x; }
[[nodiscard]] constexpr Vec3i grid_coord_3d(i32 idx,i32 w,i32 h) noexcept {
    return {idx%w,(idx/w)%h,idx/(w*h)};
}

// Grid neighbor offsets (4-connected, 8-connected, 6-face, 26-neighbor)
inline constexpr Vec2i NEIGHBORS_4[4]={{1,0},{-1,0},{0,1},{0,-1}};
inline constexpr Vec2i NEIGHBORS_8[8]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}};
inline constexpr Vec3i NEIGHBORS_6[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

// ---------------------------------------------------------------------------
// Quadtree node helpers
// ---------------------------------------------------------------------------
// Linear quadtree: child of node at depth d, index i, quadrant q(0-3)
[[nodiscard]] constexpr u64 quadtree_child(u64 node, i32 quadrant) noexcept { return node*4+quadrant+1; }
[[nodiscard]] constexpr u64 quadtree_parent(u64 node) noexcept { return (node-1)/4; }
[[nodiscard]] constexpr i32 quadtree_depth(u64 node) noexcept {
    i32 d=0; u64 n=node+1; while(n>1){n=(n+3)/4;++d;} return d;
}
[[nodiscard]] constexpr Vec2i quadtree_quadrant_offset(i32 q) noexcept {
    return {q&1,(q>>1)&1};
}

// Octree
[[nodiscard]] constexpr u64 octree_child(u64 node, i32 oct) noexcept { return node*8+oct+1; }
[[nodiscard]] constexpr u64 octree_parent(u64 node) noexcept { return (node-1)/8; }
[[nodiscard]] constexpr Vec3i octree_octant_offset(i32 o) noexcept {
    return {o&1,(o>>1)&1,(o>>2)&1};
}

// ---------------------------------------------------------------------------
// KD-tree helper types (build data — actual tree storage is in qs_asset)
// ---------------------------------------------------------------------------
struct KDSplit {
    i32 axis{0};  // 0=x, 1=y, 2=z
    f32 pos{0.f};
};

[[nodiscard]] constexpr KDSplit best_split_axis(AABBf bounds) noexcept {
    Vec3f size=bounds.size();
    i32 axis=0;
    if(size.y>size.x&&size.y>size.z) axis=1;
    else if(size.z>size.x&&size.z>size.y) axis=2;
    f32 mid=(axis==0)?bounds.center().x:(axis==1)?bounds.center().y:bounds.center().z;
    return {axis,mid};
}

// Surface area heuristic for KD-tree
[[nodiscard]] constexpr f32 kd_sah(AABBf parent, AABBf left, AABBf right,
                                    i32 n_left, i32 n_right) noexcept {
    f32 pa=parent.surface_area();
    if(pa<EPS_F32) return F32_INF;
    return (left.surface_area()/pa)*n_left + (right.surface_area()/pa)*n_right;
}

// ---------------------------------------------------------------------------
// Frustum vs grid cell — fast coarse cull
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr bool grid_cell_in_frustum(Vec3i cell, f32 cell_size, const Frustumf& frustum) noexcept {
    Vec3f mn{f32(cell.x)*cell_size, f32(cell.y)*cell_size, f32(cell.z)*cell_size};
    Vec3f mx=mn+Vec3f{cell_size};
    return frustum.intersects_aabb(AABBf{mn,mx});
}

} // namespace qs::math::spatial
