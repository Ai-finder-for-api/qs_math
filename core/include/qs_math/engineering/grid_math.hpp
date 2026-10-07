#pragma once
// qs_math/grid_math.hpp — World-space grid, polar grid, isometric, hex grid (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <numbers>
#include <tuple>

namespace qs {

// ─── Cartesian grid ───────────────────────────────────────────────────────────
struct Grid2D_Config {
    Vec2f origin{};
    f32   cell_size{1.f};
    u32   subdivisions{1};
};

// Snap world position to nearest grid cell center
[[nodiscard]] inline Vec2f snap_to_grid_2d(Vec2f pos, f32 cell_size, Vec2f origin={0,0}) {
    Vec2f local=pos-origin;
    return origin+Vec2f{std::round(local.x/cell_size)*cell_size,
                        std::round(local.y/cell_size)*cell_size};
}

// World position to grid cell indices
[[nodiscard]] inline Vec2i world_to_cell(Vec2f pos, f32 cell_size, Vec2f origin={0,0}) {
    Vec2f local=pos-origin;
    return {i32(std::floor(local.x/cell_size)), i32(std::floor(local.y/cell_size))};
}
[[nodiscard]] inline Vec2f cell_to_world(Vec2i cell, f32 cell_size, Vec2f origin={0,0}) {
    return origin+Vec2f{(cell.x+0.5f)*cell_size,(cell.y+0.5f)*cell_size};
}

// Grid line vertices for rendering (returns pairs of start/end points)
[[nodiscard]] inline std::vector<std::pair<Vec3f,Vec3f>> grid_lines(
    Vec2f min_corner, Vec2f max_corner, f32 cell_size, f32 y=0)
{
    std::vector<std::pair<Vec3f,Vec3f>> lines;
    for(f32 x=min_corner.x;x<=max_corner.x+cell_size*0.5f;x+=cell_size)
        lines.push_back({{x,y,min_corner.y},{x,y,max_corner.y}});
    for(f32 z=min_corner.y;z<=max_corner.y+cell_size*0.5f;z+=cell_size)
        lines.push_back({{min_corner.x,y,z},{max_corner.x,y,z}});
    return lines;
}

// ─── Polar / radial grid ──────────────────────────────────────────────────────
[[nodiscard]] inline Vec2f polar_to_cartesian(f32 r, f32 theta) {
    return {r*std::cos(theta), r*std::sin(theta)};
}
[[nodiscard]] inline std::pair<f32,f32> cartesian_to_polar(Vec2f p) {
    return {length(p), std::atan2(p.y,p.x)};
}
// Snap to nearest polar grid point
[[nodiscard]] inline Vec2f snap_polar(Vec2f p, f32 r_step, f32 theta_step_deg) {
    auto [r,th]=cartesian_to_polar(p);
    f32 ts=theta_step_deg*f32(std::numbers::pi)/180.f;
    r=std::round(r/r_step)*r_step;
    th=std::round(th/ts)*ts;
    return polar_to_cartesian(r,th);
}

// ─── Isometric grid ───────────────────────────────────────────────────────────
// Standard 2:1 isometric projection
[[nodiscard]] inline Vec2f iso_to_screen(Vec2f tile, f32 tile_w=64, f32 tile_h=32) {
    return {(tile.x-tile.y)*tile_w*0.5f, (tile.x+tile.y)*tile_h*0.5f};
}
[[nodiscard]] inline Vec2f screen_to_iso(Vec2f screen, f32 tile_w=64, f32 tile_h=32) {
    f32 x=screen.x/(tile_w*0.5f), y=screen.y/(tile_h*0.5f);
    return {(x+y)*0.5f, (y-x)*0.5f};
}
// Isometric tile under screen point (floor to tile)
[[nodiscard]] inline Vec2i screen_to_iso_tile(Vec2f screen, f32 tile_w=64, f32 tile_h=32) {
    Vec2f iso=screen_to_iso(screen,tile_w,tile_h);
    return {i32(std::floor(iso.x)),i32(std::floor(iso.y))};
}
// Isometric depth sort key (painter's algorithm): sort by x+y
[[nodiscard]] inline f32 iso_depth(Vec2i tile) { return f32(tile.x+tile.y); }

// ─── Hexagonal grid (axial coordinates) ───────────────────────────────────────
// Pointy-top hexagons using cube coordinates (q,r,s) where q+r+s=0
struct HexCoord {
    i32 q{0},r{0};
    [[nodiscard]] i32 s() const { return -q-r; }
    [[nodiscard]] bool operator==(const HexCoord& o) const { return q==o.q&&r==o.r; }
};

// Distance between two hex cells
[[nodiscard]] inline i32 hex_distance(HexCoord a, HexCoord b) {
    return (std::abs(a.q-b.q)+std::abs(a.q-b.q+a.r-b.r)+std::abs(a.r-b.r))/2;
}

// Hex to world position (pointy-top)
[[nodiscard]] inline Vec2f hex_to_world(HexCoord h, f32 size) {
    return {size*(std::numbers::sqrt3_v<f32>*h.q+std::numbers::sqrt3_v<f32>/2*h.r),
            size*(1.5f*h.r)};
}
// Flat-top variant
[[nodiscard]] inline Vec2f hex_to_world_flat(HexCoord h, f32 size) {
    return {size*(1.5f*h.q),
            size*(std::numbers::sqrt3_v<f32>/2*h.q+std::numbers::sqrt3_v<f32>*h.r)};
}

// World to hex (pointy-top) — fractional
[[nodiscard]] inline HexCoord world_to_hex(Vec2f p, f32 size) {
    f32 q=(std::numbers::sqrt3_v<f32>/3*p.x-1.f/3*p.y)/size;
    f32 r=(2.f/3*p.y)/size;
    // Round to nearest hex
    f32 s=-q-r;
    i32 rq=i32(std::round(q)),rr=i32(std::round(r)),rs=i32(std::round(s));
    f32 dq=std::abs(q-rq),dr=std::abs(r-rr),ds=std::abs(s-rs);
    if(dq>dr&&dq>ds) rq=-rr-rs;
    else if(dr>ds)   rr=-rq-rs;
    return {rq,rr};
}

// Hex neighbours (6 directions)
static constexpr HexCoord HEX_DIRS[6]={{1,0},{1,-1},{0,-1},{-1,0},{-1,1},{0,1}};
[[nodiscard]] inline HexCoord hex_neighbour(HexCoord h, u32 dir) {
    return {h.q+HEX_DIRS[dir].q, h.r+HEX_DIRS[dir].r};
}
[[nodiscard]] inline std::array<HexCoord,6> hex_neighbours(HexCoord h) {
    std::array<HexCoord,6> nb;
    for(u32 i=0;i<6;++i) nb[i]=hex_neighbour(h,i);
    return nb;
}

// Hex ring at radius R
[[nodiscard]] inline std::vector<HexCoord> hex_ring(HexCoord center, u32 R) {
    if(R==0) return {center};
    std::vector<HexCoord> ring;
    HexCoord h{center.q+HEX_DIRS[4].q*i32(R), center.r+HEX_DIRS[4].r*i32(R)};
    for(u32 i=0;i<6;++i) for(u32 j=0;j<R;++j) { ring.push_back(h); h=hex_neighbour(h,i); }
    return ring;
}

// Hex spiral (all hexes within radius R)
[[nodiscard]] inline std::vector<HexCoord> hex_spiral(HexCoord center, u32 R) {
    std::vector<HexCoord> result={center};
    for(u32 r=1;r<=R;++r) { auto ring=hex_ring(center,r); result.insert(result.end(),ring.begin(),ring.end()); }
    return result;
}

// Hex field of view (simple cone)
[[nodiscard]] inline std::vector<HexCoord> hex_fov(HexCoord origin, u32 radius,
    const std::function<bool(HexCoord)>& blocks)
{
    auto all=hex_spiral(origin,radius);
    std::vector<HexCoord> visible;
    for(auto h:all) {
        // Line-of-sight along hex line from origin to h
        i32 n=hex_distance(origin,h);
        bool los=true;
        for(i32 step=1;step<n;++step) {
            f32 t=f32(step)/n;
            HexCoord mid=world_to_hex(hex_to_world(origin,1)*( 1-t)+hex_to_world(h,1)*t,1);
            if(blocks(mid)){los=false;break;}
        }
        if(los) visible.push_back(h);
    }
    return visible;
}

// ─── 3D voxel grid ────────────────────────────────────────────────────────────
struct VoxelCoord { i32 x,y,z; };

[[nodiscard]] inline VoxelCoord world_to_voxel(Vec3f p, f32 voxel_size) {
    return {i32(std::floor(p.x/voxel_size)),i32(std::floor(p.y/voxel_size)),i32(std::floor(p.z/voxel_size))};
}
[[nodiscard]] inline Vec3f voxel_to_world(VoxelCoord v, f32 vs) {
    return {(v.x+0.5f)*vs,(v.y+0.5f)*vs,(v.z+0.5f)*vs};
}

// DDA (Digital Differential Analyzer) voxel traversal along a ray
struct DDAResult { VoxelCoord cell; Vec3f t_enter,t_exit; i32 face_normal_axis{0}; i32 face_normal_sign{0}; };
[[nodiscard]] inline std::vector<DDAResult> dda_voxel_traverse(
    Vec3f ray_origin, Vec3f ray_dir, f32 voxel_size, u32 max_steps=100)
{
    std::vector<DDAResult> results;
    Vec3f inv_dir{1.f/(ray_dir.x+1e-20f),1.f/(ray_dir.y+1e-20f),1.f/(ray_dir.z+1e-20f)};
    VoxelCoord cur=world_to_voxel(ray_origin,voxel_size);
    Vec3f step={ray_dir.x>0?1.f:-1.f,ray_dir.y>0?1.f:-1.f,ray_dir.z>0?1.f:-1.f};
    Vec3f cell_bound{(step.x>0?(cur.x+1)*voxel_size:cur.x*voxel_size),
                     (step.y>0?(cur.y+1)*voxel_size:cur.y*voxel_size),
                     (step.z>0?(cur.z+1)*voxel_size:cur.z*voxel_size)};
    Vec3f t_max{(cell_bound.x-ray_origin.x)*inv_dir.x,
                (cell_bound.y-ray_origin.y)*inv_dir.y,
                (cell_bound.z-ray_origin.z)*inv_dir.z};
    Vec3f t_delta{voxel_size*std::abs(inv_dir.x),voxel_size*std::abs(inv_dir.y),voxel_size*std::abs(inv_dir.z)};
    for(u32 s=0;s<max_steps;++s) {
        DDAResult r; r.cell=cur;
        results.push_back(r);
        if(t_max.x<t_max.y&&t_max.x<t_max.z){ cur.x+=i32(step.x); t_max.x+=t_delta.x; r.face_normal_axis=0; r.face_normal_sign=i32(-step.x); }
        else if(t_max.y<t_max.z)              { cur.y+=i32(step.y); t_max.y+=t_delta.y; r.face_normal_axis=1; r.face_normal_sign=i32(-step.y); }
        else                                   { cur.z+=i32(step.z); t_max.z+=t_delta.z; r.face_normal_axis=2; r.face_normal_sign=i32(-step.z); }
    }
    return results;
}

} // namespace qs
