#pragma once
// render_math.hpp — Rendering mathematics: rasterization, z-buffer,
//                   tile-based rendering, visibility, LOD, batching math,
//                   render state math, draw call sorting, indirect draw
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "geometry3d.hpp"
#include "scalar_math.hpp"
#include "color_science.hpp"

namespace qs::math::render {

// ---------------------------------------------------------------------------
// Tile math (tiled forward / deferred rendering)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec2i screen_to_tile(Vec2f screen_pos, i32 tile_size=16) noexcept {
    return {i32(screen_pos.x)/tile_size, i32(screen_pos.y)/tile_size};
}
[[nodiscard]] constexpr i32 tile_index(Vec2i tile, i32 tiles_x) noexcept {
    return tile.y*tiles_x+tile.x;
}
[[nodiscard]] constexpr i32 tile_count(i32 screen_w, i32 screen_h, i32 tile_size=16) noexcept {
    return ((screen_w+tile_size-1)/tile_size)*((screen_h+tile_size-1)/tile_size);
}
[[nodiscard]] constexpr Rect2f tile_screen_rect(Vec2i tile, i32 tile_size=16) noexcept {
    Vec2f min{f32(tile.x*tile_size),f32(tile.y*tile_size)};
    return {min, min+Vec2f{f32(tile_size)}};
}

// Build tile frustum for light culling
[[nodiscard]] inline Frustumf tile_frustum(
    Vec2i tile, i32 tile_size, i32 screen_w, i32 screen_h,
    f32 near_z, f32 far_z, Mat4f inv_proj) noexcept
{
    // NDC bounds of tile
    f32 tw=f32(tile_size)/screen_w, th=f32(tile_size)/screen_h;
    f32 tx=-1.f+f32(tile.x)*2.f*tw, ty=-1.f+f32(tile.y)*2.f*th;
    // Build planes from tile corners
    Frustumf f;
    // Simplified: use screen-space tile rect projected to view space
    f.planes[4]={Vec3f{0,0,-1},near_z}; // near
    f.planes[5]={Vec3f{0,0, 1},-far_z}; // far
    // Left, right, bottom, top from tile NDC
    f.planes[0]={Vec3f{ 1,0,-(tx)}.normalized(),0.f};
    f.planes[1]={Vec3f{-1,0, tx+2.f*tw}.normalized(),0.f};
    f.planes[2]={Vec3f{0, 1,-(ty)}.normalized(),0.f};
    f.planes[3]={Vec3f{0,-1, ty+2.f*th}.normalized(),0.f};
    return f;
}

// ---------------------------------------------------------------------------
// Z-buffer math
// ---------------------------------------------------------------------------
// Perspective-correct depth linearization
[[nodiscard]] constexpr f32 ndc_to_linear_depth(f32 ndc_z, f32 near, f32 far) noexcept {
    return 2.f*near*far/(far+near-ndc_z*(far-near));
}

// Reversed-Z depth (near=1, far=0 — better precision)
[[nodiscard]] constexpr f32 linear_to_reversed_z(f32 linear_z, f32 near, f32 far) noexcept {
    return near/(linear_z*(1.f-near/far)+near/far);
}

// Hi-Z occlusion check: is a sphere occluded by a coarser depth value?
[[nodiscard]] constexpr bool hiz_occluded(
    f32 sphere_min_z, f32 hiz_depth) noexcept
{
    return sphere_min_z>hiz_depth; // reversed-Z: closer = larger value
}

// ---------------------------------------------------------------------------
// LOD selection
// ---------------------------------------------------------------------------
struct LODRange {
    f32 distances[8];   // max distance for each LOD level
    i32 count{4};

    [[nodiscard]] constexpr i32 select_lod(f32 distance) const noexcept {
        for(i32 i=0;i<count;++i) if(distance<=distances[i]) return i;
        return count; // culled
    }
};

// Projected screen size of bounding sphere
[[nodiscard]] constexpr f32 projected_sphere_size(
    f32 world_radius, f32 distance_to_center, f32 fov_y, f32 screen_height) noexcept
{
    if(distance_to_center<EPS_F32) return screen_height;
    f32 angular_size=atan(world_radius/distance_to_center)*2.f;
    return angular_size/(fov_y)*screen_height;
}

// ---------------------------------------------------------------------------
// Draw call sorting key
// ---------------------------------------------------------------------------
union DrawSortKey {
    u64 value{0};
    struct {
        u64 depth       : 24; // quantized depth (front-to-back for opaque)
        u64 material_id : 16;
        u64 mesh_id     : 12;
        u64 layer       : 8;
        u64 translucent : 1;
        u64 shadow_cast : 1;
        u64 reserved    : 2;
    };
};

[[nodiscard]] constexpr DrawSortKey make_sort_key(
    f32 depth, i32 mat_id, i32 mesh_id, i32 layer, bool translucent) noexcept
{
    DrawSortKey k;
    // Quantize depth to 24 bits
    u32 depth_bits=u32(saturate(depth)*16777215.f);
    // For opaque: front-to-back (smaller depth = earlier draw)
    // For translucent: back-to-front (larger depth = earlier draw)
    k.depth       =translucent?16777215-depth_bits:depth_bits;
    k.material_id =u64(mat_id&0xFFFF);
    k.mesh_id     =u64(mesh_id&0xFFF);
    k.layer       =u64(layer&0xFF);
    k.translucent =translucent?1:0;
    return k;
}

// ---------------------------------------------------------------------------
// Clustered shading math
// ---------------------------------------------------------------------------
struct ClusterGrid {
    i32 tiles_x{16}, tiles_y{9}, slices_z{24};
    f32 near_z, far_z;
    Vec2f screen_size;

    [[nodiscard]] constexpr i32 total_clusters() const noexcept { return tiles_x*tiles_y*slices_z; }

    // Convert linear depth to cluster Z slice
    [[nodiscard]] constexpr i32 depth_to_slice(f32 view_z) const noexcept {
        // Exponential slicing: slice = floor(log(z/near) / log(far/near) * slices_z)
        f32 ratio=log(abs(view_z)/near_z)/log(far_z/near_z);
        return clamp(i32(ratio*slices_z),0,slices_z-1);
    }

    [[nodiscard]] constexpr i32 cluster_index(i32 tile_x, i32 tile_y, i32 slice_z) const noexcept {
        return slice_z*(tiles_x*tiles_y)+tile_y*tiles_x+tile_x;
    }
    [[nodiscard]] constexpr i32 cluster_for_position(
        Vec2f screen_pos, f32 view_z, i32 tile_size=16) const noexcept
    {
        i32 tx=clamp(i32(screen_pos.x)/tile_size,0,tiles_x-1);
        i32 ty=clamp(i32(screen_pos.y)/tile_size,0,tiles_y-1);
        i32 tz=depth_to_slice(view_z);
        return cluster_index(tx,ty,tz);
    }
};

// ---------------------------------------------------------------------------
// Indirect draw math
// ---------------------------------------------------------------------------
struct DrawIndirectArgs {
    u32 vertex_count;
    u32 instance_count;
    u32 first_vertex;
    u32 first_instance;
};
struct DrawIndexedIndirectArgs {
    u32 index_count;
    u32 instance_count;
    u32 first_index;
    i32 vertex_offset;
    u32 first_instance;
};

// Pack instance data into GPU buffer
struct InstanceData {
    Mat4f transform;
    Vec4f color;
    Vec4f params; // x=lod, y=material_id, z=custom0, w=custom1
};

// ---------------------------------------------------------------------------
// Cascaded shadow map math
// ---------------------------------------------------------------------------
struct CSMCascade {
    f32     z_near, z_far;
    Mat4f   light_view_proj;
    Vec2f   texel_size;
};

// Compute cascade splits (logarithmic + practical)
inline void compute_csm_splits(
    f32 near_z, f32 far_z, i32 n_cascades,
    f32* splits, f32 lambda=0.7f) noexcept
{
    for(i32 i=0;i<n_cascades;++i){
        f32 t=f32(i+1)/n_cascades;
        f32 log_split=near_z*pow(far_z/near_z,t);
        f32 lin_split=near_z+(far_z-near_z)*t;
        splits[i]=lerp(lin_split,log_split,lambda);
    }
}

// Compute light frustum for a CSM cascade
[[nodiscard]] inline Mat4f cascade_light_matrix(
    Vec3f light_dir, const Vec3f* frustum_corners, i32 n,
    f32 texel_snap=0.f) noexcept
{
    Vec3f center{0.f};
    for(i32 i=0;i<n;++i) center=center+frustum_corners[i];
    center=center*(1.f/n);

    Mat4f light_view=Mat4f::look_at(center-light_dir,center,Vec3f::up());

    // Transform corners to light space and find AABB
    AABBf light_aabb{};
    for(i32 i=0;i<n;++i){
        Vec3f lp=light_view.transform_point(frustum_corners[i]);
        light_aabb=(i==0)?AABBf{lp,lp}:light_aabb.merge(lp);
    }

    // Snap to texel size to reduce shadow shimmer
    if(texel_snap>EPS_F32){
        light_aabb.min_pt.x=floor(light_aabb.min_pt.x/texel_snap)*texel_snap;
        light_aabb.min_pt.y=floor(light_aabb.min_pt.y/texel_snap)*texel_snap;
        light_aabb.max_pt.x=ceil(light_aabb.max_pt.x/texel_snap)*texel_snap;
        light_aabb.max_pt.y=ceil(light_aabb.max_pt.y/texel_snap)*texel_snap;
    }

    Vec3f s=light_aabb.size();
    Mat4f light_proj=Mat4f::orthographic(
        light_aabb.min_pt.x,light_aabb.max_pt.x,
        light_aabb.min_pt.y,light_aabb.max_pt.y,
        light_aabb.min_pt.z-s.z*0.5f,light_aabb.max_pt.z+s.z*0.5f);

    return light_proj*light_view;
}

// ---------------------------------------------------------------------------
// Ambient occlusion math
// ---------------------------------------------------------------------------
// GTAO bent normal and AO factor
[[nodiscard]] inline f32 gtao_integrate_horizon(
    f32* h, f32 N_angle, i32 n_samples) noexcept
{
    f32 ao=0.f;
    for(i32 i=0;i<n_samples;++i){
        f32 d=h[i]-N_angle;
        ao+=1.f-cos(max(0.f,d));
    }
    return 1.f-ao/n_samples;
}

// Bent normal from AO samples
[[nodiscard]] inline Vec3f bent_normal(
    Vec3f N, Vec3f T, Vec3f B,
    const Vec2f* sample_dirs, const f32* ao_values, i32 n) noexcept
{
    Vec3f bent{0.f};
    for(i32 i=0;i<n;++i){
        f32 theta=sample_dirs[i].x*HALF_PI;
        f32 phi=sample_dirs[i].y*TWO_PI;
        Vec3f dir=N*cos(theta)+T*sin(theta)*cos(phi)+B*sin(theta)*sin(phi);
        bent=bent+dir*ao_values[i];
    }
    return bent.length()>EPS_F32?bent.normalized():N;
}

} // namespace qs::math::render
