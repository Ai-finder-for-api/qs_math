#pragma once
// qs_math/city_gen_math.hpp — Lot subdivision, road networks, block generation (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <span>
#include <queue>

namespace qs {

// ─── Road segment ─────────────────────────────────────────────────────────────

struct RoadSegment {
    Vec2f start{}, end{};
    f32   width{6.f};
    u32   type{0};  // 0=local, 1=arterial, 2=highway
    u32   id{};
};

[[nodiscard]] inline f32 road_length(const RoadSegment& r) {
    return length(r.end-r.start);
}

[[nodiscard]] inline Vec2f road_midpoint(const RoadSegment& r) {
    return (r.start+r.end)*0.5f;
}

[[nodiscard]] inline Vec2f road_dir(const RoadSegment& r) {
    return normalize(r.end-r.start);
}

// ─── Grid road network ───────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<RoadSegment> generate_grid_roads(
    const Vec2f& origin, f32 block_size,
    u32 blocks_x, u32 blocks_y, f32 road_width=6.f)
{
    std::vector<RoadSegment> segs;
    for (u32 y=0;y<=blocks_y;++y) {
        f32 py=origin.y+f32(y)*block_size;
        segs.push_back({
            {origin.x,py},
            {origin.x+f32(blocks_x)*block_size,py},
            road_width, y%4==0?1u:0u,u32(segs.size())});
    }
    for (u32 x=0;x<=blocks_x;++x) {
        f32 px=origin.x+f32(x)*block_size;
        segs.push_back({
            {px,origin.y},
            {px,origin.y+f32(blocks_y)*block_size},
            road_width, x%4==0?1u:0u,u32(segs.size())});
    }
    return segs;
}

// ─── L-System road generation (Parish & Müller) ──────────────────────────────

struct RoadGrowthParams {
    f32  segment_length{20.f};
    f32  branch_angle{90.f};    // degrees
    f32  forward_angle_var{5.f};
    u32  max_depth{8};
    f32  min_segment_len{5.f};
    f32  highway_density{0.3f};
};

struct RoadBranch {
    Vec2f pos{};
    Vec2f dir{};
    u32   depth{};
    u32   type{};  // 0=street, 1=highway
};

[[nodiscard]] inline std::vector<RoadSegment> lsystem_roads(
    const Vec2f& start, const Vec2f& initial_dir,
    const RoadGrowthParams& p, u64 seed=42)
{
    std::vector<RoadSegment> roads;
    std::vector<RoadBranch> queue;
    queue.push_back({start, initial_dir, 0, 0});
    Pcg32 rng{seed};

    while (!queue.empty()) {
        auto branch = queue.back(); queue.pop_back();
        if (branch.depth >= p.max_depth) continue;

        // Perturb direction slightly
        f32 angle_var = (rng.next_f32()-0.5f)*2.f*p.forward_angle_var * k_deg_to_rad;
        f32 c=std::cos(angle_var), s_a=std::sin(angle_var);
        Vec2f dir{branch.dir.x*c-branch.dir.y*s_a, branch.dir.x*s_a+branch.dir.y*c};
        Vec2f end=branch.pos+dir*p.segment_length;
        roads.push_back({branch.pos,end,branch.type==1?10.f:6.f,branch.type,u32(roads.size())});

        // Forward
        queue.push_back({end,dir,branch.depth+1,branch.type});

        // Branch
        if (branch.depth%2==0) {
            f32 ba=p.branch_angle*k_deg_to_rad*(rng.next_f32()>0.5f?1.f:-1.f);
            f32 cb=std::cos(ba), sb=std::sin(ba);
            Vec2f bdir{dir.x*cb-dir.y*sb, dir.x*sb+dir.y*cb};
            if (rng.next_f32()<p.highway_density)
                queue.push_back({end,bdir,branch.depth+1,1});
            else
                queue.push_back({end,bdir,branch.depth+2,0});
        }
    }
    return roads;
}

// ─── City block ──────────────────────────────────────────────────────────────

struct CityBlock {
    std::vector<Vec2f> boundary;   // convex polygon
    u32 zone_type{};               // 0=residential, 1=commercial, 2=industrial, 3=park
    f32 area{};
};

[[nodiscard]] inline f32 polygon_area_2d(std::span<const Vec2f> poly) {
    f32 a=0.f; u32 N=u32(poly.size());
    for (u32 i=0;i<N;++i) {
        const Vec2f& p=poly[i],&q=poly[(i+1)%N];
        a+=p.x*q.y-q.x*p.y;
    }
    return std::abs(a)*0.5f;
}

[[nodiscard]] inline Vec2f polygon_centroid_2d(std::span<const Vec2f> poly) {
    Vec2f c{0,0}; f32 a=0.f; u32 N=u32(poly.size());
    for (u32 i=0;i<N;++i) {
        const Vec2f& p=poly[i],&q=poly[(i+1)%N];
        f32 cross=p.x*q.y-q.x*p.y;
        c.x+=(p.x+q.x)*cross; c.y+=(p.y+q.y)*cross; a+=cross;
    }
    return c*(1.f/(3.f*a));
}

// ─── Lot subdivision ─────────────────────────────────────────────────────────

struct LotParams {
    f32 min_width{8.f};
    f32 min_depth{12.f};
    f32 target_area{200.f};
    f32 irregularity{0.1f};  // boundary noise
};

// Recursive OBB subdivision of a convex polygon into lots
[[nodiscard]] inline std::vector<std::vector<Vec2f>> subdivide_block(
    std::span<const Vec2f> block_poly,
    const LotParams& p, u64 seed=0)
{
    Pcg32 rng{seed};
    f32 area = polygon_area_2d(block_poly);
    if (area < p.target_area*0.5f || block_poly.size() < 3) {
        return {{block_poly.begin(),block_poly.end()}};
    }

    // Split along longest axis
    Vec2f mn=block_poly[0],mx=block_poly[0];
    for (const auto& v:block_poly){
        mn.x=std::min(mn.x,v.x);mn.y=std::min(mn.y,v.y);
        mx.x=std::max(mx.x,v.x);mx.y=std::max(mx.y,v.y);
    }
    Vec2f range=mx-mn;
    f32 split_t=0.4f+rng.next_f32()*0.2f; // 40-60% split
    f32 split_coord, axis_val;
    bool split_x = range.x>range.y;

    std::vector<Vec2f> poly_a, poly_b;
    if (split_x) {
        split_coord=mn.x+split_t*range.x;
        for (u32 i=0;i<u32(block_poly.size());++i) {
            const Vec2f& p0=block_poly[i];
            const Vec2f& p1=block_poly[(i+1)%block_poly.size()];
            if(p0.x<=split_coord) poly_a.push_back(p0);
            else poly_b.push_back(p0);
            // Edge crossing
            if((p0.x<split_coord)!=(p1.x<split_coord)) {
                f32 t=(split_coord-p0.x)/(p1.x-p0.x+1e-8f);
                Vec2f inter{split_coord,p0.y+(p1.y-p0.y)*t};
                poly_a.push_back(inter);
                poly_b.push_back(inter);
            }
        }
    } else {
        split_coord=mn.y+split_t*range.y;
        for (u32 i=0;i<u32(block_poly.size());++i) {
            const Vec2f& p0=block_poly[i];
            const Vec2f& p1=block_poly[(i+1)%block_poly.size()];
            if(p0.y<=split_coord) poly_a.push_back(p0);
            else poly_b.push_back(p0);
            if((p0.y<split_coord)!=(p1.y<split_coord)) {
                f32 t=(split_coord-p0.y)/(p1.y-p0.y+1e-8f);
                Vec2f inter{p0.x+(p1.x-p0.x)*t,split_coord};
                poly_a.push_back(inter);
                poly_b.push_back(inter);
            }
        }
    }

    std::vector<std::vector<Vec2f>> result;
    auto sub_a=subdivide_block(poly_a,p,seed^0xABCD);
    auto sub_b=subdivide_block(poly_b,p,seed^0x1234);
    for(auto&s:sub_a) result.push_back(std::move(s));
    for(auto&s:sub_b) result.push_back(std::move(s));
    return result;
}

// ─── Building footprint ───────────────────────────────────────────────────────

struct BuildingParams {
    f32 setback_front{3.f};
    f32 setback_side{1.5f};
    f32 setback_rear{4.f};
    f32 footprint_ratio{0.6f}; // lot coverage
    f32 height_min{3.f};
    f32 height_max{30.f};
    f32 floor_height{3.f};
};

[[nodiscard]] inline std::vector<Vec2f> building_footprint(
    std::span<const Vec2f> lot, const BuildingParams& p, u64 seed=0)
{
    if (lot.size()<3) return {};
    Vec2f mn=lot[0],mx=lot[0];
    for(const auto&v:lot){mn.x=std::min(mn.x,v.x);mn.y=std::min(mn.y,v.y);mx.x=std::max(mx.x,v.x);mx.y=std::max(mx.y,v.y);}
    // Erode lot by setbacks (simplified: shrink AABB)
    Vec2f inner_mn{mn.x+p.setback_side, mn.y+p.setback_front};
    Vec2f inner_mx{mx.x-p.setback_side, mx.y-p.setback_rear};
    if(inner_mn.x>=inner_mx.x||inner_mn.y>=inner_mx.y) return {};
    // Scale to footprint ratio
    Vec2f range=inner_mx-inner_mn;
    f32 scale=std::sqrt(p.footprint_ratio);
    Vec2f center=(inner_mn+inner_mx)*0.5f;
    Vec2f half=range*scale*0.5f;
    return {{center.x-half.x,center.y-half.y},{center.x+half.x,center.y-half.y},
            {center.x+half.x,center.y+half.y},{center.x-half.x,center.y+half.y}};
}

// ─── Height field population density ─────────────────────────────────────────

[[nodiscard]] inline f32 building_height(
    const Vec2f& pos, const Vec2f& city_center,
    const BuildingParams& p, u64 seed=0)
{
    Pcg32 rng{seed};
    f32 dist=length(pos-city_center);
    f32 max_r=500.f;
    f32 density=std::exp(-dist/max_r);
    f32 h=p.height_min+(p.height_max-p.height_min)*density*(0.5f+rng.next_f32()*0.5f);
    return std::floor(h/p.floor_height)*p.floor_height;
}

} // namespace qs
