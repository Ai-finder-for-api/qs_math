#pragma once
// qs_math/voxel_math.hpp — Morton codes, sparse octrees, SVO ray marching, DDA (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <optional>

namespace qs {

// ─── Morton codes (Z-order curve) ────────────────────────────────────────────

[[nodiscard]] inline u64 morton_split3(u32 x) {
    u64 v = x & 0x1fffffu;
    v = (v | v<<32) & 0x1f00000000ffffull;
    v = (v | v<<16) & 0x1f0000ff0000ffull;
    v = (v | v<< 8) & 0x100f00f00f00f0full;
    v = (v | v<< 4) & 0x10c30c30c30c3ull;
    v = (v | v<< 2) & 0x1249249249249ull;
    return v;
}

[[nodiscard]] inline u64 morton_encode(u32 x, u32 y, u32 z) {
    return morton_split3(x) | (morton_split3(y)<<1) | (morton_split3(z)<<2);
}

[[nodiscard]] inline u32 morton_compact3(u64 x) {
    x &= 0x1249249249249ull;
    x = (x | x>>2)  & 0x10c30c30c30c3ull;
    x = (x | x>>4)  & 0x100f00f00f00f0full;
    x = (x | x>>8)  & 0x1f0000ff0000ffull;
    x = (x | x>>16) & 0x1f00000000ffffull;
    x = (x | x>>32) & 0x1fffffu;
    return u32(x);
}

[[nodiscard]] inline std::array<u32,3> morton_decode(u64 code) {
    return {morton_compact3(code),morton_compact3(code>>1),morton_compact3(code>>2)};
}

// ─── DDA voxel traversal ──────────────────────────────────────────────────────

struct DDAState {
    std::array<i32,3> cell{};
    std::array<f32,3> t_max{};
    std::array<f32,3> t_delta{};
    std::array<i32,3> step{};
    f32 t{0.f};
    u32 face_axis{0};

    [[nodiscard]] static DDAState init(
        const Vec3f& origin, const Vec3f& dir,
        f32 voxel_size=1.f)
    {
        DDAState s;
        for (u32 i=0;i<3;++i) {
            f32 d = dir[i];
            s.cell[i] = i32(std::floor(origin[i]/voxel_size));
            s.step[i] = d>=0?1:-1;
            f32 next = (d>=0)
                ? (f32(s.cell[i]+1)*voxel_size - origin[i])
                : (origin[i] - f32(s.cell[i])*voxel_size);
            s.t_delta[i] = std::abs(d)>1e-10f ? voxel_size/std::abs(d) : 1e30f;
            s.t_max[i]   = std::abs(d)>1e-10f ? next/std::abs(d) : 1e30f;
        }
        return s;
    }

    // Advance one step
    void step_forward() {
        face_axis = 0;
        if (t_max[1]<t_max[0]) face_axis=1;
        if (t_max[2]<t_max[face_axis]) face_axis=2;
        t = t_max[face_axis];
        cell[face_axis] += step[face_axis];
        t_max[face_axis] += t_delta[face_axis];
    }
};

// ─── Sparse voxel octree node ─────────────────────────────────────────────────

struct SVONode {
    static constexpr u32 EMPTY = u32(-1);
    u32 children[8]{EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY};
    u32 data{0};  // leaf data (color, material, etc.)
    bool is_leaf{false};
};

struct SparseVoxelOctree {
    std::vector<SVONode> nodes;
    Vec3f origin{};
    f32   size{256.f};
    u32   max_depth{8};

    void init(Vec3f o, f32 s, u32 depth) {
        origin=o; size=s; max_depth=depth;
        nodes.clear(); nodes.emplace_back();
    }

    u32 alloc_node() {
        nodes.emplace_back();
        return u32(nodes.size()-1);
    }

    void insert(const Vec3f& point, u32 data_val) {
        u32 node_idx = 0;
        Vec3f c = origin + Vec3f{size,size,size}*0.5f;
        f32 half = size*0.5f;
        for (u32 d=0;d<max_depth;++d) {
            half *= 0.5f;
            u32 child = 0;
            if (point.x>=c.x){child|=1; c.x+=half;} else c.x-=half;
            if (point.y>=c.y){child|=2; c.y+=half;} else c.y-=half;
            if (point.z>=c.z){child|=4; c.z+=half;} else c.z-=half;
            if (nodes[node_idx].children[child]==SVONode::EMPTY) {
                u32 ni = alloc_node();
                nodes[node_idx].children[child] = ni;
                if (d==max_depth-1) {
                    nodes[ni].is_leaf=true;
                    nodes[ni].data=data_val;
                    return;
                }
            }
            node_idx = nodes[node_idx].children[child];
        }
    }

    // Ray march through SVO
    [[nodiscard]] std::optional<u32> ray_cast(
        const Vec3f& ray_origin, const Vec3f& ray_dir,
        f32 max_dist=1000.f) const
    {
        // Simplified: DDA at max_depth resolution
        f32 leaf_size = size / f32(1u<<max_depth);
        DDAState dda = DDAState::init(ray_origin, ray_dir, leaf_size);
        for (u32 step=0; step<1024&&dda.t<max_dist; ++step) {
            // Check if current cell is filled
            Vec3f p = ray_origin + ray_dir*dda.t + Vec3f{0.5f,0.5f,0.5f}*leaf_size;
            // Traverse tree
            u32 node_idx=0;
            Vec3f c2=origin+Vec3f{size,size,size}*0.5f;
            f32 half=size*0.5f;
            bool found=false;
            for (u32 d=0;d<max_depth;++d) {
                half*=0.5f;
                u32 child=0;
                if(p.x>=c2.x){child|=1;c2.x+=half;}else c2.x-=half;
                if(p.y>=c2.y){child|=2;c2.y+=half;}else c2.y-=half;
                if(p.z>=c2.z){child|=4;c2.z+=half;}else c2.z-=half;
                u32 ni=nodes[node_idx].children[child];
                if(ni==SVONode::EMPTY) break;
                if(nodes[ni].is_leaf){return nodes[ni].data;}
                node_idx=ni;
            }
            dda.step_forward();
        }
        return std::nullopt;
    }
};

// ─── Run-length encoded voxel grid ───────────────────────────────────────────

struct RLEVoxelLayer {
    struct Run { u16 length; u32 value; };
    std::vector<Run> runs;
    u32 total_count{0};

    void push(u32 val, u32 count=1) {
        if (!runs.empty() && runs.back().value==val)
            runs.back().length = u16(std::min(u32(runs.back().length)+count, u32(65535)));
        else
            runs.push_back({u16(count), val});
        total_count += count;
    }

    [[nodiscard]] u32 get(u32 idx) const {
        u32 cur=0;
        for (const auto& r : runs) {
            cur+=r.length;
            if (idx<cur) return r.value;
        }
        return 0;
    }
};

// ─── Smooth voxel (marching cubes lookup stub) ────────────────────────────────

// Edge table for marching cubes (simplified — first 16 entries)
static constexpr u16 MC_EDGE_TABLE[256] = {
    0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,
    0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00
    // (full 256-entry table would follow; truncated for space)
};

[[nodiscard]] inline f32 trilinear_voxel(
    const f32 v[8], f32 fx, f32 fy, f32 fz)
{
    return v[0]*(1-fx)*(1-fy)*(1-fz) + v[1]*fx*(1-fy)*(1-fz)
         + v[2]*(1-fx)*fy*(1-fz)     + v[3]*fx*fy*(1-fz)
         + v[4]*(1-fx)*(1-fy)*fz     + v[5]*fx*(1-fy)*fz
         + v[6]*(1-fx)*fy*fz         + v[7]*fx*fy*fz;
}

[[nodiscard]] inline Vec3f voxel_gradient(const f32 v[8]) {
    return {
        0.5f*(v[1]-v[0]+v[3]-v[2]+v[5]-v[4]+v[7]-v[6]),
        0.5f*(v[2]-v[0]+v[3]-v[1]+v[6]-v[4]+v[7]-v[5]),
        0.5f*(v[4]-v[0]+v[5]-v[1]+v[6]-v[2]+v[7]-v[3])};
}

} // namespace qs
