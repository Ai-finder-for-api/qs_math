#pragma once
// qs_math/render_graph_math.hpp — Pass dependency, resource aliasing, bandwidth estimation (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <span>
#include <array>
#include <optional>
#include <bitset>

namespace qs {

// ─── Resource descriptor ──────────────────────────────────────────────────────

enum class ResourceType { Texture2D, Texture3D, Buffer, RenderTarget, DepthStencil };
enum class ResourceUsage : u32 { Read=1, Write=2, RenderTarget=4, DepthWrite=8, Shader=16 };

struct ResourceDesc {
    std::string name{};
    ResourceType type{ResourceType::Texture2D};
    u32  width{}, height{}, depth{1};
    u32  mip_levels{1};
    u32  format_bytes{4};    // bytes per pixel/element
    bool transient{true};    // can be aliased
    u32  id{};
};

[[nodiscard]] inline u64 resource_size(const ResourceDesc& r) {
    u64 mip_factor = 0;
    u64 s = 1;
    for(u32 m=0;m<r.mip_levels;++m){mip_factor+=s;s=(s+3)/4;}
    return u64(r.width)*u64(r.height)*u64(r.depth)*u64(r.format_bytes)*mip_factor;
}

// ─── Render pass ─────────────────────────────────────────────────────────────

struct PassResource {
    u32 resource_id{};
    ResourceUsage usage{};
};

struct RenderPass {
    std::string name{};
    u32  id{};
    std::vector<PassResource> inputs, outputs;
    f32  estimated_ms{0.f};    // GPU time estimate
    bool async_compute{false};

    void read(u32 rid)  { inputs.push_back({rid,ResourceUsage::Read}); }
    void write(u32 rid) { outputs.push_back({rid,ResourceUsage::Write}); }
    void render_target(u32 rid) { outputs.push_back({rid,ResourceUsage::RenderTarget}); }
};

// ─── Dependency graph ─────────────────────────────────────────────────────────

struct RenderGraph {
    std::vector<RenderPass>    passes;
    std::vector<ResourceDesc>  resources;

    u32 add_resource(ResourceDesc desc) {
        desc.id=u32(resources.size());
        resources.push_back(std::move(desc));
        return resources.back().id;
    }

    u32 add_pass(RenderPass pass) {
        pass.id=u32(passes.size());
        passes.push_back(std::move(pass));
        return passes.back().id;
    }

    // Topological sort of passes
    [[nodiscard]] std::vector<u32> topological_sort() const {
        u32 N=u32(passes.size());
        std::vector<u32> in_degree(N,0);
        std::vector<std::vector<u32>> adj(N);

        // Build adjacency: pass A -> pass B if A writes something B reads
        for(u32 a=0;a<N;++a)
        for(const auto& out:passes[a].outputs)
        for(u32 b=0;b<N;++b) if(b!=a)
        for(const auto& in:passes[b].inputs)
            if(in.resource_id==out.resource_id){adj[a].push_back(b);++in_degree[b];}

        std::vector<u32> queue, result;
        for(u32 i=0;i<N;++i) if(!in_degree[i]) queue.push_back(i);
        while(!queue.empty()){
            u32 v=queue.back(); queue.pop_back();
            result.push_back(v);
            for(u32 nb:adj[v]) if(--in_degree[nb]==0) queue.push_back(nb);
        }
        return result;
    }

    // Cull passes not contributing to final outputs
    [[nodiscard]] std::vector<bool> cull_passes(std::span<const u32> final_resources) const {
        u32 N=u32(passes.size());
        std::vector<bool> needed(N,false);
        std::vector<bool> res_needed(resources.size(),false);
        for(u32 r:final_resources) if(r<res_needed.size()) res_needed[r]=true;
        // Reverse traversal
        auto order=topological_sort();
        for(i32 i=i32(order.size())-1;i>=0;--i){
            u32 p=order[i];
            bool writes_needed=false;
            for(const auto& out:passes[p].outputs)
                if(out.resource_id<res_needed.size()&&res_needed[out.resource_id]){writes_needed=true;break;}
            if(writes_needed){
                needed[p]=true;
                for(const auto& in:passes[p].inputs) if(in.resource_id<res_needed.size()) res_needed[in.resource_id]=true;
            }
        }
        return needed;
    }

    // Estimate total GPU bandwidth (bytes read + written per frame)
    [[nodiscard]] u64 estimate_bandwidth() const {
        u64 bw=0;
        for(const auto& pass:passes)
        for(const auto& res:pass.inputs)
            if(res.resource_id<resources.size()) bw+=resource_size(resources[res.resource_id]);
        for(const auto& pass:passes)
        for(const auto& res:pass.outputs)
            if(res.resource_id<resources.size()) bw+=resource_size(resources[res.resource_id]);
        return bw;
    }
};

// ─── Resource aliasing (memory overlap) ───────────────────────────────────────

struct LifetimeInterval { u32 pass_begin{}, pass_end{}; };

[[nodiscard]] inline bool intervals_overlap(const LifetimeInterval& a, const LifetimeInterval& b) {
    return a.pass_begin<=b.pass_end && b.pass_begin<=a.pass_end;
}

// Greedy memory aliasing: pack non-overlapping resources into same heap offset
struct AliasedAllocation {
    u32 resource_id{};
    u64 heap_offset{};
    u64 size{};
};

[[nodiscard]] inline std::vector<AliasedAllocation> alias_resources(
    std::span<const ResourceDesc> resources,
    std::span<const LifetimeInterval> lifetimes)
{
    u32 N=u32(resources.size());
    std::vector<AliasedAllocation> allocs(N);
    std::vector<std::pair<u64,u64>> free_ranges; // (offset, size)
    u64 total_size=0;

    for(u32 i=0;i<N;++i) {
        if(!resources[i].transient){
            allocs[i]={i,total_size,resource_size(resources[i])};
            total_size+=resource_size(resources[i]); continue;
        }
        u64 sz=resource_size(resources[i]);
        // Try to reuse a free range
        bool placed=false;
        for(auto& [off,fsz]:free_ranges){
            if(fsz>=sz){
                allocs[i]={i,off,sz};
                off+=sz; fsz-=sz; placed=true; break;
            }
        }
        if(!placed){
            allocs[i]={i,total_size,sz};
            total_size+=sz;
        }
        // Release resources whose lifetime ended before this pass
        // (simplified: just return to pool after pass_end)
    }
    return allocs;
}

// ─── Frame time budget ────────────────────────────────────────────────────────

struct FrameBudget {
    f32 target_ms{16.67f};  // 60fps
    f32 gpu_overhead_ms{0.5f};
    f32 cpu_submit_ms{1.0f};

    [[nodiscard]] f32 available_gpu_ms() const {
        return target_ms - gpu_overhead_ms - cpu_submit_ms;
    }

    [[nodiscard]] bool fits(f32 pass_ms) const {
        return pass_ms <= available_gpu_ms();
    }

    [[nodiscard]] f32 utilization(f32 total_pass_ms) const {
        return total_pass_ms / available_gpu_ms();
    }
};

// ─── Tile-based rendering math ────────────────────────────────────────────────

struct TileGrid {
    u32 tile_w{16}, tile_h{16};
    u32 screen_w{}, screen_h{};

    [[nodiscard]] u32 num_tiles_x() const { return (screen_w+tile_w-1)/tile_w; }
    [[nodiscard]] u32 num_tiles_y() const { return (screen_h+tile_h-1)/tile_h; }
    [[nodiscard]] u32 total_tiles() const { return num_tiles_x()*num_tiles_y(); }

    [[nodiscard]] std::array<u32,4> tile_rect(u32 tx, u32 ty) const {
        return {tx*tile_w, ty*tile_h,
                std::min((tx+1)*tile_w,screen_w),
                std::min((ty+1)*tile_h,screen_h)};
    }

    // Check if light sphere intersects tile frustum (simplified 2D AABB test)
    [[nodiscard]] bool light_in_tile(
        u32 tx, u32 ty, const Vec3f& light_pos_ss, f32 radius_ss) const
    {
        auto [x0,y0,x1,y1]=tile_rect(tx,ty);
        f32 cx=(f32(x0)+f32(x1))*0.5f, cy=(f32(y0)+f32(y1))*0.5f;
        f32 hw=(f32(x1)-f32(x0))*0.5f, hh=(f32(y1)-f32(y0))*0.5f;
        f32 dx=std::max(0.f,std::abs(light_pos_ss.x-cx)-hw);
        f32 dy=std::max(0.f,std::abs(light_pos_ss.y-cy)-hh);
        return dx*dx+dy*dy<=radius_ss*radius_ss;
    }
};

// ─── Visibility buffer / meshlet math ─────────────────────────────────────────

struct Meshlet {
    u32 vertex_offset{}, vertex_count{};
    u32 prim_offset{},   prim_count{};
    Vec3f center{};
    f32   radius{};
    Vec3f cone_apex{};
    Vec3f cone_axis{};
    f32   cone_cutoff{};  // cos of half-angle
};

[[nodiscard]] inline bool meshlet_backface_cull(
    const Meshlet& m, const Vec3f& cam_pos)
{
    Vec3f view=normalize(m.cone_apex-cam_pos);
    return dot(view,m.cone_axis)>=m.cone_cutoff;
}

[[nodiscard]] inline bool meshlet_frustum_cull(
    const Meshlet& m, const Vec4f frustum_planes[6])
{
    for(u32 i=0;i<6;++i)
        if(frustum_planes[i].x*m.center.x+frustum_planes[i].y*m.center.y
          +frustum_planes[i].z*m.center.z+frustum_planes[i].w<-m.radius) return true;
    return false;
}

} // namespace qs
