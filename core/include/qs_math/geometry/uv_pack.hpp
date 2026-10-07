#pragma once
// qs_math/uv_pack.hpp — Atlas packing, island detection, border tracing, RBSP (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "geometry2d.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <cassert>

namespace qs {

// ─── UV island ────────────────────────────────────────────────────────────────
struct UVIsland {
    std::vector<Vec2f> uvs;      // UV coordinates (local, 0-based)
    std::vector<std::array<u32,3>> tris; // triangle indices into uvs
    Vec2f min_uv{1e30f,1e30f},max_uv{-1e30f,-1e30f};
    Vec2f packed_offset{};       // position in atlas after packing
    f32   scale{1.f};
    f32   angle{0};              // rotation angle

    void compute_bounds() {
        min_uv={1e30f,1e30f}; max_uv={-1e30f,-1e30f};
        for(const auto& uv:uvs){min_uv.x=std::min(min_uv.x,uv.x);min_uv.y=std::min(min_uv.y,uv.y);max_uv.x=std::max(max_uv.x,uv.x);max_uv.y=std::max(max_uv.y,uv.y);}
    }
    [[nodiscard]] Vec2f size() const { return max_uv-min_uv; }
    [[nodiscard]] f32   area() const { auto s=size(); return s.x*s.y; }
    [[nodiscard]] f32   uv_area() const {
        f32 a=0;
        for(const auto& t:tris){ Vec2f u=uvs[t[1]]-uvs[t[0]],v=uvs[t[2]]-uvs[t[0]]; a+=std::abs(u.x*v.y-u.y*v.x)*0.5f; }
        return a;
    }
    // Normalise: shift uvs so min is at origin
    void normalise() {
        compute_bounds();
        for(auto& uv:uvs){ uv.x-=min_uv.x; uv.y-=min_uv.y; }
        max_uv=max_uv-min_uv; min_uv={0,0};
    }
    // Try rotating by 90°: sometimes gives better packing
    void rotate90() {
        for(auto& uv:uvs){ f32 tmp=uv.x; uv.x=-uv.y; uv.y=tmp; }
        normalise(); angle+=f32(std::numbers::pi)*0.5f;
    }
};

// ─── Island detection (connected components in UV space) ─────────────────────
[[nodiscard]] inline std::vector<UVIsland> detect_islands(
    const std::vector<Vec2f>& uvs,
    const std::vector<std::array<u32,3>>& tris,
    f32 seam_eps=1e-5f)
{
    u32 nf=u32(tris.size());
    // Build face adjacency through shared UV edges
    std::unordered_map<u64,std::vector<u32>> edge_to_faces;
    auto ek=[](Vec2f a,Vec2f b)->u64{
        // Hash two UV points
        u32 ha=u32(std::abs(a.x)*10000)^(u32(std::abs(a.y)*10000)<<16);
        u32 hb=u32(std::abs(b.x)*10000)^(u32(std::abs(b.y)*10000)<<16);
        if(ha>hb) std::swap(ha,hb);
        return (u64(ha)<<32)|hb;
    };
    for(u32 fi=0;fi<nf;++fi) for(u32 e=0;e<3;++e) {
        Vec2f a=uvs[tris[fi][e]],b=uvs[tris[fi][(e+1)%3]];
        edge_to_faces[ek(a,b)].push_back(fi);
    }
    // BFS to find connected components
    std::vector<i32> comp(nf,-1); u32 nc=0;
    for(u32 fi=0;fi<nf;++fi) {
        if(comp[fi]>=0) continue;
        std::vector<u32> stack={fi};
        while(!stack.empty()) {
            u32 f=stack.back(); stack.pop_back();
            if(comp[f]>=0) continue;
            comp[f]=i32(nc);
            for(u32 e=0;e<3;++e) {
                Vec2f a=uvs[tris[f][e]],b=uvs[tris[f][(e+1)%3]];
                for(u32 nf2:edge_to_faces[ek(a,b)]) if(comp[nf2]<0) stack.push_back(nf2);
            }
        }
        ++nc;
    }
    // Build islands
    std::vector<UVIsland> islands(nc);
    for(u32 fi=0;fi<nf;++fi) {
        auto& isl=islands[u32(comp[fi])];
        // Map global UV indices to local
        for(u32 v:tris[fi]) {
            bool found=false;
            for(u32 i=0;i<u32(isl.uvs.size());++i) if(dist_sq(isl.uvs[i],uvs[v])<seam_eps*seam_eps){found=true;break;}
            if(!found) isl.uvs.push_back(uvs[v]);
        }
        std::array<u32,3> local_tri;
        for(u32 e=0;e<3;++e) {
            Vec2f uv=uvs[tris[fi][e]];
            for(u32 i=0;i<u32(isl.uvs.size());++i) if(dist_sq(isl.uvs[i],uv)<seam_eps*seam_eps){local_tri[e]=i;break;}
        }
        isl.tris.push_back(local_tri);
    }
    for(auto& isl:islands) isl.normalise();
    return islands;
}

// ─── Border tracing ───────────────────────────────────────────────────────────
[[nodiscard]] inline std::vector<Vec2f> trace_island_border(const UVIsland& island) {
    // Find boundary edges (appear once)
    std::unordered_map<u64,std::pair<u32,u32>> edge_count;
    auto ek=[](u32 a,u32 b)->u64{return (u64(std::min(a,b))<<32)|std::max(a,b);};
    for(const auto& t:island.tris) for(u32 e=0;e<3;++e) {
        u32 a=t[e],b=t[(e+1)%3];
        ++edge_count[ek(a,b)].first;
        edge_count[ek(a,b)].second=a; // store directed info
    }
    // Build boundary adjacency
    std::unordered_map<u32,u32> next_vert;
    for(const auto& t:island.tris) for(u32 e=0;e<3;++e) {
        u32 a=t[e],b=t[(e+1)%3];
        if(edge_count[ek(a,b)].first==1) next_vert[a]=b; // boundary edge a→b
    }
    if(next_vert.empty()) return {};
    // Walk boundary
    u32 start=next_vert.begin()->first;
    std::vector<Vec2f> border;
    u32 cur=start;
    do {
        border.push_back(island.uvs[cur]);
        auto it=next_vert.find(cur);
        if(it==next_vert.end()) break;
        cur=it->second;
    } while(cur!=start&&border.size()<island.uvs.size()+10);
    return border;
}

// ─── Shelf-bin packing (FFDH — First Fit Decreasing Height) ──────────────────
struct AtlasPackResult {
    std::vector<Vec2f> offsets; // per-island UV offset
    std::vector<f32>   scales;  // per-island scale factor
    std::vector<bool>  rotated; // whether island was rotated 90°
    f32 utilisation{0};         // area used / atlas area
    f32 atlas_size{1.f};
};

[[nodiscard]] inline AtlasPackResult pack_islands_shelf(
    std::vector<UVIsland>& islands,
    f32 atlas_size=1.f, f32 padding=0.002f)
{
    u32 n=u32(islands.size());
    AtlasPackResult result;
    result.offsets.resize(n); result.scales.resize(n,1.f); result.rotated.resize(n,false);
    // Sort islands by height (descending)
    std::vector<u32> order(n); std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](u32 a,u32 b){
        Vec2f sa=islands[a].size(),sb=islands[b].size();
        return std::max(sa.x,sa.y)>std::max(sb.x,sb.y);
    });
    // Shelf packing
    struct Shelf { f32 x{0},y{0},h{0}; };
    std::vector<Shelf> shelves={{0,0,0}};
    f32 total_area_used=0;
    for(u32 idx:order) {
        auto& isl=islands[idx];
        Vec2f sz=isl.size();
        f32 scale=std::min(atlas_size*0.9f/std::max(sz.x,sz.y+1e-10f),1.f);
        f32 w=sz.x*scale+padding,h=sz.y*scale+padding;
        // Try each shelf
        bool placed=false;
        for(auto& shelf:shelves) {
            if(shelf.x+w<=atlas_size&&shelf.y+h<=atlas_size) {
                result.offsets[idx]={shelf.x+padding*0.5f,shelf.y+padding*0.5f};
                result.scales[idx]=scale;
                shelf.h=std::max(shelf.h,h);
                shelf.x+=w;
                total_area_used+=w*h;
                placed=true; break;
            }
        }
        if(!placed) {
            f32 new_y=0; for(auto& s:shelves) new_y=std::max(new_y,s.y+s.h);
            if(new_y+h<=atlas_size) {
                shelves.push_back({w,new_y,h});
                result.offsets[idx]={padding*0.5f,new_y+padding*0.5f};
                result.scales[idx]=scale;
                total_area_used+=w*h;
            } else {
                // Doesn't fit — place at 0 with smaller scale
                scale*=0.5f; w=sz.x*scale; h=sz.y*scale;
                result.offsets[idx]={0,0}; result.scales[idx]=scale;
            }
        }
    }
    result.atlas_size=atlas_size;
    result.utilisation=total_area_used/(atlas_size*atlas_size);
    return result;
}

// Apply packing result to island UVs
inline void apply_pack(std::vector<UVIsland>& islands, const AtlasPackResult& pack) {
    for(u32 i=0;i<u32(islands.size());++i) {
        f32 s=pack.scales[i]; Vec2f off=pack.offsets[i];
        for(auto& uv:islands[i].uvs){ uv.x=uv.x*s+off.x; uv.y=uv.y*s+off.y; }
    }
}

// ─── Overlap detection ────────────────────────────────────────────────────────
[[nodiscard]] inline bool islands_overlap(const UVIsland& a, const UVIsland& b) {
    // AABB test first
    Vec2f a_min=a.min_uv+a.packed_offset, a_max=a.max_uv+a.packed_offset;
    Vec2f b_min=b.min_uv+b.packed_offset, b_max=b.max_uv+b.packed_offset;
    if(a_max.x<=b_min.x||b_max.x<=a_min.x||a_max.y<=b_min.y||b_max.y<=a_min.y) return false;
    // SAT on borders would be needed for exact test — simplified here
    return true;
}

// Utilisation: ratio of UV triangle area to atlas area
[[nodiscard]] inline f32 atlas_utilisation(const std::vector<UVIsland>& islands, f32 atlas_size) {
    f32 used=0; for(const auto& isl:islands) used+=isl.uv_area();
    return used/(atlas_size*atlas_size);
}

// ─── Lightmap-specific packing (add margin for texel bleed) ───────────────────
struct LightmapPackConfig {
    u32  resolution{512};
    u32  texel_margin{2};
    bool allow_rotation{true};
    f32  scale_bias{1.f}; // multiply all island scales by this
};

[[nodiscard]] inline AtlasPackResult pack_lightmap(std::vector<UVIsland>& islands,
    const LightmapPackConfig& cfg)
{
    f32 texel_size=1.f/cfg.resolution;
    f32 padding=cfg.texel_margin*texel_size;
    // Scale islands to match world-space density
    for(auto& isl:islands) {
        Vec2f sz=isl.size();
        f32 s=std::min(1.f/std::max(sz.x,sz.y+1e-10f),cfg.scale_bias);
        for(auto& uv:isl.uvs){uv.x*=s;uv.y*=s;}
        isl.normalise();
    }
    return pack_islands_shelf(islands,1.f,padding);
}

} // namespace qs
