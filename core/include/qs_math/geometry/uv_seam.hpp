#pragma once
// qs_math/uv_seam.hpp — Seam cutting (minimum spanning tree), virtual nodes, cone singularities (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "geodesic_math.hpp"
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <numbers>
#include <functional>
#include <cassert>

namespace qs {

// ─── Seam graph ───────────────────────────────────────────────────────────────
struct SeamEdge { u32 v0,v1; f32 cost; };

// Build seam graph: candidate seams are high-curvature or high-distortion edges
[[nodiscard]] inline std::vector<SeamEdge> build_seam_candidates(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& normals) // per-vertex normals
{
    std::vector<SeamEdge> candidates;
    std::unordered_map<u64,u32> edge_faces;
    auto ek=[](u32 a,u32 b)->u64{if(a>b)std::swap(a,b);return (u64(a)<<32)|b;};
    for(const auto& t:tris) for(u32 e=0;e<3;++e) {
        u32 a=t[e],b=t[(e+1)%3];
        ++edge_faces[ek(a,b)];
    }
    // Only interior edges (appear twice) are seam candidates
    for(auto& [k,cnt]:edge_faces) {
        if(cnt<2) continue;
        u32 a=u32(k>>32),b=u32(k&0xFFFFFFFF);
        // Cost: dihedral angle deviation from flat (high angle = good seam)
        f32 curvature=1-dot(normals[a],normals[b]); // 0=flat,2=sharp
        f32 len=dist(verts[a],verts[b]);
        candidates.push_back({a,b,curvature*len});
    }
    return candidates;
}

// ─── Minimum spanning tree seam selection (Prim's algorithm) ─────────────────
// Finds the set of seam edges that makes the mesh topologically a disk
// by connecting all genus/boundary in a tree pattern

struct SeamResult {
    std::vector<SeamEdge> seams;  // selected seam edges
    std::vector<bool>     is_seam; // per-edge flag (indexed same as candidates)
    f32 total_cost{0};
};

[[nodiscard]] inline SeamResult cut_to_disk(
    u32 n_verts,
    const std::vector<SeamEdge>& candidates,
    u32 euler_target=1) // 1 for disk topology
{
    // Union-Find to track connected components (we want to CUT edges to reduce genus)
    // Approach: use spanning tree on the dual graph — seams = spanning tree edges
    // Simple: select edges with highest cost (most "natural" cuts) until mesh is a disk
    // For a closed genus-g mesh: need 2g independent cuts + connect to boundary

    // Build adjacency for spanning tree
    std::vector<u32> parent(n_verts);
    std::iota(parent.begin(),parent.end(),0);
    std::function<u32(u32)> find=[&](u32 x)->u32{return parent[x]==x?x:parent[x]=find(parent[x]);};

    // Sort by cost descending (cut most expensive edges first)
    std::vector<u32> order(candidates.size()); std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](u32 a,u32 b){return candidates[a].cost>candidates[b].cost;});

    SeamResult res; res.is_seam.assign(candidates.size(),false);
    u32 components=n_verts;
    for(u32 i:order) {
        u32 a=find(candidates[i].v0),b=find(candidates[i].v1);
        if(a!=b) {
            parent[a]=b; --components;
            res.seams.push_back(candidates[i]);
            res.is_seam[i]=true;
            res.total_cost+=candidates[i].cost;
        }
        if(components<=u32(euler_target)) break;
    }
    return res;
}

// ─── Virtual nodes (duplicate vertices along seams) ────────────────────────────
// Given a set of seam edges, split vertices that appear on seam boundary
// so each UV island has its own copy

struct SplitMesh {
    std::vector<Vec3f> verts;             // may be larger than original (split vertices)
    std::vector<std::array<u32,3>> tris;  // remapped triangles
    std::vector<u32> original_vert;       // maps new vert → original vert
    std::vector<std::vector<u32>> copies; // copies[orig] = list of virtual nodes
};

[[nodiscard]] inline SplitMesh split_seams(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<SeamEdge>& seams)
{
    u32 nv=u32(verts.size());
    // Mark seam vertices
    std::vector<bool> on_seam(nv,false);
    for(const auto& s:seams){ on_seam[s.v0]=true; on_seam[s.v1]=true; }
    SplitMesh result;
    result.verts=verts;
    result.original_vert.resize(nv); std::iota(result.original_vert.begin(),result.original_vert.end(),0);
    result.copies.resize(nv);
    for(u32 i=0;i<nv;++i) result.copies[i]={i};
    result.tris=tris;
    // BFS to assign distinct virtual nodes to each island on seam vertices
    std::vector<i32> tri_island(tris.size(),-1);
    u32 n_islands=0;
    for(u32 fi=0;fi<u32(tris.size());++fi) {
        if(tri_island[fi]>=0) continue;
        std::vector<u32> stack={fi};
        while(!stack.empty()) {
            u32 f=stack.back(); stack.pop_back();
            if(tri_island[f]>=0) continue;
            tri_island[f]=i32(n_islands);
            // Find adjacent faces not across a seam edge
            for(u32 e=0;e<3;++e) {
                u32 va=tris[f][e],vb=tris[f][(e+1)%3];
                bool is_seam_edge=false;
                for(const auto& s:seams) if((s.v0==va&&s.v1==vb)||(s.v0==vb&&s.v1==va)){is_seam_edge=true;break;}
                if(is_seam_edge) continue;
                // Find neighbor sharing this edge
                for(u32 fi2=0;fi2<u32(tris.size());++fi2) if(fi2!=f&&tri_island[fi2]<0) {
                    bool has_a=false,has_b=false;
                    for(u32 v:tris[fi2]){if(v==va)has_a=true;if(v==vb)has_b=true;}
                    if(has_a&&has_b) stack.push_back(fi2);
                }
            }
        }
        ++n_islands;
    }
    // Assign per-island copies of seam vertices
    std::unordered_map<u64,u32> island_vert; // (island,orig_vert) → new vert
    auto ivk=[](u32 isl,u32 v)->u64{return (u64(isl)<<32)|v;};
    for(u32 fi=0;fi<u32(result.tris.size());++fi) {
        u32 isl=u32(tri_island[fi]);
        for(u32 e=0;e<3;++e) {
            u32 ov=tris[fi][e];
            if(!on_seam[ov]) continue;
            auto it=island_vert.find(ivk(isl,ov));
            u32 new_v;
            if(it==island_vert.end()) {
                new_v=u32(result.verts.size());
                result.verts.push_back(verts[ov]);
                result.original_vert.push_back(ov);
                result.copies[ov].push_back(new_v);
                island_vert[ivk(isl,ov)]=new_v;
            } else new_v=it->second;
            result.tris[fi][e]=new_v;
        }
    }
    return result;
}

// ─── Cone singularities ────────────────────────────────────────────────────────
// Gaussian curvature = 2π - sum of angles at vertex
// Singularity: vertex where curvature ≠ 0
struct ConeVertex { u32 vertex; f32 angle_defect; f32 cone_angle; };

[[nodiscard]] inline std::vector<ConeVertex> compute_cone_singularities(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris,
    f32 threshold=0.1f) // angle defect threshold in radians
{
    u32 nv=u32(verts.size());
    std::vector<f32> angle_sum(nv,0);
    for(const auto& t:tris) for(u32 i=0;i<3;++i) {
        Vec3f e1=normalize(verts[t[(i+1)%3]]-verts[t[i]]);
        Vec3f e2=normalize(verts[t[(i+2)%3]]-verts[t[i]]);
        angle_sum[t[i]]+=std::acos(std::clamp(dot(e1,e2),-1.f,1.f));
    }
    std::vector<ConeVertex> cones;
    for(u32 v=0;v<nv;++v) {
        f32 defect=2*f32(std::numbers::pi)-angle_sum[v];
        if(std::abs(defect)>threshold)
            cones.push_back({v,defect,angle_sum[v]});
    }
    // Sort by absolute defect magnitude
    std::sort(cones.begin(),cones.end(),[](const ConeVertex& a,const ConeVertex& b){return std::abs(a.angle_defect)>std::abs(b.angle_defect);});
    return cones;
}

// Total Gaussian curvature (Gauss-Bonnet theorem: = 2π χ for closed manifold)
[[nodiscard]] inline f32 total_gaussian_curvature(const std::vector<ConeVertex>& cones) {
    f32 total=0; for(const auto& c:cones) total+=c.angle_defect; return total;
}

// Place cone singularities optimally (Springborn et al. algorithm sketch)
// For a disk topology, one cone at each corner of a boundary quad layout
[[nodiscard]] inline std::vector<u32> select_cone_positions(
    const std::vector<ConeVertex>& cones, u32 n_cones=4)
{
    u32 n=std::min(n_cones,u32(cones.size()));
    std::vector<u32> selected;
    for(u32 i=0;i<n;++i) selected.push_back(cones[i].vertex);
    return selected;
}

} // namespace qs
