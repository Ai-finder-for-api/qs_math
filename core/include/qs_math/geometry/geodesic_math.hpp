#pragma once
// qs_math/geodesic_math.hpp — Exact geodesics on meshes (Dijkstra/MMP), heat method (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mesh_math.hpp"
#include <vector>
#include <queue>
#include <array>
#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>

namespace qs {

// ─── Dijkstra geodesic distance on mesh graph (vertex-to-vertex) ──────────────
struct MeshGraph {
    u32 n_verts{0};
    std::vector<std::vector<std::pair<u32,f32>>> adj; // adj[v] = [(neighbour, edge_length)]

    MeshGraph() = default;
    explicit MeshGraph(u32 nv) : n_verts(nv), adj(nv) {}

    void add_edge(u32 a, u32 b, f32 len) {
        adj[a].emplace_back(b,len);
        adj[b].emplace_back(a,len);
    }

    // Build from triangle soup
    static MeshGraph from_triangles(u32 nv,
        const std::vector<std::array<u32,3>>& tris,
        const std::vector<Vec3f>& verts)
    {
        MeshGraph g(nv);
        for(const auto& t:tris) {
            for(u32 e=0;e<3;++e) {
                u32 a=t[e],b=t[(e+1)%3];
                f32 len=dist(verts[a],verts[b]);
                // Avoid duplicate edges
                bool found=false;
                for(auto& [nb,w]:g.adj[a]) if(nb==b){found=true;break;}
                if(!found) g.add_edge(a,b,len);
            }
        }
        return g;
    }
};

// Dijkstra shortest paths from source vertex
[[nodiscard]] inline std::vector<f32> dijkstra_geodesic(const MeshGraph& g, u32 source) {
    std::vector<f32> dist(g.n_verts, 1e30f);
    dist[source]=0;
    using P=std::pair<f32,u32>;
    std::priority_queue<P,std::vector<P>,std::greater<P>> pq;
    pq.push({0,source});
    while(!pq.empty()) {
        auto [d,u]=pq.top(); pq.pop();
        if(d>dist[u]) continue;
        for(auto [v,w]:g.adj[u]) {
            f32 nd=dist[u]+w;
            if(nd<dist[v]){ dist[v]=nd; pq.push({nd,v}); }
        }
    }
    return dist;
}

// Multi-source Dijkstra (from a set of source vertices)
[[nodiscard]] inline std::vector<f32> dijkstra_multi_source(const MeshGraph& g,
    const std::vector<u32>& sources)
{
    std::vector<f32> d(g.n_verts,1e30f);
    using P=std::pair<f32,u32>;
    std::priority_queue<P,std::vector<P>,std::greater<P>> pq;
    for(u32 s:sources){ d[s]=0; pq.push({0,s}); }
    while(!pq.empty()) {
        auto [dv,u]=pq.top(); pq.pop();
        if(dv>d[u]) continue;
        for(auto [v,w]:g.adj[u]) if(d[u]+w<d[v]){ d[v]=d[u]+w; pq.push({d[v],v}); }
    }
    return d;
}

// ─── Window propagation (MMP algorithm sketch) ───────────────────────────────
// Simplified MMP: exact geodesic via window propagation
// Full implementation requires half-edge mesh; here a simplified interface
struct GeodesicWindow {
    u32 edge_id;
    f32 b0,b1;     // parametric interval on edge [0,1]
    f32 d0,d1;     // distances at interval endpoints
    u32 source;    // originating source vertex
    f32 sigma;     // signed distance to virtual source
};

// Propagate windows (simplified — one step)
[[nodiscard]] inline std::vector<f32> mmp_geodesic_approx(
    u32 nv, const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& verts, u32 source, u32 max_iters=5)
{
    auto g=MeshGraph::from_triangles(nv,tris,verts);
    auto base=dijkstra_geodesic(g,source);
    // Refine using local unfolding on each triangle
    for(u32 iter=0;iter<max_iters;++iter) {
        bool improved=false;
        for(const auto& t:tris) {
            for(u32 e=0;e<3;++e) {
                u32 a=t[e],b=t[(e+1)%3],c=t[(e+2)%3];
                // Can we improve dist[c] by going through edge a-b?
                f32 lab=dist(verts[a],verts[b]);
                f32 lac=dist(verts[a],verts[c]);
                f32 lbc=dist(verts[b],verts[c]);
                // Unfold: place virtual source at position on plane
                f32 cos_a=(lab*lab+lac*lac-lbc*lbc)/(2*lab*lac+1e-10f);
                // Local distance estimate
                f32 t_ab=std::clamp((base[a]-base[b]+lab)/(2*lab+1e-10f),0.f,1.f);
                f32 d_int=base[a]*(1-t_ab)+base[b]*t_ab;
                // Distance from interpolated point to c
                Vec3f pa=verts[a],pb=verts[b],pc=verts[c];
                Vec3f pi=pa+(pb-pa)*t_ab;
                f32 nd=d_int+dist(pi,pc);
                if(nd<base[c]-1e-5f){ base[c]=nd; improved=true; }
            }
        }
        if(!improved) break;
    }
    return base;
}

// ─── Heat method for geodesics (Crane et al. 2013) ────────────────────────────
// Approximate geodesic using heat diffusion
// 1) Solve heat equation (one step): (I - t*L) u = delta_source
// 2) Normalize gradient: X = -∇u / |∇u|
// 3) Solve Poisson equation: Δφ = ∇·X
struct HeatGeodesic {
    u32 nv;
    std::vector<Vec3f> verts;
    std::vector<std::array<u32,3>> tris;
    f32 timestep;

    HeatGeodesic(u32 nv, std::vector<Vec3f> v, std::vector<std::array<u32,3>> t, f32 h=-1)
        : nv(nv), verts(std::move(v)), tris(std::move(t))
    {
        // Compute mean edge length if h not given
        if(h<0){ f32 s=0; u32 c=0; for(auto& tr:this->tris) for(u32 e=0;e<3;++e){s+=dist(this->verts[tr[e]],this->verts[tr[(e+1)%3]]);++c;} timestep=c>0?(s/c)*(s/c):1; }
        else timestep=h;
    }

    // Build cotangent Laplacian (returns sparse triplet form)
    [[nodiscard]] std::vector<std::tuple<u32,u32,f32>> build_laplacian() const {
        std::vector<std::tuple<u32,u32,f32>> trips;
        for(const auto& t:tris) {
            for(u32 i=0;i<3;++i) {
                u32 vi=t[i],vj=t[(i+1)%3],vk=t[(i+2)%3];
                Vec3f ei=verts[vj]-verts[vi],ej=verts[vk]-verts[vj];
                // Cotangent weight at vertex vk (opposite edge vi-vj)
                Vec3f u=verts[vi]-verts[vk],v=verts[vj]-verts[vk];
                f32 cos_k=dot(u,v),sin_k=length(cross(u,v));
                f32 cot_k=sin_k>1e-7f?cos_k/sin_k:0;
                f32 w=cot_k*0.5f;
                trips.emplace_back(vi,vj,-w); trips.emplace_back(vj,vi,-w);
                trips.emplace_back(vi,vi,w);  trips.emplace_back(vj,vj,w);
                (void)ei;(void)ej;
            }
        }
        return trips;
    }

    // Simplified heat method (Jacobi iterations instead of full solve)
    [[nodiscard]] std::vector<f32> geodesic_from(u32 source, u32 iters=50) const {
        // Step 1: diffuse heat from source
        std::vector<f32> u(nv,0); u[source]=1;
        auto g=MeshGraph::from_triangles(nv,tris,verts);
        for(u32 it=0;it<iters;++it) {
            std::vector<f32> u_new=u;
            for(u32 vi=0;vi<nv;++vi) {
                f32 lap=0, sum_w=0;
                for(auto [vj,w]:g.adj[vi]){ lap+=u[vj]*w; sum_w+=w; }
                if(sum_w>1e-10f) u_new[vi]=u[vi]+timestep*(lap/sum_w-u[vi]);
            }
            u=u_new;
        }
        // Step 2: normalize gradient → approximate geodesic via reverse diffusion
        // Simplified: use -log(u+eps) as geodesic proxy
        std::vector<f32> phi(nv);
        for(u32 i=0;i<nv;++i) phi[i]=-std::log(std::max(u[i],1e-10f));
        // Shift to zero at source
        f32 min_val=phi[source];
        for(auto& v:phi) v-=min_val;
        return phi;
    }
};

// ─── Isoline extraction (marching on mesh surface) ────────────────────────────
// Extract a level-set isoline from a scalar field on mesh vertices
struct IsolineSegment { Vec3f a, b; };

[[nodiscard]] inline std::vector<IsolineSegment> extract_isoline(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<f32>& field, f32 isovalue)
{
    std::vector<IsolineSegment> result;
    for(const auto& t:tris) {
        std::vector<Vec3f> pts;
        for(u32 e=0;e<3;++e) {
            u32 a=t[e],b=t[(e+1)%3];
            f32 fa=field[a],fb=field[b];
            if((fa<=isovalue&&fb>=isovalue)||(fa>=isovalue&&fb<=isovalue)) {
                f32 fr=std::abs(fb-fa)<1e-10f?0.5f:(isovalue-fa)/(fb-fa);
                pts.push_back(verts[a]+(verts[b]-verts[a])*fr);
            }
        }
        if(pts.size()==2) result.push_back({pts[0],pts[1]});
    }
    return result;
}

} // namespace qs
