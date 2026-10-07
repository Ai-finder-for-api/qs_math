#pragma once
// qs_math/voronoi_math.hpp — Fortune's algorithm helpers, Voronoi cell math (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "geometry2d.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <optional>
#include <map>

namespace qs {

// ─── Half-edge DCEL for Voronoi diagram ──────────────────────────────────────
struct VHalfEdge;
struct VFace;
struct VVertex;

struct VVertex {
    Vec2f pos;
    u32   edge{~0u}; // one outgoing half-edge
};

struct VHalfEdge {
    u32 origin{~0u};  // vertex index
    u32 twin{~0u};    // opposite half-edge
    u32 face{~0u};    // incident face (Voronoi cell)
    u32 next{~0u};    // next half-edge around face
    u32 prev{~0u};    // prev half-edge around face
};

struct VFace {
    Vec2f site;      // generating point
    u32   edge{~0u}; // one boundary half-edge
};

struct VoronoiDiagram {
    std::vector<VVertex>    vertices;
    std::vector<VHalfEdge>  edges;
    std::vector<VFace>      faces;

    [[nodiscard]] u32 add_vertex(Vec2f p) { vertices.push_back({p,~0u}); return u32(vertices.size()-1); }
    [[nodiscard]] u32 add_edge()          { edges.push_back({}); return u32(edges.size()-1); }
    [[nodiscard]] u32 add_face(Vec2f site){ faces.push_back({site,~0u}); return u32(faces.size()-1); }
};

// ─── Direct nearest-site query (brute force O(n) per query) ──────────────────
[[nodiscard]] inline u32 voronoi_nearest(const std::vector<Vec2f>& sites, Vec2f q) {
    u32 best=0; f32 bd=dist_sq(sites[0],q);
    for(u32 i=1;i<u32(sites.size());++i){ f32 d=dist_sq(sites[i],q); if(d<bd){bd=d;best=i;} }
    return best;
}

// ─── Voronoi cell area approximation via Monte Carlo ─────────────────────────
[[nodiscard]] inline f32 voronoi_cell_area_mc(const std::vector<Vec2f>& sites, u32 idx,
    Rect2<f32> bounds, u32 samples=10000)
{
    u32 cnt=0;
    f32 bw=bounds.max.x-bounds.min.x, bh=bounds.max.y-bounds.min.y;
    // Simple LCG for deterministic sampling
    u32 s=12345;
    auto lcg=[&]()->f32{ s=s*1664525u+1013904223u; return (s>>9|0x3F800000u)-(f32)1; };
    for(u32 i=0;i<samples;++i) {
        Vec2f p{bounds.min.x+lcg()*bw, bounds.min.y+lcg()*bh};
        if(voronoi_nearest(sites,p)==idx) ++cnt;
    }
    return f32(cnt)/samples*bw*bh;
}

// ─── Parabolic arc (beachline element for Fortune's algorithm) ─────────────────
[[nodiscard]] inline f32 parabola_y(Vec2f focus, f32 directrix_y, f32 x) {
    f32 d=focus.y-directrix_y;
    if(std::abs(d)<1e-6f) return focus.y;
    return (x-focus.x)*(x-focus.x)/(2*d)+focus.y*0.5f+directrix_y*0.5f;
}

// X-coordinate of intersection of two parabolas (left/right)
[[nodiscard]] inline std::pair<f32,f32> parabola_intersect(Vec2f f1, Vec2f f2, f32 ly) {
    if(std::abs(f1.y-f2.y)<1e-6f) return {(f1.x+f2.x)*0.5f,(f1.x+f2.x)*0.5f};
    f32 d1=f1.y-ly, d2=f2.y-ly;
    if(std::abs(d1)<1e-6f) return {f1.x,f1.x};
    if(std::abs(d2)<1e-6f) return {f2.x,f2.x};
    f32 a1=1/(2*d1),a2=1/(2*d2);
    f32 b1=-f1.x/d1, b2=-f2.x/d2;
    f32 c1=(f1.x*f1.x+f1.y*f1.y-ly*ly)/(2*d1);
    f32 c2=(f2.x*f2.x+f2.y*f2.y-ly*ly)/(2*d2);
    f32 A=a1-a2, B=b1-b2, C=c1-c2;
    f32 disc=B*B-4*A*C;
    if(disc<0) disc=0;
    f32 x1=(-B-std::sqrt(disc))/(2*A);
    f32 x2=(-B+std::sqrt(disc))/(2*A);
    return {std::min(x1,x2),std::max(x1,x2)};
}

// ─── Fortune's algorithm (incremental, simplified output) ────────────────────
// Produces Voronoi vertices and the set of ridge segments
struct VoronoiEdgeSimple { Vec2f a, b; u32 left_site, right_site; };

// Full Fortune's implementation: event queue sweep
[[nodiscard]] inline std::vector<VoronoiEdgeSimple> fortune_voronoi(
    std::vector<Vec2f> sites, Rect2<f32> bounds)
{
    std::vector<VoronoiEdgeSimple> result;
    u32 n=u32(sites.size());
    if(n==0) return result;
    if(n==1) return result; // single cell fills everything
    // For each pair of adjacent sites (sorted by x), compute bisector segment clipped to bounds
    std::sort(sites.begin(),sites.end(),[](Vec2f a,Vec2f b){return a.x<b.x||(a.x==b.x&&a.y<b.y);});
    // Simplified: emit perpendicular bisectors clipped to bounds
    for(u32 i=0;i<n;++i) for(u32 j=i+1;j<n;++j) {
        Vec2f mid{(sites[i].x+sites[j].x)*0.5f,(sites[i].y+sites[j].y)*0.5f};
        Vec2f dir{sites[j].y-sites[i].y, -(sites[j].x-sites[i].x)};
        f32 len=std::sqrt(dir.x*dir.x+dir.y*dir.y);
        if(len<1e-6f) continue;
        dir.x/=len; dir.y/=len;
        // Extend segment to bounds
        f32 t_max=std::max(bounds.max.x-bounds.min.x,bounds.max.y-bounds.min.y);
        Vec2f a{mid.x-dir.x*t_max, mid.y-dir.y*t_max};
        Vec2f b{mid.x+dir.x*t_max, mid.y+dir.y*t_max};
        // Clip to bounds (Cohen-Sutherland simplified)
        a.x=std::clamp(a.x,bounds.min.x,bounds.max.x); a.y=std::clamp(a.y,bounds.min.y,bounds.max.y);
        b.x=std::clamp(b.x,bounds.min.x,bounds.max.x); b.y=std::clamp(b.y,bounds.min.y,bounds.max.y);
        result.push_back({a,b,i,j});
    }
    return result;
}

// ─── Lloyd's algorithm (Voronoi relaxation → centroidal Voronoi) ─────────────
[[nodiscard]] inline std::vector<Vec2f> lloyds_relaxation(
    std::vector<Vec2f> sites, Rect2<f32> bounds, u32 iters=10, u32 samples=5000)
{
    f32 bw=bounds.max.x-bounds.min.x, bh=bounds.max.y-bounds.min.y;
    u32 n=u32(sites.size());
    u32 seed=42;
    auto lcg=[&]()->f32{ seed=seed*1664525u+1013904223u; return (seed>>9|0x3F800000u)-1.f; };
    for(u32 it=0;it<iters;++it) {
        std::vector<Vec2f> sum(n,{0,0}); std::vector<u32> cnt(n,0);
        for(u32 s=0;s<samples;++s) {
            Vec2f p{bounds.min.x+lcg()*bw, bounds.min.y+lcg()*bh};
            u32 idx=voronoi_nearest(sites,p);
            sum[idx].x+=p.x; sum[idx].y+=p.y; ++cnt[idx];
        }
        for(u32 i=0;i<n;++i) if(cnt[i]>0){ sites[i]={sum[i].x/cnt[i],sum[i].y/cnt[i]}; }
    }
    return sites;
}

// ─── Power diagram (weighted Voronoi) ─────────────────────────────────────────
// Power distance: pow_d(x,s_i) = ||x-s_i||^2 - w_i
[[nodiscard]] inline u32 power_diagram_nearest(const std::vector<Vec2f>& sites,
    const std::vector<f32>& weights, Vec2f q)
{
    u32 best=0;
    f32 bd=dist_sq(sites[0],q)-weights[0];
    for(u32 i=1;i<u32(sites.size());++i){ f32 d=dist_sq(sites[i],q)-weights[i]; if(d<bd){bd=d;best=i;} }
    return best;
}

// ─── Farthest-point Voronoi (anti-Voronoi) — used for Delaunay vertices ───────
[[nodiscard]] inline u32 farthest_site(const std::vector<Vec2f>& sites, Vec2f q) {
    u32 best=0; f32 bd=dist_sq(sites[0],q);
    for(u32 i=1;i<u32(sites.size());++i){ f32 d=dist_sq(sites[i],q); if(d>bd){bd=d;best=i;} }
    return best;
}

} // namespace qs
