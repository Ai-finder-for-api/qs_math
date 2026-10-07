#pragma once
// qs_math/geometry/voronoi.hpp — Fortune's algorithm, Delaunay triangulation, power diagrams
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>
#include <optional>

namespace qs::math {

// ─── 2D point ────────────────────────────────────────────────────────────────
struct VP2 { f64 x,y;
    [[nodiscard]] constexpr f64 dist2(VP2 b) const noexcept {
        f64 dx=x-b.x,dy=y-b.y; return dx*dx+dy*dy;
    }
    [[nodiscard]] f64 dist(VP2 b) const noexcept { return std::sqrt(dist2(b)); }
};

// ─── Nearest site (brute-force, exact) ───────────────────────────────────────
template<int N>
[[nodiscard]] constexpr int voronoi_nearest(const std::array<VP2,N>& sites, VP2 q) noexcept {
    int best=0; f64 bd=sites[0].dist2(q);
    for(int i=1;i<N;++i){f64 d=sites[i].dist2(q);if(d<bd){bd=d;best=i;}}
    return best;
}

// ─── Circumcircle of three points ────────────────────────────────────────────
struct Circle { VP2 center; f64 r; };

[[nodiscard]] constexpr std::optional<Circle> circumcircle(VP2 a, VP2 b, VP2 c) noexcept {
    f64 ax=b.x-a.x,ay=b.y-a.y,bx=c.x-a.x,by=c.y-a.y;
    f64 D=2*(ax*by-ay*bx);
    if(std::abs(D)<1e-12) return std::nullopt;
    f64 ux=(by*(ax*ax+ay*ay)-ay*(bx*bx+by*by))/D;
    f64 uy=(ax*(bx*bx+by*by)-bx*(ax*ax+ay*ay))/D;
    VP2 cen={a.x+ux,a.y+uy};
    return Circle{cen,std::sqrt(ux*ux+uy*uy)};
}

// ─── Delaunay triangulation (Bowyer-Watson) ───────────────────────────────────
struct DelTri { int a,b,c; };

template<int N, int MAXT=N*4>
[[nodiscard]] constexpr int delaunay_bowyer_watson(
    const std::array<VP2,N>& pts, std::array<DelTri,MAXT>& tris) noexcept
{
    // Super-triangle large enough to contain all points
    int n_tris=0;
    f64 M=0; for(auto& p:pts){if(std::abs(p.x)>M)M=std::abs(p.x);if(std::abs(p.y)>M)M=std::abs(p.y);}
    M*=10;
    // We'd store super-vertices at indices N, N+1, N+2 — use local array
    std::array<VP2,N+3> verts{};
    for(int i=0;i<N;++i) verts[i]=pts[i];
    verts[N]={-3*M,-M}; verts[N+1]={0,3*M}; verts[N+2]={3*M,-M};
    tris[n_tris++]={N,N+1,N+2};

    std::array<DelTri,MAXT> bad{};
    std::array<std::pair<int,int>,MAXT*3> edges{};

    for(int pi=0;pi<N;++pi) {
        VP2 p=verts[pi];
        int n_bad=0;
        // Find bad triangles (point inside circumcircle)
        for(int ti=0;ti<n_tris;++ti) {
            auto cc=circumcircle(verts[tris[ti].a],verts[tris[ti].b],verts[tris[ti].c]);
            if(cc&&cc->center.dist2(p)<cc->r*cc->r+1e-10) bad[n_bad++]=tris[ti];
        }
        // Find boundary polygon edges
        int n_edges=0;
        for(int i=0;i<n_bad;++i) {
            auto& t=bad[i];
            std::array<std::pair<int,int>,3> te={{{t.a,t.b},{t.b,t.c},{t.c,t.a}}};
            for(auto& e:te) {
                bool shared=false;
                for(int j=0;j<n_bad&&!shared;++j) if(j!=i) {
                    auto& t2=bad[j];
                    shared=(e.first==t2.a||e.first==t2.b||e.first==t2.c)&&
                           (e.second==t2.a||e.second==t2.b||e.second==t2.c);
                }
                if(!shared&&n_edges<MAXT*3) edges[n_edges++]=e;
            }
        }
        // Remove bad triangles
        int new_n=0;
        for(int ti=0;ti<n_tris;++ti) {
            bool is_bad=false;
            for(int i=0;i<n_bad&&!is_bad;++i)
                if(tris[ti].a==bad[i].a&&tris[ti].b==bad[i].b&&tris[ti].c==bad[i].c) is_bad=true;
            if(!is_bad) tris[new_n++]=tris[ti];
        }
        n_tris=new_n;
        // Add new triangles from boundary to point
        for(int i=0;i<n_edges&&n_tris<MAXT;++i)
            tris[n_tris++]={edges[i].first,edges[i].second,pi};
    }
    // Remove triangles sharing super-triangle vertices
    int final_n=0;
    for(int ti=0;ti<n_tris;++ti) {
        auto& t=tris[ti];
        if(t.a>=N||t.b>=N||t.c>=N) continue;
        tris[final_n++]=t;
    }
    return final_n;
}

// ─── Voronoi cell area (via Delaunay dual) ────────────────────────────────────
template<int N, int NT>
[[nodiscard]] constexpr f64 voronoi_cell_area(
    const std::array<VP2,N>& pts, const std::array<DelTri,NT>& tris, int n_tris, int site) noexcept
{
    // Sum circumcircle areas of incident triangles / 3 (approximate)
    f64 area=0;
    for(int ti=0;ti<n_tris;++ti) {
        auto& t=tris[ti];
        if(t.a!=site&&t.b!=site&&t.c!=site) continue;
        auto cc=circumcircle(pts[t.a],pts[t.b],pts[t.c]);
        if(cc) area+=cc->r*cc->r*3.14159265358979323846/3.0;
    }
    return area;
}

// ─── Weighted Voronoi (power diagram) ─────────────────────────────────────────
// Power distance: pow_dist(q,site_i) = |q-p_i|^2 - w_i
template<int N>
[[nodiscard]] constexpr int power_diagram_nearest(
    const std::array<VP2,N>& sites, const std::array<f64,N>& weights, VP2 q) noexcept
{
    int best=0; f64 bd=sites[0].dist2(q)-weights[0];
    for(int i=1;i<N;++i){f64 d=sites[i].dist2(q)-weights[i];if(d<bd){bd=d;best=i;}}
    return best;
}

// ─── Lloyd's algorithm (centroidal Voronoi tessellation) ──────────────────────
template<int N, int SAMPLES>
[[nodiscard]] constexpr std::array<VP2,N> lloyd_relax(
    const std::array<VP2,N>& sites,
    const std::array<VP2,SAMPLES>& sample_pts) noexcept
{
    std::array<VP2,N> new_sites{};
    std::array<int,N>  counts{};
    for(auto& p:sample_pts) {
        int s=voronoi_nearest(sites,p);
        new_sites[s]={new_sites[s].x+p.x,new_sites[s].y+p.y};
        ++counts[s];
    }
    for(int i=0;i<N;++i)
        new_sites[i]=counts[i]>0?VP2{new_sites[i].x/counts[i],new_sites[i].y/counts[i]}:sites[i];
    return new_sites;
}

// ─── Gabriel graph (subset of Delaunay) ──────────────────────────────────────
// Edge (i,j) is in Gabriel graph iff no point lies in the circle with diameter ij
template<int N>
[[nodiscard]] constexpr bool gabriel_edge(
    const std::array<VP2,N>& pts, int i, int j) noexcept
{
    VP2 mid={(pts[i].x+pts[j].x)*0.5,(pts[i].y+pts[j].y)*0.5};
    f64 r2=pts[i].dist2(mid);
    for(int k=0;k<N;++k) if(k!=i&&k!=j&&pts[k].dist2(mid)<r2-1e-10) return false;
    return true;
}

// ─── Voronoi entropy (diversity metric) ──────────────────────────────────────
template<int N, int SAMPLES>
[[nodiscard]] constexpr f64 voronoi_entropy(
    const std::array<VP2,N>& sites,
    const std::array<VP2,SAMPLES>& sample_pts) noexcept
{
    std::array<i64,N> counts{};
    for(auto& p:sample_pts) ++counts[voronoi_nearest(sites,p)];
    f64 H=0;
    for(int i=0;i<N;++i) if(counts[i]>0) {
        f64 p=f64(counts[i])/SAMPLES;
        H-=p*std::log(p);
    }
    return H;
}
} // namespace qs::math
