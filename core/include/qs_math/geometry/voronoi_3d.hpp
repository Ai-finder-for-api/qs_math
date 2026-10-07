#pragma once
// qs_math/voronoi_3d.hpp — 3D Voronoi, power diagram, Apollonius circles (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "geometry3d.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <span>

namespace qs {

// ─── Voronoi site ─────────────────────────────────────────────────────────────

struct VoronoiSite3D {
    Vec3f pos{};
    f32   weight{0.f};  // for power diagram / weighted Voronoi
    u32   id{};
};

// ─── Cell / half-edge structures ──────────────────────────────────────────────

struct VFace {
    std::vector<Vec3f> vertices;  // convex polygon in 3D
    u32 site_a{}, site_b{};       // two sites sharing this face
};

struct VCell {
    u32   site_id{};
    std::vector<VFace> faces;
    Vec3f centroid{};
    f32   volume{};
};

// ─── Nearest Voronoi site ────────────────────────────────────────────────────

[[nodiscard]] inline u32 voronoi_nearest(
    const Vec3f& p, std::span<const VoronoiSite3D> sites)
{
    u32 best=0;
    f32 best_d=length_sq(p-sites[0].pos);
    for (u32 i=1;i<u32(sites.size());++i) {
        f32 d=length_sq(p-sites[i].pos);
        if(d<best_d){best_d=d;best=i;}
    }
    return best;
}

// Power distance: ||p - s||^2 - w^2
[[nodiscard]] inline f32 power_distance(
    const Vec3f& p, const VoronoiSite3D& site)
{
    return length_sq(p-site.pos) - site.weight*site.weight;
}

[[nodiscard]] inline u32 power_nearest(
    const Vec3f& p, std::span<const VoronoiSite3D> sites)
{
    u32 best=0;
    f32 best_d=power_distance(p,sites[0]);
    for (u32 i=1;i<u32(sites.size());++i) {
        f32 d=power_distance(p,sites[i]);
        if(d<best_d){best_d=d;best=i;}
    }
    return best;
}

// ─── Bisector plane between two sites ────────────────────────────────────────

struct BisectorPlane {
    Vec3f normal{};
    f32   d{};  // dot(normal, point_on_plane) = d

    [[nodiscard]] f32 side(const Vec3f& p) const {
        return dot(normal,p) - d;
    }
};

[[nodiscard]] inline BisectorPlane voronoi_bisector(
    const VoronoiSite3D& a, const VoronoiSite3D& b)
{
    Vec3f n = b.pos - a.pos;
    f32 len = length(n);
    if(len<1e-8f) return {{1,0,0},0.f};
    n = n/len;
    Vec3f mid = (a.pos+b.pos)*0.5f;
    // For power diagram: adjust midpoint
    if(std::abs(a.weight-b.weight)>1e-6f) {
        f32 wa2=a.weight*a.weight, wb2=b.weight*b.weight;
        f32 t=0.5f+(wb2-wa2)/(2.f*len*len);
        mid = a.pos + n*(t*len);
    }
    return {n, dot(n,mid)};
}

// ─── Voronoi cell clipping by half-spaces ─────────────────────────────────────

// Sutherland-Hodgman in 3D for convex cells
// Clip convex polyhedron (list of faces) against half-plane normal*x <= d
inline void clip_face_by_plane(
    const std::vector<Vec3f>& in_poly,
    std::vector<Vec3f>& out_poly,
    const Vec3f& n, f32 d)
{
    out_poly.clear();
    u32 N=u32(in_poly.size());
    for (u32 i=0;i<N;++i) {
        const Vec3f& cur=in_poly[i];
        const Vec3f& nxt=in_poly[(i+1)%N];
        f32 dc=dot(n,cur)-d, dn=dot(n,nxt)-d;
        if(dc<=0.f) out_poly.push_back(cur);
        if((dc<0.f)!=(dn<0.f)){
            f32 t=dc/(dc-dn);
            out_poly.push_back(cur+(nxt-cur)*t);
        }
    }
}

// Build approximate Voronoi cell for site i by clipping a large box
[[nodiscard]] inline VCell build_voronoi_cell(
    u32 site_idx,
    std::span<const VoronoiSite3D> sites,
    const Vec3f& box_min, const Vec3f& box_max)
{
    // Start with box faces
    Vec3f c=sites[site_idx].pos;
    f32 bx=box_max.x-box_min.x;
    // Represent cell as list of planes (half-spaces)
    std::vector<std::pair<Vec3f,f32>> half_spaces;
    // Box half-spaces
    half_spaces.push_back({{-1,0,0},-box_min.x});
    half_spaces.push_back({{ 1,0,0}, box_max.x});
    half_spaces.push_back({{0,-1,0},-box_min.y});
    half_spaces.push_back({{0, 1,0}, box_max.y});
    half_spaces.push_back({{0,0,-1},-box_min.z});
    half_spaces.push_back({{0,0, 1}, box_max.z});
    // Bisector half-spaces
    for (u32 j=0;j<u32(sites.size());++j) {
        if(j==site_idx) continue;
        auto bp=voronoi_bisector(sites[site_idx],sites[j]);
        half_spaces.push_back({bp.normal,bp.d});
    }

    VCell cell;
    cell.site_id=site_idx;
    // Approximate centroid
    Vec3f centroid{0,0,0}; f32 cnt=0.f;
    // Monte Carlo volume estimate
    u32 N_samples=1000;
    for (u32 k=0;k<N_samples;++k) {
        // Deterministic quasi-random sample
        f32 fx=f32(k*3+1)/f32(N_samples);
        f32 fy=f32(k*7+3)/f32(N_samples);
        f32 fz=f32(k*13+5)/f32(N_samples);
        // Halton-like
        auto halton=[](u32 i,u32 base)->f32{
            f32 r=0,f=1;
            while(i>0){f/=f32(base);r+=f32(i%base)*f;i/=base;}
            return r;
        };
        Vec3f p=box_min+Vec3f{
            halton(k+1,2)*(box_max.x-box_min.x),
            halton(k+1,3)*(box_max.y-box_min.y),
            halton(k+1,5)*(box_max.z-box_min.z)};
        if(voronoi_nearest(p,sites)==site_idx){
            centroid=centroid+p; cnt+=1.f;
        }
    }
    if(cnt>0.f) cell.centroid=centroid*(1.f/cnt);
    cell.volume=cnt/f32(N_samples)*(box_max.x-box_min.x)*(box_max.y-box_min.y)*(box_max.z-box_min.z);
    return cell;
}

// ─── Lloyd relaxation ────────────────────────────────────────────────────────

inline void lloyd_relax(
    std::vector<VoronoiSite3D>& sites,
    const Vec3f& box_min, const Vec3f& box_max,
    u32 iterations=10, u32 mc_samples=2000)
{
    for (u32 iter=0;iter<iterations;++iter) {
        std::vector<Vec3f> centroids(sites.size(),{0,0,0});
        std::vector<f32> counts(sites.size(),0.f);
        auto halton=[](u32 i,u32 base)->f32{
            f32 r=0,f=1;
            while(i>0){f/=f32(base);r+=f32(i%base)*f;i/=base;}
            return r;
        };
        for (u32 k=0;k<mc_samples;++k) {
            Vec3f p=box_min+Vec3f{
                halton(k+1,2)*(box_max.x-box_min.x),
                halton(k+1,3)*(box_max.y-box_min.y),
                halton(k+1,5)*(box_max.z-box_min.z)};
            u32 nearest=voronoi_nearest(p,sites);
            centroids[nearest]=centroids[nearest]+p;
            counts[nearest]+=1.f;
        }
        for (u32 i=0;i<u32(sites.size());++i)
            if(counts[i]>0.f) sites[i].pos=centroids[i]*(1.f/counts[i]);
    }
}

// ─── Apollonius circles (2D weighted Voronoi) ─────────────────────────────────

struct ApolloCircle2D {
    Vec2f center{};
    f32   radius{};
};

// Apollonius circle of two weighted 2D points
[[nodiscard]] inline ApolloCircle2D apollonius_circle(
    Vec2f p1, f32 w1, Vec2f p2, f32 w2)
{
    // Points at power-distance ratio w1:w2
    if (std::abs(w1-w2)<1e-8f) {
        // Perpendicular bisector (infinite radius)
        return {(p1+p2)*0.5f, 1e30f};
    }
    f32 r1sq=w1*w1, r2sq=w2*w2;
    f32 d2=length_sq(p2-p1);
    // Apollonius circle center on line p1-p2
    f32 t = r1sq/(r1sq-r2sq);
    Vec2f c = p1 + (p2-p1)*(t);
    f32 radius = std::sqrt(std::abs(r1sq-d2*t*(1.f-t))+1e-8f);
    return {c, radius};
}

// ─── 3D power cell centroid ───────────────────────────────────────────────────

[[nodiscard]] inline Vec3f power_cell_centroid(
    u32 site_idx, std::span<const VoronoiSite3D> sites,
    const Vec3f& box_min, const Vec3f& box_max, u32 samples=500)
{
    Vec3f acc{0,0,0}; f32 cnt=0.f;
    auto halton=[](u32 i,u32 b)->f32{f32 r=0,f=1;while(i>0){f/=f32(b);r+=f32(i%b)*f;i/=b;}return r;};
    for (u32 k=0;k<samples;++k) {
        Vec3f p=box_min+Vec3f{
            halton(k+1,2)*(box_max.x-box_min.x),
            halton(k+1,3)*(box_max.y-box_min.y),
            halton(k+1,5)*(box_max.z-box_min.z)};
        if(power_nearest(p,sites)==site_idx){acc=acc+p;cnt+=1.f;}
    }
    return cnt>0.f?acc*(1.f/cnt):sites[site_idx].pos;
}

} // namespace qs
