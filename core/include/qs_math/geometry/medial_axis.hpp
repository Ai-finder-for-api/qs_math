#pragma once
// qs_math/medial_axis.hpp — MAT, Voronoi skeleton, thinning, chordal axis (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "geometry2d.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <span>

namespace qs {

// ─── Medial ball ─────────────────────────────────────────────────────────────

struct MedialBall {
    Vec3f center{};
    f32   radius{};
    u32   feature_a{}, feature_b{}; // closest boundary features
};

// Fit maximal inscribed circle in 2D polygon at a given query point
[[nodiscard]] inline f32 inscribed_radius_2d(
    const Vec2f& query,
    std::span<const Vec2f> polygon)
{
    f32 min_d = std::numeric_limits<f32>::max();
    u32 N = u32(polygon.size());
    for (u32 i=0;i<N;++i) {
        const Vec2f& a=polygon[i], &b=polygon[(i+1)%N];
        Vec2f ab=b-a, ap=query-a;
        f32 t=std::clamp(dot(ap,ab)/dot(ab,ab),0.f,1.f);
        f32 d=length(query-(a+ab*t));
        min_d=std::min(min_d,d);
    }
    return min_d;
}

// ─── Grid-based MAT (distance transform skeleton) ────────────────────────────

struct GridMAT {
    std::vector<f32> dist;      // distance transform
    std::vector<u8>  skeleton;  // 1 = skeleton voxel
    u32 width{}, height{};

    void compute_from_binary(const std::vector<u8>& binary, u32 w, u32 h) {
        width=w; height=h;
        dist.assign(w*h, std::numeric_limits<f32>::max());
        skeleton.assign(w*h, 0);

        // Chamfer distance (approximate EDT)
        // Forward pass
        for (u32 y=0;y<h;++y)
        for (u32 x=0;x<w;++x) {
            if (!binary[y*w+x]) { dist[y*w+x]=0.f; continue; }
            f32 d=std::numeric_limits<f32>::max();
            if(x>0)   d=std::min(d,dist[y*w+x-1]+1.f);
            if(y>0)   d=std::min(d,dist[(y-1)*w+x]+1.f);
            if(x>0&&y>0) d=std::min(d,dist[(y-1)*w+x-1]+1.414f);
            if(x<w-1&&y>0) d=std::min(d,dist[(y-1)*w+x+1]+1.414f);
            dist[y*w+x]=d;
        }
        // Backward pass
        for (i32 y=i32(h)-1;y>=0;--y)
        for (i32 x=i32(w)-1;x>=0;--x) {
            f32& d=dist[y*w+x];
            if(u32(x)<w-1)   d=std::min(d,dist[y*w+x+1]+1.f);
            if(u32(y)<h-1)   d=std::min(d,dist[(y+1)*w+x]+1.f);
            if(u32(x)<w-1&&u32(y)<h-1) d=std::min(d,dist[(y+1)*w+x+1]+1.414f);
            if(x>0&&u32(y)<h-1) d=std::min(d,dist[(y+1)*w+x-1]+1.414f);
        }
    }

    // Extract skeleton via local maxima of distance transform
    void extract_skeleton(f32 threshold=0.1f) {
        for (u32 y=1;y<height-1;++y)
        for (u32 x=1;x<width-1;++x) {
            f32 d=dist[y*width+x];
            if(d<1.f) continue;
            bool is_max=true;
            for(i32 dy=-1;dy<=1&&is_max;++dy)
            for(i32 dx=-1;dx<=1&&is_max;++dx) {
                if(dx==0&&dy==0) continue;
                if(dist[(y+dy)*width+(x+dx)]>d+threshold) is_max=false;
            }
            skeleton[y*width+x]=is_max?1:0;
        }
    }
};

// ─── Voronoi-based skeleton ───────────────────────────────────────────────────

struct SkeletonEdge {
    Vec2f a{}, b{};
    f32 radius_a{}, radius_b{};
};

// Sample polygon boundary, compute Voronoi, filter to interior MAT edges
[[nodiscard]] inline std::vector<SkeletonEdge> voronoi_skeleton_2d(
    std::span<const Vec2f> polygon_samples,
    std::span<const Vec2f> interior_candidates)
{
    std::vector<SkeletonEdge> edges;
    u32 N=u32(polygon_samples.size());
    u32 M=u32(interior_candidates.size());

    // For each interior point, find 2 nearest boundary samples
    for (u32 i=0;i<M;++i) {
        const Vec2f& p=interior_candidates[i];
        // Find two nearest
        f32 d0=std::numeric_limits<f32>::max(), d1=std::numeric_limits<f32>::max();
        u32 j0=0,j1=1;
        for (u32 j=0;j<N;++j) {
            f32 d=length_sq(p-polygon_samples[j]);
            if(d<d0){d1=d0;j1=j0;d0=d;j0=j;}
            else if(d<d1){d1=d;j1=j;}
        }
        // A skeleton point has (at least) two equidistant boundary features
        f32 r0=std::sqrt(d0), r1=std::sqrt(d1);
        if(std::abs(r0-r1)<0.5f) {
            // This is approximately on the MAT
            if(!edges.empty()) {
                edges.back().b=p;
                edges.back().radius_b=r0;
            }
            edges.push_back({p,p,r0,r0});
        }
    }
    return edges;
}

// ─── Thinning (Zhang-Suen 2D binary thinning) ────────────────────────────────

inline void zhang_suen_thin(std::vector<u8>& img, u32 w, u32 h) {
    auto at=[&](i32 x,i32 y)->u8{
        return (x<0||y<0||u32(x)>=w||u32(y)>=h)?0:img[u32(y)*w+u32(x)];};
    bool changed=true;
    while(changed) {
        changed=false;
        for (u32 pass=0;pass<2;++pass) {
            std::vector<u32> remove;
            for (u32 y=1;y<h-1;++y)
            for (u32 x=1;x<w-1;++x) {
                if(!img[y*w+x]) continue;
                u8 p2=at(x,y-1),p3=at(x+1,y-1),p4=at(x+1,y),p5=at(x+1,y+1);
                u8 p6=at(x,y+1),p7=at(x-1,y+1),p8=at(x-1,y),p9=at(x-1,y-1);
                u32 B=p2+p3+p4+p5+p6+p7+p8+p9;
                if(B<2||B>6) continue;
                u32 A=0;
                if(!p2&&p3)++A; if(!p3&&p4)++A; if(!p4&&p5)++A; if(!p5&&p6)++A;
                if(!p6&&p7)++A; if(!p7&&p8)++A; if(!p8&&p9)++A; if(!p9&&p2)++A;
                if(A!=1) continue;
                if(pass==0&&(p2*p4*p6||p4*p6*p8)) continue;
                if(pass==1&&(p2*p4*p8||p2*p6*p8)) continue;
                remove.push_back(y*w+x);
            }
            for(u32 idx:remove){img[idx]=0;changed=true;}
        }
    }
}

// ─── Chordal axis transform ───────────────────────────────────────────────────

struct ChordalSegment {
    Vec2f p{}, q{};   // chord endpoints on boundary
    Vec2f center{};   // chord midpoint (approximate MAT center)
    f32   radius{};   // half chord length
};

[[nodiscard]] inline std::vector<ChordalSegment> chordal_axis(
    std::span<const Vec2f> triangulation_centers,
    std::span<const f32>   inscribed_radii,
    f32 pruning_threshold=0.2f)
{
    std::vector<ChordalSegment> segs;
    u32 N=u32(triangulation_centers.size());
    for (u32 i=0;i<N&&i+1<N;++i) {
        f32 r=inscribed_radii[i];
        if(r<pruning_threshold) continue;
        segs.push_back({triangulation_centers[i]-Vec2f{r,0},
                        triangulation_centers[i]+Vec2f{r,0},
                        triangulation_centers[i], r});
    }
    return segs;
}

// ─── Shrink-wrapping for 3D MAT ───────────────────────────────────────────────

[[nodiscard]] inline std::vector<MedialBall> sample_medial_balls_3d(
    std::span<const Vec3f> surface_samples,
    u32 interior_samples=200, f32 shrink=0.5f)
{
    // Find bounding box
    Vec3f mn=surface_samples[0], mx=surface_samples[0];
    for(const auto& p:surface_samples){
        mn.x=std::min(mn.x,p.x);mn.y=std::min(mn.y,p.y);mn.z=std::min(mn.z,p.z);
        mx.x=std::max(mx.x,p.x);mx.y=std::max(mx.y,p.y);mx.z=std::max(mx.z,p.z);
    }
    Vec3f range=mx-mn;

    std::vector<MedialBall> balls;
    auto halton=[](u32 i,u32 b)->f32{f32 r=0,f=1;while(i>0){f/=f32(b);r+=f32(i%b)*f;i/=b;}return r;};
    for (u32 k=0;k<interior_samples;++k) {
        Vec3f p=mn+Vec3f{halton(k+1,2)*range.x,halton(k+1,3)*range.y,halton(k+1,5)*range.z};
        // Find two nearest surface points
        f32 d0=1e30f,d1=1e30f; u32 j0=0,j1=0;
        for(u32 j=0;j<u32(surface_samples.size());++j) {
            f32 d=length(p-surface_samples[j]);
            if(d<d0){d1=d0;j1=j0;d0=d;j0=j;}
            else if(d<d1){d1=d;j1=j;}
        }
        if(std::abs(d0-d1)<d0*0.1f)
            balls.push_back({p,d0,j0,j1});
    }
    return balls;
}

} // namespace qs
