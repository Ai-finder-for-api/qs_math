#pragma once
// qs_math/sculpt_math.hpp — Brush falloff curves, stroke accumulation, dyntopo edge criteria (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "geometry3d.hpp"
#include "noise.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <functional>
#include <numbers>
#include <algorithm>

namespace qs {

// ─── Brush falloff curves ─────────────────────────────────────────────────────
enum class FalloffType : u8 {
    Smooth,       // smooth step
    Sphere,       // √(1-r²)
    Root,         // √(1-r)
    Sharp,        // 1-r²
    Linear,       // 1-r
    Constant,     // 1
    Gaussian,     // exp(-r²)
    Custom        // user-defined
};

[[nodiscard]] inline f32 brush_falloff(f32 r, FalloffType type, f32 hardness=0.5f) {
    r=std::clamp(r,0.f,1.f);
    switch(type) {
        case FalloffType::Smooth:   return r<1?1-r*r*(3-2*r):0;
        case FalloffType::Sphere:   return std::sqrt(std::max(0.f,1-r*r));
        case FalloffType::Root:     return std::sqrt(std::max(0.f,1-r));
        case FalloffType::Sharp:    return std::max(0.f,1-r*r);
        case FalloffType::Linear:   return std::max(0.f,1-r);
        case FalloffType::Constant: return 1.f;
        case FalloffType::Gaussian: return std::exp(-r*r*4*hardness);
        default: return std::max(0.f,1-r);
    }
}

// Anisotropic falloff (elliptical brush)
[[nodiscard]] inline f32 brush_falloff_ellipse(Vec2f delta, f32 rx, f32 ry,
    f32 angle, FalloffType type)
{
    f32 c=std::cos(angle),s=std::sin(angle);
    f32 lx=(delta.x*c+delta.y*s)/rx;
    f32 ly=(-delta.x*s+delta.y*c)/ry;
    f32 r=std::sqrt(lx*lx+ly*ly);
    return brush_falloff(r,type);
}

// ─── Brush stroke ─────────────────────────────────────────────────────────────
struct BrushStroke {
    Vec3f  hit_pos{};          // intersection point on mesh
    Vec3f  hit_normal{};       // surface normal at hit
    f32    radius{0.1f};
    f32    strength{0.5f};
    f32    hardness{0.5f};
    FalloffType falloff{FalloffType::Smooth};
    bool   invert{false};
    u32    symmetry_flags{0};  // bitmask: X=1,Y=2,Z=4

    [[nodiscard]] f32 influence(Vec3f v_pos) const {
        f32 d=dist(v_pos,hit_pos);
        if(d>=radius) return 0;
        f32 r=d/radius;
        f32 f=brush_falloff(r,falloff,hardness)*strength;
        return invert?-f:f;
    }
};

// ─── Sculpt operations ────────────────────────────────────────────────────────

// Draw (pull/push along normal)
inline void sculpt_draw(std::vector<Vec3f>& verts, const std::vector<Vec3f>& normals,
    const BrushStroke& brush)
{
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=brush.influence(verts[i]);
        if(std::abs(w)>1e-6f) verts[i]=verts[i]+normals[i]*w;
    }
}

// Smooth brush (Laplacian)
inline void sculpt_smooth(std::vector<Vec3f>& verts,
    const std::vector<std::vector<u32>>& adj,
    const BrushStroke& brush, f32 factor=1.f)
{
    std::vector<Vec3f> new_verts=verts;
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=brush.influence(verts[i]);
        if(std::abs(w)<1e-6f||adj[i].empty()) continue;
        Vec3f avg{}; for(u32 nb:adj[i]) avg=avg+verts[nb]; avg=avg*(1.f/adj[i].size());
        new_verts[i]=verts[i]+(avg-verts[i])*(w*factor);
    }
    verts=new_verts;
}

// Flatten brush (project towards average plane)
inline void sculpt_flatten(std::vector<Vec3f>& verts, const BrushStroke& brush) {
    // Compute average plane in brush radius
    Vec3f avg_pos{}, avg_normal{};
    f32 total_w=0;
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=std::abs(brush.influence(verts[i]));
        avg_pos=avg_pos+verts[i]*w; total_w+=w;
    }
    if(total_w<1e-7f) return;
    avg_pos=avg_pos*(1/total_w);
    Vec3f plane_normal=brush.hit_normal;
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=brush.influence(verts[i]);
        if(std::abs(w)<1e-6f) continue;
        f32 dist_to_plane=dot(verts[i]-avg_pos,plane_normal);
        verts[i]=verts[i]-plane_normal*(dist_to_plane*w);
    }
}

// Inflate brush (push along local normal)
inline void sculpt_inflate(std::vector<Vec3f>& verts, const std::vector<Vec3f>& normals,
    const BrushStroke& brush)
{
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=brush.influence(verts[i]);
        if(std::abs(w)>1e-6f) verts[i]=verts[i]+normals[i]*w;
    }
}

// Crease brush (draw + negative smooth)
inline void sculpt_crease(std::vector<Vec3f>& verts, const std::vector<Vec3f>& normals,
    const std::vector<std::vector<u32>>& adj, const BrushStroke& brush)
{
    sculpt_draw(verts,normals,brush);
    BrushStroke inv_brush=brush; inv_brush.strength*=-0.5f;
    sculpt_smooth(verts,adj,inv_brush);
}

// Pinch brush (attract vertices toward stroke axis)
inline void sculpt_pinch(std::vector<Vec3f>& verts, const BrushStroke& brush) {
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=brush.influence(verts[i]);
        if(std::abs(w)<1e-6f) continue;
        Vec3f to=brush.hit_pos-verts[i];
        verts[i]=verts[i]+to*w*0.5f;
    }
}

// Grab brush (move vertices rigidly)
inline void sculpt_grab(std::vector<Vec3f>& verts, const BrushStroke& brush,
    Vec3f delta)
{
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=brush.influence(verts[i]);
        if(std::abs(w)>1e-6f) verts[i]=verts[i]+delta*w;
    }
}

// Clay strips (draw + buildup accumulation)
struct ClayAccumulator {
    std::vector<f32> accumulated; // per-vertex accumulated height
    f32 max_height{0.1f};

    void reset(u32 nv) { accumulated.assign(nv,0); }

    void apply(std::vector<Vec3f>& verts, const std::vector<Vec3f>& normals,
        const BrushStroke& brush, f32 dt)
    {
        for(u32 i=0;i<u32(verts.size());++i) {
            f32 w=brush.influence(verts[i])*dt;
            f32 remaining=max_height-std::abs(accumulated[i]);
            f32 actual=std::min(std::abs(w),remaining)*std::copysign(1.f,w);
            accumulated[i]+=actual;
            verts[i]=verts[i]+normals[i]*actual;
        }
    }
};

// ─── Symmetry helpers ─────────────────────────────────────────────────────────
[[nodiscard]] inline std::vector<BrushStroke> symmetry_strokes(const BrushStroke& base) {
    std::vector<BrushStroke> strokes={base};
    if(base.symmetry_flags&0x1) { // X mirror
        BrushStroke s=base; s.hit_pos.x=-s.hit_pos.x; s.hit_normal.x=-s.hit_normal.x;
        strokes.push_back(s);
    }
    if(base.symmetry_flags&0x2) { // Y mirror
        BrushStroke s=base; s.hit_pos.y=-s.hit_pos.y; s.hit_normal.y=-s.hit_normal.y;
        strokes.push_back(s);
    }
    if(base.symmetry_flags&0x4) { // Z mirror
        BrushStroke s=base; s.hit_pos.z=-s.hit_pos.z; s.hit_normal.z=-s.hit_normal.z;
        strokes.push_back(s);
    }
    return strokes;
}

// ─── Dynamic topology (dyntopo) edge criteria ─────────────────────────────────
struct DyntopoConfig {
    f32 detail_size{0.01f};    // target edge length (detail level)
    f32 constant_detail{0.f};  // if > 0, use constant world-space detail
    f32 relative_detail{0.1f}; // relative to brush radius
    bool collapse_enabled{true};
    bool split_enabled{true};
};

// Should edge be split (too long relative to brush)?
[[nodiscard]] inline bool dyntopo_should_split(f32 edge_len, f32 brush_radius,
    const DyntopoConfig& cfg)
{
    f32 target=cfg.constant_detail>0?cfg.constant_detail:brush_radius*cfg.relative_detail;
    return edge_len>target*4.f/3;
}

// Should edge be collapsed (too short)?
[[nodiscard]] inline bool dyntopo_should_collapse(f32 edge_len, f32 brush_radius,
    const DyntopoConfig& cfg)
{
    f32 target=cfg.constant_detail>0?cfg.constant_detail:brush_radius*cfg.relative_detail;
    return edge_len<target*4.f/5;
}

// Dyntopo detail: geometry complexity in brush area
[[nodiscard]] inline f32 dyntopo_local_density(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris,
    Vec3f brush_center, f32 brush_radius)
{
    u32 count=0;
    for(const auto& t:tris) {
        Vec3f c=(verts[t[0]]+verts[t[1]]+verts[t[2]])*(1.f/3);
        if(dist(c,brush_center)<brush_radius) ++count;
    }
    return f32(count)/(f32(std::numbers::pi)*brush_radius*brush_radius);
}

// ─── Multi-resolution sculpting ───────────────────────────────────────────────
struct MeshLevel {
    std::vector<Vec3f>            verts;
    std::vector<std::array<u32,3>>tris;
    std::vector<Vec3f>            displacements; // detail stored as displacements
};

struct MultiResMesh {
    std::vector<MeshLevel> levels; // levels[0] = coarsest

    [[nodiscard]] u32 num_levels() const { return u32(levels.size()); }

    // Subdivide level n to create level n+1 (simple midpoint subdivision)
    void subdivide() {
        if(levels.empty()) return;
        auto& src=levels.back();
        MeshLevel dst;
        dst.verts=src.verts;
        std::unordered_map<u64,u32> edge_midpoints;
        auto ek=[](u32 a,u32 b)->u64{ if(a>b)std::swap(a,b); return (u64(a)<<32)|b; };
        auto get_mid=[&](u32 a,u32 b)->u32 {
            u64 k=ek(a,b);
            auto it=edge_midpoints.find(k);
            if(it!=edge_midpoints.end()) return it->second;
            u32 idx=u32(dst.verts.size());
            dst.verts.push_back((src.verts[a]+src.verts[b])*0.5f);
            edge_midpoints[k]=idx; return idx;
        };
        for(const auto& t:src.tris) {
            u32 m01=get_mid(t[0],t[1]),m12=get_mid(t[1],t[2]),m20=get_mid(t[2],t[0]);
            dst.tris.push_back({t[0],m01,m20});
            dst.tris.push_back({m01,t[1],m12});
            dst.tris.push_back({m20,m12,t[2]});
            dst.tris.push_back({m01,m12,m20});
        }
        dst.displacements.assign(dst.verts.size(),{});
        levels.push_back(dst);
    }

    // Apply displacements at given level to get final vertex positions
    [[nodiscard]] std::vector<Vec3f> get_displaced(u32 level) const {
        auto verts=levels[level].verts;
        for(u32 i=0;i<u32(verts.size());++i) verts[i]=verts[i]+levels[level].displacements[i];
        return verts;
    }
};

// ─── Noise-based sculpting (texture displacement) ─────────────────────────────
inline void sculpt_noise_displace(std::vector<Vec3f>& verts,
    const std::vector<Vec3f>& normals, f32 amplitude, f32 frequency,
    const BrushStroke& brush)
{
    for(u32 i=0;i<u32(verts.size());++i) {
        f32 w=brush.influence(verts[i]);
        if(std::abs(w)<1e-6f) continue;
        f32 n=perlin_3d(verts[i].x*frequency,verts[i].y*frequency,verts[i].z*frequency);
        verts[i]=verts[i]+normals[i]*(n*amplitude*w);
    }
}

} // namespace qs
