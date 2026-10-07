#pragma once
// qs_math/displacement_brush.hpp — Brush radius/strength/falloff, multi-res displacement (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "sculpt_math.hpp"
#include "curves.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <array>
#include <numbers>

namespace qs {

// ─── Advanced brush parameters ────────────────────────────────────────────────
struct BrushParams {
    f32  radius{0.1f};
    f32  strength{0.5f};
    f32  hardness{0.5f};
    f32  spacing{0.1f};       // fraction of radius between dabs
    f32  jitter{0};           // position randomness [0,1]
    f32  scatter{0};          // scatter radius [0,1]
    u32  count{1};            // number of dabs per step
    f32  angle{0};            // brush rotation
    f32  size_pressure{1};    // pressure-to-size mapping
    f32  strength_pressure{1};// pressure-to-strength mapping
    f32  alpha{1};            // opacity/alpha
    FalloffType falloff{FalloffType::Smooth};

    // Apply pressure curve (bezier or linear)
    [[nodiscard]] f32 effective_radius(f32 pressure) const { return radius*std::pow(pressure,1.f/size_pressure); }
    [[nodiscard]] f32 effective_strength(f32 pressure) const { return strength*std::pow(pressure,1.f/strength_pressure); }
};

// ─── Stroke interpolation (dab placement along cursor path) ───────────────────
struct StrokePath {
    std::vector<Vec3f> positions;
    std::vector<Vec3f> normals;
    std::vector<f32>   pressures;
    std::vector<f32>   arc_lengths; // cumulative arc length

    void add(Vec3f pos, Vec3f normal, f32 pressure) {
        if(!positions.empty()) {
            f32 dl=dist(pos,positions.back());
            arc_lengths.push_back(arc_lengths.back()+dl);
        } else { arc_lengths.push_back(0); }
        positions.push_back(pos);
        normals.push_back(normal);
        pressures.push_back(pressure);
    }

    [[nodiscard]] f32 total_length() const { return arc_lengths.empty()?0:arc_lengths.back(); }

    // Sample stroke at arc length s (linear interpolation)
    struct Sample { Vec3f pos,normal; f32 pressure; };
    [[nodiscard]] Sample sample_at(f32 s) const {
        u32 n=u32(positions.size());
        if(n==0) return {};
        if(s<=0) return {positions[0],normals[0],pressures[0]};
        if(s>=arc_lengths.back()) return {positions[n-1],normals[n-1],pressures[n-1]};
        u32 lo=0,hi=n-1;
        while(hi-lo>1){ u32 mid=(lo+hi)/2; if(arc_lengths[mid]<s)lo=mid;else hi=mid; }
        f32 t=(arc_lengths[hi]-arc_lengths[lo])>1e-7f?(s-arc_lengths[lo])/(arc_lengths[hi]-arc_lengths[lo]):0;
        return {
            positions[lo]+(positions[hi]-positions[lo])*t,
            normalize(normals[lo]+(normals[hi]-normals[lo])*t),
            pressures[lo]+(pressures[hi]-pressures[lo])*t
        };
    }

    // Generate evenly-spaced dab positions along the stroke
    [[nodiscard]] std::vector<Sample> dab_positions(f32 spacing_world) const {
        std::vector<Sample> dabs;
        f32 len=total_length();
        if(len<1e-7f){ if(!positions.empty()) dabs.push_back({positions[0],normals[0],pressures[0]}); return dabs; }
        for(f32 s=0;s<=len+spacing_world*0.5f;s+=spacing_world) dabs.push_back(sample_at(s));
        return dabs;
    }
};

// ─── Displacement layer ────────────────────────────────────────────────────────
struct DisplacementLayer {
    std::vector<f32> heights;   // per-vertex displacement magnitude
    std::vector<Vec3f> directions; // per-vertex displacement direction (usually normal)
    f32 scale{1.f};
    bool enabled{true};

    DisplacementLayer() = default;
    explicit DisplacementLayer(u32 nv) : heights(nv,0), directions(nv,{0,1,0}) {}

    void apply_to_verts(std::vector<Vec3f>& verts) const {
        if(!enabled) return;
        for(u32 i=0;i<u32(verts.size())&&i<u32(heights.size());++i)
            verts[i]=verts[i]+directions[i]*(heights[i]*scale);
    }

    void add_dab(u32 vi, Vec3f normal, f32 h_delta, f32 weight) {
        if(vi>=u32(heights.size())) return;
        heights[vi]+=h_delta*weight;
        directions[vi]=normalize(directions[vi]+normal*weight);
    }

    void clear() { std::fill(heights.begin(),heights.end(),0); }

    // Blend two layers
    void blend(const DisplacementLayer& other, f32 t) {
        for(u32 i=0;i<u32(heights.size());++i) heights[i]=heights[i]*(1-t)+other.heights[i]*t;
    }
};

// ─── Brush texture (alpha mask) ───────────────────────────────────────────────
struct BrushTexture {
    std::vector<f32> data; // [width × height], normalized [0,1]
    u32 width{64},height{64};

    BrushTexture() { data.assign(64*64,1.f); }

    [[nodiscard]] f32 sample(Vec2f uv) const {
        u32 x=u32(std::clamp(uv.x,0.f,1.f)*(width-1));
        u32 y=u32(std::clamp(uv.y,0.f,1.f)*(height-1));
        return data[y*width+x];
    }

    // Sample bilinearly
    [[nodiscard]] f32 sample_bilinear(Vec2f uv) const {
        f32 fx=std::clamp(uv.x,0.f,1.f)*(width-1);
        f32 fy=std::clamp(uv.y,0.f,1.f)*(height-1);
        u32 x0=u32(fx),y0=u32(fy);
        u32 x1=std::min(x0+1,width-1),y1=std::min(y0+1,height-1);
        f32 tx=fx-x0,ty=fy-y0;
        f32 v00=data[y0*width+x0],v10=data[y0*width+x1];
        f32 v01=data[y1*width+x0],v11=data[y1*width+x1];
        return (v00*(1-tx)+v10*tx)*(1-ty)+(v01*(1-tx)+v11*tx)*ty;
    }

    // Generate procedural textures
    static BrushTexture make_round(u32 sz=64) {
        BrushTexture t; t.width=sz; t.height=sz; t.data.resize(sz*sz);
        for(u32 y=0;y<sz;++y) for(u32 x=0;x<sz;++x) {
            f32 dx=f32(x)/sz-0.5f,dy=f32(y)/sz-0.5f;
            f32 r=std::sqrt(dx*dx+dy*dy)*2;
            t.data[y*sz+x]=r<1?1-r*r:0;
        }
        return t;
    }
    static BrushTexture make_star(u32 sz=64, u32 points=5) {
        BrushTexture t; t.width=sz; t.height=sz; t.data.resize(sz*sz,0);
        for(u32 y=0;y<sz;++y) for(u32 x=0;x<sz;++x) {
            f32 dx=f32(x)/sz-0.5f,dy=f32(y)/sz-0.5f;
            f32 r=std::sqrt(dx*dx+dy*dy)*2;
            f32 angle=std::atan2(dy,dx);
            f32 star=std::cos(angle*points)*0.5f+0.5f;
            t.data[y*sz+x]=r<star?1.f-r/star:0.f;
        }
        return t;
    }
};

// ─── Multi-resolution displacement mapping ────────────────────────────────────
struct MultiResDisplacement {
    static constexpr u32 MAX_LEVELS=8;
    std::array<DisplacementLayer,MAX_LEVELS> levels;
    u32 num_levels{1};
    u32 current_level{0};

    void add_level(u32 nv) {
        if(num_levels<MAX_LEVELS){ levels[num_levels++]=DisplacementLayer(nv); }
    }

    DisplacementLayer& current() { return levels[current_level]; }

    // Accumulate all levels into final displacement
    [[nodiscard]] DisplacementLayer bake(u32 nv) const {
        DisplacementLayer result(nv);
        for(u32 l=0;l<num_levels;++l) {
            if(!levels[l].enabled) continue;
            for(u32 i=0;i<nv;++i) {
                result.heights[i]+=levels[l].heights[i]*levels[l].scale;
                result.directions[i]=normalize(result.directions[i]+levels[l].directions[i]);
            }
        }
        return result;
    }
};

// ─── PN triangles (point-normal subdivision for displacement) ──────────────────
struct PNTriangle {
    // Cubic control points for position (10 points)
    Vec3f b300,b030,b003;
    Vec3f b210,b120,b021,b012,b102,b201;
    Vec3f b111;
    // Quadratic control points for normal (6 points)
    Vec3f n200,n020,n002,n110,n011,n101;

    // Build from triangle
    static PNTriangle build(Vec3f p0,Vec3f p1,Vec3f p2,Vec3f n0,Vec3f n1,Vec3f n2) {
        PNTriangle pn;
        pn.b300=p0; pn.b030=p1; pn.b003=p2;
        // Edge control points: b_ij = p_i + (1/3) * (p_i - p_j - dot(p_j-p_i,n_i)*n_i)... simplified:
        auto edge_cp=[&](Vec3f pi,Vec3f pj,Vec3f ni)->Vec3f{
            return (pi*2+pj-ni*dot(pj-pi,ni))*(1.f/3);
        };
        pn.b210=edge_cp(p0,p1,n0); pn.b120=edge_cp(p1,p0,n1);
        pn.b021=edge_cp(p1,p2,n1); pn.b012=edge_cp(p2,p1,n2);
        pn.b102=edge_cp(p2,p0,n2); pn.b201=edge_cp(p0,p2,n0);
        Vec3f E=(pn.b210+pn.b120+pn.b021+pn.b012+pn.b102+pn.b201)*(1.f/6);
        Vec3f V=(p0+p1+p2)*(1.f/3);
        pn.b111=E+(E-V)*0.5f;
        // Normal control points
        auto nij=[&](Vec3f pi,Vec3f pj,Vec3f ni)->Vec3f{
            return ni-2*dot(pj-pi,ni)/(dot(pj-pi,pj-pi)+1e-10f)*(pj-pi)+2*dot(pj-pi,ni)/(dot(ni+nj,pj-pi)+1e-10f)*(ni+n1);
            // simplified: reflect normal
        };
        pn.n200=n0; pn.n020=n1; pn.n002=n2;
        pn.n110=normalize(n0+n1); pn.n011=normalize(n1+n2); pn.n101=normalize(n0+n2);
        return pn;
    }

    // Evaluate at barycentric coordinates (u,v,w=1-u-v)
    [[nodiscard]] Vec3f position(f32 u,f32 v) const {
        f32 w=1-u-v;
        return b300*(w*w*w)+b030*(u*u*u)+b003*(v*v*v)
              +b210*(3*w*w*u)+b120*(3*w*u*u)+b021*(3*u*u*v)
              +b012*(3*u*v*v)+b102*(3*w*v*v)+b201*(3*w*w*v)
              +b111*(6*w*u*v);
    }
    [[nodiscard]] Vec3f normal(f32 u,f32 v) const {
        f32 w=1-u-v;
        Vec3f n=n200*(w*w)+n020*(u*u)+n002*(v*v)+n110*(w*u)+n011*(u*v)+n101*(w*v);
        return normalize(n);
    }
};

// ─── Displacement shader helper (GPU-ready struct) ────────────────────────────
struct DisplacementShaderData {
    f32 height_scale{0.1f};
    f32 uv_scale{1.f};
    u32 texture_idx{0};       // index into texture array
    u32 channel{0};           // R/G/B/A channel
    bool invert{false};
    f32 midlevel{0.5f};       // neutral displacement height

    // GPU-ready packed params (16 bytes)
    [[nodiscard]] std::array<f32,4> packed() const {
        return {height_scale, uv_scale, midlevel, f32(texture_idx|(invert?0x80000000u:0u))};
    }
};

} // namespace qs
