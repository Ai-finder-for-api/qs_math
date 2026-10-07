#pragma once
// terrain_math.hpp — Terrain heightmap math: sampling, normals, LOD,
//                    erosion, river networks, terrain analysis,
//                    chunking, quadtree LOD, CDLOD
#include "vec2.hpp"
#include "vec3.hpp"
#include "noise.hpp"
#include "scalar_math.hpp"
#include "geometry3d.hpp"

namespace qs::math::terrain {

// ---------------------------------------------------------------------------
// Heightmap sampling
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 sample_heightmap_bilinear(
    const f32* heights, i32 width, i32 height,
    f32 u, f32 v) noexcept
{
    u=saturate(u)*(width-1);
    v=saturate(v)*(height-1);
    i32 ix=(i32)u, iy=(i32)v;
    f32 fx=u-ix, fy=v-iy;
    ix=min(ix,width-2); iy=min(iy,height-2);
    f32 h00=heights[iy*width+ix],     h10=heights[iy*width+ix+1];
    f32 h01=heights[(iy+1)*width+ix], h11=heights[(iy+1)*width+ix+1];
    return bilinear(h00,h10,h01,h11,fx,fy);
}

// Smooth bicubic sample
[[nodiscard]] inline f32 sample_heightmap_bicubic(
    const f32* hm, i32 w, i32 h, f32 u, f32 v) noexcept
{
    f32 x=saturate(u)*(w-1), y=saturate(v)*(h-1);
    i32 ix=(i32)x, iy=(i32)y;
    f32 fx=x-ix, fy=y-iy;
    // Sample 4x4 neighborhood
    f32 grid[4][4];
    for(i32 dy=0;dy<4;++dy) for(i32 dx=0;dx<4;++dx){
        i32 sx=clamp(ix+dx-1,0,w-1), sy=clamp(iy+dy-1,0,h-1);
        grid[dx][dy]=hm[sy*w+sx];
    }
    return bicubic(grid,fx,fy);
}

// Normal from heightmap (central differences)
[[nodiscard]] inline Vec3f heightmap_normal(
    const f32* heights, i32 width, i32 height,
    i32 x, i32 y, f32 cell_size=1.f, f32 height_scale=1.f) noexcept
{
    auto h=[&](i32 xi,i32 yi)->f32{
        return heights[clamp(yi,0,height-1)*width+clamp(xi,0,width-1)];
    };
    f32 dz_dx=(h(x+1,y)-h(x-1,y))/(2.f*cell_size)*height_scale;
    f32 dz_dy=(h(x,y+1)-h(x,y-1))/(2.f*cell_size)*height_scale;
    return Vec3f{-dz_dx,1.f,-dz_dy}.normalized();
}

// ---------------------------------------------------------------------------
// Procedural terrain generation
// ---------------------------------------------------------------------------
inline void generate_terrain_fbm(
    f32* heights, i32 width, i32 height,
    f32 scale=0.01f, i32 octaves=8,
    f32 lacunarity=2.f, f32 gain=0.5f, f32 seed_offset=0.f) noexcept
{
    FBMParams params{octaves,lacunarity,gain,0.5f};
    for(i32 y=0;y<height;++y) for(i32 x=0;x<width;++x){
        f32 nx=(x+seed_offset)*scale, ny=(y+seed_offset)*scale;
        heights[y*width+x]=fbm2(nx,ny,params)*0.5f+0.5f;
    }
}

// Domain-warped terrain
inline void generate_terrain_warped(
    f32* heights, i32 width, i32 height, f32 scale=0.005f) noexcept
{
    for(i32 y=0;y<height;++y) for(i32 x=0;x<width;++x){
        f32 nx=x*scale, ny=y*scale;
        heights[y*width+x]=domain_warp2(nx,ny,0.8f)*0.5f+0.5f;
    }
}

// Hydraulic erosion (simplified)
inline void erode_hydraulic(f32* heights, i32 w, i32 h,
                             i32 iterations=50000, PCG32& rng=g_rng) noexcept
{
    for(i32 iter=0;iter<iterations;++iter){
        // Drop a water particle at random position
        f32 px=rng.range_f32(1.f,f32(w-2));
        f32 py=rng.range_f32(1.f,f32(h-2));
        f32 sediment=0.f, water=1.f, speed=0.f;
        Vec2f dir{0.f};
        for(i32 step=0;step<64;++step){
            i32 ix=(i32)px, iy=(i32)py;
            f32 fx=px-ix, fy=py-iy;
            // Bilinear gradient
            f32 h00=heights[iy*w+ix],     h10=heights[iy*w+ix+1];
            f32 h01=heights[(iy+1)*w+ix], h11=heights[(iy+1)*w+ix+1];
            Vec2f grad{(h10-h00)*(1.f-fy)+(h11-h01)*fy,
                       (h01-h00)*(1.f-fx)+(h11-h10)*fx};
            dir=dir*0.05f-grad*0.95f;
            if(dir.length_sq()<EPS_F32) break;
            dir=dir.normalized();
            f32 new_px=px+dir.x, new_py=py+dir.y;
            if(new_px<0||new_px>=w-1||new_py<0||new_py>=h-1) break;
            f32 old_h=bilinear(h00,h10,h01,h11,fx,fy);
            f32 new_h=sample_heightmap_bilinear(heights,w,h,new_px/(w-1),new_py/(h-1));
            f32 dh=new_h-old_h;
            f32 capacity=max(-dh,0.01f)*speed*water*4.f;
            if(sediment>capacity){
                // Deposit
                f32 deposit=(sediment-capacity)*0.3f;
                heights[iy*w+ix]+=deposit*(1.f-fx)*(1.f-fy);
                heights[iy*w+ix+1]+=deposit*fx*(1.f-fy);
                heights[(iy+1)*w+ix]+=deposit*(1.f-fx)*fy;
                heights[(iy+1)*w+ix+1]+=deposit*fx*fy;
                sediment-=deposit;
            } else {
                // Erode
                f32 erode=min((capacity-sediment)*0.3f,abs(dh));
                heights[iy*w+ix]-=erode*(1.f-fx)*(1.f-fy);
                sediment+=erode;
            }
            speed=sqrt(abs(dh)*9.8f);
            water*=0.99f;
            px=new_px; py=new_py;
        }
    }
}

// Thermal erosion (talus formation)
inline void erode_thermal(f32* heights, i32 w, i32 h,
                           f32 talus_angle=0.1f, i32 iterations=100) noexcept
{
    for(i32 iter=0;iter<iterations;++iter){
        for(i32 y=1;y<h-1;++y) for(i32 x=1;x<w-1;++x){
            f32 cur=heights[y*w+x];
            static constexpr i32 dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1};
            f32 total_diff=0.f; f32 diffs[4]={};
            for(i32 d=0;d<4;++d){
                f32 diff=cur-heights[(y+dy[d])*w+(x+dx[d])];
                if(diff>talus_angle){ diffs[d]=diff; total_diff+=diff; }
            }
            if(total_diff<EPS_F32) continue;
            f32 move=total_diff*0.5f/total_diff;
            for(i32 d=0;d<4;++d) if(diffs[d]>0.f){
                f32 dm=move*diffs[d];
                heights[y*w+x]-=dm;
                heights[(y+dy[d])*w+(x+dx[d])]+=dm;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain analysis
// ---------------------------------------------------------------------------
// Slope in radians
[[nodiscard]] inline f32 terrain_slope(
    const f32* hm, i32 w, i32 h, i32 x, i32 y,
    f32 cell_size=1.f, f32 height_scale=1.f) noexcept
{
    Vec3f N=heightmap_normal(hm,w,h,x,y,cell_size,height_scale);
    return safe_acos(N.y);
}

// Aspect (direction of steepest descent, radians from north)
[[nodiscard]] inline f32 terrain_aspect(const f32* hm, i32 w, i32 h, i32 x, i32 y) noexcept {
    auto hxy=[&](i32 xi,i32 yi)->f32{
        return hm[clamp(yi,0,h-1)*w+clamp(xi,0,w-1)];
    };
    f32 dz_dx=(hxy(x+1,y)-hxy(x-1,y))*0.5f;
    f32 dz_dy=(hxy(x,y+1)-hxy(x,y-1))*0.5f;
    return atan2(dz_dy,-dz_dx);
}

// Curvature (profile curvature)
[[nodiscard]] inline f32 terrain_curvature(const f32* hm, i32 w, i32 h, i32 x, i32 y) noexcept {
    auto h=[&](i32 xi,i32 yi)->f32{
        return hm[clamp(yi,0,h-1)*w+clamp(xi,0,w-1)];
    };
    f32 c=h(x,y);
    f32 d2x=h(x+1,y)-2.f*c+h(x-1,y);
    f32 d2y=h(x,y+1)-2.f*c+h(x,y-1);
    return d2x+d2y;
}

// Topographic Wetness Index
[[nodiscard]] constexpr f32 twi(f32 upslope_area, f32 slope_radians) noexcept {
    return log(upslope_area/(tan(max(slope_radians,0.01f))+EPS_F32));
}

// ---------------------------------------------------------------------------
// CDLOD (Continuous Distance-Dependent Level of Detail) helpers
// ---------------------------------------------------------------------------
struct CDLODNode {
    Vec2f min_xy, max_xy;
    f32   min_h, max_h;
    i32   lod_level;
    bool  is_leaf;
};

[[nodiscard]] inline f32 cdlod_morph_alpha(
    Vec3f camera_pos, Vec3f node_center,
    f32 lod_range, f32 morph_start=0.6f) noexcept
{
    f32 dist=(camera_pos-node_center).length();
    return smoothstep(lod_range*morph_start,lod_range,dist);
}

// Morphed vertex position for seamless LOD transitions
[[nodiscard]] inline Vec3f cdlod_morph_vertex(
    Vec3f base_pos, Vec3f coarser_pos, f32 morph_alpha) noexcept
{
    return lerp(base_pos,coarser_pos,morph_alpha);
}

// Quadtree split decision
[[nodiscard]] constexpr bool cdlod_should_split(
    Vec3f camera_pos, Vec3f node_center, f32 node_size, i32 lod, f32 base_range=64.f) noexcept
{
    f32 range=base_range*pow(2.f,f32(lod));
    return (camera_pos-node_center).length()<range*1.5f&&node_size>1.f;
}

} // namespace qs::math::terrain
