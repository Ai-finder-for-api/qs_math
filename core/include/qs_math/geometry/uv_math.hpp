#pragma once
// uv_math.hpp — UV parameterization, LSCM seed, ABF helpers,
//               seam detection, UV chart analysis, texture fetch math,
//               UV transform, tiling, projection methods
#include "vec2.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "scalar_math.hpp"
#include "geometry2d.hpp"
#include "geometry3d.hpp"
#include "mesh_math.hpp"

namespace qs::math::uv {

// ---------------------------------------------------------------------------
// UV projection methods
// ---------------------------------------------------------------------------
// Planar projection
[[nodiscard]] constexpr Vec2f project_planar(Vec3f P, Vec3f normal, Vec3f origin,
                                              Vec3f tangent, Vec3f bitangent) noexcept
{
    Vec3f d=P-origin;
    return {dot(d,tangent), dot(d,bitangent)};
}

// Spherical projection
[[nodiscard]] inline Vec2f project_spherical(Vec3f P, Vec3f center) noexcept {
    Vec3f d=(P-center).normalized();
    f32 phi=atan2(d.z,d.x);
    f32 theta=safe_acos(clamp(d.y,-1.f,1.f));
    return {(phi/TWO_PI)+0.5f, theta/PI};
}

// Cylindrical projection
[[nodiscard]] inline Vec2f project_cylindrical(Vec3f P, Vec3f axis, Vec3f center) noexcept {
    Vec3f d=P-center;
    f32 h=dot(d,axis);
    Vec3f radial=d-axis*h;
    f32 angle=atan2(radial.z,radial.x);
    return {angle/TWO_PI+0.5f, h};
}

// Cube projection (box mapping)
[[nodiscard]] inline Vec2f project_cube(Vec3f P) noexcept {
    auto coord=cubemap_coord(P.normalized());
    return coord.uv;
}

// Triplanar projection (world-space)
[[nodiscard]] inline Vec2f project_triplanar(Vec3f P, Vec3f N, f32 scale=1.f) noexcept {
    Vec3f weights=N.abs();
    weights={pow(weights.x,4.f),pow(weights.y,4.f),pow(weights.z,4.f)};
    f32 inv=1.f/(weights.x+weights.y+weights.z+EPS_F32);
    weights=weights*inv;
    Vec2f uv_x={P.y*scale,P.z*scale};
    Vec2f uv_y={P.x*scale,P.z*scale};
    Vec2f uv_z={P.x*scale,P.y*scale};
    return uv_x*weights.x+uv_y*weights.y+uv_z*weights.z;
}

// ---------------------------------------------------------------------------
// UV distortion metrics
// ---------------------------------------------------------------------------
// Stretch metric (per-triangle, 0=perfect, 1=max stretch)
[[nodiscard]] inline f32 uv_stretch(
    Vec3f p0,Vec3f p1,Vec3f p2,
    Vec2f uv0,Vec2f uv1,Vec2f uv2) noexcept
{
    Vec3f e1=p1-p0, e2=p2-p0;
    Vec2f d1=uv1-uv0, d2=uv2-uv0;
    f32 det=d1.x*d2.y-d1.y*d2.x;
    if(abs(det)<EPS_F32) return 1.f;
    f32 inv=1.f/det;
    Vec3f Ss=(e1*d2.y-e2*d1.y)*inv;
    Vec3f St=(e2*d1.x-e1*d2.x)*inv;
    f32 a=dot(Ss,Ss), b=dot(Ss,St), c=dot(St,St);
    f32 disc=sqrt(max(0.f,(a-c)*(a-c)+4.f*b*b));
    f32 sigma_max=sqrt(max(0.f,(a+c+disc)*0.5f));
    f32 sigma_min=sqrt(max(0.f,(a+c-disc)*0.5f));
    f32 L2_stretch=sqrt((sigma_max*sigma_max+sigma_min*sigma_min)*0.5f);
    return saturate(L2_stretch-1.f);
}

// Angle distortion (MIPS metric)
[[nodiscard]] inline f32 uv_angle_distortion(
    Vec3f p0,Vec3f p1,Vec3f p2,
    Vec2f uv0,Vec2f uv1,Vec2f uv2) noexcept
{
    f32 area3d=triangle_area(p0,p1,p2);
    f32 area2d=abs(polygon_signed_area(&uv0,3));
    if(area3d<EPS_F32||area2d<EPS_F32) return F32_INF;
    // MIPS = (sigma1^2 + sigma2^2) / (sigma1*sigma2)
    Vec3f e1=p1-p0, e2=p2-p0;
    Vec2f d1=uv1-uv0, d2=uv2-uv0;
    f32 det=d1.x*d2.y-d1.y*d2.x;
    if(abs(det)<EPS_F32) return F32_INF;
    f32 inv=1.f/det;
    Vec3f Ss=(e1*d2.y-e2*d1.y)*inv, St=(e2*d1.x-e1*d2.x)*inv;
    f32 a=dot(Ss,Ss), c=dot(St,St), b=dot(Ss,St);
    f32 sigma1=sqrt(max(0.f,0.5f*(a+c+sqrt(max(0.f,(a-c)*(a-c)+4.f*b*b)))));
    f32 sigma2=sqrt(max(0.f,0.5f*(a+c-sqrt(max(0.f,(a-c)*(a-c)+4.f*b*b)))));
    if(sigma2<EPS_F32) return F32_INF;
    return (sigma1*sigma1+sigma2*sigma2)/(sigma1*sigma2);
}

// UV flipped (negative area)
[[nodiscard]] constexpr bool uv_is_flipped(Vec2f uv0,Vec2f uv1,Vec2f uv2) noexcept {
    return polygon_signed_area(&uv0,3)<0.f;
}

// UV overlap detection (brute force, O(n^2))
[[nodiscard]] inline bool uvs_overlap(
    const Vec2f* uva, const Vec2f* uvb) noexcept // each is 3 vertices
{
    // Check if any vertex of A is inside B or vice versa
    return point_in_polygon(uva[0],uvb,3)||point_in_polygon(uvb[0],uva,3);
}

// ---------------------------------------------------------------------------
// UV transform helpers
// ---------------------------------------------------------------------------
struct UVTransform {
    Vec2f offset{0.f};
    Vec2f scale{1.f};
    f32   rotation{0.f}; // radians

    [[nodiscard]] Vec2f apply(Vec2f uv) const noexcept {
        uv=(uv-Vec2f{0.5f})*scale;
        f32 c=cos(rotation),s=sin(rotation);
        uv={uv.x*c-uv.y*s, uv.x*s+uv.y*c};
        return uv+Vec2f{0.5f}+offset;
    }

    [[nodiscard]] Vec2f apply_inverse(Vec2f uv) const noexcept {
        uv=uv-Vec2f{0.5f}-offset;
        f32 c=cos(-rotation),s=sin(-rotation);
        uv={uv.x*c-uv.y*s, uv.x*s+uv.y*c};
        return uv/scale+Vec2f{0.5f};
    }

    static constexpr UVTransform identity() noexcept { return {}; }
    static constexpr UVTransform flip_u() noexcept { return {Vec2f{1.f,0.f},Vec2f{-1.f,1.f},0.f}; }
    static constexpr UVTransform flip_v() noexcept { return {Vec2f{0.f,1.f},Vec2f{1.f,-1.f},0.f}; }
};

// ---------------------------------------------------------------------------
// Seam detection
// ---------------------------------------------------------------------------
// Find UV seam edges (edges where UV coords discontinuous)
inline i32 find_seam_edges(
    const u32* vert_indices,  // triangle vertex indices (into position array)
    const u32* uv_indices,    // triangle UV indices (into UV array)
    i32 tri_count,
    u32* seam_edges,          // output: pairs of vertex indices
    i32 max_seams) noexcept
{
    i32 count=0;
    for(i32 t=0;t<tri_count&&count<max_seams;++t)
        for(i32 e=0;e<3&&count<max_seams;++e){
            u32 vi0=vert_indices[t*3+e],    vi1=vert_indices[t*3+(e+1)%3];
            u32 ui0=uv_indices[t*3+e],      ui1=uv_indices[t*3+(e+1)%3];
            // Find adjacent triangle sharing vi0,vi1 with different UV indices
            for(i32 t2=0;t2<tri_count;++t2){
                if(t2==t) continue;
                for(i32 e2=0;e2<3;++e2){
                    u32 vj0=vert_indices[t2*3+e2], vj1=vert_indices[t2*3+(e2+1)%3];
                    u32 uj0=uv_indices[t2*3+e2],   uj1=uv_indices[t2*3+(e2+1)%3];
                    if((vi0==vj1&&vi1==vj0)&&(ui0!=uj1||ui1!=uj0)){
                        seam_edges[count*2]=vi0; seam_edges[count*2+1]=vi1;
                        ++count;
                    }
                }
            }
        }
    return count;
}

// ---------------------------------------------------------------------------
// Lightmap UV generation helpers (simple, for static geometry)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec2f lightmap_uv_for_tile(
    i32 tile_x, i32 tile_y, i32 atlas_w, i32 atlas_h,
    Vec2f local_uv, f32 padding=0.5f) noexcept
{
    f32 tx=(tile_x+local_uv.x*(1.f-2.f*padding/atlas_w)+padding/atlas_w)/(f32)atlas_w;
    f32 ty=(tile_y+local_uv.y*(1.f-2.f*padding/atlas_h)+padding/atlas_h)/(f32)atlas_h;
    return {tx,ty};
}

// Luma weight for lightmap texel coverage
[[nodiscard]] constexpr f32 lightmap_texel_area(
    Vec3f p0, Vec3f p1, Vec3f p2,
    Vec2f uv0, Vec2f uv1, Vec2f uv2,
    f32 texels_per_world_unit) noexcept
{
    f32 world_area=triangle_area(p0,p1,p2);
    f32 uv_area=abs(polygon_signed_area(&uv0,3));
    if(uv_area<EPS_F32) return 0.f;
    return world_area/(uv_area*texels_per_world_unit*texels_per_world_unit);
}

// ---------------------------------------------------------------------------
// Texture fetch helpers (CPU-side, for baking)
// ---------------------------------------------------------------------------
// Bilinear texture fetch
[[nodiscard]] inline Vec4f tex_fetch_bilinear(
    const Vec4f* texels, i32 w, i32 h, Vec2f uv) noexcept
{
    uv.x=wrap(uv.x,0.f,1.f); uv.y=wrap(uv.y,0.f,1.f);
    f32 fx=uv.x*(w-1), fy=uv.y*(h-1);
    i32 ix=(i32)fx, iy=(i32)fy;
    f32 bx=fx-ix, by=fy-iy;
    ix=min(ix,w-2); iy=min(iy,h-2);
    Vec4f t00=texels[iy*w+ix],     t10=texels[iy*w+ix+1];
    Vec4f t01=texels[(iy+1)*w+ix], t11=texels[(iy+1)*w+ix+1];
    return t00.lerp(t10,bx).lerp(t01.lerp(t11,bx),by);
}

// Mip level selection
[[nodiscard]] constexpr f32 mip_level(Vec2f ddx, Vec2f ddy, i32 tex_w, i32 tex_h) noexcept {
    f32 lx=Vec2f{ddx.x*tex_w,ddx.y*tex_h}.length_sq();
    f32 ly=Vec2f{ddy.x*tex_w,ddy.y*tex_h}.length_sq();
    return 0.5f*log2(max(lx,ly));
}

} // namespace qs::math::uv
