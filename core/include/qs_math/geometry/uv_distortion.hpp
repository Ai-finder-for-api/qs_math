#pragma once
// qs_math/uv_distortion.hpp — Stretch metric L2/MIPS/angle, area preservation, conformal energy (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "parameterization.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace qs {

// ─── Per-triangle distortion metrics ─────────────────────────────────────────

// Compute the 2x2 Jacobian of the UV map for a triangle
// Maps 2D surface (parameterized) to 2D UV
[[nodiscard]] inline std::array<f32,4> triangle_jacobian(
    Vec3f p0,Vec3f p1,Vec3f p2,  // 3D positions
    Vec2f u0,Vec2f u1,Vec2f u2)  // UV coordinates
{
    // Build orthonormal 2D frame on triangle in 3D
    Vec3f e1=p1-p0,e2=p2-p0;
    Vec3f n=normalize(cross(e1,e2));
    Vec3f ex=normalize(e1);
    Vec3f ey=cross(n,ex);
    // 2D coordinates in surface frame
    f32 s1=length(e1);
    f32 s2x=dot(e2,ex),s2y=dot(e2,ey);
    // UV differences
    Vec2f du1=u1-u0,du2=u2-u0;
    // J = [du/ds1  du/ds2; dv/ds1  dv/ds2]
    // We solve: [s1 s2x; 0 s2y] * J^T = [du1; du2]
    // J^T = [s1 0; s2x s2y]^{-1} * [du1^T; du2^T]
    f32 det=s1*s2y;
    if(std::abs(det)<1e-10f) return {0,0,0,0};
    f32 inv_det=1.f/det;
    // J = [[du1.x du2.x] [du1.y du2.y]] * inv([[s1 s2x][0 s2y]])
    f32 J00=(du1.x*s2y-du2.x*0)*inv_det;
    f32 J01=(du2.x*s1-du1.x*s2x)*inv_det;
    f32 J10=(du1.y*s2y-du2.y*0)*inv_det;
    f32 J11=(du2.y*s1-du1.y*s2x)*inv_det;
    return {J00,J01,J10,J11};
}

// Singular values of 2x2 matrix via closed form
[[nodiscard]] inline std::pair<f32,f32> svd2x2(f32 J00,f32 J01,f32 J10,f32 J11) {
    f32 a=J00*J00+J10*J10,b=J00*J01+J10*J11,c=J01*J01+J11*J11;
    f32 trace=a+c, det=a*c-b*b;
    f32 disc=std::sqrt(std::max(0.f,(trace*trace*0.25f-det)));
    f32 sv1=std::sqrt(std::max(0.f,trace*0.5f+disc));
    f32 sv2=std::sqrt(std::max(0.f,trace*0.5f-disc));
    return {sv1,sv2};
}

// L2 stretch metric for a triangle: σ = sqrt((σ1²+σ2²)/2)
// = 1 for isometric mapping
[[nodiscard]] inline f32 l2_stretch_tri(Vec3f p0,Vec3f p1,Vec3f p2,Vec2f u0,Vec2f u1,Vec2f u2) {
    auto J=triangle_jacobian(p0,p1,p2,u0,u1,u2);
    auto [s1,s2]=svd2x2(J[0],J[1],J[2],J[3]);
    return std::sqrt((s1*s1+s2*s2)*0.5f);
}

// MIPS (Most Isometric Parameterizations) energy: σ1/σ2 + σ2/σ1
// Minimised when σ1=σ2 (conformal)
[[nodiscard]] inline f32 mips_energy_tri(Vec3f p0,Vec3f p1,Vec3f p2,Vec2f u0,Vec2f u1,Vec2f u2) {
    auto J=triangle_jacobian(p0,p1,p2,u0,u1,u2);
    auto [s1,s2]=svd2x2(J[0],J[1],J[2],J[3]);
    if(s2<1e-10f||s1<1e-10f) return 1e6f;
    return s1/s2+s2/s1;
}

// Angle distortion: measures deviation from conformal mapping
// Perfect = 1 (all angles preserved)
[[nodiscard]] inline f32 angle_distortion_tri(Vec3f p0,Vec3f p1,Vec3f p2,Vec2f u0,Vec2f u1,Vec2f u2) {
    auto J=triangle_jacobian(p0,p1,p2,u0,u1,u2);
    auto [s1,s2]=svd2x2(J[0],J[1],J[2],J[3]);
    if(std::min(s1,s2)<1e-10f) return 1e6f;
    return std::max(s1/s2,s2/s1); // 1=conformal, >1=distorted
}

// Area distortion: ratio of UV area to 3D surface area
[[nodiscard]] inline f32 area_distortion_tri(Vec3f p0,Vec3f p1,Vec3f p2,Vec2f u0,Vec2f u1,Vec2f u2) {
    f32 A3d=length(cross(p1-p0,p2-p0))*0.5f;
    f32 A2d=std::abs((u1.x-u0.x)*(u2.y-u0.y)-(u2.x-u0.x)*(u1.y-u0.y))*0.5f;
    if(A3d<1e-10f) return 0;
    return A2d/A3d;
}

// ─── Mesh-wide distortion statistics ─────────────────────────────────────────
struct DistortionStats {
    f32 l2_mean{0},l2_max{0};
    f32 mips_mean{0},mips_max{0};
    f32 angle_mean{0},angle_max{0};
    f32 area_mean{0},area_variance{0};
    u32 flipped_triangles{0}; // triangles with negative area in UV
    f32 total_area_3d{0},total_area_uv{0};
};

[[nodiscard]] inline DistortionStats compute_distortion(
    const std::vector<Vec3f>& verts3d,
    const std::vector<Vec2f>& uvs,
    const std::vector<std::array<u32,3>>& tris)
{
    DistortionStats s;
    u32 n=u32(tris.size());
    if(n==0) return s;
    std::vector<f32> area_ratios(n);
    for(u32 i=0;i<n;++i) {
        auto& t=tris[i];
        Vec3f p0=verts3d[t[0]],p1=verts3d[t[1]],p2=verts3d[t[2]];
        Vec2f u0=uvs[t[0]],u1=uvs[t[1]],u2=uvs[t[2]];
        f32 l2=l2_stretch_tri(p0,p1,p2,u0,u1,u2);
        f32 mi=mips_energy_tri(p0,p1,p2,u0,u1,u2);
        f32 an=angle_distortion_tri(p0,p1,p2,u0,u1,u2);
        f32 ar=area_distortion_tri(p0,p1,p2,u0,u1,u2);
        f32 A3=length(cross(p1-p0,p2-p0))*0.5f;
        f32 A2=std::abs((u1.x-u0.x)*(u2.y-u0.y)-(u2.x-u0.x)*(u1.y-u0.y))*0.5f;
        f32 A2_signed=(u1.x-u0.x)*(u2.y-u0.y)-(u2.x-u0.x)*(u1.y-u0.y);
        if(A2_signed<0) ++s.flipped_triangles;
        s.l2_mean+=l2; s.l2_max=std::max(s.l2_max,l2);
        s.mips_mean+=mi; s.mips_max=std::max(s.mips_max,mi);
        s.angle_mean+=an; s.angle_max=std::max(s.angle_max,an);
        s.area_mean+=ar; area_ratios[i]=ar;
        s.total_area_3d+=A3; s.total_area_uv+=A2;
    }
    s.l2_mean/=n; s.mips_mean/=n; s.angle_mean/=n; s.area_mean/=n;
    f32 var=0; for(auto r:area_ratios) var+=(r-s.area_mean)*(r-s.area_mean);
    s.area_variance=var/n;
    return s;
}

// ─── Conformal energy (Dirichlet) ────────────────────────────────────────────
[[nodiscard]] inline f64 dirichlet_energy(
    const std::vector<Vec3f>& verts3d,
    const std::vector<Vec2f>& uvs,
    const std::vector<std::array<u32,3>>& tris)
{
    f64 E=0;
    for(const auto& t:tris) {
        Vec3f p0=verts3d[t[0]],p1=verts3d[t[1]],p2=verts3d[t[2]];
        Vec2f u0=uvs[t[0]],u1=uvs[t[1]],u2=uvs[t[2]];
        auto J=triangle_jacobian(p0,p1,p2,u0,u1,u2);
        f32 A3=length(cross(p1-p0,p2-p0))*0.5f;
        f64 trace_JtJ=J[0]*J[0]+J[2]*J[2]+J[1]*J[1]+J[3]*J[3];
        E+=f64(A3)*trace_JtJ;
    }
    return E;
}

// Symmetric Dirichlet energy (prevents fold-overs): E = Σ(σ1²+σ2²+1/σ1²+1/σ2²) * A
[[nodiscard]] inline f64 symmetric_dirichlet_energy(
    const std::vector<Vec3f>& verts3d,
    const std::vector<Vec2f>& uvs,
    const std::vector<std::array<u32,3>>& tris)
{
    f64 E=0;
    for(const auto& t:tris) {
        Vec3f p0=verts3d[t[0]],p1=verts3d[t[1]],p2=verts3d[t[2]];
        Vec2f u0=uvs[t[0]],u1=uvs[t[1]],u2=uvs[t[2]];
        auto J=triangle_jacobian(p0,p1,p2,u0,u1,u2);
        auto [s1,s2]=svd2x2(J[0],J[1],J[2],J[3]);
        f32 A3=length(cross(p1-p0,p2-p0))*0.5f;
        if(s1<1e-10f||s2<1e-10f) E+=1e8; // degenerate
        else E+=f64(A3)*(s1*s1+s2*s2+1.0f/(s1*s1)+1.0f/(s2*s2));
    }
    return E;
}

// ─── UV repair: fix flipped triangles ────────────────────────────────────────
inline void fix_flipped_uvs(std::vector<Vec2f>& uvs,
    const std::vector<std::array<u32,3>>& tris,
    u32 max_iters=100)
{
    for(u32 it=0;it<max_iters;++it) {
        bool any_flip=false;
        for(const auto& t:tris) {
            Vec2f u0=uvs[t[0]],u1=uvs[t[1]],u2=uvs[t[2]];
            f32 signed_area=(u1.x-u0.x)*(u2.y-u0.y)-(u2.x-u0.x)*(u1.y-u0.y);
            if(signed_area<0) {
                // Average towards centroid
                Vec2f c=(u0+u1+u2)*(1.f/3);
                for(u32 v:t) uvs[v]=uvs[v]+(c-uvs[v])*0.1f;
                any_flip=true;
            }
        }
        if(!any_flip) break;
    }
}

// ─── Optimal UV scale (match world-space texel density) ──────────────────────
// Returns scale factor to apply to UV island so texels match world density (pixels/unit)
[[nodiscard]] inline f32 optimal_uv_scale(
    const std::vector<Vec3f>& verts3d,
    const std::vector<Vec2f>& uvs,
    const std::vector<std::array<u32,3>>& tris,
    f32 target_texel_density=1.f)
{
    f32 total_3d=0,total_uv=0;
    for(const auto& t:tris) {
        Vec3f p0=verts3d[t[0]],p1=verts3d[t[1]],p2=verts3d[t[2]];
        Vec2f u0=uvs[t[0]],u1=uvs[t[1]],u2=uvs[t[2]];
        total_3d+=length(cross(p1-p0,p2-p0))*0.5f;
        total_uv+=std::abs((u1.x-u0.x)*(u2.y-u0.y)-(u2.x-u0.x)*(u1.y-u0.y))*0.5f;
    }
    if(total_uv<1e-10f) return target_texel_density;
    return std::sqrt(total_3d/(total_uv+1e-10f))*target_texel_density;
}

} // namespace qs
