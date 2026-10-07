#pragma once
// scalar_field.hpp — Scalar field mathematics: isosurfaces, marching cubes tables,
//                    level sets, signed distance fields, eikonal equation,
//                    potential energy surfaces, field operations
#include "vec3.hpp"
#include "vec2.hpp"
#include "scalar_math.hpp"
#include "geometry3d.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Discrete scalar field
// ---------------------------------------------------------------------------
struct ScalarField3D {
    f32*  data{nullptr};
    i32   w{0},h{0},d{0};
    f32   dx{1.f},dy{1.f},dz{1.f};
    Vec3f origin{0.f};

    [[nodiscard]] f32 sample(Vec3f p) const noexcept {
        if(!data) return 0.f;
        f32 fx=(p.x-origin.x)/dx, fy=(p.y-origin.y)/dy, fz=(p.z-origin.z)/dz;
        i32 ix=clamp(i32(fx),0,w-2), iy=clamp(i32(fy),0,h-2), iz=clamp(i32(fz),0,d-2);
        f32 bx=fx-ix, by=fy-iy, bz=fz-iz;
        auto g=[&](i32 x,i32 y,i32 z)->f32{
            return data[clamp(z,0,d-1)*h*w+clamp(y,0,h-1)*w+clamp(x,0,w-1)];
        };
        return trilinear(g(ix,iy,iz),g(ix+1,iy,iz),g(ix,iy+1,iz),g(ix+1,iy+1,iz),
                         g(ix,iy,iz+1),g(ix+1,iy,iz+1),g(ix,iy+1,iz+1),g(ix+1,iy+1,iz+1),bx,by,bz);
    }

    [[nodiscard]] Vec3f gradient_at(i32 x, i32 y, i32 z) const noexcept {
        auto g=[&](i32 xi,i32 yi,i32 zi)->f32{
            return data[clamp(zi,0,d-1)*h*w+clamp(yi,0,h-1)*w+clamp(xi,0,w-1)];
        };
        return {(g(x+1,y,z)-g(x-1,y,z))/(2.f*dx),
                (g(x,y+1,z)-g(x,y-1,z))/(2.f*dy),
                (g(x,y,z+1)-g(x,y,z-1))/(2.f*dz)};
    }
};

// ---------------------------------------------------------------------------
// Marching cubes lookup tables
// ---------------------------------------------------------------------------
// Edge table (which edges are crossed for each case)
extern const i32 MC_EDGE_TABLE[256];
// Triangle table (which triangles to generate for each case)
extern const i32 MC_TRI_TABLE[256][16];

// Interpolate edge point between v0 and v1
[[nodiscard]] constexpr Vec3f mc_interp(Vec3f p0, Vec3f p1, f32 v0, f32 v1, f32 iso) noexcept {
    if(abs(iso-v0)<EPS_F32) return p0;
    if(abs(iso-v1)<EPS_F32) return p1;
    if(abs(v0-v1)<EPS_F32) return p0;
    f32 t=(iso-v0)/(v1-v0);
    return lerp(p0,p1,t);
}

// Compute marching cubes case index for a voxel
[[nodiscard]] constexpr i32 mc_case(f32 v[8], f32 iso) noexcept {
    i32 idx=0;
    for(i32 i=0;i<8;++i) if(v[i]<iso) idx|=(1<<i);
    return idx;
}

// Voxel corner offsets
inline constexpr Vec3f MC_CORNERS[8]={
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};

// ---------------------------------------------------------------------------
// Level set operations
// ---------------------------------------------------------------------------
// Signed distance field to sphere
[[nodiscard]] constexpr f32 sdf_sphere_ls(Vec3f p, Vec3f c, f32 r) noexcept { return (p-c).length()-r; }

// Reinitialize SDF (fast marching hint — simplified Sussman method)
inline void sdf_reinitialize_step(f32* phi, f32* phi0, i32 n, f32 dt=0.5f) noexcept {
    for(i32 i=0;i<n;++i){
        f32 s=phi0[i]/(sqrt(phi0[i]*phi0[i]+EPS_F32*EPS_F32));
        phi[i]-=dt*s*(abs(phi[i])-1.f);
    }
}

// SDF boolean operations
[[nodiscard]] constexpr f32 sdf_union(f32 a, f32 b) noexcept { return min(a,b); }
[[nodiscard]] constexpr f32 sdf_subtract(f32 a, f32 b) noexcept { return max(a,-b); }
[[nodiscard]] constexpr f32 sdf_intersect(f32 a, f32 b) noexcept { return max(a,b); }
[[nodiscard]] constexpr f32 sdf_offset(f32 sdf, f32 r) noexcept { return sdf-r; }
[[nodiscard]] constexpr f32 sdf_shell(f32 sdf, f32 thickness) noexcept { return abs(sdf)-thickness; }
[[nodiscard]] constexpr f32 sdf_scale(f32 sdf, f32 s) noexcept { return sdf*s; }

// Smooth min/max (for organic blending)
[[nodiscard]] constexpr f32 sdf_smin(f32 a, f32 b, f32 k) noexcept {
    f32 h=max(k-abs(a-b),0.f)/k;
    return min(a,b)-h*h*k*0.25f;
}
[[nodiscard]] constexpr f32 sdf_smax(f32 a, f32 b, f32 k) noexcept { return -sdf_smin(-a,-b,k); }

// Domain deformation
[[nodiscard]] inline Vec3f sdf_twist(Vec3f p, f32 k) noexcept {
    f32 c=cos(k*p.y), s=sin(k*p.y);
    return {c*p.x-s*p.z, p.y, s*p.x+c*p.z};
}
[[nodiscard]] inline Vec3f sdf_bend(Vec3f p, f32 k) noexcept {
    f32 c=cos(k*p.x), s=sin(k*p.x);
    return {c*p.x-s*p.y, s*p.x+c*p.y, p.z};
}
[[nodiscard]] constexpr Vec3f sdf_repeat(Vec3f p, Vec3f cell) noexcept {
    return {p.x-cell.x*round(p.x/cell.x),
            p.y-cell.y*round(p.y/cell.y),
            p.z-cell.z*round(p.z/cell.z)};
}
[[nodiscard]] constexpr Vec3f sdf_repeat_limited(Vec3f p, Vec3f cell, Vec3f lim) noexcept {
    return p-cell*p.clamp(-lim,lim)/cell;
}

// ---------------------------------------------------------------------------
// Eikonal equation (fast marching for distance computation)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 eikonal_update_2d(
    f32 Ux, f32 Uy, f32 f=1.f) noexcept
{
    // Update rule: max(u-Ux,0)^2 + max(u-Uy,0)^2 = f^2
    f32 u_min=min(Ux,Uy);
    f32 u=u_min+f;
    if(u>max(Ux,Uy)){
        // Quadratic: 2u^2 - 2(Ux+Uy)u + Ux^2+Uy^2-f^2 = 0
        f32 b=Ux+Uy, disc=2.f*f*f-(Ux-Uy)*(Ux-Uy);
        if(disc>=0.f) u=(b+sqrt(disc))*0.5f;
    }
    return u;
}

// ---------------------------------------------------------------------------
// Transfer functions (volume rendering)
// ---------------------------------------------------------------------------
struct TransferFn {
    Vec4f color{1.f};       // RGBA
    f32   iso_value{0.f};
    f32   width{0.1f};

    [[nodiscard]] constexpr Vec4f evaluate(f32 density) const noexcept {
        f32 alpha=exp(-0.5f*((density-iso_value)/width)*((density-iso_value)/width));
        return color*alpha;
    }
};

[[nodiscard]] inline Vec4f transfer_fn_composite(
    const TransferFn* fns, i32 n, f32 density) noexcept
{
    Vec4f result{0.f};
    for(i32 i=0;i<n;++i){
        Vec4f c=fns[i].evaluate(density);
        // Alpha compositing (over)
        result=result+Vec4f{c.x*c.w,c.y*c.w,c.z*c.w,c.w}*(1.f-result.w);
    }
    return result;
}

} // namespace qs::math
