#pragma once
// qs_math/gizmo_math.hpp — Translate/rotate/scale gizmo hit testing, axis projection, snapping (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "mat4.hpp"
#include "ray_tracing_math.hpp"
#include "geometry3d.hpp"
#include "intersection.hpp"
#include <optional>
#include <cmath>
#include <array>

namespace qs {

// ─── Gizmo axis IDs ──────────────────────────────────────────────────────────
enum class GizmoAxis : u8 { None, X, Y, Z, XY, XZ, YZ, XYZ, RotX, RotY, RotZ, ScaleX, ScaleY, ScaleZ, ScaleUniform };
enum class GizmoMode : u8 { Translate, Rotate, Scale };

// ─── Screen-space axis line projection ───────────────────────────────────────
// Project a 3D axis from origin onto screen, return 2D direction
[[nodiscard]] inline Vec2f project_axis_to_screen(Vec3f origin, Vec3f axis_dir,
    const Mat4f& view_proj, Vec2f viewport_size)
{
    // Project origin and tip
    auto proj=[&](Vec3f p)->Vec2f {
        Vec4f clip=view_proj*Vec4f{p.x,p.y,p.z,1};
        if(std::abs(clip.w)<1e-7f) return {0,0};
        return {(clip.x/clip.w*0.5f+0.5f)*viewport_size.x,
                (1-(clip.y/clip.w*0.5f+0.5f))*viewport_size.y};
    };
    Vec2f p0=proj(origin);
    Vec2f p1=proj(origin+axis_dir*0.1f);
    Vec2f d=p1-p0;
    f32 len=std::sqrt(d.x*d.x+d.y*d.y);
    return len>1e-5f?Vec2f{d.x/len,d.y/len}:Vec2f{1,0};
}

// ─── Translate gizmo hit testing ─────────────────────────────────────────────
// Hit test a translate gizmo axis arrow at 'origin' along 'axis'
// Returns hit distance or empty if miss
[[nodiscard]] inline std::optional<f32> hit_translate_axis(
    Vec3f ray_origin, Vec3f ray_dir,
    Vec3f gizmo_origin, Vec3f axis,
    f32 arrow_length=1.f, f32 arrow_radius=0.04f)
{
    // Cylinder along axis from origin to origin+axis*arrow_length
    Vec3f tip=gizmo_origin+axis*arrow_length;
    // Closest point between ray and cylinder axis segment
    Vec3f u=ray_dir, v=axis;
    Vec3f w=ray_origin-gizmo_origin;
    f32 a=dot(u,u), b=dot(u,v), c=dot(v,v);
    f32 d=dot(u,w), e=dot(v,w);
    f32 denom=a*c-b*b;
    f32 sc=std::abs(denom)>1e-9f?(b*e-c*d)/denom:0;
    f32 tc=(b*sc+e)/std::max(1e-9f,c);
    if(tc<0||tc>arrow_length) return std::nullopt;
    Vec3f closest_on_ray=ray_origin+ray_dir*sc;
    Vec3f closest_on_axis=gizmo_origin+axis*tc;
    f32 gap_sq=dist_sq(closest_on_ray,closest_on_axis);
    if(gap_sq>arrow_radius*arrow_radius) return std::nullopt;
    return sc>0?std::optional<f32>(sc):std::nullopt;
}

// Hit test translate plane handle (small square in XY, XZ, YZ plane)
[[nodiscard]] inline std::optional<f32> hit_translate_plane(
    Vec3f ray_origin, Vec3f ray_dir,
    Vec3f gizmo_origin, Vec3f n,    // plane normal
    Vec3f u, Vec3f v,               // plane axes
    f32 handle_size=0.2f)
{
    Plane<f32> plane{n, dot(n,gizmo_origin+u*handle_size*0.5f+v*handle_size*0.5f)};
    auto hit=ray_plane_intersect(Ray3<f32>{ray_origin,ray_dir},plane);
    if(!hit) return std::nullopt;
    Vec3f p=ray_origin+ray_dir*(*hit)-gizmo_origin;
    f32 pu=dot(p,u), pv=dot(p,v);
    if(pu<0||pu>handle_size||pv<0||pv>handle_size) return std::nullopt;
    return *hit>0?*hit:std::nullopt;
}

// ─── Rotate gizmo hit testing ─────────────────────────────────────────────────
// Hit test a rotation ring (torus approximation: circle of radius R, tube radius r)
[[nodiscard]] inline std::optional<f32> hit_rotate_ring(
    Vec3f ray_origin, Vec3f ray_dir,
    Vec3f gizmo_origin, Vec3f ring_normal,
    f32 ring_radius=1.f, f32 tube_radius=0.04f)
{
    // Intersect ray with ring plane first
    Plane<f32> plane{ring_normal, dot(ring_normal,gizmo_origin)};
    auto t_hit=ray_plane_intersect(Ray3<f32>{ray_origin,ray_dir},plane);
    if(!t_hit||*t_hit<=0) return std::nullopt;
    Vec3f p=ray_origin+ray_dir*(*t_hit);
    // Check if p is near the ring circle
    Vec3f dp=p-gizmo_origin;
    Vec3f dp_flat=dp-ring_normal*dot(dp,ring_normal);
    f32 dist_to_center=length(dp_flat);
    if(std::abs(dist_to_center-ring_radius)<tube_radius*2)
        return *t_hit;
    return std::nullopt;
}

// ─── Gizmo full hit test ──────────────────────────────────────────────────────
struct GizmoHit {
    GizmoAxis axis{GizmoAxis::None};
    f32 t{0};
};

[[nodiscard]] inline GizmoHit hit_translate_gizmo(
    Vec3f ray_origin, Vec3f ray_dir,
    Vec3f origin, f32 scale=1.f)
{
    GizmoHit best{GizmoAxis::None, 1e30f};
    struct Ax { Vec3f dir; GizmoAxis id; Vec3f color; };
    Ax axes[3]={{{1,0,0},GizmoAxis::X,{}},{{0,1,0},GizmoAxis::Y,{}},{{0,0,1},GizmoAxis::Z,{}}};
    for(auto& ax:axes) {
        auto h=hit_translate_axis(ray_origin,ray_dir,origin,ax.dir,scale,scale*0.04f);
        if(h&&*h<best.t){ best={ax.id,*h}; }
    }
    // Plane handles
    struct Pl { Vec3f n,u,v; GizmoAxis id; };
    Pl planes[3]={
        {{0,0,1},{1,0,0},{0,1,0},GizmoAxis::XY},
        {{0,1,0},{1,0,0},{0,0,1},GizmoAxis::XZ},
        {{1,0,0},{0,1,0},{0,0,1},GizmoAxis::YZ}
    };
    for(auto& pl:planes) {
        auto h=hit_translate_plane(ray_origin,ray_dir,origin,pl.n,pl.u,pl.v,scale*0.25f);
        if(h&&*h<best.t){ best={pl.id,*h}; }
    }
    return best;
}

[[nodiscard]] inline GizmoHit hit_rotate_gizmo(
    Vec3f ray_origin, Vec3f ray_dir,
    Vec3f origin, f32 scale=1.f)
{
    GizmoHit best{GizmoAxis::None, 1e30f};
    struct Rg { Vec3f n; GizmoAxis id; };
    Rg rings[3]={{{1,0,0},GizmoAxis::RotX},{{0,1,0},GizmoAxis::RotY},{{0,0,1},GizmoAxis::RotZ}};
    for(auto& rg:rings) {
        auto h=hit_rotate_ring(ray_origin,ray_dir,origin,rg.n,scale,scale*0.04f);
        if(h&&*h<best.t){ best={rg.id,*h}; }
    }
    return best;
}

// ─── Gizmo delta computation ─────────────────────────────────────────────────
// Given mouse delta and active axis, compute world-space translation delta
[[nodiscard]] inline Vec3f translate_delta_from_mouse(
    GizmoAxis axis, Vec2f mouse_delta,
    const Mat4f& view, const Mat4f& proj, Vec2f viewport_size,
    Vec3f gizmo_origin, f32 sensitivity=1.f)
{
    // Unproject mouse delta into axis-constrained world movement
    // Approximate: project axis onto screen, use screen-space delta along it
    Mat4f vp=proj*view;
    Vec3f world_axis={};
    switch(axis) {
        case GizmoAxis::X: world_axis={1,0,0}; break;
        case GizmoAxis::Y: world_axis={0,1,0}; break;
        case GizmoAxis::Z: world_axis={0,0,1}; break;
        default: return {};
    }
    Vec2f screen_axis=project_axis_to_screen(gizmo_origin,world_axis,vp,viewport_size);
    f32 proj_dist=mouse_delta.x*screen_axis.x+mouse_delta.y*screen_axis.y;
    return world_axis*(proj_dist/viewport_size.x*sensitivity*10.f);
}

// Rotation delta from mouse drag on ring
[[nodiscard]] inline f32 rotate_delta_from_mouse(
    GizmoAxis axis, Vec2f mouse_delta,
    Vec2f viewport_size, f32 sensitivity=1.f)
{
    // Screen-space delta magnitude → rotation in radians
    f32 screen_len=std::sqrt(mouse_delta.x*mouse_delta.x+mouse_delta.y*mouse_delta.y);
    return screen_len/viewport_size.x*f32(std::numbers::pi)*2*sensitivity;
}

// ─── Snapping math ────────────────────────────────────────────────────────────
[[nodiscard]] inline f32 snap_to_grid(f32 v, f32 grid_size) {
    return std::round(v/grid_size)*grid_size;
}
[[nodiscard]] inline Vec3f snap_position(Vec3f p, f32 grid=0.25f) {
    return {snap_to_grid(p.x,grid),snap_to_grid(p.y,grid),snap_to_grid(p.z,grid)};
}
[[nodiscard]] inline f32 snap_angle(f32 rad, f32 step_deg=15.f) {
    f32 step=step_deg*f32(std::numbers::pi)/180.f;
    return std::round(rad/step)*step;
}
[[nodiscard]] inline f32 snap_scale(f32 s, f32 step=0.25f) {
    return std::max(step,snap_to_grid(s,step));
}

// ─── Viewport picking ray ─────────────────────────────────────────────────────
[[nodiscard]] inline Ray3f viewport_picking_ray(
    Vec2f mouse_pos, Vec2f viewport_size,
    const Mat4f& inv_view, const Mat4f& inv_proj)
{
    // NDC coordinates
    f32 nx=(mouse_pos.x/viewport_size.x)*2-1;
    f32 ny=1-(mouse_pos.y/viewport_size.y)*2;
    // Unproject near and far plane
    Vec4f clip_near{nx,ny,-1,1}, clip_far{nx,ny,1,1};
    auto unproject=[&](Vec4f clip)->Vec3f {
        Vec4f view_space=inv_proj*clip;
        view_space=view_space*(1.f/view_space.w);
        Vec4f world=inv_view*view_space;
        return {world.x/world.w,world.y/world.w,world.z/world.w};
    };
    Vec3f p_near=unproject(clip_near);
    Vec3f p_far =unproject(clip_far);
    return {p_near, normalize(p_far-p_near)};
}

// ─── Local-space gizmo transform accumulation ─────────────────────────────────
struct GizmoTransform {
    Vec3f position{};
    Quatf rotation{};
    Vec3f scale{1,1,1};

    [[nodiscard]] Mat4f to_mat4() const {
        Mat4f T=Mat4f::translation(position);
        Mat4f R=Mat4f::from_quat(rotation);
        Mat4f S=Mat4f::scaling(scale);
        return T*R*S;
    }

    void translate(Vec3f delta) { position=position+delta; }
    void rotate_axis(Vec3f axis, f32 angle) { rotation=normalize(Quatf::from_axis_angle(axis,angle)*rotation); }
    void scale_uniform(f32 s) { scale=scale*s; }
};

} // namespace qs
