#pragma once
// qs_math/viewport_math.hpp — Orbit camera, arcball, pan/zoom, screen-space picking, marquee (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include "geometry3d.hpp"
#include <cmath>
#include <numbers>
#include <optional>

namespace qs {

// ─── Orbit camera ────────────────────────────────────────────────────────────
struct OrbitCamera {
    Vec3f target{0,0,0};      // look-at point
    f32   distance{5.f};      // distance from target
    f32   yaw{0};             // azimuth [rad]
    f32   pitch{f32(std::numbers::pi)*0.25f}; // elevation [rad]
    f32   fov_y{f32(std::numbers::pi)/3};     // vertical FOV
    f32   near_plane{0.01f}, far_plane{1000.f};
    f32   aspect{1.77f};

    // Polar limits
    f32 pitch_min{0.01f}, pitch_max{f32(std::numbers::pi)-0.01f};
    f32 dist_min{0.1f},   dist_max{1000.f};

    [[nodiscard]] Vec3f position() const {
        return target + Vec3f{
            distance*std::sin(pitch)*std::sin(yaw),
            distance*std::cos(pitch),
            distance*std::sin(pitch)*std::cos(yaw)
        };
    }

    [[nodiscard]] Mat4f view_matrix() const {
        Vec3f eye=position();
        Vec3f up{0,1,0};
        Vec3f z=normalize(eye-target);
        Vec3f x=normalize(cross(up,z));
        Vec3f y=cross(z,x);
        return {
            x.x,y.x,z.x,0,
            x.y,y.y,z.y,0,
            x.z,y.z,z.z,0,
            -dot(x,eye),-dot(y,eye),-dot(z,eye),1
        };
    }

    [[nodiscard]] Mat4f proj_matrix() const {
        return Mat4f::perspective(fov_y,aspect,near_plane,far_plane);
    }

    void orbit(f32 delta_yaw, f32 delta_pitch) {
        yaw  +=delta_yaw;
        pitch =std::clamp(pitch+delta_pitch,pitch_min,pitch_max);
    }

    void pan(Vec2f screen_delta, Vec2f viewport_size) {
        f32 scale=distance*2*std::tan(fov_y*0.5f)/viewport_size.y;
        Vec3f eye=position();
        Vec3f z=normalize(eye-target);
        Vec3f x=normalize(cross(Vec3f{0,1,0},z));
        Vec3f y=cross(z,x);
        Vec3f delta=(x*(-screen_delta.x)+y*(screen_delta.y))*scale;
        target=target+delta;
    }

    void zoom(f32 delta) {
        distance=std::clamp(distance*std::exp(-delta*0.1f),dist_min,dist_max);
    }

    void zoom_at(Vec2f mouse_ndc, f32 delta, Vec2f viewport_size) {
        f32 factor=std::exp(-delta*0.1f);
        // Move target toward mouse direction before zooming
        Vec3f eye=position();
        Vec3f ray_dir=normalize(Vec3f{mouse_ndc.x*std::tan(fov_y*0.5f)*aspect,
                                      mouse_ndc.y*std::tan(fov_y*0.5f),
                                      -1.f});
        (void)viewport_size;
        // Transform ray_dir from view to world
        Mat4f invV=view_matrix().inversed();
        Vec4f rd4=invV*Vec4f{ray_dir.x,ray_dir.y,ray_dir.z,0};
        Vec3f world_ray=normalize(Vec3f{rd4.x,rd4.y,rd4.z});
        Vec3f step=world_ray*(distance*(1-factor));
        target=target+step;
        distance=std::clamp(distance*factor,dist_min,dist_max);
    }

    // Frame a bounding sphere
    void frame(Vec3f center, f32 radius) {
        target=center;
        distance=radius/std::sin(fov_y*0.5f)*1.5f;
    }
};

// ─── Arcball rotation (Shoemake 1992) ────────────────────────────────────────
struct ArcballCamera {
    Quatf rotation{};
    Vec3f center{};
    f32   radius{1.f};    // arcball sphere radius in screen units
    Vec2f viewport{};

    // Map screen point to unit sphere
    [[nodiscard]] Vec3f screen_to_sphere(Vec2f p) const {
        Vec2f v{(p.x-viewport.x*0.5f)/radius, (viewport.y*0.5f-p.y)/radius};
        f32 r2=v.x*v.x+v.y*v.y;
        if(r2<=1.f) return normalize(Vec3f{v.x,v.y,std::sqrt(1-r2)});
        return normalize(Vec3f{v.x,v.y,0}); // outside sphere: project onto edge
    }

    // Drag from p0 to p1 → apply rotation
    void drag(Vec2f p0, Vec2f p1) {
        Vec3f v0=screen_to_sphere(p0), v1=screen_to_sphere(p1);
        Vec3f axis=cross(v0,v1);
        f32 angle=std::acos(std::clamp(dot(v0,v1),-1.f,1.f));
        if(length(axis)>1e-7f)
            rotation=normalize(Quatf::from_axis_angle(normalize(axis),angle)*rotation);
    }

    [[nodiscard]] Mat4f view_matrix(Vec3f eye_offset={0,0,5}) const {
        Vec3f eye=center+rotation.rotate(eye_offset);
        Vec3f up=rotation.rotate(Vec3f{0,1,0});
        Vec3f z=normalize(eye-center);
        Vec3f x=normalize(cross(up,z));
        Vec3f y=cross(z,x);
        return {x.x,y.x,z.x,0,x.y,y.y,z.y,0,x.z,y.z,z.z,0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1};
    }
};

// ─── Fly camera (FPS-style) ───────────────────────────────────────────────────
struct FlyCamera {
    Vec3f pos{0,0,5};
    f32   yaw{0}, pitch{0};
    f32   fov_y{f32(std::numbers::pi)/3};
    f32   near_p{0.01f}, far_p{1000.f}, aspect{1.77f};

    [[nodiscard]] Vec3f forward() const { return {std::sin(yaw)*std::cos(pitch),-std::sin(pitch),std::cos(yaw)*std::cos(pitch)}; }
    [[nodiscard]] Vec3f right()   const { return normalize(cross(forward(),{0,1,0})); }
    [[nodiscard]] Vec3f up()      const { return cross(right(),forward()); }

    void move(Vec3f delta_local) { pos=pos+right()*delta_local.x+Vec3f{0,1,0}*delta_local.y+forward()*(-delta_local.z); }
    void look(f32 dy, f32 dp) { yaw+=dy; pitch=std::clamp(pitch+dp,-f32(std::numbers::pi)*0.499f,f32(std::numbers::pi)*0.499f); }

    [[nodiscard]] Mat4f view_matrix() const {
        Vec3f f=forward(),r=right(),u=up();
        return {r.x,u.x,-f.x,0,r.y,u.y,-f.y,0,r.z,u.z,-f.z,0,-dot(r,pos),-dot(u,pos),dot(f,pos),1};
    }
    [[nodiscard]] Mat4f proj_matrix() const { return Mat4f::perspective(fov_y,aspect,near_p,far_p); }
};

// ─── Screen-space operations ──────────────────────────────────────────────────

// Unproject screen point to world ray
[[nodiscard]] inline Ray3f screen_to_ray(Vec2f screen, Vec2f viewport,
    const Mat4f& inv_view, const Mat4f& inv_proj)
{
    f32 nx=(screen.x/viewport.x)*2-1;
    f32 ny=1-(screen.y/viewport.y)*2;
    Vec4f clip_near{nx,ny,-1,1}, clip_far{nx,ny,1,1};
    auto unproj=[&](Vec4f c)->Vec3f{
        Vec4f vs=inv_proj*c; vs=vs*(1/vs.w);
        Vec4f ws=inv_view*vs; return {ws.x/ws.w,ws.y/ws.w,ws.z/ws.w};
    };
    Vec3f pn=unproj(clip_near), pf=unproj(clip_far);
    return {pn,normalize(pf-pn)};
}

// Project world point to screen
[[nodiscard]] inline std::optional<Vec2f> world_to_screen(Vec3f p, const Mat4f& vp, Vec2f viewport) {
    Vec4f clip=vp*Vec4f{p.x,p.y,p.z,1};
    if(std::abs(clip.w)<1e-7f) return std::nullopt;
    Vec3f ndc{clip.x/clip.w,clip.y/clip.w,clip.z/clip.w};
    if(ndc.z<-1||ndc.z>1) return std::nullopt;
    return Vec2f{(ndc.x*0.5f+0.5f)*viewport.x,(1-(ndc.y*0.5f+0.5f))*viewport.y};
}

// ─── Marquee (rectangle) selection ───────────────────────────────────────────
struct Marquee {
    Vec2f start{}, end{};
    bool  active{false};

    void begin(Vec2f p) { start=end=p; active=true; }
    void update(Vec2f p){ end=p; }
    void finish()       { active=false; }

    [[nodiscard]] Vec2f min_pt() const { return {std::min(start.x,end.x),std::min(start.y,end.y)}; }
    [[nodiscard]] Vec2f max_pt() const { return {std::max(start.x,end.x),std::max(start.y,end.y)}; }
    [[nodiscard]] bool  contains(Vec2f p) const { auto mn=min_pt(),mx=max_pt(); return p.x>=mn.x&&p.x<=mx.x&&p.y>=mn.y&&p.y<=mx.y; }
    [[nodiscard]] Vec2f size()    const { return max_pt()-min_pt(); }
    [[nodiscard]] bool  is_empty()const { return size().x<2&&size().y<2; }

    // Frustum from marquee for volume selection
    [[nodiscard]] std::array<Vec4f,6> selection_frustum(
        const Mat4f& inv_view, const Mat4f& inv_proj, Vec2f viewport) const
    {
        // Build 4 edge planes of marquee selection frustum
        auto mn=min_pt(),mx=max_pt();
        std::array<Vec2f,4> corners={mn,{mx.x,mn.y},mx,{mn.x,mx.y}};
        std::array<Vec3f,4> world_corners;
        for(u32 i=0;i<4;++i){
            f32 nx=(corners[i].x/viewport.x)*2-1, ny=1-(corners[i].y/viewport.y)*2;
            Vec4f clip{nx,ny,0,1}; Vec4f vs=inv_proj*clip; vs=vs*(1/vs.w);
            Vec4f ws=inv_view*vs; world_corners[i]={ws.x/ws.w,ws.y/ws.w,ws.z/ws.w};
        }
        // Camera position
        Vec3f eye{inv_view.m[0][3],inv_view.m[1][3],inv_view.m[2][3]};
        // Build planes from edge frustum (left,right,top,bottom,near,far)
        std::array<Vec4f,6> planes;
        for(u32 i=0;i<4;++i) {
            Vec3f e1=world_corners[i]-eye, e2=world_corners[(i+1)%4]-eye;
            Vec3f n=normalize(cross(e1,e2));
            planes[i]={n.x,n.y,n.z,-dot(n,eye)};
        }
        planes[4]={0,0,-1, 0}; // near/far simplified
        planes[5]={0,0, 1,-1000};
        return planes;
    }
};

// ─── Depth linearization ──────────────────────────────────────────────────────
[[nodiscard]] inline f32 linearize_depth(f32 depth_ndc, f32 near_p, f32 far_p) {
    return 2*near_p*far_p/(far_p+near_p-(2*depth_ndc-1)*(far_p-near_p));
}
[[nodiscard]] inline f32 linear_depth_to_ndc(f32 linear_d, f32 near_p, f32 far_p) {
    return (far_p+near_p-2*near_p*far_p/linear_d)/(far_p-near_p)*0.5f+0.5f;
}

// ─── Camera animation (smooth follow) ────────────────────────────────────────
// Spring-damper smoothing for camera position/target
[[nodiscard]] inline Vec3f spring_follow(Vec3f current, Vec3f target_pos, Vec3f velocity,
    f32 stiffness, f32 damping, f32 dt, Vec3f& velocity_out)
{
    Vec3f spring_force=(target_pos-current)*stiffness;
    Vec3f damp_force=velocity*(-damping);
    Vec3f acc=spring_force+damp_force;
    velocity_out=velocity+acc*dt;
    return current+velocity_out*dt;
}

} // namespace qs
