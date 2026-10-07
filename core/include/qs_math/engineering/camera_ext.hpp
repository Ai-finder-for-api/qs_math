#pragma once
// camera_ext.hpp — Full camera model: perspective/ortho/fisheye/panoramic,
//                  depth of field, motion blur, chromatic aberration helpers,
//                  vignette, lens distortion, multi-camera rigs
#include "camera_math.hpp"
#include "mat4.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "trig.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// Camera model (full lens)
// ---------------------------------------------------------------------------
struct LensModel {
    f32 focal_length{50.f};     // mm
    f32 aperture{1.8f};         // f-stop
    f32 focus_distance{5.f};    // world units
    f32 film_size_mm{36.f};     // horizontal sensor size in mm
    f32 sensor_width{1920.f};   // pixels
    f32 sensor_height{1080.f};

    [[nodiscard]] f32 fov_horizontal_rad() const noexcept {
        return 2.f*atan(film_size_mm/(2.f*focal_length));
    }
    [[nodiscard]] f32 fov_vertical_rad() const noexcept {
        f32 aspect=sensor_width/sensor_height;
        f32 film_v=film_size_mm/aspect;
        return 2.f*atan(film_v/(2.f*focal_length));
    }
    [[nodiscard]] f32 aspect_ratio() const noexcept { return sensor_width/sensor_height; }
    [[nodiscard]] f32 coc_diameter(f32 object_distance) const noexcept {
        // Circle of confusion diameter (mm)
        if(abs(object_distance-focus_distance)<EPS_F32) return 0.f;
        f32 fd=focus_distance, f=focal_length, ap=aperture;
        return abs(f*f*(object_distance-fd)/(ap*object_distance*(fd-f)));
    }
    [[nodiscard]] f32 depth_of_field_near() const noexcept {
        f32 NC=0.03f; // acceptable CoC in mm (35mm full frame)
        f32 H=focal_length*focal_length/(aperture*NC);
        return (H*focus_distance)/(H+focus_distance);
    }
    [[nodiscard]] f32 depth_of_field_far() const noexcept {
        f32 NC=0.03f;
        f32 H=focal_length*focal_length/(aperture*NC);
        f32 dof_far=(H*focus_distance)/(H-focus_distance);
        return dof_far<0.f?F32_INF:dof_far;
    }
};

// ---------------------------------------------------------------------------
// Lens distortion (Brown-Conrady model)
// ---------------------------------------------------------------------------
struct LensDistortion {
    f32 k1{0.f},k2{0.f},k3{0.f};  // radial
    f32 p1{0.f},p2{0.f};           // tangential

    [[nodiscard]] Vec2f distort(Vec2f normalized_pt) const noexcept {
        f32 r2=normalized_pt.length_sq();
        f32 r4=r2*r2, r6=r4*r2;
        f32 radial=1.f+k1*r2+k2*r4+k3*r6;
        Vec2f rad=normalized_pt*radial;
        Vec2f tang{2.f*p1*normalized_pt.x*normalized_pt.y+p2*(r2+2.f*normalized_pt.x*normalized_pt.x),
                   p1*(r2+2.f*normalized_pt.y*normalized_pt.y)+2.f*p2*normalized_pt.x*normalized_pt.y};
        return rad+tang;
    }

    [[nodiscard]] Vec2f undistort(Vec2f distorted, i32 iterations=10) const noexcept {
        Vec2f p=distorted;
        for(i32 i=0;i<iterations;++i){
            Vec2f d=distort(p)-distorted;
            p=p-d;
        }
        return p;
    }
};

// ---------------------------------------------------------------------------
// Chromatic aberration offset
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec2f chromatic_aberration_offset(
    Vec2f uv, f32 strength, i32 channel) noexcept
{
    // Red=0, Green=1, Blue=2; shift red outward, blue inward
    Vec2f center{0.5f};
    Vec2f dir=uv-center;
    f32 offsets[3]={strength, 0.f, -strength};
    return uv+dir*offsets[channel];
}

// ---------------------------------------------------------------------------
// Fisheye projection
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec2f fisheye_project(Vec3f dir, f32 fov_rad) noexcept {
    f32 theta=atan2(Vec2f{dir.x,dir.y}.length(),dir.z);
    f32 phi=atan2(dir.y,dir.x);
    f32 r=theta/(fov_rad*0.5f);
    return {r*cos(phi)*0.5f+0.5f, r*sin(phi)*0.5f+0.5f};
}

[[nodiscard]] inline Vec3f fisheye_unproject(Vec2f uv, f32 fov_rad) noexcept {
    Vec2f centered=(uv-Vec2f{0.5f})*2.f;
    f32 r=centered.length();
    if(r>1.f) return Vec3f{0.f};
    f32 theta=r*fov_rad*0.5f;
    f32 phi=atan2(centered.y,centered.x);
    return {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};
}

// ---------------------------------------------------------------------------
// Equirectangular (360°) mapping
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec2f equirectangular_project(Vec3f dir) noexcept {
    f32 phi=atan2(dir.z,dir.x);
    f32 theta=safe_acos(clamp(dir.y,-1.f,1.f));
    return {(phi/TWO_PI)+0.5f, theta/PI};
}

[[nodiscard]] inline Vec3f equirectangular_unproject(Vec2f uv) noexcept {
    f32 phi=(uv.x-0.5f)*TWO_PI;
    f32 theta=uv.y*PI;
    return {sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi)};
}

// ---------------------------------------------------------------------------
// Cubemap face + UV from direction
// ---------------------------------------------------------------------------
struct CubemapCoord { i32 face; Vec2f uv; };
[[nodiscard]] inline CubemapCoord cubemap_coord(Vec3f dir) noexcept {
    f32 ax=abs(dir.x),ay=abs(dir.y),az=abs(dir.z);
    i32 face; f32 u,v,ma;
    if(ax>=ay&&ax>=az){ face=dir.x>0?0:1; ma=ax; u=dir.x>0?-dir.z:dir.z; v=-dir.y; }
    else if(ay>=az)   { face=dir.y>0?2:3; ma=ay; u=dir.x; v=dir.y>0?dir.z:-dir.z; }
    else              { face=dir.z>0?4:5; ma=az; u=dir.z>0?dir.x:-dir.x; v=-dir.y; }
    return {face,{(u/ma+1.f)*0.5f,(v/ma+1.f)*0.5f}};
}

// ---------------------------------------------------------------------------
// Vignette
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 vignette(Vec2f uv, f32 strength, f32 radius) noexcept {
    Vec2f centered=(uv-Vec2f{0.5f})*2.f;
    f32 d=centered.length();
    return 1.f-smoothstep(radius-0.1f,radius+0.1f,d)*strength;
}

// ---------------------------------------------------------------------------
// Camera path interpolation (for cinematics)
// ---------------------------------------------------------------------------
struct CameraKeyframe {
    Vec3f  position;
    Quatf  orientation;
    f32    fov_y_rad;
    f32    time;
};

[[nodiscard]] inline CameraMatrices camera_keyframe_interp(
    const CameraKeyframe& a, const CameraKeyframe& b,
    f32 t, f32 aspect) noexcept
{
    Vec3f pos=lerp(a.position,b.position,t);
    Quatf rot=Quatf::slerp(a.orientation,b.orientation,t);
    f32 fov=lerp(a.fov_y_rad,b.fov_y_rad,t);
    Vec3f forward=rot.rotate(Vec3f{0,0,-1});
    Vec3f up=rot.rotate(Vec3f{0,1,0});
    return build_camera(pos,pos+forward,up,fov*RAD_TO_DEG,aspect,0.1f,1000.f);
}

// ---------------------------------------------------------------------------
// Multi-camera rig (stereo / 360° arrays)
// ---------------------------------------------------------------------------
struct StereoCameraRig {
    f32 ipd{0.063f}; // inter-pupillary distance in world units (default 63mm)

    [[nodiscard]] CameraMatrices left_camera(
        Vec3f eye, Vec3f target, Vec3f up,
        f32 fov, f32 aspect, f32 near_z, f32 far_z) const noexcept
    {
        Vec3f right=(target-eye).cross(up).normalized();
        return build_camera(eye-right*(ipd*0.5f),target,up,fov,aspect,near_z,far_z);
    }

    [[nodiscard]] CameraMatrices right_camera(
        Vec3f eye, Vec3f target, Vec3f up,
        f32 fov, f32 aspect, f32 near_z, f32 far_z) const noexcept
    {
        Vec3f right=(target-eye).cross(up).normalized();
        return build_camera(eye+right*(ipd*0.5f),target,up,fov,aspect,near_z,far_z);
    }

    // Off-axis stereo (more correct for VR)
    [[nodiscard]] Mat4f offaxis_projection(
        f32 left, f32 right, f32 bottom, f32 top,
        f32 near_z, f32 far_z) const noexcept
    {
        return Mat4f::orthographic(left,right,bottom,top,near_z,far_z);
    }
};

} // namespace qs::math
