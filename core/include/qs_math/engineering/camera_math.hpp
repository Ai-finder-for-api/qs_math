#pragma once
// qs_math/camera_math.hpp — Projection, frustum, arcball, fly cam, cinematic shake (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include "ray_tracing_math.hpp"
#include <cmath>
#include <algorithm>

namespace qs {

// ─── View matrix ─────────────────────────────────────────────────────────────

[[nodiscard]] inline Mat4f look_at(
    const Vec3f& eye, const Vec3f& center, const Vec3f& up)
{
    Vec3f f = normalize(center - eye);
    Vec3f s = normalize(cross(f, up));
    Vec3f u = cross(s, f);
    Mat4f m = Mat4f::identity();
    m.m[0][0]=s.x; m.m[0][1]=s.y; m.m[0][2]=s.z;
    m.m[1][0]=u.x; m.m[1][1]=u.y; m.m[1][2]=u.z;
    m.m[2][0]=-f.x;m.m[2][1]=-f.y;m.m[2][2]=-f.z;
    m.m[0][3]=-dot(s,eye); m.m[1][3]=-dot(u,eye); m.m[2][3]=dot(f,eye);
    return m;
}

// ─── Projection matrices ──────────────────────────────────────────────────────

[[nodiscard]] inline Mat4f perspective_rh(f32 fov_y, f32 aspect, f32 near, f32 far) {
    f32 f = 1.f / std::tan(fov_y*0.5f);
    Mat4f m{};
    m.m[0][0]=f/aspect; m.m[1][1]=f;
    m.m[2][2]=-(far+near)/(far-near);
    m.m[2][3]=-2.f*far*near/(far-near);
    m.m[3][2]=-1.f;
    return m;
}

[[nodiscard]] inline Mat4f perspective_infinite_rh(f32 fov_y, f32 aspect, f32 near) {
    f32 f = 1.f / std::tan(fov_y*0.5f);
    Mat4f m{};
    m.m[0][0]=f/aspect; m.m[1][1]=f;
    m.m[2][2]=-1.f; m.m[2][3]=-2.f*near;
    m.m[3][2]=-1.f;
    return m;
}

// Reversed-Z for better precision
[[nodiscard]] inline Mat4f perspective_reversed_z(f32 fov_y, f32 aspect, f32 near, f32 far) {
    f32 f = 1.f / std::tan(fov_y*0.5f);
    Mat4f m{};
    m.m[0][0]=f/aspect; m.m[1][1]=f;
    m.m[2][2]=near/(far-near);
    m.m[2][3]=near*far/(far-near);
    m.m[3][2]=-1.f;
    return m;
}

[[nodiscard]] inline Mat4f orthographic(
    f32 l, f32 r, f32 b, f32 t, f32 near, f32 far)
{
    Mat4f m{};
    m.m[0][0]=2.f/(r-l); m.m[1][1]=2.f/(t-b); m.m[2][2]=-2.f/(far-near);
    m.m[0][3]=-(r+l)/(r-l); m.m[1][3]=-(t+b)/(t-b); m.m[2][3]=-(far+near)/(far-near);
    m.m[3][3]=1.f;
    return m;
}

// ─── Frustum ─────────────────────────────────────────────────────────────────

struct Frustum {
    Vec4f planes[6]; // left,right,bottom,top,near,far (normal,d: ax+by+cz+d>0 = inside)

    static Frustum from_view_proj(const Mat4f& VP) {
        Frustum f;
        // Rows
        auto r0 = Vec4f{VP.m[0][0],VP.m[0][1],VP.m[0][2],VP.m[0][3]};
        auto r1 = Vec4f{VP.m[1][0],VP.m[1][1],VP.m[1][2],VP.m[1][3]};
        auto r2 = Vec4f{VP.m[2][0],VP.m[2][1],VP.m[2][2],VP.m[2][3]};
        auto r3 = Vec4f{VP.m[3][0],VP.m[3][1],VP.m[3][2],VP.m[3][3]};
        auto norm = [](Vec4f p){
            f32 l=std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
            return l>0.f?Vec4f{p.x/l,p.y/l,p.z/l,p.w/l}:p;};
        f->planes[0]=norm(r3+r0); // left
        f->planes[1]=norm(r3-r0); // right
        f->planes[2]=norm(r3+r1); // bottom
        f->planes[3]=norm(r3-r1); // top
        f->planes[4]=norm(r3+r2); // near
        f->planes[5]=norm(r3-r2); // far
        return f;
    }

    [[nodiscard]] bool test_sphere(const Vec3f& c, f32 r) const {
        for (const auto& p : planes)
            if (p.x*c.x+p.y*c.y+p.z*c.z+p.w < -r) return false;
        return true;
    }

    [[nodiscard]] bool test_aabb(const Vec3f& mn, const Vec3f& mx) const {
        for (const auto& p : planes) {
            Vec3f positive{p.x>0?mx.x:mn.x, p.y>0?mx.y:mn.y, p.z>0?mx.z:mn.z};
            if (p.x*positive.x+p.y*positive.y+p.z*positive.z+p.w < 0.f) return false;
        }
        return true;
    }
};

// ─── Arcball camera ───────────────────────────────────────────────────────────

struct ArcballCamera {
    Vec3f target{0,0,0};
    f32   distance{5.f};
    f32   yaw{0.f}, pitch{0.f};
    f32   min_pitch{-k_pi_over_two+0.01f};
    f32   max_pitch{ k_pi_over_two-0.01f};
    f32   min_dist{0.1f}, max_dist{100.f};

    void orbit(f32 dyaw, f32 dpitch) {
        yaw   += dyaw;
        pitch = std::clamp(pitch+dpitch, min_pitch, max_pitch);
    }

    void zoom(f32 delta) {
        distance = std::clamp(distance*std::exp(-delta*0.1f), min_dist, max_dist);
    }

    void pan(f32 dx, f32 dy) {
        Vec3f right{std::cos(yaw), 0, std::sin(yaw)};
        Vec3f up{0, 1, 0};
        target = target + right*(-dx*distance*0.001f) + up*(dy*distance*0.001f);
    }

    [[nodiscard]] Vec3f eye() const {
        return target + Vec3f{
            std::cos(pitch)*std::sin(yaw),
            std::sin(pitch),
            std::cos(pitch)*std::cos(yaw)} * distance;
    }

    [[nodiscard]] Mat4f view_matrix() const {
        return look_at(eye(), target, {0,1,0});
    }
};

// ─── Fly camera ───────────────────────────────────────────────────────────────

struct FlyCamera {
    Vec3f pos{0,0,0};
    f32   yaw{0.f}, pitch{0.f};
    f32   move_speed{5.f};
    f32   mouse_sensitivity{0.002f};

    void look(f32 dyaw, f32 dpitch) {
        yaw   += dyaw;
        pitch = std::clamp(pitch+dpitch, -k_pi_over_two+0.01f, k_pi_over_two-0.01f);
    }

    [[nodiscard]] Vec3f forward() const {
        return {std::cos(pitch)*std::sin(yaw), std::sin(pitch), std::cos(pitch)*std::cos(yaw)};
    }
    [[nodiscard]] Vec3f right() const {
        return normalize(cross(forward(), {0,1,0}));
    }

    void move(f32 fw, f32 rt, f32 up, f32 dt) {
        pos = pos + forward()*(fw*move_speed*dt)
                  + right()*(rt*move_speed*dt)
                  + Vec3f{0,1,0}*(up*move_speed*dt);
    }

    [[nodiscard]] Mat4f view_matrix() const {
        return look_at(pos, pos+forward(), {0,1,0});
    }
};

// ─── Camera shake ────────────────────────────────────────────────────────────

struct CameraShake {
    f32 trauma{0.f};    // 0..1; squared for shake magnitude
    f32 decay{1.0f};    // trauma per second decay
    f32 magnitude{0.05f};
    f32 frequency{10.f};
    f32 t{0.f};

    void add_trauma(f32 amount) { trauma = std::min(1.f, trauma+amount); }

    void update(f32 dt) { trauma = std::max(0.f, trauma-decay*dt); t+=dt; }

    [[nodiscard]] Vec3f offset() const {
        f32 shake = trauma*trauma;
        return Vec3f{
            magnitude*shake*std::sin(t*frequency*2.17f),
            magnitude*shake*std::sin(t*frequency*3.13f+1.57f),
            magnitude*shake*std::sin(t*frequency*1.79f+0.78f)};
    }
};

// ─── Screen to ray ────────────────────────────────────────────────────────────

[[nodiscard]] inline Ray screen_to_ray(
    Vec2f ndc, // [-1,1] x [-1,1]
    const Mat4f& inv_view, const Mat4f& inv_proj)
{
    Vec4f clip{ndc.x, ndc.y, -1.f, 1.f};
    Vec4f eye_space = inv_proj * clip;
    eye_space.z = -1.f; eye_space.w = 0.f;
    Vec4f world = inv_view * eye_space;
    Vec3f dir = normalize(Vec3f{world.x,world.y,world.z});
    Vec3f origin{inv_view.m[0][3],inv_view.m[1][3],inv_view.m[2][3]};
    return {origin, dir};
}

// ─── DOF circle of confusion ─────────────────────────────────────────────────

[[nodiscard]] inline f32 circle_of_confusion(
    f32 depth, f32 focus_dist, f32 aperture, f32 focal_length)
{
    // Thin lens formula approximation
    f32 num = aperture * focal_length * std::abs(depth - focus_dist);
    f32 den = depth * std::abs(focus_dist - focal_length) + 1e-6f;
    return num / den;
}

} // namespace qs
