#pragma once
// qs_math/animation_blending.hpp — Pose blending, blend trees, additive, IK integration (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "quat.hpp"
#include "mat4.hpp"
#include "animation_math.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <functional>
#include <string>

namespace qs {

// ─── Bone transform ───────────────────────────────────────────────────────────

struct BoneTransform {
    Vec3f translation{0,0,0};
    Quatf rotation{0,0,0,1};
    Vec3f scale{1,1,1};

    [[nodiscard]] static BoneTransform identity() {
        return {{0,0,0},{0,0,0,1},{1,1,1}};
    }

    [[nodiscard]] Mat4f to_matrix() const {
        Mat4f r = rotation.to_matrix4();
        Mat4f s = Mat4f::scale(scale);
        Mat4f t = Mat4f::translation(translation);
        return t * r * s;
    }
};

// ─── Pose ─────────────────────────────────────────────────────────────────────

struct Pose {
    std::vector<BoneTransform> bones;
    u32 num_bones{};

    void resize(u32 n) { num_bones=n; bones.resize(n, BoneTransform::identity()); }

    [[nodiscard]] static Pose lerp(const Pose& a, const Pose& b, f32 t) {
        Pose out; out.resize(a.num_bones);
        for (u32 i=0;i<a.num_bones&&i<b.num_bones;++i) {
            out.bones[i].translation = a.bones[i].translation*(1-t)+b.bones[i].translation*t;
            out.bones[i].rotation    = slerp(a.bones[i].rotation, b.bones[i].rotation, t);
            out.bones[i].scale       = a.bones[i].scale*(1-t)+b.bones[i].scale*t;
        }
        return out;
    }

    // Additive blend: base + (additive - ref) * weight
    [[nodiscard]] static Pose additive(
        const Pose& base, const Pose& add, const Pose& ref, f32 weight)
    {
        Pose out; out.resize(base.num_bones);
        for (u32 i=0;i<base.num_bones;++i) {
            Vec3f dt = (add.bones[i].translation - ref.bones[i].translation)*weight;
            out.bones[i].translation = base.bones[i].translation + dt;
            Quatf dq = add.bones[i].rotation * inverse(ref.bones[i].rotation);
            out.bones[i].rotation = slerp(Quatf{0,0,0,1}, dq, weight) * base.bones[i].rotation;
            out.bones[i].scale    = base.bones[i].scale;
        }
        return out;
    }
};

// ─── Blend mask ───────────────────────────────────────────────────────────────

struct BlendMask {
    std::vector<f32> weights; // per bone, 0=ignore 1=full blend

    void resize(u32 n, f32 default_w=1.f) { weights.assign(n, default_w); }

    void set_bone_range(u32 start, u32 end, f32 w) {
        for (u32 i=start;i<end&&i<weights.size();++i) weights[i]=w;
    }
};

[[nodiscard]] inline Pose masked_lerp(
    const Pose& a, const Pose& b, f32 t, const BlendMask& mask)
{
    Pose out = a;
    for (u32 i=0;i<a.num_bones&&i<b.num_bones&&i<mask.weights.size();++i) {
        f32 w = mask.weights[i]*t;
        out.bones[i].translation = a.bones[i].translation*(1-w)+b.bones[i].translation*w;
        out.bones[i].rotation    = slerp(a.bones[i].rotation, b.bones[i].rotation, w);
        out.bones[i].scale       = a.bones[i].scale*(1-w)+b.bones[i].scale*w;
    }
    return out;
}

// ─── Blend tree node ─────────────────────────────────────────────────────────

enum class BlendNodeType { Clip, Lerp1D, Lerp2D, Additive, Override, Mask };

struct BlendNode {
    BlendNodeType type{BlendNodeType::Clip};
    std::string name{};
    f32 param_x{0.f}, param_y{0.f};
    std::vector<u32> children;   // indices into node list
    std::vector<f32> thresholds; // for 1D blend
    u32 clip_id{0};
    BlendMask mask;
};

// ─── 1D blend space ───────────────────────────────────────────────────────────

[[nodiscard]] inline Pose blend1d(
    const std::vector<Pose>& poses,
    const std::vector<f32>& thresholds,
    f32 param)
{
    u32 N = u32(poses.size());
    if (N == 0) return {};
    if (N == 1) return poses[0];
    // Find surrounding pair
    for (u32 i=0;i<N-1;++i) {
        if (param <= thresholds[i+1]) {
            f32 t = (param-thresholds[i])/(thresholds[i+1]-thresholds[i]+1e-8f);
            return Pose::lerp(poses[i], poses[i+1], std::clamp(t,0.f,1.f));
        }
    }
    return poses[N-1];
}

// ─── 2D blend space (bilinear) ────────────────────────────────────────────────

struct BlendPoint2D {
    f32 x{}, y{};
    Pose pose;
};

[[nodiscard]] inline Pose blend2d_nearest_triangle(
    const std::vector<BlendPoint2D>& pts, f32 px, f32 py)
{
    // Inverse distance weighting fallback
    f32 total_w = 0.f;
    std::vector<f32> weights(pts.size());
    for (u32 i=0;i<u32(pts.size());++i) {
        f32 d = std::sqrt((pts[i].x-px)*(pts[i].x-px)+(pts[i].y-py)*(pts[i].y-py));
        weights[i] = d < 1e-6f ? 1e6f : 1.f/d;
        total_w += weights[i];
    }
    // Weighted average
    Pose out; out.resize(pts[0].pose.num_bones);
    for (u32 i=0;i<u32(pts.size());++i) {
        f32 w = weights[i]/total_w;
        for (u32 b=0;b<out.num_bones;++b) {
            out.bones[b].translation = out.bones[b].translation + pts[i].pose.bones[b].translation*w;
            out.bones[b].rotation    = slerp(out.bones[b].rotation, pts[i].pose.bones[b].rotation, w);
            out.bones[b].scale       = out.bones[b].scale + pts[i].pose.bones[b].scale*w;
        }
    }
    return out;
}

// ─── Motion matching helpers ──────────────────────────────────────────────────

struct MotionFeature {
    Vec3f root_vel{};
    Vec3f foot_l_pos{}, foot_r_pos{};
    Vec3f foot_l_vel{}, foot_r_vel{};
    Vec3f facing{};
};

[[nodiscard]] inline f32 motion_feature_cost(
    const MotionFeature& a, const MotionFeature& b,
    f32 w_vel=1.f, f32 w_foot=2.f, f32 w_facing=0.5f)
{
    f32 c = length_sq(a.root_vel-b.root_vel)*w_vel
           +length_sq(a.foot_l_pos-b.foot_l_pos)*w_foot
           +length_sq(a.foot_r_pos-b.foot_r_pos)*w_foot
           +length_sq(a.foot_l_vel-b.foot_l_vel)*w_foot*0.5f
           +length_sq(a.foot_r_vel-b.foot_r_vel)*w_foot*0.5f
           +length_sq(a.facing-b.facing)*w_facing;
    return c;
}

// ─── Foot IK (ground clamping) ────────────────────────────────────────────────

struct FootIKParams {
    f32 max_step_height{0.4f};
    f32 foot_height_offset{0.05f};
    f32 blend_speed{8.f};
    f32 hip_drop_limit{0.3f};
};

[[nodiscard]] inline Vec3f foot_ik_target(
    const Vec3f& animated_foot, const Vec3f& ground_hit,
    const Vec3f& ground_normal, const FootIKParams& p)
{
    Vec3f target = ground_hit + ground_normal * p.foot_height_offset;
    f32 dy = target.y - animated_foot.y;
    if (dy > p.max_step_height) dy = p.max_step_height;
    return {animated_foot.x, animated_foot.y+dy, animated_foot.z};
}

// ─── Pose retargeting ─────────────────────────────────────────────────────────

struct RetargetMap {
    std::vector<i32> src_to_dst; // -1 = no mapping
    std::vector<Quatf> bind_corrections;
};

[[nodiscard]] inline Pose retarget_pose(
    const Pose& src, const RetargetMap& map, u32 dst_num_bones)
{
    Pose out; out.resize(dst_num_bones);
    for (u32 s=0;s<u32(map.src_to_dst.size())&&s<src.num_bones;++s) {
        i32 d = map.src_to_dst[s];
        if (d < 0) continue;
        out.bones[u32(d)].translation = src.bones[s].translation;
        Quatf corr = s < map.bind_corrections.size() ? map.bind_corrections[s] : Quatf{0,0,0,1};
        out.bones[u32(d)].rotation = corr * src.bones[s].rotation * inverse(corr);
        out.bones[u32(d)].scale    = src.bones[s].scale;
    }
    return out;
}

} // namespace qs
