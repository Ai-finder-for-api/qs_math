#pragma once
// qs_math/ik_math.hpp — FABRIK, CCD, Jacobian IK, chain setup (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "quat.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>

namespace qs {

struct IKJoint {
    Vec3f pos{};
    f32   length{1.f};      // length to next joint
    Vec3f axis{0,1,0};      // preferred rotation axis
    f32   min_angle{-k_pi};
    f32   max_angle{ k_pi};
};

struct IKChain {
    std::vector<IKJoint> joints;
    Vec3f root{};

    [[nodiscard]] u32 size() const { return u32(joints.size()); }
    [[nodiscard]] f32 total_length() const {
        f32 t=0.f; for(const auto& j:joints) t+=j.length; return t;
    }
};

// ─── FABRIK ───────────────────────────────────────────────────────────────────

struct FABRIKParams {
    u32  max_iterations{10};
    f32  tolerance{0.001f};
    bool constrain_root{true};
};

[[nodiscard]] inline bool fabrik_solve(
    IKChain& chain, const Vec3f& target, const FABRIKParams& p)
{
    u32 N = chain.size();
    if (N == 0) return false;
    Vec3f root_pos = chain.root;

    // Unreachable check
    if (length(target - root_pos) >= chain.total_length() * 0.9999f) {
        Vec3f d = normalize(target - root_pos);
        chain.joints[0].pos = root_pos;
        for (u32 i=1;i<N;++i)
            chain.joints[i].pos = chain.joints[i-1].pos + d*chain.joints[i-1].length;
        return false;
    }

    for (u32 iter=0; iter<p.max_iterations; ++iter) {
        // Forward pass
        chain.joints[N-1].pos = target;
        for (i32 i=i32(N)-2; i>=0; --i) {
            Vec3f d = normalize(chain.joints[i].pos - chain.joints[i+1].pos);
            chain.joints[i].pos = chain.joints[i+1].pos + d*chain.joints[i].length;
        }
        // Backward pass
        if (p.constrain_root) chain.joints[0].pos = root_pos;
        for (u32 i=0;i<N-1;++i) {
            Vec3f d = normalize(chain.joints[i+1].pos - chain.joints[i].pos);
            chain.joints[i+1].pos = chain.joints[i].pos + d*chain.joints[i].length;
        }
        if (length(chain.joints[N-1].pos - target) < p.tolerance) return true;
    }
    return false;
}

// ─── CCD (Cyclic Coordinate Descent) ─────────────────────────────────────────

[[nodiscard]] inline bool ccd_solve(
    IKChain& chain, const Vec3f& target,
    u32 max_iter=20, f32 tol=0.001f)
{
    u32 N = chain.size();
    for (u32 iter=0; iter<max_iter; ++iter) {
        for (i32 i=i32(N)-2; i>=0; --i) {
            Vec3f end_pos = chain.joints[N-1].pos;
            Vec3f joint_pos = chain.joints[i].pos;
            Vec3f to_end = normalize(end_pos - joint_pos);
            Vec3f to_target = normalize(target - joint_pos);
            f32 cos_a = std::clamp(dot(to_end, to_target), -1.f, 1.f);
            if (std::abs(cos_a) > 0.9999f) continue;
            Vec3f axis = normalize(cross(to_end, to_target));
            f32 angle = std::acos(cos_a);
            Quatf rot = Quatf::from_axis_angle(axis, angle);
            // Rotate all subsequent joints around this joint
            for (u32 j=i32(i)+1; j<N; ++j) {
                chain.joints[j].pos = joint_pos + rot.rotate(chain.joints[j].pos - joint_pos);
            }
        }
        if (length(chain.joints[N-1].pos - target) < tol) return true;
    }
    return false;
}

// ─── Jacobian IK ─────────────────────────────────────────────────────────────

// Compute Jacobian column for joint i (cross product of axis and (end-joint))
[[nodiscard]] inline Vec3f jacobian_column(
    const Vec3f& joint_pos, const Vec3f& axis,
    const Vec3f& end_effector)
{
    return cross(axis, end_effector - joint_pos);
}

// Damped Least Squares solve step (one iteration)
inline void jacobian_dls_step(
    IKChain& chain, const Vec3f& target,
    f32 step_size=0.1f, f32 damping=0.5f)
{
    u32 N = chain.size();
    Vec3f end = chain.joints[N-1].pos;
    Vec3f error = target - end;
    f32 err_len = length(error);
    if (err_len < 0.001f) return;

    // Build 3×N Jacobian
    std::vector<Vec3f> J(N-1);
    for (u32 i=0;i<N-1;++i)
        J[i] = jacobian_column(chain.joints[i].pos, chain.joints[i].axis, end);

    // DLS: dq = Jt * (J*Jt + lambda^2*I)^-1 * e
    // Simplified: dq_i = dot(J_i, e) / (sum(|J_i|^2) + damping^2)
    f32 JtJ = 0.f;
    for (auto& j : J) JtJ += dot(j,j);
    f32 denom = JtJ + damping*damping;

    for (u32 i=0;i<N-1;++i) {
        f32 dq = dot(J[i], error) / denom * step_size;
        Quatf rot = Quatf::from_axis_angle(chain.joints[i].axis, dq);
        for (u32 j=i+1;j<N;++j)
            chain.joints[j].pos = chain.joints[i].pos
                                 + rot.rotate(chain.joints[j].pos - chain.joints[i].pos);
    }
}

// ─── 2-bone IK (analytic) ────────────────────────────────────────────────────

struct TwoBoneIKResult {
    Vec3f mid{};
    bool  reachable{};
};

[[nodiscard]] inline TwoBoneIKResult two_bone_ik(
    const Vec3f& root, const Vec3f& target,
    f32 len_a, f32 len_b, const Vec3f& pole_hint)
{
    Vec3f to_target = target - root;
    f32 dist = length(to_target);
    dist = std::clamp(dist, std::abs(len_a-len_b)+1e-4f, len_a+len_b-1e-4f);

    // Law of cosines: angle at root
    f32 cos_a = (len_a*len_a + dist*dist - len_b*len_b) / (2.f*len_a*dist);
    cos_a = std::clamp(cos_a, -1.f, 1.f);
    f32 angle_a = std::acos(cos_a);

    // Plane formed by root, target, pole
    Vec3f fwd = normalize(to_target);
    Vec3f side = normalize(cross(pole_hint - root, fwd));
    Vec3f up2  = cross(fwd, side);

    // Mid position
    f32 c = std::cos(angle_a), s = std::sin(angle_a);
    Vec3f mid = root + (fwd*c + up2*s)*len_a;
    return {mid, true};
}

// ─── Spine/spline IK (arc-length parameterized) ───────────────────────────────

inline void spine_ik_along_curve(
    IKChain& chain,
    const std::vector<Vec3f>& curve_points)
{
    if (curve_points.size() < 2) return;
    // Compute arc lengths
    std::vector<f32> arc(curve_points.size(), 0.f);
    for (u32 i=1;i<u32(curve_points.size());++i)
        arc[i] = arc[i-1] + length(curve_points[i]-curve_points[i-1]);
    f32 total_arc = arc.back();

    // Place joints along curve by cumulative chain length
    f32 cum = 0.f;
    chain.joints[0].pos = curve_points[0];
    for (u32 j=1;j<chain.size();++j) {
        cum += chain.joints[j-1].length;
        f32 t = std::clamp(cum/total_arc, 0.f, 1.f) * total_arc;
        // Find segment
        for (u32 i=1;i<u32(curve_points.size());++i) {
            if (arc[i] >= t) {
                f32 frac = (t-arc[i-1])/(arc[i]-arc[i-1]+1e-8f);
                chain.joints[j].pos = curve_points[i-1]*(1-frac)+curve_points[i]*frac;
                break;
            }
        }
    }
}

} // namespace qs
