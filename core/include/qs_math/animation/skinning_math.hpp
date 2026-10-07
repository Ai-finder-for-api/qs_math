#pragma once
// qs_math/skinning_math.hpp — LBS, DQS, blend shapes, corrective shapes (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <array>

namespace qs {

// ─── Bind-pose data ───────────────────────────────────────────────────────────

struct SkinVertex {
    Vec3f position{};
    Vec3f normal{};
    Vec4f weights{};    // up to 4 bone weights (must sum to 1)
    std::array<u32,4> indices{};
};

// ─── Linear Blend Skinning (LBS) ─────────────────────────────────────────────

[[nodiscard]] inline Vec3f lbs_position(
    const SkinVertex& v,
    std::span<const Mat4f> skin_matrices) // bone_world * inv_bind_pose
{
    Vec4f p{v.position.x, v.position.y, v.position.z, 1.f};
    Vec4f acc{0,0,0,0};
    for (u32 i=0;i<4;++i) {
        if (v.weights[i] < 1e-5f) continue;
        acc = acc + skin_matrices[v.indices[i]] * p * v.weights[i];
    }
    return {acc.x, acc.y, acc.z};
}

[[nodiscard]] inline Vec3f lbs_normal(
    const SkinVertex& v,
    std::span<const Mat4f> skin_matrices)
{
    Vec4f n{v.normal.x, v.normal.y, v.normal.z, 0.f};
    Vec4f acc{0,0,0,0};
    for (u32 i=0;i<4;++i) {
        if (v.weights[i] < 1e-5f) continue;
        // Use upper-left 3x3 for normals (transpose of inverse for non-uniform scale)
        acc = acc + skin_matrices[v.indices[i]] * n * v.weights[i];
    }
    Vec3f nr{acc.x,acc.y,acc.z};
    return normalize(nr);
}

// Batch LBS over entire mesh
inline void lbs_mesh(
    std::span<const SkinVertex> verts,
    std::span<const Mat4f> skin_matrices,
    std::span<Vec3f> out_pos,
    std::span<Vec3f> out_nrm)
{
    for (u32 i=0;i<u32(verts.size());++i) {
        out_pos[i] = lbs_position(verts[i], skin_matrices);
        out_nrm[i] = lbs_normal(verts[i], skin_matrices);
    }
}

// ─── Dual Quaternion Skinning (DQS) ──────────────────────────────────────────

struct DualQuat {
    Quatf real{0,0,0,1}; // rotation
    Quatf dual{0,0,0,0}; // translation encoded

    [[nodiscard]] static DualQuat from_rt(const Quatf& rot, const Vec3f& t) {
        DualQuat dq;
        dq.real = rot;
        // dual = 0.5 * t_quat * rot
        Quatf tq{t.x*0.5f, t.y*0.5f, t.z*0.5f, 0.f};
        dq.dual = tq * rot;
        return dq;
    }

    [[nodiscard]] static DualQuat from_mat4(const Mat4f& m) {
        // Extract rotation & translation
        Quatf r = Quatf::from_matrix(m);
        Vec3f t{m.m[0][3], m.m[1][3], m.m[2][3]};
        return from_rt(r, t);
    }

    DualQuat operator+(const DualQuat& o) const {
        return {real+o.real, dual+o.dual};
    }
    DualQuat operator*(f32 s) const { return {real*s, dual*s}; }

    void normalize() {
        f32 l = std::sqrt(dot_quat(real,real));
        if (l < 1e-8f) return;
        real = real*(1.f/l);
        dual = dual*(1.f/l);
    }

    // Transform point
    [[nodiscard]] Vec3f transform_point(const Vec3f& p) const {
        // Extract rotation and translation
        Vec3f t{2.f*(real.w*dual.x - dual.w*real.x + real.y*dual.z - real.z*dual.y),
                2.f*(real.w*dual.y - dual.w*real.y + real.z*dual.x - real.x*dual.z),
                2.f*(real.w*dual.z - dual.w*real.z + real.x*dual.y - real.y*dual.x)};
        return real.rotate(p) + t;
    }
    [[nodiscard]] Vec3f transform_normal(const Vec3f& n) const {
        return real.rotate(n);
    }
private:
    static f32 dot_quat(const Quatf& a, const Quatf& b) {
        return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
    }
};

[[nodiscard]] inline Vec3f dqs_position(
    const SkinVertex& v,
    std::span<const DualQuat> bone_dqs)
{
    DualQuat blended{{0,0,0,0},{0,0,0,0}};
    // Antipodality fix: flip if dot < 0
    f32 ref_sign = 1.f;
    for (u32 i=0;i<4;++i) {
        if (v.weights[i] < 1e-5f) continue;
        const DualQuat& dq = bone_dqs[v.indices[i]];
        f32 sign = (i==0||v.weights[0]<1e-5f) ? 1.f
            : (dq.real.x*blended.real.x+dq.real.y*blended.real.y
              +dq.real.z*blended.real.z+dq.real.w*blended.real.w < 0.f ? -1.f : 1.f);
        blended.real.x += dq.real.x*v.weights[i]*sign;
        blended.real.y += dq.real.y*v.weights[i]*sign;
        blended.real.z += dq.real.z*v.weights[i]*sign;
        blended.real.w += dq.real.w*v.weights[i]*sign;
        blended.dual.x += dq.dual.x*v.weights[i]*sign;
        blended.dual.y += dq.dual.y*v.weights[i]*sign;
        blended.dual.z += dq.dual.z*v.weights[i]*sign;
        blended.dual.w += dq.dual.w*v.weights[i]*sign;
    }
    blended.normalize();
    return blended.transform_point(v.position);
}

// ─── Blend Shapes (Morph Targets) ────────────────────────────────────────────

struct BlendShape {
    std::string name{};
    std::vector<Vec3f> delta_pos;  // indexed by vertex
    std::vector<Vec3f> delta_nrm;
    f32 weight{0.f};
};

inline void apply_blend_shapes(
    std::span<Vec3f> positions, std::span<Vec3f> normals,
    std::span<const Vec3f> base_pos, std::span<const Vec3f> base_nrm,
    std::span<const BlendShape> shapes)
{
    u32 N = u32(base_pos.size());
    for (u32 i=0;i<N;++i) { positions[i]=base_pos[i]; normals[i]=base_nrm[i]; }
    for (const auto& s : shapes) {
        if (std::abs(s.weight)<1e-5f) continue;
        for (u32 i=0;i<N&&i<s.delta_pos.size();++i)
            positions[i] = positions[i] + s.delta_pos[i]*s.weight;
        for (u32 i=0;i<N&&i<s.delta_nrm.size();++i)
            normals[i] = normals[i] + s.delta_nrm[i]*s.weight;
    }
    for (auto& n : normals) n = normalize(n);
}

// ─── Corrective blend shapes (pose-space deformation) ────────────────────────

struct CorrectiveShape {
    BlendShape shape;
    u32   bone_index{};
    Quatf trigger_rotation{0,0,0,1};
    f32   trigger_angle{0.f};   // activation angle in radians
    f32   falloff{0.3f};        // angle falloff width

    [[nodiscard]] f32 compute_weight(const Quatf& current_rot) const {
        Quatf diff = current_rot * inverse(trigger_rotation);
        f32 angle = 2.f*std::acos(std::clamp(std::abs(diff.w), 0.f, 1.f));
        f32 d = std::abs(angle - trigger_angle);
        return std::max(0.f, 1.f - d/falloff);
    }
};

// ─── Skin weight smoothing ────────────────────────────────────────────────────

inline void smooth_skin_weights(
    std::span<SkinVertex> verts,
    std::span<const std::vector<u32>> adjacency, // per-vertex neighbor list
    u32 iterations=3, f32 smoothing=0.5f)
{
    u32 N = u32(verts.size());
    for (u32 iter=0;iter<iterations;++iter) {
        for (u32 i=0;i<N;++i) {
            if (adjacency[i].empty()) continue;
            Vec4f avg_w{0,0,0,0};
            for (u32 nb : adjacency[i]) {
                avg_w.x+=verts[nb].weights.x; avg_w.y+=verts[nb].weights.y;
                avg_w.z+=verts[nb].weights.z; avg_w.w+=verts[nb].weights.w;
            }
            f32 k=1.f/f32(adjacency[i].size());
            avg_w.x*=k; avg_w.y*=k; avg_w.z*=k; avg_w.w*=k;
            verts[i].weights.x += (avg_w.x-verts[i].weights.x)*smoothing;
            verts[i].weights.y += (avg_w.y-verts[i].weights.y)*smoothing;
            verts[i].weights.z += (avg_w.z-verts[i].weights.z)*smoothing;
            verts[i].weights.w += (avg_w.w-verts[i].weights.w)*smoothing;
            // Renormalize
            f32 sum=verts[i].weights.x+verts[i].weights.y+verts[i].weights.z+verts[i].weights.w;
            if (sum>1e-6f){f32 inv=1.f/sum;verts[i].weights.x*=inv;verts[i].weights.y*=inv;verts[i].weights.z*=inv;verts[i].weights.w*=inv;}
        }
    }
}

} // namespace qs
