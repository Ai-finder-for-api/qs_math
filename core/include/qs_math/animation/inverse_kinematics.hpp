#pragma once
// qs_math/animation/inverse_kinematics.hpp — FABRIK, Jacobian IK, constraints, pose blending
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

struct IKV3 { f64 x,y,z;
    [[nodiscard]] constexpr IKV3 operator+(IKV3 b) const noexcept{return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr IKV3 operator-(IKV3 b) const noexcept{return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr IKV3 operator*(f64 s)  const noexcept{return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64  dot(IKV3 b)  const noexcept{return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr f64  len2()       const noexcept{return dot(*this);}
    [[nodiscard]]           f64  len()        const noexcept{return std::sqrt(len2());}
    [[nodiscard]]           IKV3 norm()       const noexcept{f64 l=len();return l>1e-12?IKV3{x/l,y/l,z/l}:*this;}
    [[nodiscard]] constexpr f64  dist(IKV3 b) const noexcept{return (*this-b).len();}
};

// ─── FABRIK (Forward And Backward Reaching IK) ────────────────────────────────
template<int N>
struct FABRIKChain {
    std::array<IKV3,N> joints{};
    std::array<f64,N-1> lengths{}; // bone lengths
    IKV3 root{};
    f64 tolerance{0.001};

    constexpr void init_lengths() noexcept {
        for(int i=0;i<N-1;++i) lengths[i]=joints[i+1].dist(joints[i]);
    }
    [[nodiscard]] constexpr f64 total_length() const noexcept {
        f64 s=0; for(auto l:lengths) s+=l; return s;
    }
    [[nodiscard]] constexpr bool target_reachable(IKV3 target) const noexcept {
        return root.dist(target)<=total_length();
    }

    constexpr void solve(IKV3 target, int max_iter=30) noexcept {
        IKV3 root_pos=joints[0];
        for(int iter=0;iter<max_iter;++iter) {
            if(joints[N-1].dist(target)<tolerance) break;
            // Forward pass
            joints[N-1]=target;
            for(int i=N-2;i>=0;--i) {
                IKV3 dir=(joints[i]-joints[i+1]).norm();
                joints[i]=joints[i+1]+dir*lengths[i];
            }
            // Backward pass
            joints[0]=root_pos;
            for(int i=0;i<N-1;++i) {
                IKV3 dir=(joints[i+1]-joints[i]).norm();
                joints[i+1]=joints[i]+dir*lengths[i];
            }
        }
    }

    // Solve with multiple targets (constrained FABRIK)
    constexpr void solve_constrained(IKV3 target,
                                      const std::array<f64,N>& angle_limits,
                                      int max_iter=30) noexcept
    {
        solve(target,max_iter);
        // Apply angle limits (clamping)
        for(int i=1;i<N-1;++i) {
            IKV3 v1=(joints[i-1]-joints[i]).norm();
            IKV3 v2=(joints[i+1]-joints[i]).norm();
            f64 cos_a=v1.dot(v2);
            f64 angle=std::acos(std::max(-1.0,std::min(1.0,cos_a)));
            if(angle>angle_limits[i]) {
                // Clamp bone i+1
                f64 max_a=angle_limits[i];
                f64 blend=max_a/angle;
                IKV3 mid=v1*blend+v2*(1-blend);
                joints[i+1]=joints[i]+mid.norm()*lengths[i];
            }
        }
    }
};

// ─── Jacobian transpose IK ────────────────────────────────────────────────────
template<int N_JOINTS>
[[nodiscard]] constexpr std::array<f64,N_JOINTS> jacobian_transpose_ik(
    const std::array<std::array<f64,3>,N_JOINTS>& joint_axes, // rotation axes
    const std::array<IKV3,N_JOINTS>& joint_positions,
    IKV3 end_effector, IKV3 target,
    f64 alpha=0.1) noexcept
{
    IKV3 e=target-end_effector;
    std::array<f64,N_JOINTS> dq{};
    for(int i=0;i<N_JOINTS;++i) {
        // Column of Jacobian: axis × (end_effector - joint_i)
        IKV3 r=end_effector-joint_positions[i];
        IKV3 ax={joint_axes[i][0],joint_axes[i][1],joint_axes[i][2]};
        IKV3 J_col=ax.cross(r);
        dq[i]=alpha*(J_col.x*e.x+J_col.y*e.y+J_col.z*e.z);
    }
    return dq;
}

// ─── Pose blending (quaternion SLERP chain) ────────────────────────────────────
struct Pose {
    std::array<std::array<f64,4>,32> joint_rots{}; // quaternions per joint
    std::array<std::array<f64,3>,32> joint_pos{};
    int n_joints{0};
};

[[nodiscard]] constexpr std::array<f64,4> quat_slerp(
    const std::array<f64,4>& q0, const std::array<f64,4>& q1, f64 t) noexcept
{
    f64 dot=q0[0]*q1[0]+q0[1]*q1[1]+q0[2]*q1[2]+q0[3]*q1[3];
    std::array<f64,4> q1p=q1; if(dot<0){for(auto& v:q1p)v=-v;dot=-dot;}
    if(dot>0.9995f){
        std::array<f64,4> r;
        for(int i=0;i<4;++i)r[i]=q0[i]+(q1p[i]-q0[i])*t;
        f64 n=std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+r[3]*r[3]);
        for(auto& v:r)v/=n; return r;
    }
    f64 theta=std::acos(dot);
    f64 s0=std::sin((1-t)*theta)/std::sin(theta);
    f64 s1=std::sin(t*theta)/std::sin(theta);
    std::array<f64,4> r;
    for(int i=0;i<4;++i)r[i]=s0*q0[i]+s1*q1p[i];
    return r;
}

[[nodiscard]] constexpr Pose blend_poses(
    const Pose& A, const Pose& B, f64 t) noexcept
{
    Pose result=A;
    for(int j=0;j<A.n_joints;++j) {
        result.joint_rots[j]=quat_slerp(A.joint_rots[j],B.joint_rots[j],t);
        for(int k=0;k<3;++k)
            result.joint_pos[j][k]=A.joint_pos[j][k]*(1-t)+B.joint_pos[j][k]*t;
    }
    return result;
}

// ─── Foot planting (IK constraint for grounded contact) ──────────────────────
struct FootPlant {
    IKV3 planted_pos{};
    bool active{false};
    f64 blend{0};

    constexpr void update(IKV3 animated_pos, f64 height_threshold=0.05) noexcept {
        if(!active && animated_pos.y < height_threshold) {
            planted_pos={animated_pos.x,0,animated_pos.z};
            active=true;
        }
        if(active && animated_pos.y > height_threshold*3) active=false;
        blend=active?std::min(1.0,blend+0.1):std::max(0.0,blend-0.1);
    }
    [[nodiscard]] constexpr IKV3 get_target(IKV3 animated_pos) const noexcept {
        return animated_pos*(1-blend)+planted_pos*blend;
    }
};

// ─── Look-at IK (orient bone toward target) ───────────────────────────────────
[[nodiscard]] constexpr std::array<f64,4> look_at_quat(
    IKV3 forward, IKV3 target_dir) noexcept
{
    target_dir=target_dir.norm();
    f64 dot=forward.dot(target_dir);
    if(dot>0.9999f) return{1,0,0,0};
    if(dot<-0.9999f) return{0,1,0,0}; // 180° rotation around any perp axis
    IKV3 axis=forward.cross(target_dir).norm();
    f64 angle=std::acos(std::max(-1.0,std::min(1.0,dot)));
    f64 s=std::sin(angle*0.5);
    return{std::cos(angle*0.5),axis.x*s,axis.y*s,axis.z*s};
}

// ─── Reachability map (precomputed workspace) ─────────────────────────────────
template<int NX, int NY, int NZ, int N_JOINTS>
[[nodiscard]] constexpr std::array<std::array<std::array<bool,NZ>,NY>,NX>
build_reachability_map(
    const FABRIKChain<N_JOINTS>& chain,
    f64 x_min, f64 x_max,
    f64 y_min, f64 y_max,
    f64 z_min, f64 z_max) noexcept
{
    std::array<std::array<std::array<bool,NZ>,NY>,NX> map{};
    f64 total=chain.total_length();
    for(int ix=0;ix<NX;++ix) for(int iy=0;iy<NY;++iy) for(int iz=0;iz<NZ;++iz) {
        IKV3 pt{x_min+(x_max-x_min)*ix/(NX-1),
                y_min+(y_max-y_min)*iy/(NY-1),
                z_min+(z_max-z_min)*iz/(NZ-1)};
        map[ix][iy][iz]=chain.root.dist(pt)<=total;
    }
    return map;
}
} // namespace qs::math
