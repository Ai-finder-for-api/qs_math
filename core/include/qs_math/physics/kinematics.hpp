#pragma once
// qs_math/physics/kinematics.hpp — Kinematics: forward/inverse kinematics,
//   Denavit-Hartenberg, Jacobian, FABRIK, CCD, trajectory planning,
//   velocity/acceleration profiles, workspace analysis (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/mat4.hpp"
#include "../linear_algebra/quat.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <span>
#include <functional>

namespace qs::math::physics {

// ─── Denavit-Hartenberg parameters ───────────────────────────────────────
struct DHParams {
    f32 a;     // link length
    f32 alpha; // link twist
    f32 d;     // link offset
    f32 theta; // joint angle (variable)
};

// DH transformation matrix: T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
[[nodiscard]] inline Mat4f dh_matrix(const DHParams& dh) noexcept {
    f32 ct=std::cos(dh.theta), st=std::sin(dh.theta);
    f32 ca=std::cos(dh.alpha), sa=std::sin(dh.alpha);
    return Mat4f{
        {ct,   -st*ca,  st*sa, dh.a*ct},
        {st,    ct*ca, -ct*sa, dh.a*st},
        {0,     sa,     ca,    dh.d   },
        {0,     0,      0,     1      }
    };
}

// Forward kinematics: compute end-effector pose from joint angles
[[nodiscard]] inline Mat4f forward_kinematics(
    std::span<const DHParams> chain) noexcept
{
    Mat4f T = Mat4f::identity();
    for (auto& dh : chain) T = T * dh_matrix(dh);
    return T;
}

// ─── Chain definition ─────────────────────────────────────────────────────
struct Joint {
    Vec3f pos;           // current world position
    Vec3f axis;          // rotation axis (world space)
    f32   angle;         // current angle
    f32   lo_limit=-f32(PI), hi_limit=f32(PI);
    bool  is_fixed=false;
};

struct IKChain {
    std::vector<Joint> joints;
    Vec3f target;
    f32   tolerance=0.001f;

    [[nodiscard]] Vec3f end_effector() const noexcept {
        return joints.empty()?Vec3f{0,0,0}:joints.back().pos;
    }
    [[nodiscard]] f32 link_length(int i) const noexcept {
        if(i+1>=int(joints.size())) return 0;
        return (joints[i+1].pos-joints[i].pos).length();
    }
};

// ─── FABRIK (Forward And Backward Reaching IK) ────────────────────────────
inline void fabrik_solve(IKChain& chain, int max_iter=20) noexcept {
    int n=int(chain.joints.size());
    if(n<2) return;

    // Cache link lengths
    std::vector<f32> lens(n-1);
    for(int i=0;i<n-1;++i) lens[i]=chain.link_length(i);

    // Check if target is reachable
    f32 total_len=0; for(auto l:lens) total_len+=l;
    Vec3f base=chain.joints[0].pos;
    if((chain.target-base).length()>=total_len){
        // Stretch toward target
        Vec3f dir=(chain.target-base).normalized();
        for(int i=1;i<n;++i) chain.joints[i].pos=chain.joints[i-1].pos+dir*lens[i-1];
        return;
    }

    for(int iter=0;iter<max_iter;++iter){
        // Forward pass: pull toward target
        chain.joints[n-1].pos=chain.target;
        for(int i=n-2;i>=0;--i){
            Vec3f dir=(chain.joints[i].pos-chain.joints[i+1].pos).normalized();
            chain.joints[i].pos=chain.joints[i+1].pos+dir*lens[i];
        }
        // Backward pass: restore base
        chain.joints[0].pos=base;
        for(int i=0;i<n-1;++i){
            Vec3f dir=(chain.joints[i+1].pos-chain.joints[i].pos).normalized();
            chain.joints[i+1].pos=chain.joints[i].pos+dir*lens[i];
        }
        if((chain.end_effector()-chain.target).length()<chain.tolerance) break;
    }
}

// ─── CCD (Cyclic Coordinate Descent) IK ───────────────────────────────────
inline void ccd_solve(IKChain& chain, int max_iter=50) noexcept {
    int n=int(chain.joints.size());
    if(n<2) return;
    for(int iter=0;iter<max_iter;++iter){
        bool converged=true;
        for(int i=n-2;i>=0;--i){
            if(chain.joints[i].is_fixed) continue;
            Vec3f to_ee =(chain.end_effector()-chain.joints[i].pos).normalized();
            Vec3f to_tgt=(chain.target-chain.joints[i].pos).normalized();
            if(dot(to_ee,to_tgt)>0.9999f) continue;
            Vec3f axis=cross(to_ee,to_tgt);
            f32 alen=axis.length();
            if(alen<1e-7f) continue;
            f32 angle=std::asin(clamp(alen,-1.f,1.f));
            Quatf rot=Quatf::from_axis_angle(axis*(1.f/alen),angle);
            // Rotate all joints from i+1 onward around joint i
            Vec3f pivot=chain.joints[i].pos;
            for(int j=i+1;j<n;++j)
                chain.joints[j].pos=pivot+rot*(chain.joints[j].pos-pivot);
            // Apply joint limits
            chain.joints[i].angle=clamp(chain.joints[i].angle+angle,
                chain.joints[i].lo_limit, chain.joints[i].hi_limit);
            if((chain.end_effector()-chain.target).length()>chain.tolerance) converged=false;
        }
        if(converged) break;
    }
}

// ─── Jacobian IK ──────────────────────────────────────────────────────────
// Compute 3xN Jacobian of end-effector position w.r.t. joint angles
// (revolute joints only, rotation axis = Y)
[[nodiscard]] inline std::vector<f32> compute_jacobian(const IKChain& chain) noexcept {
    int n=int(chain.joints.size())-1; // n DOF
    std::vector<f32> J(3*n,0);
    Vec3f ee=chain.end_effector();
    for(int i=0;i<n;++i){
        Vec3f r=ee-chain.joints[i].pos;
        Vec3f col=cross(chain.joints[i].axis,r);
        J[0*n+i]=col.x; J[1*n+i]=col.y; J[2*n+i]=col.z;
    }
    return J;
}

// Pseudoinverse-based Jacobian IK step (damped least squares)
inline void jacobian_ik_step(
    IKChain& chain, f32 lambda=0.01f, f32 step=0.1f) noexcept
{
    int n=int(chain.joints.size())-1;
    Vec3f err=chain.target-chain.end_effector();
    auto J=compute_jacobian(chain);
    // dq = J^T (J J^T + lambda^2 I)^{-1} err  (DLS)
    // For small n, compute JJ^T (3x3) and invert
    f32 JJT[9]={};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        for(int k=0;k<n;++k) JJT[i*3+j]+=J[i*n+k]*J[j*n+k];
    JJT[0]+=lambda*lambda; JJT[4]+=lambda*lambda; JJT[8]+=lambda*lambda;
    // Cramer's rule inverse of 3x3
    f32 det=JJT[0]*(JJT[4]*JJT[8]-JJT[5]*JJT[7])
           -JJT[1]*(JJT[3]*JJT[8]-JJT[5]*JJT[6])
           +JJT[2]*(JJT[3]*JJT[7]-JJT[4]*JJT[6]);
    if(std::abs(det)<1e-10f) return;
    f32 inv_det=1.f/det;
    f32 inv[9];
    inv[0]=(JJT[4]*JJT[8]-JJT[5]*JJT[7])*inv_det;
    inv[1]=(JJT[2]*JJT[7]-JJT[1]*JJT[8])*inv_det;
    inv[2]=(JJT[1]*JJT[5]-JJT[2]*JJT[4])*inv_det;
    inv[3]=(JJT[5]*JJT[6]-JJT[3]*JJT[8])*inv_det;
    inv[4]=(JJT[0]*JJT[8]-JJT[2]*JJT[6])*inv_det;
    inv[5]=(JJT[2]*JJT[3]-JJT[0]*JJT[5])*inv_det;
    inv[6]=(JJT[3]*JJT[7]-JJT[4]*JJT[6])*inv_det;
    inv[7]=(JJT[1]*JJT[6]-JJT[0]*JJT[7])*inv_det;
    inv[8]=(JJT[0]*JJT[4]-JJT[1]*JJT[3])*inv_det;
    // v = inv * err (3)
    f32 v[3]={inv[0]*err.x+inv[1]*err.y+inv[2]*err.z,
               inv[3]*err.x+inv[4]*err.y+inv[5]*err.z,
               inv[6]*err.x+inv[7]*err.y+inv[8]*err.z};
    // dq = J^T * v
    for(int k=0;k<n;++k){
        f32 dq=(J[0*n+k]*v[0]+J[1*n+k]*v[1]+J[2*n+k]*v[2])*step;
        chain.joints[k].angle=clamp(chain.joints[k].angle+dq,
            chain.joints[k].lo_limit,chain.joints[k].hi_limit);
    }
}

// ─── Trajectory planning ──────────────────────────────────────────────────
// Trapezoidal velocity profile: ramp up, cruise, ramp down
struct TrapProfile {
    f32 t_acc; // acceleration phase duration
    f32 t_cru; // cruise phase duration
    f32 t_dec; // deceleration phase duration
    f32 v_max; // max velocity
    f32 a_max; // max acceleration
    f32 dist;  // total distance

    static TrapProfile plan(f32 dist, f32 v_max, f32 a_max) noexcept {
        f32 t_acc=v_max/a_max;
        f32 d_acc=0.5f*a_max*t_acc*t_acc;
        if(2*d_acc>dist){
            // Triangular profile (no cruise)
            t_acc=std::sqrt(dist/a_max);
            return {t_acc,0,t_acc,a_max*t_acc,a_max,dist};
        }
        f32 t_cru=(dist-2*d_acc)/v_max;
        return {t_acc,t_cru,t_acc,v_max,a_max,dist};
    }

    [[nodiscard]] f32 position(f32 t) const noexcept {
        if(t<0) return 0;
        if(t<t_acc) return 0.5f*a_max*t*t;
        f32 d_acc=0.5f*a_max*t_acc*t_acc;
        if(t<t_acc+t_cru) return d_acc+v_max*(t-t_acc);
        f32 t3=t-(t_acc+t_cru);
        if(t3>t_dec) return dist;
        f32 d_cru=v_max*t_cru;
        return d_acc+d_cru+v_max*t3-0.5f*a_max*t3*t3;
    }

    [[nodiscard]] f32 velocity(f32 t) const noexcept {
        if(t<0) return 0;
        if(t<t_acc) return a_max*t;
        if(t<t_acc+t_cru) return v_max;
        f32 t3=t-(t_acc+t_cru);
        if(t3>t_dec) return 0;
        return v_max-a_max*t3;
    }

    [[nodiscard]] f32 total_time() const noexcept { return t_acc+t_cru+t_dec; }
};

// S-curve velocity profile (7th-order polynomial, jerk-limited)
struct SCurveProfile {
    f32 j_max, a_max, v_max, dist;
    f32 T; // total time (simplified, equal-phase)

    static SCurveProfile plan(f32 d, f32 v, f32 a, f32 j) noexcept {
        f32 t_j=a/j;
        f32 t_a=v/a;
        f32 t_total=t_j+t_a+t_j; // simplified
        return {j,a,v,d,t_total};
    }

    [[nodiscard]] f32 position(f32 t) const noexcept {
        // 7-segment interpolation (simplified: just cubic blend)
        f32 t_norm=clamp01(t/T);
        // Smooth step ^ 3 approximation
        f32 s=t_norm*t_norm*(3-2*t_norm);
        return s*dist;
    }
    [[nodiscard]] f32 velocity(f32 t) const noexcept {
        f32 tn=clamp01(t/T);
        f32 ds=6*tn*(1-tn);
        return ds*dist/T;
    }
};

// ─── Workspace analysis ───────────────────────────────────────────────────
// Check if point is in reachable workspace of planar 2-link arm
[[nodiscard]] inline bool in_workspace_2link(
    f32 l1, f32 l2, Vec3f target, Vec3f base={0,0,0}) noexcept
{
    f32 r=(target-base).length();
    return r>=std::abs(l1-l2) && r<=(l1+l2);
}

// 2-link planar IK (2 solutions)
struct IK2LinkSolution { f32 theta1, theta2; bool valid; };

[[nodiscard]] inline std::array<IK2LinkSolution,2> ik_2link(
    f32 l1, f32 l2, Vec2f target) noexcept
{
    f32 x=target.x, y=target.y;
    f32 r2=x*x+y*y;
    f32 cos2=(r2-l1*l1-l2*l2)/(2*l1*l2);
    if(std::abs(cos2)>1.f) return {{0,0,false},{0,0,false}};
    f32 sin2_pos=std::sqrt(1-cos2*cos2), sin2_neg=-sin2_pos;
    auto solve=[&](f32 sin2)->IK2LinkSolution{
        f32 t2=std::atan2(sin2,cos2);
        f32 k1=l1+l2*cos2, k2=l2*sin2;
        f32 t1=std::atan2(y,x)-std::atan2(k2,k1);
        return {t1,t2,true};
    };
    return {solve(sin2_pos), solve(sin2_neg)};
}

// ─── Smooth trajectory (Bezier waypoints) ─────────────────────────────────
[[nodiscard]] inline Vec3f smooth_trajectory(
    std::span<const Vec3f> waypoints, f32 t) noexcept
{
    // Bezier along waypoints via De Casteljau
    int n=int(waypoints.size());
    std::vector<Vec3f> pts(waypoints.begin(),waypoints.end());
    for(int r=1;r<n;++r)
        for(int i=0;i<n-r;++i)
            pts[i]=pts[i]*(1.f-t)+pts[i+1]*t;
    return pts[0];
}

} // namespace qs::math::physics
