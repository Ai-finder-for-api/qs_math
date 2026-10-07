#pragma once
// qs_math/animation/motion_math.hpp — Kinematics, IK, blend trees, motion capture math
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── 3D vector (animation context) ────────────────────────────────────────────
struct AV3 { f64 x,y,z;
    [[nodiscard]] constexpr AV3 operator+(AV3 b) const noexcept{return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr AV3 operator-(AV3 b) const noexcept{return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr AV3 operator*(f64 s) const noexcept{return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64  dot(AV3 b)  const noexcept{return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr AV3  cross(AV3 b) const noexcept{return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    [[nodiscard]] constexpr f64  len2() const noexcept{return dot(*this);}
    [[nodiscard]]           f64  len()  const noexcept{return std::sqrt(len2());}
    [[nodiscard]]           AV3  norm() const noexcept{f64 l=len();return l>1e-12?AV3{x/l,y/l,z/l}:*this;}
    [[nodiscard]] constexpr AV3  lerp(AV3 b,f64 t) const noexcept{return *this+(b-*this)*t;}
};

// ─── Dual quaternion (rigid transform) ────────────────────────────────────────
struct DualQuat {
    // q = real + ε*dual
    f64 rw,rx,ry,rz; // real part
    f64 dw,dx,dy,dz; // dual part

    [[nodiscard]] constexpr DualQuat operator+(DualQuat b) const noexcept {
        return{rw+b.rw,rx+b.rx,ry+b.ry,rz+b.rz,dw+b.dw,dx+b.dx,dy+b.dy,dz+b.dz};
    }
    [[nodiscard]] constexpr DualQuat operator*(f64 s) const noexcept {
        return{rw*s,rx*s,ry*s,rz*s,dw*s,dx*s,dy*s,dz*s};
    }
    [[nodiscard]] constexpr DualQuat conjugate() const noexcept {
        return{rw,-rx,-ry,-rz,dw,-dx,-dy,-dz};
    }
    [[nodiscard]] constexpr f64 real_norm() const noexcept {
        return rw*rw+rx*rx+ry*ry+rz*rz;
    }
    [[nodiscard]] constexpr DualQuat normalize() const noexcept {
        f64 n=std::sqrt(real_norm());
        if(n<1e-12) return *this;
        return *this*(1.0/n);
    }
    // From rotation quaternion and translation
    static constexpr DualQuat from_rt(f64 qw,f64 qx,f64 qy,f64 qz,
                                       f64 tx,f64 ty,f64 tz) noexcept {
        // d = 0.5 * t * q
        return{qw,qx,qy,qz,
               -0.5*(tx*qx+ty*qy+tz*qz),
                0.5*(tx*qw+ty*qz-tz*qy),
                0.5*(-tx*qz+ty*qw+tz*qx),
                0.5*(tx*qy-ty*qx+tz*qw)};
    }
    // Transform point
    [[nodiscard]] constexpr AV3 transform_point(AV3 p) const noexcept {
        // t = 2*(d*conj(r)).xyz
        f64 tx=2*(dw*(-rx)+dx*rw-dy*rz+dz*ry);
        f64 ty=2*(dw*(-ry)+dx*rz+dy*rw-dz*rx);
        f64 tz=2*(dw*(-rz)-dx*ry+dy*rx+dz*rw);
        // Rotate p then add t
        f64 rx2=rx*rx,ry2=ry*ry,rz2=rz*rz;
        f64 rxry=rx*ry,rxrz=rx*rz,ryrz=ry*rz,rwrx=rw*rx,rwry=rw*ry,rwrz=rw*rz;
        return{(1-2*(ry2+rz2))*p.x+2*(rxry-rwrz)*p.y+2*(rxrz+rwry)*p.z+tx,
               2*(rxry+rwrz)*p.x+(1-2*(rx2+rz2))*p.y+2*(ryrz-rwrx)*p.z+ty,
               2*(rxrz-rwry)*p.x+2*(ryrz+rwrx)*p.y+(1-2*(rx2+ry2))*p.z+tz};
    }
};

// DLB (dual quaternion linear blending)
[[nodiscard]] constexpr DualQuat dqb_blend(
    const DualQuat* dqs, const f64* weights, int n) noexcept
{
    DualQuat result{};
    // Flip sign to ensure shortest path
    for(int i=0;i<n;++i) {
        DualQuat dq=dqs[i];
        if(i>0&&(dq.rw*dqs[0].rw+dq.rx*dqs[0].rx+dq.ry*dqs[0].ry+dq.rz*dqs[0].rz)<0)
            dq=dq*(-1.0);
        result=result+dq*weights[i];
    }
    return result.normalize();
}

// ─── Cubic Hermite spline for animation curves ────────────────────────────────
[[nodiscard]] constexpr f64 hermite_interp(f64 p0,f64 p1,f64 m0,f64 m1,f64 t) noexcept {
    f64 t2=t*t,t3=t2*t;
    return (2*t3-3*t2+1)*p0+(t3-2*t2+t)*m0+(-2*t3+3*t2)*p1+(t3-t2)*m1;
}
[[nodiscard]] constexpr f64 hermite_deriv(f64 p0,f64 p1,f64 m0,f64 m1,f64 t) noexcept {
    f64 t2=t*t;
    return (6*t2-6*t)*p0+(3*t2-4*t+1)*m0+(-6*t2+6*t)*p1+(3*t2-2*t)*m1;
}

// Catmull-Rom spline (auto-tangent)
[[nodiscard]] constexpr f64 catmull_rom(f64 p0,f64 p1,f64 p2,f64 p3,f64 t) noexcept {
    f64 m1=(p2-p0)*0.5, m2=(p3-p1)*0.5;
    return hermite_interp(p1,p2,m1,m2,t);
}

// ─── Blend tree weights (normalized radial) ───────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> blend_weights_radial(
    const std::array<f64,N>& sample_dirs, // 1D angles or blendspace x values
    f64 query) noexcept
{
    std::array<f64,N> w{};
    f64 total=0;
    for(int i=0;i<N;++i) {
        f64 d=std::abs(query-sample_dirs[i]);
        w[i]=d>1e-6?1.0/d:1e6;
        total+=w[i];
    }
    if(total>1e-12) for(auto& v:w) v/=total;
    return w;
}

// ─── 2D blendspace (bilinear interpolation in parameter space) ────────────────
template<int NX, int NY>
[[nodiscard]] constexpr std::array<f64,NX*NY> blendspace_2d_weights(
    const std::array<f64,NX>& xs, const std::array<f64,NY>& ys,
    f64 qx, f64 qy) noexcept
{
    std::array<f64,NX*NY> w{};
    // Find bounding cell
    int ix=0,iy=0;
    for(int i=0;i<NX-1;++i) if(qx>=xs[i]&&qx<=xs[i+1])ix=i;
    for(int j=0;j<NY-1;++j) if(qy>=ys[j]&&qy<=ys[j+1])iy=j;
    f64 tx=(xs[ix+1]-xs[ix])>1e-12?(qx-xs[ix])/(xs[ix+1]-xs[ix]):0;
    f64 ty=(ys[iy+1]-ys[iy])>1e-12?(qy-ys[iy])/(ys[iy+1]-ys[iy]):0;
    w[iy*NX+ix]    =(1-tx)*(1-ty);
    w[iy*NX+ix+1]  = tx*(1-ty);
    w[(iy+1)*NX+ix]=(1-tx)*ty;
    w[(iy+1)*NX+ix+1]=tx*ty;
    return w;
}

// ─── Inverse kinematics (CCD, 2-bone analytical) ─────────────────────────────
// 2-bone IK (arm: shoulder→elbow→wrist)
struct TwoBoneIK {
    f64 l1,l2; // upper and lower bone lengths
    [[nodiscard]] constexpr std::pair<f64,f64> solve(f64 target_dist) const noexcept {
        // Elbow angle via law of cosines
        f64 d=std::min(target_dist,l1+l2-1e-6);
        f64 cos_e=(l1*l1+l2*l2-d*d)/(2*l1*l2);
        cos_e=std::max(-1.0,std::min(1.0,cos_e));
        f64 elbow_angle=std::acos(cos_e);
        // Shoulder angle
        f64 cos_s=(l1*l1+d*d-l2*l2)/(2*l1*d);
        cos_s=std::max(-1.0,std::min(1.0,cos_s));
        f64 shoulder_angle=std::acos(cos_s);
        return{shoulder_angle,elbow_angle};
    }
};

// CCD IK step (one iteration for one joint)
[[nodiscard]] constexpr f64 ccd_joint_rotation(
    AV3 joint_pos, AV3 end_effector, AV3 target) noexcept
{
    AV3 to_ee=(end_effector-joint_pos).norm();
    AV3 to_tgt=(target-joint_pos).norm();
    f64 dot=std::max(-1.0,std::min(1.0,to_ee.dot(to_tgt)));
    return std::acos(dot);
}

// ─── Motion retargeting (proportional) ───────────────────────────────────────
[[nodiscard]] constexpr f64 retarget_bone_length(
    f64 src_length, f64 dst_length, f64 angle) noexcept
{
    // Scale translation component proportionally
    return angle; // rotation preserved, translation scaled by ratio
}
[[nodiscard]] constexpr f64 bone_length_ratio(f64 src, f64 dst) noexcept {
    return src>1e-12?dst/src:1.0;
}

// ─── Spring dynamics (secondary motion) ──────────────────────────────────────
struct SpringBone {
    AV3 position{}, velocity{};
    f64 stiffness{50},damping{5},mass{1};

    constexpr void update(AV3 target, f64 dt) noexcept {
        AV3 force=(target-position)*stiffness-velocity*damping;
        velocity=velocity+force*(dt/mass);
        position=position+velocity*dt;
    }
};
} // namespace qs::math
