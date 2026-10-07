#pragma once
// qs_math/engineering/kinematics.hpp — Robot kinematics, trajectory planning, workspace analysis
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── SE(3) homogeneous transform ──────────────────────────────────────────────
struct Transform4 {
    f64 m[4][4]{};
    static constexpr Transform4 identity() noexcept {
        Transform4 T; T.m[0][0]=T.m[1][1]=T.m[2][2]=T.m[3][3]=1; return T;
    }
    [[nodiscard]] constexpr Transform4 operator*(const Transform4& B) const noexcept {
        Transform4 C;
        for(int i=0;i<4;++i) for(int k=0;k<4;++k) if(m[i][k])
            for(int j=0;j<4;++j) C.m[i][j]+=m[i][k]*B.m[k][j];
        return C;
    }
    [[nodiscard]] constexpr std::array<f64,3> apply(f64 x,f64 y,f64 z) const noexcept {
        f64 w=m[3][0]*x+m[3][1]*y+m[3][2]*z+m[3][3];
        return{(m[0][0]*x+m[0][1]*y+m[0][2]*z+m[0][3])/w,
               (m[1][0]*x+m[1][1]*y+m[1][2]*z+m[1][3])/w,
               (m[2][0]*x+m[2][1]*y+m[2][2]*z+m[2][3])/w};
    }
    // Extract position
    [[nodiscard]] constexpr std::array<f64,3> translation() const noexcept {
        return{m[0][3],m[1][3],m[2][3]};
    }
    // Inverse (for rigid transforms: T^{-1} = [R^T, -R^T*t; 0,0,0,1])
    [[nodiscard]] constexpr Transform4 inverse() const noexcept {
        Transform4 inv;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) inv.m[i][j]=m[j][i];
        for(int i=0;i<3;++i) {
            inv.m[i][3]=0;
            for(int k=0;k<3;++k) inv.m[i][3]-=inv.m[i][k]*m[k][3];
        }
        inv.m[3][3]=1;
        return inv;
    }
};

// Rotation matrix from axis-angle
[[nodiscard]] constexpr Transform4 rot_axis_angle(f64 nx,f64 ny,f64 nz,f64 theta) noexcept {
    f64 c=std::cos(theta),s=std::sin(theta),t=1-c;
    Transform4 R=Transform4::identity();
    R.m[0][0]=t*nx*nx+c;     R.m[0][1]=t*nx*ny-s*nz; R.m[0][2]=t*nx*nz+s*ny;
    R.m[1][0]=t*nx*ny+s*nz;  R.m[1][1]=t*ny*ny+c;    R.m[1][2]=t*ny*nz-s*nx;
    R.m[2][0]=t*nx*nz-s*ny;  R.m[2][1]=t*ny*nz+s*nx; R.m[2][2]=t*nz*nz+c;
    return R;
}
[[nodiscard]] constexpr Transform4 rot_x(f64 t) noexcept {return rot_axis_angle(1,0,0,t);}
[[nodiscard]] constexpr Transform4 rot_y(f64 t) noexcept {return rot_axis_angle(0,1,0,t);}
[[nodiscard]] constexpr Transform4 rot_z(f64 t) noexcept {return rot_axis_angle(0,0,1,t);}
[[nodiscard]] constexpr Transform4 translate(f64 x,f64 y,f64 z) noexcept {
    auto T=Transform4::identity(); T.m[0][3]=x;T.m[1][3]=y;T.m[2][3]=z; return T;
}

// ─── Denavit-Hartenberg forward kinematics ────────────────────────────────────
struct DHLink { f64 a,alpha,d,theta; bool revolute{true}; };

[[nodiscard]] constexpr Transform4 dh_transform(const DHLink& link, f64 q) noexcept {
    f64 theta=link.revolute?link.theta+q:link.theta;
    f64 d=link.revolute?link.d:link.d+q;
    f64 ca=std::cos(link.alpha),sa=std::sin(link.alpha);
    f64 ct=std::cos(theta),st=std::sin(theta);
    Transform4 T;
    T.m[0][0]=ct;T.m[0][1]=-st*ca;T.m[0][2]=st*sa; T.m[0][3]=link.a*ct;
    T.m[1][0]=st;T.m[1][1]=ct*ca; T.m[1][2]=-ct*sa;T.m[1][3]=link.a*st;
    T.m[2][0]=0; T.m[2][1]=sa;    T.m[2][2]=ca;     T.m[2][3]=d;
    T.m[3][3]=1;
    return T;
}

template<int N>
[[nodiscard]] constexpr Transform4 fk(
    const std::array<DHLink,N>& links,
    const std::array<f64,N>& q) noexcept
{
    Transform4 T=Transform4::identity();
    for(int i=0;i<N;++i) T=T*dh_transform(links[i],q[i]);
    return T;
}

// ─── Jacobian (numerical, spatial) ───────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,6> jacobian_numerical(
    const std::array<DHLink,N>& links,
    const std::array<f64,N>& q, f64 h=1e-6) noexcept
{
    std::array<std::array<f64,N>,6> J{};
    auto T0=fk(links,q);
    auto p0=T0.translation();
    for(int i=0;i<N;++i) {
        auto qp=q; qp[i]+=h;
        auto Tp=fk(links,qp);
        auto pp=Tp.translation();
        // Linear velocity (position difference)
        J[0][i]=(pp[0]-p0[0])/h;
        J[1][i]=(pp[1]-p0[1])/h;
        J[2][i]=(pp[2]-p0[2])/h;
        // Angular velocity: z-axis of joint i × (p_e - p_i)
        auto Ti=fk<N>(links,q); // T up to joint i
        // Simplified: use z-axis of frame i
        J[3][i]=Ti.m[2][0]/h;
        J[4][i]=Ti.m[2][1]/h;
        J[5][i]=Ti.m[2][2]/h;
    }
    return J;
}

// Manipulability measure: w = sqrt(det(J*J'))
template<int N>
[[nodiscard]] constexpr f64 manipulability(
    const std::array<std::array<f64,N>,6>& J) noexcept
{
    // Compute J*J' (6×6)
    std::array<std::array<f64,6>,6> JJt{};
    for(int i=0;i<6;++i) for(int j=0;j<6;++j)
        for(int k=0;k<N;++k) JJt[i][j]+=J[i][k]*J[j][k];
    // Simplified: trace as approximation of manipulability
    f64 tr=0; for(int i=0;i<6;++i) tr+=JJt[i][i];
    return std::sqrt(tr);
}

// ─── Trajectory planning ──────────────────────────────────────────────────────
// Cubic polynomial: q(t)=a0+a1*t+a2*t²+a3*t³
struct CubicTraj {
    f64 a0,a1,a2,a3;
    [[nodiscard]] constexpr f64 pos(f64 t) const noexcept{return a0+t*(a1+t*(a2+t*a3));}
    [[nodiscard]] constexpr f64 vel(f64 t) const noexcept{return a1+t*(2*a2+3*t*a3);}
    [[nodiscard]] constexpr f64 acc(f64 t) const noexcept{return 2*a2+6*t*a3;}
};
// Boundary conditions: q(0)=q0,q(T)=q1,q'(0)=v0,q'(T)=v1
[[nodiscard]] constexpr CubicTraj cubic_traj(
    f64 q0,f64 q1,f64 v0,f64 v1,f64 T) noexcept
{
    f64 a0=q0,a1=v0;
    f64 a2=(3*(q1-q0)-T*(2*v0+v1))/(T*T);
    f64 a3=(2*(q0-q1)+T*(v0+v1))/(T*T*T);
    return{a0,a1,a2,a3};
}

// Quintic polynomial (zero velocity AND acceleration at endpoints)
struct QuinticTraj { f64 c[6]{}; 
    [[nodiscard]] constexpr f64 pos(f64 t) const noexcept{
        f64 r=c[0]; f64 tn=t;
        for(int i=1;i<6;++i){r+=c[i]*tn;tn*=t;} return r;}
    [[nodiscard]] constexpr f64 vel(f64 t) const noexcept{
        f64 r=c[1]+2*c[2]*t+3*c[3]*t*t+4*c[4]*t*t*t+5*c[5]*t*t*t*t; return r;}
};
[[nodiscard]] constexpr QuinticTraj quintic_traj(f64 q0,f64 q1,f64 T) noexcept {
    f64 T3=T*T*T,T4=T3*T,T5=T4*T;
    return{{q0,0,0,10*(q1-q0)/T3,-15*(q1-q0)/T4,6*(q1-q0)/T5}};
}

// ─── Trapezoidal velocity profile ─────────────────────────────────────────────
struct TrapVelProfile {
    f64 q0,q1,v_max,a_max;
    f64 t_acc,t_const,t_total;

    constexpr TrapVelProfile(f64 _q0,f64 _q1,f64 _vmax,f64 _amax) noexcept
        :q0(_q0),q1(_q1),v_max(_vmax),a_max(_amax)
    {
        f64 d=std::abs(q1-q0);
        t_acc=v_max/a_max;
        f64 d_acc=0.5*a_max*t_acc*t_acc;
        if(2*d_acc>d){t_acc=std::sqrt(d/a_max);t_const=0;}
        else t_const=(d-2*d_acc)/v_max;
        t_total=2*t_acc+t_const;
    }
    [[nodiscard]] constexpr f64 pos(f64 t) const noexcept {
        f64 sign=(q1>q0)?1:-1,d=std::abs(q1-q0);
        if(t<=t_acc) return q0+sign*0.5*a_max*t*t;
        if(t<=t_acc+t_const) return q0+sign*(0.5*a_max*t_acc*t_acc+v_max*(t-t_acc));
        f64 dt=t_total-t;
        return q1-sign*0.5*a_max*dt*dt;
    }
    [[nodiscard]] constexpr f64 vel(f64 t) const noexcept {
        f64 sign=(q1>q0)?1:-1;
        if(t<=t_acc) return sign*a_max*t;
        if(t<=t_acc+t_const) return sign*v_max;
        return sign*a_max*(t_total-t);
    }
};

// ─── Workspace analysis ───────────────────────────────────────────────────────
// Reachable workspace for 2-link planar robot: annular region
[[nodiscard]] constexpr bool in_workspace_2link(
    f64 px, f64 py, f64 l1, f64 l2) noexcept
{
    f64 r2=px*px+py*py;
    f64 r_max=l1+l2, r_min=std::abs(l1-l2);
    return r2<=r_max*r_max&&r2>=r_min*r_min;
}
// 2-link IK: elbow up solution
[[nodiscard]] constexpr std::pair<f64,f64> ik_2link(
    f64 px,f64 py,f64 l1,f64 l2,bool elbow_up=true) noexcept
{
    f64 r2=px*px+py*py;
    f64 c2=(r2-l1*l1-l2*l2)/(2*l1*l2);
    c2=std::max(-1.0,std::min(1.0,c2));
    f64 theta2=elbow_up?-std::acos(c2):std::acos(c2);
    f64 theta1=std::atan2(py,px)-std::atan2(l2*std::sin(theta2),l1+l2*std::cos(theta2));
    return{theta1,theta2};
}
} // namespace qs::math
