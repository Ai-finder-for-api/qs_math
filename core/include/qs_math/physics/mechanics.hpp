#pragma once
// qs_math/physics/mechanics.hpp — Rigid body, constraints, contact, multibody dynamics
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── 3D vector (reusing pattern) ─────────────────────────────────────────────
struct V3m { f64 x,y,z;
    [[nodiscard]] constexpr V3m operator+(V3m b) const noexcept{return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr V3m operator-(V3m b) const noexcept{return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr V3m operator*(f64 s) const noexcept{return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64  dot(V3m b) const noexcept{return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr V3m  cross(V3m b) const noexcept{return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    [[nodiscard]] constexpr f64  len2() const noexcept{return dot(*this);}
    [[nodiscard]]           f64  len()  const noexcept{return std::sqrt(len2());}
    [[nodiscard]]           V3m  norm() const noexcept{f64 l=len();return l>1e-12?V3m{x/l,y/l,z/l}:*this;}
};

// ─── Quaternion (unit = rotation) ────────────────────────────────────────────
struct Quat { f64 w,x,y,z;
    [[nodiscard]] constexpr Quat operator*(Quat b) const noexcept {
        return{w*b.w-x*b.x-y*b.y-z*b.z,
               w*b.x+x*b.w+y*b.z-z*b.y,
               w*b.y-x*b.z+y*b.w+z*b.x,
               w*b.z+x*b.y-y*b.x+z*b.w};
    }
    [[nodiscard]] constexpr Quat conj() const noexcept{return{w,-x,-y,-z};}
    [[nodiscard]] constexpr f64  mag2() const noexcept{return w*w+x*x+y*y+z*z;}
    [[nodiscard]]           Quat norm() const noexcept{f64 m=std::sqrt(mag2());return m>1e-12?Quat{w/m,x/m,y/m,z/m}:*this;}
    [[nodiscard]] constexpr V3m  rotate(V3m v) const noexcept {
        Quat p={0,v.x,v.y,v.z};
        Quat r=(*this)*p*conj();
        return{r.x,r.y,r.z};
    }
    static [[nodiscard]] constexpr Quat from_axis_angle(V3m axis, f64 angle) noexcept {
        f64 s=std::sin(angle*0.5); axis=axis.norm();
        return{std::cos(angle*0.5),axis.x*s,axis.y*s,axis.z*s};
    }
    // SLERP
    [[nodiscard]] constexpr Quat slerp(Quat b, f64 t) const noexcept {
        f64 dot=w*b.w+x*b.x+y*b.y+z*b.z;
        if(dot<0){b={-b.w,-b.x,-b.y,-b.z};dot=-dot;}
        if(dot>0.9995f){return Quat{w+(b.w-w)*t,x+(b.x-x)*t,y+(b.y-y)*t,z+(b.z-z)*t}.norm();}
        f64 theta=std::acos(dot),s0=std::sin((1-t)*theta),s1=std::sin(t*theta),d=std::sin(theta);
        return Quat{(w*s0+b.w*s1)/d,(x*s0+b.x*s1)/d,(y*s0+b.y*s1)/d,(z*s0+b.z*s1)/d};
    }
    // Euler angles (ZYX convention)
    [[nodiscard]] constexpr std::array<f64,3> to_euler() const noexcept {
        f64 roll=std::atan2(2*(w*x+y*z),1-2*(x*x+y*y));
        f64 pitch=std::asin(std::max(-1.0,std::min(1.0,2*(w*y-z*x))));
        f64 yaw=std::atan2(2*(w*z+x*y),1-2*(y*y+z*z));
        return{roll,pitch,yaw};
    }
};

// ─── 3×3 inertia tensor ──────────────────────────────────────────────────────
struct InertiaTensor {
    f64 Ixx,Iyy,Izz,Ixy,Ixz,Iyz;
    // For solid box: I_xx=(m/12)*(y²+z²), etc.
    static constexpr InertiaTensor solid_box(f64 m,f64 lx,f64 ly,f64 lz) noexcept {
        return{m/12*(ly*ly+lz*lz),m/12*(lx*lx+lz*lz),m/12*(lx*lx+ly*ly),0,0,0};
    }
    static constexpr InertiaTensor solid_sphere(f64 m, f64 r) noexcept {
        f64 I=2.0/5*m*r*r; return{I,I,I,0,0,0};
    }
    static constexpr InertiaTensor solid_cylinder(f64 m, f64 r, f64 h) noexcept {
        f64 Iz=0.5*m*r*r,Ix=m*(3*r*r+h*h)/12;
        return{Ix,Ix,Iz,0,0,0};
    }
    // Apply: Iω (angular momentum)
    [[nodiscard]] constexpr V3m apply(V3m omega) const noexcept {
        return{Ixx*omega.x-Ixy*omega.y-Ixz*omega.z,
              -Ixy*omega.x+Iyy*omega.y-Iyz*omega.z,
              -Ixz*omega.x-Iyz*omega.y+Izz*omega.z};
    }
    // Parallel axis theorem: I_cm + m*d²
    [[nodiscard]] constexpr InertiaTensor parallel_axis(f64 m, V3m d) const noexcept {
        return{Ixx+m*(d.y*d.y+d.z*d.z),Iyy+m*(d.x*d.x+d.z*d.z),Izz+m*(d.x*d.x+d.y*d.y),
               Ixy-m*d.x*d.y,Ixz-m*d.x*d.z,Iyz-m*d.y*d.z};
    }
};

// ─── Rigid body state ─────────────────────────────────────────────────────────
struct RigidBody {
    f64 mass{1};
    InertiaTensor I{};
    V3m pos{},vel{},force{},torque{};
    Quat orient{1,0,0,0};
    V3m omega{};  // angular velocity (body frame)

    // Equations of motion (Newton-Euler)
    constexpr void integrate(f64 dt) noexcept {
        // Linear
        V3m acc=force*(1.0/mass);
        vel=vel+acc*dt;
        pos=pos+vel*dt;
        // Angular (Euler's equations in body frame)
        V3m L=I.apply(omega);
        V3m alpha_body={
            (torque.x-omega.y*L.z+omega.z*L.y)/I.Ixx,
            (torque.y-omega.z*L.x+omega.x*L.z)/I.Iyy,
            (torque.z-omega.x*L.y+omega.y*L.x)/I.Izz
        };
        omega=omega+alpha_body*dt;
        // Update quaternion: q̇ = 0.5 * q * ω_quat
        Quat omega_q{0,omega.x,omega.y,omega.z};
        Quat dq=orient*omega_q;
        orient={orient.w+0.5*dt*dq.w,orient.x+0.5*dt*dq.x,
                orient.y+0.5*dt*dq.y,orient.z+0.5*dt*dq.z};
        orient=orient.norm();
        force={}; torque={};
    }

    // Apply world-space impulse at world-space point
    constexpr void apply_impulse(V3m J, V3m pt) noexcept {
        vel=vel+J*(1.0/mass);
        V3m r=pt-pos;
        V3m tau=r.cross(J); // world-space torque impulse
        // Convert to body frame
        V3m tau_body=orient.conj().rotate(tau);
        omega=omega+V3m{tau_body.x/I.Ixx,tau_body.y/I.Iyy,tau_body.z/I.Izz};
    }

    // Velocity of a point on the body (world frame)
    [[nodiscard]] constexpr V3m point_velocity(V3m world_pt) const noexcept {
        V3m r=world_pt-pos;
        V3m omega_world=orient.rotate(omega);
        return vel+omega_world.cross(r);
    }
};

// ─── Contact / collision response ─────────────────────────────────────────────
struct ContactPoint { V3m pos,normal; f64 depth; };

[[nodiscard]] constexpr f64 collision_impulse(
    const RigidBody& A, const RigidBody& B,
    const ContactPoint& c, f64 restitution=0.5) noexcept
{
    V3m vA=A.point_velocity(c.pos), vB=B.point_velocity(c.pos);
    f64 vrel=(vA-vB).dot(c.normal);
    if(vrel>0) return 0; // separating
    V3m rA=c.pos-A.pos, rB=c.pos-B.pos;
    V3m omA=A.orient.rotate(rA).cross(c.normal);
    V3m omB=B.orient.rotate(rB).cross(c.normal);
    f64 inv_eff_mass=1.0/A.mass+1.0/B.mass
        +V3m{omA.x/A.I.Ixx,omA.y/A.I.Iyy,omA.z/A.I.Izz}.dot(omA)
        +V3m{omB.x/B.I.Ixx,omB.y/B.I.Iyy,omB.z/B.I.Izz}.dot(omB);
    return -(1+restitution)*vrel/inv_eff_mass;
}

// ─── Spring-damper constraint ─────────────────────────────────────────────────
struct SpringDamper { f64 k,c,rest_len;
    [[nodiscard]] constexpr f64 force(f64 len, f64 vel) const noexcept {
        return -k*(len-rest_len)-c*vel;
    }
};

// ─── Jacobian-based constraint solver (MLCP, simple projection) ───────────────
// Ball joint: positional error = posA + rA - posB - rB = 0
[[nodiscard]] constexpr V3m ball_joint_error(
    V3m posA, Quat orientA, V3m rA_local,
    V3m posB, Quat orientB, V3m rB_local) noexcept
{
    return posA+orientA.rotate(rA_local)-posB-orientB.rotate(rB_local);
}

// ─── Kinematic chains (Denavit-Hartenberg) ────────────────────────────────────
struct DHParams { f64 a,alpha,d,theta; };

// DH matrix for a link
[[nodiscard]] constexpr std::array<std::array<f64,4>,4> dh_matrix(DHParams p) noexcept {
    f64 ca=std::cos(p.alpha),sa=std::sin(p.alpha);
    f64 ct=std::cos(p.theta),st=std::sin(p.theta);
    return{{
        {ct,-st*ca,st*sa,p.a*ct},
        {st,ct*ca,-ct*sa,p.a*st},
        {0,sa,ca,p.d},
        {0,0,0,1}
    }};
}

// Multiply two 4×4 homogeneous matrices
[[nodiscard]] constexpr std::array<std::array<f64,4>,4>
h_mul(const std::array<std::array<f64,4>,4>& A,
      const std::array<std::array<f64,4>,4>& B) noexcept {
    std::array<std::array<f64,4>,4> C{};
    for(int i=0;i<4;++i) for(int k=0;k<4;++k) if(A[i][k])
        for(int j=0;j<4;++j) C[i][j]+=A[i][k]*B[k][j];
    return C;
}

template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,4>,4>
forward_kinematics(const std::array<DHParams,N>& links) noexcept {
    auto T=dh_matrix(links[0]);
    for(int i=1;i<N;++i) T=h_mul(T,dh_matrix(links[i]));
    return T;
}
} // namespace qs::math
