#pragma once
// qs_math/physics/rigid_body.hpp — Rigid body dynamics: inertia tensors,
//   impulse resolution, constraints (joints), torque/force integration,
//   contact manifolds, restitution, friction (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/mat3.hpp"
#include "../linear_algebra/quat.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <cmath>

namespace qs::math::physics {

// ─── Inertia tensors (solid primitives) ──────────────────────────────────
// All return world-space 3x3 inertia tensor (symmetric, as 6 unique values)
// Stored: {Ixx, Iyy, Izz, Ixy, Ixz, Iyz}
struct InertiaTensor {
    f32 Ixx,Iyy,Izz,Ixy,Ixz,Iyz;

    // Apply parallel axis theorem: shift by offset d from CoM
    [[nodiscard]] InertiaTensor shift(Vec3f d, f32 mass) const noexcept {
        return {
            Ixx+mass*(d.y*d.y+d.z*d.z), Iyy+mass*(d.x*d.x+d.z*d.z),
            Izz+mass*(d.x*d.x+d.y*d.y),
            Ixy-mass*d.x*d.y, Ixz-mass*d.x*d.z, Iyz-mass*d.y*d.z
        };
    }

    [[nodiscard]] InertiaTensor operator+(const InertiaTensor& o) const noexcept {
        return {Ixx+o.Ixx,Iyy+o.Iyy,Izz+o.Izz,Ixy+o.Ixy,Ixz+o.Ixz,Iyz+o.Iyz};
    }

    // Multiply omega: I * omega (angular momentum from angular velocity)
    [[nodiscard]] Vec3f apply(Vec3f w) const noexcept {
        return {Ixx*w.x+Ixy*w.y+Ixz*w.z,
                Ixy*w.x+Iyy*w.y+Iyz*w.z,
                Ixz*w.x+Iyz*w.y+Izz*w.z};
    }

    // 3x3 inverse (analytic, for diagonal or full tensor)
    [[nodiscard]] InertiaTensor inverse() const noexcept {
        f32 det = Ixx*(Iyy*Izz-Iyz*Iyz)
                 -Ixy*(Ixy*Izz-Iyz*Ixz)
                 +Ixz*(Ixy*Iyz-Iyy*Ixz);
        if(std::abs(det)<1e-30f) return {1e30f,1e30f,1e30f,0,0,0};
        f32 inv=1.f/det;
        return {
            (Iyy*Izz-Iyz*Iyz)*inv,
            (Ixx*Izz-Ixz*Ixz)*inv,
            (Ixx*Iyy-Ixy*Ixy)*inv,
           -(Ixy*Izz-Ixz*Iyz)*inv,
            (Ixy*Iyz-Iyz*Ixz)*inv,  // note: simplified
           -(Ixx*Iyz-Ixy*Ixz)*inv
        };
    }
};

[[nodiscard]] inline InertiaTensor inertia_sphere(f32 mass, f32 radius) noexcept {
    f32 I=0.4f*mass*radius*radius;
    return {I,I,I,0,0,0};
}
[[nodiscard]] inline InertiaTensor inertia_box(f32 mass, Vec3f half) noexcept {
    f32 f=mass/3.f;
    return {f*(half.y*half.y+half.z*half.z)*4.f,
            f*(half.x*half.x+half.z*half.z)*4.f,
            f*(half.x*half.x+half.y*half.y)*4.f, 0,0,0};
}
[[nodiscard]] inline InertiaTensor inertia_cylinder(f32 mass, f32 radius, f32 height) noexcept {
    f32 Iyz=mass*(3.f*radius*radius+height*height)/12.f;
    return {0.5f*mass*radius*radius, Iyz, Iyz, 0,0,0};
}
[[nodiscard]] inline InertiaTensor inertia_cone(f32 mass, f32 radius, f32 height) noexcept {
    f32 Ix=3.f*mass*radius*radius/10.f;
    f32 Iyz=3.f*mass*(radius*radius/4.f+height*height)/5.f;
    return {Ix, Iyz, Iyz, 0,0,0};
}
[[nodiscard]] inline InertiaTensor inertia_capsule(f32 mass, f32 radius, f32 cyl_height) noexcept {
    // Solid cylinder + 2 hemispheres
    f32 mc=mass*(cyl_height/(cyl_height+4.f*radius/3.f));
    f32 ms=mass-mc;
    auto Ic=inertia_cylinder(mc,radius,cyl_height);
    auto Is=inertia_sphere(ms,radius);
    // Shift hemispheres to ends
    f32 shift=cyl_height/2.f+3.f*radius/8.f;
    return {Ic.Ixx+Is.Ixx,
            Ic.Iyy+Is.shift({0,shift,0},ms*0.5f).Iyy+Is.shift({0,-shift,0},ms*0.5f).Iyy,
            Ic.Izz+Is.Izz, 0,0,0};
}

// ─── Rigid body state ─────────────────────────────────────────────────────
struct RigidBody {
    // Primary state
    Vec3f position    = {0,0,0};
    Quatf orientation = Quatf::identity();
    Vec3f linear_mom  = {0,0,0}; // p = m*v
    Vec3f angular_mom = {0,0,0}; // L = I*omega

    // Constant properties
    f32   mass        = 1.f;
    f32   inv_mass    = 1.f;
    InertiaTensor body_inertia   = inertia_box(1.f,{0.5f,0.5f,0.5f});
    InertiaTensor body_inv_inertia;

    // Derived (updated each step)
    Vec3f velocity    = {0,0,0};
    Vec3f omega       = {0,0,0}; // angular velocity

    // Accumulated forces/torques this step
    Vec3f force_accum  = {0,0,0};
    Vec3f torque_accum = {0,0,0};

    // Material
    f32 restitution = 0.3f;
    f32 friction    = 0.5f;
    bool is_static  = false;

    void init() noexcept {
        inv_mass = is_static ? 0.f : 1.f/mass;
        body_inv_inertia = is_static ? InertiaTensor{0,0,0,0,0,0} : body_inertia.inverse();
    }

    // World-space inverse inertia tensor (rotated by orientation)
    [[nodiscard]] InertiaTensor world_inv_inertia() const noexcept {
        // R * I_body_inv * R^T  (diagonal case simplification)
        Mat3f R = orientation.to_mat3();
        // Full tensor rotation: I_world_inv = R * I_body_inv * R^T
        InertiaTensor ii = body_inv_inertia;
        f32 m[3][3]={{ii.Ixx,ii.Ixy,ii.Ixz},{ii.Ixy,ii.Iyy,ii.Iyz},{ii.Ixz,ii.Iyz,ii.Izz}};
        f32 r[3][3]={{R.c[0].x,R.c[1].x,R.c[2].x},{R.c[0].y,R.c[1].y,R.c[2].y},{R.c[0].z,R.c[1].z,R.c[2].z}};
        // T = R * m
        f32 T[3][3]={};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) T[i][j]+=r[i][k]*m[k][j];
        // W = T * R^T
        f32 W[3][3]={};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) W[i][j]+=T[i][k]*r[j][k];
        return {W[0][0],W[1][1],W[2][2],W[0][1],W[0][2],W[1][2]};
    }

    void apply_force(Vec3f f) noexcept { force_accum=force_accum+f; }
    void apply_torque(Vec3f t) noexcept { torque_accum=torque_accum+t; }
    void apply_force_at_point(Vec3f f, Vec3f world_pt) noexcept {
        force_accum=force_accum+f;
        torque_accum=torque_accum+cross(world_pt-position, f);
    }

    void clear_forces() noexcept { force_accum={0,0,0}; torque_accum={0,0,0}; }

    // Integrate state (symplectic Euler)
    void integrate(f32 dt) noexcept {
        if(is_static) return;
        // Update momentum
        linear_mom  = linear_mom  + force_accum  * dt;
        angular_mom = angular_mom + torque_accum * dt;
        // Derive velocity
        velocity = linear_mom * inv_mass;
        auto Iinv = world_inv_inertia();
        omega = Iinv.apply(angular_mom);
        // Update position/orientation
        position = position + velocity * dt;
        // dq/dt = 0.5 * [omega,0] * q
        Quatf dq = Quatf{omega.x*0.5f*dt, omega.y*0.5f*dt, omega.z*0.5f*dt, 0.f} * orientation;
        orientation = Quatf{
            orientation.x+dq.x, orientation.y+dq.y,
            orientation.z+dq.z, orientation.w+dq.w}.normalized();
        clear_forces();
    }
};

// ─── Contact point ────────────────────────────────────────────────────────
struct ContactPoint {
    Vec3f point;        // world-space contact point
    Vec3f normal;       // from B into A (unit)
    f32   penetration;  // depth of overlap
};

// ─── Impulse-based collision resolution ───────────────────────────────────
// Resolves a single contact between two rigid bodies.
inline void resolve_contact(RigidBody& A, RigidBody& B, const ContactPoint& c) noexcept {
    Vec3f rA = c.point - A.position;
    Vec3f rB = c.point - B.position;
    Vec3f vA = A.velocity + cross(A.omega, rA);
    Vec3f vB = B.velocity + cross(B.omega, rB);
    Vec3f vrel = vA - vB;
    f32 vn = dot(vrel, c.normal);
    if(vn > 0) return; // separating

    f32 e = min(A.restitution, B.restitution);
    auto IinvA = A.world_inv_inertia();
    auto IinvB = B.world_inv_inertia();

    // Effective mass
    Vec3f rAn = cross(rA, c.normal);
    Vec3f rBn = cross(rB, c.normal);
    f32 eff_mass = A.inv_mass + B.inv_mass
        + dot(IinvA.apply(rAn), rAn)
        + dot(IinvB.apply(rBn), rBn);

    f32 j = -(1.f+e)*vn / eff_mass;
    Vec3f impulse = c.normal * j;

    A.linear_mom  = A.linear_mom  + impulse;
    B.linear_mom  = B.linear_mom  - impulse;
    A.angular_mom = A.angular_mom + cross(rA, impulse);
    B.angular_mom = B.angular_mom - cross(rB, impulse);
    A.velocity = A.linear_mom * A.inv_mass;
    B.velocity = B.linear_mom * B.inv_mass;
    A.omega = IinvA.apply(A.angular_mom);
    B.omega = IinvB.apply(B.angular_mom);

    // Friction impulse (Coulomb)
    Vec3f vrel2 = (A.velocity+cross(A.omega,rA)) - (B.velocity+cross(B.omega,rB));
    Vec3f vt = vrel2 - c.normal*dot(vrel2,c.normal);
    f32 vt_len = vt.length();
    if(vt_len < 1e-6f) return;
    Vec3f t = vt*(1.f/vt_len);
    Vec3f rAt = cross(rA,t), rBt = cross(rB,t);
    f32 eff_t = A.inv_mass+B.inv_mass+dot(IinvA.apply(rAt),rAt)+dot(IinvB.apply(rBt),rBt);
    f32 jt = -vt_len/eff_t;
    f32 mu = std::sqrt(A.friction*B.friction);
    jt = clamp(jt, -mu*j, mu*j);
    Vec3f ft = t*jt;
    A.linear_mom  = A.linear_mom  + ft;
    B.linear_mom  = B.linear_mom  - ft;
    A.angular_mom = A.angular_mom + cross(rA,ft);
    B.angular_mom = B.angular_mom - cross(rB,ft);
    A.velocity=A.linear_mom*A.inv_mass; B.velocity=B.linear_mom*B.inv_mass;
    A.omega=IinvA.apply(A.angular_mom); B.omega=IinvB.apply(B.angular_mom);
}

// ─── Kinetic energy ───────────────────────────────────────────────────────
[[nodiscard]] inline f32 kinetic_energy(const RigidBody& b) noexcept {
    f32 ke_lin = 0.5f*b.mass*dot(b.velocity,b.velocity);
    InertiaTensor Iw = b.world_inv_inertia(); // world inv I
    // KE_rot = 0.5 * L^T * I_inv * L
    Vec3f IL = Iw.apply(b.angular_mom); // I^{-1} L = omega
    f32 ke_rot = 0.5f*dot(b.angular_mom, IL);
    return ke_lin+ke_rot;
}

// ─── Distance joint (ball-socket constraint) ──────────────────────────────
struct BallSocketJoint {
    RigidBody* A;
    RigidBody* B;
    Vec3f localA; // attachment point in A's local frame
    Vec3f localB;
    f32 stiffness = 100.f;
    f32 damping   = 0.1f;

    void apply(f32 dt) noexcept {
        Mat3f RA = A->orientation.to_mat3();
        Mat3f RB = B->orientation.to_mat3();
        Vec3f wA = A->position + RA*localA;
        Vec3f wB = B->position + RB*localB;
        Vec3f err = wA - wB;
        Vec3f vA = A->velocity + cross(A->omega, wA-A->position);
        Vec3f vB = B->velocity + cross(B->omega, wB-B->position);
        Vec3f verr = vA - vB;
        Vec3f force = err*(-stiffness) - verr*damping;
        A->apply_force_at_point(-force, wA);
        B->apply_force_at_point( force, wB);
    }
};

// ─── Hinge joint (1-DOF rotation constraint) ──────────────────────────────
struct HingeJoint {
    RigidBody* A; RigidBody* B;
    Vec3f localA, localB;    // anchor points
    Vec3f axisA, axisB;      // hinge axis in local frames
    f32 lo_limit = -PI, hi_limit = PI;
    f32 stiffness=50.f, damping=0.05f;

    void apply(f32 /*dt*/) noexcept {
        Mat3f RA=A->orientation.to_mat3(), RB=B->orientation.to_mat3();
        Vec3f wa=A->position+RA*localA, wb=B->position+RB*localB;
        Vec3f err=wa-wb;
        A->apply_force_at_point(-err*stiffness, wa);
        B->apply_force_at_point( err*stiffness, wb);
        // Angular constraint: keep axes aligned (simplified)
        Vec3f axA=RA*axisA, axB=RB*axisB;
        Vec3f ang_err=cross(axA,axB);
        A->apply_torque(-ang_err*stiffness*0.5f);
        B->apply_torque( ang_err*stiffness*0.5f);
    }
};

// ─── World gravity integration ────────────────────────────────────────────
inline void apply_gravity(std::vector<RigidBody>& bodies, Vec3f g={0,-9.81f,0}) noexcept {
    for(auto& b:bodies) if(!b.is_static) b.apply_force(g*b.mass);
}

inline void integrate_all(std::vector<RigidBody>& bodies, f32 dt) noexcept {
    for(auto& b:bodies) b.integrate(dt);
}

} // namespace qs::math::physics
