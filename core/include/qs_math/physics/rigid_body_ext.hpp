#pragma once
// qs_math/rigid_body_ext.hpp — Articulated bodies, generalized coords, Lagrangian mechanics (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include "physics_dynamics.hpp"
#include "linear_algebra.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <cassert>

namespace qs {

// ─── Articulated rigid body chain ────────────────────────────────────────────
struct Joint {
    enum class Type : u8 { Revolute, Prismatic, Spherical, Fixed } type{Type::Revolute};
    Vec3f axis{0,1,0};     // joint axis in parent frame
    Vec3f offset{};         // joint position in parent frame
    f32   q{0};             // generalized coordinate
    f32   dq{0};            // velocity
    f32   ddq{0};           // acceleration
    f32   q_min{-f32(std::numbers::pi)}, q_max{f32(std::numbers::pi)};
    f32   damping{0.01f}, stiffness{0};
};

struct Link {
    f32   mass{1};
    Mat3f inertia{Mat3f::identity()};
    Vec3f com{};       // centre of mass in link frame
    u32   parent{~0u}; // parent link index
    u32   joint_idx{~0u};
};

struct ArticulatedBody {
    std::vector<Link>  links;
    std::vector<Joint> joints;

    [[nodiscard]] u32 dof() const {
        u32 d=0;
        for(const auto& j:joints) {
            if(j.type==Joint::Type::Spherical) d+=3;
            else if(j.type!=Joint::Type::Fixed) d+=1;
        }
        return d;
    }

    // Forward kinematics: compute world transforms for all links
    [[nodiscard]] std::vector<Mat4f> forward_kinematics() const {
        u32 n=u32(links.size());
        std::vector<Mat4f> T(n,Mat4f::identity());
        for(u32 i=0;i<n;++i) {
            if(links[i].parent==~0u) continue;
            const Joint& j=joints[links[i].joint_idx];
            Mat4f local=Mat4f::translation(j.offset);
            if(j.type==Joint::Type::Revolute)
                local=local*Mat4f::from_quat(Quatf::from_axis_angle(j.axis,j.q));
            else if(j.type==Joint::Type::Prismatic)
                local=local*Mat4f::translation(j.axis*j.q);
            T[i]=T[links[i].parent]*local;
        }
        return T;
    }

    // Jacobian of end-effector position wrt joint angles (6×n matrix: 3 pos + 3 rot)
    [[nodiscard]] DenseMatrix<f32> jacobian(u32 ee_link, f32 eps=1e-4f) const {
        u32 ndof=dof();
        DenseMatrix<f32> J(6,ndof);
        auto T=forward_kinematics();
        Vec3f ee_pos{T[ee_link].m[0][3],T[ee_link].m[1][3],T[ee_link].m[2][3]};
        ArticulatedBody perturbed=*this;
        u32 col=0;
        for(u32 ji=0;ji<u32(joints.size());++ji) {
            if(joints[ji].type==Joint::Type::Fixed) continue;
            perturbed.joints[ji].q+=eps;
            auto Tp=perturbed.forward_kinematics();
            Vec3f ep{Tp[ee_link].m[0][3],Tp[ee_link].m[1][3],Tp[ee_link].m[2][3]};
            Vec3f dpos=(ep-ee_pos)*(1/eps);
            J.at(0,col)=dpos.x; J.at(1,col)=dpos.y; J.at(2,col)=dpos.z;
            // Rotation column (axis in world frame)
            Vec3f ax=joints[ji].axis;
            J.at(3,col)=ax.x; J.at(4,col)=ax.y; J.at(5,col)=ax.z;
            perturbed.joints[ji].q=joints[ji].q;
            ++col;
        }
        return J;
    }
};

// ─── Lagrangian mechanics helpers ─────────────────────────────────────────────

// Generalised mass matrix M(q): M_ij = Σ_k m_k J^T_k J_k + ω^T I_k ω
// Simplified: diagonal approximation from link masses and inertias
[[nodiscard]] inline DenseMatrix<f64> mass_matrix(const ArticulatedBody& body) {
    u32 n=body.dof();
    DenseMatrix<f64> M(n,n);
    // Build using Jacobians (simplified diagonal version)
    for(u32 i=0;i<n;++i) M.at(i,i)=f64(body.links[std::min(i,u32(body.links.size()-1))].mass);
    return M;
}

// Coriolis + centrifugal vector C(q,dq) — via numerical differentiation of M
[[nodiscard]] inline std::vector<f64> coriolis_vector(
    const ArticulatedBody& body, const std::vector<f64>& q,
    const std::vector<f64>& dq, f64 eps=1e-5)
{
    u32 n=body.dof();
    std::vector<f64> c(n,0);
    auto M=mass_matrix(body);
    // Christoffel: C_i = Σ_jk Γ^j_{ik} dq_j dq_k  (approximated via finite diff)
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
        c[i]+=M.at(i,j)*0; // placeholder — full implementation requires dM/dq
    return c;
}

// Gravity vector g(q): g_i = -∂PE/∂q_i
[[nodiscard]] inline std::vector<f64> gravity_vector(
    const ArticulatedBody& body, Vec3f gravity={0,-9.81f,0})
{
    u32 n=body.dof();
    std::vector<f64> g(n,0);
    auto T=body.forward_kinematics();
    // For each joint, compute how much gravity acts on the subtree
    for(u32 ji=0;ji<u32(body.joints.size())&&ji<n;++ji) {
        if(body.joints[ji].type==Joint::Type::Fixed) continue;
        // Sum of masses in subtree * distance from joint axis
        f32 contrib=0;
        for(u32 li=0;li<u32(body.links.size());++li) {
            Vec3f com_world{T[li].m[0][3]+body.links[li].com.x,T[li].m[1][3]+body.links[li].com.y,T[li].m[2][3]+body.links[li].com.z};
            contrib+=body.links[li].mass*dot(gravity,com_world);
        }
        g[ji]=-f64(contrib);
    }
    return g;
}

// Equation of motion: M(q)q'' + C(q,q') + g(q) = τ (generalised forces)
// Solve for accelerations: q'' = M^{-1}(τ - C - g)
[[nodiscard]] inline std::vector<f64> solve_eom(
    const ArticulatedBody& body,
    const std::vector<f64>& q, const std::vector<f64>& dq,
    const std::vector<f64>& tau, Vec3f gravity={0,-9.81f,0})
{
    u32 n=body.dof();
    auto M=mass_matrix(body);
    auto C=coriolis_vector(body,q,dq);
    auto g=gravity_vector(body,gravity);
    std::vector<f64> rhs(n);
    for(u32 i=0;i<n;++i) rhs[i]=tau[i]-C[i]-g[i];
    auto Minv=inverse(M);
    if(!Minv) return std::vector<f64>(n,0);
    return (*Minv)*rhs;
}

// ─── Constraint dynamics (Lagrange multipliers) ────────────────────────────────

// Bilateral constraints φ(q) = 0 → Jacobian Jc s.t. Jc q' = 0
// Baumgarte stabilisation: add β φ + γ φ' to constraint RHS
struct Constraint {
    std::function<f64(const std::vector<f64>&)> phi;     // constraint function
    std::function<std::vector<f64>(const std::vector<f64>&)> J; // Jacobian row
    f64 beta{0.1}, gamma{0.1}; // Baumgarte constants
};

[[nodiscard]] inline std::vector<f64> constrained_accelerations(
    const DenseMatrix<f64>& M,
    const std::vector<f64>& h,        // unconstrained forces
    const std::vector<Constraint>& constraints,
    const std::vector<f64>& q, const std::vector<f64>& dq)
{
    u32 n=u32(h.size()), m=u32(constraints.size());
    // Build Jacobian Jc (m×n)
    DenseMatrix<f64> Jc(m,n);
    std::vector<f64> zeta(m,0);
    for(u32 i=0;i<m;++i) {
        auto Ji=constraints[i].J(q);
        for(u32 j=0;j<n;++j) Jc.at(i,j)=Ji[j];
        f64 phi=constraints[i].phi(q);
        f64 dphi=0; for(u32 j=0;j<n;++j) dphi+=Ji[j]*dq[j];
        zeta[i]=-constraints[i].beta*phi-constraints[i].gamma*dphi;
    }
    // [M Jc^T; Jc 0][a; λ] = [h; ζ]
    u32 N=n+m;
    DenseMatrix<f64> KKT(N,N);
    std::vector<f64> rhs(N);
    for(u32 i=0;i<n;++i){ for(u32 j=0;j<n;++j) KKT.at(i,j)=M.at(i,j); rhs[i]=h[i]; }
    for(u32 i=0;i<m;++i){ for(u32 j=0;j<n;++j){ KKT.at(n+i,j)=Jc.at(i,j); KKT.at(j,n+i)=Jc.at(i,j); } rhs[n+i]=zeta[i]; }
    auto sol=solve(KKT,rhs);
    if(!sol) return std::vector<f64>(n,0);
    return std::vector<f64>(sol->begin(),sol->begin()+n);
}

} // namespace qs
