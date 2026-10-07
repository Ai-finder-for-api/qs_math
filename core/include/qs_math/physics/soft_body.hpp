#pragma once
// qs_math/physics/soft_body.hpp — Soft body physics: mass-spring networks,
//   Position-Based Dynamics (PBD), XPBD constraints, cloth simulation,
//   deformable mesh, finite element stiffness, shape matching (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/mat3.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <span>
#include <functional>

namespace qs::math::physics {

// ─── Particle ─────────────────────────────────────────────────────────────
struct SoftParticle {
    Vec3f pos, prev_pos, vel;
    f32   mass=1.f, inv_mass=1.f;
    bool  fixed=false;

    void set_fixed(bool f) noexcept { fixed=f; inv_mass=f?0.f:1.f/mass; }
};

// ─── Spring constraint ────────────────────────────────────────────────────
struct SpringConstraint {
    int a, b;
    f32 rest_length;
    f32 stiffness=1.f;  // PBD compliance = 0
    f32 damping=0.f;

    void solve_pbd(std::vector<SoftParticle>& pts) const noexcept {
        auto& pa=pts[a]; auto& pb=pts[b];
        Vec3f d=pb.pos-pa.pos;
        f32 len=d.length();
        if(len<1e-7f) return;
        f32 C=len-rest_length;
        Vec3f n=d*(1.f/len);
        f32 w=pa.inv_mass+pb.inv_mass;
        if(w<1e-10f) return;
        Vec3f dx=n*(stiffness*C/w);
        if(!pa.fixed) pa.pos=pa.pos+dx*pa.inv_mass;
        if(!pb.fixed) pb.pos=pb.pos-dx*pb.inv_mass;
    }

    // XPBD: alpha=compliance/(dt^2)
    void solve_xpbd(std::vector<SoftParticle>& pts,
                    std::vector<f32>& lambdas, int idx,
                    f32 alpha, f32 dt) const noexcept
    {
        auto& pa=pts[a]; auto& pb=pts[b];
        Vec3f d=pb.pos-pa.pos;
        f32 len=d.length();
        if(len<1e-7f) return;
        f32 C=len-rest_length;
        Vec3f grad_a=-(d/len), grad_b=(d/len);
        f32 w=pa.inv_mass*(grad_a.x*grad_a.x+grad_a.y*grad_a.y+grad_a.z*grad_a.z)
            +pb.inv_mass*(grad_b.x*grad_b.x+grad_b.y*grad_b.y+grad_b.z*grad_b.z);
        f32 dlambda=(-C-alpha*lambdas[idx])/(w+alpha);
        lambdas[idx]+=dlambda;
        if(!pa.fixed) pa.pos=pa.pos+grad_a*(dlambda*pa.inv_mass);
        if(!pb.fixed) pb.pos=pb.pos+grad_b*(dlambda*pb.inv_mass);
    }
};

// ─── Volume constraint (for jelly-like objects) ───────────────────────────
struct VolumeConstraint {
    std::array<int,4> tet; // tetrahedron vertex indices
    f32 rest_volume;
    f32 stiffness=1.f;

    [[nodiscard]] static f32 tet_volume(Vec3f a,Vec3f b,Vec3f c,Vec3f d) noexcept {
        return dot(a-d,cross(b-d,c-d))/6.f;
    }

    void solve_pbd(std::vector<SoftParticle>& pts) const noexcept {
        Vec3f& p0=pts[tet[0]].pos, &p1=pts[tet[1]].pos,
               &p2=pts[tet[2]].pos, &p3=pts[tet[3]].pos;
        f32 V=tet_volume(p0,p1,p2,p3);
        f32 C=V-rest_volume;
        // Gradients: dV/dp_i = (cross of other 3)/6
        Vec3f g0=cross(p1-p3,p2-p3)*(1.f/6.f);
        Vec3f g1=cross(p2-p3,p0-p3)*(1.f/6.f);
        Vec3f g2=cross(p0-p3,p1-p3)*(1.f/6.f);
        Vec3f g3=-(g0+g1+g2);
        f32 w=pts[tet[0]].inv_mass*dot(g0,g0)+pts[tet[1]].inv_mass*dot(g1,g1)
             +pts[tet[2]].inv_mass*dot(g2,g2)+pts[tet[3]].inv_mass*dot(g3,g3);
        if(w<1e-10f) return;
        f32 s=stiffness*C/w;
        if(!pts[tet[0]].fixed) p0=p0-g0*s*pts[tet[0]].inv_mass;
        if(!pts[tet[1]].fixed) p1=p1-g1*s*pts[tet[1]].inv_mass;
        if(!pts[tet[2]].fixed) p2=p2-g2*s*pts[tet[2]].inv_mass;
        if(!pts[tet[3]].fixed) p3=p3-g3*s*pts[tet[3]].inv_mass;
    }
};

// ─── Bending constraint (dihedral angle for cloth) ────────────────────────
struct BendingConstraint {
    int p1,p2,p3,p4; // shared edge p3-p4, triangles (p1,p3,p4) and (p2,p4,p3)
    f32 rest_angle;
    f32 stiffness=0.01f;

    void solve_pbd(std::vector<SoftParticle>& pts) const noexcept {
        Vec3f& a=pts[p1].pos, &b=pts[p2].pos,
               &c=pts[p3].pos, &d=pts[p4].pos;
        Vec3f n1=cross(c-a,d-a), n2=cross(d-b,c-b);
        f32 n1l=n1.length(), n2l=n2.length();
        if(n1l<1e-6f||n2l<1e-6f) return;
        n1=n1*(1.f/n1l); n2=n2*(1.f/n2l);
        f32 cos_a=clamp(dot(n1,n2),-1.f,1.f);
        f32 angle=std::acos(cos_a);
        f32 C=angle-rest_angle;
        if(std::abs(C)<1e-7f) return;
        // Gradient computation (simplified)
        Vec3f e=d-c; f32 el=e.length();
        if(el<1e-6f) return;
        Vec3f grad_a=-cross(n1,e)*(1.f/n1l);
        Vec3f grad_b= cross(n2,e)*(1.f/n2l);
        f32 w=pts[p1].inv_mass*dot(grad_a,grad_a)+pts[p2].inv_mass*dot(grad_b,grad_b);
        if(w<1e-10f) return;
        f32 s=stiffness*C/w;
        if(!pts[p1].fixed) a=a+grad_a*(s*pts[p1].inv_mass);
        if(!pts[p2].fixed) b=b-grad_b*(s*pts[p2].inv_mass);
    }
};

// ─── PBD Simulation ───────────────────────────────────────────────────────
struct PBDSimulation {
    std::vector<SoftParticle> particles;
    std::vector<SpringConstraint> springs;
    std::vector<BendingConstraint> bends;
    std::vector<VolumeConstraint> volumes;
    Vec3f gravity={0,-9.81f,0};
    f32 damping=0.99f;

    void substep(f32 dt, int iters=10) noexcept {
        // External forces + predict
        for(auto& p:particles){
            if(p.fixed) continue;
            p.vel=p.vel+gravity*dt;
            p.vel=p.vel*damping;
            p.prev_pos=p.pos;
            p.pos=p.pos+p.vel*dt;
        }
        // Constraint solving
        for(int it=0;it<iters;++it){
            for(auto& s:springs) s.solve_pbd(particles);
            for(auto& b:bends)   b.solve_pbd(particles);
            for(auto& v:volumes) v.solve_pbd(particles);
        }
        // Update velocity
        f32 inv_dt=1.f/dt;
        for(auto& p:particles){
            if(p.fixed){p.vel={0,0,0};continue;}
            p.vel=(p.pos-p.prev_pos)*inv_dt;
        }
    }

    void ground_collision(f32 y=0.f, f32 friction=0.3f) noexcept {
        for(auto& p:particles){
            if(p.pos.y<y){
                p.pos.y=y;
                p.vel.y=max(0.f,p.vel.y);
                p.vel.x*=(1.f-friction);
                p.vel.z*=(1.f-friction);
            }
        }
    }
};

// ─── Shape matching ────────────────────────────────────────────────────────
// Müller et al. 2005 — rigid/linear shape matching
struct ShapeMatchingBody {
    std::vector<Vec3f> rest_pos; // rest positions (relative to CoM)
    std::vector<f32>   masses;
    f32 stiffness=0.8f;

    // Extract translation and linear deformation from current positions
    void solve(std::vector<Vec3f>& current) const noexcept {
        int n=int(rest_pos.size());
        // Compute current CoM
        Vec3f cm_cur{0,0,0}; f32 total_mass=0;
        for(int i=0;i<n;++i){cm_cur=cm_cur+current[i]*masses[i];total_mass+=masses[i];}
        cm_cur=cm_cur*(1.f/total_mass);
        // Compute rest CoM
        Vec3f cm_rest{0,0,0};
        for(int i=0;i<n;++i) cm_rest=cm_rest+rest_pos[i]*masses[i];
        cm_rest=cm_rest*(1.f/total_mass);
        // Apq = sum_i mi*(q_i - cm_rest) * (p_i - cm_cur)^T — 3x3 matrix (column-major)
        f32 Apq[9]={};
        for(int i=0;i<n;++i){
            Vec3f q=rest_pos[i]-cm_rest, p=current[i]-cm_cur;
            f32 m=masses[i];
            Apq[0]+=m*q.x*p.x; Apq[1]+=m*q.y*p.x; Apq[2]+=m*q.z*p.x;
            Apq[3]+=m*q.x*p.y; Apq[4]+=m*q.y*p.y; Apq[5]+=m*q.z*p.y;
            Apq[6]+=m*q.x*p.z; Apq[7]+=m*q.y*p.z; Apq[8]+=m*q.z*p.z;
        }
        // Polar decompose Apq = R*S; extract R via iterative method
        f32 R[9]={1,0,0,0,1,0,0,0,1};
        for(int it=0;it<10;++it){
            // Approximate R^{-T} via adjugate
            f32 det=R[0]*(R[4]*R[8]-R[5]*R[7])-R[1]*(R[3]*R[8]-R[5]*R[6])+R[2]*(R[3]*R[7]-R[4]*R[6]);
            if(std::abs(det)<1e-10f) break;
            f32 inv=1.f/det;
            f32 Rinv[9]={
                (R[4]*R[8]-R[5]*R[7])*inv,-(R[1]*R[8]-R[2]*R[7])*inv,(R[1]*R[5]-R[2]*R[4])*inv,
               -(R[3]*R[8]-R[5]*R[6])*inv, (R[0]*R[8]-R[2]*R[6])*inv,-(R[0]*R[5]-R[2]*R[3])*inv,
                (R[3]*R[7]-R[4]*R[6])*inv,-(R[0]*R[7]-R[1]*R[6])*inv,(R[0]*R[4]-R[1]*R[3])*inv
            };
            for(int j=0;j<9;++j) R[j]=0.5f*(R[j]+Rinv[j]);
        }
        // Compute goal positions and blend
        for(int i=0;i<n;++i){
            Vec3f q=rest_pos[i]-cm_rest;
            Vec3f goal_local={R[0]*q.x+R[3]*q.y+R[6]*q.z,
                              R[1]*q.x+R[4]*q.y+R[7]*q.z,
                              R[2]*q.x+R[5]*q.y+R[8]*q.z};
            Vec3f goal=cm_cur+goal_local;
            current[i]=lerp(current[i],goal,stiffness);
        }
    }
};

// ─── Cloth mesh builder ───────────────────────────────────────────────────
struct ClothMesh {
    std::vector<SoftParticle> particles;
    std::vector<SpringConstraint> structural, shear, bending_springs;
    int rows, cols;

    void build(int r, int c, f32 spacing=0.1f,
               f32 struct_k=1.f, f32 shear_k=0.5f, f32 bend_k=0.2f) noexcept
    {
        rows=r; cols=c;
        particles.resize(r*c);
        for(int i=0;i<r;++i) for(int j=0;j<c;++j){
            auto& p=particles[i*c+j];
            p.pos={j*spacing,0.f,i*spacing};
            p.prev_pos=p.pos;
            p.mass=1.f; p.inv_mass=1.f;
        }
        // Structural springs (adjacent)
        auto add_spring=[&](int a,int b,f32 k){
            f32 rest=(particles[a].pos-particles[b].pos).length();
            structural.push_back({a,b,rest,k});
        };
        auto add_bend=[&](int a,int b,f32 k){
            f32 rest=(particles[a].pos-particles[b].pos).length();
            bending_springs.push_back({a,b,rest,k});
        };
        for(int i=0;i<r;++i) for(int j=0;j<c;++j){
            int idx=i*c+j;
            if(j+1<c) add_spring(idx,idx+1,struct_k);   // horizontal
            if(i+1<r) add_spring(idx,idx+c,struct_k);   // vertical
            if(i+1<r&&j+1<c){ add_spring(idx,idx+c+1,shear_k); // shear
                               add_spring(idx+1,idx+c,shear_k); }
            if(j+2<c) add_bend(idx,idx+2,bend_k);   // bending horizontal
            if(i+2<r) add_bend(idx,idx+2*c,bend_k); // bending vertical
        }
        // Fix top row
        for(int j=0;j<c;++j) particles[j].set_fixed(true);
    }
};

// ─── Finite element stiffness (linear tet) ───────────────────────────────
// Compute element stiffness matrix K (12x12) for linear tetrahedron
inline void linear_tet_stiffness(
    Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3,
    f32 E, f32 nu, f32 K[144]) noexcept
{
    // Compute volume
    Vec3f a=p1-p0, b=p2-p0, c=p3-p0;
    f32 V=std::abs(dot(a,cross(b,c)))/6.f;
    if(V<1e-10f){ for(int i=0;i<144;++i) K[i]=0.f; return; }
    // Shape function gradients (constant for linear tet)
    // B = [dN/dx dN/dy dN/dz] for each node
    f32 lam=E*nu/((1+nu)*(1-2*nu));
    f32 mu=E/(2*(1+nu));
    // Simplified: diagonal K approximation (mass-lumped)
    f32 k_diag=(lam+2*mu)*V/4.f;
    for(int i=0;i<144;++i) K[i]=0.f;
    for(int i=0;i<12;++i) K[i*12+i]=k_diag;
}

} // namespace qs::math::physics
