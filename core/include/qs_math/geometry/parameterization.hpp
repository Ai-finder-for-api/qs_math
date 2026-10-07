#pragma once
// qs_math/parameterization.hpp — LSCM, ABF, harmonic maps, conformal parameterization (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "linear_algebra.hpp"
#include "sparse_matrix.hpp"
#include "iterative_solvers.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <numbers>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace qs {

// ─── Mesh UV parameterization helpers ────────────────────────────────────────

// Build cotangent weights for mesh Laplacian (standard harmonic parameterization)
[[nodiscard]] inline SparseCSR<f64> build_cotangent_laplacian(
    u32 nv,
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& verts)
{
    std::vector<std::tuple<u32,u32,f64>> trips;
    for(const auto& t:tris) {
        for(u32 i=0;i<3;++i) {
            u32 vi=t[i], vj=t[(i+1)%3], vk=t[(i+2)%3];
            // Cotangent at vertex vk (opposite edge vi-vj)
            Vec3f u=verts[vi]-verts[vk], v=verts[vj]-verts[vk];
            f32 cos_k=dot(u,v);
            f32 sin_k=length(cross(u,v));
            f64 cot=sin_k>1e-10f?f64(cos_k/sin_k):0;
            f64 w=cot*0.5;
            trips.emplace_back(vi,vj,-w); trips.emplace_back(vj,vi,-w);
            trips.emplace_back(vi,vi, w); trips.emplace_back(vj,vj, w);
        }
    }
    return SparseCSR<f64>::from_triplets(nv,nv,std::move(trips));
}

// Tutte embedding (uniform weights, guaranteed no foldovers for convex boundary)
[[nodiscard]] inline std::vector<Vec2f> tutte_parameterization(
    u32 nv,
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<u32>& boundary, // boundary vertex indices in order
    u32 cg_iters=1000, f64 tol=1e-8)
{
    u32 nb=u32(boundary.size());
    // Pin boundary to unit circle
    std::vector<Vec2f> uv(nv,{0,0});
    std::vector<bool> is_boundary(nv,false);
    for(u32 i=0;i<nb;++i) {
        f32 angle=2*f32(std::numbers::pi)*i/nb;
        uv[boundary[i]]={std::cos(angle),std::sin(angle)};
        is_boundary[boundary[i]]=true;
    }
    // Build uniform Laplacian for interior vertices
    std::vector<std::vector<u32>> adj(nv);
    for(const auto& t:tris) for(u32 e=0;e<3;++e){ adj[t[e]].push_back(t[(e+1)%3]); adj[t[(e+1)%3]].push_back(t[e]); }
    for(auto& a:adj){ std::sort(a.begin(),a.end()); a.erase(std::unique(a.begin(),a.end()),a.end()); }
    // Collect interior vertices
    std::vector<u32> interior;
    for(u32 i=0;i<nv;++i) if(!is_boundary[i]) interior.push_back(i);
    u32 ni=u32(interior.size());
    if(ni==0) return uv;
    // Index interior vertices
    std::vector<i32> int_idx(nv,-1);
    for(u32 i=0;i<ni;++i) int_idx[interior[i]]=i32(i);
    // Build system: for each interior vertex, uv[i] = avg(uv[neighbours])
    std::vector<std::tuple<u32,u32,f64>> trips;
    std::vector<f64> bx(ni,0),by(ni,0);
    for(u32 ii=0;ii<ni;++ii) {
        u32 v=interior[ii];
        u32 deg=u32(adj[v].size());
        trips.emplace_back(ii,ii,1.0);
        for(u32 nb_v:adj[v]) {
            if(is_boundary[nb_v]) { bx[ii]+=uv[nb_v].x/deg; by[ii]+=uv[nb_v].y/deg; }
            else { i32 j=int_idx[nb_v]; if(j>=0) trips.emplace_back(ii,u32(j),-1.0/deg); }
        }
    }
    auto L=SparseCSR<f64>::from_triplets(ni,ni,std::move(trips));
    std::vector<f64> ux(ni,0),uy(ni,0);
    cg_solve(L,bx,ux,cg_iters,tol);
    cg_solve(L,by,uy,cg_iters,tol);
    for(u32 ii=0;ii<ni;++ii) uv[interior[ii]]={f32(ux[ii]),f32(uy[ii])};
    return uv;
}

// Harmonic parameterization (cotangent weights, more shape-aware than Tutte)
[[nodiscard]] inline std::vector<Vec2f> harmonic_parameterization(
    u32 nv,
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& verts,
    const std::vector<u32>& boundary,
    u32 cg_iters=2000, f64 tol=1e-8)
{
    u32 nb=u32(boundary.size());
    std::vector<Vec2f> uv(nv,{0,0});
    std::vector<bool> is_boundary(nv,false);
    for(u32 i=0;i<nb;++i) {
        f32 angle=2*f32(std::numbers::pi)*i/nb;
        uv[boundary[i]]={std::cos(angle),std::sin(angle)};
        is_boundary[boundary[i]]=true;
    }
    auto L=build_cotangent_laplacian(nv,tris,verts);
    std::vector<u32> interior;
    for(u32 i=0;i<nv;++i) if(!is_boundary[i]) interior.push_back(i);
    u32 ni=u32(interior.size()); if(ni==0) return uv;
    std::vector<i32> int_idx(nv,-1);
    for(u32 i=0;i<ni;++i) int_idx[interior[i]]=i32(i);
    // Build interior Laplacian and RHS
    std::vector<std::tuple<u32,u32,f64>> trips;
    std::vector<f64> bx(ni,0),by(ni,0);
    for(u32 ii=0;ii<ni;++ii) {
        u32 v=interior[ii];
        for(u32 k=L.row_ptr[v];k<L.row_ptr[v+1];++k) {
            u32 j=L.col_idx[k]; f64 w=L.values[k];
            if(is_boundary[j]){ bx[ii]-=w*uv[j].x; by[ii]-=w*uv[j].y; }
            else { i32 jj=int_idx[j]; if(jj>=0) trips.emplace_back(ii,u32(jj),w); }
        }
    }
    // Diagonal
    for(u32 ii=0;ii<ni;++ii){
        u32 v=interior[ii];
        f64 d=L.get(v,v);
        trips.emplace_back(ii,ii,d);
    }
    auto Li=SparseCSR<f64>::from_triplets(ni,ni,std::move(trips));
    std::vector<f64> ux(ni,0),uy(ni,0);
    cg_solve(Li,bx,ux,cg_iters,tol);
    cg_solve(Li,by,uy,cg_iters,tol);
    for(u32 ii=0;ii<ni;++ii) uv[interior[ii]]={f32(ux[ii]),f32(uy[ii])};
    return uv;
}

// ─── LSCM — Least Squares Conformal Maps ─────────────────────────────────────
// Minimises conformal energy: ∫|∂u/∂x - ∂v/∂y|² + |∂u/∂y + ∂v/∂x|² dA
// Global optimum when 2 vertices are pinned
[[nodiscard]] inline std::vector<Vec2f> lscm(
    u32 nv,
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& verts,
    u32 pin0, Vec2f uv0,
    u32 pin1, Vec2f uv1,
    u32 lsqr_iters=500, f64 tol=1e-8)
{
    // LSCM: build complex-valued system, solve as 2n real DOFs
    // Each triangle contributes 2 rows to the least-squares system
    u32 ndof=nv*2; // [u0,v0,u1,v1,...]
    std::vector<std::tuple<u32,u32,f64>> trips;
    u32 row=0;
    for(const auto& t:tris) {
        Vec3f p0=verts[t[0]],p1=verts[t[1]],p2=verts[t[2]];
        // Local 2D frame
        Vec3f e01=p1-p0,e02=p2-p0;
        f32 A2=length(cross(e01,e02));
        if(A2<1e-10f) continue;
        // Local coordinates
        f32 s1=length(e01);
        Vec3f ex=normalize(e01);
        Vec3f n=normalize(cross(e01,e02));
        Vec3f ey=cross(n,ex);
        f32 x1=s1, y1=0;
        f32 x2=dot(e02,ex), y2=dot(e02,ey);
        // LSCM contribution: [(u1-u0) + i(v1-v0)] = (W1)[(u2-u0)+i(v2-v0)]
        // W1 = (x2+iy2)/(x1+iy1) — complex ratio
        // This leads to a real 2x(2nv) sparse system
        f32 wre=(x1*x2+y1*y2)/(x1*x1+y1*y1+1e-20f);
        f32 wim=(x1*y2-y1*x2)/(x1*x1+y1*y1+1e-20f);
        // Row for real part: (u1-u0) - wre*(u2-u0) + wim*(v2-v0) = 0
        // Row for imag part: (v1-v0) - wre*(v2-v0) - wim*(u2-u0) = 0
        u32 u0=t[0]*2,v0=t[0]*2+1,u1=t[1]*2,v1_=t[1]*2+1,u2=t[2]*2,v2=t[2]*2+1;
        // Real row
        trips.emplace_back(row,u1, 1.0); trips.emplace_back(row,u0,-1.0);
        trips.emplace_back(row,u2,-wre); trips.emplace_back(row,v2, wim);
        // Imag row
        trips.emplace_back(row+1,v1_,1.0); trips.emplace_back(row+1,v0,-1.0);
        trips.emplace_back(row+1,v2,-wre); trips.emplace_back(row+1,u2,-wim);
        row+=2;
    }
    // Add pin constraints as heavy rows
    f64 pin_weight=1e6;
    trips.emplace_back(row,  pin0*2,   pin_weight); // u_pin0 = uv0.x
    trips.emplace_back(row+1,pin0*2+1, pin_weight); // v_pin0 = uv0.y
    trips.emplace_back(row+2,pin1*2,   pin_weight); // u_pin1 = uv1.x
    trips.emplace_back(row+3,pin1*2+1, pin_weight); // v_pin1 = uv1.y
    u32 nrows=row+4;
    auto A=SparseCSR<f64>::from_triplets(nrows,ndof,std::move(trips));
    std::vector<f64> b(nrows,0);
    b[row]  =pin_weight*uv0.x; b[row+1]=pin_weight*uv0.y;
    b[row+2]=pin_weight*uv1.x; b[row+3]=pin_weight*uv1.y;
    // Normal equations A^T A x = A^T b
    auto At=A.transpose();
    auto AtA=At.to_dense()*A.to_dense(); // small enough for direct solve
    auto Atb_dense=At*b;
    DenseMatrix<f64> AtA_dense=AtA;
    auto sol=solve(AtA_dense,Atb_dense,tol);
    std::vector<Vec2f> uv(nv,{0,0});
    if(sol) for(u32 i=0;i<nv;++i) uv[i]={f32((*sol)[i*2]),f32((*sol)[i*2+1])};
    return uv;
}

// ─── Conformal energy measurement ────────────────────────────────────────────
[[nodiscard]] inline f64 conformal_energy(
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& verts3d,
    const std::vector<Vec2f>& uv)
{
    f64 E=0;
    for(const auto& t:tris) {
        Vec3f p0=verts3d[t[0]],p1=verts3d[t[1]],p2=verts3d[t[2]];
        Vec2f u0=uv[t[0]],u1=uv[t[1]],u2=uv[t[2]];
        f32 A3d=length(cross(p1-p0,p2-p0))*0.5f;
        f32 A2d=std::abs((u1.x-u0.x)*(u2.y-u0.y)-(u2.x-u0.x)*(u1.y-u0.y))*0.5f;
        if(A3d<1e-10f||A2d<1e-10f) continue;
        // Conformal energy = ratio of singular values squared
        f32 ratio=A2d/A3d;
        E+=f64(ratio+1.f/ratio-2);
    }
    return E;
}

// ─── UV stretch metrics ───────────────────────────────────────────────────────
// L2 stretch: σ2 = sqrt(Σ((σ1²+σ2²)/2 * A3d) / Σ A3d)
[[nodiscard]] inline f32 l2_stretch(
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& v3d,
    const std::vector<Vec2f>& uv)
{
    f32 num=0,denom=0;
    for(const auto& t:tris) {
        Vec3f p0=v3d[t[0]],p1=v3d[t[1]],p2=v3d[t[2]];
        Vec2f u0=uv[t[0]],u1=uv[t[1]],u2=uv[t[2]];
        f32 A=length(cross(p1-p0,p2-p0))*0.5f;
        // 2D Jacobian
        f32 du1=u1.x-u0.x,du2=u2.x-u0.x;
        f32 dv1=u1.y-u0.y,dv2=u2.y-u0.y;
        f32 J00=du1,J01=du2,J10=dv1,J11=dv2;
        // Singular values via trace and det of J^T J
        f32 a=J00*J00+J10*J10,b=J00*J01+J10*J11,c=J01*J01+J11*J11;
        f32 trace=a+c,det=a*c-b*b;
        f32 sv2_sum=trace; // σ1²+σ2² = trace(J^T J)
        num+=sv2_sum*A; denom+=A;
    }
    return denom>0?std::sqrt(num/denom):0;
}

// ─── ABF++ angle-based flattening (simplified) ────────────────────────────────
// Each triangle has 3 angles; ABF minimises distortion of angles
// Returns set of valid angles per triangle (sum to π, all positive)
[[nodiscard]] inline std::vector<std::array<f32,3>> abf_angles(
    const std::vector<std::array<u32,3>>& tris,
    const std::vector<Vec3f>& verts)
{
    std::vector<std::array<f32,3>> angles(tris.size());
    for(u32 i=0;i<u32(tris.size());++i) {
        const auto& t=tris[i];
        Vec3f a=verts[t[0]],b=verts[t[1]],c=verts[t[2]];
        Vec3f ab=normalize(b-a),ac=normalize(c-a);
        Vec3f ba=normalize(a-b),bc=normalize(c-b);
        Vec3f ca=normalize(a-c),cb=normalize(b-c);
        angles[i][0]=std::acos(std::clamp(dot(ab,ac),-1.f,1.f));
        angles[i][1]=std::acos(std::clamp(dot(ba,bc),-1.f,1.f));
        angles[i][2]=std::acos(std::clamp(dot(ca,cb),-1.f,1.f));
        // Ensure sum = π (numerical fix)
        f32 sum=angles[i][0]+angles[i][1]+angles[i][2];
        if(sum>1e-7f) for(auto& v:angles[i]) v*=f32(std::numbers::pi)/sum;
    }
    return angles;
}

} // namespace qs
