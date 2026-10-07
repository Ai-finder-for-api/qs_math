#pragma once
// qs_math/differential_geometry.hpp — Manifolds, geodesics, parallel transport, curvature (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "vec4.hpp"
#include <cmath>
#include <array>
#include <vector>
#include <functional>
#include <span>

namespace qs {

// ─── Surface (parametric) ─────────────────────────────────────────────────────

// Parametric surface r(u,v) with first/second fundamental forms
struct ParametricSurface {
    std::function<Vec3f(f32,f32)> r;   // position
    std::function<Vec3f(f32,f32)> ru;  // dr/du
    std::function<Vec3f(f32,f32)> rv;  // dr/dv

    [[nodiscard]] Vec3f normal(f32 u,f32 v) const {
        return normalize(cross(ru(u,v),rv(u,v)));
    }

    // First fundamental form coefficients E,F,G
    [[nodiscard]] std::array<f32,3> first_form(f32 u,f32 v) const {
        Vec3f U=ru(u,v),V=rv(u,v);
        return {dot(U,U),dot(U,V),dot(V,V)};
    }

    // Second fundamental form coefficients e,f,g
    [[nodiscard]] std::array<f32,3> second_form(f32 u,f32 v, f32 eps=0.001f) const {
        Vec3f N=normal(u,v);
        // Finite difference for r_uu, r_uv, r_vv
        Vec3f ruu=(ru(u+eps,v)-ru(u-eps,v))/(2.f*eps);
        Vec3f ruv=(ru(u,v+eps)-ru(u,v-eps))/(2.f*eps);
        Vec3f rvv=(rv(u,v+eps)-rv(u,v-eps))/(2.f*eps);
        return {dot(ruu,N),dot(ruv,N),dot(rvv,N)};
    }

    // Gaussian curvature K = (eg - f^2)/(EG - F^2)
    [[nodiscard]] f32 gaussian_curvature(f32 u,f32 v) const {
        auto[E,F,G]=first_form(u,v);
        auto[e,f2,g]=second_form(u,v);
        f32 denom=E*G-F*F;
        return denom>1e-8f?(e*g-f2*f2)/denom:0.f;
    }

    // Mean curvature H = (eG - 2fF + gE) / (2(EG-F^2))
    [[nodiscard]] f32 mean_curvature(f32 u,f32 v) const {
        auto[E,F,G]=first_form(u,v);
        auto[e,f2,g]=second_form(u,v);
        f32 denom=2.f*(E*G-F*F);
        return denom>1e-8f?(e*G-2.f*f2*F+g*E)/denom:0.f;
    }

    // Principal curvatures k1,k2 from H and K
    [[nodiscard]] std::pair<f32,f32> principal_curvatures(f32 u,f32 v) const {
        f32 H=mean_curvature(u,v), K=gaussian_curvature(u,v);
        f32 disc=std::max(0.f,H*H-K);
        return {H-std::sqrt(disc),H+std::sqrt(disc)};
    }
};

// ─── Christoffel symbols ──────────────────────────────────────────────────────

// Gamma^k_ij from metric g_ij (2D manifold, 4 Christoffel symbols each)
// g is 2x2 symmetric: g[0]=g_uu, g[1]=g_uv, g[2]=g_vv
struct ChristoffelSymbols {
    f32 G[2][2][2]{}; // G[k][i][j]

    [[nodiscard]] static ChristoffelSymbols from_metric(
        f32 guu, f32 guv, f32 gvv,
        f32 dEdu, f32 dEDv, f32 dFdu, f32 dFdv, f32 dGdu, f32 dGdv)
    {
        // g^{ij} (inverse metric)
        f32 det=guu*gvv-guv*guv;
        if(std::abs(det)<1e-10f) return {};
        f32 guu_inv=gvv/det, guv_inv=-guv/det, gvv_inv=guu/det;
        ChristoffelSymbols C;
        // Gamma^u_uu, Gamma^u_uv, Gamma^u_vv, Gamma^v_uu, Gamma^v_uv, Gamma^v_vv
        // Gamma^k_ij = 0.5 * g^kl (partial_i g_jl + partial_j g_il - partial_l g_ij)
        // 1D partial derivatives of g passed in as parameters
        C.G[0][0][0]=0.5f*(guu_inv*dEdu + guv_inv*(2.f*dFdu-dEDv));
        C.G[0][0][1]=0.5f*(guu_inv*dEDv + guv_inv*dGdu);
        C.G[0][1][0]=C.G[0][0][1];
        C.G[0][1][1]=0.5f*(guu_inv*(2.f*dFdv-dGdu) + guv_inv*dGdv);
        C.G[1][0][0]=0.5f*(guv_inv*dEdu + gvv_inv*(2.f*dFdu-dEDv));
        C.G[1][0][1]=0.5f*(guv_inv*dEDv + gvv_inv*dGdu);
        C.G[1][1][0]=C.G[1][0][1];
        C.G[1][1][1]=0.5f*(guv_inv*(2.f*dFdv-dGdu) + gvv_inv*dGdv);
        return C;
    }
};

// ─── Geodesic ODE ─────────────────────────────────────────────────────────────

struct GeodesicState {
    f32 u{},v{};      // position on surface
    f32 du{},dv{};    // velocity (tangent vector)
};

// One RK4 step of geodesic equation:
// d^2u/dt^2 + Gamma^u_ij (du^i/dt)(du^j/dt) = 0
inline GeodesicState geodesic_step(
    const GeodesicState& s, const ChristoffelSymbols& C, f32 dt)
{
    auto rhs=[&](const GeodesicState& q)->GeodesicState{
        f32 ddu=-(C.G[0][0][0]*q.du*q.du+2.f*C.G[0][0][1]*q.du*q.dv+C.G[0][1][1]*q.dv*q.dv);
        f32 ddv=-(C.G[1][0][0]*q.du*q.du+2.f*C.G[1][0][1]*q.du*q.dv+C.G[1][1][1]*q.dv*q.dv);
        return {q.du,q.dv,ddu,ddv};
    };
    auto k1=rhs(s);
    GeodesicState s2{s.u+k1.u*dt/2.f,s.v+k1.v*dt/2.f,s.du+k1.du*dt/2.f,s.dv+k1.dv*dt/2.f};
    auto k2=rhs(s2);
    GeodesicState s3{s.u+k2.u*dt/2.f,s.v+k2.v*dt/2.f,s.du+k2.du*dt/2.f,s.dv+k2.dv*dt/2.f};
    auto k3=rhs(s3);
    GeodesicState s4{s.u+k3.u*dt,s.v+k3.v*dt,s.du+k3.du*dt,s.dv+k3.dv*dt};
    auto k4=rhs(s4);
    return {
        s.u+(k1.u+2.f*k2.u+2.f*k3.u+k4.u)*dt/6.f,
        s.v+(k1.v+2.f*k2.v+2.f*k3.v+k4.v)*dt/6.f,
        s.du+(k1.du+2.f*k2.du+2.f*k3.du+k4.du)*dt/6.f,
        s.dv+(k1.dv+2.f*k2.dv+2.f*k3.dv+k4.dv)*dt/6.f};
}

// Trace geodesic from starting point/direction
[[nodiscard]] inline std::vector<std::pair<f32,f32>> trace_geodesic(
    f32 u0,f32 v0,f32 du0,f32 dv0,
    const ChristoffelSymbols& C, f32 length, u32 steps=100)
{
    std::vector<std::pair<f32,f32>> path;
    GeodesicState s{u0,v0,du0,dv0};
    f32 dt=length/f32(steps);
    for(u32 i=0;i<=steps;++i){
        path.push_back({s.u,s.v});
        s=geodesic_step(s,C,dt);
    }
    return path;
}

// ─── Parallel transport ───────────────────────────────────────────────────────

struct ParallelTransportState {
    f32 u{},v{};       // position
    f32 du{},dv{};     // velocity
    f32 wu{},wv{};     // vector being transported
};

inline ParallelTransportState parallel_transport_step(
    const ParallelTransportState& s, const ChristoffelSymbols& C, f32 dt)
{
    f32 dwu=-(C.G[0][0][0]*s.wu*s.du+C.G[0][0][1]*(s.wu*s.dv+s.wv*s.du)+C.G[0][1][1]*s.wv*s.dv);
    f32 dwv=-(C.G[1][0][0]*s.wu*s.du+C.G[1][0][1]*(s.wu*s.dv+s.wv*s.du)+C.G[1][1][1]*s.wv*s.dv);
    return {s.u+s.du*dt,s.v+s.dv*dt,s.du,s.dv,s.wu+dwu*dt,s.wv+dwv*dt};
}

// Holonomy angle (rotation of vector after parallel transport around loop)
[[nodiscard]] inline f32 holonomy_angle(
    std::span<const std::pair<f32,f32>> loop, // (u,v) pairs forming closed loop
    const ChristoffelSymbols& C)
{
    if(loop.size()<2) return 0.f;
    ParallelTransportState s{loop[0].first,loop[0].second,0.f,0.f,1.f,0.f};
    u32 N=u32(loop.size());
    for(u32 i=0;i+1<N;++i){
        f32 du=loop[i+1].first-loop[i].first;
        f32 dv=loop[i+1].second-loop[i].second;
        s.du=du; s.dv=dv;
        s=parallel_transport_step(s,C,1.f);
    }
    return std::atan2(s.wv,s.wu); // angle of transported vector
}

// ─── Lie bracket on vector fields ────────────────────────────────────────────

// [X,Y]f = X(Yf) - Y(Xf), Lie bracket in R^3
[[nodiscard]] inline Vec3f lie_bracket(
    const std::function<Vec3f(Vec3f)>& X,
    const std::function<Vec3f(Vec3f)>& Y,
    const Vec3f& p, f32 eps=0.001f)
{
    auto jac_vf=[&](const std::function<Vec3f(Vec3f)>& F, const Vec3f& x)->Mat3f{
        Vec3f dx=(F({x.x+eps,x.y,x.z})-F({x.x-eps,x.y,x.z}))/(2.f*eps);
        Vec3f dy=(F({x.x,x.y+eps,x.z})-F({x.x,x.y-eps,x.z}))/(2.f*eps);
        Vec3f dz=(F({x.x,x.y,x.z+eps})-F({x.x,x.y,x.z-eps}))/(2.f*eps);
        return Mat3f{dx.x,dy.x,dz.x, dx.y,dy.y,dz.y, dx.z,dy.z,dz.z};
    };
    Mat3f JX=jac_vf(X,p), JY=jac_vf(Y,p);
    Vec3f Yp=Y(p), Xp=X(p);
    return JY*Xp-JX*Yp;
}

// ─── Riemann curvature tensor (2D surface, scalar) ────────────────────────────

// R^u_vuv from Christoffel symbols (Gauss curvature = R^u_vuv / sqrt(det g))
[[nodiscard]] inline f32 riemann_scalar_2d(
    const ChristoffelSymbols& C, const ChristoffelSymbols& dC_u, const ChristoffelSymbols& dC_v)
{
    // R^0_101 = partial_0 Gamma^0_11 - partial_1 Gamma^0_01 + sum Gamma^0_0k Gamma^k_11 - Gamma^0_1k Gamma^k_01
    f32 R=dC_u.G[0][1][1]-dC_v.G[0][0][1]
         +C.G[0][0][0]*C.G[0][1][1]+C.G[0][0][1]*C.G[1][1][1]
         -C.G[0][1][0]*C.G[0][0][1]-C.G[0][1][1]*C.G[1][0][1];
    return R;
}

// ─── Shape operator (Weingarten map) ─────────────────────────────────────────

[[nodiscard]] inline std::array<f32,4> shape_operator(
    f32 E,f32 F,f32 G, f32 e,f32 f2,f32 g)
{
    // S = I^{-1} II (matrix product of inverse first form with second form)
    f32 det=E*G-F*F;
    if(std::abs(det)<1e-10f) return {0,0,0,0};
    f32 s00=(e*G-f2*F)/det, s01=(f2*G-g*F)/det;
    f32 s10=(f2*E-e*F)/det, s11=(g*E-f2*F)/det;
    return {s00,s01,s10,s11};
}

} // namespace qs
