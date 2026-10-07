#pragma once
// differential_geom.hpp — Differential geometry: manifolds, curvature,
//                          geodesics, connection, Lie groups, Riemannian metric,
//                          covariant derivative, parallel transport on surfaces
#include "vec2.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "quat.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "curves.hpp"

namespace qs::math::diffgeom {

// ---------------------------------------------------------------------------
// Surface differential geometry
// ---------------------------------------------------------------------------
struct SurfacePatch {
    // Parametric surface r(u,v)
    Vec3f position;       // r(u,v)
    Vec3f ru, rv;         // ∂r/∂u, ∂r/∂v
    Vec3f ruu,ruv,rvv;    // second derivatives

    [[nodiscard]] Vec3f normal() const noexcept { return cross(ru,rv).normalized(); }

    // First fundamental form coefficients (metric tensor)
    [[nodiscard]] f32 E() const noexcept { return dot(ru,ru); }
    [[nodiscard]] f32 F() const noexcept { return dot(ru,rv); }
    [[nodiscard]] f32 G() const noexcept { return dot(rv,rv); }

    // Second fundamental form coefficients (curvature)
    [[nodiscard]] f32 L() const noexcept { return dot(ruu,normal()); }
    [[nodiscard]] f32 M() const noexcept { return dot(ruv,normal()); }
    [[nodiscard]] f32 N_() const noexcept { return dot(rvv,normal()); }

    // Gaussian curvature K = (LN-M^2)/(EG-F^2)
    [[nodiscard]] f32 gaussian_curvature() const noexcept {
        f32 EG=E()*G()-F()*F();
        return abs(EG)>EPS_F32?(L()*N_()-M()*M())/EG:0.f;
    }

    // Mean curvature H = (EN - 2FM + GL)/(2(EG-F^2))
    [[nodiscard]] f32 mean_curvature() const noexcept {
        f32 EG=E()*G()-F()*F();
        return abs(EG)>EPS_F32?(E()*N_()-2.f*F()*M()+G()*L())/(2.f*EG):0.f;
    }

    // Principal curvatures k1, k2
    [[nodiscard]] void principal_curvatures(f32& k1, f32& k2) const noexcept {
        f32 H=mean_curvature(), K=gaussian_curvature();
        f32 disc=max(0.f,H*H-K);
        f32 sq=sqrt(disc);
        k1=H+sq; k2=H-sq;
    }

    // Principal directions (eigenvectors of shape operator)
    [[nodiscard]] Vec2f principal_dir1() const noexcept {
        f32 k1,k2; principal_curvatures(k1,k2);
        // Solve (II - k*I) v = 0
        f32 a=L()-k1*E(), b=M()-k1*F();
        if(abs(a)<EPS_F32&&abs(b)<EPS_F32) return {1.f,0.f};
        return Vec2f{-b,a}.normalized();
    }

    // Geodesic curvature of a curve on the surface
    [[nodiscard]] f32 geodesic_curvature(Vec3f tangent, Vec3f dtangent) const noexcept {
        Vec3f n=normal();
        return dot(dtangent,cross(n,tangent));
    }
};

// ---------------------------------------------------------------------------
// Geodesic on implicit surface (gradient flow)
// ---------------------------------------------------------------------------
// Gradient descent on surface: project velocity onto tangent plane
[[nodiscard]] constexpr Vec3f project_to_tangent(Vec3f v, Vec3f n) noexcept {
    return v-n*dot(v,n);
}

// Geodesic step: move along surface following tangent
template<typename SDF>
[[nodiscard]] inline Vec3f geodesic_step(Vec3f p, Vec3f dir, SDF&& sdf,
                                          f32 dt=0.01f) noexcept
{
    f32 eps=0.001f;
    Vec3f n=Vec3f{sdf({p.x+eps,p.y,p.z})-sdf({p.x-eps,p.y,p.z}),
                  sdf({p.x,p.y+eps,p.z})-sdf({p.x,p.y-eps,p.z}),
                  sdf({p.x,p.y,p.z+eps})-sdf({p.x,p.y,p.z-eps})}.normalized();
    Vec3f tangent=project_to_tangent(dir,n).normalized();
    Vec3f np=p+tangent*dt;
    // Project back to surface
    for(i32 iter=0;iter<10;++iter){
        f32 d=sdf(np);
        Vec3f gn=Vec3f{sdf({np.x+eps,np.y,np.z})-sdf({np.x-eps,np.y,np.z}),
                       sdf({np.x,np.y+eps,np.z})-sdf({np.x,np.y-eps,np.z}),
                       sdf({np.x,np.y,np.z+eps})-sdf({np.x,np.y,np.z-eps})};
        f32 gn_len=gn.length();
        if(gn_len<EPS_F32) break;
        np=np-gn*(d/gn_len);
        if(abs(d)<EPS_F32*0.1f) break;
    }
    return np;
}

// ---------------------------------------------------------------------------
// Lie groups (SO(3), SE(3))
// ---------------------------------------------------------------------------
// SO(3): rotation group
// Exponential map: so(3) → SO(3) via Rodrigues
[[nodiscard]] inline Mat3f so3_exp(Vec3f omega) noexcept {
    f32 theta=omega.length();
    if(theta<EPS_ANGLE) return Mat3f::identity();
    Vec3f k=omega*(1.f/theta);
    f32 c=cos(theta),s=sin(theta);
    // Rodrigues: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
    Mat3f K{Vec3f{0.f,k.z,-k.y},Vec3f{-k.z,0.f,k.x},Vec3f{k.y,-k.x,0.f}};
    return Mat3f::identity()+K*s+K*K*(1.f-c);
}

// Logarithm map: SO(3) → so(3)
[[nodiscard]] inline Vec3f so3_log(Mat3f R) noexcept {
    f32 trace=R[0].x+R[1].y+R[2].z;
    f32 theta=safe_acos(clamp((trace-1.f)*0.5f,-1.f,1.f));
    if(abs(theta)<EPS_ANGLE) return {0.f};
    f32 factor=theta/(2.f*sin(theta));
    return Vec3f{R[2].y-R[1].z,R[0].z-R[2].x,R[1].x-R[0].y}*factor;
}

// SE(3): rigid body transformation group
struct SE3 {
    Mat3f R{Mat3f::identity()};
    Vec3f t{0.f};

    SE3() noexcept = default;
    SE3(Mat3f R, Vec3f t) noexcept : R(R),t(t) {}

    [[nodiscard]] Vec3f apply(Vec3f p) const noexcept { return R*p+t; }
    [[nodiscard]] SE3 inverse() const noexcept { return {R.transposed(),R.transposed()*(-t)}; }
    [[nodiscard]] SE3 operator*(SE3 o) const noexcept { return {R*o.R,R*o.t+t}; }
};

// SE(3) exponential map (from twist ξ = [v; ω])
[[nodiscard]] inline SE3 se3_exp(Vec3f v, Vec3f omega) noexcept {
    f32 theta=omega.length();
    Mat3f R=so3_exp(omega);
    Vec3f tt;
    if(theta<EPS_ANGLE){
        tt=v;
    } else {
        Mat3f K{Vec3f{0.f,omega.z,-omega.y},Vec3f{-omega.z,0.f,omega.x},Vec3f{omega.y,-omega.x,0.f}};
        f32 t2=theta*theta;
        tt=(Mat3f::identity()+K*((1.f-cos(theta))/t2)+K*K*((theta-sin(theta))/(t2*theta)))*v;
    }
    return {R,tt};
}

// ---------------------------------------------------------------------------
// Riemannian metric tensor (pullback of Euclidean metric)
// ---------------------------------------------------------------------------
struct RiemannianMetric2D {
    f32 g11{1.f},g12{0.f},g22{1.f}; // metric tensor [g_ij]

    [[nodiscard]] f32 determinant() const noexcept { return g11*g22-g12*g12; }
    [[nodiscard]] f32 christoffel(i32 k, i32 i, i32 j) const noexcept { return 0.f; } // simplified flat

    // Arc length element: ds^2 = g_ij du^i du^j
    [[nodiscard]] constexpr f32 arc_length_elem(Vec2f du) const noexcept {
        return g11*du.x*du.x+2.f*g12*du.x*du.y+g22*du.y*du.y;
    }

    // Area element: dA = sqrt(det(g)) du dv
    [[nodiscard]] f32 area_element() const noexcept { return sqrt(max(0.f,determinant())); }
};

// ---------------------------------------------------------------------------
// Parallel transport on a surface (discrete, along a path)
// ---------------------------------------------------------------------------
inline void parallel_transport_path(
    const Vec3f* path, i32 n,  // positions on surface
    Vec3f initial_vec,          // vector to transport
    Vec3f* transported) noexcept
{
    transported[0]=initial_vec;
    for(i32 i=1;i<n;++i){
        Vec3f t=(path[i]-path[i-1]).normalized();
        Vec3f prev_t=(i>1?(path[i-1]-path[i-2]).normalized():t);
        Vec3f rot_axis=cross(prev_t,t);
        f32   rot_len=rot_axis.length();
        if(rot_len<EPS_F32){ transported[i]=transported[i-1]; continue; }
        f32 angle=safe_acos(clamp(dot(prev_t,t),-1.f,1.f));
        Vec3f k=rot_axis*(1.f/rot_len);
        // Rodrigues rotation
        transported[i]=transported[i-1]*cos(angle)+cross(k,transported[i-1])*sin(angle)
                       +k*dot(k,transported[i-1])*(1.f-cos(angle));
    }
}

} // namespace qs::math::diffgeom
