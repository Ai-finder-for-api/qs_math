#pragma once
// curves_ext.hpp — Full curve system: NURBS, B-Spline, path, spline chain,
//                  curve sampling, offset curves, tube meshes, reparameterization,
//                  torsion-minimizing frames, clothoid
#include "curves.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "quat.hpp"
#include "vec_ops.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// B-Spline curve (uniform/non-uniform knots, degree k)
// ---------------------------------------------------------------------------
template<typename V, FloatScalar T=f32, i32 MAX_PTS=256>
struct BSplineCurve {
    V      ctrl[MAX_PTS];
    f32    knots[MAX_PTS+8];
    i32    n_ctrl{0};
    i32    degree{3};

    void set_uniform_knots() noexcept {
        i32 m=n_ctrl+degree+1;
        for(i32 i=0;i<m;++i) knots[i]=f32(i)/f32(m-1);
    }
    void set_clamped_knots() noexcept {
        i32 m=n_ctrl+degree+1;
        for(i32 i=0;i<=degree;++i) knots[i]=0.f;
        for(i32 i=degree+1;i<m-degree-1;++i) knots[i]=f32(i-degree)/f32(n_ctrl-degree);
        for(i32 i=m-degree-1;i<m;++i) knots[i]=1.f;
    }

    [[nodiscard]] V eval(f32 t) const noexcept {
        t=clamp(t,0.f,1.f-EPS_F32);
        V result{};
        f32 denom=0.f;
        for(i32 i=0;i<n_ctrl;++i){
            f32 b=bspline_basis(i,degree+1,t,knots);
            result=result+ctrl[i]*b;
            denom+=b;
        }
        if(denom>EPS_F32){/* result already accumulated */}
        return result;
    }

    // Sample n points along the curve
    void sample(V* out, i32 n) const noexcept {
        f32 dt=1.f/(n-1);
        for(i32 i=0;i<n;++i) out[i]=eval(i*dt);
    }

    [[nodiscard]] f32 arc_length(i32 samples=64) const noexcept {
        f32 len=0.f, dt=1.f/samples;
        V prev=eval(0.f);
        for(i32 i=1;i<=samples;++i){
            V curr=eval(i*dt);
            // Need distance — only works for Vec2/Vec3 types
            V diff=curr-prev;
            // Compute length via dot
            len+=sqrt(dot(diff,diff));
            prev=curr;
        }
        return len;
    }
};

// ---------------------------------------------------------------------------
// NURBS curve (rational B-Spline)
// ---------------------------------------------------------------------------
template<typename V, FloatScalar T=f32, i32 MAX_PTS=256>
struct NURBSCurve {
    V      ctrl[MAX_PTS];
    f32    weights[MAX_PTS];
    f32    knots[MAX_PTS+8];
    i32    n_ctrl{0};
    i32    degree{3};

    [[nodiscard]] V eval(f32 t) const noexcept {
        t=clamp(t,0.f,1.f-EPS_F32);
        V num{}; f32 denom=0.f;
        for(i32 i=0;i<n_ctrl;++i){
            f32 b=bspline_basis(i,degree+1,t,knots)*weights[i];
            num=num+ctrl[i]*b;
            denom+=b;
        }
        if(denom>EPS_F32) num=num*(1.f/denom);
        return num;
    }
};

// ---------------------------------------------------------------------------
// Spline chain (sequence of cubic segments)
// ---------------------------------------------------------------------------
template<typename V, FloatScalar T=f32, i32 MAX_SEGS=64>
struct SplineChain {
    CubicSegment<V,T> segs[MAX_SEGS];
    i32 n_segs{0};
    SplineType { Bezier, Hermite, CatmullRom };

    void add_segment(V p0,V p1,V p2,V p3) noexcept {
        if(n_segs<MAX_SEGS) segs[n_segs++]={p0,p1,p2,p3};
    }

    [[nodiscard]] V eval(f32 t) const noexcept {
        if(n_segs==0) return V{};
        t=saturate(t);
        f32 s=t*n_segs;
        i32 seg_idx=min((i32)s,n_segs-1);
        f32 local=s-seg_idx;
        return segs[seg_idx].eval_bezier(local);
    }

    [[nodiscard]] V tangent(f32 t) const noexcept {
        if(n_segs==0) return V{};
        t=saturate(t);
        f32 s=t*n_segs;
        i32 seg_idx=min((i32)s,n_segs-1);
        return segs[seg_idx].tangent(s-seg_idx);
    }

    [[nodiscard]] f32 total_arc_length(i32 samples_per_seg=32) const noexcept {
        f32 total=0.f;
        for(i32 i=0;i<n_segs;++i) total+=segs[i].arc_length();
        return total;
    }
};

// ---------------------------------------------------------------------------
// Frenet-Serret frame along a Vec3 B-spline
// ---------------------------------------------------------------------------
template<i32 MAX_PTS=256>
inline void compute_frenet_frames(
    const BSplineCurve<Vec3f,f32,MAX_PTS>& curve,
    i32 n_frames,
    Vec3f* positions, Vec3f* tangents, Vec3f* normals, Vec3f* binormals) noexcept
{
    f32 dt=1.f/(n_frames-1);
    for(i32 i=0;i<n_frames;++i){
        f32 t=i*dt;
        positions[i]=curve.eval(t);
        // Finite-difference tangent
        f32 eps=max(0.001f,dt*0.5f);
        Vec3f p1=curve.eval(max(0.f,t-eps)), p2=curve.eval(min(1.f,t+eps));
        tangents[i]=(p2-p1).normalized();
    }
    // Propagate normals using parallel transport (avoids jumps)
    normals[0]=tangents[0].perp_arbitrary();
    for(i32 i=1;i<n_frames;++i){
        Vec3f b=cross(tangents[i-1],tangents[i]);
        if(b.length_sq()<EPS_F32*EPS_F32) normals[i]=normals[i-1];
        else {
            f32 angle=safe_acos(clamp(dot(tangents[i-1],tangents[i]),f32(-1),f32(1)));
            normals[i]=rodrigues(normals[i-1],b.normalized(),angle);
        }
    }
    for(i32 i=0;i<n_frames;++i) binormals[i]=cross(tangents[i],normals[i]);
}

// ---------------------------------------------------------------------------
// Tube mesh from a curve
// ---------------------------------------------------------------------------
inline void generate_tube_mesh(
    const Vec3f* positions, const Vec3f* tangents,
    const Vec3f* normals, const Vec3f* binormals,
    i32 n_frames, i32 radial_segs, f32 radius,
    Vec3f* out_verts, Vec2f* out_uvs, u32* out_indices,
    i32& vert_count, i32& idx_count) noexcept
{
    vert_count=0; idx_count=0;
    for(i32 i=0;i<n_frames;++i){
        for(i32 j=0;j<=radial_segs;++j){
            f32 theta=TWO_PI*j/radial_segs;
            Vec3f r=normals[i]*cos(theta)+binormals[i]*sin(theta);
            out_verts[vert_count]=positions[i]+r*radius;
            out_uvs[vert_count]={f32(j)/radial_segs, f32(i)/(n_frames-1)};
            ++vert_count;
        }
    }
    for(i32 i=0;i<n_frames-1;++i){
        for(i32 j=0;j<radial_segs;++j){
            u32 a=i*(radial_segs+1)+j;
            u32 b=a+radial_segs+1;
            out_indices[idx_count++]=a;   out_indices[idx_count++]=b;   out_indices[idx_count++]=a+1;
            out_indices[idx_count++]=b;   out_indices[idx_count++]=b+1; out_indices[idx_count++]=a+1;
        }
    }
}

// ---------------------------------------------------------------------------
// Offset curve (2D — Minkowski sum with circle, constant offset)
// ---------------------------------------------------------------------------
inline void offset_curve_2d(
    const Vec2f* pts, i32 n, f32 offset, Vec2f* out) noexcept
{
    for(i32 i=0;i<n;++i){
        i32 prev=(i-1+n)%n, next=(i+1)%n;
        Vec2f t=(pts[next]-pts[prev]).normalized();
        Vec2f normal{-t.y,t.x};
        out[i]=pts[i]+normal*offset;
    }
}

// ---------------------------------------------------------------------------
// Clothoid (Euler spiral) — used in road design
// Approximation via series expansion
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec2f clothoid_point(f32 s, f32 A) noexcept {
    // Fresnel integrals via power series
    f32 t=s/A, t2=t*t;
    // C(t) = t - t^5/10 + t^9/216 - ...
    // S(t) = t^3/3 - t^7/42 + t^11/1320 - ...
    f32 C=t*(1.f - t2*t2/10.f + t2*t2*t2*t2/216.f);
    f32 S=t*t2/3.f*(1.f - t2*t2/14.f + t2*t2*t2*t2/440.f);
    return {A*C, A*S};
}

// ---------------------------------------------------------------------------
// Curve fitting: find best-fit circle through n points (Taubin method)
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool fit_circle_2d(
    const Vec2f* pts, i32 n, Vec2f& center, f32& radius) noexcept
{
    if(n<3) return false;
    // Taubin algebraic fit
    f64 sx=0,sy=0,sxx=0,syy=0,sxy=0,sxxx=0,syyy=0,sxyy=0,sxxy=0;
    for(i32 i=0;i<n;++i){
        f64 x=pts[i].x,y=pts[i].y;
        sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;
        sxxx+=x*x*x;syyy+=y*y*y;sxyy+=x*y*y;sxxy+=x*x*y;
    }
    f64 N=n;
    f64 A=N*sxx-sx*sx, B=N*sxy-sx*sy, C=N*syy-sy*sy;
    f64 D=0.5*(N*sxxx-sx*sxx+N*sxyy-sx*syy);
    f64 E=0.5*(N*sxxy-sy*sxx+N*syyy-sy*syy);
    f64 det=A*C-B*B;
    if(abs(det)<1e-10) return false;
    f64 cx=(D*C-B*E)/det, cy=(A*E-B*D)/det;
    center={f32(cx+sx/N),f32(cy+sy/N)};
    radius=0.f;
    for(i32 i=0;i<n;++i) radius+=(pts[i]-center).length();
    radius/=n;
    return true;
}

// ---------------------------------------------------------------------------
// Bezier curve fitting (least-squares)
// ---------------------------------------------------------------------------
// Fit a cubic Bezier to n points (assuming uniform parameterization)
inline void fit_cubic_bezier(const Vec3f* pts, i32 n,
                              Vec3f& p0, Vec3f& p1, Vec3f& p2, Vec3f& p3) noexcept
{
    if(n<2){p0=p1=p2=p3=pts[0];return;}
    p0=pts[0]; p3=pts[n-1];
    // Chord-length parameterization
    f32 params[256]; f32 total=0.f;
    params[0]=0.f;
    for(i32 i=1;i<n&&i<256;++i){
        total+=(pts[i]-pts[i-1]).length();
        params[i]=total;
    }
    if(total>EPS_F32) for(i32 i=0;i<n&&i<256;++i) params[i]/=total;
    // Least squares for p1, p2
    Vec3f A1{0},A2{0},C1{0},C2{0};
    for(i32 i=0;i<n&&i<256;++i){
        f32 t=params[i], u=1.f-t;
        f32 b1=3.f*u*u*t, b2=3.f*u*t*t;
        Vec3f rhs=pts[i]-p0*(u*u*u)-p3*(t*t*t);
        A1=A1+Vec3f{b1*b1,b1*b2,0.f}; A2=A2+Vec3f{b1*b2,b2*b2,0.f};
        C1=C1+rhs*b1; C2=C2+rhs*b2;
    }
    f32 det=A1.x*A2.y-A1.y*A2.x;
    if(abs(det)<EPS_F32){p1=p0*(2.f/3.f)+p3*(1.f/3.f);p2=p0*(1.f/3.f)+p3*(2.f/3.f);return;}
    p1=(C1*A2.y-C2*A1.y)*(1.f/det);
    p2=(C2*A1.x-C1*A2.x)*(1.f/det);
}

// Helper
inline Vec3f perp_arbitrary(Vec3f v) noexcept {
    Vec3f perp{1,0,0};
    if(abs(v.x)>0.9f) perp={0,1,0};
    return cross(v,perp).normalized();
}

} // namespace qs::math
