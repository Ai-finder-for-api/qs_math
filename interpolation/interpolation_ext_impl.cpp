
// qs_math/interpolation/interpolation_ext_impl.cpp (C++20)
#include "include/qs_math_interpolation.hpp"
#include <cmath>
#include <vector>
#include <cassert>
namespace qs::math::interp {

// ─── SLERP ────────────────────────────────────────────────────────────────────
Quat<f32> slerp(Quat<f32> a, Quat<f32> b, f32 t) noexcept {
    f32 dot = a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;
    if(dot<0){b.w=-b.w;b.x=-b.x;b.y=-b.y;b.z=-b.z;dot=-dot;}
    if(dot>0.9995f) return nlerp(a,b,t);
    const f32 theta=std::acos(dot);
    const f32 st=std::sin(theta);
    const f32 s0=std::sin((1-t)*theta)/st;
    const f32 s1=std::sin(t*theta)/st;
    return {a.w*s0+b.w*s1,a.x*s0+b.x*s1,a.y*s0+b.y*s1,a.z*s0+b.z*s1};
}
Quat<f32> nlerp(Quat<f32> a, Quat<f32> b, f32 t) noexcept {
    f32 dot=a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;
    if(dot<0){b.w=-b.w;b.x=-b.x;b.y=-b.y;b.z=-b.z;}
    Quat<f32> r{a.w+(b.w-a.w)*t,a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,a.z+(b.z-a.z)*t};
    const f32 len=std::sqrt(r.w*r.w+r.x*r.x+r.y*r.y+r.z*r.z);
    if(len>1e-7f){r.w/=len;r.x/=len;r.y/=len;r.z/=len;}
    return r;
}
static Quat<f32> qlog(Quat<f32> q) noexcept {
    const f32 vlen=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
    const f32 theta=vlen<1e-7f?0:std::atan2(vlen,q.w)/vlen;
    return {0,q.x*theta,q.y*theta,q.z*theta};
}
static Quat<f32> qexp(Quat<f32> q) noexcept {
    const f32 vlen=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
    const f32 s=vlen<1e-7f?1:std::sin(vlen)/vlen;
    return {std::cos(vlen),q.x*s,q.y*s,q.z*s};
}
static Quat<f32> qmul(Quat<f32> a,Quat<f32> b) noexcept {
    return{a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
           a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
           a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
           a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w};
}
static Quat<f32> qconj(Quat<f32> q){return{q.w,-q.x,-q.y,-q.z};}
static Quat<f32> qadd(Quat<f32> a,Quat<f32> b){return{a.w+b.w,a.x+b.x,a.y+b.y,a.z+b.z};}
static Quat<f32> qscale(Quat<f32> a,f32 s){return{a.w*s,a.x*s,a.y*s,a.z*s};}
Quat<f32> squad_tangent(Quat<f32> p,Quat<f32> q,Quat<f32> n) noexcept {
    Quat<f32> qinv=qconj(q);
    return qmul(q,qexp(qscale(qadd(qlog(qmul(qinv,p)),qlog(qmul(qinv,n))),-0.25f)));
}
Quat<f32> squad(Quat<f32> q0,Quat<f32> q1,Quat<f32> s0,Quat<f32> s1,f32 t) noexcept {
    return slerp(slerp(q0,q1,t),slerp(s0,s1,t),2*t*(1-t));
}
// ─── RBF ─────────────────────────────────────────────────────────────────────
f64 RBFInterp::phi(f64 r) const noexcept {
    switch(kernel){
    case Kernel::Gaussian:        return std::exp(-epsilon*epsilon*r*r);
    case Kernel::Multiquadric:    return std::sqrt(1+epsilon*epsilon*r*r);
    case Kernel::InvMultiquadric: return 1.0/std::sqrt(1+epsilon*epsilon*r*r);
    case Kernel::ThinPlate:       return r>1e-14?r*r*std::log(r):0;
    default: return 0;
    }
}
void RBFInterp::fit(const Vec2<f32>* pts,const f64* vals,i32 n) noexcept {
    centers.assign(pts,pts+n);
    // Build n×n matrix A[i][j]=phi(|pts[i]-pts[j]|)
    std::vector<std::vector<f64>> A(n,std::vector<f64>(n));
    for(i32 i=0;i<n;++i) for(i32 j=0;j<n;++j){
        f64 r=std::sqrt((f64)((pts[i]-pts[j]).length_sq()));
        A[i][j]=phi(r);
    }
    // Solve A*w=vals via Gaussian elimination
    std::vector<f64> b(vals,vals+n);
    std::vector<std::vector<f64>> M(n,std::vector<f64>(n+1));
    for(i32 i=0;i<n;++i){for(i32 j=0;j<n;++j)M[i][j]=A[i][j];M[i][n]=b[i];}
    for(i32 col=0;col<n;++col){
        i32 piv=col;
        for(i32 r=col+1;r<n;++r) if(std::abs(M[r][col])>std::abs(M[piv][col]))piv=r;
        std::swap(M[col],M[piv]);
        if(std::abs(M[col][col])<1e-14) continue;
        const f64 inv=1.0/M[col][col];
        for(i32 r=col+1;r<n;++r){
            const f64 f=M[r][col]*inv;
            for(i32 j=col;j<=n;++j)M[r][j]-=f*M[col][j];
        }
    }
    weights.resize(n);
    for(i32 i=n-1;i>=0;--i){
        f64 s=M[i][n];
        for(i32 j=i+1;j<n;++j)s-=M[i][j]*weights[j];
        weights[i]=std::abs(M[i][i])<1e-14?0:s/M[i][i];
    }
}
f64 RBFInterp::eval(Vec2<f32> p) const noexcept {
    f64 sum=0;
    for(i32 i=0;i<(i32)centers.size();++i){
        f64 r=std::sqrt((f64)((p-centers[i]).length_sq()));
        sum+=weights[i]*phi(r);
    }
    return sum;
}
// ─── Trilinear field ─────────────────────────────────────────────────────────
f64 trilinear_field(const f64 c[8],f64 u,f64 v,f64 w) noexcept {
    return c[0]*(1-u)*(1-v)*(1-w)+c[1]*u*(1-v)*(1-w)+
           c[2]*(1-u)*v*(1-w)   +c[3]*u*v*(1-w)+
           c[4]*(1-u)*(1-v)*w   +c[5]*u*(1-v)*w+
           c[6]*(1-u)*v*w       +c[7]*u*v*w;
}
// ─── Bezier surface ───────────────────────────────────────────────────────────
static Vec3<f32> bez_row(const Vec3<f32>* p,f32 t){
    f32 it=1-t;
    return p[0]*(it*it*it)+p[1]*(3*it*it*t)+p[2]*(3*it*t*t)+p[3]*(t*t*t);
}
Vec3<f32> bezier_surface(const Vec3<f32> ctrl[4][4],f32 u,f32 v) noexcept {
    Vec3<f32> rows[4];
    for(i32 i=0;i<4;++i) rows[i]=bez_row(ctrl[i],u);
    return bez_row(rows,v);
}
Vec3<f32> bezier_surface_normal(const Vec3<f32> ctrl[4][4],f32 u,f32 v) noexcept {
    // Partial derivatives via finite differences
    const f32 h=1e-4f;
    Vec3<f32> du=(bezier_surface(ctrl,u+h,v)-bezier_surface(ctrl,u-h,v))*(0.5f/h);
    Vec3<f32> dv=(bezier_surface(ctrl,u,v+h)-bezier_surface(ctrl,u,v-h))*(0.5f/h);
    return du.cross(dv).normalized();
}
Vec3<f32> bspline_surface(const Vec3<f32>* ctrl,i32 nu,i32 nv,i32 pu,i32 pv,
    const f64* ku,const f64* kv,f64 u,f64 v) noexcept
{
    // Evaluate as tensor product of two 1D B-splines
    // Simplified: evaluate along v-direction then u-direction
    std::vector<Vec3<f32>> col(nv);
    for(i32 jv=0;jv<nv;++jv){
        // Extract control polygon at constant v-index jv
        std::vector<Vec3<f32>> row(nu);
        for(i32 iu=0;iu<nu;++iu) row[iu]=ctrl[iu*nv+jv];
        // Evaluate B-spline in u: use de Boor
        // Simplified nearest-knot evaluation
        col[jv]=row[std::min(nu-1,(i32)(u*(nu-pu)))];
    }
    return col[std::min(nv-1,(i32)(v*(nv-pv)))];
}
} // namespace qs::math::interp
