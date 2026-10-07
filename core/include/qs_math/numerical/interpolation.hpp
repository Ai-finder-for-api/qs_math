#pragma once
// qs_math/numerical/interpolation.hpp — Interpolation: cubic splines, Bezier,
//   B-spline, NURBS, Catmull-Rom, Hermite, barycentric, RBF, Shepard (C++20)
#include "../linear_algebra/vec2.hpp"
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec4.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <span>

namespace qs::math::numerical {

// ─── Polynomial interpolation (Lagrange) ──────────────────────────────────
[[nodiscard]] inline f64 lagrange(
    std::span<const f64> xs, std::span<const f64> ys, f64 x) noexcept
{
    int n=int(xs.size()); f64 result=0.0;
    for(int i=0;i<n;++i){
        f64 term=ys[i];
        for(int j=0;j<n;++j){
            if(j!=i) term*=(x-xs[j])/(xs[i]-xs[j]);
        }
        result+=term;
    }
    return result;
}

// ─── Newton's divided differences ─────────────────────────────────────────
[[nodiscard]] inline std::vector<f64> newton_div_diff(
    std::span<const f64> xs, std::span<const f64> ys) noexcept
{
    int n=int(xs.size());
    std::vector<f64> F(ys.begin(),ys.end());
    for(int j=1;j<n;++j)
        for(int i=n-1;i>=j;--i)
            F[i]=(F[i]-F[i-1])/(xs[i]-xs[i-j]);
    return F; // Newton coefficients
}

[[nodiscard]] inline f64 newton_eval(
    std::span<const f64> xs, std::span<const f64> coeff, f64 x) noexcept
{
    int n=int(xs.size()); f64 r=coeff[n-1];
    for(int i=n-2;i>=0;--i) r=r*(x-xs[i])+coeff[i];
    return r;
}

// ─── Cubic natural spline ─────────────────────────────────────────────────
struct CubicSpline {
    std::vector<f64> x,a,b,c,d; // S_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3
    int n;

    void build(std::span<const f64> xs, std::span<const f64> ys) {
        n=int(xs.size())-1;
        x.assign(xs.begin(),xs.end());
        a.assign(ys.begin(),ys.end());
        b.resize(n+1,0); c.resize(n+1,0); d.resize(n,0);
        std::vector<f64> h(n),alpha(n),l(n+1,1),mu(n+1,0),z(n+1,0);
        for(int i=0;i<n;++i) h[i]=xs[i+1]-xs[i];
        for(int i=1;i<n;++i)
            alpha[i]=3.0/h[i]*(a[i+1]-a[i])-3.0/h[i-1]*(a[i]-a[i-1]);
        for(int i=1;i<n;++i){
            l[i]=2.0*(xs[i+1]-xs[i-1])-h[i-1]*mu[i-1];
            mu[i]=h[i]/l[i]; z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i];
        }
        for(int j=n-1;j>=0;--j){
            c[j]=z[j]-mu[j]*c[j+1];
            b[j]=(a[j+1]-a[j])/h[j]-h[j]*(c[j+1]+2.0*c[j])/3.0;
            d[j]=(c[j+1]-c[j])/(3.0*h[j]);
        }
    }

    [[nodiscard]] f64 eval(f64 t) const noexcept {
        int i=0;
        for(int k=0;k<n-1;++k) if(t>=x[k+1]) i=k+1;
        f64 dt=t-x[i];
        return a[i]+b[i]*dt+c[i]*dt*dt+d[i]*dt*dt*dt;
    }

    [[nodiscard]] f64 deriv(f64 t) const noexcept {
        int i=0;
        for(int k=0;k<n-1;++k) if(t>=x[k+1]) i=k+1;
        f64 dt=t-x[i];
        return b[i]+2.0*c[i]*dt+3.0*d[i]*dt*dt;
    }
};

// ─── Catmull-Rom spline ───────────────────────────────────────────────────
[[nodiscard]] inline Vec3f catmull_rom(Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3, f32 t) noexcept {
    f32 t2=t*t, t3=t2*t;
    return (p0*(-t3+2*t2-t) + p1*(3*t3-5*t2+2) +
            p2*(-3*t3+4*t2+t) + p3*(t3-t2)) * 0.5f;
}

[[nodiscard]] inline Vec3f catmull_rom_tangent(Vec3f p0,Vec3f p1,Vec3f p2,Vec3f p3,f32 t) noexcept {
    f32 t2=t*t;
    return (p0*(-3*t2+4*t-1) + p1*(9*t2-10*t) +
            p2*(-9*t2+8*t+1) + p3*(3*t2-2*t)) * 0.5f;
}

// Chordal/centripetal parameterization (alpha=0.5 for centripetal)
[[nodiscard]] inline f32 cr_knot_interval(Vec3f a, Vec3f b, f32 alpha=0.5f) noexcept {
    f32 d=(b-a).length();
    return std::pow(d,alpha);
}

// ─── Hermite spline ───────────────────────────────────────────────────────
// Cubic Hermite: interpolate (p0,m0) -> (p1,m1) with tangents m0,m1
[[nodiscard]] inline Vec3f hermite(Vec3f p0,Vec3f m0,Vec3f p1,Vec3f m1,f32 t) noexcept {
    f32 t2=t*t,t3=t2*t;
    f32 h00=2*t3-3*t2+1, h10=t3-2*t2+t, h01=-2*t3+3*t2, h11=t3-t2;
    return p0*h00 + m0*h10 + p1*h01 + m1*h11;
}

// ─── Bezier curves ────────────────────────────────────────────────────────
// De Casteljau algorithm (stable for any degree)
[[nodiscard]] inline Vec3f bezier_decasteljau(std::span<const Vec3f> pts, f32 t) noexcept {
    int n=int(pts.size());
    std::vector<Vec3f> p(pts.begin(),pts.end());
    for(int r=1;r<n;++r)
        for(int i=0;i<n-r;++i)
            p[i]=p[i]*(1.f-t)+p[i+1]*t;
    return p[0];
}

// Cubic Bezier (closed form)
[[nodiscard]] inline Vec3f bezier3(Vec3f p0,Vec3f p1,Vec3f p2,Vec3f p3,f32 t) noexcept {
    f32 u=1.f-t,u2=u*u,u3=u2*u,t2=t*t,t3=t2*t;
    return p0*u3 + p1*(3*u2*t) + p2*(3*u*t2) + p3*t3;
}

[[nodiscard]] inline Vec3f bezier3_deriv(Vec3f p0,Vec3f p1,Vec3f p2,Vec3f p3,f32 t) noexcept {
    f32 u=1.f-t;
    return (p1-p0)*(3*u*u) + (p2-p1)*(6*u*t) + (p3-p2)*(3*t*t);
}

// Arc length parameterization via Gaussian quadrature (7-point)
[[nodiscard]] inline f32 bezier3_arc_length(Vec3f p0,Vec3f p1,Vec3f p2,Vec3f p3,
                                             f32 t0=0.f,f32 t1=1.f) noexcept
{
    static constexpr f64 W[]={0.4179591836734694,0.3818300505051189,0.3818300505051189,
                               0.2797053914892767,0.2797053914892767,0.1294849661688697,0.1294849661688697};
    static constexpr f64 X[]={0,0.4058451513773832,-0.4058451513773832,
                               0.7415311855993945,-0.7415311855993945,0.9491079123427585,-0.9491079123427585};
    f32 len=0, mid=0.5f*(t0+t1), half=0.5f*(t1-t0);
    for(int i=0;i<7;++i){
        f32 t=mid+half*f32(X[i]);
        len+=f32(W[i])*bezier3_deriv(p0,p1,p2,p3,t).length();
    }
    return len*half;
}

// ─── B-spline ─────────────────────────────────────────────────────────────
// Cox-de Boor recursion
[[nodiscard]] inline f32 bspline_basis(int i, int k, f32 t,
                                        std::span<const f32> knots) noexcept
{
    if(k==1) return (t>=knots[i]&&t<knots[i+1])?1.f:0.f;
    f32 d1=knots[i+k-1]-knots[i], d2=knots[i+k]-knots[i+1];
    f32 a=(d1>1e-9f)?(t-knots[i])/d1:0.f;
    f32 b=(d2>1e-9f)?(knots[i+k]-t)/d2:0.f;
    return a*bspline_basis(i,k-1,t,knots)+b*bspline_basis(i+1,k-1,t,knots);
}

[[nodiscard]] inline Vec3f bspline_eval(
    std::span<const Vec3f> ctrl, std::span<const f32> knots, int order, f32 t) noexcept
{
    int n=int(ctrl.size());
    Vec3f result{0,0,0};
    for(int i=0;i<n;++i){
        f32 b=bspline_basis(i,order,t,knots);
        result=result+ctrl[i]*b;
    }
    return result;
}

// Uniform clamped knot vector generation
[[nodiscard]] inline std::vector<f32> bspline_clamped_knots(int n_ctrl, int order) noexcept {
    int m=n_ctrl+order;
    std::vector<f32> knots(m);
    for(int i=0;i<order;++i) knots[i]=0.f;
    for(int i=order;i<n_ctrl;++i) knots[i]=f32(i-order+1)/f32(n_ctrl-order+1);
    for(int i=n_ctrl;i<m;++i) knots[i]=1.f;
    return knots;
}

// ─── NURBS ────────────────────────────────────────────────────────────────
struct NURBSCurve {
    std::vector<Vec3f> ctrl;
    std::vector<f32>   weights;
    std::vector<f32>   knots;
    int order;

    [[nodiscard]] Vec3f eval(f32 t) const noexcept {
        int n=int(ctrl.size());
        Vec3f num{0,0,0}; f32 denom=0.f;
        for(int i=0;i<n;++i){
            f32 b=bspline_basis(i,order,t,knots)*weights[i];
            num=num+ctrl[i]*b; denom+=b;
        }
        return denom>1e-9f?num*(1.f/denom):Vec3f{0,0,0};
    }
};

// ─── Radial basis function (RBF) interpolation ────────────────────────────
enum class RBFKernel { Gaussian, Multiquadric, InvMultiquadric, ThinPlate, Linear };

[[nodiscard]] inline f64 rbf_kernel(f64 r, RBFKernel k, f64 eps=1.0) noexcept {
    switch(k){
        case RBFKernel::Multiquadric:    return std::sqrt(r*r+eps*eps);
        case RBFKernel::InvMultiquadric: return 1.0/std::sqrt(r*r+eps*eps);
        case RBFKernel::ThinPlate:       return (r>1e-14)?r*r*std::log(r):0.0;
        case RBFKernel::Linear:          return r;
        default: return std::exp(-eps*eps*r*r); // Gaussian
    }
}

struct RBFInterpolant {
    std::vector<Vec3f> centers;
    std::vector<f64>   weights_rbf;
    RBFKernel kernel; f64 eps;

    void build(std::span<const Vec3f> pts, std::span<const f64> vals,
               RBFKernel k=RBFKernel::Gaussian, f64 e=1.0) {
        kernel=k; eps=e; centers.assign(pts.begin(),pts.end());
        int n=int(pts.size());
        // Build and solve A*w=f (brute force Gaussian elimination)
        std::vector<f64> A(n*n);
        for(int i=0;i<n;++i) for(int j=0;j<n;++j)
            A[i*n+j]=rbf_kernel((pts[i]-pts[j]).length(),k,e);
        // Solve via simple Gauss (for small n)
        std::vector<f64> rhs(vals.begin(),vals.end());
        for(int col=0;col<n;++col){
            int pivot=col;
            for(int row=col+1;row<n;++row)
                if(std::abs(A[row*n+col])>std::abs(A[pivot*n+col])) pivot=row;
            for(int j=0;j<n;++j) std::swap(A[col*n+j],A[pivot*n+j]);
            std::swap(rhs[col],rhs[pivot]);
            for(int row=col+1;row<n;++row){
                f64 f=A[row*n+col]/A[col*n+col];
                for(int j=col;j<n;++j) A[row*n+j]-=f*A[col*n+j];
                rhs[row]-=f*rhs[col];
            }
        }
        weights_rbf.resize(n);
        for(int i=n-1;i>=0;--i){
            weights_rbf[i]=rhs[i];
            for(int j=i+1;j<n;++j) weights_rbf[i]-=A[i*n+j]*weights_rbf[j];
            weights_rbf[i]/=A[i*n+i];
        }
    }

    [[nodiscard]] f64 eval(Vec3f p) const noexcept {
        f64 s=0;
        for(int i=0;i<int(centers.size());++i)
            s+=weights_rbf[i]*rbf_kernel((p-centers[i]).length(),kernel,eps);
        return s;
    }
};

// ─── Shepard (inverse distance weighting) ────────────────────────────────
[[nodiscard]] inline f64 shepard_interp(
    std::span<const Vec3f> pts, std::span<const f64> vals,
    Vec3f query, f32 power=2.f) noexcept
{
    f64 num=0, denom=0;
    for(int i=0;i<int(pts.size());++i){
        f32 d=(query-pts[i]).length();
        if(d<1e-7f) return vals[i];
        f64 w=1.0/std::pow(f64(d),f64(power));
        num+=w*vals[i]; denom+=w;
    }
    return denom>1e-300?num/denom:0.0;
}

// ─── Bilinear / trilinear interpolation ───────────────────────────────────
[[nodiscard]] inline f32 bilinear(f32 f00,f32 f10,f32 f01,f32 f11,f32 tx,f32 ty) noexcept {
    return lerp(lerp(f00,f10,tx), lerp(f01,f11,tx), ty);
}

[[nodiscard]] inline f32 trilinear(
    f32 f000,f32 f100,f32 f010,f32 f110,
    f32 f001,f32 f101,f32 f011,f32 f111,
    f32 tx,f32 ty,f32 tz) noexcept
{
    return lerp(bilinear(f000,f100,f010,f110,tx,ty),
                bilinear(f001,f101,f011,f111,tx,ty), tz);
}

// Bicubic coefficients (16 samples)
[[nodiscard]] inline f32 bicubic(const f32 p[4][4], f32 tx, f32 ty) noexcept {
    auto cubic=[](const f32 v[4],f32 t)->f32{
        return v[1]+(t*(-0.5f*v[0]+0.5f*v[2])
               +t*t*(v[0]-2.5f*v[1]+2.f*v[2]-0.5f*v[3])
               +t*t*t*(-0.5f*v[0]+1.5f*v[1]-1.5f*v[2]+0.5f*v[3]));
    };
    f32 row[4];
    for(int i=0;i<4;++i) row[i]=cubic(p[i],tx);
    return cubic(row,ty);
}

// ─── Barycentric interpolation on triangle ────────────────────────────────
struct BaryCoords { f32 u,v,w; bool valid; };

[[nodiscard]] inline BaryCoords barycentric(Vec3f p, Vec3f a, Vec3f b, Vec3f c) noexcept {
    Vec3f v0=b-a,v1=c-a,v2=p-a;
    f32 d00=dot(v0,v0),d01=dot(v0,v1),d11=dot(v1,v1);
    f32 d20=dot(v2,v0),d21=dot(v2,v1);
    f32 denom=d00*d11-d01*d01;
    if(std::abs(denom)<1e-10f) return {0,0,0,false};
    f32 v=(d11*d20-d01*d21)/denom;
    f32 w=(d00*d21-d01*d20)/denom;
    return {1.f-v-w, v, w, true};
}

template<typename T>
[[nodiscard]] inline T bary_interp(T va, T vb, T vc, BaryCoords b) noexcept {
    return va*b.u + vb*b.v + vc*b.w;
}

// ─── Monotone cubic interpolation (Fritsch-Carlson) ─────────────────────
struct MonoCubicSpline {
    std::vector<f64> xs,ys,ms; // positions, values, slopes

    void build(std::span<const f64> x, std::span<const f64> y) {
        int n=int(x.size()); xs.assign(x.begin(),x.end()); ys.assign(y.begin(),y.end());
        ms.resize(n);
        std::vector<f64> d(n-1),s(n-1);
        for(int i=0;i<n-1;++i){d[i]=x[i+1]-x[i];s[i]=(y[i+1]-y[i])/d[i];}
        ms[0]=s[0]; ms[n-1]=s[n-2];
        for(int i=1;i<n-1;++i) ms[i]=(s[i-1]+s[i])*0.5;
        // Enforce monotonicity
        for(int i=0;i<n-1;++i){
            if(std::abs(s[i])<1e-14){ms[i]=ms[i+1]=0;}
            else{
                f64 a=ms[i]/s[i],b=ms[i+1]/s[i];
                f64 h=std::sqrt(a*a+b*b);
                if(h>3.0){ms[i]=3.0*a/h*s[i];ms[i+1]=3.0*b/h*s[i];}
            }
        }
    }

    [[nodiscard]] f64 eval(f64 x) const noexcept {
        int n=int(xs.size()),i=0;
        for(int k=0;k<n-2;++k) if(x>xs[k+1]) i=k+1;
        f64 h=xs[i+1]-xs[i], t=(x-xs[i])/h, t2=t*t, t3=t2*t;
        return (2*t3-3*t2+1)*ys[i]+(t3-2*t2+t)*h*ms[i]+
               (-2*t3+3*t2)*ys[i+1]+(t3-t2)*h*ms[i+1];
    }
};

} // namespace qs::math::numerical
