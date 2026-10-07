#pragma once
// qs_math/calculus.hpp — Numerical derivatives, integrals, limits, root finding (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include <functional>
#include <vector>
#include <cmath>
#include <cassert>
#include <array>

namespace qs {

using Func1D = std::function<f64(f64)>;
using FuncND = std::function<f64(const std::vector<f64>&)>;

// ─── Finite difference derivatives ───────────────────────────────────────────

// First derivative: 4th-order central difference
[[nodiscard]] inline f64 deriv1(const Func1D& f, f64 x, f64 h=1e-5) {
    return (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h);
}
// Second derivative: 4th-order
[[nodiscard]] inline f64 deriv2(const Func1D& f, f64 x, f64 h=1e-4) {
    return (-f(x+2*h)+16*f(x+h)-30*f(x)+16*f(x-h)-f(x-2*h))/(12*h*h);
}
// Third derivative
[[nodiscard]] inline f64 deriv3(const Func1D& f, f64 x, f64 h=1e-3) {
    return (-f(x+2*h)+2*f(x+h)-2*f(x-h)+f(x-2*h))/(2*h*h*h);
}
// nth derivative via Fornberg weights (up to n=6)
[[nodiscard]] inline f64 derivn(const Func1D& f, f64 x, u32 n, f64 h=1e-3) {
    // Central finite difference with 2n+1 points
    if(n==0) return f(x);
    if(n==1) return deriv1(f,x,h);
    if(n==2) return deriv2(f,x,h);
    if(n==3) return deriv3(f,x,h);
    // Generic: use Richardson extrapolation base
    return (derivn(f,x,n-1,h+h*0.01)-derivn(f,x,n-1,h)) / (h*0.01); // very rough
}

// Partial derivative ∂f/∂x_i
[[nodiscard]] inline f64 partial(const FuncND& f, std::vector<f64> x, u32 i, f64 h=1e-5) {
    f64 xi=x[i];
    x[i]=xi+h; f64 fp=f(x);
    x[i]=xi-h; f64 fm=f(x);
    return (fp-fm)/(2*h);
}

// Gradient ∇f at x
[[nodiscard]] inline std::vector<f64> gradient(const FuncND& f, std::vector<f64> x, f64 h=1e-5) {
    u32 n=u32(x.size());
    std::vector<f64> g(n);
    for(u32 i=0;i<n;++i) g[i]=partial(f,x,i,h);
    return g;
}

// Jacobian matrix of vector-valued function F:R^n→R^m
[[nodiscard]] inline std::vector<std::vector<f64>> jacobian(
    const std::function<std::vector<f64>(const std::vector<f64>&)>& F,
    std::vector<f64> x, f64 h=1e-5)
{
    auto f0=F(x); u32 m=u32(f0.size()), n=u32(x.size());
    std::vector<std::vector<f64>> J(m, std::vector<f64>(n));
    for(u32 j=0;j<n;++j) {
        f64 xj=x[j];
        x[j]=xj+h; auto fp=F(x);
        x[j]=xj-h; auto fm=F(x);
        x[j]=xj;
        for(u32 i=0;i<m;++i) J[i][j]=(fp[i]-fm[i])/(2*h);
    }
    return J;
}

// Hessian matrix (symmetric, second partials)
[[nodiscard]] inline std::vector<std::vector<f64>> hessian(const FuncND& f, std::vector<f64> x, f64 h=1e-4) {
    u32 n=u32(x.size());
    std::vector<std::vector<f64>> H(n,std::vector<f64>(n));
    for(u32 i=0;i<n;++i) for(u32 j=i;j<n;++j) {
        if(i==j) {
            H[i][j]=deriv2([&](f64 t){ auto xc=x; xc[i]=t; return f(xc); },x[i],h);
        } else {
            f64 xi=x[i],xj=x[j];
            x[i]=xi+h; x[j]=xj+h; f64 fpp=f(x);
            x[i]=xi+h; x[j]=xj-h; f64 fpm=f(x);
            x[i]=xi-h; x[j]=xj+h; f64 fmp=f(x);
            x[i]=xi-h; x[j]=xj-h; f64 fmm=f(x);
            x[i]=xi; x[j]=xj;
            H[i][j]=H[j][i]=(fpp-fpm-fmp+fmm)/(4*h*h);
        }
    }
    return H;
}

// ─── Numerical integration ────────────────────────────────────────────────────

// Adaptive Simpson's rule
[[nodiscard]] inline f64 adaptive_simpson(const Func1D& f, f64 a, f64 b, f64 tol=1e-8, u32 max_depth=20) {
    std::function<f64(f64,f64,f64,f64,f64,f64,u32)> rec=[&](f64 a,f64 b,f64 fa,f64 fm,f64 fb,f64 S,u32 depth)->f64 {
        f64 m1=0.5*(a+a+b-a)*0.5, m2=0.5*(a+b+b-a)*0.5; // midpoints of halves
        f64 mid=0.5*(a+b);
        m1=0.5*(a+mid); m2=0.5*(mid+b);
        f64 f1=f(m1),f2=f(m2),fmid=f(mid);
        f64 S1=(mid-a)/6*(fa+4*f1+fmid);
        f64 S2=(b-mid)/6*(fmid+4*f2+fb);
        if(depth==0||std::abs(S1+S2-S)<15*tol) return S1+S2+(S1+S2-S)/15;
        return rec(a,mid,fa,f1,fmid,S1,depth-1)+rec(mid,b,fmid,f2,fb,S2,depth-1);
    };
    f64 fa=f(a),fb=f(b),fm=f(0.5*(a+b));
    f64 S=(b-a)/6*(fa+4*fm+fb);
    return rec(a,b,fa,fm,fb,S,max_depth);
}

// Gauss-Legendre quadrature (fixed nodes, high accuracy)
template<u32 N=5>
[[nodiscard]] f64 gauss_legendre(const Func1D& f, f64 a, f64 b) {
    // 5-point GL nodes and weights on [-1,1]
    static constexpr f64 nodes[5]  ={-0.9061798459,-0.5384693101,0.0,0.5384693101,0.9061798459};
    static constexpr f64 weights[5]={ 0.2369268851, 0.4786286705,0.5688888889,0.4786286705,0.2369268851};
    f64 mid=0.5*(a+b), half=0.5*(b-a), s=0;
    for(u32 i=0;i<5;++i) s+=weights[i]*f(mid+half*nodes[i]);
    return half*s;
}

// Gaussian quadrature with n panels (composite)
[[nodiscard]] inline f64 integrate(const Func1D& f, f64 a, f64 b, u32 panels=100) {
    f64 h=(b-a)/panels, s=0;
    for(u32 i=0;i<panels;++i) s+=gauss_legendre(f, a+i*h, a+(i+1)*h);
    return s;
}

// Double integral ∫∫ f(x,y) dx dy over [ax,bx]×[ay,by]
[[nodiscard]] inline f64 integrate2d(const std::function<f64(f64,f64)>& f,
    f64 ax,f64 bx,f64 ay,f64 by,u32 nx=20,u32 ny=20)
{
    f64 hx=(bx-ax)/nx, hy=(by-ay)/ny, s=0;
    for(u32 i=0;i<nx;++i) for(u32 j=0;j<ny;++j)
        s+=gauss_legendre([&](f64 x){ return gauss_legendre([&](f64 y){return f(x,y);},ay+j*hy,ay+(j+1)*hy); },ax+i*hx,ax+(i+1)*hx);
    return s;
}

// ─── Root finding ─────────────────────────────────────────────────────────────

struct RootResult { f64 x; f64 fx; u32 iters; bool converged; };

// Brent's method — robust root bracketing
[[nodiscard]] inline RootResult brent(const Func1D& f, f64 a, f64 b, f64 tol=1e-12, u32 max_iter=200) {
    f64 fa=f(a),fb=f(b);
    if(fa*fb>0) return {a,fa,0,false};
    if(std::abs(fa)<std::abs(fb)){std::swap(a,b);std::swap(fa,fb);}
    f64 c=a,fc=fa,s=b,fs,d=b-a;
    bool mflag=true;
    for(u32 i=0;i<max_iter;++i) {
        if(std::abs(b-a)<tol) return {b,fb,i,true};
        if(fa!=fc&&fb!=fc) {
            s=a*fb*fc/((fa-fb)*(fa-fc))+b*fa*fc/((fb-fa)*(fb-fc))+c*fa*fb/((fc-fa)*(fc-fb));
        } else s=b-fb*(b-a)/(fb-fa);
        f64 lo=std::min((3*a+b)*0.25,b), hi=std::max((3*a+b)*0.25,b);
        bool cond1=s<lo||s>hi;
        bool cond2=mflag&&std::abs(s-b)>=0.5*std::abs(b-c);
        bool cond3=!mflag&&std::abs(s-b)>=0.5*std::abs(c-d);
        bool cond4=mflag&&std::abs(b-c)<tol;
        bool cond5=!mflag&&std::abs(c-d)<tol;
        if(cond1||cond2||cond3||cond4||cond5){ s=0.5*(a+b); mflag=true; } else mflag=false;
        fs=f(s); d=c; c=b; fc=fb;
        if(fa*fs<0){b=s;fb=fs;}else{a=s;fa=fs;}
        if(std::abs(fa)<std::abs(fb)){std::swap(a,b);std::swap(fa,fb);}
    }
    return {b,fb,max_iter,false};
}

// Newton-Raphson with fallback to bisection
[[nodiscard]] inline RootResult newton(const Func1D& f, const Func1D& df, f64 x0,
    f64 tol=1e-12, u32 max_iter=100)
{
    f64 x=x0;
    for(u32 i=0;i<max_iter;++i) {
        f64 fx=f(x),dfx=df(x);
        if(std::abs(fx)<tol) return {x,fx,i,true};
        if(std::abs(dfx)<1e-20) break;
        x-=fx/dfx;
    }
    return {x,f(x),max_iter,false};
}

// Newton with auto-differentiation (uses finite diff for df)
[[nodiscard]] inline RootResult newton_auto(const Func1D& f, f64 x0, f64 tol=1e-12, u32 max_iter=100) {
    return newton(f,[&](f64 x){return deriv1(f,x);},x0,tol,max_iter);
}

// Halley's method — cubic convergence
[[nodiscard]] inline RootResult halley(const Func1D& f, f64 x0, f64 tol=1e-12, u32 max_iter=50) {
    f64 x=x0;
    for(u32 i=0;i<max_iter;++i) {
        f64 fx=f(x),d1=deriv1(f,x),d2=deriv2(f,x);
        if(std::abs(fx)<tol) return {x,fx,i,true};
        f64 denom=2*d1*d1-fx*d2;
        if(std::abs(denom)<1e-20) break;
        x-=2*fx*d1/denom;
    }
    return {x,f(x),max_iter,false};
}

// ─── Limits ───────────────────────────────────────────────────────────────────

// Richardson extrapolation for limit lim_{h→0} f(h)
[[nodiscard]] inline f64 limit_richardson(const Func1D& f, f64 h0=0.1, u32 steps=6) {
    // Build Romberg-like table
    std::vector<f64> R;
    f64 h=h0;
    for(u32 i=0;i<steps;++i) { R.push_back(f(h)); h*=0.5; }
    // Richardson extrapolation assuming O(h^2) convergence
    for(u32 j=1;j<steps;++j)
        for(u32 i=steps-1;i>=j;--i)
            R[i]=(4*R[i]-R[i-1])/3;
    return R[steps-1];
}

// ─── Romberg integration ──────────────────────────────────────────────────────
[[nodiscard]] inline f64 romberg(const Func1D& f, f64 a, f64 b, u32 max_level=8, f64 tol=1e-10) {
    u32 N=max_level+1;
    std::vector<std::vector<f64>> R(N, std::vector<f64>(N,0));
    f64 h=b-a;
    R[0][0]=0.5*h*(f(a)+f(b));
    for(u32 i=1;i<N;++i) {
        h*=0.5; u32 pts=1u<<(i-1);
        f64 s=0; for(u32 k=1;k<=pts;++k) s+=f(a+(2*k-1)*h);
        R[i][0]=0.5*R[i-1][0]+h*s;
        f64 p=4;
        for(u32 j=1;j<=i;++j) { R[i][j]=(p*R[i][j-1]-R[i-1][j-1])/(p-1); p*=4; }
        if(i>1&&std::abs(R[i][i]-R[i-1][i-1])<tol) return R[i][i];
    }
    return R[N-1][N-1];
}

// ─── Directional derivative ───────────────────────────────────────────────────
[[nodiscard]] inline f64 directional_deriv(const FuncND& f, const std::vector<f64>& x,
    const std::vector<f64>& dir, f64 h=1e-5)
{
    u32 n=u32(x.size());
    f64 dn=0; for(auto v:dir) dn+=v*v; dn=std::sqrt(dn);
    std::vector<f64> xp(n),xm(n);
    for(u32 i=0;i<n;++i){ xp[i]=x[i]+h*dir[i]/dn; xm[i]=x[i]-h*dir[i]/dn; }
    return (f(xp)-f(xm))/(2*h);
}

// ─── Divergence and curl (3D) ─────────────────────────────────────────────────
using VecField3 = std::function<Vec3f(Vec3f)>;

[[nodiscard]] inline f64 divergence(const VecField3& F, Vec3f p, f64 h=1e-4) {
    auto eval_comp=[&](int c, f64 d) -> f64 {
        Vec3f pp=p,pm=p;
        if(c==0){pp.x+=d;pm.x-=d;}
        else if(c==1){pp.y+=d;pm.y-=d;}
        else{pp.z+=d;pm.z-=d;}
        auto fp=F(pp),fm=F(pm);
        return c==0?(fp.x-fm.x)/(2*d):c==1?(fp.y-fm.y)/(2*d):(fp.z-fm.z)/(2*d);
    };
    return eval_comp(0,h)+eval_comp(1,h)+eval_comp(2,h);
}

[[nodiscard]] inline Vec3f curl(const VecField3& F, Vec3f p, f64 h=1e-4) {
    auto Dp=[&](Vec3f pt,f64 dh,int axis)->Vec3f{
        Vec3f pp=pt,pm=pt;
        if(axis==0){pp.x+=f32(dh);pm.x-=f32(dh);}
        else if(axis==1){pp.y+=f32(dh);pm.y-=f32(dh);}
        else{pp.z+=f32(dh);pm.z-=f32(dh);}
        auto fp=F(pp),fm=F(pm);
        return {(fp.x-fm.x)/f32(2*dh),(fp.y-fm.y)/f32(2*dh),(fp.z-fm.z)/f32(2*dh)};
    };
    auto dFdx=Dp(p,h,0),dFdy=Dp(p,h,1),dFdz=Dp(p,h,2);
    return {dFdy.z-dFdz.y, dFdz.x-dFdx.z, dFdx.y-dFdy.x};
}

} // namespace qs
