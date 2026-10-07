#pragma once
// qs_math/numerical/root_finding.hpp — Bisection, Newton-Raphson, Brent, Secant,
//   Halley, Ridder, Illinois, fixed-point iteration, polynomial root finding (C++20)
#include "../arithmetic/scalar_math.hpp"
#include <functional>
#include <vector>
#include <complex>
#include <cmath>
#include <optional>
#include <array>

namespace qs::math::numerical {

using Fn1  = std::function<f64(f64)>;
using Fn1d = std::function<f64(f64)>; // derivative

struct RootResult {
    f64  root;
    f64  residual;
    int  iterations;
    bool converged;
};

// ─── Bisection ────────────────────────────────────────────────────────────
[[nodiscard]] inline RootResult bisection(
    Fn1 f, f64 a, f64 b, f64 tol=1e-12, int max_iter=200) noexcept
{
    f64 fa=f(a), fb=f(b);
    if(fa*fb>0) return {0,1e30,0,false};
    for(int i=0;i<max_iter;++i){
        f64 m=0.5*(a+b), fm=f(m);
        if(std::abs(fm)<tol||0.5*(b-a)<tol) return {m,fm,i+1,true};
        (fa*fm<0)?(b=m,fb=fm):(a=m,fa=fm);
    }
    return {0.5*(a+b),f(0.5*(a+b)),max_iter,false};
}

// ─── Newton-Raphson ───────────────────────────────────────────────────────
[[nodiscard]] inline RootResult newton(
    Fn1 f, Fn1d df, f64 x0, f64 tol=1e-12, int max_iter=100) noexcept
{
    f64 x=x0;
    for(int i=0;i<max_iter;++i){
        f64 fx=f(x), dfx=df(x);
        if(std::abs(dfx)<1e-300) break;
        f64 dx=fx/dfx; x-=dx;
        if(std::abs(dx)<tol) return {x,f(x),i+1,true};
    }
    return {x,f(x),max_iter,false};
}

// ─── Halley's method (cubic convergence) ─────────────────────────────────
[[nodiscard]] inline RootResult halley(
    Fn1 f, Fn1d df, Fn1d d2f, f64 x0, f64 tol=1e-12, int max_iter=50) noexcept
{
    f64 x=x0;
    for(int i=0;i<max_iter;++i){
        f64 fx=f(x), dfx=df(x), d2fx=d2f(x);
        f64 denom=2.0*dfx*dfx - fx*d2fx;
        if(std::abs(denom)<1e-300) break;
        f64 dx=2.0*fx*dfx/denom; x-=dx;
        if(std::abs(dx)<tol) return {x,f(x),i+1,true};
    }
    return {x,f(x),max_iter,false};
}

// ─── Brent's method (bracketed, superlinear convergence) ──────────────────
[[nodiscard]] inline RootResult brent(
    Fn1 f, f64 a, f64 b, f64 tol=1e-12, int max_iter=200) noexcept
{
    f64 fa=f(a),fb=f(b);
    if(fa*fb>0) return {0,1e30,0,false};
    if(std::abs(fa)<std::abs(fb)){std::swap(a,b);std::swap(fa,fb);}
    f64 c=a,fc=fa,d=b-a,e=d;
    for(int i=0;i<max_iter;++i){
        if(std::abs(fb)<tol) return {b,fb,i+1,true};
        if(fa!=fc&&fb!=fc){
            // Inverse quadratic interpolation
            f64 s=a*fb*fc/((fa-fb)*(fa-fc))
                 +b*fa*fc/((fb-fa)*(fb-fc))
                 +c*fa*fb/((fc-fa)*(fc-fb));
            f64 p=s-b, tol1=std::abs(tol*(2.0*std::abs(b)+0.5));
            bool cond1=!(0.75*(a-b)<p&&p<0);
            bool cond2=(std::abs(e)<=tol1);
            bool cond3=(std::abs(p)>=0.5*std::abs(e));
            if(cond1||cond2||cond3){d=e=0.5*(a-b);}
            else{e=d;d=p-b;}
        } else {
            d=e=0.5*(a-b);
        }
        c=b; fc=fb;
        b+=(std::abs(d)>std::abs(tol))?d:(d>0?tol:-tol);
        fb=f(b);
        if(fa*fb>0){a=c;fa=fc;e=d=b-a;}
        if(std::abs(fa)<std::abs(fb)){std::swap(a,b);std::swap(fa,fb);}
    }
    return {b,fb,max_iter,false};
}

// ─── Secant method ────────────────────────────────────────────────────────
[[nodiscard]] inline RootResult secant(
    Fn1 f, f64 x0, f64 x1, f64 tol=1e-12, int max_iter=100) noexcept
{
    f64 fx0=f(x0),fx1=f(x1);
    for(int i=0;i<max_iter;++i){
        f64 denom=fx1-fx0;
        if(std::abs(denom)<1e-300) break;
        f64 x2=x1-fx1*(x1-x0)/denom;
        x0=x1;fx0=fx1;x1=x2;fx1=f(x1);
        if(std::abs(x1-x0)<tol) return {x1,fx1,i+1,true};
    }
    return {x1,f(x1),max_iter,false};
}

// ─── Ridder's method ──────────────────────────────────────────────────────
[[nodiscard]] inline RootResult ridder(
    Fn1 f, f64 a, f64 b, f64 tol=1e-12, int max_iter=100) noexcept
{
    f64 fa=f(a),fb=f(b);
    if(fa*fb>0) return {0,1e30,0,false};
    for(int i=0;i<max_iter;++i){
        f64 m=0.5*(a+b), fm=f(m);
        f64 w=std::sqrt(fm*fm-fa*fb);
        if(std::abs(w)<1e-300) return {m,fm,i+1,true};
        f64 xnew=m+(m-a)*(fa<fb?1:-1)*fm/w;
        f64 fxnew=f(xnew);
        if(std::abs(fxnew)<tol) return {xnew,fxnew,i+1,true};
        if(fm*fxnew<0){a=m;fa=fm;b=xnew;fb=fxnew;}
        else if(fa*fxnew<0){b=xnew;fb=fxnew;}
        else{a=xnew;fa=fxnew;}
        if(std::abs(b-a)<tol) return {0.5*(a+b),f(0.5*(a+b)),i+1,true};
    }
    return {0.5*(a+b),f(0.5*(a+b)),max_iter,false};
}

// ─── Illinois algorithm (regula falsi + anti-stagnation) ──────────────────
[[nodiscard]] inline RootResult illinois(
    Fn1 f, f64 a, f64 b, f64 tol=1e-12, int max_iter=200) noexcept
{
    f64 fa=f(a),fb=f(b);
    if(fa*fb>0) return {0,1e30,0,false};
    int side=0;
    for(int i=0;i<max_iter;++i){
        f64 c=(fb*a-fa*b)/(fb-fa);
        f64 fc=f(c);
        if(std::abs(fc)<tol||std::abs(b-a)<tol) return {c,fc,i+1,true};
        if(fc*fb<0){a=b;fa=fb;side=0;}
        else{if(side==1) fb*=0.5; side=1;}
        b=c;fb=fc;
    }
    return {b,fb,max_iter,false};
}

// ─── Fixed-point iteration ────────────────────────────────────────────────
[[nodiscard]] inline RootResult fixed_point(
    Fn1 g, f64 x0, f64 tol=1e-10, int max_iter=200) noexcept
{
    f64 x=x0;
    for(int i=0;i<max_iter;++i){
        f64 xnew=g(x);
        if(std::abs(xnew-x)<tol) return {xnew,std::abs(xnew-x),i+1,true};
        x=xnew;
    }
    return {x,0,max_iter,false};
}

// ─── Steffensen's method (fixed-point with Aitken acceleration) ───────────
[[nodiscard]] inline RootResult steffensen(
    Fn1 g, f64 x0, f64 tol=1e-12, int max_iter=100) noexcept
{
    f64 x=x0;
    for(int i=0;i<max_iter;++i){
        f64 gx=g(x), ggx=g(gx);
        f64 denom=ggx-2.0*gx+x;
        if(std::abs(denom)<1e-300) break;
        f64 xnew=x-(gx-x)*(gx-x)/denom;
        if(std::abs(xnew-x)<tol) return {xnew,0,i+1,true};
        x=xnew;
    }
    return {x,0,max_iter,false};
}

// ─── Polynomial roots (Aberth-Ehrlich method) ─────────────────────────────
using Cx64 = std::complex<f64>;

// Evaluate polynomial p at z (Horner)
[[nodiscard]] inline Cx64 poly_eval(const std::vector<Cx64>& p, Cx64 z) noexcept {
    Cx64 r=p.back();
    for(int i=int(p.size())-2;i>=0;--i) r=r*z+p[i];
    return r;
}

// Derivative via Horner
[[nodiscard]] inline Cx64 poly_deriv(const std::vector<Cx64>& p, Cx64 z) noexcept {
    int n=int(p.size())-1;
    Cx64 r=p[n]*Cx64(n);
    for(int i=n-1;i>=1;--i) r=r*z+p[i]*Cx64(i);
    return r;
}

[[nodiscard]] inline std::vector<Cx64> aberth_ehrlich(
    std::vector<Cx64> p, f64 tol=1e-12, int max_iter=200) noexcept
{
    int n=int(p.size())-1;
    // Initial approximations on circle of radius 1+max|coeff|^(1/n)
    f64 r=0; for(int i=0;i<n;++i) r=std::max(r,std::abs(p[i]/p[n]));
    r=1.0+std::pow(r,1.0/n);
    std::vector<Cx64> z(n);
    for(int k=0;k<n;++k){
        f64 ang=2.0*PI*k/n+0.1;
        z[k]={r*std::cos(ang),r*std::sin(ang)};
    }
    for(int iter=0;iter<max_iter;++iter){
        f64 max_step=0;
        for(int k=0;k<n;++k){
            Cx64 pz=poly_eval(p,z[k]);
            Cx64 dpz=poly_deriv(p,z[k]);
            if(std::abs(dpz)<1e-300) continue;
            Cx64 sum{0,0};
            for(int j=0;j<n;++j) if(j!=k){
                Cx64 diff=z[k]-z[j];
                if(std::abs(diff)>1e-300) sum+=Cx64(1)/diff;
            }
            Cx64 w=pz/dpz/(Cx64(1)-pz/dpz*sum);
            z[k]-=w; max_step=std::max(max_step,std::abs(w));
        }
        if(max_step<tol) break;
    }
    return z;
}

// Real polynomial roots via companion matrix + eigenvalue deflation (Laguerre)
[[nodiscard]] inline RootResult laguerre(
    const std::vector<f64>& p, Cx64 x0={0.5,0.5}, f64 tol=1e-12, int max_iter=200) noexcept
{
    int n=int(p.size())-1;
    Cx64 x=x0;
    for(int iter=0;iter<max_iter;++iter){
        std::vector<Cx64> cp(p.size());
        for(size_t i=0;i<p.size();++i) cp[i]={p[i],0};
        Cx64 pv=poly_eval(cp,x);
        if(std::abs(pv)<tol) return {x.real(),std::abs(pv),iter,true};
        Cx64 dpv=poly_deriv(cp,x);
        // Second derivative via Horner
        Cx64 d2pv={0,0};
        for(int i=n;i>=2;--i) d2pv=d2pv*x+cp[i]*Cx64(i*(i-1));
        Cx64 G=dpv/pv, H=G*G-d2pv/pv;
        Cx64 sq=std::sqrt(Cx64(n-1)*(Cx64(n)*H-G*G));
        Cx64 denom=(std::abs(G+sq)>std::abs(G-sq))?(G+sq):(G-sq);
        if(std::abs(denom)<1e-300) break;
        Cx64 dx=Cx64(n)/denom; x-=dx;
        if(std::abs(dx)<tol) return {x.real(),std::abs(poly_eval(cp,x)),iter+1,true};
    }
    return {x.real(),0,max_iter,false};
}

// ─── Multi-dimensional Newton (2D and 3D) ─────────────────────────────────
// Solve F(x,y)=0 where F:R^2->R^2, J is 2x2 Jacobian
using F2 = std::function<std::array<f64,2>(f64,f64)>;
using J2 = std::function<std::array<std::array<f64,2>,2>(f64,f64)>;

[[nodiscard]] inline std::pair<f64,f64> newton2d(
    F2 F, J2 J, f64 x0, f64 y0, f64 tol=1e-10, int max_iter=100) noexcept
{
    f64 x=x0,y=y0;
    for(int i=0;i<max_iter;++i){
        auto f=F(x,y);
        auto j=J(x,y);
        f64 det=j[0][0]*j[1][1]-j[0][1]*j[1][0];
        if(std::abs(det)<1e-300) break;
        f64 dx=( f[0]*j[1][1]-f[1]*j[0][1])/det;
        f64 dy=(-f[0]*j[1][0]+f[1]*j[0][0])/det;
        x-=dx; y-=dy;
        if(dx*dx+dy*dy<tol*tol) break;
    }
    return {x,y};
}

// ─── Inverse quadratic interpolation (standalone) ─────────────────────────
[[nodiscard]] inline f64 inv_quad_interp(
    f64 x0, f64 x1, f64 x2,
    f64 f0, f64 f1, f64 f2) noexcept
{
    return x0*f1*f2/((f0-f1)*(f0-f2))
          +x1*f0*f2/((f1-f0)*(f1-f2))
          +x2*f0*f1/((f2-f0)*(f2-f1));
}

} // namespace qs::math::numerical
