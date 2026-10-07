#pragma once
// qs_math/numerical/optimization.hpp — Unconstrained/constrained optimization:
//   gradient descent variants (SGD/Adam/RMSProp/Adagrad), LBFGS, Nelder-Mead,
//   simulated annealing, differential evolution, golden section, ternary search (C++20)
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

namespace qs::math::numerical {

using VecD  = std::vector<f64>;
using ObjFn = std::function<f64(const VecD&)>;
using GradFn= std::function<VecD(const VecD&)>;

// ─── Vector helpers ───────────────────────────────────────────────────────
inline VecD vaxpy(f64 a, const VecD& x, const VecD& y){
    VecD r(x.size()); for(size_t i=0;i<x.size();++i) r[i]=a*x[i]+y[i]; return r;
}
inline f64 vdot(const VecD& a,const VecD& b){ f64 s=0; for(size_t i=0;i<a.size();++i) s+=a[i]*b[i]; return s; }
inline f64 vnorm(const VecD& v){ return std::sqrt(vdot(v,v)); }
inline VecD vscale(f64 s,const VecD& v){ VecD r(v.size()); for(size_t i=0;i<v.size();++i) r[i]=s*v[i]; return r; }
inline VecD vsub(const VecD& a,const VecD& b){ VecD r(a.size()); for(size_t i=0;i<a.size();++i) r[i]=a[i]-b[i]; return r; }
inline VecD vadd(const VecD& a,const VecD& b){ VecD r(a.size()); for(size_t i=0;i<a.size();++i) r[i]=a[i]+b[i]; return r; }

struct OptResult { VecD x; f64 f_val; int iterations; bool converged; };

// ─── Numerical gradient ───────────────────────────────────────────────────
[[nodiscard]] inline VecD numerical_gradient(const ObjFn& f, const VecD& x, f64 h=1e-5) {
    VecD g(x.size()); VecD xp=x,xm=x;
    for(size_t i=0;i<x.size();++i){
        xp[i]+=h; xm[i]-=h;
        g[i]=(f(xp)-f(xm))/(2*h);
        xp[i]=xm[i]=x[i];
    }
    return g;
}

// ─── Gradient Descent ────────────────────────────────────────────────────
[[nodiscard]] inline OptResult gradient_descent(
    const ObjFn& f, const GradFn& grad, VecD x0,
    f64 lr=0.01, f64 tol=1e-8, int max_iter=10000) noexcept
{
    VecD x=x0;
    for(int i=0;i<max_iter;++i){
        VecD g=grad(x);
        f64 gn=vnorm(g);
        if(gn<tol) return {x,f(x),i+1,true};
        x=vsub(x,vscale(lr,g));
    }
    return {x,f(x),max_iter,false};
}

// ─── Adam optimizer ───────────────────────────────────────────────────────
struct AdamParams { f64 lr=1e-3,beta1=0.9,beta2=0.999,eps=1e-8; };

[[nodiscard]] inline OptResult adam(
    const ObjFn& f, const GradFn& grad, VecD x0,
    AdamParams p={}, f64 tol=1e-8, int max_iter=10000) noexcept
{
    int n=int(x0.size()); VecD x=x0,m(n,0),v(n,0);
    for(int t=1;t<=max_iter;++t){
        VecD g=grad(x);
        f64 gn=vnorm(g); if(gn<tol) return {x,f(x),t,true};
        for(int i=0;i<n;++i){
            m[i]=p.beta1*m[i]+(1-p.beta1)*g[i];
            v[i]=p.beta2*v[i]+(1-p.beta2)*g[i]*g[i];
            f64 mh=m[i]/(1-std::pow(p.beta1,t));
            f64 vh=v[i]/(1-std::pow(p.beta2,t));
            x[i]-=p.lr*mh/(std::sqrt(vh)+p.eps);
        }
    }
    return {x,f(x),max_iter,false};
}

// ─── RMSProp ─────────────────────────────────────────────────────────────
[[nodiscard]] inline OptResult rmsprop(
    const ObjFn& f, const GradFn& grad, VecD x0,
    f64 lr=1e-3, f64 decay=0.9, f64 eps=1e-8,
    f64 tol=1e-8, int max_iter=10000) noexcept
{
    int n=int(x0.size()); VecD x=x0,eg2(n,0);
    for(int t=0;t<max_iter;++t){
        VecD g=grad(x);
        if(vnorm(g)<tol) return {x,f(x),t,true};
        for(int i=0;i<n;++i){
            eg2[i]=decay*eg2[i]+(1-decay)*g[i]*g[i];
            x[i]-=lr*g[i]/(std::sqrt(eg2[i])+eps);
        }
    }
    return {x,f(x),max_iter,false};
}

// ─── Adagrad ─────────────────────────────────────────────────────────────
[[nodiscard]] inline OptResult adagrad(
    const ObjFn& f, const GradFn& grad, VecD x0,
    f64 lr=0.01, f64 eps=1e-8, f64 tol=1e-8, int max_iter=10000) noexcept
{
    int n=int(x0.size()); VecD x=x0,G(n,0);
    for(int t=0;t<max_iter;++t){
        VecD g=grad(x);
        if(vnorm(g)<tol) return {x,f(x),t,true};
        for(int i=0;i<n;++i){ G[i]+=g[i]*g[i]; x[i]-=lr*g[i]/(std::sqrt(G[i])+eps); }
    }
    return {x,f(x),max_iter,false};
}

// ─── L-BFGS ──────────────────────────────────────────────────────────────
[[nodiscard]] inline OptResult lbfgs(
    const ObjFn& f, const GradFn& grad, VecD x0,
    int m=10, f64 tol=1e-8, int max_iter=500) noexcept
{
    VecD x=x0,g=grad(x);
    std::vector<VecD> s_hist,y_hist; std::vector<f64> rho_hist;
    for(int it=0;it<max_iter;++it){
        if(vnorm(g)<tol) return {x,f(x),it,true};
        // Two-loop recursion
        VecD q=g; int hm=int(s_hist.size());
        std::vector<f64> alpha(hm);
        for(int i=hm-1;i>=0;--i){
            alpha[i]=rho_hist[i]*vdot(s_hist[i],q);
            q=vsub(q,vscale(alpha[i],y_hist[i]));
        }
        VecD r=q;
        if(hm>0){
            f64 scale=vdot(s_hist.back(),y_hist.back())/vdot(y_hist.back(),y_hist.back());
            r=vscale(scale,q);
        }
        for(int i=0;i<hm;++i){
            f64 beta=rho_hist[i]*vdot(y_hist[i],r);
            r=vadd(r,vscale(alpha[i]-beta,s_hist[i]));
        }
        VecD dir=vscale(-1,r);
        // Line search (Wolfe conditions — simplified backtracking)
        f64 step=1.0,f0=f(x),c1=1e-4;
        VecD xnew,gnew;
        for(int ls=0;ls<50;++ls){
            xnew=vadd(x,vscale(step,dir));
            f64 fnew=f(xnew);
            if(fnew<=f0+c1*step*vdot(g,dir)) break;
            step*=0.5;
        }
        xnew=vadd(x,vscale(step,dir));
        gnew=grad(xnew);
        VecD sk=vsub(xnew,x), yk=vsub(gnew,g);
        f64 rho=vdot(yk,sk);
        if(std::abs(rho)>1e-14){
            rho=1.0/rho;
            s_hist.push_back(sk); y_hist.push_back(yk); rho_hist.push_back(rho);
            if(int(s_hist.size())>m){ s_hist.erase(s_hist.begin()); y_hist.erase(y_hist.begin()); rho_hist.erase(rho_hist.begin()); }
        }
        x=xnew; g=gnew;
    }
    return {x,f(x),max_iter,false};
}

// ─── Nelder-Mead simplex ─────────────────────────────────────────────────
[[nodiscard]] inline OptResult nelder_mead(
    const ObjFn& f, VecD x0, f64 scale=1.0,
    f64 tol=1e-8, int max_iter=5000) noexcept
{
    int n=int(x0.size());
    std::vector<VecD> simplex(n+1);
    simplex[0]=x0;
    for(int i=0;i<n;++i){ simplex[i+1]=x0; simplex[i+1][i]+=scale; }
    std::vector<f64> fvals(n+1);
    for(int i=0;i<=n;++i) fvals[i]=f(simplex[i]);
    std::vector<int> idx(n+1); std::iota(idx.begin(),idx.end(),0);
    for(int it=0;it<max_iter;++it){
        std::sort(idx.begin(),idx.end(),[&](int a,int b){return fvals[a]<fvals[b];});
        if(vnorm(vsub(simplex[idx[0]],simplex[idx[n]]))<tol) return {simplex[idx[0]],fvals[idx[0]],it,true};
        // Centroid of all but worst
        VecD cen(n,0);
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) cen[j]+=simplex[idx[i]][j];
        for(auto& v:cen) v/=n;
        // Reflect
        VecD xr=vsub(vscale(2,cen),simplex[idx[n]]);
        f64 fr=f(xr);
        if(fr<fvals[idx[0]]){ // Expand
            VecD xe=vsub(vscale(3,cen),vscale(2,simplex[idx[n]]));
            f64 fe=f(xe);
            if(fe<fr){ simplex[idx[n]]=xe;fvals[idx[n]]=fe; }
            else{ simplex[idx[n]]=xr;fvals[idx[n]]=fr; }
        } else if(fr<fvals[idx[n-1]]){
            simplex[idx[n]]=xr; fvals[idx[n]]=fr;
        } else { // Contract
            VecD xc=vadd(vscale(0.5,cen),vscale(0.5,simplex[idx[n]]));
            f64 fc=f(xc);
            if(fc<fvals[idx[n]]){ simplex[idx[n]]=xc;fvals[idx[n]]=fc; }
            else { // Shrink
                for(int i=1;i<=n;++i){
                    simplex[idx[i]]=vadd(simplex[idx[0]],vscale(0.5,vsub(simplex[idx[i]],simplex[idx[0]])));
                    fvals[idx[i]]=f(simplex[idx[i]]);
                }
            }
        }
    }
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return fvals[a]<fvals[b];});
    return {simplex[idx[0]],fvals[idx[0]],max_iter,false};
}

// ─── Simulated Annealing ──────────────────────────────────────────────────
[[nodiscard]] inline OptResult simulated_annealing(
    const ObjFn& f, VecD x0, f64 T0=1.0, f64 T_final=1e-6,
    f64 cooling=0.99, f64 step_size=0.1, int steps_per_temp=100,
    u32 seed=42) noexcept
{
    std::mt19937 rng(seed);
    std::normal_distribution<f64> nd(0,step_size);
    std::uniform_real_distribution<f64> ud(0,1);
    VecD x=x0,best_x=x0;
    f64 fx=f(x),best_f=fx,T=T0;
    int total=0;
    while(T>T_final){
        for(int s=0;s<steps_per_temp;++s){
            VecD xn=x;
            for(auto& v:xn) v+=nd(rng);
            f64 fn=f(xn);
            f64 dE=fn-fx;
            if(dE<0||ud(rng)<std::exp(-dE/T)){x=xn;fx=fn;}
            if(fx<best_f){best_f=fx;best_x=x;}
            ++total;
        }
        T*=cooling;
    }
    return {best_x,best_f,total,true};
}

// ─── Differential Evolution ───────────────────────────────────────────────
[[nodiscard]] inline OptResult differential_evolution(
    const ObjFn& f, int n_dim, VecD lb, VecD ub,
    int pop=20, f64 F=0.8, f64 CR=0.9,
    int max_iter=1000, u32 seed=42) noexcept
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> ud(0,1);
    std::uniform_int_distribution<int> ri(0,pop-1);
    // Initialize population
    std::vector<VecD> pop_v(pop,VecD(n_dim));
    std::vector<f64>  pop_f(pop);
    for(auto& ind:pop_v){
        for(int j=0;j<n_dim;++j) ind[j]=lb[j]+ud(rng)*(ub[j]-lb[j]);
    }
    for(int i=0;i<pop;++i) pop_f[i]=f(pop_v[i]);
    int best=int(std::min_element(pop_f.begin(),pop_f.end())-pop_f.begin());
    for(int it=0;it<max_iter;++it){
        for(int i=0;i<pop;++i){
            int a,b,c;
            do{a=ri(rng);}while(a==i);
            do{b=ri(rng);}while(b==i||b==a);
            do{c=ri(rng);}while(c==i||c==a||c==b);
            VecD trial=pop_v[i];
            int jrand=int(ud(rng)*n_dim);
            for(int j=0;j<n_dim;++j){
                if(ud(rng)<CR||j==jrand){
                    trial[j]=pop_v[a][j]+F*(pop_v[b][j]-pop_v[c][j]);
                    trial[j]=std::clamp(trial[j],lb[j],ub[j]);
                }
            }
            f64 ft=f(trial);
            if(ft<pop_f[i]){pop_v[i]=trial;pop_f[i]=ft;if(ft<pop_f[best]) best=i;}
        }
    }
    return {pop_v[best],pop_f[best],max_iter*pop,true};
}

// ─── Golden section search (1D unimodal) ──────────────────────────────────
[[nodiscard]] inline std::pair<f64,f64> golden_section(
    std::function<f64(f64)> f, f64 a, f64 b,
    f64 tol=1e-10, int max_iter=200) noexcept
{
    static constexpr f64 phi=(std::sqrt(5.0)-1)/2;
    f64 c=b-phi*(b-a), d=a+phi*(b-a);
    f64 fc=f(c),fd=f(d);
    for(int i=0;i<max_iter;++i){
        if(std::abs(b-a)<tol) break;
        if(fc<fd){b=d;d=c;fd=fc;c=b-phi*(b-a);fc=f(c);}
        else{a=c;c=d;fc=fd;d=a+phi*(b-a);fd=f(d);}
    }
    f64 x=0.5*(a+b);
    return {x,f(x)};
}

// ─── Ternary search ───────────────────────────────────────────────────────
[[nodiscard]] inline std::pair<f64,f64> ternary_search(
    std::function<f64(f64)> f, f64 a, f64 b,
    f64 tol=1e-10, int max_iter=200) noexcept
{
    for(int i=0;i<max_iter;++i){
        if(b-a<tol) break;
        f64 m1=a+(b-a)/3, m2=b-(b-a)/3;
        if(f(m1)<f(m2)) b=m2; else a=m1;
    }
    f64 x=0.5*(a+b); return {x,f(x)};
}

// ─── Wolfe line search (strong Wolfe conditions) ───────────────────────────
[[nodiscard]] inline f64 wolfe_line_search(
    const ObjFn& f, const GradFn& grad,
    const VecD& x, const VecD& dir,
    f64 f0, const VecD& g0, f64 c1=1e-4, f64 c2=0.9) noexcept
{
    f64 alpha=1.0,alpha_lo=0,alpha_hi=1e10,gd=vdot(g0,dir);
    for(int i=0;i<50;++i){
        VecD xnew=vadd(x,vscale(alpha,dir));
        f64 fnew=f(xnew);
        if(fnew>f0+c1*alpha*gd||(i>0&&fnew>=f(vadd(x,vscale(alpha_lo,dir))))){
            // Zoom into [alpha_lo,alpha]
            alpha_hi=alpha;
        } else {
            VecD gnew=grad(xnew); f64 gnew_d=vdot(gnew,dir);
            if(std::abs(gnew_d)<=-c2*gd) return alpha;
            if(gnew_d*(alpha_hi-alpha_lo)>=0) alpha_hi=alpha_lo;
            alpha_lo=alpha;
        }
        if(alpha_hi<1e9) alpha=0.5*(alpha_lo+alpha_hi);
        else alpha*=2;
    }
    return alpha;
}

// ─── Projected gradient (box-constrained) ────────────────────────────────
[[nodiscard]] inline OptResult projected_gradient(
    const ObjFn& f, const GradFn& grad,
    VecD x0, const VecD& lb, const VecD& ub,
    f64 lr=0.01, f64 tol=1e-8, int max_iter=10000) noexcept
{
    VecD x=x0;
    for(int i=0;i<int(x.size());++i) x[i]=std::clamp(x[i],lb[i],ub[i]);
    for(int it=0;it<max_iter;++it){
        VecD g=grad(x);
        if(vnorm(g)<tol) return {x,f(x),it,true};
        VecD xnew=x;
        for(int i=0;i<int(x.size());++i)
            xnew[i]=std::clamp(x[i]-lr*g[i],lb[i],ub[i]);
        x=xnew;
    }
    return {x,f(x),max_iter,false};
}

} // namespace qs::math::numerical
