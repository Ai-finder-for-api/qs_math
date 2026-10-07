#pragma once
// qs_math/optimization.hpp — Convex/non-convex overview, KKT conditions, duality (C++20)
#include "scalar_types.hpp"
#include "calculus.hpp"
#include "linear_algebra.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <optional>
#include <cassert>
#include <limits>

namespace qs {

// ─── Optimality conditions ────────────────────────────────────────────────────

// Check first-order KKT conditions for constrained optimisation
// min f(x) s.t. g_i(x) <= 0, h_j(x) = 0
struct KKTResult {
    bool primal_feasible{false};
    bool dual_feasible{false};   // λ_i >= 0
    bool complementary_slack{false}; // λ_i * g_i(x) ≈ 0
    bool stationarity{false};    // ∇f + Σλ_i ∇g_i + Σμ_j ∇h_j ≈ 0
    bool is_kkt{false};
};

[[nodiscard]] inline KKTResult check_kkt(
    const std::vector<f64>& x,
    const FuncND& f,
    const std::vector<FuncND>& g_ineq, // g_i <= 0
    const std::vector<FuncND>& h_eq,   // h_j = 0
    const std::vector<f64>& lambda,    // multipliers for ineq
    const std::vector<f64>& mu,        // multipliers for eq
    f64 tol=1e-6)
{
    KKTResult r;
    u32 m=u32(g_ineq.size()), p_=u32(h_eq.size());
    // Primal feasibility
    r.primal_feasible=true;
    for(auto& g:g_ineq) if(g(x)>tol){r.primal_feasible=false;break;}
    if(r.primal_feasible) for(auto& h:h_eq) if(std::abs(h(x))>tol){r.primal_feasible=false;break;}
    // Dual feasibility
    r.dual_feasible=true;
    for(auto l:lambda) if(l<-tol){r.dual_feasible=false;break;}
    // Complementary slackness
    r.complementary_slack=true;
    for(u32 i=0;i<m;++i) if(std::abs(lambda[i]*g_ineq[i](x))>tol){r.complementary_slack=false;break;}
    // Stationarity: ∇L ≈ 0
    auto gf=gradient(f,x);
    r.stationarity=true;
    for(u32 k=0;k<u32(x.size());++k) {
        f64 grad_L=gf[k];
        for(u32 i=0;i<m;++i){ auto gg=gradient(g_ineq[i],x); grad_L+=lambda[i]*gg[k]; }
        for(u32 j=0;j<p_;++j){ auto gh=gradient(h_eq[j],x); grad_L+=mu[j]*gh[k]; }
        if(std::abs(grad_L)>tol){r.stationarity=false;break;}
    }
    r.is_kkt=r.primal_feasible&&r.dual_feasible&&r.complementary_slack&&r.stationarity;
    return r;
}

// ─── Gradient descent variants ────────────────────────────────────────────────
struct OptResult {
    std::vector<f64> x;
    f64 f_val{0};
    u32 iterations{0};
    bool converged{false};
    f64 grad_norm{0};
};

// Vanilla gradient descent with backtracking line search
[[nodiscard]] inline OptResult gradient_descent(
    const FuncND& f, std::vector<f64> x0,
    f64 lr=0.01, u32 max_iter=1000, f64 tol=1e-6)
{
    OptResult res; res.x=x0;
    for(u32 it=0;it<max_iter;++it) {
        auto g=gradient(f,res.x);
        f64 gnorm=0; for(auto v:g) gnorm+=v*v; gnorm=std::sqrt(gnorm);
        res.grad_norm=gnorm;
        if(gnorm<tol){ res.converged=true; res.iterations=it; break; }
        // Backtracking Armijo line search
        f64 step=lr; f64 fx=f(res.x);
        std::vector<f64> xn(res.x.size());
        for(u32 ls=0;ls<20;++ls) {
            for(u32 i=0;i<u32(res.x.size());++i) xn[i]=res.x[i]-step*g[i];
            if(f(xn)<=fx-0.5*step*gnorm*gnorm) break;
            step*=0.5;
        }
        res.x=xn;
        res.iterations=it+1;
    }
    res.f_val=f(res.x);
    return res;
}

// Adam optimizer
[[nodiscard]] inline OptResult adam(const FuncND& f, std::vector<f64> x0,
    f64 lr=1e-3, f64 beta1=0.9, f64 beta2=0.999, f64 eps=1e-8,
    u32 max_iter=1000, f64 tol=1e-6)
{
    u32 n=u32(x0.size());
    OptResult res; res.x=x0;
    std::vector<f64> m(n,0),v(n,0);
    for(u32 it=0;it<max_iter;++it) {
        auto g=gradient(f,res.x);
        f64 gnorm=0; for(auto gv:g) gnorm+=gv*gv; gnorm=std::sqrt(gnorm);
        res.grad_norm=gnorm;
        if(gnorm<tol){res.converged=true;res.iterations=it;break;}
        f64 t=f64(it+1);
        for(u32 i=0;i<n;++i) {
            m[i]=beta1*m[i]+(1-beta1)*g[i];
            v[i]=beta2*v[i]+(1-beta2)*g[i]*g[i];
            f64 m_hat=m[i]/(1-std::pow(beta1,t));
            f64 v_hat=v[i]/(1-std::pow(beta2,t));
            res.x[i]-=lr*m_hat/(std::sqrt(v_hat)+eps);
        }
        res.iterations=it+1;
    }
    res.f_val=f(res.x);
    return res;
}

// Nesterov Accelerated Gradient
[[nodiscard]] inline OptResult nag(const FuncND& f, std::vector<f64> x0,
    f64 lr=0.01, u32 max_iter=1000, f64 tol=1e-6)
{
    u32 n=u32(x0.size());
    OptResult res; res.x=x0;
    std::vector<f64> y=x0;
    f64 t=1;
    for(u32 it=0;it<max_iter;++it) {
        auto g=gradient(f,y);
        f64 gnorm=0; for(auto gv:g) gnorm+=gv*gv; gnorm=std::sqrt(gnorm);
        res.grad_norm=gnorm;
        if(gnorm<tol){res.converged=true;res.iterations=it;break;}
        std::vector<f64> x_new(n);
        for(u32 i=0;i<n;++i) x_new[i]=y[i]-lr*g[i];
        f64 t_new=(1+std::sqrt(1+4*t*t))*0.5;
        f64 beta=(t-1)/t_new;
        for(u32 i=0;i<n;++i) y[i]=x_new[i]+beta*(x_new[i]-res.x[i]);
        res.x=x_new; t=t_new;
        res.iterations=it+1;
    }
    res.f_val=f(res.x);
    return res;
}

// ─── Nelder-Mead (derivative-free) ───────────────────────────────────────────
[[nodiscard]] inline OptResult nelder_mead(const FuncND& f, std::vector<f64> x0,
    f64 step=0.1, u32 max_iter=5000, f64 tol=1e-8)
{
    u32 n=u32(x0.size());
    // Build initial simplex
    std::vector<std::vector<f64>> simplex(n+1,x0);
    for(u32 i=0;i<n;++i) simplex[i+1][i]+=step;
    std::vector<f64> fvals(n+1);
    for(u32 i=0;i<=n;++i) fvals[i]=f(simplex[i]);

    auto sort_simplex=[&](){ // sort by f value
        std::vector<u32> idx(n+1); std::iota(idx.begin(),idx.end(),0);
        std::sort(idx.begin(),idx.end(),[&](u32 a,u32 b){return fvals[a]<fvals[b];});
        auto sv=simplex; auto fv=fvals;
        for(u32 i=0;i<=n;++i){simplex[i]=sv[idx[i]];fvals[i]=fv[idx[i]];}
    };

    OptResult res;
    for(u32 it=0;it<max_iter;++it) {
        sort_simplex();
        // Convergence check
        f64 spread=0; for(u32 i=1;i<=n;++i) spread+=std::abs(fvals[i]-fvals[0]);
        if(spread<tol){res.converged=true;res.iterations=it;break;}
        // Centroid of all but worst
        std::vector<f64> cen(n,0);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) cen[j]+=simplex[i][j]; for(auto& v:cen) v/=n;
        // Reflect
        std::vector<f64> xr(n); for(u32 j=0;j<n;++j) xr[j]=cen[j]+1*(cen[j]-simplex[n][j]);
        f64 fr=f(xr);
        if(fr<fvals[0]) { // expand
            std::vector<f64> xe(n); for(u32 j=0;j<n;++j) xe[j]=cen[j]+2*(cen[j]-simplex[n][j]);
            f64 fe=f(xe);
            simplex[n]=fe<fr?xe:xr; fvals[n]=fe<fr?fe:fr;
        } else if(fr<fvals[n-1]) {
            simplex[n]=xr; fvals[n]=fr;
        } else { // contract
            std::vector<f64> xc(n); for(u32 j=0;j<n;++j) xc[j]=cen[j]+0.5*(simplex[n][j]-cen[j]);
            f64 fc=f(xc);
            if(fc<fvals[n]){simplex[n]=xc;fvals[n]=fc;}
            else { // shrink
                for(u32 i=1;i<=n;++i){ for(u32 j=0;j<n;++j) simplex[i][j]=simplex[0][j]+0.5*(simplex[i][j]-simplex[0][j]); fvals[i]=f(simplex[i]); }
            }
        }
        res.iterations=it+1;
    }
    sort_simplex();
    res.x=simplex[0]; res.f_val=fvals[0];
    return res;
}

// ─── Convexity checks ─────────────────────────────────────────────────────────

// Check if f is convex along a line segment [x0, x1] (Jensen's inequality)
[[nodiscard]] inline bool is_convex_along_segment(const FuncND& f,
    const std::vector<f64>& x0, const std::vector<f64>& x1,
    u32 checks=20, f64 tol=1e-6)
{
    u32 n=u32(x0.size()); f64 fx0=f(x0),fx1=f(x1);
    for(u32 k=1;k<checks;++k) {
        f64 t=k/f64(checks);
        std::vector<f64> xt(n); for(u32 i=0;i<n;++i) xt[i]=(1-t)*x0[i]+t*x1[i];
        if(f(xt)>(1-t)*fx0+t*fx1+tol) return false;
    }
    return true;
}

// Lagrangian: L(x,λ,μ) = f(x) + Σλ_i g_i(x) + Σμ_j h_j(x)
[[nodiscard]] inline f64 lagrangian(const FuncND& f, const std::vector<FuncND>& g,
    const std::vector<FuncND>& h,
    const std::vector<f64>& x, const std::vector<f64>& lam, const std::vector<f64>& mu)
{
    f64 L=f(x);
    for(u32 i=0;i<u32(g.size());++i) L+=lam[i]*g[i](x);
    for(u32 j=0;j<u32(h.size());++j) L+=mu[j]*h[j](x);
    return L;
}

// Dual function: g(λ,μ) = inf_x L(x,λ,μ) — evaluated by gradient descent on x
[[nodiscard]] inline f64 dual_function(const FuncND& f,const std::vector<FuncND>& g,
    const std::vector<FuncND>& h,
    const std::vector<f64>& x0,const std::vector<f64>& lam,const std::vector<f64>& mu)
{
    u32 n=u32(x0.size());
    FuncND L=[&](const std::vector<f64>& x){ return lagrangian(f,g,h,x,lam,mu); };
    auto res=gradient_descent(L,x0,0.01,500,1e-6);
    return res.f_val;
}

// Proximal operator: prox_{λf}(v) = argmin_x {f(x) + ||x-v||²/(2λ)}
[[nodiscard]] inline std::vector<f64> proximal_operator(const FuncND& f,
    const std::vector<f64>& v, f64 lambda_prox=1.0)
{
    u32 n=u32(v.size());
    FuncND g=[&](const std::vector<f64>& x) {
        f64 s=f(x);
        for(u32 i=0;i<n;++i) s+=(x[i]-v[i])*(x[i]-v[i])/(2*lambda_prox);
        return s;
    };
    return gradient_descent(g,v,0.01,200,1e-7).x;
}

} // namespace qs
