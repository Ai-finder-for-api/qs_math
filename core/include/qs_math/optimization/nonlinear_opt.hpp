#pragma once
// qs_math/nonlinear_opt.hpp — L-BFGS, trust region, Levenberg-Marquardt, SQP (C++20)
#include "scalar_types.hpp"
#include "calculus.hpp"
#include "linear_algebra.hpp"
#include "optimization.hpp"
#include <vector>
#include <cmath>
#include <deque>
#include <functional>

namespace qs {

// ─── L-BFGS (Limited-memory BFGS) ───────────────────────────────────────────
[[nodiscard]] inline OptResult lbfgs(const FuncND& f, std::vector<f64> x0,
    u32 memory=10, u32 max_iter=1000, f64 tol=1e-6)
{
    u32 n=u32(x0.size());
    OptResult res; res.x=x0;
    auto g=gradient(f,res.x);
    std::deque<std::vector<f64>> s_hist, y_hist;
    std::deque<f64> rho_hist;
    for(u32 it=0;it<max_iter;++it) {
        f64 gnorm=0; for(auto v:g) gnorm+=v*v; gnorm=std::sqrt(gnorm);
        res.grad_norm=gnorm;
        if(gnorm<tol){res.converged=true;res.iterations=it;break;}
        // Two-loop recursion for search direction
        std::vector<f64> q=g;
        u32 m=u32(s_hist.size());
        std::vector<f64> alpha_v(m);
        for(i32 i=i32(m)-1;i>=0;--i) {
            u32 ui=u32(i);
            f64 rho=rho_hist[ui];
            f64 a=rho*(std::inner_product(s_hist[ui].begin(),s_hist[ui].end(),q.begin(),0.0));
            alpha_v[ui]=a;
            for(u32 j=0;j<n;++j) q[j]-=a*y_hist[ui][j];
        }
        // Initial Hessian approximation
        f64 hk=1;
        if(m>0){ f64 sy=std::inner_product(s_hist.back().begin(),s_hist.back().end(),y_hist.back().begin(),0.0);
                  f64 yy=std::inner_product(y_hist.back().begin(),y_hist.back().end(),y_hist.back().begin(),0.0);
                  hk=sy/(yy+1e-30); }
        std::vector<f64> r(n); for(u32 j=0;j<n;++j) r[j]=hk*q[j];
        for(u32 i=0;i<m;++i) {
            f64 beta=rho_hist[i]*(std::inner_product(y_hist[i].begin(),y_hist[i].end(),r.begin(),0.0));
            for(u32 j=0;j<n;++j) r[j]+=(alpha_v[i]-beta)*s_hist[i][j];
        }
        std::vector<f64> d(n); for(u32 j=0;j<n;++j) d[j]=-r[j];
        // Wolfe line search (strong)
        f64 step=1.0; f64 fx=f(res.x); f64 gd=std::inner_product(g.begin(),g.end(),d.begin(),0.0);
        std::vector<f64> xn(n);
        for(u32 ls=0;ls<30;++ls) {
            for(u32 j=0;j<n;++j) xn[j]=res.x[j]+step*d[j];
            f64 fxn=f(xn);
            auto gn=gradient(f,xn);
            f64 gnd=std::inner_product(gn.begin(),gn.end(),d.begin(),0.0);
            if(fxn>fx+1e-4*step*gd){ step*=0.5; continue; }
            if(std::abs(gnd)<=0.9*std::abs(gd)) break;
            step*=2;
        }
        for(u32 j=0;j<n;++j) xn[j]=res.x[j]+step*d[j];
        auto gn=gradient(f,xn);
        // Update history
        std::vector<f64> sk(n),yk(n);
        for(u32 j=0;j<n;++j){sk[j]=xn[j]-res.x[j];yk[j]=gn[j]-g[j];}
        f64 sy=std::inner_product(sk.begin(),sk.end(),yk.begin(),0.0);
        if(sy>1e-20){
            if(s_hist.size()>=memory){s_hist.pop_front();y_hist.pop_front();rho_hist.pop_front();}
            s_hist.push_back(sk); y_hist.push_back(yk); rho_hist.push_back(1/sy);
        }
        res.x=xn; g=gn;
        res.iterations=it+1;
    }
    res.f_val=f(res.x);
    return res;
}

// ─── Levenberg-Marquardt (nonlinear least squares) ───────────────────────────
// Minimise ||r(x)||^2 where r is a residual vector
using ResidualFunc = std::function<std::vector<f64>(const std::vector<f64>&)>;

[[nodiscard]] inline OptResult levenberg_marquardt(
    const ResidualFunc& residual,
    std::vector<f64> x0,
    u32 max_iter=200, f64 tol=1e-8)
{
    u32 n=u32(x0.size());
    OptResult res; res.x=x0;
    f64 lam=1e-3;
    for(u32 it=0;it<max_iter;++it) {
        auto r=residual(res.x);
        f64 cost=std::inner_product(r.begin(),r.end(),r.begin(),0.0);
        // Jacobian via finite differences
        u32 m=u32(r.size());
        DenseMatrix<f64> J(m,n);
        f64 h=1e-5;
        for(u32 j=0;j<n;++j) {
            auto xp=res.x; xp[j]+=h;
            auto rp=residual(xp);
            for(u32 i=0;i<m;++i) J.at(i,j)=(rp[i]-r[i])/h;
        }
        // JTJ + λI
        auto JT=J.transpose();
        auto JTJ=JT*J;
        for(u32 i=0;i<n;++i) JTJ.at(i,i)+=lam;
        // JTr
        std::vector<f64> JTr(n,0);
        for(u32 i=0;i<n;++i) for(u32 k=0;k<m;++k) JTr[i]+=JT.at(i,k)*r[k];
        // Solve for step
        std::vector<f64> neg_JTr(n); for(u32 i=0;i<n;++i) neg_JTr[i]=-JTr[i];
        auto step_opt=solve(JTJ,neg_JTr);
        if(!step_opt) { lam*=10; continue; }
        auto& dx=*step_opt;
        std::vector<f64> xn(n); for(u32 i=0;i<n;++i) xn[i]=res.x[i]+dx[i];
        auto rn=residual(xn);
        f64 cost_new=std::inner_product(rn.begin(),rn.end(),rn.begin(),0.0);
        f64 rho=(cost-cost_new)/(0.001+cost); // approximate gain ratio
        if(cost_new<cost){ res.x=xn; lam=std::max(1e-10,lam/10); }
        else lam=std::min(1e10,lam*10);
        // Convergence
        f64 step_norm=0; for(auto v:dx) step_norm+=v*v; step_norm=std::sqrt(step_norm);
        res.iterations=it+1;
        if(step_norm<tol){res.converged=true;break;}
    }
    res.f_val=0; auto r=residual(res.x); for(auto v:r) res.f_val+=v*v;
    return res;
}

// ─── Trust-region method (dogleg) ────────────────────────────────────────────
[[nodiscard]] inline OptResult trust_region_dogleg(const FuncND& f, std::vector<f64> x0,
    f64 delta0=1.0, u32 max_iter=500, f64 tol=1e-7)
{
    u32 n=u32(x0.size());
    OptResult res; res.x=x0;
    f64 delta=delta0;
    for(u32 it=0;it<max_iter;++it) {
        auto g=gradient(f,res.x);
        f64 gnorm=0; for(auto v:g) gnorm+=v*v; gnorm=std::sqrt(gnorm);
        res.grad_norm=gnorm;
        if(gnorm<tol){res.converged=true;res.iterations=it;break;}
        auto H_data=hessian(f,res.x);
        DenseMatrix<f64> H(n,n);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) H.at(i,j)=H_data[i][j];
        // Cauchy point: p_c = -t * g, t = min(gnorm^3/(g^T H g), delta)/gnorm
        f64 gHg=0; for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) gHg+=g[i]*H.at(i,j)*g[j];
        f64 t_c=gHg>1e-14?std::min(gnorm*gnorm*gnorm/gHg,delta)/gnorm:delta/gnorm;
        std::vector<f64> p_c(n); for(u32 i=0;i<n;++i) p_c[i]=-t_c*g[i];
        // Newton step
        std::vector<f64> neg_g(n); for(u32 i=0;i<n;++i) neg_g[i]=-g[i];
        auto p_n_opt=solve(H,neg_g);
        std::vector<f64> p(n);
        if(!p_n_opt){ p=p_c; }
        else {
            auto& p_n=*p_n_opt;
            f64 pn_norm=0; for(auto v:p_n) pn_norm+=v*v; pn_norm=std::sqrt(pn_norm);
            if(pn_norm<=delta) p=p_n;
            else {
                // Dogleg: p = p_c + tau*(p_n - p_c) scaled to trust radius
                f64 pc_norm=0; for(auto v:p_c) pc_norm+=v*v; pc_norm=std::sqrt(pc_norm);
                if(pc_norm>=delta){ for(u32 i=0;i<n;++i) p[i]=delta*p_c[i]/pc_norm; }
                else {
                    std::vector<f64> dp(n); for(u32 i=0;i<n;++i) dp[i]=p_n[i]-p_c[i];
                    f64 a=0; for(auto v:dp) a+=v*v;
                    f64 b=0; for(u32 i=0;i<n;++i) b+=p_c[i]*dp[i];
                    f64 c_=0; for(auto v:p_c) c_+=v*v; c_-=delta*delta;
                    f64 disc=b*b-a*c_;
                    f64 tau=disc>=0?(-b+std::sqrt(disc))/a:0;
                    for(u32 i=0;i<n;++i) p[i]=p_c[i]+tau*dp[i];
                }
            }
        }
        // Actual vs predicted reduction
        f64 fx=f(res.x);
        std::vector<f64> xn(n); for(u32 i=0;i<n;++i) xn[i]=res.x[i]+p[i];
        f64 fxn=f(xn);
        f64 pred_red=0; for(u32 i=0;i<n;++i){pred_red-=g[i]*p[i]; for(u32 j=0;j<n;++j) pred_red-=0.5*p[i]*H.at(i,j)*p[j];}
        f64 rho_tr=pred_red>1e-12?(fx-fxn)/pred_red:0;
        if(rho_tr>0.1){res.x=xn;}
        if(rho_tr>0.75){f64 pnorm=0;for(auto v:p)pnorm+=v*v;pnorm=std::sqrt(pnorm);if(pnorm>0.8*delta)delta*=2;}
        else if(rho_tr<0.25) delta*=0.25;
        res.iterations=it+1;
    }
    res.f_val=f(res.x);
    return res;
}

// ─── Sequential Quadratic Programming (SQP, single step) ─────────────────────
// Solves: min f(x) s.t. h(x)=0 (equality constraints only, for simplicity)
// Each iteration solves a QP subproblem
[[nodiscard]] inline OptResult sqp(const FuncND& f,
    const std::vector<FuncND>& h_eq,
    std::vector<f64> x0, u32 max_iter=100, f64 tol=1e-7)
{
    u32 n=u32(x0.size()), p_=u32(h_eq.size());
    OptResult res; res.x=x0;
    std::vector<f64> lam(p_,0);
    for(u32 it=0;it<max_iter;++it) {
        auto gf=gradient(f,res.x);
        // Build KKT system for QP subproblem: [H A^T; A 0][dx; dlam] = [-gL; -h]
        auto H_data=hessian(f,res.x);
        u32 N=n+p_;
        DenseMatrix<f64> KKT(N,N);
        std::vector<f64> rhs(N);
        // Hessian block
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) KKT.at(i,j)=H_data[i][j];
        // Constraint Jacobian blocks
        for(u32 k=0;k<p_;++k) {
            auto gh=gradient(h_eq[k],res.x);
            for(u32 i=0;i<n;++i){ KKT.at(i,n+k)=gh[i]; KKT.at(n+k,i)=gh[i]; }
        }
        // RHS: gradient of Lagrangian
        for(u32 i=0;i<n;++i){
            rhs[i]=-gf[i];
            for(u32 k=0;k<p_;++k){auto gh=gradient(h_eq[k],res.x);rhs[i]+=lam[k]*gh[i];}
            rhs[i]=-rhs[i];
        }
        for(u32 k=0;k<p_;++k) rhs[n+k]=-h_eq[k](res.x);
        auto sol=solve(KKT,rhs);
        if(!sol) break;
        f64 step=1; // simple unit step (no line search here)
        for(u32 i=0;i<n;++i) res.x[i]+=step*(*sol)[i];
        for(u32 k=0;k<p_;++k) lam[k]+=(*sol)[n+k];
        // Check convergence
        f64 resid=0; for(u32 k=0;k<p_;++k) resid+=std::abs(h_eq[k](res.x));
        auto gg=gradient(f,res.x); f64 gnorm=0; for(auto v:gg) gnorm+=v*v;
        res.grad_norm=std::sqrt(gnorm); res.iterations=it+1;
        if(resid<tol&&gnorm<tol*tol){res.converged=true;break;}
    }
    res.f_val=f(res.x);
    return res;
}

} // namespace qs
