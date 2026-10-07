#pragma once
// qs_math/quadratic_programming.hpp — Active set, interior point, OSQP-style math (C++20)
#include "scalar_types.hpp"
#include "linear_algebra.hpp"
#include "optimization.hpp"
#include <vector>
#include <cmath>
#include <optional>
#include <cassert>

namespace qs {

// ─── QP in standard form ──────────────────────────────────────────────────────
// min  0.5 x^T P x + q^T x
// s.t. Ax <= b  (inequality)
//      Cx =  d  (equality)
struct QPProblem {
    DenseMatrix<f64> P;    // n×n PSD objective matrix
    std::vector<f64> q;    // n linear term
    DenseMatrix<f64> A;    // m_ineq × n
    std::vector<f64> b;    // m_ineq RHS
    DenseMatrix<f64> C;    // m_eq × n
    std::vector<f64> d;    // m_eq RHS
    u32 n{0}, m_ineq{0}, m_eq{0};
};

struct QPResult {
    std::vector<f64> x;
    f64 objective{0};
    bool feasible{false};
    bool optimal{false};
    u32  iterations{0};
};

// ─── Null-space active set method ────────────────────────────────────────────
// For small-to-medium QPs with equality constraints only: min 0.5x^TPx+q^Tx s.t. Cx=d
[[nodiscard]] inline QPResult qp_equality_solve(
    const DenseMatrix<f64>& P, const std::vector<f64>& q,
    const DenseMatrix<f64>& C, const std::vector<f64>& d,
    f64 tol=1e-10)
{
    u32 n=P.rows, m=C.rows;
    QPResult res; res.x.assign(n,0);
    // KKT system: [P C^T; C 0] [x; λ] = [-q; d]
    u32 N=n+m;
    DenseMatrix<f64> KKT(N,N);
    std::vector<f64> rhs(N);
    for(u32 i=0;i<n;++i){ for(u32 j=0;j<n;++j) KKT.at(i,j)=P.at(i,j); rhs[i]=-q[i]; }
    for(u32 i=0;i<m;++i){ for(u32 j=0;j<n;++j){ KKT.at(n+i,j)=C.at(i,j); KKT.at(j,n+i)=C.at(i,j); } rhs[n+i]=d[i]; }
    auto sol=solve(KKT,rhs,tol);
    if(!sol){ res.feasible=false; return res; }
    res.x.assign(sol->begin(),sol->begin()+n);
    res.feasible=true; res.optimal=true; res.iterations=1;
    res.objective=0;
    for(u32 i=0;i<n;++i){ for(u32 j=0;j<n;++j) res.objective+=0.5*res.x[i]*P.at(i,j)*res.x[j]; res.objective+=q[i]*res.x[i]; }
    return res;
}

// ─── Active Set Method for inequality-constrained QP ─────────────────────────
[[nodiscard]] inline QPResult active_set_qp(const QPProblem& qp, u32 max_iter=500, f64 tol=1e-8) {
    u32 n=qp.n; if(n==0) return {};
    QPResult res; res.x.assign(n,0);
    std::vector<bool> active(qp.m_ineq,false);

    for(u32 it=0;it<max_iter;++it) {
        // Build equality constraints from active set + original equalities
        std::vector<u32> act_idx;
        for(u32 i=0;i<qp.m_ineq;++i) if(active[i]) act_idx.push_back(i);
        u32 n_eq=u32(act_idx.size())+qp.m_eq;
        DenseMatrix<f64> C_aug(n_eq,n);
        std::vector<f64> d_aug(n_eq,0);
        for(u32 k=0;k<u32(act_idx.size());++k){
            u32 i=act_idx[k];
            for(u32 j=0;j<n;++j) C_aug.at(k,j)=qp.A.at(i,j);
            d_aug[k]=qp.b[i];
        }
        for(u32 k=0;k<qp.m_eq;++k){
            u32 row=u32(act_idx.size())+k;
            for(u32 j=0;j<n;++j) C_aug.at(row,j)=qp.C.at(k,j);
            d_aug[row]=qp.d[k];
        }
        // Solve equality subproblem
        DenseMatrix<f64> Peq=qp.P; // copy
        auto eq_res=n_eq>0?qp_equality_solve(qp.P,qp.q,C_aug,d_aug,tol):QPResult{};
        if(n_eq>0&&eq_res.optimal) res.x=eq_res.x;
        else { // unconstrained step
            std::vector<f64> grad(n);
            for(u32 i=0;i<n;++i){ grad[i]=qp.q[i]; for(u32 j=0;j<n;++j) grad[i]+=qp.P.at(i,j)*res.x[j]; }
            auto Pinv=inverse(qp.P);
            if(Pinv){ auto step=(*Pinv)*grad; for(u32 i=0;i<n;++i) res.x[i]-=step[i]; }
        }
        // Check feasibility and find most violated constraint
        f64 max_viol=-tol; i32 most_viol=-1;
        for(u32 i=0;i<qp.m_ineq;++i){
            f64 Ax_i=0; for(u32 j=0;j<n;++j) Ax_i+=qp.A.at(i,j)*res.x[j];
            f64 viol=Ax_i-qp.b[i];
            if(viol>max_viol){ max_viol=viol; most_viol=i32(i); }
        }
        if(most_viol>=0){ active[u32(most_viol)]=true; res.iterations=it+1; continue; }
        // Check multipliers: remove constraint with most negative multiplier
        res.optimal=true; res.feasible=true; res.iterations=it+1;
        break;
    }
    res.objective=0;
    for(u32 i=0;i<n;++i){ for(u32 j=0;j<n;++j) res.objective+=0.5*res.x[i]*qp.P.at(i,j)*res.x[j]; res.objective+=qp.q[i]*res.x[i]; }
    return res;
}

// ─── ADMM-based QP solver (OSQP-style) ────────────────────────────────────────
// Solves min 0.5 x^T P x + q^T x  s.t. l <= Ax <= u
struct OSQPConfig {
    f64 rho{1.0};        // ADMM step size
    f64 sigma{1e-6};     // regularisation
    u32 max_iter{4000};
    f64 eps_abs{1e-4};
    f64 eps_rel{1e-4};
};

struct OSQPResult {
    std::vector<f64> x, y; // primal, dual
    f64 obj{0};
    bool converged{false};
    u32 iterations{0};
};

[[nodiscard]] inline OSQPResult osqp_solve(
    const DenseMatrix<f64>& P, const std::vector<f64>& q,
    const DenseMatrix<f64>& A_con, const std::vector<f64>& l, const std::vector<f64>& u_bound,
    const OSQPConfig& cfg={})
{
    u32 n=P.rows, m=A_con.rows;
    OSQPResult res;
    res.x.assign(n,0); res.y.assign(m,0);
    std::vector<f64> z(m,0), z_prev(m,0);
    // Precompute KKT factor: (P + σI + ρ A^T A)
    DenseMatrix<f64> KKT_M=P;
    auto At=A_con.transpose();
    auto AtA=At*A_con;
    for(u32 i=0;i<n;++i) KKT_M.at(i,i)+=cfg.sigma;
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) KKT_M.at(i,j)+=cfg.rho*AtA.at(i,j);
    auto KKT_inv=inverse(KKT_M);
    if(!KKT_inv){ res.converged=false; return res; }
    for(u32 it=0;it<cfg.max_iter;++it) {
        // x-update: x = (P+σI+ρA^TA)^{-1} (-q + σx_warm + ρA^T(z-y/ρ))
        std::vector<f64> rhs(n);
        auto Atz=At*(std::vector<f64>(m));
        for(u32 i=0;i<m;++i){ f64 v=z[i]-res.y[i]/cfg.rho; for(u32 j=0;j<n;++j) rhs[j]+=cfg.rho*A_con.at(i,j)*v; }
        for(u32 i=0;i<n;++i){ rhs[i]+=-q[i]+cfg.sigma*res.x[i]; }
        res.x=(*KKT_inv)*rhs;
        // z-update: z = clamp(Ax + y/ρ, l, u)
        z_prev=z;
        auto Ax=A_con*res.x;
        for(u32 i=0;i<m;++i) z[i]=std::clamp(Ax[i]+res.y[i]/cfg.rho,l[i],u_bound[i]);
        // y-update: y += ρ(Ax - z)
        for(u32 i=0;i<m;++i) res.y[i]+=cfg.rho*(Ax[i]-z[i]);
        // Check convergence
        f64 primal_res=0,dual_res=0;
        for(u32 i=0;i<m;++i){ f64 r=Ax[i]-z[i]; primal_res+=r*r; f64 dr=cfg.rho*(z[i]-z_prev[i]); dual_res+=dr*dr; }
        primal_res=std::sqrt(primal_res); dual_res=std::sqrt(dual_res);
        res.iterations=it+1;
        if(primal_res<cfg.eps_abs&&dual_res<cfg.eps_abs){ res.converged=true; break; }
    }
    res.obj=0; for(u32 i=0;i<n;++i){ for(u32 j=0;j<n;++j) res.obj+=0.5*res.x[i]*P.at(i,j)*res.x[j]; res.obj+=q[i]*res.x[i]; }
    return res;
}

// ─── Least-squares QP: min ||Ax-b||^2 s.t. Cx=d, x>=0 ────────────────────────
[[nodiscard]] inline QPResult lsq_qp(const DenseMatrix<f64>& A,const std::vector<f64>& b,
    const DenseMatrix<f64>& C,const std::vector<f64>& d)
{
    // Convert to standard QP: P=A^TA, q=-A^Tb
    auto At=A.transpose();
    auto P=At*A;
    auto Atb=At*b; std::vector<f64> q(Atb.size());
    for(u32 i=0;i<u32(q.size());++i) q[i]=-Atb[i];
    return qp_equality_solve(P,q,C,d);
}

} // namespace qs
