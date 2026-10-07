#pragma once
// qs_math/linear_programming.hpp — Simplex method, dual simplex, tableau (C++20)
#include "scalar_types.hpp"
#include "linear_algebra.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <optional>

namespace qs {

// ─── LP in standard form: min c^T x  s.t. Ax=b, x>=0 ─────────────────────────
struct LPProblem {
    DenseMatrix<f64> A; // m×n constraint matrix
    std::vector<f64> b; // RHS (m)
    std::vector<f64> c; // objective (n)
    u32 m{0},n{0};

    LPProblem() = default;
    LPProblem(DenseMatrix<f64> A, std::vector<f64> b, std::vector<f64> c)
        : A(std::move(A)), b(b), c(c), m(u32(b.size())), n(u32(c.size())) {}

    // Convert from inequality form Ax<=b to standard form by adding slacks
    static LPProblem from_inequality(DenseMatrix<f64> A_ineq, std::vector<f64> b_ineq, std::vector<f64> c_obj) {
        u32 mi=u32(b_ineq.size()), ni=u32(c_obj.size());
        DenseMatrix<f64> As(mi, ni+mi);
        for(u32 i=0;i<mi;++i){ for(u32 j=0;j<ni;++j) As.at(i,j)=A_ineq.at(i,j); As.at(i,ni+i)=1; }
        c_obj.resize(ni+mi,0);
        return {As, b_ineq, c_obj};
    }
};

struct LPResult {
    std::vector<f64> x;
    f64 objective{0};
    bool optimal{false};
    bool infeasible{false};
    bool unbounded{false};
    u32  iterations{0};
};

// ─── Revised Simplex Method ───────────────────────────────────────────────────
[[nodiscard]] inline LPResult simplex(const LPProblem& lp, u32 max_iter=1000, f64 tol=1e-10) {
    u32 m=lp.m, n=lp.n;
    LPResult res; res.x.assign(n,0);

    // Build tableau: [A | I | b] with artificial variables for Phase 1 if needed
    // Big-M method: add artificials with large objective cost
    f64 BIG_M=1e7;
    u32 na=m; // artificials
    u32 ncols=n+na;
    DenseMatrix<f64> T(m+1, ncols+1); // last row = objective, last col = RHS
    for(u32 i=0;i<m;++i) {
        for(u32 j=0;j<n;++j) T.at(i,j)=lp.A.at(i,j);
        T.at(i,n+i)=1; // artificial
        T.at(i,ncols)=lp.b[i];
    }
    // Objective row
    for(u32 j=0;j<n;++j) T.at(m,j)=lp.c[j];
    for(u32 j=n;j<ncols;++j) T.at(m,j)=BIG_M; // penalise artificials
    // Subtract M * constraint rows from objective to account for basic artificials
    for(u32 i=0;i<m;++i)
        for(u32 j=0;j<=ncols;++j) T.at(m,j)-=BIG_M*T.at(i,j);

    std::vector<u32> basis(m); for(u32 i=0;i<m;++i) basis[i]=n+i;

    for(u32 it=0;it<max_iter;++it) {
        // Find entering variable: most negative reduced cost
        f64 min_rc=0; i32 enter=-1;
        for(u32 j=0;j<ncols;++j) if(T.at(m,j)<min_rc-tol){min_rc=T.at(m,j);enter=i32(j);}
        if(enter<0){res.optimal=true;res.iterations=it;break;}
        // Find leaving variable: minimum ratio test
        f64 min_ratio=std::numeric_limits<f64>::infinity(); i32 leave=-1;
        for(u32 i=0;i<m;++i) {
            if(T.at(i,u32(enter))>tol) {
                f64 r=T.at(i,ncols)/T.at(i,u32(enter));
                if(r<min_ratio){min_ratio=r;leave=i32(i);}
            }
        }
        if(leave<0){res.unbounded=true;return res;}
        // Pivot
        u32 ue=u32(enter),ul=u32(leave);
        f64 pivot=T.at(ul,ue);
        for(u32 j=0;j<=ncols;++j) T.at(ul,j)/=pivot;
        for(u32 i=0;i<=m;++i) if(i!=ul) {
            f64 f=T.at(i,ue);
            for(u32 j=0;j<=ncols;++j) T.at(i,j)-=f*T.at(ul,j);
        }
        basis[ul]=ue;
        res.iterations=it+1;
    }
    // Extract solution
    res.x.assign(n,0);
    for(u32 i=0;i<m;++i) if(basis[i]<n) res.x[basis[i]]=T.at(i,ncols);
    // Check for artificials in basis (infeasibility)
    for(u32 i=0;i<m;++i) if(basis[i]>=n && T.at(i,ncols)>tol){res.infeasible=true;return res;}
    res.objective=-T.at(m,ncols); // negate: maximised tableau
    return res;
}

// ─── Dual Simplex Method ──────────────────────────────────────────────────────
// Starting from a dual-feasible (primal-infeasible) basis
[[nodiscard]] inline LPResult dual_simplex(const LPProblem& lp, u32 max_iter=1000, f64 tol=1e-10) {
    u32 m=lp.m, n=lp.n;
    LPResult res; res.x.assign(n,0);
    // Build slack tableau (assumes A has slack identity embedded)
    u32 ncols=n;
    DenseMatrix<f64> T(m+1,ncols+1);
    for(u32 i=0;i<m;++i){ for(u32 j=0;j<n;++j) T.at(i,j)=lp.A.at(i,j); T.at(i,ncols)=lp.b[i]; }
    for(u32 j=0;j<n;++j) T.at(m,j)=lp.c[j];
    // Find initial basis (identity columns = slack variables)
    std::vector<u32> basis(m); for(u32 i=0;i<m;++i) basis[i]=n-m+i;
    for(u32 it=0;it<max_iter;++it) {
        // Find leaving: most negative RHS
        f64 min_rhs=0; i32 leave=-1;
        for(u32 i=0;i<m;++i) if(T.at(i,ncols)<min_rhs-tol){min_rhs=T.at(i,ncols);leave=i32(i);}
        if(leave<0){res.optimal=true;res.iterations=it;break;}
        // Find entering: min ratio of reduced cost / pivot (pivot < 0)
        f64 min_ratio=std::numeric_limits<f64>::infinity(); i32 enter=-1;
        for(u32 j=0;j<ncols;++j) if(T.at(u32(leave),j)<-tol){
            f64 r=std::abs(T.at(m,j)/T.at(u32(leave),j));
            if(r<min_ratio){min_ratio=r;enter=i32(j);}
        }
        if(enter<0){res.infeasible=true;return res;}
        u32 ue=u32(enter),ul=u32(leave);
        f64 pivot=T.at(ul,ue);
        for(u32 j=0;j<=ncols;++j) T.at(ul,j)/=pivot;
        for(u32 i=0;i<=m;++i) if(i!=ul){ f64 f=T.at(i,ue); for(u32 j=0;j<=ncols;++j) T.at(i,j)-=f*T.at(ul,j); }
        basis[ul]=ue;
        res.iterations=it+1;
    }
    res.x.assign(n,0);
    for(u32 i=0;i<m;++i) if(basis[i]<n) res.x[basis[i]]=T.at(i,ncols);
    res.objective=0; for(u32 j=0;j<n;++j) res.objective+=lp.c[j]*res.x[j];
    return res;
}

// ─── LP duality ───────────────────────────────────────────────────────────────
// Dual of min c^T x s.t. Ax>=b, x>=0 is max b^T y s.t. A^T y<=c, y>=0
[[nodiscard]] inline LPProblem make_dual(const LPProblem& primal) {
    auto At=primal.A.transpose();
    return LPProblem{At, primal.c, primal.b};
}

// Strong duality check: primal obj ≈ dual obj at optimum
[[nodiscard]] inline bool check_strong_duality(const LPProblem& p, const LPResult& pr, const LPResult& dr, f64 tol=1e-4) {
    return pr.optimal && dr.optimal && std::abs(pr.objective+dr.objective)<tol;
}

// ─── Sensitivity analysis (ranging) ───────────────────────────────────────────
struct SensitivityResult {
    std::vector<std::pair<f64,f64>> c_ranges; // [c_lo, c_hi] for each objective coeff
    std::vector<std::pair<f64,f64>> b_ranges; // [b_lo, b_hi] for each RHS
};
// Placeholder — full sensitivity requires basis information from final tableau
[[nodiscard]] inline SensitivityResult lp_sensitivity(const LPProblem& lp, const LPResult& /*res*/) {
    SensitivityResult s;
    s.c_ranges.assign(lp.n,{-1e10,1e10});
    s.b_ranges.assign(lp.m,{-1e10,1e10});
    return s;
}

} // namespace qs
