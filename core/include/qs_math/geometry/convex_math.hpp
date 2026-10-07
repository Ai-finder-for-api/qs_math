#pragma once
// qs_math/convex_math.hpp — Convex sets, supporting hyperplanes, Bregman divergence (C++20)
#include "scalar_types.hpp"
#include "optimization.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <cassert>
#include <numeric>

namespace qs {

// ─── Convex set predicates ────────────────────────────────────────────────────

// Projection onto non-negative orthant
[[nodiscard]] inline std::vector<f64> proj_nonneg(std::vector<f64> x) {
    for(auto& v:x) v=std::max(0.0,v); return x;
}

// Projection onto L2 ball ||x||<=r
[[nodiscard]] inline std::vector<f64> proj_l2_ball(std::vector<f64> x, f64 r=1) {
    f64 n=0; for(auto v:x) n+=v*v; n=std::sqrt(n);
    if(n<=r) return x;
    for(auto& v:x) v*=r/n; return x;
}

// Projection onto L∞ ball: |x_i|<=r
[[nodiscard]] inline std::vector<f64> proj_linf_ball(std::vector<f64> x, f64 r=1) {
    for(auto& v:x) v=std::clamp(v,-r,r); return x;
}

// Projection onto L1 ball: ||x||_1<=r (Duchi et al. algorithm)
[[nodiscard]] inline std::vector<f64> proj_l1_ball(std::vector<f64> x, f64 r=1) {
    u32 n=u32(x.size());
    f64 s=0; for(auto v:x) s+=std::abs(v);
    if(s<=r) return x;
    std::vector<f64> u(n); for(u32 i=0;i<n;++i) u[i]=std::abs(x[i]);
    std::sort(u.rbegin(),u.rend());
    f64 cssv=0; i32 rho=-1;
    for(u32 i=0;i<n;++i){ cssv+=u[i]; if(u[i]-(cssv-r)/(i+1)>0) rho=i32(i); }
    f64 theta=(cssv-r)/(rho+1);
    for(u32 i=0;i<n;++i) x[i]=std::copysign(std::max(0.0,std::abs(x[i])-theta),x[i]);
    return x;
}

// Projection onto probability simplex: Σx_i=1, x_i>=0
[[nodiscard]] inline std::vector<f64> proj_simplex(std::vector<f64> x) {
    return proj_l1_ball(x,1); // equivalent when all components positive — more robust:
    u32 n=u32(x.size());
    std::vector<f64> u=x; std::sort(u.rbegin(),u.rend());
    f64 cssv=0,rho=0;
    for(u32 i=0;i<n;++i){ cssv+=u[i]; if(u[i]-(cssv-1)/(i+1)>0) rho=f64(i+1); }
    f64 theta=(cssv-1)/rho;
    for(u32 i=0;i<n;++i) x[i]=std::max(0.0,x[i]-theta);
    return x;
}

// Projection onto affine subspace {x : Ax=b} (unconstrained)
[[nodiscard]] inline std::vector<f64> proj_affine(const std::vector<f64>& x,
    const DenseMatrix<f64>& A, const std::vector<f64>& b)
{
    // x* = x - A^T (AA^T)^{-1} (Ax - b)
    auto At=A.transpose();
    auto Ax=A*x;
    std::vector<f64> res_=b; for(u32 i=0;i<u32(b.size());++i) res_[i]-=Ax[i];
    auto AAt=A*At;
    auto lam=solve(AAt,res_);
    if(!lam) return x;
    auto AtLam=At*(*lam);
    std::vector<f64> proj(x.size());
    for(u32 i=0;i<u32(x.size());++i) proj[i]=x[i]+AtLam[i];
    return proj;
}

// ─── Supporting hyperplane: find normal n, offset b s.t. n^T x >= b for all x in C ─
// Gradient at boundary point for a convex function f defines supporting hyperplane
[[nodiscard]] inline std::pair<std::vector<f64>,f64> supporting_hyperplane(
    const FuncND& f, const std::vector<f64>& x)
{
    auto n=gradient(f,x);
    f64 b=0; for(u32 i=0;i<u32(x.size());++i) b+=n[i]*x[i];
    return {n,b};
}

// ─── Convex hull (2D) — Graham scan ──────────────────────────────────────────
struct Point2D { f64 x,y; };

[[nodiscard]] inline std::vector<Point2D> convex_hull_2d(std::vector<Point2D> pts) {
    u32 n=u32(pts.size()); if(n<3) return pts;
    // Find lowest-y point
    u32 lo=0; for(u32 i=1;i<n;++i) if(pts[i].y<pts[lo].y||(pts[i].y==pts[lo].y&&pts[i].x<pts[lo].x)) lo=i;
    std::swap(pts[0],pts[lo]);
    Point2D pivot=pts[0];
    std::sort(pts.begin()+1,pts.end(),[&](const Point2D& a,const Point2D& b){
        f64 cross=(a.x-pivot.x)*(b.y-pivot.y)-(a.y-pivot.y)*(b.x-pivot.x);
        if(std::abs(cross)>1e-12) return cross>0;
        f64 da=(a.x-pivot.x)*(a.x-pivot.x)+(a.y-pivot.y)*(a.y-pivot.y);
        f64 db=(b.x-pivot.x)*(b.x-pivot.x)+(b.y-pivot.y)*(b.y-pivot.y);
        return da<db;
    });
    std::vector<Point2D> hull={pts[0],pts[1]};
    for(u32 i=2;i<n;++i) {
        while(hull.size()>1) {
            auto& a=hull[hull.size()-2]; auto& b=hull.back();
            f64 cross=(b.x-a.x)*(pts[i].y-a.y)-(b.y-a.y)*(pts[i].x-a.x);
            if(cross>0) break;
            hull.pop_back();
        }
        hull.push_back(pts[i]);
    }
    return hull;
}

// ─── Bregman divergence D_φ(x,y) = φ(x) - φ(y) - <∇φ(y), x-y> ─────────────
// For various φ functions
struct BregmanDivergence {
    // KL divergence: φ(x) = Σ x_i log x_i (for probability vectors)
    [[nodiscard]] static f64 kl(const std::vector<f64>& x, const std::vector<f64>& y) {
        f64 d=0;
        for(u32 i=0;i<u32(x.size());++i)
            if(x[i]>1e-15) d+=x[i]*std::log(x[i]/(y[i]+1e-30));
        return d;
    }
    // Squared Euclidean: φ(x) = ||x||^2
    [[nodiscard]] static f64 squared_euclidean(const std::vector<f64>& x, const std::vector<f64>& y) {
        f64 d=0; for(u32 i=0;i<u32(x.size());++i) d+=(x[i]-y[i])*(x[i]-y[i]); return d;
    }
    // Itakura-Saito: φ(x) = -log x - 1
    [[nodiscard]] static f64 itakura_saito(const std::vector<f64>& x, const std::vector<f64>& y) {
        f64 d=0;
        for(u32 i=0;i<u32(x.size());++i) if(x[i]>1e-15&&y[i]>1e-15)
            d+=x[i]/y[i]-std::log(x[i]/y[i])-1;
        return d;
    }
    // Beta divergence (generalises KL and IS)
    [[nodiscard]] static f64 beta_div(const std::vector<f64>& x,const std::vector<f64>& y,f64 beta) {
        if(std::abs(beta)<1e-9) return itakura_saito(x,y);
        if(std::abs(beta-1)<1e-9) return kl(x,y);
        f64 d=0;
        for(u32 i=0;i<u32(x.size());++i) if(x[i]>0&&y[i]>0)
            d+=(1/(beta*(beta-1)))*(std::pow(x[i],beta)+( beta-1)*std::pow(y[i],beta)-beta*x[i]*std::pow(y[i],beta-1));
        return d;
    }
    // Centre of mass under Bregman divergence (generalised mean)
    [[nodiscard]] static std::vector<f64> centroid_kl(const std::vector<std::vector<f64>>& pts) {
        u32 n=u32(pts.size()),d=u32(pts[0].size());
        std::vector<f64> c(d,0);
        for(const auto& p:pts) for(u32 j=0;j<d;++j) c[j]+=std::log(p[j]+1e-30);
        for(u32 j=0;j<d;++j) c[j]=std::exp(c[j]/n);
        return c;
    }
};

// ─── Moreau envelope and proximal point algorithm ─────────────────────────────
// Moreau envelope: f_λ(x) = min_y {f(y) + ||x-y||²/(2λ)}
[[nodiscard]] inline f64 moreau_envelope(const FuncND& f, const std::vector<f64>& x, f64 lambda=1) {
    u32 n=u32(x.size());
    FuncND g=[&](const std::vector<f64>& y){
        f64 s=f(y);
        for(u32 i=0;i<n;++i) s+=(y[i]-x[i])*(y[i]-x[i])/(2*lambda);
        return s;
    };
    auto res=gradient_descent(g,x,0.01,500,1e-8);
    return res.f_val;
}

// Proximal point algorithm: x_{k+1} = prox_{λf}(x_k)
[[nodiscard]] inline OptResult proximal_point(const FuncND& f, std::vector<f64> x0,
    f64 lambda=1.0, u32 max_iter=200, f64 tol=1e-7)
{
    OptResult res; res.x=x0;
    for(u32 it=0;it<max_iter;++it) {
        u32 n=u32(res.x.size());
        FuncND g=[&](const std::vector<f64>& y){
            f64 s=f(y);
            for(u32 i=0;i<n;++i) s+=(y[i]-res.x[i])*(y[i]-res.x[i])/(2*lambda);
            return s;
        };
        auto xn=gradient_descent(g,res.x,0.01,100,1e-8).x;
        f64 diff=0; for(u32 i=0;i<n;++i) diff+=(xn[i]-res.x[i])*(xn[i]-res.x[i]);
        res.x=xn; res.iterations=it+1;
        if(diff<tol*tol){res.converged=true;break;}
    }
    res.f_val=f(res.x);
    return res;
}

// ─── Frank-Wolfe (conditional gradient) for constrained convex optimisation ──
// min f(x) s.t. x in C, given linear oracle: s = argmin_{y in C} <∇f(x), y>
[[nodiscard]] inline OptResult frank_wolfe(const FuncND& f,
    const std::function<std::vector<f64>(const std::vector<f64>&)>& linear_oracle,
    std::vector<f64> x0, u32 max_iter=500, f64 tol=1e-7)
{
    OptResult res; res.x=x0;
    for(u32 it=0;it<max_iter;++it) {
        auto g=gradient(f,res.x);
        auto s=linear_oracle(g);
        // Duality gap
        u32 n=u32(res.x.size());
        f64 gap=0; for(u32 i=0;i<n;++i) gap+=g[i]*(res.x[i]-s[i]);
        res.grad_norm=gap;
        if(gap<tol){res.converged=true;res.iterations=it;break;}
        // Line search
        f64 gamma=2.0/(it+2); // 2/(k+2) schedule
        for(u32 i=0;i<n;++i) res.x[i]=(1-gamma)*res.x[i]+gamma*s[i];
        res.iterations=it+1;
    }
    res.f_val=f(res.x);
    return res;
}

// ─── ADMM (Alternating Direction Method of Multipliers) ──────────────────────
// Solve: min f(x) + g(z) s.t. Ax + Bz = c
struct ADMMConfig { f64 rho{1.0}; u32 max_iter{500}; f64 tol{1e-6}; };

struct ADMMResult {
    std::vector<f64> x,z,u; // primal, primal, dual
    f64 primal_res{0},dual_res{0};
    u32 iterations{0};
    bool converged{false};
};

[[nodiscard]] inline ADMMResult admm_lasso(
    const DenseMatrix<f64>& A, const std::vector<f64>& b, f64 lambda,
    const ADMMConfig& cfg={})
{
    // Lasso: min 0.5||Ax-b||^2 + lambda||z||_1  s.t. x=z
    u32 n=u32(A.cols);
    ADMMResult res;
    res.x.assign(n,0); res.z.assign(n,0); res.u.assign(n,0);
    // Precompute (A^TA + ρI)^{-1}
    auto At=A.transpose();
    auto AtA=At*A;
    for(u32 i=0;i<n;++i) AtA.at(i,i)+=cfg.rho;
    auto AtA_inv=inverse(AtA);
    auto Atb=At*b;
    for(u32 it=0;it<cfg.max_iter;++it) {
        // x-update: x = (A^TA + ρI)^{-1} (A^Tb + ρ(z-u))
        std::vector<f64> rhs(n);
        for(u32 i=0;i<n;++i) rhs[i]=Atb[i]+cfg.rho*(res.z[i]-res.u[i]);
        if(AtA_inv) res.x=(*AtA_inv)*rhs;
        // z-update: soft thresholding
        std::vector<f64> z_old=res.z;
        for(u32 i=0;i<n;++i){ f64 v=res.x[i]+res.u[i]; res.z[i]=std::copysign(std::max(0.0,std::abs(v)-lambda/cfg.rho),v); }
        // u-update
        for(u32 i=0;i<n;++i) res.u[i]+=res.x[i]-res.z[i];
        // Residuals
        res.primal_res=0; for(u32 i=0;i<n;++i) res.primal_res+=(res.x[i]-res.z[i])*(res.x[i]-res.z[i]);
        res.dual_res=0; for(u32 i=0;i<n;++i) res.dual_res+=(res.z[i]-z_old[i])*(res.z[i]-z_old[i]); res.dual_res*=cfg.rho*cfg.rho;
        res.primal_res=std::sqrt(res.primal_res); res.dual_res=std::sqrt(res.dual_res);
        res.iterations=it+1;
        if(res.primal_res<cfg.tol&&res.dual_res<cfg.tol){res.converged=true;break;}
    }
    return res;
}

} // namespace qs
