#pragma once
// qs_math/optimization/convex_opt.hpp — LP, QP, interior point, ADMM, proximal methods
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Gradient descent ────────────────────────────────────────────────────────
template<int N, typename Grad>
[[nodiscard]] constexpr std::array<f64,N> gradient_descent(
    Grad&& grad, std::array<f64,N> x,
    f64 lr=1e-3, f64 tol=1e-8, int max_iter=10000) noexcept
{
    for(int iter=0;iter<max_iter;++iter) {
        auto g=grad(x);
        f64 gnorm=0; for(auto v:g) gnorm+=v*v;
        if(gnorm<tol*tol) break;
        for(int i=0;i<N;++i) x[i]-=lr*g[i];
    }
    return x;
}

// ─── Momentum (Heavy Ball) ────────────────────────────────────────────────────
template<int N, typename Grad>
[[nodiscard]] constexpr std::array<f64,N> heavy_ball(
    Grad&& grad, std::array<f64,N> x,
    f64 lr=1e-3, f64 beta=0.9, f64 tol=1e-8, int max_iter=10000) noexcept
{
    std::array<f64,N> m{};
    for(int iter=0;iter<max_iter;++iter) {
        auto g=grad(x);
        f64 gnorm=0; for(auto v:g) gnorm+=v*v;
        if(gnorm<tol*tol) break;
        for(int i=0;i<N;++i){ m[i]=beta*m[i]+(1-beta)*g[i]; x[i]-=lr*m[i]; }
    }
    return x;
}

// ─── Adam optimizer ──────────────────────────────────────────────────────────
template<int N, typename Grad>
[[nodiscard]] constexpr std::array<f64,N> adam(
    Grad&& grad, std::array<f64,N> x,
    f64 lr=1e-3, f64 b1=0.9, f64 b2=0.999, f64 eps=1e-8,
    f64 tol=1e-8, int max_iter=10000) noexcept
{
    std::array<f64,N> m{},v{};
    for(int t=1;t<=max_iter;++t) {
        auto g=grad(x);
        f64 gnorm=0; for(auto vi:g) gnorm+=vi*vi;
        if(gnorm<tol*tol) break;
        for(int i=0;i<N;++i) {
            m[i]=b1*m[i]+(1-b1)*g[i];
            v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
            f64 mh=m[i]/(1-std::pow(b1,t));
            f64 vh=v[i]/(1-std::pow(b2,t));
            x[i]-=lr*mh/(std::sqrt(vh)+eps);
        }
    }
    return x;
}

// ─── AdaGrad ─────────────────────────────────────────────────────────────────
template<int N, typename Grad>
[[nodiscard]] constexpr std::array<f64,N> adagrad(
    Grad&& grad, std::array<f64,N> x,
    f64 lr=0.01, f64 eps=1e-8, int max_iter=10000) noexcept
{
    std::array<f64,N> G{};
    for(int iter=0;iter<max_iter;++iter) {
        auto g=grad(x);
        for(int i=0;i<N;++i) { G[i]+=g[i]*g[i]; x[i]-=lr*g[i]/(std::sqrt(G[i])+eps); }
    }
    return x;
}

// ─── L-BFGS (limited-memory BFGS) ────────────────────────────────────────────
template<int N, int M=10, typename Func, typename Grad>
[[nodiscard]] constexpr std::array<f64,N> lbfgs(
    Func&& f, Grad&& grad, std::array<f64,N> x,
    f64 tol=1e-8, int max_iter=500) noexcept
{
    std::array<std::array<f64,N>,M> s_hist{},y_hist{};
    std::array<f64,M> rho_hist{};
    int hist_sz=0;

    auto dot=[](const std::array<f64,N>& a,const std::array<f64,N>& b){
        f64 s=0;for(int i=0;i<N;++i)s+=a[i]*b[i];return s;
    };
    auto axpy=[](f64 a,const std::array<f64,N>& u,std::array<f64,N> v){
        for(int i=0;i<N;++i)v[i]+=a*u[i];return v;
    };
    auto scale=[](std::array<f64,N> v,f64 a){for(auto& x:v)x*=a;return v;};

    auto g=grad(x);
    for(int iter=0;iter<max_iter;++iter) {
        f64 gnorm=0;for(auto v:g)gnorm+=v*v;
        if(gnorm<tol*tol) break;

        // Two-loop L-BFGS direction
        auto q=g;
        std::array<f64,M> alpha_arr{};
        int actual=hist_sz<M?hist_sz:M;
        for(int i=actual-1;i>=0;--i) {
            int idx=(hist_sz-actual+i)%M;
            alpha_arr[i]=rho_hist[idx]*dot(s_hist[idx],q);
            q=axpy(-alpha_arr[i],y_hist[idx],q);
        }
        // Scale
        f64 gamma=1;
        if(actual>0){
            int last=(hist_sz-1)%M;
            f64 ys=dot(y_hist[last],s_hist[last]);
            f64 yy=dot(y_hist[last],y_hist[last]);
            if(yy>1e-15) gamma=ys/yy;
        }
        auto r=scale(q,gamma);
        for(int i=0;i<actual;++i) {
            int idx=(hist_sz-actual+i)%M;
            f64 beta=rho_hist[idx]*dot(y_hist[idx],r);
            r=axpy(alpha_arr[i]-beta,s_hist[idx],r);
        }

        // Line search (Armijo)
        auto d=scale(r,-1.0);
        f64 f0=f(x), step=1.0;
        for(int ls=0;ls<20;++ls) {
            auto xn=axpy(step,d,x);
            if(f(xn)<=f0+1e-4*step*dot(g,d)) break;
            step*=0.5;
        }
        auto xn=axpy(step,d,x);
        auto gn=grad(xn);

        // Update history
        int idx=hist_sz%M;
        s_hist[idx]=axpy(-1,x,xn);       // s=xn-x
        y_hist[idx]=axpy(-1,g,gn);        // y=gn-g
        f64 ys=dot(s_hist[idx],y_hist[idx]);
        rho_hist[idx]=ys>1e-15?1.0/ys:0;
        ++hist_sz;

        x=xn; g=gn;
    }
    return x;
}

// ─── Projected gradient (box constraints) ─────────────────────────────────────
template<int N, typename Grad>
[[nodiscard]] constexpr std::array<f64,N> projected_gradient(
    Grad&& grad, std::array<f64,N> x,
    const std::array<f64,N>& lo, const std::array<f64,N>& hi,
    f64 lr=1e-3, f64 tol=1e-8, int max_iter=5000) noexcept
{
    for(int iter=0;iter<max_iter;++iter) {
        auto g=grad(x);
        f64 gnorm=0; for(auto v:g) gnorm+=v*v;
        if(gnorm<tol*tol) break;
        for(int i=0;i<N;++i) {
            x[i]-=lr*g[i];
            x[i]=x[i]<lo[i]?lo[i]:(x[i]>hi[i]?hi[i]:x[i]);
        }
    }
    return x;
}

// ─── ADMM (Alternating Direction Method of Multipliers) ───────────────────────
// Solve: min f(x)+g(z) s.t. Ax+Bz=c
// Simplified: min f(x)+g(z), z=x (consensus form)
template<int N, typename ProxF, typename ProxG>
[[nodiscard]] constexpr std::array<f64,N> admm(
    ProxF&& prox_f, ProxG&& prox_g,
    std::array<f64,N> x={}, f64 rho=1.0,
    f64 tol=1e-6, int max_iter=500) noexcept
{
    std::array<f64,N> z=x,u{};
    auto axpy=[](f64 a,const std::array<f64,N>& p,std::array<f64,N> q){
        for(int i=0;i<N;++i)q[i]+=a*p[i];return q;
    };
    for(int iter=0;iter<max_iter;++iter) {
        x=prox_f(axpy(-1,u,z),1.0/rho);  // x-update: prox_{f/rho}(z-u)
        z=prox_g(axpy(1,u,x),1.0/rho);   // z-update: prox_{g/rho}(x+u)
        for(int i=0;i<N;++i) u[i]+=x[i]-z[i]; // dual update
        // Primal residual
        f64 prim=0; for(int i=0;i<N;++i) prim+=(x[i]-z[i])*(x[i]-z[i]);
        if(prim<tol*tol) break;
    }
    return x;
}

// ─── Proximal operators ───────────────────────────────────────────────────────
// Prox of L1: soft thresholding
template<int N>
[[nodiscard]] constexpr std::array<f64,N> prox_l1(
    const std::array<f64,N>& v, f64 lambda) noexcept
{
    std::array<f64,N> x{};
    for(int i=0;i<N;++i) {
        f64 av=v[i]<0?-v[i]:v[i];
        x[i]=(av>lambda)?(v[i]>0?av-lambda:-(av-lambda)):0;
    }
    return x;
}
// Prox of L2²: ridge
template<int N>
[[nodiscard]] constexpr std::array<f64,N> prox_l2sq(
    const std::array<f64,N>& v, f64 lambda) noexcept
{
    std::array<f64,N> x{};
    f64 s=1.0/(1+2*lambda);
    for(int i=0;i<N;++i) x[i]=v[i]*s;
    return x;
}
// Prox of indicator of box [lo,hi]
template<int N>
[[nodiscard]] constexpr std::array<f64,N> prox_box(
    const std::array<f64,N>& v,
    const std::array<f64,N>& lo, const std::array<f64,N>& hi) noexcept
{
    std::array<f64,N> x{};
    for(int i=0;i<N;++i) x[i]=v[i]<lo[i]?lo[i]:(v[i]>hi[i]?hi[i]:v[i]);
    return x;
}

// ─── Frank-Wolfe (conditional gradient) ──────────────────────────────────────
template<int N, typename Grad, typename LinearMin>
[[nodiscard]] constexpr std::array<f64,N> frank_wolfe(
    Grad&& grad, LinearMin&& lin_min,
    std::array<f64,N> x, int max_iter=500) noexcept
{
    for(int t=1;t<=max_iter;++t) {
        auto g=grad(x);
        auto s=lin_min(g);               // minimize <g, s> over feasible set
        f64 gamma=2.0/(t+2);             // diminishing step
        for(int i=0;i<N;++i) x[i]+= gamma*(s[i]-x[i]);
    }
    return x;
}

// ─── Ellipsoid method (feasibility for convex sets) ───────────────────────────
template<int N>
struct Ellipsoid {
    std::array<f64,N>              center{};
    std::array<std::array<f64,N>,N> P{};  // shape matrix

    constexpr void update(const std::array<f64,N>& g) noexcept {
        // g: subgradient at center; update ellipsoid to contain minimizer
        f64 g_Pg=0;
        std::array<f64,N> Pg{};
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) Pg[i]+=P[i][j]*g[j];
        for(int i=0;i<N;++i) g_Pg+=g[i]*Pg[i];
        if(g_Pg<1e-15) return;
        f64 sq=std::sqrt(g_Pg);
        f64 n2=f64(N*N);
        // Center update
        for(int i=0;i<N;++i) center[i]-=Pg[i]/((N+1)*sq);
        // Shape update
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            P[i][j]=n2/(n2-1)*(P[i][j]-2*Pg[i]*Pg[j]/((N+1)*g_Pg));
    }
};
} // namespace qs::math
