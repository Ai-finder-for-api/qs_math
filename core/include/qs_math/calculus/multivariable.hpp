#pragma once
// qs_math/calculus/multivariable.hpp — Gradient, Hessian, Jacobian, curl, div, Laplacian
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Numerical gradient ───────────────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr std::array<f64,N> gradient(F&& f,
    const std::array<f64,N>& x, f64 h=1e-6) noexcept
{
    std::array<f64,N> g{};
    for(int i=0;i<N;++i) {
        std::array<f64,N> xp=x,xm=x;
        xp[i]+=h; xm[i]-=h;
        g[i]=(f(xp)-f(xm))/(2*h);
    }
    return g;
}

// ─── Hessian matrix ───────────────────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N> hessian(F&& f,
    const std::array<f64,N>& x, f64 h=1e-4) noexcept
{
    std::array<std::array<f64,N>,N> H{};
    f64 f0=f(x);
    for(int i=0;i<N;++i) {
        for(int j=i;j<N;++j) {
            std::array<f64,N> xpp=x,xpm=x,xmp=x,xmm=x;
            xpp[i]+=h;xpp[j]+=h;
            xpm[i]+=h;xpm[j]-=h;
            xmp[i]-=h;xmp[j]+=h;
            xmm[i]-=h;xmm[j]-=h;
            H[i][j]=H[j][i]=(f(xpp)-f(xpm)-f(xmp)+f(xmm))/(4*h*h);
        }
    }
    return H;
}

// ─── Jacobian of vector function F:R^M → R^N ──────────────────────────────────
template<int M, int N, typename F>
[[nodiscard]] constexpr std::array<std::array<f64,M>,N> jacobian_num(F&& f,
    const std::array<f64,M>& x, f64 h=1e-6) noexcept
{
    std::array<std::array<f64,M>,N> J{};
    for(int j=0;j<M;++j) {
        std::array<f64,M> xp=x,xm=x;
        xp[j]+=h; xm[j]-=h;
        auto fp=f(xp),fm=f(xm);
        for(int i=0;i<N;++i) J[i][j]=(fp[i]-fm[i])/(2*h);
    }
    return J;
}

// ─── Divergence of vector field ───────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr f64 divergence(F&& field,
    const std::array<f64,N>& x, f64 h=1e-6) noexcept
{
    f64 div=0;
    for(int i=0;i<N;++i) {
        std::array<f64,N> xp=x,xm=x;
        xp[i]+=h; xm[i]-=h;
        div+=(field(xp)[i]-field(xm)[i])/(2*h);
    }
    return div;
}

// ─── Curl (3D only) ───────────────────────────────────────────────────────────
template<typename F>
[[nodiscard]] constexpr std::array<f64,3> curl(F&& field,
    const std::array<f64,3>& x, f64 h=1e-6) noexcept
{
    auto df=[&](int comp, int dir) {
        std::array<f64,3> xp=x,xm=x;
        xp[dir]+=h;xm[dir]-=h;
        return (field(xp)[comp]-field(xm)[comp])/(2*h);
    };
    return {df(2,1)-df(1,2), df(0,2)-df(2,0), df(1,0)-df(0,1)};
}

// ─── Laplacian ────────────────────────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr f64 laplacian(F&& f,
    const std::array<f64,N>& x, f64 h=1e-4) noexcept
{
    f64 lap=0, f0=f(x);
    for(int i=0;i<N;++i) {
        std::array<f64,N> xp=x,xm=x;
        xp[i]+=h;xm[i]-=h;
        lap+=(f(xp)-2*f0+f(xm))/(h*h);
    }
    return lap;
}

// ─── Directional derivative ───────────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr f64 directional_deriv(F&& f,
    const std::array<f64,N>& x, const std::array<f64,N>& dir, f64 h=1e-6) noexcept
{
    std::array<f64,N> xp=x,xm=x;
    for(int i=0;i<N;++i){xp[i]+=h*dir[i];xm[i]-=h*dir[i];}
    return (f(xp)-f(xm))/(2*h);
}

// ─── Line integral ∫_C f·dr ───────────────────────────────────────────────────
template<int N, typename F, typename Gamma>
[[nodiscard]] constexpr f64 line_integral(F&& field, Gamma&& path,
    f64 t0, f64 t1, int steps=200) noexcept
{
    f64 h=(t1-t0)/steps, result=0;
    for(int k=0;k<steps;++k) {
        f64 t=(k+0.5)*h+t0;
        auto p=path(t);
        auto fp=field(p);
        // Approximate dr/dt via finite difference
        auto pp=path(t+h*0.5), pm=path(t-h*0.5);
        f64 dot=0;
        for(int i=0;i<N;++i) dot+=fp[i]*(pp[i]-pm[i]);
        result+=dot;
    }
    return result;
}

// ─── Surface integral ─────────────────────────────────────────────────────────
template<typename F, typename Sigma>
[[nodiscard]] constexpr f64 surface_integral(F&& f, Sigma&& surface,
    f64 u0,f64 u1,f64 v0,f64 v1, int nu=50, int nv=50) noexcept
{
    f64 hu=(u1-u0)/nu, hv=(v1-v0)/nv, result=0;
    for(int i=0;i<nu;++i) for(int j=0;j<nv;++j) {
        f64 u=u0+(i+0.5)*hu, v=v0+(j+0.5)*hv;
        auto p=surface(u,v);
        // Normal via cross product of partial derivatives
        auto pu=surface(u+1e-5,v), pd=surface(u-1e-5,v);
        auto pv=surface(u,v+1e-5), pm=surface(u,v-1e-5);
        std::array<f64,3> ru={(pu[0]-pd[0])/2e-5,(pu[1]-pd[1])/2e-5,(pu[2]-pd[2])/2e-5};
        std::array<f64,3> rv={(pv[0]-pm[0])/2e-5,(pv[1]-pm[1])/2e-5,(pv[2]-pm[2])/2e-5};
        // Cross product
        f64 nx=ru[1]*rv[2]-ru[2]*rv[1];
        f64 ny=ru[2]*rv[0]-ru[0]*rv[2];
        f64 nz=ru[0]*rv[1]-ru[1]*rv[0];
        f64 area=std::sqrt(nx*nx+ny*ny+nz*nz);
        result+=f(p[0],p[1],p[2])*area*hu*hv;
    }
    return result;
}

// ─── Taylor expansion ─────────────────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr f64 taylor_approx(F&& f,
    const std::array<f64,N>& x0, const std::array<f64,N>& x, int order=2) noexcept
{
    f64 val=f(x0);
    if(order<1) return val;
    auto g=gradient(f,x0);
    std::array<f64,N> dx{};
    for(int i=0;i<N;++i){dx[i]=x[i]-x0[i];val+=g[i]*dx[i];}
    if(order<2) return val;
    auto H=hessian(f,x0);
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) val+=0.5*H[i][j]*dx[i]*dx[j];
    return val;
}

// ─── Newton's method for systems ──────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr std::array<f64,N> newton_system(F&& F_vec,
    std::array<f64,N> x, f64 tol=1e-10, int max_iter=100) noexcept
{
    for(int iter=0;iter<max_iter;++iter) {
        auto Fx=F_vec(x);
        f64 norm=0; for(auto v:Fx) norm+=v*v;
        if(norm<tol*tol) break;
        auto J=jacobian_num<N,N>(F_vec,x);
        // Solve J*dx=-Fx (Gaussian elimination)
        std::array<std::array<f64,N+1>,N> aug{};
        for(int i=0;i<N;++i){for(int j=0;j<N;++j)aug[i][j]=J[i][j];aug[i][N]=-Fx[i];}
        for(int k=0;k<N;++k) {
            int p=k; for(int i=k+1;i<N;++i) if(std::abs(aug[i][k])>std::abs(aug[p][k]))p=i;
            std::swap(aug[p],aug[k]);
            if(std::abs(aug[k][k])<1e-15) break;
            for(int i=k+1;i<N;++i){f64 f=aug[i][k]/aug[k][k];for(int j=k;j<=N;++j)aug[i][j]-=f*aug[k][j];}
        }
        std::array<f64,N> dx{};
        for(int i=N-1;i>=0;--i){dx[i]=aug[i][N];for(int j=i+1;j<N;++j)dx[i]-=aug[i][j]*dx[j];dx[i]/=aug[i][i]+1e-15;}
        for(int i=0;i<N;++i) x[i]+=dx[i];
    }
    return x;
}

// ─── Green's theorem verification (2D) ────────────────────────────────────────
// ∮_C (P dx + Q dy) = ∬_D (∂Q/∂x - ∂P/∂y) dA
template<typename P, typename Q>
[[nodiscard]] constexpr f64 greens_theorem_integrand(P&& Pfn, Q&& Qfn,
    f64 x, f64 y, f64 h=1e-6) noexcept
{
    auto dQ_dx=[&](f64 xx,f64 yy){return (Qfn(xx+h,yy)-Qfn(xx-h,yy))/(2*h);};
    auto dP_dy=[&](f64 xx,f64 yy){return (Pfn(xx,yy+h)-Pfn(xx,yy-h))/(2*h);};
    return dQ_dx(x,y)-dP_dy(x,y);
}
} // namespace qs::math
