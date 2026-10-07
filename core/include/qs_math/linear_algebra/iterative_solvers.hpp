#pragma once
// qs_math/linear_algebra/iterative_solvers.hpp — GMRES, BiCGSTAB, SOR, multigrid sketch
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Dense GMRES (restarted, restart=K) ──────────────────────────────────────
template<int N, int K=30>
struct GMRES {
    std::array<std::array<f64,N>,K+1> V{};
    std::array<std::array<f64,K>,K+1> H{};
    std::array<f64,K+1> g{},cs{},sn{};

    template<typename Matvec>
    [[nodiscard]] constexpr std::array<f64,N> solve(
        Matvec&& A, const std::array<f64,N>& b,
        std::array<f64,N> x={}, f64 tol=1e-10, int max_iter=200) noexcept
    {
        auto dot=[](const std::array<f64,N>& u,const std::array<f64,N>& v){
            f64 s=0; for(int i=0;i<N;++i) s+=u[i]*v[i]; return s;
        };
        auto norm=[&](const std::array<f64,N>& u){return std::sqrt(dot(u,u));};
        auto axpy=[](f64 a,const std::array<f64,N>& u,std::array<f64,N> v){
            for(int i=0;i<N;++i) v[i]+=a*u[i]; return v;
        };

        for(int restart=0;restart<max_iter/K+1;++restart) {
            auto r=axpy(-1,A(x),b);
            f64 beta=norm(r);
            if(beta<tol) return x;
            for(int i=0;i<N;++i) V[0][i]=r[i]/beta;
            g={}; g[0]=beta;

            int j=0;
            for(;j<K;++j) {
                auto w=A(V[j]);
                // Arnoldi
                for(int i=0;i<=j;++i) {
                    H[i][j]=dot(w,V[i]);
                    w=axpy(-H[i][j],V[i],w);
                }
                f64 wnorm=norm(w);
                H[j+1][j]=wnorm;
                if(wnorm>1e-12) for(int i=0;i<N;++i) V[j+1][i]=w[i]/wnorm;
                // Apply Givens rotations
                for(int i=0;i<j;++i) {
                    f64 h1=H[i][j],h2=H[i+1][j];
                    H[i][j]=cs[i]*h1+sn[i]*h2;
                    H[i+1][j]=-sn[i]*h1+cs[i]*h2;
                }
                f64 r1=H[j][j],r2=H[j+1][j],rr=std::sqrt(r1*r1+r2*r2);
                cs[j]=rr>1e-15?r1/rr:1; sn[j]=rr>1e-15?r2/rr:0;
                H[j][j]=cs[j]*r1+sn[j]*r2; H[j+1][j]=0;
                g[j+1]=-sn[j]*g[j]; g[j]*=cs[j];
                if(std::abs(g[j+1])<tol) {++j;break;}
            }
            // Solve upper triangular system H[0..j-1][0..j-1]*y=g[0..j-1]
            std::array<f64,K> y{};
            for(int i=j-1;i>=0;--i) {
                y[i]=g[i];
                for(int k=i+1;k<j;++k) y[i]-=H[i][k]*y[k];
                if(std::abs(H[i][i])>1e-15) y[i]/=H[i][i];
            }
            // Update solution x += Vy
            for(int i=0;i<j;++i) x=axpy(y[i],V[i],x);
        }
        return x;
    }
};

// ─── BiCGSTAB ────────────────────────────────────────────────────────────────
template<int N, typename Matvec>
[[nodiscard]] constexpr std::array<f64,N> bicgstab(
    Matvec&& A, const std::array<f64,N>& b,
    std::array<f64,N> x={}, f64 tol=1e-10, int max_iter=500) noexcept
{
    auto dot=[](const std::array<f64,N>& u,const std::array<f64,N>& v){
        f64 s=0;for(int i=0;i<N;++i) s+=u[i]*v[i];return s;
    };
    auto axpy=[](f64 a,const std::array<f64,N>& u,std::array<f64,N> v){
        for(int i=0;i<N;++i) v[i]+=a*u[i];return v;
    };

    auto r=axpy(-1,A(x),b);
    auto r0=r,p=r;
    f64 rho=dot(r0,r),alpha=1,omega=1;
    std::array<f64,N> v{},s{},t{};

    for(int iter=0;iter<max_iter;++iter) {
        f64 rho_new=dot(r0,r);
        if(std::abs(rho)<1e-30) break;
        f64 beta=(rho_new/rho)*(alpha/omega);
        p=axpy(-omega,v,axpy(beta,p,r));
        v=A(p);
        f64 rv=dot(r0,v);
        if(std::abs(rv)<1e-30) break;
        alpha=rho_new/rv;
        s=axpy(-alpha,v,r);
        f64 snorm=dot(s,s);
        if(snorm<tol*tol){x=axpy(alpha,p,x);break;}
        t=A(s);
        f64 tt=dot(t,t);
        omega=tt>1e-30?dot(t,s)/tt:0;
        x=axpy(alpha,p,axpy(omega,s,x));
        r=axpy(-omega,t,s);
        rho=rho_new;
        if(dot(r,r)<tol*tol) break;
    }
    return x;
}

// ─── SOR (Successive Over-Relaxation) ────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> sor(
    const std::array<std::array<f64,N>,N>& A,
    const std::array<f64,N>& b,
    std::array<f64,N> x={},
    f64 omega=1.5, f64 tol=1e-10, int max_iter=1000) noexcept
{
    for(int iter=0;iter<max_iter;++iter) {
        f64 err=0;
        for(int i=0;i<N;++i) {
            f64 sigma=0;
            for(int j=0;j<N;++j) if(j!=i) sigma+=A[i][j]*x[j];
            f64 xi_new=(std::abs(A[i][i])>1e-15)?(b[i]-sigma)/A[i][i]:x[i];
            f64 xi_sor=(1-omega)*x[i]+omega*xi_new;
            err+=std::abs(xi_sor-x[i]);
            x[i]=xi_sor;
        }
        if(err<tol) break;
    }
    return x;
}

// ─── Gauss-Seidel ────────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> gauss_seidel(
    const std::array<std::array<f64,N>,N>& A,
    const std::array<f64,N>& b,
    std::array<f64,N> x={},
    f64 tol=1e-10, int max_iter=1000) noexcept
{
    return sor(A,b,x,1.0,tol,max_iter);
}

// ─── Jacobi iteration ────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> jacobi(
    const std::array<std::array<f64,N>,N>& A,
    const std::array<f64,N>& b,
    std::array<f64,N> x={},
    f64 tol=1e-10, int max_iter=1000) noexcept
{
    std::array<f64,N> xn{};
    for(int iter=0;iter<max_iter;++iter) {
        f64 err=0;
        for(int i=0;i<N;++i) {
            f64 s=b[i];
            for(int j=0;j<N;++j) if(j!=i) s-=A[i][j]*x[j];
            xn[i]=(std::abs(A[i][i])>1e-15)?s/A[i][i]:x[i];
            err+=std::abs(xn[i]-x[i]);
        }
        x=xn;
        if(err<tol) break;
    }
    return x;
}

// ─── MINRES (symmetric indefinite) ────────────────────────────────────────────
template<int N, typename Matvec>
[[nodiscard]] constexpr std::array<f64,N> minres(
    Matvec&& A, const std::array<f64,N>& b,
    std::array<f64,N> x={},
    f64 tol=1e-10, int max_iter=500) noexcept
{
    auto dot=[](const std::array<f64,N>& u,const std::array<f64,N>& v){
        f64 s=0;for(int i=0;i<N;++i)s+=u[i]*v[i];return s;
    };
    auto axpy=[](f64 a,const std::array<f64,N>& u,std::array<f64,N> v){
        for(int i=0;i<N;++i)v[i]+=a*u[i];return v;
    };
    auto scale=[](const std::array<f64,N>& u,f64 a){
        std::array<f64,N> v;for(int i=0;i<N;++i)v[i]=u[i]*a;return v;
    };

    auto r=axpy(-1,A(x),b);
    std::array<f64,N> v=r,v_old={},w={},w_old={};
    f64 beta=std::sqrt(dot(v,v)),beta_old=1,cs=-1,sn=0,eta=beta;
    if(beta>1e-15) v=scale(v,1/beta);
    for(int i=0;i<max_iter;++i) {
        auto Av=A(v);
        f64 alpha=dot(v,Av);
        auto v_new=axpy(-alpha,v,axpy(-beta,v_old,Av));
        f64 beta_new=std::sqrt(dot(v_new,v_new));
        // Lanczos tridiagonal element: alpha on diagonal, beta_new off-diagonal
        f64 delta=cs*alpha+sn*beta_old;
        f64 ep   =-sn*alpha+cs*beta_old; // would be stored for next step
        f64 denom=std::sqrt(alpha*alpha+beta_new*beta_new);
        cs=(denom>1e-15)?alpha/denom:1;
        sn=(denom>1e-15)?beta_new/denom:0;
        if(std::abs(eta*sn)<tol) break;
        auto w_new=axpy(-delta,w,axpy(-ep,w_old,v));
        if(denom>1e-15) w_new=scale(w_new,1/denom);
        x=axpy(cs*eta,w_new,x);
        eta*=-sn;
        v_old=v; v=beta_new>1e-15?scale(v_new,1/beta_new):v_new;
        beta_old=beta; beta=beta_new; w_old=w; w=w_new;
    }
    return x;
}
} // namespace qs::math
