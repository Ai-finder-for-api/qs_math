
// qs_math/optimization/optimization_impl.cpp (C++20)
#include "include/qs_math_optimization.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::opt {

static void numerical_grad(ObjFn f, const f64* x, i32 n, f64 h, f64* g) noexcept {
    std::vector<f64> xp(x,x+n),xm(x,x+n);
    for(i32 i=0;i<n;++i){
        xp[i]+=h;xm[i]-=h;
        g[i]=(f(xp.data())-f(xm.data()))/(2*h);
        xp[i]=xm[i]=x[i];
    }
}

OptResult gradient_descent(ObjFn f,const f64* x0,i32 n,f64 lr,f64 tol,i32 max_iter) noexcept {
    OptResult r; r.x.assign(x0,x0+n);
    std::vector<f64> g(n);
    for(r.iters=0;r.iters<max_iter;++r.iters){
        numerical_grad(f,r.x.data(),n,1e-6,g.data());
        f64 gnorm=0; for(i32 i=0;i<n;++i) gnorm+=g[i]*g[i];
        if(std::sqrt(gnorm)<tol){r.converged=true;break;}
        // Backtracking line search
        f64 step=lr; const f64 f0=f(r.x.data());
        std::vector<f64> xn(n);
        for(i32 ls=0;ls<20;++ls){
            for(i32 i=0;i<n;++i) xn[i]=r.x[i]-step*g[i];
            if(f(xn.data())<f0-1e-4*step*gnorm) break;
            step*=0.5;
        }
        r.x=xn;
    }
    r.value=f(r.x.data()); return r;
}

OptResult adam(ObjFn f,const f64* x0,i32 n,f64 lr,f64 b1,f64 b2,f64 eps,f64 tol,i32 max_iter) noexcept {
    OptResult r; r.x.assign(x0,x0+n);
    std::vector<f64> m(n,0),v(n,0),g(n);
    for(r.iters=1;r.iters<=max_iter;++r.iters){
        numerical_grad(f,r.x.data(),n,1e-6,g.data());
        f64 gnorm=0;
        for(i32 i=0;i<n;++i) gnorm+=g[i]*g[i];
        if(std::sqrt(gnorm)<tol){r.converged=true;break;}
        const f64 bc1=1-std::pow(b1,r.iters),bc2=1-std::pow(b2,r.iters);
        for(i32 i=0;i<n;++i){
            m[i]=b1*m[i]+(1-b1)*g[i];
            v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
            const f64 mh=m[i]/bc1, vh=v[i]/bc2;
            r.x[i]-=lr*mh/(std::sqrt(vh)+eps);
        }
    }
    r.value=f(r.x.data()); return r;
}

f64 golden_section(std::function<f64(f64)> f,f64 a,f64 b,f64 tol) noexcept {
    constexpr f64 phi=(std::sqrt(5.0)-1)*0.5;
    f64 c=b-phi*(b-a),d=a+phi*(b-a);
    f64 fc=f(c),fd=f(d);
    while(b-a>tol){
        if(fc<fd){b=d;d=c;fd=fc;c=b-phi*(b-a);fc=f(c);}
        else{a=c;c=d;fc=fd;d=a+phi*(b-a);fd=f(d);}
    }
    return (a+b)*0.5;
}

OptResult nelder_mead(ObjFn f,const f64* x0,i32 n,f64 step,f64 tol,i32 max_iter) noexcept {
    OptResult r; r.x.assign(x0,x0+n);
    i32 np1=n+1;
    std::vector<std::vector<f64>> simplex(np1,std::vector<f64>(n));
    std::vector<f64> fv(np1);
    simplex[0]=r.x;
    for(i32 i=1;i<=n;++i){simplex[i]=r.x;simplex[i][i-1]+=step;}
    for(i32 i=0;i<np1;++i)fv[i]=f(simplex[i].data());
    std::vector<i32> idx(np1); std::iota(idx.begin(),idx.end(),0);
    for(r.iters=0;r.iters<max_iter;++r.iters){
        std::sort(idx.begin(),idx.end(),[&](i32 a,i32 b){return fv[a]<fv[b];});
        if(fv[idx[0]]<tol){r.converged=true;break;}
        // Centroid of all but worst
        std::vector<f64> cen(n,0);
        for(i32 i=0;i<n;++i)for(i32 j=0;j<n;++j)cen[j]+=simplex[idx[i]][j]/n;
        // Reflection
        std::vector<f64> xr(n); for(i32 j=0;j<n;++j)xr[j]=cen[j]*2-simplex[idx[n]][j];
        const f64 fr=f(xr.data());
        if(fr<fv[idx[0]]){
            std::vector<f64> xe(n);for(i32 j=0;j<n;++j)xe[j]=cen[j]+2*(xr[j]-cen[j]);
            const f64 fe=f(xe.data());
            simplex[idx[n]]=(fe<fr)?xe:xr; fv[idx[n]]=(fe<fr)?fe:fr;
        }else if(fr<fv[idx[n-1]]){simplex[idx[n]]=xr;fv[idx[n]]=fr;}
        else{
            std::vector<f64> xc(n);for(i32 j=0;j<n;++j)xc[j]=cen[j]+0.5*(simplex[idx[n]][j]-cen[j]);
            const f64 fc=f(xc.data());
            if(fc<fv[idx[n]]){simplex[idx[n]]=xc;fv[idx[n]]=fc;}
            else{
                for(i32 i=1;i<=n;++i)for(i32 j=0;j<n;++j)
                    simplex[idx[i]][j]=simplex[idx[0]][j]+0.5*(simplex[idx[i]][j]-simplex[idx[0]][j]);
                for(i32 i=1;i<=n;++i)fv[idx[i]]=f(simplex[idx[i]].data());
            }
        }
    }
    r.x=simplex[idx[0]]; r.value=fv[idx[0]]; return r;
}

OptResult simulated_annealing(ObjFn f,const f64* x0,i32 n,f64 T0,f64 Tf,f64 cool,f64 step,i32 max_iter,u64 seed) noexcept {
    OptResult r; r.x.assign(x0,x0+n);
    std::vector<f64> best=r.x; f64 fbest=f(r.x.data()),fcur=fbest;
    u64 rng=seed;
    auto randu=[&]()->f64{rng=rng*6364136223846793005ULL+1442695040888963407ULL;return (rng>>11)*(1.0/9007199254740992.0);};
    f64 T=T0;
    for(r.iters=0;r.iters<max_iter&&T>Tf;++r.iters,T*=cool){
        std::vector<f64> xn=r.x;
        i32 idx=(i32)(randu()*n); xn[idx]+=step*(randu()*2-1);
        const f64 fn=f(xn.data());
        if(fn<fcur||randu()<std::exp(-(fn-fcur)/T)){r.x=xn;fcur=fn;}
        if(fcur<fbest){fbest=fcur;best=r.x;}
    }
    r.x=best; r.value=fbest; r.converged=true; return r;
}

OptResult differential_evolution(ObjFn f,i32 n,const f64* lo,const f64* hi,
    i32 pop,f64 F,f64 CR,i32 max_iter,u64 seed) noexcept
{
    u64 rng=seed;
    auto randu=[&]()->f64{rng=rng*6364136223846793005ULL+1442695040888963407ULL;return(rng>>11)*(1.0/9007199254740992.0);};
    auto randi=[&](i32 n_)->i32{return(i32)(randu()*n_);};
    std::vector<std::vector<f64>> pop_v(pop,std::vector<f64>(n));
    std::vector<f64> fv(pop);
    for(i32 i=0;i<pop;++i){
        for(i32 j=0;j<n;++j) pop_v[i][j]=lo[j]+randu()*(hi[j]-lo[j]);
        fv[i]=f(pop_v[i].data());
    }
    OptResult r; r.iters=0;
    for(;r.iters<max_iter;++r.iters){
        for(i32 i=0;i<pop;++i){
            i32 a=randi(pop),b=randi(pop),c=randi(pop);
            while(a==i)a=randi(pop);while(b==i||b==a)b=randi(pop);while(c==i||c==a||c==b)c=randi(pop);
            i32 R=randi(n);
            std::vector<f64> trial(n);
            for(i32 j=0;j<n;++j)
                trial[j]=(j==R||randu()<CR)?std::max(lo[j],std::min(hi[j],pop_v[a][j]+F*(pop_v[b][j]-pop_v[c][j]))):pop_v[i][j];
            const f64 ft=f(trial.data());
            if(ft<fv[i]){pop_v[i]=trial;fv[i]=ft;}
        }
    }
    i32 best=0;for(i32 i=1;i<pop;++i)if(fv[i]<fv[best])best=i;
    r.x=pop_v[best];r.value=fv[best];r.converged=true;return r;
}

OptResult lbfgs(ObjFn f,const f64* x0,i32 n,i32 m,f64 tol,i32 max_iter) noexcept {
    OptResult r; r.x.assign(x0,x0+n);
    std::vector<f64> g(n),g_prev(n),x_prev(n);
    std::vector<std::vector<f64>> s(m,std::vector<f64>(n)),y(m,std::vector<f64>(n));
    std::vector<f64> rho(m);
    i32 k=0;
    numerical_grad(f,r.x.data(),n,1e-6,g.data());
    for(r.iters=0;r.iters<max_iter;++r.iters){
        f64 gnorm=0;for(i32 i=0;i<n;++i)gnorm+=g[i]*g[i];
        if(std::sqrt(gnorm)<tol){r.converged=true;break;}
        // Two-loop recursion
        std::vector<f64> q(g),alpha(m);
        for(i32 i=std::min(r.iters,m)-1;i>=0;--i){
            f64 dot=0;for(i32 j=0;j<n;++j)dot+=s[i][j]*q[j];
            alpha[i]=rho[i]*dot;
            for(i32 j=0;j<n;++j)q[j]-=alpha[i]*y[i][j];
        }
        // Scale
        if(r.iters>0){
            f64 yy=0,sy=0;i32 idx=(k-1+m)%m;
            for(i32 j=0;j<n;++j){yy+=y[idx][j]*y[idx][j];sy+=s[idx][j]*y[idx][j];}
            const f64 scale=sy/(yy+1e-14);
            for(i32 j=0;j<n;++j)q[j]*=scale;
        }
        for(i32 i=0;i<std::min(r.iters,m);++i){
            f64 dot=0;for(i32 j=0;j<n;++j)dot+=y[i][j]*q[j];
            for(i32 j=0;j<n;++j)q[j]+=(alpha[i]-rho[i]*dot)*s[i][j];
        }
        // Line search (simple backtracking)
        f64 step=1;const f64 f0=f(r.x.data());
        std::vector<f64> xn(n);
        for(i32 ls=0;ls<20;++ls){
            for(i32 j=0;j<n;++j)xn[j]=r.x[j]-step*q[j];
            if(f(xn.data())<f0)break; step*=0.5;
        }
        // Update s,y,rho
        i32 ki=k%m;
        for(i32 j=0;j<n;++j)s[ki][j]=xn[j]-r.x[j];
        x_prev=r.x; g_prev=g; r.x=xn;
        numerical_grad(f,r.x.data(),n,1e-6,g.data());
        for(i32 j=0;j<n;++j)y[ki][j]=g[j]-g_prev[j];
        f64 sy=0;for(i32 j=0;j<n;++j)sy+=s[ki][j]*y[ki][j];
        rho[ki]=std::abs(sy)>1e-14?1.0/sy:0;
        k++;
    }
    r.value=f(r.x.data());return r;
}

OptResult conjugate_gradient(ObjFn f,const f64* x0,i32 n,f64 tol,i32 max_iter) noexcept {
    OptResult r;r.x.assign(x0,x0+n);
    std::vector<f64> g(n),g_prev(n),d(n);
    numerical_grad(f,r.x.data(),n,1e-6,g.data());
    for(i32 i=0;i<n;++i)d[i]=-g[i];
    for(r.iters=0;r.iters<max_iter;++r.iters){
        f64 gnorm=0;for(i32 i=0;i<n;++i)gnorm+=g[i]*g[i];
        if(std::sqrt(gnorm)<tol){r.converged=true;break;}
        // Line search
        f64 step=1;const f64 f0=f(r.x.data());
        std::vector<f64> xn(n);
        for(i32 ls=0;ls<30;++ls){
            for(i32 i=0;i<n;++i)xn[i]=r.x[i]+step*d[i];
            if(f(xn.data())<f0-1e-4*step*gnorm)break;step*=0.5;
        }
        r.x=xn;g_prev=g;
        numerical_grad(f,r.x.data(),n,1e-6,g.data());
        // Polak-Ribière beta
        f64 num=0,den=0;
        for(i32 i=0;i<n;++i){num+=g[i]*(g[i]-g_prev[i]);den+=g_prev[i]*g_prev[i];}
        const f64 beta=den>1e-14?std::max(0.0,num/den):0;
        for(i32 i=0;i<n;++i)d[i]=-g[i]+beta*d[i];
    }
    r.value=f(r.x.data());return r;
}
} // namespace qs::math::opt
