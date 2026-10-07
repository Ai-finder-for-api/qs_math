
// qs_math/optimization/optimization_ext_impl.cpp (C++20)
// Extended optimization: evolutionary strategies, particle swarm,
// constrained optimization, multi-objective, trust region.
#include "include/qs_math_optimization.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>
namespace qs::math::opt {

// ─── CMA-ES (Covariance Matrix Adaptation Evolution Strategy) ─────────────────
OptResult cmaes(ObjFn f, const f64* x0, i32 n,
                f64 sigma0=0.3, i32 max_iter=1000,
                f64 tol=1e-12, u64 seed=1) noexcept
{
    // Population size (default heuristic)
    const i32 lam = 4 + (i32)(3*std::log(n));
    const i32 mu  = lam / 2;

    // Weights
    std::vector<f64> weights(mu);
    f64 sum_w=0;
    for(i32 i=0;i<mu;++i){weights[i]=std::log(mu+0.5)-std::log(i+1);sum_w+=weights[i];}
    for(auto& w:weights) w/=sum_w;
    f64 mueff=0; for(auto w:weights)mueff+=w*w; mueff=1/mueff;

    // Step size control
    const f64 cs = (mueff+2)/(n+mueff+5);
    const f64 ds = 1+2*std::max(0.0,std::sqrt((mueff-1)/(n+1))-1)+cs;
    const f64 chi_n = std::sqrt(n)*(1-1.0/(4*n)+1.0/(21*n*n));

    // Covariance matrix adaptation
    const f64 cc  = (4+mueff/n)/(n+4+2*mueff/n);
    const f64 c1  = 2.0/((n+1.3)*(n+1.3)+mueff);
    const f64 cmu = std::min(1-c1, 2*(mueff-2+1/mueff)/((n+2)*(n+2)+mueff));

    // State
    std::vector<f64> m(x0,x0+n);
    std::vector<f64> ps(n,0), pc(n,0);
    std::vector<f64> C(n*n,0); for(i32 i=0;i<n;++i) C[i*n+i]=1;
    std::vector<f64> D(n,1); // eigenvalues
    std::vector<f64> B(n*n,0); for(i32 i=0;i<n;++i) B[i*n+i]=1; // eigenvectors
    f64 sigma=sigma0;

    u64 rng=seed;
    auto randn=[&]()->f64{
        rng=rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u1=std::max(1e-14,(rng>>11)*(1.0/9007199254740992.0));
        rng=rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u2=(rng>>11)*(1.0/9007199254740992.0);
        return std::sqrt(-2*std::log(u1))*std::cos(6.28318530718*u2);
    };

    std::vector<std::vector<f64>> arz(lam,std::vector<f64>(n));
    std::vector<std::vector<f64>> arx(lam,std::vector<f64>(n));
    std::vector<f64> arfitness(lam);

    OptResult r; r.x=m; r.value=f(m.data());

    for(r.iters=0;r.iters<max_iter;++r.iters){
        // Sample population: x = m + sigma * B * D * z
        for(i32 k=0;k<lam;++k){
            for(i32 i=0;i<n;++i) arz[k][i]=randn();
            // y = B * (D * z)
            std::vector<f64> y(n,0);
            for(i32 i=0;i<n;++i)
                for(i32 j=0;j<n;++j)
                    y[i]+=B[i*n+j]*D[j]*arz[k][j];
            for(i32 i=0;i<n;++i) arx[k][i]=m[i]+sigma*y[i];
            arfitness[k]=f(arx[k].data());
        }
        // Sort by fitness
        std::vector<i32> idx(lam); std::iota(idx.begin(),idx.end(),0);
        std::sort(idx.begin(),idx.end(),[&](i32 a,i32 b){return arfitness[a]<arfitness[b];});

        if(arfitness[idx[0]]<r.value){ r.value=arfitness[idx[0]]; r.x=arx[idx[0]]; }

        // New mean
        std::vector<f64> m_new(n,0);
        for(i32 i=0;i<mu;++i)
            for(i32 j=0;j<n;++j)
                m_new[j]+=weights[i]*arx[idx[i]][j];

        // Step size control (CSA)
        // Simplified: just update sigma using success probability
        std::vector<f64> y_w(n,0);
        for(i32 i=0;i<mu;++i)
            for(i32 j=0;j<n;++j)
                y_w[j]+=weights[i]*arz[idx[i]][j];

        // ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * B * y_w
        const f64 c1ps=std::sqrt(cs*(2-cs)*mueff);
        for(i32 i=0;i<n;++i){
            f64 By=0; for(i32 j=0;j<n;++j) By+=B[i*n+j]*y_w[j];
            ps[i]=(1-cs)*ps[i]+c1ps*By;
        }
        f64 ps_norm=0; for(auto v:ps)ps_norm+=v*v; ps_norm=std::sqrt(ps_norm);
        sigma*=std::exp((cs/ds)*(ps_norm/chi_n-1));
        m=m_new;

        // Check convergence
        const f64 dx_norm=sigma*std::sqrt(std::inner_product(y_w.begin(),y_w.end(),y_w.begin(),0.0));
        if(dx_norm<tol&&r.iters>10){ r.converged=true; break; }
    }
    return r;
}

// ─── Particle Swarm Optimization ─────────────────────────────────────────────
OptResult pso(ObjFn f, const f64* lo, const f64* hi, i32 n,
              i32 n_particles=40, i32 max_iter=1000,
              f64 w=0.729, f64 c1=1.494, f64 c2=1.494,
              u64 seed=7, f64 tol=1e-10) noexcept
{
    u64 rng=seed;
    auto randu=[&]()->f64{rng=rng*6364136223846793005ULL+1442695040888963407ULL;return(rng>>11)*(1.0/9007199254740992.0);};

    // Initialise particles
    std::vector<std::vector<f64>> pos(n_particles,std::vector<f64>(n));
    std::vector<std::vector<f64>> vel(n_particles,std::vector<f64>(n,0));
    std::vector<std::vector<f64>> pbest(n_particles);
    std::vector<f64> pbest_f(n_particles,1e300);
    std::vector<f64> gbest(n); f64 gbest_f=1e300;

    for(i32 p=0;p<n_particles;++p){
        for(i32 d=0;d<n;++d) pos[p][d]=lo[d]+randu()*(hi[d]-lo[d]);
        const f64 fp=f(pos[p].data());
        pbest[p]=pos[p]; pbest_f[p]=fp;
        if(fp<gbest_f){gbest_f=fp;gbest=pos[p];}
    }
    OptResult r; r.x=gbest; r.value=gbest_f;
    for(r.iters=0;r.iters<max_iter;++r.iters){
        for(i32 p=0;p<n_particles;++p){
            for(i32 d=0;d<n;++d){
                vel[p][d]=w*vel[p][d]
                    +c1*randu()*(pbest[p][d]-pos[p][d])
                    +c2*randu()*(gbest[d]-pos[p][d]);
                pos[p][d]=std::max(lo[d],std::min(hi[d],pos[p][d]+vel[p][d]));
            }
            const f64 fp=f(pos[p].data());
            if(fp<pbest_f[p]){pbest_f[p]=fp;pbest[p]=pos[p];}
            if(fp<gbest_f){gbest_f=fp;gbest=pos[p];}
        }
        if(gbest_f<tol){r.converged=true;break;}
    }
    r.x=gbest; r.value=gbest_f; return r;
}

// ─── NSGA-II multi-objective optimizer ───────────────────────────────────────
// (Simplified version — Pareto front computation)
struct MOOResult {
    std::vector<std::vector<f64>> pareto_front_x;
    std::vector<std::vector<f64>> pareto_front_f;
};

static bool pareto_dominates(const std::vector<f64>& a, const std::vector<f64>& b) noexcept {
    bool any_better=false;
    for(i32 i=0;i<(i32)a.size();++i){
        if(a[i]>b[i]) return false;
        if(a[i]<b[i]) any_better=true;
    }
    return any_better;
}

MOOResult nsga2(std::vector<std::function<f64(const f64*)>> objectives,
    const f64* lo, const f64* hi, i32 n, i32 pop_size=100,
    i32 max_gen=200, u64 seed=1) noexcept
{
    const i32 n_obj=(i32)objectives.size();
    u64 rng=seed;
    auto randu=[&]()->f64{rng=rng*6364136223846793005ULL+1442695040888963407ULL;return(rng>>11)*(1.0/9007199254740992.0);};

    // Init population
    std::vector<std::vector<f64>> pop(pop_size,std::vector<f64>(n));
    std::vector<std::vector<f64>> fvals(pop_size,std::vector<f64>(n_obj));
    for(i32 p=0;p<pop_size;++p){
        for(i32 d=0;d<n;++d) pop[p][d]=lo[d]+randu()*(hi[d]-lo[d]);
        for(i32 o=0;o<n_obj;++o) fvals[p][o]=objectives[o](pop[p].data());
    }

    for(i32 gen=0;gen<max_gen;++gen){
        // Create offspring via tournament selection + mutation
        std::vector<std::vector<f64>> offspring(pop_size,std::vector<f64>(n));
        for(i32 p=0;p<pop_size;++p){
            // Tournament: pick 2 random parents
            i32 a=(i32)(randu()*pop_size), b=(i32)(randu()*pop_size);
            const std::vector<f64>& parent=pareto_dominates(fvals[a],fvals[b])?pop[a]:pop[b];
            for(i32 d=0;d<n;++d){
                offspring[p][d]=parent[d]+(randu()-0.5)*0.1*(hi[d]-lo[d]);
                offspring[p][d]=std::max(lo[d],std::min(hi[d],offspring[p][d]));
            }
        }
        // Combine and select
        std::vector<std::vector<f64>> combined=pop;
        std::vector<std::vector<f64>> combined_f=fvals;
        for(auto& o:offspring){combined.push_back(o);
            std::vector<f64> fv(n_obj);
            for(i32 oo=0;oo<n_obj;++oo)fv[oo]=objectives[oo](o.data());
            combined_f.push_back(fv);}
        // Non-dominated sort (simplified: keep first pop_size non-dominated)
        const i32 total=(i32)combined.size();
        std::vector<bool> dominated(total,false);
        for(i32 i=0;i<total;++i)
            for(i32 j=0;j<total;++j)
                if(i!=j&&pareto_dominates(combined_f[j],combined_f[i]))
                    {dominated[i]=true;break;}
        std::vector<i32> nd_idx;
        for(i32 i=0;i<total;++i)if(!dominated[i])nd_idx.push_back(i);
        if((i32)nd_idx.size()<pop_size)
            for(i32 i=0;i<total&&(i32)nd_idx.size()<pop_size;++i)
                if(dominated[i])nd_idx.push_back(i);
        pop.clear(); fvals.clear();
        for(i32 i=0;i<pop_size&&i<(i32)nd_idx.size();++i){
            pop.push_back(combined[nd_idx[i]]);
            fvals.push_back(combined_f[nd_idx[i]]);
        }
    }
    // Extract Pareto front
    MOOResult r;
    const i32 sz=(i32)pop.size();
    std::vector<bool> dominated(sz,false);
    for(i32 i=0;i<sz;++i)
        for(i32 j=0;j<sz;++j)
            if(i!=j&&pareto_dominates(fvals[j],fvals[i])){dominated[i]=true;break;}
    for(i32 i=0;i<sz;++i)if(!dominated[i]){r.pareto_front_x.push_back(pop[i]);r.pareto_front_f.push_back(fvals[i]);}
    return r;
}

// ─── Trust region method (dogleg step) ───────────────────────────────────────
OptResult trust_region(ObjFn f, const f64* x0, i32 n,
    f64 delta0=1.0, f64 tol=1e-8, i32 max_iter=500) noexcept
{
    OptResult r; r.x.assign(x0,x0+n);
    std::vector<f64> g(n),p(n);
    f64 delta=delta0;

    auto grad=[&](const std::vector<f64>& x)->std::vector<f64>{
        std::vector<f64> gv(n);
        const f64 h=1e-6;
        std::vector<f64> xp=x,xm=x;
        for(i32 i=0;i<n;++i){
            xp[i]+=h;xm[i]-=h;
            gv[i]=(f(xp.data())-f(xm.data()))/(2*h);
            xp[i]=xm[i]=x[i];
        }
        return gv;
    };

    for(r.iters=0;r.iters<max_iter;++r.iters){
        g=grad(r.x);
        f64 gnorm=0; for(auto v:g)gnorm+=v*v; gnorm=std::sqrt(gnorm);
        if(gnorm<tol){r.converged=true;break;}

        // Cauchy point step
        const f64 gc_norm=gnorm;
        std::vector<f64> p_c(n);
        for(i32 i=0;i<n;++i) p_c[i]=-g[i]/gc_norm*std::min(delta,gc_norm);

        // Propose step = Cauchy point (simplified trust region)
        const f64 f0=f(r.x.data());
        std::vector<f64> xnew(n);
        for(i32 i=0;i<n;++i) xnew[i]=r.x[i]+p_c[i];
        const f64 f1=f(xnew.data());

        // Actual vs predicted reduction
        const f64 actual=f0-f1;
        f64 pred_step=0; for(i32 i=0;i<n;++i)pred_step+=g[i]*p_c[i];
        const f64 predicted=-pred_step;
        const f64 rho=predicted>1e-14?actual/predicted:0;

        if(rho>0.1){ r.x=xnew; r.value=f1; }
        if(rho>0.75) delta=std::min(2*delta,10.0);
        else if(rho<0.25) delta*=0.5;
        if(delta<1e-12){r.converged=true;break;}
    }
    r.value=f(r.x.data());
    return r;
}

// ─── Augmented Lagrangian method (equality constraints) ───────────────────────
OptResult augmented_lagrangian(
    ObjFn f,
    std::vector<std::function<f64(const f64*)>> eq_constraints,
    const f64* x0, i32 n,
    f64 rho0=1.0, f64 tol=1e-8, i32 max_outer=50) noexcept
{
    const i32 m=(i32)eq_constraints.size();
    std::vector<f64> lambda(m,0);
    f64 rho=rho0;
    std::vector<f64> x(x0,x0+n);

    // Augmented objective
    auto AL=[&](const f64* xv)->f64{
        f64 val=f(xv);
        for(i32 i=0;i<m;++i){
            const f64 c=eq_constraints[i](xv);
            val+=lambda[i]*c+0.5*rho*c*c;
        }
        return val;
    };

    OptResult r;
    for(i32 outer=0;outer<max_outer;++outer){
        // Minimise augmented Lagrangian
        auto inner=gradient_descent(AL,x.data(),n,0.001,1e-8,5000);
        x=inner.x;
        // Update multipliers
        f64 cnorm=0;
        for(i32 i=0;i<m;++i){
            const f64 c=eq_constraints[i](x.data());
            lambda[i]+=rho*c;
            cnorm+=c*c;
        }
        cnorm=std::sqrt(cnorm);
        if(cnorm<tol){r.converged=true;break;}
        rho=std::min(rho*1.5,1e6);
        r.iters+=inner.iters;
    }
    r.x=x; r.value=f(x.data()); return r;
}

} // namespace qs::math::opt
