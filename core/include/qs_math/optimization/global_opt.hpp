#pragma once
// qs_math/optimization/global_opt.hpp — Simulated annealing, CMA-ES, differential evolution
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Simple PRNG (xorshift64, no external dependency) ─────────────────────────
struct Xorshift64 {
    u64 state{12345678901234567ULL};
    [[nodiscard]] constexpr u64 next() noexcept {
        state^=state<<13; state^=state>>7; state^=state<<17; return state;
    }
    [[nodiscard]] constexpr f64 uniform() noexcept {
        return f64(next()>>11)*5.421010862427522e-20; // /2^53
    }
    [[nodiscard]] constexpr f64 normal() noexcept {
        // Box-Muller
        f64 u1=uniform()+1e-15,u2=uniform();
        return std::sqrt(-2*std::log(u1))*std::cos(2*3.14159265358979323846*u2);
    }
};

// ─── Simulated annealing ──────────────────────────────────────────────────────
template<int N, typename Func>
[[nodiscard]] constexpr std::array<f64,N> simulated_annealing(
    Func&& f, std::array<f64,N> x,
    f64 T0=1.0, f64 alpha=0.99, f64 step=0.1,
    int max_iter=100000, u64 seed=42) noexcept
{
    Xorshift64 rng{seed};
    f64 fx=f(x),T=T0;
    std::array<f64,N> best=x; f64 best_f=fx;

    for(int iter=0;iter<max_iter;++iter) {
        // Perturb
        std::array<f64,N> xn=x;
        int dim=int(rng.next()%N);
        xn[dim]+=step*rng.normal();
        f64 fxn=f(xn);
        f64 delta=fxn-fx;
        if(delta<0||rng.uniform()<std::exp(-delta/T)){
            x=xn; fx=fxn;
            if(fx<best_f){best=x;best_f=fx;}
        }
        T*=alpha;
    }
    return best;
}

// ─── CMA-ES (Covariance Matrix Adaptation Evolution Strategy) ────────────────
template<int N, typename Func>
[[nodiscard]] constexpr std::array<f64,N> cma_es(
    Func&& f, std::array<f64,N> mean,
    f64 sigma0=0.3, int max_iter=500, u64 seed=99) noexcept
{
    constexpr int lam=4+int(3*std::log(f64(N))+1); // population size
    constexpr int mu=lam/2;                          // parents

    Xorshift64 rng{seed};
    std::array<f64,N> xmean=mean;
    f64 sigma=sigma0;
    std::array<std::array<f64,N>,N> C{}; // covariance
    for(int i=0;i<N;++i) C[i][i]=1;     // identity init
    std::array<f64,N> pc{},ps{};         // evolution paths
    f64 chiN=std::sqrt(f64(N))*(1-1.0/(4*N)+1.0/(21*N*N));
    f64 cc=4.0/(N+4), cs=4.0/(N+4), c1=2.0/((N+1.3)*(N+1.3));
    f64 cmu=2.0*(0.25-1.0/(N+4))/(N*N), damps=1+2*std::max(0.0,std::sqrt((mu-1.0)/(N+1))-1)+cs;

    std::array<f64,lam>                      fits{};
    std::array<std::array<f64,N>,lam>        pop{};
    std::array<int,lam>                       order{};

    for(int gen=0;gen<max_iter;++gen) {
        // Sample population
        for(int k=0;k<lam;++k) {
            std::array<f64,N> z{};
            for(int i=0;i<N;++i) z[i]=rng.normal();
            // x = mean + sigma*C^{1/2}*z (approximate: diagonal only for header simplicity)
            for(int i=0;i<N;++i) {
                pop[k][i]=xmean[i];
                for(int j=0;j<N;++j) pop[k][i]+=sigma*C[i][j]*z[j];
            }
            fits[k]=f(pop[k]);
            order[k]=k;
        }
        // Sort by fitness
        for(int i=0;i<lam-1;++i)
            for(int j=i+1;j<lam;++j)
                if(fits[order[j]]<fits[order[i]]) std::swap(order[i],order[j]);

        // Weights (log-sum)
        std::array<f64,mu> w{};
        f64 wsum=0;
        for(int i=0;i<mu;++i){w[i]=std::log(mu+0.5)-std::log(f64(i+1));wsum+=w[i];}
        for(auto& wi:w) wi/=wsum;
        f64 mueff=0; { f64 s=0;for(int i=0;i<mu;++i)s+=w[i]*w[i];mueff=1/s; }

        // New mean
        std::array<f64,N> old_mean=xmean;
        xmean={};
        for(int i=0;i<mu;++i) for(int d=0;d<N;++d) xmean[d]+=w[i]*pop[order[i]][d];

        // Update ps, pc, sigma
        std::array<f64,N> step_vec{};
        for(int d=0;d<N;++d) step_vec[d]=(xmean[d]-old_mean[d])/sigma;
        for(int d=0;d<N;++d) ps[d]=(1-cs)*ps[d]+std::sqrt(cs*(2-cs)*mueff)*step_vec[d];
        f64 ps_norm=0; for(auto v:ps) ps_norm+=v*v; ps_norm=std::sqrt(ps_norm);
        sigma*=std::exp((cs/damps)*(ps_norm/chiN-1));

        for(int d=0;d<N;++d) pc[d]=(1-cc)*pc[d]+std::sqrt(cc*(2-cc)*mueff)*step_vec[d];
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            C[i][j]=(1-c1-cmu)*C[i][j]+c1*pc[i]*pc[j];
        for(int k=0;k<mu;++k) {
            std::array<f64,N> dv{};
            for(int d=0;d<N;++d) dv[d]=(pop[order[k]][d]-old_mean[d])/sigma;
            for(int i=0;i<N;++i) for(int j=0;j<N;++j)
                C[i][j]+=cmu*w[k]*dv[i]*dv[j];
        }
        if(sigma<1e-12) break;
    }
    return xmean;
}

// ─── Differential Evolution ───────────────────────────────────────────────────
template<int N, int POP=10, typename Func>
[[nodiscard]] constexpr std::array<f64,N> differential_evolution(
    Func&& f,
    const std::array<f64,N>& lo, const std::array<f64,N>& hi,
    f64 F=0.8, f64 CR=0.9,
    int max_gen=1000, u64 seed=7) noexcept
{
    Xorshift64 rng{seed};
    std::array<std::array<f64,N>,POP> pop{};
    std::array<f64,POP> fits{};

    // Initialize
    for(int i=0;i<POP;++i) {
        for(int d=0;d<N;++d) pop[i][d]=lo[d]+rng.uniform()*(hi[d]-lo[d]);
        fits[i]=f(pop[i]);
    }

    for(int gen=0;gen<max_gen;++gen) {
        for(int i=0;i<POP;++i) {
            // Pick 3 distinct indices != i
            int a=int(rng.next()%POP),b,c;
            do{b=int(rng.next()%POP);}while(b==i||b==a);
            do{c=int(rng.next()%POP);}while(c==i||c==a||c==b);
            // Mutant vector
            std::array<f64,N> trial{};
            int R=int(rng.next()%N);
            for(int d=0;d<N;++d) {
                if(d==R||rng.uniform()<CR) {
                    trial[d]=pop[a][d]+F*(pop[b][d]-pop[c][d]);
                    trial[d]=trial[d]<lo[d]?lo[d]:(trial[d]>hi[d]?hi[d]:trial[d]);
                } else trial[d]=pop[i][d];
            }
            f64 ft=f(trial);
            if(ft<fits[i]){pop[i]=trial;fits[i]=ft;}
        }
    }
    // Return best
    int best=0; for(int i=1;i<POP;++i) if(fits[i]<fits[best]) best=i;
    return pop[best];
}

// ─── Particle Swarm Optimization ──────────────────────────────────────────────
template<int N, int POP=30, typename Func>
[[nodiscard]] constexpr std::array<f64,N> pso(
    Func&& f,
    const std::array<f64,N>& lo, const std::array<f64,N>& hi,
    int max_iter=500, u64 seed=13) noexcept
{
    Xorshift64 rng{seed};
    constexpr f64 w=0.729,c1=1.494,c2=1.494;
    std::array<std::array<f64,N>,POP> pos{},vel{},pbest{};
    std::array<f64,POP> pbest_f{};
    std::array<f64,N> gbest{};
    f64 gbest_f=1e300;

    for(int i=0;i<POP;++i) {
        for(int d=0;d<N;++d) {
            pos[i][d]=lo[d]+rng.uniform()*(hi[d]-lo[d]);
            vel[i][d]=(rng.uniform()-0.5)*(hi[d]-lo[d]);
        }
        pbest[i]=pos[i];
        pbest_f[i]=f(pos[i]);
        if(pbest_f[i]<gbest_f){gbest=pbest[i];gbest_f=pbest_f[i];}
    }

    for(int iter=0;iter<max_iter;++iter) {
        for(int i=0;i<POP;++i) {
            for(int d=0;d<N;++d) {
                f64 r1=rng.uniform(),r2=rng.uniform();
                vel[i][d]=w*vel[i][d]+c1*r1*(pbest[i][d]-pos[i][d])+c2*r2*(gbest[d]-pos[i][d]);
                pos[i][d]+=vel[i][d];
                pos[i][d]=pos[i][d]<lo[d]?lo[d]:(pos[i][d]>hi[d]?hi[d]:pos[i][d]);
            }
            f64 fi=f(pos[i]);
            if(fi<pbest_f[i]){pbest[i]=pos[i];pbest_f[i]=fi;}
            if(fi<gbest_f){gbest=pos[i];gbest_f=fi;}
        }
    }
    return gbest;
}

// ─── Nelder-Mead simplex ─────────────────────────────────────────────────────
template<int N, typename Func>
[[nodiscard]] constexpr std::array<f64,N> nelder_mead(
    Func&& f, std::array<f64,N> x0, f64 step=0.1,
    f64 tol=1e-8, int max_iter=5000) noexcept
{
    constexpr int V=N+1;
    std::array<std::array<f64,N>,V> simplex{};
    std::array<f64,V> fvals{};
    simplex[0]=x0; fvals[0]=f(x0);
    for(int i=1;i<V;++i){
        simplex[i]=x0; simplex[i][i-1]+=step; fvals[i]=f(simplex[i]);
    }

    for(int iter=0;iter<max_iter;++iter) {
        // Sort
        for(int i=0;i<V-1;++i) for(int j=i+1;j<V;++j)
            if(fvals[j]<fvals[i]){std::swap(simplex[i],simplex[j]);std::swap(fvals[i],fvals[j]);}

        // Check convergence
        f64 spread=0;for(int i=1;i<V;++i)spread+=std::abs(fvals[i]-fvals[0]);
        if(spread<tol) break;

        // Centroid (exclude worst)
        std::array<f64,N> cen{};
        for(int i=0;i<N;++i) for(int d=0;d<N;++d) cen[d]+=simplex[i][d]/N;

        // Reflect
        std::array<f64,N> xr{};
        for(int d=0;d<N;++d) xr[d]=cen[d]+1.0*(cen[d]-simplex[N][d]);
        f64 fr=f(xr);

        if(fr<fvals[0]) {
            // Expand
            std::array<f64,N> xe{};
            for(int d=0;d<N;++d) xe[d]=cen[d]+2.0*(xr[d]-cen[d]);
            f64 fe=f(xe);
            simplex[N]=(fe<fr)?xe:xr; fvals[N]=(fe<fr)?fe:fr;
        } else if(fr<fvals[N-1]) {
            simplex[N]=xr; fvals[N]=fr;
        } else {
            // Contract
            std::array<f64,N> xc{};
            for(int d=0;d<N;++d) xc[d]=cen[d]+0.5*(simplex[N][d]-cen[d]);
            f64 fc=f(xc);
            if(fc<fvals[N]){simplex[N]=xc;fvals[N]=fc;}
            else { // Shrink
                for(int i=1;i<V;++i){
                    for(int d=0;d<N;++d) simplex[i][d]=simplex[0][d]+0.5*(simplex[i][d]-simplex[0][d]);
                    fvals[i]=f(simplex[i]);
                }
            }
        }
    }
    return simplex[0];
}
} // namespace qs::math
