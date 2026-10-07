#pragma once
// qs_math/evolutionary.hpp — Genetic algorithm, CMA-ES, differential evolution (C++20)
#include "scalar_types.hpp"
#include "random.hpp"
#include "optimization.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>

namespace qs {

using FitnessFunc = std::function<f64(const std::vector<f64>&)>;

// ─── Genetic Algorithm (real-valued) ─────────────────────────────────────────
struct GAConfig {
    u32  pop_size{100};
    u32  max_gen{500};
    f64  crossover_rate{0.9};
    f64  mutation_rate{0.01};
    f64  mutation_std{0.1};
    f64  tol{1e-8};
    bool elitism{true};
};

struct GAResult {
    std::vector<f64> best_x;
    f64 best_f{0};
    u32 generations{0};
    std::vector<f64> history; // best fitness per generation
};

[[nodiscard]] inline GAResult genetic_algorithm(
    const FitnessFunc& f, // minimise
    u32 dim, f64 lo, f64 hi,
    const GAConfig& cfg={}, PCG32* rng_ptr=nullptr)
{
    PCG32 local_rng; if(!rng_ptr) rng_ptr=&local_rng;
    PCG32& rng=*rng_ptr;
    u32 N=cfg.pop_size;
    // Initialise population
    std::vector<std::vector<f64>> pop(N, std::vector<f64>(dim));
    std::vector<f64> fit(N);
    for(u32 i=0;i<N;++i){
        for(u32 j=0;j<dim;++j) pop[i][j]=lo+(hi-lo)*rng.next_float();
        fit[i]=f(pop[i]);
    }
    GAResult res; res.best_x=pop[0]; res.best_f=fit[0];
    for(u32 gen=0;gen<cfg.max_gen;++gen) {
        // Tournament selection
        auto select=[&]()->u32{
            u32 a=rng.next_u32()%N, b=rng.next_u32()%N;
            return fit[a]<fit[b]?a:b;
        };
        std::vector<std::vector<f64>> new_pop(N);
        u32 start=0;
        if(cfg.elitism){ // keep best
            u32 best_idx=u32(std::min_element(fit.begin(),fit.end())-fit.begin());
            new_pop[0]=pop[best_idx]; start=1;
        }
        for(u32 i=start;i<N;++i) {
            u32 p1=select(), p2=select();
            new_pop[i].resize(dim);
            // BLX-α crossover
            if(rng.next_float()<cfg.crossover_rate) {
                for(u32 j=0;j<dim;++j) {
                    f64 alpha=0.5;
                    f64 mn=std::min(pop[p1][j],pop[p2][j])-alpha*std::abs(pop[p1][j]-pop[p2][j]);
                    f64 mx=std::max(pop[p1][j],pop[p2][j])+alpha*std::abs(pop[p1][j]-pop[p2][j]);
                    new_pop[i][j]=mn+(mx-mn)*rng.next_float();
                }
            } else { new_pop[i]=pop[p1]; }
            // Gaussian mutation
            for(u32 j=0;j<dim;++j)
                if(rng.next_float()<cfg.mutation_rate)
                    new_pop[i][j]+=rng.next_gaussian()*cfg.mutation_std*(hi-lo);
            // Clamp
            for(u32 j=0;j<dim;++j) new_pop[i][j]=std::clamp(new_pop[i][j],lo,hi);
        }
        pop=new_pop;
        for(u32 i=0;i<N;++i){ fit[i]=f(pop[i]); if(fit[i]<res.best_f){res.best_f=fit[i];res.best_x=pop[i];} }
        res.history.push_back(res.best_f);
        res.generations=gen+1;
        if(res.best_f<cfg.tol) break;
    }
    return res;
}

// ─── Differential Evolution (DE/rand/1/bin) ──────────────────────────────────
struct DEConfig {
    u32 pop_size{50};
    u32 max_gen{1000};
    f64 F{0.8};   // differential weight
    f64 CR{0.9};  // crossover probability
    f64 tol{1e-8};
    // Strategy: 0=rand/1, 1=best/1, 2=rand/2
    u32 strategy{0};
};

[[nodiscard]] inline GAResult differential_evolution(
    const FitnessFunc& f, u32 dim, f64 lo, f64 hi,
    const DEConfig& cfg={}, PCG32* rng_ptr=nullptr)
{
    PCG32 local_rng; if(!rng_ptr) rng_ptr=&local_rng;
    PCG32& rng=*rng_ptr;
    u32 N=cfg.pop_size;
    std::vector<std::vector<f64>> pop(N,std::vector<f64>(dim));
    std::vector<f64> fit(N);
    for(u32 i=0;i<N;++i){ for(u32 j=0;j<dim;++j) pop[i][j]=lo+(hi-lo)*rng.next_float(); fit[i]=f(pop[i]); }
    GAResult res;
    res.best_f=*std::min_element(fit.begin(),fit.end());
    res.best_x=pop[u32(std::min_element(fit.begin(),fit.end())-fit.begin())];
    for(u32 gen=0;gen<cfg.max_gen;++gen) {
        u32 best_idx=u32(std::min_element(fit.begin(),fit.end())-fit.begin());
        for(u32 i=0;i<N;++i) {
            // Pick 3 random distinct indices
            u32 a,b,c;
            do{a=rng.next_u32()%N;}while(a==i);
            do{b=rng.next_u32()%N;}while(b==i||b==a);
            do{c=rng.next_u32()%N;}while(c==i||c==a||c==b);
            std::vector<f64> base=cfg.strategy==1?pop[best_idx]:pop[a];
            std::vector<f64> trial(dim);
            u32 jrand=rng.next_u32()%dim;
            for(u32 j=0;j<dim;++j) {
                if(rng.next_float()<cfg.CR||j==jrand) trial[j]=std::clamp(base[j]+cfg.F*(pop[b][j]-pop[c][j]),lo,hi);
                else trial[j]=pop[i][j];
            }
            if(cfg.strategy==2){ // rand/2: add second difference vector
                u32 d,e;
                do{d=rng.next_u32()%N;}while(d==i||d==a||d==b||d==c);
                do{e=rng.next_u32()%N;}while(e==i||e==a||e==b||e==c||e==d);
                for(u32 j=0;j<dim;++j) trial[j]=std::clamp(trial[j]+cfg.F*(pop[d][j]-pop[e][j]),lo,hi);
            }
            f64 ft=f(trial);
            if(ft<=fit[i]){ pop[i]=trial; fit[i]=ft; if(ft<res.best_f){res.best_f=ft;res.best_x=trial;} }
        }
        res.history.push_back(res.best_f);
        res.generations=gen+1;
        if(res.best_f<cfg.tol) break;
    }
    return res;
}

// ─── CMA-ES (Covariance Matrix Adaptation Evolution Strategy) ─────────────────
struct CMAESConfig {
    u32  max_iter{10000};
    f64  tol{1e-10};
    f64  sigma0{0.3}; // initial step size
};

[[nodiscard]] inline GAResult cmaes(
    const FitnessFunc& f, std::vector<f64> x0,
    const CMAESConfig& cfg={}, PCG32* rng_ptr=nullptr)
{
    PCG32 local_rng; if(!rng_ptr) rng_ptr=&local_rng;
    PCG32& rng=*rng_ptr;
    u32 n=u32(x0.size());
    u32 lam=4+u32(3*std::log(n));  // population size
    u32 mu=lam/2;                  // number of parents
    f64 sigma=cfg.sigma0;
    // Weights
    std::vector<f64> w(mu);
    for(u32 i=0;i<mu;++i) w[i]=std::log(mu+0.5)-std::log(f64(i+1));
    f64 wsum=0; for(auto v:w) wsum+=v; for(auto& v:w) v/=wsum;
    f64 mueff=0; { f64 s1=0,s2=0; for(auto v:w){s1+=v;s2+=v*v;} mueff=s1*s1/s2; }
    // Adaptation constants
    f64 cs=(mueff+2)/(n+mueff+5);
    f64 ds=1+cs+2*std::max(0.0,std::sqrt((mueff-1)/(n+1))-1);
    f64 cc=(4+mueff/n)/(n+4+2*mueff/n);
    f64 c1=2/((n+1.3)*(n+1.3)+mueff);
    f64 cmu=std::min(1-c1, 2*(mueff-2+1/mueff)/((n+2)*(n+2)+mueff));
    f64 chiN=std::sqrt(f64(n))*(1-1.0/(4*n)+1.0/(21*n*n));
    // State
    std::vector<f64> xmean=x0;
    std::vector<f64> ps(n,0), pc(n,0);
    std::vector<std::vector<f64>> C(n,std::vector<f64>(n,0)); for(u32 i=0;i<n;++i) C[i][i]=1;
    std::vector<std::vector<f64>> B(n,std::vector<f64>(n,0)); for(u32 i=0;i<n;++i) B[i][i]=1;
    std::vector<f64> D(n,1);
    GAResult res; res.best_x=x0; res.best_f=f(x0);
    for(u32 it=0;it<cfg.max_iter;++it) {
        // Sample population
        std::vector<std::vector<f64>> arz(lam,std::vector<f64>(n));
        std::vector<std::vector<f64>> arx(lam,std::vector<f64>(n));
        std::vector<f64> arfitness(lam);
        for(u32 k=0;k<lam;++k) {
            for(u32 i=0;i<n;++i) arz[k][i]=rng.next_gaussian();
            // x = mean + sigma * B * D * z
            for(u32 i=0;i<n;++i) {
                f64 s=0; for(u32 j=0;j<n;++j) s+=B[i][j]*D[j]*arz[k][j];
                arx[k][i]=xmean[i]+sigma*s;
            }
            arfitness[k]=f(arx[k]);
            if(arfitness[k]<res.best_f){res.best_f=arfitness[k];res.best_x=arx[k];}
        }
        // Sort by fitness
        std::vector<u32> idx(lam); std::iota(idx.begin(),idx.end(),0);
        std::sort(idx.begin(),idx.end(),[&](u32 a,u32 b){return arfitness[a]<arfitness[b];});
        // Update mean
        std::vector<f64> xold=xmean;
        for(u32 i=0;i<n;++i){ xmean[i]=0; for(u32 k=0;k<mu;++k) xmean[i]+=w[k]*arx[idx[k]][i]; }
        // Cumulation for step-size control
        for(u32 i=0;i<n;++i) {
            f64 inv_sigma_Cps=0;
            for(u32 j=0;j<n;++j) inv_sigma_Cps+=B[j][i]*(xmean[j]-xold[j])/(sigma*D[i]);
            ps[i]=(1-cs)*ps[i]+std::sqrt(cs*(2-cs)*mueff)*inv_sigma_Cps;
        }
        f64 ps_norm=0; for(auto v:ps) ps_norm+=v*v; ps_norm=std::sqrt(ps_norm);
        f64 hsig=ps_norm/std::sqrt(1-std::pow(1-cs,2*(it+1)))/chiN < 1.4+2/(n+1)?1:0;
        // Cumulation for covariance
        for(u32 i=0;i<n;++i)
            pc[i]=(1-cc)*pc[i]+hsig*std::sqrt(cc*(2-cc)*mueff)*(xmean[i]-xold[i])/sigma;
        // Adapt covariance
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) {
            C[i][j]=(1-c1-cmu)*C[i][j]+c1*(pc[i]*pc[j]+(1-hsig)*cc*(2-cc)*C[i][j]);
            for(u32 k=0;k<mu;++k)
                C[i][j]+=cmu*w[k]*(arx[idx[k]][i]-xold[i])*(arx[idx[k]][j]-xold[j])/(sigma*sigma);
        }
        // Adapt step size
        sigma*=std::exp((cs/ds)*(ps_norm/chiN-1));
        // Eigendecomposition of C every n/10+1 iterations
        if(it%(n/10+1)==0) {
            auto eig=jacobi_eigen(DenseMatrix<f64>(n,n)); // simplified: use diagonal
            for(u32 i=0;i<n;++i) D[i]=std::sqrt(std::max(1e-20,C[i][i]));
        }
        res.history.push_back(res.best_f);
        res.generations=it+1;
        if(sigma<cfg.tol||res.best_f<cfg.tol) break;
    }
    return res;
}

// ─── Particle Swarm Optimization (PSO) ───────────────────────────────────────
struct PSOConfig { u32 pop{50},max_iter{500}; f64 w{0.7},c1{1.5},c2{1.5},tol{1e-8}; };

[[nodiscard]] inline GAResult pso(const FitnessFunc& f, u32 dim, f64 lo, f64 hi,
    const PSOConfig& cfg={}, PCG32* rng_ptr=nullptr)
{
    PCG32 local_rng; if(!rng_ptr) rng_ptr=&local_rng;
    PCG32& rng=*rng_ptr;
    u32 N=cfg.pop;
    std::vector<std::vector<f64>> pos(N,std::vector<f64>(dim));
    std::vector<std::vector<f64>> vel(N,std::vector<f64>(dim,0));
    std::vector<std::vector<f64>> pbest(N);
    std::vector<f64> pbest_f(N,std::numeric_limits<f64>::max());
    std::vector<f64> gbest; f64 gbest_f=std::numeric_limits<f64>::max();
    for(u32 i=0;i<N;++i){
        for(u32 j=0;j<dim;++j){ pos[i][j]=lo+(hi-lo)*rng.next_float(); vel[i][j]=(hi-lo)*0.1*(rng.next_float()-0.5); }
        f64 fv=f(pos[i]); pbest[i]=pos[i]; pbest_f[i]=fv;
        if(fv<gbest_f){gbest_f=fv;gbest=pos[i];}
    }
    GAResult res; res.best_x=gbest; res.best_f=gbest_f;
    for(u32 it=0;it<cfg.max_iter;++it) {
        for(u32 i=0;i<N;++i){
            for(u32 j=0;j<dim;++j){
                f64 r1=rng.next_float(),r2=rng.next_float();
                vel[i][j]=cfg.w*vel[i][j]+cfg.c1*r1*(pbest[i][j]-pos[i][j])+cfg.c2*r2*(gbest[j]-pos[i][j]);
                pos[i][j]=std::clamp(pos[i][j]+vel[i][j],lo,hi);
            }
            f64 fv=f(pos[i]);
            if(fv<pbest_f[i]){pbest_f[i]=fv;pbest[i]=pos[i];}
            if(fv<gbest_f){gbest_f=fv;gbest=pos[i];res.best_f=fv;res.best_x=gbest;}
        }
        res.history.push_back(res.best_f);
        res.generations=it+1;
        if(res.best_f<cfg.tol) break;
    }
    return res;
}

} // namespace qs
