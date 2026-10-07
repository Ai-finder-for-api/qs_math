#pragma once
// qs_math/monte_carlo.hpp — Variance reduction: IS, CV, AV, QMC, MCMC (C++20)
#include "scalar_types.hpp"
#include "random.hpp"
#include "distributions.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <numeric>
#include <cassert>

namespace qs {

// ─── Basic Monte Carlo integrator ────────────────────────────────────────────
struct MCResult {
    f64 estimate{0};
    f64 std_error{0};
    f64 variance{0};
    u32 samples{0};
};

[[nodiscard]] inline MCResult mc_integrate(
    const std::function<f64(f64)>& f, f64 a, f64 b, u32 N, PCG32& rng)
{
    f64 h=b-a, sum=0, sum2=0;
    for(u32 i=0;i<N;++i) {
        f64 x=a+h*rng.next_float();
        f64 v=h*f(x);
        sum+=v; sum2+=v*v;
    }
    f64 est=sum/N;
    f64 var=sum2/N-est*est;
    return {est, std::sqrt(var/N), var, N};
}

// ─── Antithetic variates: use f(u) and f(1-u) to reduce variance ─────────────
[[nodiscard]] inline MCResult mc_antithetic(
    const std::function<f64(f64)>& f, f64 a, f64 b, u32 N, PCG32& rng)
{
    f64 h=b-a; N=N/2*2; // ensure even
    f64 sum=0,sum2=0;
    for(u32 i=0;i<N/2;++i) {
        f64 u=rng.next_float();
        f64 v1=h*f(a+h*u);
        f64 v2=h*f(a+h*(1-u));
        f64 v=0.5*(v1+v2);
        sum+=v; sum2+=v*v;
    }
    f64 est=sum/(N/2);
    f64 var=sum2/(N/2)-est*est;
    return {est, std::sqrt(var/(N/2)), var, N};
}

// ─── Control variates: use a known-mean auxiliary function to reduce variance ─
// E[f] = E[f - c(g - E[g])] + c*E[g]
// Optimal c = Cov(f,g)/Var(g)
[[nodiscard]] inline MCResult mc_control_variate(
    const std::function<f64(f64)>& f,
    const std::function<f64(f64)>& g,
    f64 E_g, // known mean of g
    f64 a, f64 b, u32 N, PCG32& rng)
{
    f64 h=b-a;
    std::vector<f64> fv(N),gv(N);
    for(u32 i=0;i<N;++i){ f64 x=a+h*rng.next_float(); fv[i]=h*f(x); gv[i]=h*g(x); }
    f64 mf=0,mg=0; for(u32 i=0;i<N;++i){mf+=fv[i];mg+=gv[i];} mf/=N;mg/=N;
    f64 cov=0,vg=0;
    for(u32 i=0;i<N;++i){cov+=(fv[i]-mf)*(gv[i]-mg);vg+=(gv[i]-mg)*(gv[i]-mg);}
    f64 c=vg>1e-30?cov/vg:0;
    f64 sum=0,sum2=0;
    for(u32 i=0;i<N;++i){ f64 v=fv[i]-c*(gv[i]-h*E_g); sum+=v; sum2+=v*v; }
    f64 est=sum/N; f64 var=sum2/N-est*est;
    return {est,std::sqrt(var/N),var,N};
}

// ─── Importance sampling: sample from q, reweight to p ───────────────────────
[[nodiscard]] inline MCResult mc_importance_sampling(
    const std::function<f64(f64)>& integrand, // f(x)
    const std::function<f64(f64)>& p,          // target density
    const std::function<f64(f64)>& q,          // proposal density
    const std::function<f64(PCG32&)>& sampler, // samples from q
    u32 N, PCG32& rng)
{
    f64 sum=0,sum2=0;
    for(u32 i=0;i<N;++i) {
        f64 x=sampler(rng);
        f64 qx=q(x);
        if(qx<1e-300) continue;
        f64 w=p(x)/qx;
        f64 v=integrand(x)*w;
        sum+=v; sum2+=v*v;
    }
    f64 est=sum/N; f64 var=sum2/N-est*est;
    return {est,std::sqrt(var/N),var,N};
}

// Self-normalised IS (when p is known only up to constant)
[[nodiscard]] inline f64 self_normalised_is(
    const std::function<f64(f64)>& f,
    const std::function<f64(f64)>& log_p_unnorm,
    const std::function<f64(f64)>& log_q,
    const std::function<f64(PCG32&)>& sampler,
    u32 N, PCG32& rng)
{
    std::vector<f64> xs(N);
    std::vector<f64> log_w(N);
    for(u32 i=0;i<N;++i){ xs[i]=sampler(rng); log_w[i]=log_p_unnorm(xs[i])-log_q(xs[i]); }
    f64 lmax=*std::max_element(log_w.begin(),log_w.end());
    f64 sum_w=0,sum_fw=0;
    for(u32 i=0;i<N;++i){ f64 w=std::exp(log_w[i]-lmax); sum_w+=w; sum_fw+=w*f(xs[i]); }
    return sum_fw/sum_w;
}

// ─── Stratified sampling ──────────────────────────────────────────────────────
[[nodiscard]] inline MCResult mc_stratified(
    const std::function<f64(f64)>& f, f64 a, f64 b, u32 N, u32 strata, PCG32& rng)
{
    u32 per_stratum=N/strata; f64 h=b-a; f64 dh=h/strata;
    f64 sum=0,sum2=0; u32 total=0;
    for(u32 s=0;s<strata;++s) {
        f64 sa=a+s*dh, sb=sa+dh;
        for(u32 i=0;i<per_stratum;++i) {
            f64 x=sa+(sb-sa)*rng.next_float();
            f64 v=h*f(x)/strata;
            sum+=v; sum2+=v*v; ++total;
        }
    }
    f64 est=sum/total; f64 var=sum2/total-est*est;
    return {est,std::sqrt(var/total),var,total};
}

// ─── Quasi-Monte Carlo (Halton sequence) ──────────────────────────────────────
[[nodiscard]] inline f64 halton(u32 index, u32 base) {
    f64 f=1, r=0;
    u32 i=index;
    while(i>0){ f/=base; r+=f*(i%base); i/=base; }
    return r;
}

[[nodiscard]] inline MCResult mc_qmc_halton(
    const std::function<f64(f64)>& f, f64 a, f64 b, u32 N)
{
    f64 h=b-a, sum=0, sum2=0;
    for(u32 i=1;i<=N;++i) {
        f64 x=a+h*halton(i,2);
        f64 v=h*f(x);
        sum+=v; sum2+=v*v;
    }
    f64 est=sum/N; f64 var=sum2/N-est*est;
    return {est,std::sqrt(var/N),var,N};
}

// ─── MCMC: Metropolis-Hastings ────────────────────────────────────────────────
struct MHChain {
    f64 current;
    f64 proposal_std{1.0};
    u32 accepted{0}, total{0};
    std::function<f64(f64)> log_target;

    MHChain(f64 init, f64 prop_std, std::function<f64(f64)> lt)
        : current(init), proposal_std(prop_std), log_target(std::move(lt)) {}

    f64 step(PCG32& rng) {
        f64 proposal=current+rng.next_gaussian()*proposal_std;
        f64 log_alpha=log_target(proposal)-log_target(current);
        f64 log_u=std::log(rng.next_float()+1e-300);
        ++total;
        if(log_u<log_alpha){ current=proposal; ++accepted; }
        return current;
    }

    std::vector<f64> sample(u32 N, u32 burn_in, PCG32& rng) {
        for(u32 i=0;i<burn_in;++i) step(rng);
        std::vector<f64> samples(N);
        for(u32 i=0;i<N;++i) samples[i]=step(rng);
        return samples;
    }

    [[nodiscard]] f64 acceptance_rate() const { return total>0?f64(accepted)/total:0; }
};

// ─── MCMC: Gibbs sampler (2D example structure) ──────────────────────────────
struct GibbsSampler2D {
    // Sample x|y and y|x using full conditionals
    std::function<f64(f64,f64,PCG32&)> sample_x_given_y;
    std::function<f64(f64,f64,PCG32&)> sample_y_given_x;
    f64 x{0}, y{0};

    std::vector<std::pair<f64,f64>> sample(u32 N, u32 burn_in, PCG32& rng) {
        for(u32 i=0;i<burn_in;++i){ x=sample_x_given_y(x,y,rng); y=sample_y_given_x(x,y,rng); }
        std::vector<std::pair<f64,f64>> out(N);
        for(u32 i=0;i<N;++i){ x=sample_x_given_y(x,y,rng); y=sample_y_given_x(x,y,rng); out[i]={x,y}; }
        return out;
    }
};

// ─── Convergence diagnostics ─────────────────────────────────────────────────

// Gelman-Rubin R̂ statistic (should be <1.1 for convergence)
[[nodiscard]] inline f64 gelman_rubin(const std::vector<std::vector<f64>>& chains) {
    u32 m=u32(chains.size()),n=u32(chains[0].size());
    std::vector<f64> chain_means(m);
    for(u32 j=0;j<m;++j){ for(auto v:chains[j]) chain_means[j]+=v; chain_means[j]/=n; }
    f64 grand_mean=0; for(auto v:chain_means) grand_mean+=v; grand_mean/=m;
    // Between-chain variance B
    f64 B=0; for(auto v:chain_means) B+=(v-grand_mean)*(v-grand_mean); B*=n/(m-1.0);
    // Within-chain variance W
    f64 W=0;
    for(u32 j=0;j<m;++j){ f64 s=0; for(auto v:chains[j]) s+=(v-chain_means[j])*(v-chain_means[j]); W+=s/(n-1); }
    W/=m;
    f64 var_est=(1-1.0/n)*W+B/n;
    return std::sqrt(var_est/(W+1e-30));
}

// Effective sample size for MCMC (autocorrelation-based)
[[nodiscard]] inline f64 ess_mcmc(const std::vector<f64>& chain) {
    u32 n=u32(chain.size());
    f64 mean=0; for(auto v:chain) mean+=v; mean/=n;
    f64 var=0; for(auto v:chain) var+=(v-mean)*(v-mean); var/=n;
    if(var<1e-30) return f64(n);
    // Compute autocorrelations until they become negative
    f64 rho_sum=1;
    for(u32 lag=1;lag<n/2;++lag) {
        f64 rho=0;
        for(u32 i=0;i<n-lag;++i) rho+=(chain[i]-mean)*(chain[i+lag]-mean);
        rho/=(n-lag)*var;
        if(rho<0) break;
        rho_sum+=2*rho;
    }
    return f64(n)/rho_sum;
}

} // namespace qs
