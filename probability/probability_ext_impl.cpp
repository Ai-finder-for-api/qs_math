
// qs_math/probability/probability_ext_impl.cpp (C++20)
// Extended probability: MCMC, variational inference, graphical models.
#include "include/qs_math_probability.hpp"
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
namespace qs::math::prob {

// ─── Metropolis-Hastings MCMC ─────────────────────────────────────────────────
std::vector<std::vector<f64>> metropolis_hastings(
    std::function<f64(const std::vector<f64>&)> log_target,
    std::vector<f64> x0,
    i32 n_samples, f64 step_size=0.1, u64 seed=42) noexcept
{
    u64 rng = seed;
    auto randn = [&]() -> f64 {
        // Box-Muller
        rng = rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u1 = (rng>>11)*(1.0/9007199254740992.0);
        rng = rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u2 = (rng>>11)*(1.0/9007199254740992.0)+1e-14;
        return std::sqrt(-2*std::log(u2))*std::cos(6.28318530718*u1);
    };
    auto randu = [&]() -> f64 {
        rng = rng*6364136223846793005ULL+1442695040888963407ULL;
        return (rng>>11)*(1.0/9007199254740992.0);
    };

    std::vector<std::vector<f64>> samples;
    samples.reserve(n_samples);
    std::vector<f64> x = x0;
    f64 log_px = log_target(x);

    for (i32 i = 0; i < n_samples; ++i) {
        // Propose new sample
        std::vector<f64> x_prop = x;
        for (auto& v : x_prop) v += randn() * step_size;
        const f64 log_px_prop = log_target(x_prop);

        // Accept/reject (log space)
        if (std::log(randu()) < log_px_prop - log_px) {
            x = x_prop;
            log_px = log_px_prop;
        }
        samples.push_back(x);
    }
    return samples;
}

// ─── Hamiltonian Monte Carlo ───────────────────────────────────────────────────
std::vector<std::vector<f64>> hmc(
    std::function<f64(const std::vector<f64>&)> log_target,
    std::function<std::vector<f64>(const std::vector<f64>&)> grad_log_target,
    std::vector<f64> x0,
    i32 n_samples, f64 step_size=0.01, i32 n_leapfrog=10, u64 seed=1) noexcept
{
    u64 rng = seed;
    auto randn = [&]() -> f64 {
        rng = rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u1 = std::max(1e-14, (rng>>11)*(1.0/9007199254740992.0));
        rng = rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u2 = (rng>>11)*(1.0/9007199254740992.0);
        return std::sqrt(-2*std::log(u1))*std::cos(6.28318530718*u2);
    };
    auto randu = [&]() -> f64 {
        rng = rng*6364136223846793005ULL+1442695040888963407ULL;
        return (rng>>11)*(1.0/9007199254740992.0);
    };

    const i32 dim = (i32)x0.size();
    std::vector<std::vector<f64>> samples;
    samples.reserve(n_samples);
    std::vector<f64> q = x0;

    for (i32 s = 0; s < n_samples; ++s) {
        // Sample momentum
        std::vector<f64> p(dim);
        for (auto& pi : p) pi = randn();

        // Current H = -log_target(q) + 0.5 * p^2
        f64 H_curr = -log_target(q);
        for (auto pi : p) H_curr += 0.5*pi*pi;

        // Leapfrog
        std::vector<f64> q_new = q, p_new = p;
        auto g = grad_log_target(q_new);
        // Half step for p
        for (i32 i = 0; i < dim; ++i) p_new[i] += 0.5*step_size*g[i];
        // Full steps
        for (i32 l = 0; l < n_leapfrog; ++l) {
            for (i32 i = 0; i < dim; ++i) q_new[i] += step_size * p_new[i];
            g = grad_log_target(q_new);
            const bool last = (l == n_leapfrog-1);
            for (i32 i = 0; i < dim; ++i) p_new[i] += (last?0.5:1)*step_size*g[i];
        }
        // Accept/reject
        f64 H_prop = -log_target(q_new);
        for (auto pi : p_new) H_prop += 0.5*pi*pi;

        if (std::log(randu()) < H_curr - H_prop) q = q_new;
        samples.push_back(q);
    }
    return samples;
}

// ─── Variational inference (mean-field Gaussian) ─────────────────────────────
struct MeanFieldGaussian {
    std::vector<f64> mu, log_sigma;

    void fit(std::function<f64(const std::vector<f64>&)> log_joint,
             i32 n_iter=1000, f64 lr=0.01, u64 seed=99) noexcept
    {
        const i32 dim = (i32)mu.size();
        u64 rng = seed;
        auto randn = [&]() -> f64 {
            rng = rng*6364136223846793005ULL+1442695040888963407ULL;
            const f64 u1 = std::max(1e-14,(rng>>11)*(1.0/9007199254740992.0));
            rng = rng*6364136223846793005ULL+1442695040888963407ULL;
            const f64 u2 = (rng>>11)*(1.0/9007199254740992.0);
            return std::sqrt(-2*std::log(u1))*std::cos(6.28318530718*u2);
        };

        const i32 MC = 10; // MC gradient samples
        for (i32 iter = 0; iter < n_iter; ++iter) {
            std::vector<f64> grad_mu(dim,0), grad_ls(dim,0);
            for (i32 m = 0; m < MC; ++m) {
                std::vector<f64> eps(dim), z(dim);
                for (i32 i = 0; i < dim; ++i) {
                    eps[i] = randn();
                    z[i] = mu[i] + std::exp(log_sigma[i]) * eps[i];
                }
                const f64 lj = log_joint(z);
                const f64 h = 1e-5;
                for (i32 i = 0; i < dim; ++i) {
                    // Numerical gradient w.r.t. mu[i]
                    mu[i] += h;
                    const f64 lj_p = log_joint(z); // z unchanged, but reparameterisation
                    mu[i] -= h;
                    grad_mu[i] += (lj_p - lj) / (h * MC);
                    grad_ls[i] += lj * eps[i] * std::exp(log_sigma[i]) / MC;
                }
                (void)lj;
            }
            // Entropy gradient for log_sigma: +1 each
            for (i32 i = 0; i < dim; ++i) {
                mu[i]         += lr * grad_mu[i];
                log_sigma[i]  += lr * (grad_ls[i] + 1.0);
            }
        }
    }

    [[nodiscard]] f64 elbo_sample(std::function<f64(const std::vector<f64>&)> log_joint,
        u64 seed=77) const noexcept
    {
        const i32 dim = (i32)mu.size();
        u64 rng = seed;
        auto randn = [&]() -> f64 {
            rng=rng*6364136223846793005ULL+1442695040888963407ULL;
            const f64 u1=std::max(1e-14,(rng>>11)*(1.0/9007199254740992.0));
            rng=rng*6364136223846793005ULL+1442695040888963407ULL;
            const f64 u2=(rng>>11)*(1.0/9007199254740992.0);
            return std::sqrt(-2*std::log(u1))*std::cos(6.28318530718*u2);
        };
        std::vector<f64> z(dim);
        for (i32 i=0;i<dim;++i) z[i]=mu[i]+std::exp(log_sigma[i])*randn();
        f64 entropy=0; for(auto ls:log_sigma) entropy+=ls+0.5*std::log(2*M_E*M_PI);
        return log_joint(z)+entropy;
    }
};

// ─── Belief propagation (sum-product, factor graph) ──────────────────────────
struct DiscreteFactorGraph {
    i32 n_vars{0}, n_states{0};
    std::vector<std::vector<f64>> unary;   // n_vars × n_states
    std::vector<std::tuple<i32,i32,std::vector<f64>>> pairwise; // var_i,var_j,n_states×n_states

    void add_unary(i32 var, const std::vector<f64>& log_probs) {
        if ((i32)unary.size() <= var) unary.resize(var+1, std::vector<f64>(n_states,0));
        unary[var] = log_probs;
    }
    void add_pairwise(i32 i, i32 j, const std::vector<f64>& log_potentials) {
        pairwise.emplace_back(i, j, log_potentials);
    }

    // Loopy belief propagation — returns marginal log-probabilities per variable
    std::vector<std::vector<f64>> run_lbp(i32 max_iter=50) const noexcept {
        const i32 n = n_vars;
        // Messages: msg[src*n+dst][state]
        std::vector<std::vector<f64>> msgs(n*n, std::vector<f64>(n_states, 0));
        std::vector<std::vector<f64>> beliefs(n, std::vector<f64>(n_states, 0));

        for (i32 iter = 0; iter < max_iter; ++iter) {
            for (auto& [vi, vj, lp] : pairwise) {
                for (i32 pass = 0; pass < 2; ++pass) {
                    const i32 src = pass?vj:vi, dst = pass?vi:vj;
                    // msg[src→dst][s_dst] = log sum_{s_src} ( lp[s_src,s_dst] + unary[src][s_src]
                    //                              + sum_{nb\dst} msg[nb→src][s_src] )
                    auto& msg = msgs[src*n+dst];
                    std::fill(msg.begin(),msg.end(),-1e18);
                    for (i32 sd=0;sd<n_states;++sd)
                        for (i32 ss=0;ss<n_states;++ss){
                            f64 v=(pass?lp[ss*n_states+sd]:lp[sd*n_states+ss]);
                            if((i32)unary.size()>src)v+=unary[src][ss];
                            // Sum incoming messages to src except from dst
                            for(auto& [a,b,lp2]:pairwise){
                                (void)lp2;
                                i32 nb=-1;
                                if(a==src&&b!=dst)nb=b;
                                else if(b==src&&a!=dst)nb=a;
                                if(nb>=0)v+=msgs[nb*n+src][ss];
                            }
                            msg[sd]=std::max(msg[sd],v); // max-product (approximation)
                        }
                }
            }
        }
        // Compute beliefs
        for (i32 v=0;v<n;++v){
            std::fill(beliefs[v].begin(),beliefs[v].end(),0);
            if((i32)unary.size()>v)beliefs[v]=unary[v];
            for(auto& [a,b,lp]:pairwise){
                i32 nb=-1;
                if(a==v)nb=b; else if(b==v)nb=a;
                if(nb>=0)for(i32 s=0;s<n_states;++s)beliefs[v][s]+=msgs[nb*n+v][s];
            }
            // Log-normalise
            const f64 mx=*std::max_element(beliefs[v].begin(),beliefs[v].end());
            f64 Z=0;for(auto& b2:beliefs[v])Z+=std::exp(b2-mx);
            for(auto& b2:beliefs[v])b2=std::exp(b2-mx)/Z;
        }
        return beliefs;
    }
};

} // namespace qs::math::prob
