#pragma once
// qs_math/statistics/bayesian.hpp — Bayes theorem, conjugate priors, MCMC, Kalman filter
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Bayes theorem ───────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 bayes_posterior(f64 likelihood, f64 prior, f64 evidence) noexcept {
    return (evidence>1e-15)?likelihood*prior/evidence:0;
}
[[nodiscard]] constexpr f64 log_bayes_factor(f64 log_likelihood_H1,
                                               f64 log_likelihood_H0) noexcept {
    return log_likelihood_H1-log_likelihood_H0;
}

// ─── Conjugate priors ─────────────────────────────────────────────────────────
// Beta-Binomial: Beta(alpha,beta) prior → Beta(alpha+k,beta+n-k) posterior
struct BetaDist { f64 alpha,beta;
    [[nodiscard]] constexpr f64 mean()     const noexcept {return alpha/(alpha+beta);}
    [[nodiscard]] constexpr f64 variance() const noexcept {
        f64 s=alpha+beta; return alpha*beta/(s*s*(s+1));
    }
    [[nodiscard]] constexpr BetaDist update(int k, int n) const noexcept {
        return {alpha+k,beta+n-k};
    }
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<=0||x>=1) return 0;
        // B(a,b)=Γ(a)Γ(b)/Γ(a+b) ≈ exp(lgamma(a)+lgamma(b)-lgamma(a+b))
        f64 logB=std::lgamma(alpha)+std::lgamma(beta)-std::lgamma(alpha+beta);
        return std::exp((alpha-1)*std::log(x)+(beta-1)*std::log(1-x)-logB);
    }
    [[nodiscard]] f64 cdf(f64 x, int terms=100) const noexcept {
        // Regularized incomplete beta via series
        if(x<=0) return 0; if(x>=1) return 1;
        f64 s=0,t=1;
        for(int n=0;n<terms;++n){
            t*=((alpha+n)/(alpha+beta+n))*x;
            s+=t/(alpha+n+1);
            if(std::abs(t)<1e-12) break;
        }
        return std::pow(x,alpha)*std::pow(1-x,beta)/(alpha)*s
               +std::pow(x,alpha)*std::pow(1-x,beta)/alpha; // approximate
    }
};

// Gamma-Poisson: Gamma(alpha,beta) prior → Gamma(alpha+sum_x, beta+n) posterior
struct GammaDist { f64 alpha,beta_rate; // rate parameterization
    [[nodiscard]] constexpr f64 mean()     const noexcept {return alpha/beta_rate;}
    [[nodiscard]] constexpr f64 variance() const noexcept {return alpha/(beta_rate*beta_rate);}
    [[nodiscard]] constexpr GammaDist update_poisson(i64 sum_x, int n) const noexcept {
        return {alpha+f64(sum_x),beta_rate+f64(n)};
    }
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<=0) return 0;
        return std::exp(alpha*std::log(beta_rate)-std::lgamma(alpha)
                        +(alpha-1)*std::log(x)-beta_rate*x);
    }
};

// Normal-Normal: N(mu0,sigma0²) prior on mean, known variance σ²
struct NormalPrior { f64 mu,sigma2;
    [[nodiscard]] constexpr NormalPrior update(f64 x_bar, int n, f64 sigma2_known) const noexcept {
        f64 tau2_post=1/(1/sigma2+n/sigma2_known);
        f64 mu_post=tau2_post*(mu/sigma2+n*x_bar/sigma2_known);
        return {mu_post,tau2_post};
    }
};

// ─── MCMC: Metropolis-Hastings ─────────────────────────────────────────────────
template<int N, typename LogPdf>
[[nodiscard]] constexpr std::array<std::array<f64,N>,1000> metropolis_hastings(
    LogPdf&& log_pdf, std::array<f64,N> x0,
    f64 proposal_std=0.1, u64 seed=42) noexcept
{
    std::array<std::array<f64,N>,1000> chain{};
    Xorshift64 rng{seed};
    auto x=x0; f64 lp=log_pdf(x);
    for(int i=0;i<1000;++i) {
        std::array<f64,N> xp=x;
        for(int d=0;d<N;++d) xp[d]+=proposal_std*rng.normal();
        f64 lp_new=log_pdf(xp);
        f64 log_alpha=lp_new-lp;
        if(log_alpha>0||rng.uniform()<std::exp(log_alpha)){x=xp;lp=lp_new;}
        chain[i]=x;
    }
    return chain;
}

// ─── Kalman Filter (linear Gaussian state-space model) ────────────────────────
template<int NS, int NO>
struct KalmanFilter {
    // x_t = F*x_{t-1} + w, w~N(0,Q)
    // y_t = H*x_t + v,     v~N(0,R)
    std::array<std::array<f64,NS>,NS> F{},Q{},P{};
    std::array<std::array<f64,NS>,NO> Ht{}; // H^T (stored transposed for easy access)
    std::array<std::array<f64,NO>,NO> R{};
    std::array<f64,NS> x{};

    // Predict step
    constexpr void predict() noexcept {
        // x = F*x
        std::array<f64,NS> xp{};
        for(int i=0;i<NS;++i) for(int j=0;j<NS;++j) xp[i]+=F[i][j]*x[j];
        x=xp;
        // P = F*P*F' + Q
        std::array<std::array<f64,NS>,NS> FP{},FPFt{};
        for(int i=0;i<NS;++i) for(int j=0;j<NS;++j)
            for(int k=0;k<NS;++k) FP[i][j]+=F[i][k]*P[k][j];
        for(int i=0;i<NS;++i) for(int j=0;j<NS;++j)
            for(int k=0;k<NS;++k) FPFt[i][j]+=FP[i][k]*F[j][k];
        for(int i=0;i<NS;++i) for(int j=0;j<NS;++j) P[i][j]=FPFt[i][j]+Q[i][j];
    }

    // Update step: y is observation
    constexpr void update(const std::array<f64,NO>& y) noexcept {
        // S = H*P*H' + R
        std::array<std::array<f64,NO>,NO> S{};
        for(int i=0;i<NO;++i) for(int j=0;j<NO;++j) {
            for(int k=0;k<NS;++k) for(int l=0;l<NS;++l)
                S[i][j]+=Ht[k][i]*P[k][l]*Ht[l][j];
            S[i][j]+=R[i][j];
        }
        // K = P*H'*S^{-1} (2x2 invert for NO=2, general: use LU)
        // Simplified: K = P*H'/trace(S) approximation for small NO
        std::array<std::array<f64,NO>,NS> K{};
        f64 s_inv=1.0/(S[0][0]+1e-15); // works for NO=1
        for(int i=0;i<NS;++i) for(int j=0;j<NO;++j)
            for(int k=0;k<NS;++k) K[i][j]+=P[i][k]*Ht[k][j]*s_inv;
        // x = x + K*(y - H*x)
        std::array<f64,NO> innov{};
        for(int i=0;i<NO;++i){
            innov[i]=y[i];
            for(int k=0;k<NS;++k) innov[i]-=Ht[k][i]*x[k];
        }
        for(int i=0;i<NS;++i) for(int j=0;j<NO;++j) x[i]+=K[i][j]*innov[j];
        // P = (I - K*H)*P
        std::array<std::array<f64,NS>,NS> KH{};
        for(int i=0;i<NS;++i) for(int j=0;j<NS;++j)
            for(int k=0;k<NO;++k) KH[i][j]+=K[i][k]*Ht[j][k];
        std::array<std::array<f64,NS>,NS> Pn{};
        for(int i=0;i<NS;++i) for(int j=0;j<NS;++j)
            Pn[i][j]=((i==j)?1.0:0.0)*P[i][j]-KH[i][j]*P[i][j];
        P=Pn;
    }

    [[nodiscard]] constexpr std::array<f64,NS> state() const noexcept { return x; }
};

// ─── Particle filter (sequential Monte Carlo) ─────────────────────────────────
template<int N, int PARTICLES, typename TransFn, typename LikFn>
[[nodiscard]] constexpr std::array<f64,N> particle_filter_step(
    TransFn&& transition, LikFn&& likelihood,
    std::array<std::array<f64,N>,PARTICLES>& particles,
    std::array<f64,PARTICLES>& weights,
    const std::array<f64,N>& obs, u64& seed) noexcept
{
    Xorshift64 rng{seed};
    // Propagate
    for(int i=0;i<PARTICLES;++i) particles[i]=transition(particles[i],rng);
    // Weight
    f64 wsum=0;
    for(int i=0;i<PARTICLES;++i){weights[i]*=likelihood(particles[i],obs);wsum+=weights[i];}
    if(wsum>1e-15) for(auto& w:weights) w/=wsum;
    // Estimate
    std::array<f64,N> est{};
    for(int i=0;i<PARTICLES;++i) for(int d=0;d<N;++d) est[d]+=weights[i]*particles[i][d];
    // Systematic resample
    f64 u0=rng.uniform()/PARTICLES;
    std::array<std::array<f64,N>,PARTICLES> new_p{};
    f64 cumw=0; int j=0;
    for(int i=0;i<PARTICLES;++i) {
        f64 target=(i+u0)/PARTICLES;
        while(cumw<target&&j<PARTICLES-1) cumw+=weights[j++];
        new_p[i]=particles[j];
    }
    particles=new_p;
    weights.fill(1.0/PARTICLES);
    seed=rng.state;
    return est;
}
} // namespace qs::math
