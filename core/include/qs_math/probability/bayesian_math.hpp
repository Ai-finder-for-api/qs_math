#pragma once
// qs_math/bayesian_math.hpp — Prior/posterior, conjugate pairs, variational Bayes (C++20)
#include "scalar_types.hpp"
#include "distributions.hpp"
#include "probability.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <cassert>

namespace qs {

// ─── Conjugate prior-posterior pairs ─────────────────────────────────────────

// Beta-Binomial: Beta(α,β) prior on p, Binomial(n,p) likelihood
// Posterior: Beta(α + successes, β + failures)
struct BetaBinomialModel {
    f64 alpha{1}, beta_param{1}; // prior parameters (Beta distribution)

    [[nodiscard]] Beta prior()   const { return Beta(alpha, beta_param); }
    [[nodiscard]] Beta posterior(u32 successes, u32 trials) const {
        return Beta(alpha+successes, beta_param+(trials-successes));
    }
    [[nodiscard]] f64 posterior_mean(u32 s, u32 n)     const { return (alpha+s)/(alpha+beta_param+n); }
    [[nodiscard]] f64 posterior_mode(u32 s, u32 n)     const {
        f64 a=alpha+s, b=beta_param+(n-s); return (a-1)/(a+b-2);
    }
    // Predictive probability for next observation
    [[nodiscard]] f64 posterior_predictive(u32 s, u32 n) const { return posterior_mean(s,n); }
    // Credible interval (approximate via beta quantile)
    [[nodiscard]] std::pair<f64,f64> credible_interval(u32 s, u32 n, f64 level=0.95) const {
        Beta post=posterior(s,n);
        // Bisection to find quantiles
        f64 lo_p=(1-level)*0.5, hi_p=1-lo_p;
        // Find x such that post.cdf(x)=p via bisection
        auto inv_cdf=[&](f64 p)->f64{
            f64 a=0,b=1;
            for(u32 i=0;i<60;++i){ f64 m=0.5*(a+b); if(post.cdf(m)<p)a=m;else b=m; }
            return 0.5*(a+b);
        };
        return {inv_cdf(lo_p), inv_cdf(hi_p)};
    }
};

// Gamma-Poisson (Normal-Gamma): Gamma(α,β) prior on λ, Poisson(λ) likelihood
struct GammaPoissonModel {
    f64 alpha{1}, beta_rate{1};

    [[nodiscard]] Gamma prior()   const { return Gamma(alpha,beta_rate); }
    [[nodiscard]] Gamma posterior(u32 count, u32 observations) const {
        return Gamma(alpha+count, beta_rate+observations);
    }
    [[nodiscard]] f64 posterior_mean(u32 cnt, u32 obs) const { return (alpha+cnt)/(beta_rate+obs); }
    [[nodiscard]] f64 predictive_pmf(u32 k, u32 cnt, u32 obs) const {
        // Negative Binomial predictive
        f64 r=alpha+cnt, p_nb=1.0/(1+(beta_rate+obs));
        return std::exp(std::lgamma(r+k)-std::lgamma(r)-std::lgamma(k+1)+k*std::log(p_nb)+(r)*std::log(1-p_nb));
    }
};

// Normal-Normal (known variance): N(μ0,τ²) prior on μ, N(μ,σ²) likelihood
struct NormalNormalModel {
    f64 mu0{0}, tau2{1}; // prior mean and variance
    f64 sigma2{1};       // likelihood variance (assumed known)

    [[nodiscard]] std::pair<f64,f64> posterior(const std::vector<f64>& data) const {
        u32 n=u32(data.size());
        f64 xbar=0; for(auto v:data) xbar+=v; xbar/=n;
        f64 tau2_n=1.0/(1/tau2+n/sigma2);
        f64 mu_n=tau2_n*(mu0/tau2+n*xbar/sigma2);
        return {mu_n,tau2_n};
    }
    [[nodiscard]] f64 posterior_predictive_var(u32 n) const { return sigma2+1.0/(1/tau2+n/sigma2); }
};

// Normal-InverseGamma: unknown mean and variance
struct NormalInverseGammaModel {
    f64 mu0{0},lambda{1},alpha{2},beta{1};

    struct NIGParams { f64 mu,lambda,alpha,beta; };
    [[nodiscard]] NIGParams posterior(const std::vector<f64>& x) const {
        u32 n=u32(x.size());
        f64 xbar=0; for(auto v:x) xbar+=v; xbar/=n;
        f64 S=0; for(auto v:x) S+=(v-xbar)*(v-xbar);
        f64 mu_n=(lambda*mu0+n*xbar)/(lambda+n);
        f64 lambda_n=lambda+n;
        f64 alpha_n=alpha+n*0.5;
        f64 beta_n=beta+0.5*S+lambda*n*(xbar-mu0)*(xbar-mu0)/(2*(lambda+n));
        return {mu_n,lambda_n,alpha_n,beta_n};
    }
    // Posterior predictive: Student's t distribution
    [[nodiscard]] StudentT predictive_distribution(const std::vector<f64>& x) const {
        auto [mu_n,lam_n,alp_n,bet_n]=posterior(x);
        return StudentT{2*alp_n};
    }
};

// Dirichlet-Multinomial: Dirichlet(α) prior on categorical probabilities
struct DirichletMultinomialModel {
    std::vector<f64> alpha; // concentration parameters

    [[nodiscard]] std::vector<f64> posterior_params(const std::vector<u32>& counts) const {
        assert(counts.size()==alpha.size());
        std::vector<f64> post(alpha.size());
        for(u32 i=0;i<u32(alpha.size());++i) post[i]=alpha[i]+counts[i];
        return post;
    }
    [[nodiscard]] std::vector<f64> posterior_mean(const std::vector<u32>& counts) const {
        auto post=posterior_params(counts);
        f64 sum=0; for(auto v:post) sum+=v;
        std::vector<f64> m(post.size());
        for(u32 i=0;i<u32(post.size());++i) m[i]=post[i]/sum;
        return m;
    }
    // Dirichlet entropy
    [[nodiscard]] f64 entropy(const std::vector<f64>& a) const {
        f64 sum=0; for(auto v:a) sum+=v;
        f64 H=std::lgamma(sum);
        for(auto v:a) H-=std::lgamma(v);
        H+=(sum-a.size())*std::digamma(sum); // digamma approximation below
        for(auto v:a) H-=(v-1)*std::digamma(v);
        return H;
    }
    [[nodiscard]] f64 digamma(f64 x) const {
        // Stirling series approximation for digamma
        if(x<6) return digamma(x+1)-1/x;
        return std::log(x)-1/(2*x)-1/(12*x*x)+1/(120*x*x*x*x);
    }
};

// ─── Variational Bayes (ELBO maximisation) ────────────────────────────────────

// Mean-field VI for a simple Gaussian mixture component
// Minimises KL(q||p) by coordinate ascent variational inference
struct CAVI_Gaussian {
    // Variational parameters for N(μ_q, σ²_q)
    f64 mu_q{0}, sigma2_q{1};
    // Hyperparameters
    f64 mu_prior{0}, sigma2_prior{1}, sigma2_lik{1};

    // Update variational parameters given data x
    void update(const std::vector<f64>& x) {
        u32 n=u32(x.size());
        f64 xbar=0; for(auto v:x) xbar+=v; xbar/=n;
        // Variational posterior for Gaussian likelihood
        sigma2_q = 1.0/(1/sigma2_prior + n/sigma2_lik);
        mu_q     = sigma2_q*(mu_prior/sigma2_prior + n*xbar/sigma2_lik);
    }

    // Evidence Lower BOund
    [[nodiscard]] f64 elbo(const std::vector<f64>& x) const {
        u32 n=u32(x.size());
        f64 E=0;
        // E_q[log p(x|μ)] = -0.5*n/σ²_lik * (Σ(x_i-μ_q)² + n*σ²_q)
        for(auto xi:x) E-=0.5*(xi-mu_q)*(xi-mu_q)/sigma2_lik;
        E-=0.5*n*sigma2_q/sigma2_lik;
        // E_q[log p(μ)] = -0.5*(μ_q-μ_prior)²/σ²_prior - 0.5*σ²_q/σ²_prior
        E-=0.5*(mu_q-mu_prior)*(mu_q-mu_prior)/sigma2_prior;
        E-=0.5*sigma2_q/sigma2_prior;
        // -E_q[log q(μ)] = 0.5*log(2πe*σ²_q)
        E+=0.5*(1+std::log(2*std::numbers::pi_v<f64>*sigma2_q));
        return E;
    }
};

// ─── Bayesian linear regression ───────────────────────────────────────────────
struct BayesLinReg {
    u32  d{1};             // dimensionality
    f64  alpha{1};         // prior precision on weights
    f64  beta_noise{1};    // noise precision

    struct Posterior { std::vector<f64> mean; std::vector<std::vector<f64>> cov; };

    [[nodiscard]] Posterior fit(const std::vector<std::vector<f64>>& X, const std::vector<f64>& t) const {
        u32 n=u32(X.size()), dim=u32(X[0].size());
        // S_N^{-1} = αI + β * Φ^T Φ
        std::vector<std::vector<f64>> S_inv(dim,std::vector<f64>(dim,0));
        for(u32 i=0;i<dim;++i) S_inv[i][i]=alpha;
        for(u32 k=0;k<n;++k) for(u32 i=0;i<dim;++i) for(u32 j=0;j<dim;++j)
            S_inv[i][j]+=beta_noise*X[k][i]*X[k][j];
        // m_N = β * S_N * Φ^T t
        std::vector<f64> Atb(dim,0);
        for(u32 k=0;k<n;++k) for(u32 j=0;j<dim;++j) Atb[j]+=beta_noise*X[k][j]*t[k];
        // Solve S_inv * m = Atb using simple iteration (or reuse our solver)
        // Simplified: return just the prior for now (caller should use linear_algebra::solve)
        return {Atb, S_inv}; // caller must invert S_inv
    }

    // Predictive distribution at new point x*
    [[nodiscard]] std::pair<f64,f64> predictive(const std::vector<f64>& x_star,
        const Posterior& post, const std::vector<std::vector<f64>>& S_N) const
    {
        u32 dim=u32(x_star.size());
        // m_pred = m_N^T x*
        f64 mu=0; for(u32 i=0;i<dim;++i) mu+=post.mean[i]*x_star[i];
        // σ²_pred = 1/β + x*^T S_N x*
        f64 sig2=1/beta_noise;
        for(u32 i=0;i<dim;++i) for(u32 j=0;j<dim;++j) sig2+=x_star[i]*S_N[i][j]*x_star[j];
        return {mu,sig2};
    }
};

// ─── Sequential Bayesian estimation (Kalman-like for scalar) ─────────────────
struct ScalarKalman {
    f64 x{0}, P{1};   // state mean and variance
    f64 Q{0.1};       // process noise
    f64 R{1.0};       // measurement noise

    void predict(f64 F=1, f64 B=0, f64 u=0) { x=F*x+B*u; P=F*P*F+Q; }
    void update(f64 z) {
        f64 K=P/(P+R); x+=K*(z-x); P=(1-K)*P;
    }
    [[nodiscard]] f64 log_likelihood(f64 z) const {
        f64 innov=z-x, S=P+R;
        return -0.5*(std::log(2*std::numbers::pi_v<f64>*S)+innov*innov/S);
    }
};

// ─── Importance sampling estimator ────────────────────────────────────────────
// E_p[f(x)] ≈ Σ f(x_i) w_i where w_i = p(x_i)/q(x_i) (unnormalised)
[[nodiscard]] inline f64 importance_sampling_estimate(
    const std::vector<f64>& samples,
    const std::function<f64(f64)>& f_func,
    const std::function<f64(f64)>& log_p,
    const std::function<f64(f64)>& log_q)
{
    f64 num=0, denom=0;
    for(auto x:samples) {
        f64 log_w=log_p(x)-log_q(x);
        f64 w=std::exp(log_w);
        num+=w*f_func(x); denom+=w;
    }
    return denom>1e-30?num/denom:0;
}

// Effective sample size for IS
[[nodiscard]] inline f64 effective_sample_size(const std::vector<f64>& log_weights) {
    f64 lmax=*std::max_element(log_weights.begin(),log_weights.end());
    f64 sum_w=0,sum_w2=0;
    for(auto lw:log_weights){ f64 w=std::exp(lw-lmax); sum_w+=w; sum_w2+=w*w; }
    return sum_w*sum_w/sum_w2;
}

} // namespace qs
