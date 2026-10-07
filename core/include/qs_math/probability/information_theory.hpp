#pragma once
// qs_math/information_theory.hpp — Mutual info, Fisher info, AIC/BIC, MDL (C++20)
#include "scalar_types.hpp"
#include "probability.hpp"
#include "distributions.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <numbers>
#include <cassert>

namespace qs {

// ─── Entropy measures ─────────────────────────────────────────────────────────
// Rényi entropy of order α
[[nodiscard]] inline f64 renyi_entropy(const std::vector<f64>& p, f64 alpha) {
    if(std::abs(alpha-1.0)<1e-9) return entropy(p); // → Shannon in limit
    f64 s=0; for(auto v:p) if(v>1e-15) s+=std::pow(v,alpha);
    return std::log(s)/(1-alpha);
}

// Tsallis entropy (non-extensive)
[[nodiscard]] inline f64 tsallis_entropy(const std::vector<f64>& p, f64 q=2) {
    if(std::abs(q-1.0)<1e-9) return entropy(p);
    f64 s=0; for(auto v:p) if(v>1e-15) s+=std::pow(v,q);
    return (1-s)/(q-1);
}

// Joint entropy H(X,Y)
[[nodiscard]] inline f64 joint_entropy(const std::vector<std::vector<f64>>& joint) {
    f64 H=0;
    for(const auto& row:joint) for(auto v:row) if(v>1e-15) H-=v*std::log(v);
    return H;
}

// Conditional entropy H(Y|X) = H(X,Y) - H(X)
[[nodiscard]] inline f64 conditional_entropy(const std::vector<std::vector<f64>>& joint) {
    auto pX=marginal_x(joint);
    return joint_entropy(joint)-entropy(pX);
}

// ─── Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y) ───────────────────────
[[nodiscard]] inline f64 mutual_information(const std::vector<std::vector<f64>>& joint) {
    auto pX=marginal_x(joint), pY=marginal_y(joint);
    f64 HX=entropy(pX),HY=entropy(pY),HXY=joint_entropy(joint);
    return HX+HY-HXY;
}

// Normalised mutual information (NMI) ∈ [0,1]
[[nodiscard]] inline f64 nmi(const std::vector<std::vector<f64>>& joint) {
    auto pX=marginal_x(joint),pY=marginal_y(joint);
    f64 HX=entropy(pX),HY=entropy(pY);
    if(HX+HY<1e-15) return 1;
    return 2*mutual_information(joint)/(HX+HY);
}

// Variation of information (metric): VI = H(X|Y) + H(Y|X)
[[nodiscard]] inline f64 variation_of_information(const std::vector<std::vector<f64>>& joint) {
    return joint_entropy(joint)-mutual_information(joint); // = H(X|Y)+H(Y|X) = H(X,Y)-I(X;Y)
}

// ─── Fisher information ───────────────────────────────────────────────────────
// Fisher information I(θ) = E[(d/dθ log p(X;θ))²] = -E[d²/dθ² log p(X;θ)]
// Numerical: I(θ) ≈ -E[d²/dθ² log p]
[[nodiscard]] inline f64 fisher_information_numerical(
    const std::function<f64(f64,f64)>& log_pdf,  // (x, theta)
    f64 theta, const std::vector<f64>& samples, f64 h=1e-4)
{
    f64 I=0;
    for(auto x:samples) {
        f64 d2=(-log_pdf(x,theta+2*h)+16*log_pdf(x,theta+h)-30*log_pdf(x,theta)+16*log_pdf(x,theta-h)-log_pdf(x,theta-2*h))/(12*h*h);
        I-=d2;
    }
    return I/samples.size();
}

// Fisher information matrix for multivariate (diagonal approx)
[[nodiscard]] inline std::vector<f64> fisher_information_matrix_diag(
    const std::function<f64(const std::vector<f64>&, const std::vector<f64>&)>& log_pdf,
    const std::vector<f64>& theta, const std::vector<std::vector<f64>>& data, f64 h=1e-4)
{
    u32 d=u32(theta.size());
    std::vector<f64> I(d,0);
    for(const auto& x:data) {
        for(u32 j=0;j<d;++j) {
            auto tp=theta,tm=theta; tp[j]+=h; tm[j]-=h;
            f64 dlj=(log_pdf(x,tp)-log_pdf(x,tm))/(2*h);
            I[j]+=dlj*dlj;
        }
    }
    for(auto& v:I) v/=data.size();
    return I;
}

// Cramér-Rao lower bound on variance of unbiased estimator
[[nodiscard]] inline f64 cramer_rao_bound(f64 fisher_info) {
    return fisher_info>1e-30?1/fisher_info:std::numeric_limits<f64>::infinity();
}

// ─── Model selection criteria ─────────────────────────────────────────────────

// Akaike Information Criterion: AIC = 2k - 2 ln L
[[nodiscard]] inline f64 aic(f64 log_likelihood, u32 k) {
    return 2*k - 2*log_likelihood;
}

// Corrected AIC for small samples: AICc = AIC + 2k(k+1)/(n-k-1)
[[nodiscard]] inline f64 aicc(f64 log_likelihood, u32 k, u32 n) {
    f64 a=aic(log_likelihood,k);
    if(n<=k+1) return std::numeric_limits<f64>::infinity();
    return a + 2.0*k*(k+1)/(n-k-1);
}

// Bayesian Information Criterion: BIC = k ln(n) - 2 ln L
[[nodiscard]] inline f64 bic(f64 log_likelihood, u32 k, u32 n) {
    return k*std::log(f64(n)) - 2*log_likelihood;
}

// Deviance Information Criterion (Bayesian model comparison)
// DIC = -2 E_θ[log p(y|θ)] + 2 p_D where p_D = effective number of parameters
[[nodiscard]] inline f64 dic(f64 mean_log_lik, f64 log_lik_at_mean) {
    f64 p_D=-2*mean_log_lik+2*log_lik_at_mean;
    return -2*mean_log_lik+p_D;
}

// Minimum Description Length: MDL = -log L + k/2 log n
[[nodiscard]] inline f64 mdl(f64 log_likelihood, u32 k, u32 n) {
    return -log_likelihood + 0.5*k*std::log(f64(n));
}

// Log-likelihood for Gaussian model: Σ log N(x_i; μ, σ²)
[[nodiscard]] inline f64 gaussian_log_likelihood(const std::vector<f64>& x, f64 mu, f64 sigma2) {
    u32 n=u32(x.size());
    f64 ll=-0.5*n*std::log(2*std::numbers::pi_v<f64>*sigma2);
    for(auto xi:x) ll-=(xi-mu)*(xi-mu)/(2*sigma2);
    return ll;
}

// Compare models: returns index of model with lowest AIC
[[nodiscard]] inline u32 select_model_aic(const std::vector<f64>& log_likelihoods,
    const std::vector<u32>& param_counts)
{
    u32 best=0; f64 best_aic=std::numeric_limits<f64>::infinity();
    for(u32 i=0;i<u32(log_likelihoods.size());++i) {
        f64 a=aic(log_likelihoods[i],param_counts[i]);
        if(a<best_aic){best_aic=a;best=i;}
    }
    return best;
}

// Akaike weights: exp(-0.5*ΔAIC_i) / Σ exp(-0.5*ΔAIC_j)
[[nodiscard]] inline std::vector<f64> akaike_weights(const std::vector<f64>& aic_values) {
    f64 min_aic=*std::min_element(aic_values.begin(),aic_values.end());
    std::vector<f64> w(aic_values.size());
    f64 sum=0;
    for(u32 i=0;i<u32(aic_values.size());++i){ w[i]=std::exp(-0.5*(aic_values[i]-min_aic)); sum+=w[i]; }
    for(auto& v:w) v/=sum;
    return w;
}

// ─── Channel capacity and rate-distortion ─────────────────────────────────────

// Capacity of binary symmetric channel (crossover probability p)
[[nodiscard]] inline f64 bsc_capacity(f64 p) {
    if(p<=0||p>=1) return p==0||p==1?1:0;
    return 1+p*std::log2(p)+(1-p)*std::log2(1-p);
}

// AWGN channel capacity: C = 1/2 log2(1 + SNR)
[[nodiscard]] inline f64 awgn_capacity(f64 snr) {
    return 0.5*std::log2(1+snr);
}

// Shannon's source coding theorem: minimum bits per symbol = H(X)/log(2)
[[nodiscard]] inline f64 min_bits_per_symbol(const std::vector<f64>& p) {
    return entropy(p,2.0); // entropy in bits
}

// Data processing inequality: I(X;Z) ≤ I(X;Y) if X→Y→Z Markov chain
// This returns true if the chain satisfies DPI (check with joint distributions)
[[nodiscard]] inline bool satisfies_dpi(
    const std::vector<std::vector<f64>>& joint_xy,
    const std::vector<std::vector<f64>>& joint_yz)
{
    return mutual_information(joint_xy) >= mutual_information(joint_yz);
}

// ─── Rate-distortion theory ───────────────────────────────────────────────────
// Distortion-rate function for Gaussian source with MSE: D = σ² 2^{-2R}
[[nodiscard]] inline f64 gaussian_distortion_rate(f64 sigma2, f64 R) {
    return sigma2*std::pow(2,-2*R);
}
[[nodiscard]] inline f64 gaussian_rate_distortion(f64 sigma2, f64 D) {
    if(D>=sigma2) return 0;
    return 0.5*std::log2(sigma2/D);
}

// ─── Kolmogorov complexity approximation via compression ─────────────────────
// Lower bound: length of gzip-like encoding (uses run-length as proxy)
[[nodiscard]] inline f64 kolmogorov_complexity_proxy(const std::vector<f64>& x, f64 tol=1e-6) {
    // Run-length encode and count transitions as crude complexity measure
    if(x.empty()) return 0;
    u32 runs=1;
    for(u32 i=1;i<u32(x.size());++i) if(std::abs(x[i]-x[i-1])>tol) ++runs;
    return std::log2(f64(runs)+1)*std::log2(f64(x.size())+1);
}

} // namespace qs
