#pragma once
// qs_math/probability.hpp — Probability spaces, axioms, conditional, Bayes, combinatorics (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <functional>
#include <numbers>

namespace qs {

// ─── Basic probability axioms ─────────────────────────────────────────────────

// Validate probability distribution (sums to 1, all non-negative)
[[nodiscard]] inline bool is_valid_distribution(const std::vector<f64>& p, f64 tol=1e-7) {
    if(p.empty()) return false;
    f64 sum=0;
    for(auto v:p) { if(v<-tol) return false; sum+=v; }
    return std::abs(sum-1.0)<tol;
}

// Normalize a distribution in-place
inline void normalize(std::vector<f64>& p) {
    f64 s=0; for(auto v:p) s+=v;
    if(s>1e-30) for(auto& v:p) v/=s;
}

// ─── Conditional probability: P(A|B) = P(A∩B) / P(B) ────────────────────────
[[nodiscard]] inline f64 conditional_probability(f64 p_ab, f64 p_b) {
    assert(p_b>0);
    return p_ab/p_b;
}

// Law of total probability: P(A) = Σ P(A|B_i) P(B_i)
[[nodiscard]] inline f64 total_probability(const std::vector<f64>& p_a_given_b,
    const std::vector<f64>& p_b)
{
    assert(p_a_given_b.size()==p_b.size());
    f64 s=0;
    for(u32 i=0;i<u32(p_b.size());++i) s+=p_a_given_b[i]*p_b[i];
    return s;
}

// Bayes' theorem: P(B_i|A) = P(A|B_i)P(B_i) / P(A)
[[nodiscard]] inline std::vector<f64> bayes_update(const std::vector<f64>& likelihood,
    const std::vector<f64>& prior)
{
    assert(likelihood.size()==prior.size());
    std::vector<f64> posterior(likelihood.size());
    f64 evidence=total_probability(likelihood,prior);
    for(u32 i=0;i<u32(prior.size());++i)
        posterior[i]=likelihood[i]*prior[i]/(evidence+1e-300);
    return posterior;
}

// Sequential Bayes update (multiple observations)
[[nodiscard]] inline std::vector<f64> bayes_sequential(
    const std::vector<std::vector<f64>>& likelihoods, std::vector<f64> prior)
{
    for(const auto& L:likelihoods) prior=bayes_update(L,prior);
    return prior;
}

// ─── Independence and correlation ─────────────────────────────────────────────

// Test independence: P(A∩B) ≈ P(A)*P(B)
[[nodiscard]] inline bool are_independent(f64 p_ab, f64 p_a, f64 p_b, f64 tol=1e-6) {
    return std::abs(p_ab - p_a*p_b) < tol;
}

// Joint distribution from marginals (assuming independence)
[[nodiscard]] inline std::vector<std::vector<f64>> joint_independent(
    const std::vector<f64>& pA, const std::vector<f64>& pB)
{
    std::vector<std::vector<f64>> J(pA.size(), std::vector<f64>(pB.size()));
    for(u32 i=0;i<u32(pA.size());++i) for(u32 j=0;j<u32(pB.size());++j)
        J[i][j]=pA[i]*pB[j];
    return J;
}

// Marginal from joint: p_X[i] = Σ_j J[i][j]
[[nodiscard]] inline std::vector<f64> marginal_x(const std::vector<std::vector<f64>>& J) {
    std::vector<f64> p(J.size());
    for(u32 i=0;i<u32(J.size());++i) for(auto v:J[i]) p[i]+=v;
    return p;
}
[[nodiscard]] inline std::vector<f64> marginal_y(const std::vector<std::vector<f64>>& J) {
    if(J.empty()) return {};
    std::vector<f64> p(J[0].size(),0);
    for(const auto& row:J) for(u32 j=0;j<u32(row.size());++j) p[j]+=row[j];
    return p;
}

// ─── Combinatorial probability helpers ───────────────────────────────────────

// n! (exact for n<=20)
[[nodiscard]] inline f64 factorial_f64(u32 n) {
    static constexpr f64 fac[21]={1,1,2,6,24,120,720,5040,40320,362880,
        3628800,39916800,479001600,6227020800.,87178291200.,1307674368000.,
        20922789888000.,355687428096000.,6402373705728000.,121645100408832000.,2432902008176640000.};
    if(n<=20) return fac[n];
    // Stirling for large n
    f64 r=std::sqrt(2*std::numbers::pi_v<f64>*n)*std::pow(n/std::numbers::e_v<f64>,n);
    return r*(1+1.0/(12*n));
}

[[nodiscard]] inline f64 log_factorial(u32 n) {
    f64 s=0; for(u32 i=2;i<=n;++i) s+=std::log(f64(i)); return s;
}

// C(n,k) = n! / (k! (n-k)!)
[[nodiscard]] inline f64 binomial_coeff_f64(u32 n, u32 k) {
    if(k>n) return 0; if(k==0||k==n) return 1;
    k=std::min(k,n-k);
    f64 r=1;
    for(u32 i=0;i<k;++i) r*=f64(n-i)/(i+1);
    return std::round(r);
}
[[nodiscard]] inline f64 log_binomial(u32 n, u32 k) {
    return log_factorial(n)-log_factorial(k)-log_factorial(n-k);
}

// Multinomial coefficient: n! / (k1! k2! ... km!)
[[nodiscard]] inline f64 multinomial(u32 n, const std::vector<u32>& ks) {
    f64 r=log_factorial(n);
    for(auto k:ks) r-=log_factorial(k);
    return std::exp(r);
}

// Stirling numbers of second kind S(n,k)
[[nodiscard]] inline f64 stirling2(u32 n, u32 k) {
    if(k==0) return n==0?1:0;
    if(k==n) return 1;
    if(k>n)  return 0;
    // DP
    std::vector<std::vector<f64>> S(n+1,std::vector<f64>(n+1,0));
    S[0][0]=1;
    for(u32 i=1;i<=n;++i) for(u32 j=1;j<=i;++j) S[i][j]=j*S[i-1][j]+S[i-1][j-1];
    return S[n][k];
}

// ─── Probability density utilities ───────────────────────────────────────────

// Log-sum-exp (numerically stable)
[[nodiscard]] inline f64 log_sum_exp(const std::vector<f64>& logp) {
    f64 mx=*std::max_element(logp.begin(),logp.end());
    f64 s=0; for(auto v:logp) s+=std::exp(v-mx);
    return mx+std::log(s);
}

// Softmax
[[nodiscard]] inline std::vector<f64> softmax(const std::vector<f64>& x) {
    std::vector<f64> logp=x;
    f64 lse=log_sum_exp(logp);
    for(auto& v:logp) v=std::exp(v-lse);
    return logp;
}

// KL divergence: KL(P||Q) = Σ P_i log(P_i/Q_i)
[[nodiscard]] inline f64 kl_divergence(const std::vector<f64>& P, const std::vector<f64>& Q) {
    f64 kl=0;
    for(u32 i=0;i<u32(P.size());++i)
        if(P[i]>1e-15) kl+=P[i]*std::log(P[i]/(Q[i]+1e-30));
    return kl;
}

// Jensen-Shannon divergence (symmetric, bounded)
[[nodiscard]] inline f64 js_divergence(const std::vector<f64>& P, const std::vector<f64>& Q) {
    std::vector<f64> M(P.size());
    for(u32 i=0;i<u32(M.size());++i) M[i]=0.5*(P[i]+Q[i]);
    return 0.5*kl_divergence(P,M)+0.5*kl_divergence(Q,M);
}

// Total variation distance
[[nodiscard]] inline f64 total_variation(const std::vector<f64>& P, const std::vector<f64>& Q) {
    f64 tv=0;
    for(u32 i=0;i<u32(P.size());++i) tv+=std::abs(P[i]-Q[i]);
    return 0.5*tv;
}

// Hellinger distance
[[nodiscard]] inline f64 hellinger(const std::vector<f64>& P, const std::vector<f64>& Q) {
    f64 h=0;
    for(u32 i=0;i<u32(P.size());++i){f64 d=std::sqrt(P[i])-std::sqrt(Q[i]);h+=d*d;}
    return std::sqrt(h)*std::numbers::sqrt2_v<f64>/2;
}

// ─── Cumulative distribution function helpers ─────────────────────────────────
// Build CDF from PMF
[[nodiscard]] inline std::vector<f64> pmf_to_cdf(const std::vector<f64>& pmf) {
    std::vector<f64> cdf(pmf.size());
    std::partial_sum(pmf.begin(),pmf.end(),cdf.begin());
    return cdf;
}

// Invert CDF: find smallest k such that CDF[k] >= u
[[nodiscard]] inline u32 invert_cdf(const std::vector<f64>& cdf, f64 u) {
    for(u32 i=0;i<u32(cdf.size());++i) if(cdf[i]>=u) return i;
    return u32(cdf.size())-1;
}

// Expected value E[f(X)] under distribution p
[[nodiscard]] inline f64 expectation(const std::vector<f64>& p, const std::vector<f64>& values) {
    assert(p.size()==values.size());
    f64 e=0; for(u32 i=0;i<u32(p.size());++i) e+=p[i]*values[i];
    return e;
}

// Variance Var[X] = E[X²] - E[X]²
[[nodiscard]] inline f64 variance_discrete(const std::vector<f64>& p, const std::vector<f64>& values) {
    f64 ex=expectation(p,values);
    std::vector<f64> x2(values.size()); for(u32 i=0;i<u32(values.size());++i) x2[i]=values[i]*values[i];
    return expectation(p,x2)-ex*ex;
}

// ─── Information theory ───────────────────────────────────────────────────────
// Shannon entropy H(P) = -Σ p log p (nats)
[[nodiscard]] inline f64 entropy(const std::vector<f64>& p, f64 base=std::numbers::e_v<f64>) {
    f64 H=0,logb=std::log(base);
    for(auto v:p) if(v>1e-15) H-=v*std::log(v)/logb;
    return H;
}

// Cross-entropy H(P,Q) = -Σ p log q
[[nodiscard]] inline f64 cross_entropy(const std::vector<f64>& P, const std::vector<f64>& Q) {
    f64 H=0;
    for(u32 i=0;i<u32(P.size());++i) if(P[i]>1e-15) H-=P[i]*std::log(Q[i]+1e-30);
    return H;
}

} // namespace qs
