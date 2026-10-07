#pragma once
// qs_math/probability/distributions.hpp — PDF/CDF/quantile for common distributions
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

inline constexpr f64 k_sqrt_2pi = 2.5066282746310002;
inline constexpr f64 k_sqrt_2   = 1.4142135623730951;
inline constexpr f64 k_pi_d     = 3.14159265358979323846;

// ─── Normal distribution ──────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 normal_pdf(f64 x, f64 mu=0, f64 sigma=1) noexcept {
    f64 z=(x-mu)/sigma;
    return std::exp(-0.5*z*z)/(sigma*k_sqrt_2pi);
}
[[nodiscard]] constexpr f64 normal_cdf(f64 x, f64 mu=0, f64 sigma=1) noexcept {
    return 0.5*std::erfc(-(x-mu)/(sigma*k_sqrt_2));
}
// Quantile (inverse CDF) via rational approximation (Beasley-Springer-Moro)
[[nodiscard]] constexpr f64 normal_quantile(f64 p) noexcept {
    if(p<=0) return -1e30; if(p>=1) return 1e30;
    // Rational approx valid for 0<p<1
    f64 t=(p<0.5)?std::sqrt(-2*std::log(p)):std::sqrt(-2*std::log(1-p));
    f64 c0=2.515517,c1=0.802853,c2=0.010328;
    f64 d1=1.432788,d2=0.189269,d3=0.001308;
    f64 q=t-(c0+c1*t+c2*t*t)/(1+d1*t+d2*t*t+d3*t*t*t);
    return p<0.5?-q:q;
}
[[nodiscard]] constexpr f64 normal_log_pdf(f64 x, f64 mu=0, f64 sigma=1) noexcept {
    f64 z=(x-mu)/sigma;
    return -0.5*z*z-std::log(sigma*k_sqrt_2pi);
}

// ─── Student's t distribution ─────────────────────────────────────────────────
[[nodiscard]] constexpr f64 t_pdf(f64 x, f64 nu) noexcept {
    f64 c=std::exp(std::lgamma((nu+1)/2)-std::lgamma(nu/2));
    c/=std::sqrt(nu*k_pi_d);
    return c*std::pow(1+x*x/nu,-(nu+1)/2);
}
// CDF via regularized incomplete beta
[[nodiscard]] constexpr f64 t_cdf(f64 x, f64 nu) noexcept {
    // Large nu → normal
    if(nu>1000) return normal_cdf(x);
    // Use regularized beta function approximation
    f64 v=nu/(nu+x*x);
    // Compute I_v(nu/2, 1/2) via series (simplified)
    f64 a=nu/2,b=0.5;
    f64 s=1,t_=1;
    for(int k=1;k<=50;++k){t_*=((a+b+k-1)/(a+k))*(v);if(k==1)t_*=(1-v);s+=t_;}
    f64 Iv=std::pow(v,a)*std::pow(1-v,b)*s;
    Iv/=std::exp(std::lgamma(a)+std::lgamma(b)-std::lgamma(a+b));
    Iv=std::max(0.0,std::min(1.0,Iv));
    return x>=0?1-0.5*Iv:0.5*Iv;
}

// ─── Chi-squared distribution ─────────────────────────────────────────────────
[[nodiscard]] constexpr f64 chi2_pdf(f64 x, f64 k) noexcept {
    if(x<=0) return 0;
    f64 half_k=k/2;
    return std::exp((half_k-1)*std::log(x)-x/2-half_k*std::log(2.0)-std::lgamma(half_k));
}
[[nodiscard]] constexpr f64 chi2_mean(f64 k) noexcept { return k; }
[[nodiscard]] constexpr f64 chi2_variance(f64 k) noexcept { return 2*k; }

// ─── Gamma distribution ───────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 gamma_pdf(f64 x, f64 alpha, f64 beta=1) noexcept {
    if(x<=0) return 0;
    return std::exp((alpha-1)*std::log(x)-x/beta-alpha*std::log(beta)-std::lgamma(alpha));
}
[[nodiscard]] constexpr f64 gamma_mean(f64 alpha, f64 beta) noexcept {return alpha*beta;}
[[nodiscard]] constexpr f64 gamma_variance(f64 alpha, f64 beta) noexcept {return alpha*beta*beta;}
[[nodiscard]] constexpr f64 gamma_mode(f64 alpha, f64 beta) noexcept {
    return alpha>=1?(alpha-1)*beta:0;
}

// ─── Beta distribution ────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 beta_pdf(f64 x, f64 a, f64 b) noexcept {
    if(x<=0||x>=1) return 0;
    return std::exp((a-1)*std::log(x)+(b-1)*std::log(1-x)
                    -std::lgamma(a)-std::lgamma(b)+std::lgamma(a+b));
}
[[nodiscard]] constexpr f64 beta_mean(f64 a, f64 b) noexcept {return a/(a+b);}
[[nodiscard]] constexpr f64 beta_mode(f64 a, f64 b) noexcept {
    return (a>1&&b>1)?(a-1)/(a+b-2):0;
}

// ─── Exponential distribution ─────────────────────────────────────────────────
[[nodiscard]] constexpr f64 exp_pdf(f64 x, f64 lam) noexcept {
    return x<0?0:lam*std::exp(-lam*x);
}
[[nodiscard]] constexpr f64 exp_cdf(f64 x, f64 lam) noexcept {
    return x<0?0:1-std::exp(-lam*x);
}
[[nodiscard]] constexpr f64 exp_quantile(f64 p, f64 lam) noexcept {
    return -std::log(1-p)/lam;
}
[[nodiscard]] constexpr f64 exp_mean(f64 lam) noexcept {return 1/lam;}
[[nodiscard]] constexpr f64 exp_variance(f64 lam) noexcept {return 1/(lam*lam);}

// ─── Poisson distribution ─────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 poisson_pmf(i64 k, f64 lam) noexcept {
    return std::exp(k*std::log(lam)-lam-std::lgamma(k+1));
}
[[nodiscard]] constexpr f64 poisson_cdf(i64 k, f64 lam) noexcept {
    f64 s=0; for(i64 i=0;i<=k;++i) s+=poisson_pmf(i,lam); return s;
}

// ─── Binomial distribution ────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 binomial_pmf(i64 k, i64 n, f64 p) noexcept {
    return std::exp(std::lgamma(n+1)-std::lgamma(k+1)-std::lgamma(n-k+1)
                    +k*std::log(p)+(n-k)*std::log(1-p));
}
[[nodiscard]] constexpr f64 binomial_cdf(i64 k, i64 n, f64 p) noexcept {
    f64 s=0; for(i64 i=0;i<=k;++i) s+=binomial_pmf(i,n,p); return s;
}

// ─── Weibull distribution ─────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 weibull_pdf(f64 x, f64 k, f64 lam) noexcept {
    if(x<0) return 0;
    f64 xl=x/lam;
    return (k/lam)*std::pow(xl,k-1)*std::exp(-std::pow(xl,k));
}
[[nodiscard]] constexpr f64 weibull_cdf(f64 x, f64 k, f64 lam) noexcept {
    return x<0?0:1-std::exp(-std::pow(x/lam,k));
}
[[nodiscard]] constexpr f64 weibull_quantile(f64 p, f64 k, f64 lam) noexcept {
    return lam*std::pow(-std::log(1-p),1/k);
}
[[nodiscard]] constexpr f64 weibull_mean(f64 k, f64 lam) noexcept {
    return lam*std::exp(std::lgamma(1+1/k));
}

// ─── Log-normal distribution ──────────────────────────────────────────────────
[[nodiscard]] constexpr f64 lognormal_pdf(f64 x, f64 mu, f64 sigma) noexcept {
    if(x<=0) return 0;
    return normal_pdf(std::log(x),mu,sigma)/x;
}
[[nodiscard]] constexpr f64 lognormal_cdf(f64 x, f64 mu, f64 sigma) noexcept {
    return x<=0?0:normal_cdf(std::log(x),mu,sigma);
}
[[nodiscard]] constexpr f64 lognormal_mean(f64 mu, f64 sigma) noexcept {
    return std::exp(mu+sigma*sigma/2);
}
[[nodiscard]] constexpr f64 lognormal_variance(f64 mu, f64 sigma) noexcept {
    f64 s2=sigma*sigma;
    return (std::exp(s2)-1)*std::exp(2*mu+s2);
}

// ─── Cauchy distribution ──────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 cauchy_pdf(f64 x, f64 x0=0, f64 gamma=1) noexcept {
    f64 t=(x-x0)/gamma;
    return 1.0/(k_pi_d*gamma*(1+t*t));
}
[[nodiscard]] constexpr f64 cauchy_cdf(f64 x, f64 x0=0, f64 gamma=1) noexcept {
    return 0.5+std::atan2(x-x0,gamma)/k_pi_d;
}
[[nodiscard]] constexpr f64 cauchy_quantile(f64 p, f64 x0=0, f64 gamma=1) noexcept {
    return x0+gamma*std::tan(k_pi_d*(p-0.5));
}

// ─── Dirichlet distribution (log density) ────────────────────────────────────
template<int K>
[[nodiscard]] constexpr f64 dirichlet_log_pdf(
    const std::array<f64,K>& x, const std::array<f64,K>& alpha) noexcept
{
    f64 sum_alpha=0,log_norm=0;
    for(auto a:alpha){sum_alpha+=a;log_norm+=std::lgamma(a);}
    log_norm=std::lgamma(sum_alpha)-log_norm;
    f64 logp=log_norm;
    for(int i=0;i<K;++i) logp+=(alpha[i]-1)*std::log(x[i]+1e-15);
    return logp;
}

// ─── Moment generating functions ─────────────────────────────────────────────
[[nodiscard]] constexpr f64 normal_mgf(f64 t, f64 mu=0, f64 sigma=1) noexcept {
    return std::exp(mu*t+0.5*sigma*sigma*t*t);
}
[[nodiscard]] constexpr f64 gamma_mgf(f64 t, f64 alpha, f64 beta=1) noexcept {
    return t<1/beta?std::pow(1-beta*t,-alpha):0;
}
[[nodiscard]] constexpr f64 poisson_pgf(f64 z, f64 lam) noexcept {
    return std::exp(lam*(z-1));
}
} // namespace qs::math
