#pragma once
// qs_math/probability/stochastic_processes.hpp — Markov chains, Brownian motion, Ito calculus
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Discrete Markov chain ────────────────────────────────────────────────────
template<int N>
struct MarkovChain {
    std::array<std::array<f64,N>,N> P{};  // transition matrix P[i][j] = P(i→j)

    // n-step transition: P^n
    [[nodiscard]] constexpr std::array<std::array<f64,N>,N> power(int n) const noexcept {
        std::array<std::array<f64,N>,N> result{},base=P;
        for(int i=0;i<N;++i) result[i][i]=1; // identity
        while(n>0) {
            if(n&1) { // result *= base
                std::array<std::array<f64,N>,N> tmp{};
                for(int i=0;i<N;++i) for(int k=0;k<N;++k) if(result[i][k])
                    for(int j=0;j<N;++j) tmp[i][j]+=result[i][k]*base[k][j];
                result=tmp;
            }
            std::array<std::array<f64,N>,N> tmp{};
            for(int i=0;i<N;++i) for(int k=0;k<N;++k) if(base[i][k])
                for(int j=0;j<N;++j) tmp[i][j]+=base[i][k]*base[k][j];
            base=tmp; n>>=1;
        }
        return result;
    }

    // Stationary distribution π: πP=π, sum(π)=1 (power iteration)
    [[nodiscard]] constexpr std::array<f64,N> stationary(int iters=500) const noexcept {
        std::array<f64,N> pi{};pi[0]=1;
        for(int it=0;it<iters;++it) {
            std::array<f64,N> npi{};
            for(int j=0;j<N;++j) for(int i=0;i<N;++i) npi[j]+=pi[i]*P[i][j];
            pi=npi;
        }
        return pi;
    }

    // Hitting time from state s to state t (expected)
    [[nodiscard]] constexpr f64 mean_hitting_time(int s, int t, int iters=1000) const noexcept {
        // h[i] = 1 + sum_{j≠t} P[i][j]*h[j], h[t]=0
        std::array<f64,N> h{};
        for(int it=0;it<iters;++it) {
            std::array<f64,N> hn{};
            for(int i=0;i<N;++i) {
                if(i==t){hn[i]=0;continue;}
                hn[i]=1;
                for(int j=0;j<N;++j) if(j!=t) hn[i]+=P[i][j]*h[j];
            }
            h=hn;
        }
        return h[s];
    }

    // Absorbing Markov chain: find Q (transient-transient) and N=(I-Q)^{-1}
    // Returns expected visits to transient states before absorption
    [[nodiscard]] constexpr bool is_absorbing(int state) const noexcept {
        return P[state][state]>=1.0-1e-12;
    }

    // Entropy rate H = -sum_i π_i sum_j P_{ij} log P_{ij}
    [[nodiscard]] constexpr f64 entropy_rate() const noexcept {
        auto pi=stationary();
        f64 H=0;
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            if(P[i][j]>1e-15) H-=pi[i]*P[i][j]*std::log(P[i][j]);
        return H;
    }
};

// ─── Geometric Brownian Motion ────────────────────────────────────────────────
// dS = μS dt + σS dW  →  S(t) = S0 * exp((μ-σ²/2)t + σW(t))
[[nodiscard]] constexpr f64 gbm_price(f64 S0,f64 mu,f64 sigma,f64 t,f64 W_t) noexcept {
    return S0*std::exp((mu-0.5*sigma*sigma)*t+sigma*W_t);
}

// ─── Ornstein-Uhlenbeck process ───────────────────────────────────────────────
// dX = θ(μ-X)dt + σ dW
// Exact mean and variance
[[nodiscard]] constexpr f64 ou_mean(f64 x0,f64 mu,f64 theta,f64 t) noexcept {
    return mu+(x0-mu)*std::exp(-theta*t);
}
[[nodiscard]] constexpr f64 ou_variance(f64 sigma,f64 theta,f64 t) noexcept {
    return sigma*sigma/(2*theta)*(1-std::exp(-2*theta*t));
}

// ─── Ito's lemma ─────────────────────────────────────────────────────────────
// If dX=a dt + b dW and Y=f(t,X):
// dY = (∂f/∂t + a∂f/∂x + b²/2 ∂²f/∂x²)dt + b∂f/∂x dW
template<typename F>
[[nodiscard]] constexpr std::pair<f64,f64> ito_lemma(F&& f,
    f64 t, f64 x, f64 a, f64 b, f64 h=1e-5) noexcept
{
    f64 df_dt=(f(t+h,x)-f(t-h,x))/(2*h);
    f64 df_dx=(f(t,x+h)-f(t,x-h))/(2*h);
    f64 d2f_dx2=(f(t,x+h)-2*f(t,x)+f(t,x-h))/(h*h);
    f64 drift_coeff=df_dt+a*df_dx+0.5*b*b*d2f_dx2;
    f64 diffusion_coeff=b*df_dx;
    return {drift_coeff,diffusion_coeff};
}

// ─── Feynman-Kac formula ──────────────────────────────────────────────────────
// u(t,x) = E[g(X_T) exp(-∫_t^T r ds) | X_t=x]
// satisfies PDE: ∂u/∂t + a∂u/∂x + b²/2 ∂²u/∂x² - ru = 0
// Black-Scholes as special case:
[[nodiscard]] constexpr f64 black_scholes_call(f64 S,f64 K,f64 r,f64 sigma,f64 T) noexcept {
    if(T<=0) return std::max(S-K,0.0);
    f64 d1=(std::log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*std::sqrt(T));
    f64 d2=d1-sigma*std::sqrt(T);
    // Standard normal CDF via erfc
    auto N_cdf=[](f64 x){return 0.5*std::erfc(-x/std::numbers::sqrt2_v<f64>);};
    return S*N_cdf(d1)-K*std::exp(-r*T)*N_cdf(d2);
}
[[nodiscard]] constexpr f64 black_scholes_put(f64 S,f64 K,f64 r,f64 sigma,f64 T) noexcept {
    return black_scholes_call(S,K,r,sigma,T)-S+K*std::exp(-r*T);
}

// ─── Monte Carlo European option pricing ──────────────────────────────────────
[[nodiscard]] constexpr f64 mc_option_price(f64 S0,f64 K,f64 r,f64 sigma,f64 T,
                                              bool call=true,int paths=10000,u64 seed=7) noexcept
{
    Xorshift64 rng{seed};
    f64 payoff_sum=0;
    for(int i=0;i<paths;++i) {
        f64 W=rng.normal()*std::sqrt(T);
        f64 ST=gbm_price(S0,r,sigma,T,W);
        payoff_sum+=call?std::max(ST-K,0.0):std::max(K-ST,0.0);
    }
    return std::exp(-r*T)*payoff_sum/paths;
}

// ─── Stochastic Euler-Maruyama scheme ─────────────────────────────────────────
template<int N, typename Drift, typename Diffusion>
[[nodiscard]] constexpr std::array<f64,N> euler_maruyama(
    Drift&& a, Diffusion&& b,
    const std::array<f64,N>& x0, f64 T,
    int steps=1000, u64 seed=42) noexcept
{
    Xorshift64 rng{seed};
    f64 dt=T/steps;
    std::array<f64,N> x=x0;
    for(int k=0;k<steps;++k) {
        auto drift=a(x);
        auto diff=b(x);
        for(int i=0;i<N;++i)
            x[i]+=drift[i]*dt+diff[i]*rng.normal()*std::sqrt(dt);
    }
    return x;
}

// ─── Milstein scheme (strong order 1) ────────────────────────────────────────
template<typename Drift, typename Diff, typename DDiff>
[[nodiscard]] constexpr f64 milstein_step(
    Drift&& a, Diff&& b, DDiff&& b_prime,
    f64 x, f64 dt, f64 dW) noexcept
{
    return x+a(x)*dt+b(x)*dW+0.5*b(x)*b_prime(x)*(dW*dW-dt);
}
} // namespace qs::math
