#pragma once
// qs_math/ai/reinforcement_learning.hpp — MDPs, Bellman, Q-learning, policy gradient, PPO math
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Markov Decision Process ──────────────────────────────────────────────────
template<int NS, int NA>
struct MDP {
    std::array<std::array<std::array<f64,NS>,NS>,NA> P{}; // P[a][s][s'] = transition
    std::array<std::array<f64,NS>,NA> R{};                // R[a][s] = expected reward
    f64 gamma{0.99};

    // Bellman equation: V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
    [[nodiscard]] constexpr std::array<f64,NS> bellman_optimality(
        const std::array<f64,NS>& V) const noexcept
    {
        std::array<f64,NS> V_new{};
        for(int s=0;s<NS;++s) {
            f64 best=-1e18;
            for(int a=0;a<NA;++a) {
                f64 q=R[a][s];
                for(int sp=0;sp<NS;++sp) q+=gamma*P[a][s][sp]*V[sp];
                if(q>best) best=q;
            }
            V_new[s]=best;
        }
        return V_new;
    }

    // Policy evaluation: V_π(s) = R(s,π(s)) + γ Σ P(s'|s,π(s)) V_π(s')
    [[nodiscard]] constexpr std::array<f64,NS> policy_eval(
        const std::array<int,NS>& policy, int iters=100) const noexcept
    {
        std::array<f64,NS> V{};
        for(int iter=0;iter<iters;++iter) {
            std::array<f64,NS> Vn{};
            for(int s=0;s<NS;++s) {
                int a=policy[s];
                Vn[s]=R[a][s];
                for(int sp=0;sp<NS;++sp) Vn[s]+=gamma*P[a][s][sp]*V[sp];
            }
            V=Vn;
        }
        return V;
    }

    // Policy improvement: greedy policy from V
    [[nodiscard]] constexpr std::array<int,NS> policy_improvement(
        const std::array<f64,NS>& V) const noexcept
    {
        std::array<int,NS> policy{};
        for(int s=0;s<NS;++s) {
            f64 best=-1e18; int best_a=0;
            for(int a=0;a<NA;++a) {
                f64 q=R[a][s];
                for(int sp=0;sp<NS;++sp) q+=gamma*P[a][s][sp]*V[sp];
                if(q>best){best=q;best_a=a;}
            }
            policy[s]=best_a;
        }
        return policy;
    }

    // Value iteration
    [[nodiscard]] constexpr std::array<f64,NS> value_iteration(
        f64 tol=1e-6, int max_iter=1000) const noexcept
    {
        std::array<f64,NS> V{};
        for(int iter=0;iter<max_iter;++iter) {
            auto Vn=bellman_optimality(V);
            f64 delta=0;
            for(int s=0;s<NS;++s) delta=std::max(delta,std::abs(Vn[s]-V[s]));
            V=Vn;
            if(delta<tol) break;
        }
        return V;
    }

    // Q-value: Q(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) V(s')
    [[nodiscard]] constexpr f64 q_value(int s, int a,
                                         const std::array<f64,NS>& V) const noexcept
    {
        f64 q=R[a][s];
        for(int sp=0;sp<NS;++sp) q+=gamma*P[a][s][sp]*V[sp];
        return q;
    }
};

// ─── Q-learning update ────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 q_learning_update(
    f64 Q_sa, f64 reward, f64 Q_max_next,
    f64 alpha, f64 gamma) noexcept
{
    return Q_sa+alpha*(reward+gamma*Q_max_next-Q_sa);
}

// SARSA update (on-policy)
[[nodiscard]] constexpr f64 sarsa_update(
    f64 Q_sa, f64 reward, f64 Q_snext_anext,
    f64 alpha, f64 gamma) noexcept
{
    return Q_sa+alpha*(reward+gamma*Q_snext_anext-Q_sa);
}

// Double Q-learning update
[[nodiscard]] constexpr f64 double_q_update(
    f64 Q1_sa, f64 reward, f64 Q2_eval,
    f64 alpha, f64 gamma) noexcept
{
    return Q1_sa+alpha*(reward+gamma*Q2_eval-Q1_sa);
}

// ─── Advantage function ───────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 advantage(f64 Q_sa, f64 V_s) noexcept { return Q_sa-V_s; }

// Generalized Advantage Estimation (GAE)
template<int T>
[[nodiscard]] constexpr std::array<f64,T> gae(
    const std::array<f64,T>& rewards,
    const std::array<f64,T>& values,
    f64 gamma=0.99, f64 lam=0.95) noexcept
{
    std::array<f64,T> advantages{};
    f64 gae_val=0;
    for(int t=T-1;t>=0;--t) {
        f64 next_val=(t<T-1)?values[t+1]:0;
        f64 delta=rewards[t]+gamma*next_val-values[t];
        gae_val=delta+gamma*lam*gae_val;
        advantages[t]=gae_val;
    }
    return advantages;
}

// ─── Policy gradient ─────────────────────────────────────────────────────────
// REINFORCE: ∇J(θ) = E[Σ_t ∇log π_θ(a_t|s_t) * G_t]
// Returns policy gradient for single trajectory
template<int T, int NA>
[[nodiscard]] constexpr f64 reinforce_gradient_term(
    f64 log_prob_at, f64 G_t) noexcept
{
    return log_prob_at*G_t;
}

// PPO clipped objective: L_clip = E[min(r_t*A_t, clip(r_t,1-ε,1+ε)*A_t)]
[[nodiscard]] constexpr f64 ppo_clip_objective(
    f64 prob_ratio, f64 advantage, f64 epsilon=0.2) noexcept
{
    f64 clip=std::max(1-epsilon,std::min(1+epsilon,prob_ratio));
    return std::min(prob_ratio*advantage,clip*advantage);
}
// Probability ratio r_t = π_θ(a|s) / π_θ_old(a|s)
[[nodiscard]] constexpr f64 prob_ratio(f64 log_prob_new, f64 log_prob_old) noexcept {
    return std::exp(log_prob_new-log_prob_old);
}
// KL divergence penalty (for PPO-KL variant)
[[nodiscard]] constexpr f64 kl_penalty(f64 log_prob_new, f64 log_prob_old) noexcept {
    return log_prob_old-log_prob_new; // approximate KL: E[log π_old - log π_new]
}

// ─── Entropy regularization ───────────────────────────────────────────────────
template<int NA>
[[nodiscard]] constexpr f64 policy_entropy(
    const std::array<f64,NA>& probs) noexcept
{
    f64 H=0;
    for(auto p:probs) if(p>1e-15) H-=p*std::log(p);
    return H;
}

// ─── TD(λ) eligibility traces ─────────────────────────────────────────────────
template<int NS>
struct EligibilityTraces {
    std::array<f64,NS> e{};
    f64 gamma{0.99},lam{0.9};

    constexpr void update(int s, f64 alpha, f64 td_error) noexcept {
        e[s]+=1; // accumulating traces
        for(int i=0;i<NS;++i) e[i]*=gamma*lam; // decay
    }
    constexpr void reset() noexcept { e.fill(0); }
};

// ─── Distributional RL: quantile regression loss ─────────────────────────────
[[nodiscard]] constexpr f64 huber_quantile_loss(
    f64 td_error, f64 tau, f64 kappa=1.0) noexcept
{
    f64 I=(td_error>=0)?1.0:0.0;
    f64 abs_err=std::abs(td_error);
    f64 L=abs_err<=kappa?0.5*td_error*td_error:kappa*(abs_err-0.5*kappa);
    return std::abs(tau-I)*L;
}

// ─── Multi-armed bandit: UCB1 ─────────────────────────────────────────────────
template<int K>
[[nodiscard]] constexpr int ucb1_select(
    const std::array<f64,K>& means,
    const std::array<i64,K>& counts,
    i64 total) noexcept
{
    int best=0; f64 best_val=-1e18;
    for(int k=0;k<K;++k) {
        if(counts[k]==0) return k;
        f64 val=means[k]+std::sqrt(2*std::log(f64(total))/counts[k]);
        if(val>best_val){best_val=val;best=k;}
    }
    return best;
}
// Thompson sampling update (Beta-Bernoulli)
[[nodiscard]] constexpr std::pair<f64,f64> thompson_update(
    f64 alpha, f64 beta_param, bool reward) noexcept
{
    return{alpha+(reward?1:0),beta_param+(reward?0:1)};
}
} // namespace qs::math
