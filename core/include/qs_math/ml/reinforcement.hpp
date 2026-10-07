#pragma once
// qs_math/ml/reinforcement.hpp — Reinforcement learning math: MDPs, Bellman equations,
//   Q-learning, policy gradient, advantage estimation, replay buffer math,
//   exploration strategies, reward shaping, multi-arm bandit (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <deque>

namespace qs::math::ml {

// ─── MDP definitions ──────────────────────────────────────────────────────
struct Transition {
    int   state, action, next_state;
    f32   reward;
    bool  done;
};

// ─── Discount & return computation ───────────────────────────────────────
// Monte Carlo return: G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
[[nodiscard]] inline std::vector<f32> discounted_returns(
    std::span<const f32> rewards, f32 gamma=0.99f) noexcept
{
    int n=int(rewards.size());
    std::vector<f32> G(n);
    f32 g=0;
    for(int t=n-1;t>=0;--t){
        g=rewards[t]+gamma*g;
        G[t]=g;
    }
    return G;
}

// n-step return
[[nodiscard]] inline f32 nstep_return(
    std::span<const f32> rewards, f32 bootstrap_value,
    int n, f32 gamma=0.99f) noexcept
{
    f32 g=bootstrap_value;
    int steps=std::min(n,int(rewards.size()));
    for(int i=steps-1;i>=0;--i) g=rewards[i]+gamma*g;
    return g;
}

// ─── Bellman equations ────────────────────────────────────────────────────
// Q*(s,a) = r + gamma * max_a' Q*(s',a')
[[nodiscard]] inline f32 bellman_q(
    f32 reward, f32 gamma, f32 max_next_q, bool done) noexcept
{
    return reward + (done?0.f:gamma*max_next_q);
}

// Double Q target: r + gamma * Q(s', argmax_a' Q_online(s',a'), target)
[[nodiscard]] inline f32 double_q_target(
    f32 reward, f32 gamma,
    const std::vector<f32>& online_next, const std::vector<f32>& target_next,
    bool done) noexcept
{
    int best=int(std::max_element(online_next.begin(),online_next.end())-online_next.begin());
    return reward+(done?0.f:gamma*target_next[best]);
}

// V(s) = sum_a pi(a|s) * Q(s,a)
[[nodiscard]] inline f32 state_value(
    std::span<const f32> policy, std::span<const f32> q_values) noexcept
{
    int n=int(policy.size()); f32 v=0;
    for(int a=0;a<n;++a) v+=policy[a]*q_values[a];
    return v;
}

// ─── Advantage estimation ─────────────────────────────────────────────────
// A(s,a) = Q(s,a) - V(s)
[[nodiscard]] inline f32 advantage(f32 q, f32 v) noexcept { return q-v; }

// Generalized Advantage Estimation (GAE-Lambda)
[[nodiscard]] inline std::vector<f32> gae(
    std::span<const f32> rewards,
    std::span<const f32> values,
    f32 last_value, f32 gamma=0.99f, f32 lambda=0.95f,
    std::span<const bool> dones={}) noexcept
{
    int n=int(rewards.size());
    std::vector<f32> adv(n,0.f);
    f32 gae_val=0;
    f32 next_val=last_value;
    for(int t=n-1;t>=0;--t){
        bool done=!dones.empty()&&dones[t];
        f32 nv=done?0.f:next_val;
        f32 delta=rewards[t]+gamma*nv-values[t];
        gae_val=delta+gamma*lambda*(done?0.f:gae_val);
        adv[t]=gae_val;
        next_val=values[t];
    }
    return adv;
}

// ─── Policy gradient ──────────────────────────────────────────────────────
// REINFORCE gradient: grad = log_prob * G_t
[[nodiscard]] inline std::vector<f32> reinforce_grad(
    std::span<const f32> log_probs,
    std::span<const f32> returns,
    f32 baseline=0.f) noexcept
{
    int n=int(log_probs.size());
    std::vector<f32> grads(n);
    for(int t=0;t<n;++t) grads[t]=log_probs[t]*(returns[t]-baseline);
    return grads;
}

// PPO clipped ratio objective
[[nodiscard]] inline f32 ppo_loss(
    f32 ratio, f32 advantage, f32 clip_eps=0.2f) noexcept
{
    f32 obj=ratio*advantage;
    f32 clipped=clamp(ratio,1.f-clip_eps,1.f+clip_eps)*advantage;
    return -min(obj,clipped);
}

[[nodiscard]] inline f32 ppo_ratio(f32 log_prob_new, f32 log_prob_old) noexcept {
    return std::exp(log_prob_new-log_prob_old);
}

// Entropy bonus (encourages exploration)
[[nodiscard]] inline f32 discrete_entropy(std::span<const f32> probs) noexcept {
    f32 H=0;
    for(auto p:probs) if(p>1e-10f) H-=p*std::log(p);
    return H;
}
[[nodiscard]] inline f32 gaussian_entropy(f32 log_std) noexcept {
    return log_std+0.5f*std::log(2.f*f32(PI)*f32(E));
}

// ─── Value function loss ──────────────────────────────────────────────────
[[nodiscard]] inline f32 value_loss(f32 pred, f32 target) noexcept {
    f32 d=pred-target; return 0.5f*d*d;
}
[[nodiscard]] inline f32 value_loss_clipped(
    f32 pred, f32 old_pred, f32 target, f32 clip_eps=0.2f) noexcept
{
    f32 d1=pred-target;
    f32 pred_c=old_pred+clamp(pred-old_pred,-clip_eps,clip_eps);
    f32 d2=pred_c-target;
    return max(0.5f*d1*d1, 0.5f*d2*d2);
}

// ─── Exploration strategies ───────────────────────────────────────────────
// Epsilon-greedy action selection
[[nodiscard]] inline int epsilon_greedy(
    std::span<const f32> q_values, f32 epsilon, PCG32& rng) noexcept
{
    if(rng.next_f32()<epsilon)
        return int(rng.next_u32()%u32(q_values.size()));
    return int(std::max_element(q_values.begin(),q_values.end())-q_values.begin());
}

// Boltzmann (softmax) exploration
[[nodiscard]] inline int boltzmann_action(
    std::span<const f32> q_values, f32 temperature, PCG32& rng) noexcept
{
    int n=int(q_values.size());
    std::vector<f32> probs(n);
    f32 mx=*std::max_element(q_values.begin(),q_values.end());
    f32 sum=0;
    for(int i=0;i<n;++i){probs[i]=std::exp((q_values[i]-mx)/temperature);sum+=probs[i];}
    for(auto& p:probs) p/=sum;
    f32 u=rng.next_f32(); f32 cdf=0;
    for(int i=0;i<n-1;++i){cdf+=probs[i];if(u<cdf) return i;}
    return n-1;
}

// UCB1 action selection (multi-arm bandit)
[[nodiscard]] inline int ucb1_action(
    std::span<const f32> means,
    std::span<const int> counts,
    int total_t, f32 c=std::sqrt(2.f)) noexcept
{
    int n=int(means.size()); int best=0; f32 best_val=-1e30f;
    for(int i=0;i<n;++i){
        if(counts[i]==0) return i;
        f32 ucb=means[i]+c*std::sqrt(std::log(f32(total_t))/counts[i]);
        if(ucb>best_val){best_val=ucb;best=i;}
    }
    return best;
}

// Thompson sampling (Beta distribution)
[[nodiscard]] inline int thompson_sampling(
    std::span<const f32> alpha, std::span<const f32> beta_p, PCG32& rng) noexcept
{
    int n=int(alpha.size()); int best=0; f32 best_val=-1e30f;
    for(int i=0;i<n;++i){
        // Sample from Beta(alpha_i, beta_i) via ratio of Gamma samples
        // Approximate with mean + random perturbation
        f32 a=alpha[i], b=beta_p[i];
        f32 mean=a/(a+b);
        f32 std=std::sqrt(a*b/((a+b)*(a+b)*(a+b+1.f)));
        f32 u=max(1e-7f,rng.next_f32()),v=rng.next_f32();
        f32 sample=clamp01(mean+std*std::sqrt(-2.f*std::log(u))*std::cos(f32(TWO_PI)*v));
        if(sample>best_val){best_val=sample;best=i;}
    }
    return best;
}

// ─── Replay buffer ────────────────────────────────────────────────────────
struct ReplayBuffer {
    std::deque<Transition> buf;
    int capacity;
    u32 seed;
    PCG32 rng;

    ReplayBuffer(int cap, u32 s=42):capacity(cap),seed(s),rng(s){}

    void push(Transition t) noexcept {
        if(int(buf.size())>=capacity) buf.pop_front();
        buf.push_back(t);
    }

    [[nodiscard]] std::vector<Transition> sample(int n) noexcept {
        int sz=int(buf.size());
        n=std::min(n,sz);
        std::vector<int> idx(sz); std::iota(idx.begin(),idx.end(),0);
        // Partial Fisher-Yates
        for(int i=0;i<n;++i){
            int j=i+rng.next_u32()%(sz-i);
            std::swap(idx[i],idx[j]);
        }
        std::vector<Transition> batch(n);
        for(int i=0;i<n;++i) batch[i]=buf[idx[i]];
        return batch;
    }

    [[nodiscard]] int size() const noexcept { return int(buf.size()); }
    [[nodiscard]] bool ready(int min_size) const noexcept { return size()>=min_size; }
};

// ─── Prioritized experience replay ────────────────────────────────────────
struct PrioritizedReplay {
    std::vector<Transition> buf;
    std::vector<f32>        priorities;
    int capacity, write_pos=0, cur_size=0;
    f32 alpha=0.6f, beta=0.4f, eps=1e-6f;
    PCG32 rng;

    PrioritizedReplay(int cap,u32 seed=42):buf(cap),priorities(cap,1.f),capacity(cap),rng(seed){}

    void push(Transition t, f32 priority=1.f) noexcept {
        buf[write_pos]=t;
        priorities[write_pos]=std::pow(priority+eps,alpha);
        write_pos=(write_pos+1)%capacity;
        cur_size=std::min(cur_size+1,capacity);
    }

    struct PERBatch {
        std::vector<Transition> transitions;
        std::vector<int>        indices;
        std::vector<f32>        weights;
    };

    [[nodiscard]] PERBatch sample(int n) noexcept {
        PERBatch batch; batch.transitions.resize(n); batch.indices.resize(n); batch.weights.resize(n);
        f32 sum=0; for(int i=0;i<cur_size;++i) sum+=priorities[i];
        f32 max_p=*std::max_element(priorities.begin(),priorities.begin()+cur_size);
        f32 max_w=std::pow(f32(cur_size)*max_p/sum,-beta);
        for(int i=0;i<n;++i){
            f32 u=rng.next_f32()*sum;
            f32 cumsum=0; int idx=0;
            for(int j=0;j<cur_size;++j){cumsum+=priorities[j];if(cumsum>=u){idx=j;break;}}
            batch.indices[i]=idx;
            batch.transitions[i]=buf[idx];
            f32 p=priorities[idx]/sum;
            batch.weights[i]=std::pow(f32(cur_size)*p,-beta)/max_w;
        }
        return batch;
    }

    void update_priorities(std::span<const int> indices, std::span<const f32> td_errors) noexcept {
        for(int i=0;i<int(indices.size());++i)
            priorities[indices[i]]=std::pow(std::abs(td_errors[i])+eps,alpha);
    }
};

// ─── Reward shaping ───────────────────────────────────────────────────────
// Potential-based reward shaping: F(s,a,s') = gamma*Phi(s') - Phi(s)
[[nodiscard]] inline f32 potential_shaping(
    f32 phi_s, f32 phi_s_prime, f32 gamma=0.99f) noexcept
{
    return gamma*phi_s_prime-phi_s;
}

// Intrinsic curiosity (simple count-based)
struct CountBonus {
    std::unordered_map<int,int> counts;
    f32 beta=0.1f;

    f32 bonus(int state) noexcept {
        int c=++counts[state];
        return beta/std::sqrt(f32(c));
    }
};

// ─── Multi-arm bandit ─────────────────────────────────────────────────────
struct BanditStats {
    std::vector<f32> means;
    std::vector<int> counts;
    int total=0;

    BanditStats(int n):means(n,0.f),counts(n,0){}

    void update(int arm, f32 reward) noexcept {
        ++counts[arm]; ++total;
        means[arm]+=(reward-means[arm])/counts[arm]; // incremental mean
    }

    [[nodiscard]] int ucb1(f32 c=std::sqrt(2.f)) const noexcept {
        int n=int(means.size());
        for(int i=0;i<n;++i) if(counts[i]==0) return i;
        int best=0; f32 bv=-1e30f;
        for(int i=0;i<n;++i){
            f32 v=means[i]+c*std::sqrt(std::log(f32(total))/counts[i]);
            if(v>bv){bv=v;best=i;}
        }
        return best;
    }

    [[nodiscard]] int greedy() const noexcept {
        return int(std::max_element(means.begin(),means.end())-means.begin());
    }
};

// ─── TD-Lambda ───────────────────────────────────────────────────────────
struct TDLambda {
    std::vector<f32> eligibility;
    f32 gamma, lambda;
    int n_states;

    TDLambda(int n,f32 g=0.99f,f32 l=0.9f):eligibility(n,0.f),gamma(g),lambda(l),n_states(n){}

    void update(std::vector<f32>& V, int s, int s_next, f32 reward, f32 alpha) noexcept {
        f32 delta=reward+gamma*V[s_next]-V[s];
        eligibility[s]+=1.f;
        for(int i=0;i<n_states;++i){
            V[i]+=alpha*delta*eligibility[i];
            eligibility[i]*=gamma*lambda;
        }
    }

    void reset() noexcept { std::fill(eligibility.begin(),eligibility.end(),0.f); }
};

// ─── Policy evaluation (DP) ───────────────────────────────────────────────
// Iterative policy evaluation for finite MDP
[[nodiscard]] inline std::vector<f32> policy_evaluation(
    int n_states,
    std::function<std::vector<std::pair<int,f32>>(int,int)> transitions, // (s,a)->[(s',p*r)]
    std::function<int(int)> policy,
    f32 gamma=0.99f, f32 tol=1e-6f, int max_iter=1000) noexcept
{
    std::vector<f32> V(n_states,0.f);
    for(int it=0;it<max_iter;++it){
        f32 delta=0;
        for(int s=0;s<n_states;++s){
            int a=policy(s);
            f32 v=0;
            for(auto [sp,pr]:transitions(s,a)) v+=pr+gamma*V[sp];
            delta=max(delta,std::abs(v-V[s]));
            V[s]=v;
        }
        if(delta<tol) break;
    }
    return V;
}

} // namespace qs::math::ml
