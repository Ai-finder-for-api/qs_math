#pragma once
// qs_math/reinforcement_learning_math.hpp — Q-learning, policy gradient, value functions (C++20)
#include "scalar_types.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <array>
#include <numeric>
#include <functional>
#include <deque>

namespace qs {

// ─── Value function basics ────────────────────────────────────────────────────

// Bellman equation: V(s) = R(s) + gamma * max_a Q(s,a)
[[nodiscard]] inline f32 bellman_update(
    f32 reward, f32 next_value, f32 gamma=0.99f)
{
    return reward + gamma * next_value;
}

// TD(0) update: V(s) += alpha * (R + gamma*V(s') - V(s))
inline void td0_update(f32& V_s, f32 reward, f32 V_s_next,
    f32 alpha=0.1f, f32 gamma=0.99f)
{
    V_s += alpha * (reward + gamma * V_s_next - V_s);
}

// TD error (temporal difference)
[[nodiscard]] inline f32 td_error(f32 V_s, f32 reward, f32 V_s_next,
    f32 gamma=0.99f)
{
    return reward + gamma * V_s_next - V_s;
}

// n-step return
[[nodiscard]] inline f32 n_step_return(
    std::span<const f32> rewards, f32 V_last, f32 gamma=0.99f)
{
    f32 G=V_last, discount=1.f;
    for(i32 t=i32(rewards.size())-1;t>=0;--t){
        G=rewards[t]+gamma*G;
        discount*=gamma;
    }
    return G;
}

// ─── Q-table ─────────────────────────────────────────────────────────────────

struct QTable {
    std::vector<f32> Q; // [n_states * n_actions]
    u32 n_states{}, n_actions{};
    f32 alpha{0.1f};   // learning rate
    f32 gamma{0.99f};  // discount factor

    void init(u32 s, u32 a, f32 init_val=0.f){
        n_states=s; n_actions=a;
        Q.assign(s*a, init_val);
    }

    [[nodiscard]] f32& at(u32 state, u32 action) { return Q[state*n_actions+action]; }
    [[nodiscard]] f32  at(u32 state, u32 action) const { return Q[state*n_actions+action]; }

    [[nodiscard]] u32 greedy_action(u32 state) const {
        u32 best=0; f32 best_q=at(state,0);
        for(u32 a=1;a<n_actions;++a) if(at(state,a)>best_q){best_q=at(state,a);best=a;}
        return best;
    }

    [[nodiscard]] f32 max_q(u32 state) const {
        f32 m=at(state,0);
        for(u32 a=1;a<n_actions;++a) m=std::max(m,at(state,a));
        return m;
    }

    // Q-learning update
    void update(u32 s, u32 a, f32 reward, u32 s_next, bool terminal=false){
        f32 target=terminal?reward:reward+gamma*max_q(s_next);
        at(s,a)+=alpha*(target-at(s,a));
    }

    // SARSA update (on-policy)
    void sarsa_update(u32 s, u32 a, f32 reward, u32 s_next, u32 a_next, bool terminal=false){
        f32 target=terminal?reward:reward+gamma*at(s_next,a_next);
        at(s,a)+=alpha*(target-at(s,a));
    }
};

// ─── Epsilon-greedy policy ────────────────────────────────────────────────────

struct EpsilonGreedy {
    f32 epsilon{1.0f};
    f32 epsilon_min{0.01f};
    f32 epsilon_decay{0.995f};

    [[nodiscard]] u32 select(u32 greedy_action, u32 n_actions, Pcg32& rng) const {
        if(rng.next_f32()<epsilon) return u32(rng.next_f32()*f32(n_actions));
        return greedy_action;
    }

    void decay(){ epsilon=std::max(epsilon_min,epsilon*epsilon_decay); }
};

// Boltzmann / softmax policy
[[nodiscard]] inline u32 softmax_action(
    std::span<const f32> q_values, f32 temperature, Pcg32& rng)
{
    std::vector<f32> probs(q_values.size());
    f32 max_q=*std::max_element(q_values.begin(),q_values.end());
    f32 Z=0.f;
    for(u32 i=0;i<u32(q_values.size());++i){
        probs[i]=std::exp((q_values[i]-max_q)/temperature);
        Z+=probs[i];
    }
    f32 r=rng.next_f32()*Z, cum=0.f;
    for(u32 i=0;i<u32(probs.size());++i){
        cum+=probs[i]; if(r<cum) return i;
    }
    return u32(probs.size())-1;
}

// ─── Replay buffer ────────────────────────────────────────────────────────────

struct Transition {
    u32 state{}, action{}, next_state{};
    f32 reward{};
    bool terminal{};
};

struct ReplayBuffer {
    std::deque<Transition> buffer;
    u32 max_size{10000};

    void push(const Transition& t){
        buffer.push_back(t);
        if(buffer.size()>max_size) buffer.pop_front();
    }

    [[nodiscard]] std::vector<Transition> sample(u32 batch_size, Pcg32& rng) const {
        std::vector<Transition> batch;
        u32 n=u32(buffer.size());
        for(u32 i=0;i<batch_size&&i<n;++i){
            u32 idx=u32(rng.next_f32()*f32(n));
            batch.push_back(buffer[idx]);
        }
        return batch;
    }

    [[nodiscard]] u32 size() const { return u32(buffer.size()); }
};

// ─── Policy gradient helpers ──────────────────────────────────────────────────

// REINFORCE gradient estimate: G_t * grad log pi(a|s)
[[nodiscard]] inline std::vector<f32> returns_from_episode(
    std::span<const f32> rewards, f32 gamma=0.99f)
{
    u32 T=u32(rewards.size());
    std::vector<f32> G(T,0.f);
    f32 running=0.f;
    for(i32 t=i32(T)-1;t>=0;--t){
        running=rewards[t]+gamma*running;
        G[t]=running;
    }
    return G;
}

// Normalize returns for stability
inline void normalize_returns(std::span<f32> G){
    f32 mean=0.f; for(f32 g:G) mean+=g; mean/=f32(G.size());
    f32 var=0.f; for(f32 g:G) var+=(g-mean)*(g-mean); var/=f32(G.size());
    f32 std2=std::sqrt(var+1e-8f);
    for(auto& g:G) g=(g-mean)/std2;
}

// Generalized advantage estimation (GAE)
[[nodiscard]] inline std::vector<f32> gae(
    std::span<const f32> rewards,
    std::span<const f32> values,
    f32 gamma=0.99f, f32 lam=0.95f)
{
    u32 T=u32(std::min(rewards.size(),values.size()-1));
    std::vector<f32> adv(T,0.f);
    f32 gae_val=0.f;
    for(i32 t=i32(T)-1;t>=0;--t){
        f32 delta=rewards[t]+gamma*values[t+1]-values[t];
        gae_val=delta+gamma*lam*gae_val;
        adv[t]=gae_val;
    }
    return adv;
}

// ─── UCB1 (multi-armed bandit) ────────────────────────────────────────────────

struct UCB1 {
    std::vector<f32> Q;     // estimated values
    std::vector<u32> N;     // pull counts
    u32 total_pulls{0};
    f32 c{2.f};             // exploration constant

    void init(u32 n_arms){ Q.assign(n_arms,0.f); N.assign(n_arms,0); }

    [[nodiscard]] u32 select() const {
        for(u32 i=0;i<u32(N.size());++i) if(!N[i]) return i;
        u32 best=0; f32 best_ucb=-1e30f;
        for(u32 i=0;i<u32(Q.size());++i){
            f32 ucb=Q[i]+c*std::sqrt(std::log(f32(total_pulls)+1.f)/f32(N[i]));
            if(ucb>best_ucb){best_ucb=ucb;best=i;}
        }
        return best;
    }

    void update(u32 arm, f32 reward){
        ++N[arm]; ++total_pulls;
        Q[arm]+=(reward-Q[arm])/f32(N[arm]);
    }
};

// Thompson sampling (Bernoulli bandit)
struct ThompsonBeta {
    std::vector<std::pair<f32,f32>> params; // (alpha, beta) per arm

    void init(u32 n){ params.assign(n,{1.f,1.f}); }

    [[nodiscard]] u32 select(Pcg32& rng) const {
        u32 best=0; f32 best_sample=-1.f;
        for(u32 i=0;i<u32(params.size());++i){
            // Beta distribution sample via Gamma
            auto gamma_sample=[&](f32 a)->f32{
                f32 x=0.f;
                for(u32 k=0;k<u32(std::ceil(a));++k)
                    x+=-std::log(std::max(1e-10f,rng.next_f32()));
                return x;
            };
            f32 a=gamma_sample(params[i].first);
            f32 b=gamma_sample(params[i].second);
            f32 s=a/(a+b+1e-10f);
            if(s>best_sample){best_sample=s;best=i;}
        }
        return best;
    }

    void update(u32 arm, bool success){
        if(success) params[arm].first+=1.f;
        else params[arm].second+=1.f;
    }
};

// ─── Reward shaping ───────────────────────────────────────────────────────────

// Potential-based reward shaping: F(s,s') = gamma*Phi(s') - Phi(s)
[[nodiscard]] inline f32 shaped_reward(
    f32 raw_reward, f32 phi_s, f32 phi_s_next, f32 gamma=0.99f)
{
    return raw_reward + gamma*phi_s_next - phi_s;
}

// Intrinsic curiosity (prediction error as reward)
[[nodiscard]] inline f32 curiosity_reward(
    std::span<const f32> predicted_next_state,
    std::span<const f32> actual_next_state,
    f32 scale=0.1f)
{
    f32 err=0.f;
    u32 n=u32(std::min(predicted_next_state.size(),actual_next_state.size()));
    for(u32 i=0;i<n;++i){
        f32 d=predicted_next_state[i]-actual_next_state[i];
        err+=d*d;
    }
    return scale*err/f32(n+1);
}

} // namespace qs
