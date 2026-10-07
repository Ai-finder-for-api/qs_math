#pragma once
// qs_math/optimizer_math.hpp — SGD, Adam, AdaGrad, RMSProp, LAMB, learning rate schedules (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>

namespace qs {

// ─── Parameter update state ───────────────────────────────────────────────────

struct SGDState {
    std::vector<f32> velocity;  // momentum buffer
    f32 lr{0.01f};
    f32 momentum{0.9f};
    f32 weight_decay{0.f};
    f32 nesterov{false};

    void init(u32 n) { velocity.assign(n, 0.f); }

    void step(std::span<f32> params, std::span<const f32> grads) {
        for (u32 i=0;i<u32(params.size());++i) {
            f32 g = grads[i] + weight_decay*params[i];
            velocity[i] = momentum*velocity[i] + g;
            f32 update = nesterov ? g + momentum*velocity[i] : velocity[i];
            params[i] -= lr*update;
        }
    }
};

struct AdamState {
    std::vector<f32> m, v; // 1st and 2nd moment
    u64  t{0};
    f32  lr{1e-3f};
    f32  beta1{0.9f}, beta2{0.999f};
    f32  eps{1e-8f};
    f32  weight_decay{0.f};
    bool amsgrad{false};
    std::vector<f32> v_hat_max; // for AMSGrad

    void init(u32 n) { m.assign(n,0.f); v.assign(n,0.f); v_hat_max.assign(n,0.f); }

    void step(std::span<f32> params, std::span<const f32> grads) {
        ++t;
        f32 bc1 = 1.f-std::pow(beta1,f32(t));
        f32 bc2 = 1.f-std::pow(beta2,f32(t));
        f32 step_size = lr * std::sqrt(bc2)/bc1;
        for (u32 i=0;i<u32(params.size());++i) {
            f32 g = grads[i] + weight_decay*params[i];
            m[i] = beta1*m[i]+(1.f-beta1)*g;
            v[i] = beta2*v[i]+(1.f-beta2)*g*g;
            f32 vd = v[i];
            if (amsgrad) { v_hat_max[i]=std::max(v_hat_max[i],vd); vd=v_hat_max[i]; }
            params[i] -= step_size*m[i]/(std::sqrt(vd)+eps);
        }
    }
};

struct AdamWState : AdamState {
    void step_adamw(std::span<f32> params, std::span<const f32> grads) {
        // AdamW: decouple weight decay from gradient
        ++t;
        f32 bc1=1.f-std::pow(beta1,f32(t));
        f32 bc2=1.f-std::pow(beta2,f32(t));
        f32 step_size=lr*std::sqrt(bc2)/bc1;
        for (u32 i=0;i<u32(params.size());++i) {
            params[i] *= (1.f-lr*weight_decay); // decoupled WD
            m[i]=beta1*m[i]+(1.f-beta1)*grads[i];
            v[i]=beta2*v[i]+(1.f-beta2)*grads[i]*grads[i];
            params[i]-=step_size*m[i]/(std::sqrt(v[i])+eps);
        }
    }
};

struct AdaGradState {
    std::vector<f32> G; // sum of squared gradients
    f32 lr{0.01f}, eps{1e-8f};

    void init(u32 n) { G.assign(n,0.f); }

    void step(std::span<f32> params, std::span<const f32> grads) {
        for (u32 i=0;i<u32(params.size());++i) {
            G[i]+=grads[i]*grads[i];
            params[i]-=lr*grads[i]/(std::sqrt(G[i])+eps);
        }
    }
};

struct RMSPropState {
    std::vector<f32> v, m; // EMA of squared grad, EMA of grad (if centered)
    f32  lr{1e-3f}, alpha{0.99f}, eps{1e-8f};
    f32  momentum{0.f};
    bool centered{false};
    std::vector<f32> mom_buf;

    void init(u32 n) { v.assign(n,0.f); m.assign(n,0.f); mom_buf.assign(n,0.f); }

    void step(std::span<f32> params, std::span<const f32> grads) {
        for (u32 i=0;i<u32(params.size());++i) {
            v[i]=alpha*v[i]+(1.f-alpha)*grads[i]*grads[i];
            f32 denom=v[i];
            if (centered){m[i]=alpha*m[i]+(1.f-alpha)*grads[i]; denom-=m[i]*m[i];}
            f32 update=grads[i]/(std::sqrt(denom)+eps);
            if (momentum>0.f){mom_buf[i]=momentum*mom_buf[i]+update;update=mom_buf[i];}
            params[i]-=lr*update;
        }
    }
};

// ─── LAMB (Layer-wise Adaptive Moments for Batch training) ───────────────────

struct LAMBState {
    AdamState adam;
    f32 clip_global_grad{1.f};

    void init(u32 n) { adam.init(n); }

    void step(std::span<f32> params, std::span<const f32> grads, f32 lr_scale=1.f) {
        ++adam.t;
        f32 bc1=1.f-std::pow(adam.beta1,f32(adam.t));
        f32 bc2=1.f-std::pow(adam.beta2,f32(adam.t));
        f32 p_norm=0.f, u_norm=0.f;
        std::vector<f32> update(params.size());
        for (u32 i=0;i<u32(params.size());++i) {
            adam.m[i]=adam.beta1*adam.m[i]+(1.f-adam.beta1)*grads[i];
            adam.v[i]=adam.beta2*adam.v[i]+(1.f-adam.beta2)*grads[i]*grads[i];
            f32 mh=adam.m[i]/bc1, vh=adam.v[i]/bc2;
            update[i]=mh/(std::sqrt(vh)+adam.eps)+adam.weight_decay*params[i];
            p_norm+=params[i]*params[i];
            u_norm+=update[i]*update[i];
        }
        p_norm=std::sqrt(p_norm); u_norm=std::sqrt(u_norm);
        f32 trust = (p_norm>0.f&&u_norm>0.f) ? p_norm/u_norm : 1.f;
        f32 effective_lr = adam.lr*trust*lr_scale;
        for (u32 i=0;i<u32(params.size());++i) params[i]-=effective_lr*update[i];
    }
};

// ─── Learning rate schedules ──────────────────────────────────────────────────

[[nodiscard]] inline f32 lr_constant(f32 base_lr, u64) { return base_lr; }

[[nodiscard]] inline f32 lr_step_decay(f32 base_lr, u64 step, u64 decay_every, f32 gamma=0.1f) {
    return base_lr * std::pow(gamma, f32(step/decay_every));
}

[[nodiscard]] inline f32 lr_exponential(f32 base_lr, u64 step, f32 gamma=0.99f) {
    return base_lr * std::pow(gamma, f32(step));
}

[[nodiscard]] inline f32 lr_cosine_annealing(f32 lr_max, f32 lr_min, u64 step, u64 T_max) {
    return lr_min + 0.5f*(lr_max-lr_min)*(1.f+std::cos(k_pi*f32(step)/f32(T_max)));
}

[[nodiscard]] inline f32 lr_warmup_cosine(
    f32 base_lr, u64 step, u64 warmup_steps, u64 total_steps)
{
    if (step < warmup_steps) return base_lr*f32(step)/f32(warmup_steps);
    u64 s=step-warmup_steps, T=total_steps-warmup_steps;
    return base_lr*0.5f*(1.f+std::cos(k_pi*f32(s)/f32(T)));
}

[[nodiscard]] inline f32 lr_one_cycle(
    f32 max_lr, u64 step, u64 total_steps,
    f32 pct_start=0.3f, f32 div_factor=25.f, f32 final_div=1e4f)
{
    f32 init_lr=max_lr/div_factor;
    u64 up=u64(f32(total_steps)*pct_start);
    if (step<up) return init_lr+(max_lr-init_lr)*f32(step)/f32(up);
    f32 p=f32(step-up)/f32(total_steps-up);
    return max_lr-(max_lr-max_lr/final_div)*p;
}

// ─── Gradient clipping ────────────────────────────────────────────────────────

inline f32 clip_grad_norm(std::span<f32> grads, f32 max_norm) {
    f32 norm=0.f;
    for (auto g : grads) norm+=g*g;
    norm=std::sqrt(norm);
    if (norm>max_norm) {
        f32 scale=max_norm/norm;
        for (auto& g : grads) g*=scale;
    }
    return norm;
}

inline void clip_grad_value(std::span<f32> grads, f32 clip_value) {
    for (auto& g : grads) g=std::clamp(g,-clip_value,clip_value);
}

// ─── EMA model averaging ──────────────────────────────────────────────────────

inline void ema_update(
    std::span<const f32> model, std::span<f32> shadow, f32 decay)
{
    for (u32 i=0;i<u32(std::min(model.size(),shadow.size()));++i)
        shadow[i]=decay*shadow[i]+(1.f-decay)*model[i];
}

} // namespace qs
