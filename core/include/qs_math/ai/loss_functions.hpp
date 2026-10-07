#pragma once
// qs_math/loss_functions.hpp — Cross-entropy, focal, Huber, triplet, contrastive, SSIM (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <numeric>

namespace qs {

// ─── Classification losses ────────────────────────────────────────────────────

// Cross-entropy: -sum(y_true * log(y_pred))
[[nodiscard]] inline f32 cross_entropy(
    std::span<const f32> y_pred, std::span<const f32> y_true, f32 eps=1e-7f)
{
    f32 loss=0.f;
    for (u32 i=0;i<u32(std::min(y_pred.size(),y_true.size()));++i)
        loss -= y_true[i]*std::log(std::clamp(y_pred[i],eps,1.f-eps));
    return loss;
}

// Binary cross-entropy
[[nodiscard]] inline f32 binary_cross_entropy(f32 y_pred, f32 y_true, f32 eps=1e-7f) {
    f32 p=std::clamp(y_pred,eps,1.f-eps);
    return -(y_true*std::log(p)+(1.f-y_true)*std::log(1.f-p));
}

// Focal loss (handles class imbalance)
[[nodiscard]] inline f32 focal_loss(f32 y_pred, f32 y_true, f32 gamma=2.f, f32 alpha=0.25f, f32 eps=1e-7f) {
    f32 p=std::clamp(y_pred,eps,1.f-eps);
    f32 pt=y_true>0.5f?p:1.f-p;
    f32 at=y_true>0.5f?alpha:1.f-alpha;
    return -at*std::pow(1.f-pt,gamma)*std::log(pt);
}

// Sparse cross-entropy (integer labels)
[[nodiscard]] inline f32 sparse_cross_entropy(
    std::span<const f32> logits, u32 label, f32 eps=1e-7f)
{
    // Compute softmax + CE together (numerically stable)
    f32 max_l = *std::max_element(logits.begin(),logits.end());
    f32 sum=0.f;
    for (auto v : logits) sum+=std::exp(v-max_l);
    f32 log_softmax_label = logits[label]-max_l-std::log(sum);
    return -log_softmax_label;
}

// KL divergence: sum(p * log(p/q))
[[nodiscard]] inline f32 kl_divergence(
    std::span<const f32> p, std::span<const f32> q, f32 eps=1e-7f)
{
    f32 kl=0.f;
    for (u32 i=0;i<u32(std::min(p.size(),q.size()));++i)
        if (p[i]>eps)
            kl+=p[i]*std::log(p[i]/std::max(q[i],eps));
    return kl;
}

// ─── Regression losses ────────────────────────────────────────────────────────

[[nodiscard]] inline f32 mse_loss(std::span<const f32> pred, std::span<const f32> target) {
    f32 s=0.f; u32 N=u32(std::min(pred.size(),target.size()));
    for (u32 i=0;i<N;++i){f32 d=pred[i]-target[i];s+=d*d;}
    return s/f32(N);
}

[[nodiscard]] inline f32 mae_loss(std::span<const f32> pred, std::span<const f32> target) {
    f32 s=0.f; u32 N=u32(std::min(pred.size(),target.size()));
    for (u32 i=0;i<N;++i) s+=std::abs(pred[i]-target[i]);
    return s/f32(N);
}

// Huber loss (smooth L1)
[[nodiscard]] inline f32 huber_loss(f32 pred, f32 target, f32 delta=1.f) {
    f32 err=std::abs(pred-target);
    return err<=delta ? 0.5f*err*err : delta*(err-0.5f*delta);
}

[[nodiscard]] inline f32 huber_loss_batch(
    std::span<const f32> pred, std::span<const f32> target, f32 delta=1.f)
{
    f32 s=0.f; u32 N=u32(std::min(pred.size(),target.size()));
    for (u32 i=0;i<N;++i) s+=huber_loss(pred[i],target[i],delta);
    return s/f32(N);
}

// Log-cosh loss
[[nodiscard]] inline f32 logcosh_loss(f32 pred, f32 target) {
    f32 d=pred-target;
    return std::log(std::cosh(d)+1e-12f);
}

// Quantile / Pinball loss
[[nodiscard]] inline f32 quantile_loss(f32 pred, f32 target, f32 q) {
    f32 err=target-pred;
    return err>=0.f ? q*err : (q-1.f)*err;
}

// ─── Metric learning losses ───────────────────────────────────────────────────

// Contrastive loss (Hadsell et al.)
[[nodiscard]] inline f32 contrastive_loss(
    std::span<const f32> a, std::span<const f32> b,
    bool same_class, f32 margin=1.f)
{
    f32 d=0.f;
    for (u32 i=0;i<u32(std::min(a.size(),b.size()));++i){f32 dd=a[i]-b[i]; d+=dd*dd;}
    d=std::sqrt(d);
    if (same_class) return 0.5f*d*d;
    return 0.5f*std::max(0.f, margin-d)*std::max(0.f, margin-d);
}

// Triplet loss
[[nodiscard]] inline f32 triplet_loss(
    std::span<const f32> anchor,
    std::span<const f32> positive,
    std::span<const f32> negative,
    f32 margin=1.f)
{
    f32 d_ap=0.f, d_an=0.f;
    u32 N=u32(anchor.size());
    for (u32 i=0;i<N;++i) {
        f32 da=anchor[i]-positive[i]; d_ap+=da*da;
        f32 dn=anchor[i]-negative[i]; d_an+=dn*dn;
    }
    return std::max(0.f, std::sqrt(d_ap)-std::sqrt(d_an)+margin);
}

// N-pair loss (multi-class triplet)
[[nodiscard]] inline f32 npairs_loss(
    std::span<const f32> anchor, std::span<const f32> positive,
    const std::vector<std::span<const f32>>& negatives, f32 scale=64.f)
{
    u32 D=u32(anchor.size());
    // sim(a, p)
    f32 ap=0.f;
    for (u32 i=0;i<D;++i) ap+=anchor[i]*positive[i];
    // logit = scale * (sim(a,n) - sim(a,p)) for each negative
    f32 sum_exp=0.f;
    for (const auto& neg : negatives) {
        f32 an=0.f;
        for (u32 i=0;i<D;++i) an+=anchor[i]*neg[i];
        sum_exp+=std::exp(scale*(an-ap));
    }
    return std::log(1.f+sum_exp);
}

// ─── Image quality losses ─────────────────────────────────────────────────────

// L2 pixel loss
[[nodiscard]] inline f32 pixel_l2(
    std::span<const f32> img_a, std::span<const f32> img_b)
{
    return mse_loss(img_a, img_b);
}

// SSIM (simplified, single channel)
[[nodiscard]] inline f32 ssim(
    std::span<const f32> x, std::span<const f32> y,
    f32 C1=0.0001f, f32 C2=0.0009f)
{
    u32 N=u32(std::min(x.size(),y.size()));
    if (!N) return 1.f;
    f32 mx=0,my=0;
    for (u32 i=0;i<N;++i){mx+=x[i];my+=y[i];}
    mx/=f32(N); my/=f32(N);
    f32 vx=0,vy=0,vxy=0;
    for (u32 i=0;i<N;++i){
        vx+=(x[i]-mx)*(x[i]-mx);
        vy+=(y[i]-my)*(y[i]-my);
        vxy+=(x[i]-mx)*(y[i]-my);
    }
    vx/=f32(N); vy/=f32(N); vxy/=f32(N);
    return (2.f*mx*my+C1)*(2.f*vxy+C2)/((mx*mx+my*my+C1)*(vx+vy+C2));
}

[[nodiscard]] inline f32 ssim_loss(std::span<const f32> x, std::span<const f32> y) {
    return 1.f - ssim(x, y);
}

// ─── Perceptual loss placeholder ──────────────────────────────────────────────

// Feature-space L2 (VGG-style): compute externally, combine here
[[nodiscard]] inline f32 perceptual_loss(
    std::span<const f32> feat_pred, std::span<const f32> feat_target,
    f32 lambda_feat=1.f)
{
    return lambda_feat * mse_loss(feat_pred, feat_target);
}

// ─── Gradient penalty (WGAN-GP) ──────────────────────────────────────────────

[[nodiscard]] inline f32 gradient_penalty(
    std::span<const f32> grad, f32 target_norm=1.f)
{
    f32 norm=0.f;
    for (auto v : grad) norm+=v*v;
    norm=std::sqrt(norm);
    f32 d=norm-target_norm;
    return d*d;
}

} // namespace qs
