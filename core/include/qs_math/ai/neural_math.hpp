#pragma once
// qs_math/neural_math.hpp — Forward/backprop primitives, activations, batch norm, attention (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <span>
#include <random>
#include <functional>

namespace qs {

// ─── Activation functions ─────────────────────────────────────────────────────

[[nodiscard]] inline f32 relu(f32 x) { return std::max(0.f, x); }
[[nodiscard]] inline f32 relu_grad(f32 x) { return x > 0.f ? 1.f : 0.f; }
[[nodiscard]] inline f32 leaky_relu(f32 x, f32 alpha=0.01f) { return x>=0.f?x:alpha*x; }
[[nodiscard]] inline f32 leaky_relu_grad(f32 x, f32 alpha=0.01f) { return x>=0.f?1.f:alpha; }
[[nodiscard]] inline f32 elu(f32 x, f32 alpha=1.f) { return x>=0.f?x:alpha*(std::exp(x)-1.f); }
[[nodiscard]] inline f32 elu_grad(f32 x, f32 alpha=1.f) { return x>=0.f?1.f:elu(x,alpha)+alpha; }
[[nodiscard]] inline f32 sigmoid(f32 x) { return 1.f/(1.f+std::exp(-x)); }
[[nodiscard]] inline f32 sigmoid_grad(f32 x) { f32 s=sigmoid(x); return s*(1.f-s); }
[[nodiscard]] inline f32 tanh_act(f32 x) { return std::tanh(x); }
[[nodiscard]] inline f32 tanh_grad(f32 x) { f32 t=std::tanh(x); return 1.f-t*t; }
[[nodiscard]] inline f32 silu(f32 x) { return x*sigmoid(x); }  // Swish
[[nodiscard]] inline f32 silu_grad(f32 x) { f32 s=sigmoid(x); return s*(1.f+x*(1.f-s)); }
[[nodiscard]] inline f32 gelu(f32 x) {
    return 0.5f*x*(1.f+std::tanh(0.7978845608f*(x+0.044715f*x*x*x)));
}
[[nodiscard]] inline f32 softplus(f32 x) { return std::log(1.f+std::exp(x)); }
[[nodiscard]] inline f32 mish(f32 x) { return x*std::tanh(softplus(x)); }

// Softmax (in-place)
inline void softmax(std::span<f32> x) {
    f32 max_v = *std::max_element(x.begin(), x.end());
    f32 sum = 0.f;
    for (auto& v : x) { v = std::exp(v - max_v); sum += v; }
    for (auto& v : x) v /= sum;
}

// Log-softmax
inline void log_softmax(std::span<f32> x) {
    f32 max_v = *std::max_element(x.begin(), x.end());
    f32 sum = 0.f;
    for (auto v : x) sum += std::exp(v - max_v);
    f32 log_sum = std::log(sum);
    for (auto& v : x) v = v - max_v - log_sum;
}

// ─── Dense layer ──────────────────────────────────────────────────────────────

struct DenseLayer {
    std::vector<f32> W;    // [out x in]
    std::vector<f32> b;    // [out]
    u32 in_features{}, out_features{};

    void init(u32 in, u32 out, u64 seed=42) {
        in_features=in; out_features=out;
        W.resize(in*out); b.assign(out, 0.f);
        // He initialization
        f32 std_dev = std::sqrt(2.f/f32(in));
        std::mt19937 rng(u32(seed));
        std::normal_distribution<f32> dist(0.f, std_dev);
        for (auto& w : W) w = dist(rng);
    }

    // Forward: y = W*x + b
    void forward(std::span<const f32> x, std::span<f32> y) const {
        for (u32 o=0;o<out_features;++o) {
            f32 s = b[o];
            for (u32 i=0;i<in_features;++i) s += W[o*in_features+i]*x[i];
            y[o] = s;
        }
    }

    // Backward: given dL/dy, compute dL/dx, dL/dW, dL/db
    void backward(
        std::span<const f32> x, std::span<const f32> dy,
        std::span<f32> dx, std::span<f32> dW, std::span<f32> db) const
    {
        for (u32 i=0;i<in_features;++i) dx[i]=0.f;
        for (u32 o=0;o<out_features;++o) {
            db[o] += dy[o];
            for (u32 i=0;i<in_features;++i) {
                dW[o*in_features+i] += dy[o]*x[i];
                dx[i] += W[o*in_features+i]*dy[o];
            }
        }
    }
};

// ─── Batch normalization ──────────────────────────────────────────────────────

struct BatchNorm {
    std::vector<f32> gamma, beta; // learnable scale + shift
    std::vector<f32> running_mean, running_var;
    f32 eps{1e-5f}, momentum{0.1f};
    u32 features{};

    void init(u32 n) {
        features=n;
        gamma.assign(n,1.f); beta.assign(n,0.f);
        running_mean.assign(n,0.f); running_var.assign(n,1.f);
    }

    // Forward (training mode, batch_size x features)
    void forward_train(
        std::span<const f32> x, std::span<f32> y,
        u32 batch_size,
        std::span<f32> mean_out, std::span<f32> var_out,
        std::span<f32> xhat_out) const
    {
        for (u32 f=0;f<features;++f) {
            f32 mean=0.f;
            for (u32 n=0;n<batch_size;++n) mean+=x[n*features+f];
            mean/=f32(batch_size);
            mean_out[f]=mean;
            f32 var=0.f;
            for (u32 n=0;n<batch_size;++n) {
                f32 d=x[n*features+f]-mean; var+=d*d;
            }
            var/=f32(batch_size);
            var_out[f]=var;
            f32 inv_std=1.f/std::sqrt(var+eps);
            for (u32 n=0;n<batch_size;++n) {
                f32 xh=(x[n*features+f]-mean)*inv_std;
                xhat_out[n*features+f]=xh;
                y[n*features+f]=gamma[f]*xh+beta[f];
            }
        }
    }

    // Inference mode
    void forward_infer(std::span<const f32> x, std::span<f32> y) const {
        for (u32 n=0;n<u32(x.size()/features);++n)
        for (u32 f=0;f<features;++f) {
            f32 xh=(x[n*features+f]-running_mean[f])/std::sqrt(running_var[f]+eps);
            y[n*features+f]=gamma[f]*xh+beta[f];
        }
    }
};

// ─── Layer normalization ──────────────────────────────────────────────────────

struct LayerNorm {
    std::vector<f32> gamma, beta;
    f32 eps{1e-5f};
    u32 features{};

    void init(u32 n) {
        features=n; gamma.assign(n,1.f); beta.assign(n,0.f);
    }

    void forward(std::span<const f32> x, std::span<f32> y) const {
        f32 mean=0.f;
        for (u32 i=0;i<features;++i) mean+=x[i];
        mean/=f32(features);
        f32 var=0.f;
        for (u32 i=0;i<features;++i){f32 d=x[i]-mean; var+=d*d;}
        var/=f32(features);
        f32 inv_std=1.f/std::sqrt(var+eps);
        for (u32 i=0;i<features;++i)
            y[i]=gamma[i]*(x[i]-mean)*inv_std+beta[i];
    }
};

// ─── Scaled dot-product attention ────────────────────────────────────────────

// QKV attention: Q[L x d], K[S x d], V[S x dv] -> out[L x dv]
inline void scaled_dot_product_attention(
    std::span<const f32> Q, std::span<const f32> K, std::span<const f32> V,
    std::span<f32> out,
    u32 L, u32 S, u32 d, u32 dv)
{
    f32 scale = 1.f/std::sqrt(f32(d));
    std::vector<f32> scores(L*S);
    // scores = Q @ K^T * scale
    for (u32 i=0;i<L;++i)
    for (u32 j=0;j<S;++j) {
        f32 s=0.f;
        for (u32 k=0;k<d;++k) s+=Q[i*d+k]*K[j*d+k];
        scores[i*S+j]=s*scale;
    }
    // Softmax per row
    for (u32 i=0;i<L;++i) {
        std::span<f32> row{scores.data()+i*S, S};
        softmax(row);
    }
    // out = scores @ V
    for (u32 i=0;i<L;++i)
    for (u32 k=0;k<dv;++k) {
        f32 s=0.f;
        for (u32 j=0;j<S;++j) s+=scores[i*S+j]*V[j*dv+k];
        out[i*dv+k]=s;
    }
}

// ─── Convolution (1D) ─────────────────────────────────────────────────────────

inline void conv1d(
    std::span<const f32> x, std::span<const f32> kernel,
    std::span<f32> y, u32 stride=1, u32 pad=0)
{
    u32 Lx=u32(x.size()), Lk=u32(kernel.size());
    u32 Ly=u32(y.size());
    for (u32 i=0;i<Ly;++i) {
        f32 s=0.f;
        for (u32 k=0;k<Lk;++k) {
            i32 xi=i32(i*stride+k)-i32(pad);
            if (xi>=0&&xi<i32(Lx)) s+=x[u32(xi)]*kernel[k];
        }
        y[i]=s;
    }
}

// ─── Dropout ─────────────────────────────────────────────────────────────────

inline void dropout(std::span<f32> x, f32 p, u64 seed) {
    Pcg32 rng{seed};
    f32 scale = 1.f/(1.f-p);
    for (auto& v : x) {
        if (rng.next_f32() < p) v=0.f;
        else v*=scale;
    }
}

// ─── Embedding ────────────────────────────────────────────────────────────────

struct Embedding {
    std::vector<f32> table; // [vocab x dim]
    u32 vocab{}, dim{};

    void init(u32 v, u32 d, u64 seed=0) {
        vocab=v; dim=d; table.resize(v*d);
        std::mt19937 rng(u32(seed));
        std::normal_distribution<f32> dist(0.f,0.02f);
        for (auto& w : table) w=dist(rng);
    }

    void lookup(u32 token, std::span<f32> out) const {
        for (u32 i=0;i<dim&&i<out.size();++i) out[i]=table[token*dim+i];
    }
};

// ─── RNN cell ─────────────────────────────────────────────────────────────────

struct GRUCell {
    // W_z, W_r, W_h, U_z, U_r, U_h, b_z, b_r, b_h
    u32 input_size{}, hidden_size{};
    std::vector<f32> W; // [3*hidden x input]
    std::vector<f32> U; // [3*hidden x hidden]
    std::vector<f32> b; // [3*hidden]

    void init(u32 in, u32 hid) {
        input_size=in; hidden_size=hid;
        W.assign(3*hid*in, 0.f); U.assign(3*hid*hid, 0.f);
        b.assign(3*hid, 0.f);
    }

    void forward(
        std::span<const f32> x, std::span<const f32> h_prev,
        std::span<f32> h_next) const
    {
        u32 H=hidden_size, I=input_size;
        auto lin=[&](u32 gate_off, std::span<const f32> xv, std::span<const f32> hv)->std::vector<f32>{
            std::vector<f32> r(H);
            for (u32 o=0;o<H;++o) {
                f32 s=b[gate_off*H+o];
                for (u32 i=0;i<I;++i) s+=W[(gate_off*H+o)*I+i]*xv[i];
                for (u32 j=0;j<H;++j) s+=U[(gate_off*H+o)*H+j]*hv[j];
                r[o]=s;
            }
            return r;
        };
        auto z_gate=lin(0,x,h_prev); for(auto&v:z_gate) v=sigmoid(v);
        auto r_gate=lin(1,x,h_prev); for(auto&v:r_gate) v=sigmoid(v);
        std::vector<f32> rh(H); for(u32 i=0;i<H;++i) rh[i]=r_gate[i]*h_prev[i];
        auto n_gate=lin(2,x,rh); for(auto&v:n_gate) v=std::tanh(v);
        for (u32 i=0;i<H;++i)
            h_next[i]=(1.f-z_gate[i])*n_gate[i]+z_gate[i]*h_prev[i];
    }
};

} // namespace qs
