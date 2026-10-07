#pragma once
// qs_math/ai/neural_layers.hpp — Dense, conv, attention, normalization layer math
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Activation functions ─────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 relu(f64 x)    noexcept { return x>0?x:0; }
[[nodiscard]] constexpr f64 leaky_relu(f64 x,f64 a=0.01) noexcept { return x>0?x:a*x; }
[[nodiscard]] constexpr f64 elu(f64 x,f64 a=1.0) noexcept { return x>0?x:a*(std::exp(x)-1); }
[[nodiscard]] constexpr f64 selu(f64 x) noexcept {
    constexpr f64 lam=1.0507009873554804,alpha=1.6732632423543772;
    return lam*(x>0?x:alpha*(std::exp(x)-1));
}
[[nodiscard]] constexpr f64 gelu(f64 x) noexcept {
    return 0.5*x*(1+std::tanh(0.7978845608028654*(x+0.044715*x*x*x)));
}
[[nodiscard]] constexpr f64 swish(f64 x,f64 beta=1.0) noexcept {
    return x/(1+std::exp(-beta*x));
}
[[nodiscard]] constexpr f64 mish(f64 x) noexcept {
    return x*std::tanh(std::log(1+std::exp(x)));
}
[[nodiscard]] constexpr f64 sigmoid(f64 x) noexcept { return 1.0/(1+std::exp(-x)); }
[[nodiscard]] constexpr f64 softplus(f64 x) noexcept { return std::log(1+std::exp(x)); }
[[nodiscard]] constexpr f64 hard_sigmoid(f64 x) noexcept {
    return x<-2.5?0:x>2.5?1:(x*0.2+0.5);
}
[[nodiscard]] constexpr f64 hard_swish(f64 x) noexcept {
    return x*hard_sigmoid(x);
}

// Derivatives
[[nodiscard]] constexpr f64 d_relu(f64 x) noexcept { return x>0?1:0; }
[[nodiscard]] constexpr f64 d_sigmoid(f64 x) noexcept { f64 s=sigmoid(x); return s*(1-s); }
[[nodiscard]] constexpr f64 d_tanh(f64 x) noexcept { f64 t=std::tanh(x); return 1-t*t; }
[[nodiscard]] constexpr f64 d_gelu(f64 x) noexcept {
    f64 t=std::tanh(0.7978845608028654*(x+0.044715*x*x*x));
    f64 sech2=1-t*t;
    return 0.5*(1+t)+0.5*x*sech2*0.7978845608028654*(1+3*0.044715*x*x);
}

// ─── Softmax ──────────────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> softmax(const std::array<f64,N>& x) noexcept {
    f64 mx=x[0]; for(auto v:x) if(v>mx) mx=v;
    std::array<f64,N> e{};
    f64 s=0; for(int i=0;i<N;++i){e[i]=std::exp(x[i]-mx);s+=e[i];}
    for(auto& v:e) v/=s;
    return e;
}
template<int N>
[[nodiscard]] constexpr f64 log_softmax(const std::array<f64,N>& x, int k) noexcept {
    f64 mx=x[0]; for(auto v:x) if(v>mx) mx=v;
    f64 s=0; for(auto v:x) s+=std::exp(v-mx);
    return x[k]-mx-std::log(s);
}

// ─── Dense layer ──────────────────────────────────────────────────────────────
template<int IN, int OUT>
[[nodiscard]] constexpr std::array<f64,OUT> dense_forward(
    const std::array<f64,IN>& x,
    const std::array<std::array<f64,IN>,OUT>& W,
    const std::array<f64,OUT>& b) noexcept
{
    std::array<f64,OUT> y{};
    for(int i=0;i<OUT;++i){
        y[i]=b[i];
        for(int j=0;j<IN;++j) y[i]+=W[i][j]*x[j];
    }
    return y;
}

// ─── Batch normalization ──────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> batch_norm(
    const std::array<f64,N>& x,
    const std::array<f64,N>& gamma, const std::array<f64,N>& beta,
    f64 eps=1e-5) noexcept
{
    f64 mu=0; for(auto v:x) mu+=v; mu/=N;
    f64 var=0; for(auto v:x) var+=(v-mu)*(v-mu); var/=N;
    std::array<f64,N> y{};
    f64 inv=1.0/std::sqrt(var+eps);
    for(int i=0;i<N;++i) y[i]=gamma[i]*(x[i]-mu)*inv+beta[i];
    return y;
}

// Layer norm (normalize over feature dimension)
template<int N>
[[nodiscard]] constexpr std::array<f64,N> layer_norm(
    const std::array<f64,N>& x,
    const std::array<f64,N>& gamma, const std::array<f64,N>& beta,
    f64 eps=1e-5) noexcept
{
    return batch_norm(x,gamma,beta,eps); // same math, different semantics
}

// ─── 1D convolution ──────────────────────────────────────────────────────────
template<int L, int KS>
[[nodiscard]] constexpr std::array<f64,L-KS+1> conv1d(
    const std::array<f64,L>& x, const std::array<f64,KS>& kernel) noexcept
{
    constexpr int OUT=L-KS+1;
    std::array<f64,OUT> y{};
    for(int i=0;i<OUT;++i)
        for(int k=0;k<KS;++k) y[i]+=x[i+k]*kernel[k];
    return y;
}

// 2D convolution (valid padding)
template<int H, int W, int KH, int KW>
[[nodiscard]] constexpr std::array<std::array<f64,W-KW+1>,H-KH+1> conv2d(
    const std::array<std::array<f64,W>,H>& x,
    const std::array<std::array<f64,KW>,KH>& k) noexcept
{
    constexpr int OH=H-KH+1, OW=W-KW+1;
    std::array<std::array<f64,OW>,OH> y{};
    for(int i=0;i<OH;++i) for(int j=0;j<OW;++j)
        for(int ki=0;ki<KH;++ki) for(int kj=0;kj<KW;++kj)
            y[i][j]+=x[i+ki][j+kj]*k[ki][kj];
    return y;
}

// ─── Attention mechanism (scaled dot-product) ─────────────────────────────────
template<int SEQ, int DIM>
[[nodiscard]] constexpr std::array<std::array<f64,DIM>,SEQ> scaled_dot_attention(
    const std::array<std::array<f64,DIM>,SEQ>& Q,
    const std::array<std::array<f64,DIM>,SEQ>& K,
    const std::array<std::array<f64,DIM>,SEQ>& V) noexcept
{
    f64 scale=1.0/std::sqrt(f64(DIM));
    // Compute attention scores S = QK^T / sqrt(d)
    std::array<std::array<f64,SEQ>,SEQ> S{};
    for(int i=0;i<SEQ;++i) for(int j=0;j<SEQ;++j) {
        f64 dot=0; for(int d=0;d<DIM;++d) dot+=Q[i][d]*K[j][d];
        S[i][j]=dot*scale;
    }
    // Softmax per row
    for(int i=0;i<SEQ;++i) {
        f64 mx=S[i][0]; for(auto v:S[i]) if(v>mx) mx=v;
        f64 s=0; for(auto& v:S[i]){v=std::exp(v-mx);s+=v;}
        for(auto& v:S[i]) v/=s;
    }
    // Output = S*V
    std::array<std::array<f64,DIM>,SEQ> out{};
    for(int i=0;i<SEQ;++i) for(int d=0;d<DIM;++d)
        for(int j=0;j<SEQ;++j) out[i][d]+=S[i][j]*V[j][d];
    return out;
}

// ─── Positional encoding (sinusoidal) ─────────────────────────────────────────
template<int SEQ, int DIM>
[[nodiscard]] constexpr std::array<std::array<f64,DIM>,SEQ> positional_encoding() noexcept {
    std::array<std::array<f64,DIM>,SEQ> pe{};
    for(int pos=0;pos<SEQ;++pos)
        for(int i=0;i<DIM;i+=2) {
            f64 freq=std::pow(10000.0,f64(i)/DIM);
            pe[pos][i]=std::sin(pos/freq);
            if(i+1<DIM) pe[pos][i+1]=std::cos(pos/freq);
        }
    return pe;
}

// ─── Loss functions ───────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 mse_loss(const std::array<f64,N>& pred,
                                      const std::array<f64,N>& target) noexcept {
    f64 s=0; for(int i=0;i<N;++i){f64 d=pred[i]-target[i];s+=d*d;} return s/N;
}
template<int N>
[[nodiscard]] constexpr f64 mae_loss(const std::array<f64,N>& pred,
                                      const std::array<f64,N>& target) noexcept {
    f64 s=0; for(int i=0;i<N;++i) s+=(pred[i]>target[i]?pred[i]-target[i]:target[i]-pred[i]);
    return s/N;
}
template<int N>
[[nodiscard]] constexpr f64 cross_entropy_loss(const std::array<f64,N>& pred,
                                                  const std::array<f64,N>& target) noexcept {
    f64 s=0;
    for(int i=0;i<N;++i) if(target[i]>1e-12) s-=target[i]*std::log(pred[i]+1e-12);
    return s;
}
template<int N>
[[nodiscard]] constexpr f64 huber_loss(const std::array<f64,N>& pred,
                                         const std::array<f64,N>& target, f64 delta=1.0) noexcept {
    f64 s=0;
    for(int i=0;i<N;++i) {
        f64 d=pred[i]-target[i],a=d<0?-d:d;
        s+=a<=delta?0.5*d*d:delta*(a-0.5*delta);
    }
    return s/N;
}
// Focal loss (RetinaNet)
[[nodiscard]] constexpr f64 focal_loss(f64 p_t, f64 gamma=2.0, f64 alpha=0.25) noexcept {
    return -alpha*std::pow(1-p_t,gamma)*std::log(p_t+1e-12);
}

// ─── Dropout ─────────────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> dropout_forward(
    const std::array<f64,N>& x, const std::array<bool,N>& mask, f64 p) noexcept {
    std::array<f64,N> y{};
    f64 scale=1.0/(1-p);
    for(int i=0;i<N;++i) y[i]=mask[i]?x[i]*scale:0;
    return y;
}

// ─── Weight initialization ────────────────────────────────────────────────────
// He initialization scale: sqrt(2/fan_in)
[[nodiscard]] constexpr f64 he_init_scale(int fan_in) noexcept {
    return std::sqrt(2.0/fan_in);
}
// Glorot/Xavier: sqrt(6/(fan_in+fan_out))
[[nodiscard]] constexpr f64 glorot_init_scale(int fan_in, int fan_out) noexcept {
    return std::sqrt(6.0/(fan_in+fan_out));
}
// LeCun: sqrt(1/fan_in)
[[nodiscard]] constexpr f64 lecun_init_scale(int fan_in) noexcept {
    return std::sqrt(1.0/fan_in);
}
} // namespace qs::math
