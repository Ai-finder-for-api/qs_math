#pragma once
// qs_math/ml/neural_math.hpp — Neural network math: activations, loss functions,
//   backprop primitives, attention, positional encoding, batch norm,
//   layer norm, dropout, weight init, embedding ops (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <span>

namespace qs::math::ml {

using Vec1D = std::vector<f32>;
using Mat2D = std::vector<f32>;

[[nodiscard]] inline f32 sigmoid(f32 x) noexcept { return 1.f/(1.f+std::exp(-x)); }
[[nodiscard]] inline f32 sigmoid_deriv(f32 y) noexcept { return y*(1.f-y); }
[[nodiscard]] inline f32 relu(f32 x) noexcept { return x>0.f?x:0.f; }
[[nodiscard]] inline f32 relu_deriv(f32 x) noexcept { return x>0.f?1.f:0.f; }
[[nodiscard]] inline f32 leaky_relu(f32 x, f32 alpha=0.01f) noexcept { return x>0.f?x:alpha*x; }
[[nodiscard]] inline f32 elu(f32 x, f32 alpha=1.f) noexcept {
    return x>=0.f?x:alpha*(std::exp(x)-1.f);
}
[[nodiscard]] inline f32 selu(f32 x) noexcept {
    static constexpr f32 alpha=1.6732632423543772f, scale=1.0507009873554805f;
    return scale*(x>=0.f?x:alpha*(std::exp(x)-1.f));
}
[[nodiscard]] inline f32 gelu(f32 x) noexcept {
    return 0.5f*x*(1.f+std::erf(x*0.7071067811865476f));
}
[[nodiscard]] inline f32 gelu_fast(f32 x) noexcept {
    f32 c=0.7978845608f*(x+0.044715f*x*x*x);
    return 0.5f*x*(1.f+std::tanh(c));
}
[[nodiscard]] inline f32 swish(f32 x) noexcept { return x*sigmoid(x); }
[[nodiscard]] inline f32 mish(f32 x) noexcept {
    return x*std::tanh(std::log1p(std::exp(x)));
}
[[nodiscard]] inline f32 softplus(f32 x) noexcept { return std::log1p(std::exp(x)); }
[[nodiscard]] inline f32 hardtanh(f32 x, f32 lo=-1.f, f32 hi=1.f) noexcept { return clamp(x,lo,hi); }
[[nodiscard]] inline f32 hard_sigmoid(f32 x) noexcept { return clamp(x*0.2f+0.5f,0.f,1.f); }
[[nodiscard]] inline f32 hard_swish(f32 x) noexcept { return x*hard_sigmoid(x); }
[[nodiscard]] inline f32 prelu(f32 x, f32 a) noexcept { return x>=0.f?x:a*x; }
[[nodiscard]] inline f32 rrelu(f32 x, f32 lo=0.125f, f32 hi=0.333f) noexcept {
    f32 a=(lo+hi)*0.5f; return x>=0.f?x:a*x;
}
[[nodiscard]] inline f32 celu(f32 x, f32 alpha=1.f) noexcept {
    return x>=0.f?x:alpha*(std::exp(x/alpha)-1.f);
}

inline void softmax_inplace(Vec1D& x) noexcept {
    f32 mx=*std::max_element(x.begin(),x.end());
    f32 s=0; for(auto& v:x){v=std::exp(v-mx);s+=v;}
    for(auto& v:x) v/=s;
}
inline void log_softmax_inplace(Vec1D& x) noexcept {
    f32 mx=*std::max_element(x.begin(),x.end());
    f32 s=0; for(auto v:x) s+=std::exp(v-mx);
    f32 lse=std::log(s)+mx;
    for(auto& v:x) v-=lse;
}

[[nodiscard]] inline f32 mse_loss(std::span<const f32> p, std::span<const f32> t) noexcept {
    int n=int(p.size()); f32 s=0;
    for(int i=0;i<n;++i){f32 d=p[i]-t[i];s+=d*d;}
    return s/n;
}
[[nodiscard]] inline f32 mae_loss(std::span<const f32> p, std::span<const f32> t) noexcept {
    int n=int(p.size()); f32 s=0;
    for(int i=0;i<n;++i) s+=std::abs(p[i]-t[i]);
    return s/n;
}
[[nodiscard]] inline f32 cross_entropy(std::span<const f32> log_p, int target) noexcept {
    return -log_p[target];
}
[[nodiscard]] inline f32 bce(f32 pred, f32 target) noexcept {
    pred=clamp(pred,1e-7f,1.f-1e-7f);
    return -(target*std::log(pred)+(1-target)*std::log(1-pred));
}
[[nodiscard]] inline f32 huber(f32 pred, f32 tgt, f32 delta=1.f) noexcept {
    f32 e=std::abs(pred-tgt);
    return e<=delta?0.5f*e*e:delta*(e-0.5f*delta);
}
[[nodiscard]] inline f32 kl_div(std::span<const f32> p, std::span<const f32> q) noexcept {
    int n=int(p.size()); f32 kl=0;
    for(int i=0;i<n;++i) if(p[i]>1e-10f) kl+=p[i]*std::log(p[i]/max(q[i],1e-10f));
    return kl;
}
[[nodiscard]] inline f32 focal_loss(f32 pred, f32 tgt, f32 gamma=2.f, f32 alpha=0.25f) noexcept {
    pred=clamp(pred,1e-7f,1.f-1e-7f);
    f32 pt=tgt>0.5f?pred:1-pred, w=tgt>0.5f?alpha:1-alpha;
    return -w*std::pow(1-pt,gamma)*std::log(pt);
}
[[nodiscard]] inline f32 cosine_similarity_loss(std::span<const f32> a, std::span<const f32> b) noexcept {
    int n=int(a.size()); f32 dot=0,na=0,nb=0;
    for(int i=0;i<n;++i){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return 1.f-dot/(std::sqrt(na*nb)+1e-8f);
}

inline void linear_fwd(const Mat2D& W,const Vec1D& b,const Vec1D& x,Vec1D& y,int out,int in) noexcept {
    y.assign(out,0.f);
    for(int i=0;i<out;++i){for(int j=0;j<in;++j) y[i]+=W[i*in+j]*x[j];y[i]+=b[i];}
}
inline void linear_bwd(const Mat2D& W,const Vec1D& x,const Vec1D& dy,Vec1D& dx,Mat2D& dW,Vec1D& db,int out,int in) noexcept {
    dx.assign(in,0.f);dW.assign(out*in,0.f);db=dy;
    for(int i=0;i<out;++i) for(int j=0;j<in;++j){dW[i*in+j]+=dy[i]*x[j];dx[j]+=W[i*in+j]*dy[i];}
}

struct BatchNorm {
    Vec1D gamma,beta,run_mean,run_var;
    f32 eps=1e-5f,mom=0.1f; int F;
    BatchNorm(int f):gamma(f,1.f),beta(f,0.f),run_mean(f,0.f),run_var(f,1.f),F(f){}
    [[nodiscard]] Vec1D fwd(const Vec1D& x,int B,bool train=true) noexcept {
        Vec1D y(B*F);
        Vec1D mean(F,0),var(F,0);
        if(train){
            for(int b=0;b<B;++b) for(int f=0;f<F;++f) mean[f]+=x[b*F+f];
            for(auto& m:mean) m/=B;
            for(int b=0;b<B;++b) for(int f=0;f<F;++f){f32 d=x[b*F+f]-mean[f];var[f]+=d*d;}
            for(auto& v:var) v/=B;
            for(int f=0;f<F;++f){run_mean[f]=(1-mom)*run_mean[f]+mom*mean[f];run_var[f]=(1-mom)*run_var[f]+mom*var[f];}
        } else { mean=run_mean; var=run_var; }
        for(int b=0;b<B;++b) for(int f=0;f<F;++f){
            f32 xh=(x[b*F+f]-mean[f])/std::sqrt(var[f]+eps);
            y[b*F+f]=gamma[f]*xh+beta[f];
        }
        return y;
    }
};

struct LayerNorm {
    Vec1D gamma,beta; f32 eps=1e-5f; int F;
    LayerNorm(int f):gamma(f,1.f),beta(f,0.f),F(f){}
    [[nodiscard]] Vec1D fwd(const Vec1D& x) noexcept {
        f32 m=0,v=0; for(auto vv:x) m+=vv; m/=F;
        for(auto vv:x){f32 d=vv-m;v+=d*d;} v/=F;
        Vec1D y(F);
        for(int i=0;i<F;++i) y[i]=gamma[i]*(x[i]-m)/std::sqrt(v+eps)+beta[i];
        return y;
    }
};

struct RMSNorm {
    Vec1D weight; f32 eps=1e-6f; int F;
    RMSNorm(int f):weight(f,1.f),F(f){}
    [[nodiscard]] Vec1D fwd(const Vec1D& x) noexcept {
        f32 rms=0; for(auto v:x) rms+=v*v; rms=std::sqrt(rms/F+eps);
        Vec1D y(F); for(int i=0;i<F;++i) y[i]=weight[i]*x[i]/rms;
        return y;
    }
};

inline void dropout_fwd(Vec1D& x, f32 p, PCG32& rng, bool train=true) noexcept {
    if(!train) return;
    f32 s=1.f/(1.f-p);
    for(auto& v:x) v=rng.next_f32()>p?v*s:0.f;
}

[[nodiscard]] inline Mat2D attention(
    const Mat2D& Q,const Mat2D& K,const Mat2D& V,
    int S,int dk,int dv,bool causal=false) noexcept
{
    f32 scale=1.f/std::sqrt(f32(dk));
    Mat2D sc(S*S);
    for(int i=0;i<S;++i) for(int j=0;j<S;++j){
        f32 s=0; for(int k=0;k<dk;++k) s+=Q[i*dk+k]*K[j*dk+k];
        sc[i*S+j]=(causal&&j>i)?-1e9f:s*scale;
    }
    for(int i=0;i<S;++i){
        Vec1D row(S); for(int j=0;j<S;++j) row[j]=sc[i*S+j];
        softmax_inplace(row); for(int j=0;j<S;++j) sc[i*S+j]=row[j];
    }
    Mat2D out(S*dv,0.f);
    for(int i=0;i<S;++i) for(int k=0;k<S;++k) for(int j=0;j<dv;++j)
        out[i*dv+j]+=sc[i*S+k]*V[k*dv+j];
    return out;
}

[[nodiscard]] inline Mat2D sinusoidal_pe(int S,int D) noexcept {
    Mat2D pe(S*D,0.f);
    for(int p=0;p<S;++p) for(int i=0;i<D;i+=2){
        f32 ang=f32(p)/std::pow(10000.f,f32(i)/D);
        pe[p*D+i]=std::sin(ang);
        if(i+1<D) pe[p*D+i+1]=std::cos(ang);
    }
    return pe;
}
inline void rope(Vec1D& x,int pos,int D) noexcept {
    for(int i=0;i<D;i+=2){
        f32 theta=f32(pos)/std::pow(10000.f,f32(i)/D);
        f32 c=std::cos(theta),s=std::sin(theta);
        f32 x0=x[i],x1=i+1<D?x[i+1]:0.f;
        x[i]=x0*c-x1*s; if(i+1<D) x[i+1]=x0*s+x1*c;
    }
}

inline void xavier_uniform(Mat2D& W,int fi,int fo,PCG32& rng) noexcept {
    f32 lim=std::sqrt(6.f/f32(fi+fo));
    for(auto& w:W) w=(rng.next_f32()*2.f-1.f)*lim;
}
inline void kaiming_normal(Mat2D& W,int fi,PCG32& rng) noexcept {
    f32 std=std::sqrt(2.f/f32(fi));
    for(auto& w:W){
        f32 u1=max(1e-7f,rng.next_f32()),u2=rng.next_f32();
        w=std*std::sqrt(-2.f*std::log(u1))*std::cos(f32(TWO_PI)*u2);
    }
}
inline void orthogonal_init(Mat2D& W,int r,int c,PCG32& rng) noexcept {
    // Random Gaussian then QR decomposition (simplified: normalize rows)
    for(auto& w:W){
        f32 u=max(1e-7f,rng.next_f32()),v=rng.next_f32();
        w=std::sqrt(-2.f*std::log(u))*std::cos(f32(TWO_PI)*v);
    }
    for(int i=0;i<r;++i){
        f32 n=0; for(int j=0;j<c;++j) n+=W[i*c+j]*W[i*c+j];
        n=std::sqrt(n)+1e-8f; for(int j=0;j<c;++j) W[i*c+j]/=n;
    }
}

[[nodiscard]] inline Vec1D embed_lookup(const Mat2D& T,int V,int D,int tok) noexcept {
    Vec1D out(D); int idx=clamp(tok,0,V-1);
    for(int i=0;i<D;++i) out[i]=T[idx*D+i];
    return out;
}

[[nodiscard]] inline f32 l2_reg(const Vec1D& w,f32 lam) noexcept {
    f32 s=0; for(auto v:w) s+=v*v; return 0.5f*lam*s;
}
inline void l2_grad(const Vec1D& w,Vec1D& g,f32 lam) noexcept {
    for(int i=0;i<int(w.size());++i) g[i]+=lam*w[i];
}
[[nodiscard]] inline f32 l1_reg(const Vec1D& w,f32 lam) noexcept {
    f32 s=0; for(auto v:w) s+=std::abs(v); return lam*s;
}
inline void grad_clip_norm(std::vector<Vec1D*>& gs,f32 mx) noexcept {
    f32 n=0; for(auto* g:gs) for(auto v:*g) n+=v*v; n=std::sqrt(n);
    if(n>mx){f32 s=mx/n; for(auto* g:gs) for(auto& v:*g) v*=s;}
}

[[nodiscard]] inline std::vector<f32> conv1d(
    std::span<const f32> in, std::span<const f32> ker) noexcept
{
    int n=int(in.size()),k=int(ker.size()),ol=n-k+1;
    std::vector<f32> out(ol,0.f);
    for(int i=0;i<ol;++i) for(int j=0;j<k;++j) out[i]+=in[i+j]*ker[j];
    return out;
}

[[nodiscard]] inline std::vector<f32> max_pool1d(
    std::span<const f32> x, int kernel=2, int stride=2) noexcept
{
    int n=int(x.size()), ol=(n-kernel)/stride+1;
    std::vector<f32> out(ol);
    for(int i=0;i<ol;++i){
        f32 mx=-1e30f;
        for(int j=0;j<kernel;++j) mx=max(mx,x[i*stride+j]);
        out[i]=mx;
    }
    return out;
}

[[nodiscard]] inline std::vector<f32> avg_pool1d(
    std::span<const f32> x, int kernel=2, int stride=2) noexcept
{
    int n=int(x.size()), ol=(n-kernel)/stride+1;
    std::vector<f32> out(ol);
    for(int i=0;i<ol;++i){
        f32 s=0; for(int j=0;j<kernel;++j) s+=x[i*stride+j];
        out[i]=s/kernel;
    }
    return out;
}

} // namespace qs::math::ml
