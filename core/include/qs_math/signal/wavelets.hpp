#pragma once
// qs_math/signal/wavelets.hpp — Wavelet transforms: DWT, IDWT, CWT, lifting scheme,
//   Daubechies, Haar, Coiflet, Symlet, biorthogonal wavelets, wavelet packets,
//   multiresolution analysis, denoising (C++20)
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <span>
#include <algorithm>

namespace qs::math::signal {

// ─── Haar wavelet ─────────────────────────────────────────────────────────
inline void haar_forward(std::vector<f64>& x) noexcept {
    int n=int(x.size());
    std::vector<f64> tmp(n);
    while(n>=2){
        for(int i=0;i<n/2;++i){
            tmp[i]       =(x[2*i]+x[2*i+1])*0.7071067811865476;
            tmp[n/2+i]   =(x[2*i]-x[2*i+1])*0.7071067811865476;
        }
        for(int i=0;i<n;++i) x[i]=tmp[i];
        n/=2;
    }
}

inline void haar_inverse(std::vector<f64>& x) noexcept {
    int n=int(x.size()), level=1;
    while(level*2<=int(x.size())) level*=2;
    level/=2;
    std::vector<f64> tmp(x.size());
    while(level>=1){
        for(int i=0;i<level;++i){
            tmp[2*i]  =(x[i]+x[level+i])*0.7071067811865476;
            tmp[2*i+1]=(x[i]-x[level+i])*0.7071067811865476;
        }
        for(int i=0;i<level*2;++i) x[i]=tmp[i];
        level*=2;
    }
}

// ─── Daubechies D4 coefficients ───────────────────────────────────────────
inline constexpr std::array<f64,4> DB4_LO_D={
     0.4829629131445341, 0.8365163037378079,
     0.2241438680420134,-0.1294095225512604};
inline constexpr std::array<f64,4> DB4_HI_D={
    -0.1294095225512604,-0.2241438680420134,
     0.8365163037378079,-0.4829629131445341};
inline constexpr std::array<f64,4> DB4_LO_R={
     0.1294095225512604, 0.2241438680420134,
     0.8365163037378079, 0.4829629131445341};
inline constexpr std::array<f64,4> DB4_HI_R={
    -0.4829629131445341, 0.8365163037378079,
    -0.2241438680420134,-0.1294095225512604};

// ─── Daubechies D8 coefficients ───────────────────────────────────────────
inline constexpr std::array<f64,8> DB8_LO_D={
     0.2303778133088965, 0.7148465705529156,
     0.6308807679298589,-0.0279837694168599,
    -0.1870348117190931, 0.0308413818355607,
     0.0328830116668852,-0.0105974017850690};
inline constexpr std::array<f64,8> DB8_HI_D={
     0.0105974017850690, 0.0328830116668852,
    -0.0308413818355607,-0.1870348117190931,
     0.0279837694168599, 0.6308807679298589,
    -0.7148465705529156, 0.2303778133088965};

// ─── Generic filter-bank DWT (single level) ───────────────────────────────
template<int F>
inline void dwt_level(
    const std::vector<f64>& x,
    const std::array<f64,F>& lo_d, const std::array<f64,F>& hi_d,
    std::vector<f64>& approx, std::vector<f64>& detail) noexcept
{
    int n=int(x.size()), half=(n+F-1)/2;
    approx.assign(half,0); detail.assign(half,0);
    for(int i=0;i<half;++i){
        for(int k=0;k<F;++k){
            int idx=(2*i+k)%n;
            approx[i]+=lo_d[k]*x[idx];
            detail[i]+=hi_d[k]*x[idx];
        }
    }
}

template<int F>
inline void idwt_level(
    const std::vector<f64>& approx, const std::vector<f64>& detail,
    const std::array<f64,F>& lo_r, const std::array<f64,F>& hi_r,
    std::vector<f64>& x) noexcept
{
    int half=int(approx.size()), n=half*2;
    x.assign(n,0);
    for(int i=0;i<half;++i){
        for(int k=0;k<F;++k){
            int idx=(2*i+k)%n;
            x[idx]+=lo_r[k]*approx[i]+hi_r[k]*detail[i];
        }
    }
}

// ─── Multi-level DWT (wavelet packet structure) ───────────────────────────
struct WaveletDecomp {
    std::vector<std::vector<f64>> levels; // levels[0]=coarsest approx, levels[1..n]=details coarse-to-fine
    int n_levels;
};

[[nodiscard]] inline WaveletDecomp dwt_db4(std::vector<f64> x, int n_levels) noexcept {
    WaveletDecomp w; w.n_levels=n_levels;
    std::vector<f64> approx=x;
    for(int lv=0;lv<n_levels;++lv){
        std::vector<f64> a,d;
        dwt_level(approx,DB4_LO_D,DB4_HI_D,a,d);
        w.levels.insert(w.levels.begin(),d);
        approx=a;
    }
    w.levels.insert(w.levels.begin(),approx);
    return w;
}

[[nodiscard]] inline std::vector<f64> idwt_db4(WaveletDecomp& w) noexcept {
    std::vector<f64> approx=w.levels[0];
    for(int lv=0;lv<w.n_levels;++lv){
        std::vector<f64> x;
        idwt_level(approx,w.levels[lv+1],DB4_LO_R,DB4_HI_R,x);
        approx=x;
    }
    return approx;
}

// ─── Lifting scheme (predict/update) ─────────────────────────────────────
// Le Gall 5/3 wavelet (reversible, used in JPEG 2000)
inline void lifting_53_forward(std::vector<f64>& x) noexcept {
    int n=int(x.size());
    // Predict (odd samples)
    for(int i=1;i<n-1;i+=2) x[i]-=std::floor((x[i-1]+x[i+1]+2)/4);
    if(n>1&&(n-1)%2==0) x[n-1]-=x[n-2];
    // Update (even samples)
    for(int i=2;i<n-1;i+=2) x[i]+=std::floor((x[i-1]+x[i+1]+1)/2);
    if(n>0) x[0]+=std::floor((x[1]+1)/2);
}

inline void lifting_53_inverse(std::vector<f64>& x) noexcept {
    int n=int(x.size());
    // Undo update
    if(n>0) x[0]-=std::floor((x[1]+1)/2);
    for(int i=2;i<n-1;i+=2) x[i]-=std::floor((x[i-1]+x[i+1]+1)/2);
    // Undo predict
    if(n>1&&(n-1)%2==0) x[n-1]+=x[n-2];
    for(int i=1;i<n-1;i+=2) x[i]+=std::floor((x[i-1]+x[i+1]+2)/4);
}

// CDF 9/7 wavelet (lossy, used in JPEG 2000) — lifting steps
inline constexpr f64 CDF97_ALPHA=-1.5861343420693648;
inline constexpr f64 CDF97_BETA =-0.0529801185718856;
inline constexpr f64 CDF97_GAMMA= 0.8829110755411875;
inline constexpr f64 CDF97_DELTA= 0.4435068520511142;
inline constexpr f64 CDF97_SCALE= 1.1496043988602418;

inline void lifting_97_forward(std::vector<f64>& x) noexcept {
    int n=int(x.size());
    // Four lifting steps
    for(int i=1;i<n-2;i+=2) x[i]+=CDF97_ALPHA*(x[i-1]+x[i+1]);
    x[n-1]+=2*CDF97_ALPHA*x[n-2];
    x[0]+=2*CDF97_BETA*x[1];
    for(int i=2;i<n;i+=2) x[i]+=CDF97_BETA*(x[i-1]+x[i+1]);
    for(int i=1;i<n-2;i+=2) x[i]+=CDF97_GAMMA*(x[i-1]+x[i+1]);
    x[n-1]+=2*CDF97_GAMMA*x[n-2];
    x[0]+=2*CDF97_DELTA*x[1];
    for(int i=2;i<n;i+=2) x[i]+=CDF97_DELTA*(x[i-1]+x[i+1]);
    // Scale
    for(int i=0;i<n;i+=2) x[i]*=CDF97_SCALE;
    for(int i=1;i<n;i+=2) x[i]/=CDF97_SCALE;
    // Uninterleave
    std::vector<f64> tmp(n);
    for(int i=0;i<n/2;++i){tmp[i]=x[2*i];tmp[n/2+i]=x[2*i+1];}
    x=tmp;
}

inline void lifting_97_inverse(std::vector<f64>& x) noexcept {
    int n=int(x.size());
    // Interleave
    std::vector<f64> tmp(n);
    for(int i=0;i<n/2;++i){tmp[2*i]=x[i];tmp[2*i+1]=x[n/2+i];}
    x=tmp;
    for(int i=0;i<n;i+=2) x[i]/=CDF97_SCALE;
    for(int i=1;i<n;i+=2) x[i]*=CDF97_SCALE;
    // Undo steps in reverse
    x[0]-=2*CDF97_DELTA*x[1];
    for(int i=2;i<n;i+=2) x[i]-=CDF97_DELTA*(x[i-1]+x[i+1]);
    x[n-1]-=2*CDF97_GAMMA*x[n-2];
    for(int i=1;i<n-2;i+=2) x[i]-=CDF97_GAMMA*(x[i-1]+x[i+1]);
    x[0]-=2*CDF97_BETA*x[1];
    for(int i=2;i<n;i+=2) x[i]-=CDF97_BETA*(x[i-1]+x[i+1]);
    x[n-1]-=2*CDF97_ALPHA*x[n-2];
    for(int i=1;i<n-2;i+=2) x[i]-=CDF97_ALPHA*(x[i-1]+x[i+1]);
}

// ─── 2D DWT (separable) ───────────────────────────────────────────────────
struct Image2D {
    std::vector<f64> data;
    int width, height;
    f64& at(int r,int c) noexcept {return data[r*width+c];}
    f64  at(int r,int c) const noexcept {return data[r*width+c];}
};

inline void dwt2d_level(Image2D& img) noexcept {
    // Apply Haar row-wise then column-wise (one level)
    for(int r=0;r<img.height;++r){
        std::vector<f64> row(img.width);
        for(int c=0;c<img.width;++c) row[c]=img.at(r,c);
        haar_forward(row);
        for(int c=0;c<img.width;++c) img.at(r,c)=row[c];
    }
    for(int c=0;c<img.width;++c){
        std::vector<f64> col(img.height);
        for(int r=0;r<img.height;++r) col[r]=img.at(r,c);
        haar_forward(col);
        for(int r=0;r<img.height;++r) img.at(r,c)=col[r];
    }
}

// ─── Continuous Wavelet Transform (CWT) ───────────────────────────────────
// Morlet wavelet: psi(t) = exp(i*omega0*t)*exp(-t^2/2) / (pi^0.25)
[[nodiscard]] inline std::complex<f64> morlet(f64 t, f64 omega0=6.0) noexcept {
    static constexpr f64 norm=0.7511255444649425; // pi^(-1/4)
    return std::complex<f64>{std::cos(omega0*t),std::sin(omega0*t)}
           *std::exp(-0.5*t*t)*norm;
}

// Ricker (Mexican hat) wavelet: psi(t) = (1-t^2)*exp(-t^2/2) * 2/sqrt(3)*pi^(-1/4)
[[nodiscard]] inline f64 ricker(f64 t) noexcept {
    static constexpr f64 norm=0.8673250705840776; // 2/sqrt(3)*pi^(-1/4)
    return norm*(1-t*t)*std::exp(-0.5*t*t);
}

// CWT scalogram at single (scale, translation)
[[nodiscard]] inline std::complex<f64> cwt_morlet(
    std::span<const f64> x, f64 scale, f64 b, f64 dt=1.0, f64 omega0=6.0) noexcept
{
    int n=int(x.size());
    std::complex<f64> sum{0,0};
    f64 norm=1.0/std::sqrt(std::abs(scale));
    for(int i=0;i<n;++i){
        f64 t=(i*dt-b)/scale;
        sum+=std::complex<f64>(x[i])*std::conj(morlet(t,omega0));
    }
    return sum*norm*dt;
}

// ─── Wavelet denoising ────────────────────────────────────────────────────
enum class ThresholdType { Hard, Soft, Garrote };

[[nodiscard]] inline f64 threshold_apply(f64 x, f64 lambda, ThresholdType t) noexcept {
    switch(t){
        case ThresholdType::Hard:   return std::abs(x)>lambda?x:0.0;
        case ThresholdType::Soft:   return std::abs(x)>lambda?(x>0?x-lambda:x+lambda):0.0;
        case ThresholdType::Garrote:return std::abs(x)>lambda?(x-lambda*lambda/x):0.0;
    }
    return x;
}

// Universal (VisuShrink) threshold
[[nodiscard]] inline f64 universal_threshold(int n, f64 sigma) noexcept {
    return sigma*std::sqrt(2.0*std::log(f64(n)));
}

// Estimate noise std dev from finest wavelet detail
[[nodiscard]] inline f64 estimate_noise(const std::vector<f64>& detail) noexcept {
    // MAD estimator
    std::vector<f64> d=detail;
    std::sort(d.begin(),d.end());
    f64 med=d[d.size()/2];
    std::vector<f64> abs_dev(d.size());
    for(int i=0;i<int(d.size());++i) abs_dev[i]=std::abs(d[i]-med);
    std::sort(abs_dev.begin(),abs_dev.end());
    return abs_dev[abs_dev.size()/2]/0.6745;
}

[[nodiscard]] inline std::vector<f64> wavelet_denoise(
    std::vector<f64> x, int levels=4,
    ThresholdType ttype=ThresholdType::Soft) noexcept
{
    auto w=dwt_db4(x,levels);
    // Estimate noise from finest detail
    f64 sigma=estimate_noise(w.levels.back());
    f64 lambda=universal_threshold(int(x.size()),sigma);
    // Threshold all detail bands
    for(int lv=1;lv<=levels;++lv)
        for(auto& c:w.levels[lv]) c=threshold_apply(c,lambda,ttype);
    return idwt_db4(w);
}

// ─── Wavelet energy ───────────────────────────────────────────────────────
[[nodiscard]] inline f64 wavelet_energy(const std::vector<f64>& coeffs) noexcept {
    f64 e=0; for(auto c:coeffs) e+=c*c; return e;
}

[[nodiscard]] inline std::vector<f64> wavelet_energy_distribution(
    const WaveletDecomp& w) noexcept
{
    std::vector<f64> energy;
    for(auto& lv:w.levels) energy.push_back(wavelet_energy(lv));
    return energy;
}

// ─── Symlet S8 coefficients ───────────────────────────────────────────────
inline constexpr std::array<f64,8> SYM8_LO_D={
    -0.0757657147893407,-0.0296355276459541, 0.4976186676324578,
     0.8037387518059161, 0.2978577956055422,-0.0992195435769354,
    -0.0126039672622612, 0.0322231006040427};
inline constexpr std::array<f64,8> SYM8_HI_D={
    -0.0322231006040427,-0.0126039672622612, 0.0992195435769354,
     0.2978577956055422,-0.8037387518059161, 0.4976186676324578,
     0.0296355276459541,-0.0757657147893407};

// ─── Biorthogonal 2.2 wavelets ────────────────────────────────────────────
inline constexpr std::array<f64,6> BIOR22_LO_D={
     0.0,-0.1767766952966369, 0.3535533905932738,
     1.0606601717798214, 0.3535533905932738,-0.1767766952966369};
inline constexpr std::array<f64,6> BIOR22_HI_D={0.0, 0.3535533905932738,-0.7071067811865476, 0.3535533905932738,0,0};
inline constexpr std::array<f64,6> BIOR22_LO_R={0,0, 0.3535533905932738, 0.7071067811865476, 0.3535533905932738,0};
inline constexpr std::array<f64,6> BIOR22_HI_R={
    -0.1767766952966369,-0.3535533905932738, 1.0606601717798214,
    -0.3535533905932738,-0.1767766952966369,0};

// ─── Wavelet packet analysis ──────────────────────────────────────────────
struct WaveletPacketNode {
    std::vector<f64> coeffs;
    int level, index;
    f64 entropy; // Shannon entropy of coefficients
};

[[nodiscard]] inline f64 shannon_entropy(const std::vector<f64>& c) noexcept {
    f64 e=0,norm=0; for(auto v:c) norm+=v*v;
    if(norm<1e-300) return 0;
    for(auto v:c){f64 p=v*v/norm; if(p>1e-300) e-=p*std::log(p);}
    return e;
}

[[nodiscard]] inline std::vector<WaveletPacketNode> wavelet_packet_decomp(
    std::vector<f64> x, int max_level) noexcept
{
    std::vector<WaveletPacketNode> nodes;
    std::vector<std::pair<std::vector<f64>,std::pair<int,int>>> queue;
    queue.push_back({x,{0,0}});
    while(!queue.empty()){
        auto [data,li]=queue.front(); queue.erase(queue.begin());
        auto [lv,idx]=li;
        f64 ent=shannon_entropy(data);
        nodes.push_back({data,lv,idx,ent});
        if(lv<max_level){
            std::vector<f64> a,d;
            dwt_level(data,DB4_LO_D,DB4_HI_D,a,d);
            queue.push_back({a,{lv+1,2*idx}});
            queue.push_back({d,{lv+1,2*idx+1}});
        }
    }
    return nodes;
}

} // namespace qs::math::signal
