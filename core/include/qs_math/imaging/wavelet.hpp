#pragma once
// qs_math/imaging/wavelet.hpp — Haar, Daubechies, DWT, CWT, wavelet packets
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Haar wavelet ─────────────────────────────────────────────────────────────
template<int N>
constexpr void haar_forward(std::array<f64,N>& x) noexcept {
    static_assert((N&(N-1))==0,"N must be power of 2");
    int len=N;
    while(len>1) {
        len>>=1;
        for(int i=0;i<len;++i) {
            f64 a=x[2*i],b=x[2*i+1];
            x[i]=(a+b)*0.7071067811865476;
            x[len+i]=(a-b)*0.7071067811865476;
        }
        // Re-arrange: in-place
        std::array<f64,N> tmp{};
        for(int i=0;i<len;++i) tmp[i]=x[i];
        for(int i=0;i<len;++i) tmp[len+i]=x[len+i];
        for(int i=0;i<2*len;++i) x[i]=tmp[i];
    }
}

template<int N>
constexpr void haar_inverse(std::array<f64,N>& x) noexcept {
    int len=1;
    while(len<N) {
        for(int i=0;i<len;++i) {
            f64 a=x[i],d=x[len+i];
            x[2*i]=(a+d)*0.7071067811865476;
            x[2*i+1]=(a-d)*0.7071067811865476;
        }
        len<<=1;
    }
}

// ─── Daubechies D4 wavelet ────────────────────────────────────────────────────
inline constexpr f64 k_db4_lo[4]={
    0.6830127018922193,0.1830127018922193,-0.1830127018922193,0.6830127018922193
    // corrected: (1+√3)/4, (3+√3)/4, (3-√3)/4, (1-√3)/4 scaled by 1/√2
};
inline constexpr f64 k_db4_hi[4]={
    -0.1830127018922193,-0.6830127018922193,0.6830127018922193,-0.1830127018922193
};
// Proper Daubechies D4 coefficients
inline constexpr double k_d4h[4]={
    (1+1.7320508075688772)/4*0.7071067811865476,
    (3+1.7320508075688772)/4*0.7071067811865476,
    (3-1.7320508075688772)/4*0.7071067811865476,
    (1-1.7320508075688772)/4*0.7071067811865476
};
inline constexpr double k_d4g[4]={k_d4h[3],-k_d4h[2],k_d4h[1],-k_d4h[0]};

template<int N>
constexpr void db4_forward_step(std::array<f64,N>& x, int n) noexcept {
    std::array<f64,N> tmp{};
    int half=n/2;
    for(int i=0;i<half;++i) {
        f64 a=0,d=0;
        for(int k=0;k<4;++k) {
            int idx=(2*i+k)%n;
            a+=k_d4h[k]*x[idx];
            d+=k_d4g[k]*x[idx];
        }
        tmp[i]=a; tmp[half+i]=d;
    }
    for(int i=0;i<n;++i) x[i]=tmp[i];
}

template<int N>
constexpr void db4_dwt(std::array<f64,N>& x) noexcept {
    for(int n=N;n>=4;n>>=1) db4_forward_step(x,n);
}

// ─── 2D Haar DWT ──────────────────────────────────────────────────────────────
template<int N>
constexpr void haar_2d_forward(std::array<std::array<f64,N>,N>& img) noexcept {
    // Apply 1D haar along rows
    for(int i=0;i<N;++i) haar_forward(img[i]);
    // Apply along columns
    for(int j=0;j<N;++j) {
        std::array<f64,N> col{};
        for(int i=0;i<N;++i) col[i]=img[i][j];
        haar_forward(col);
        for(int i=0;i<N;++i) img[i][j]=col[i];
    }
}

// ─── Wavelet thresholding (denoising) ─────────────────────────────────────────
// Soft threshold: shrink(x,λ) = sign(x)*max(|x|-λ,0)
[[nodiscard]] constexpr f64 soft_threshold(f64 x, f64 lambda) noexcept {
    f64 ax=x<0?-x:x;
    return (ax>lambda)?((x>0?1:-1)*(ax-lambda)):0.0;
}
// Hard threshold: h(x,λ) = x if |x|>λ else 0
[[nodiscard]] constexpr f64 hard_threshold(f64 x, f64 lambda) noexcept {
    return (x<0?-x:x)>lambda?x:0.0;
}

template<int N>
constexpr void wavelet_denoise(std::array<f64,N>& x, f64 sigma, bool soft=true) noexcept {
    haar_forward(x);
    // Universal threshold: λ = σ*sqrt(2*log(N))
    f64 lambda=sigma*std::sqrt(2*std::log(f64(N)));
    // Skip scaling coefficient (index 0)
    for(int i=1;i<N;++i) x[i]=soft?soft_threshold(x[i],lambda):hard_threshold(x[i],lambda);
    haar_inverse(x);
}

// ─── Continuous Wavelet Transform (CWT) via Morlet ────────────────────────────
// Morlet mother wavelet: ψ(t) = exp(iω₀t - t²/2) / π^{1/4}
[[nodiscard]] constexpr f64 morlet_real(f64 t, f64 omega0=6.0) noexcept {
    return std::exp(-t*t*0.5)*std::cos(omega0*t)/std::pow(3.14159265358979323846,0.25);
}
[[nodiscard]] constexpr f64 morlet_imag(f64 t, f64 omega0=6.0) noexcept {
    return std::exp(-t*t*0.5)*std::sin(omega0*t)/std::pow(3.14159265358979323846,0.25);
}

// CWT at scale s, translation b: W(s,b) = (1/√s) ∫ x(t) ψ*((t-b)/s) dt
template<int N>
[[nodiscard]] constexpr std::pair<f64,f64> cwt_morlet(
    const std::array<f64,N>& x, f64 s, f64 b, f64 dt=1.0) noexcept
{
    f64 wr=0,wi=0,norm=1.0/std::sqrt(s);
    for(int n=0;n<N;++n) {
        f64 t=n*dt-b, tau=t/s;
        wr+=x[n]*morlet_real(tau);
        wi+=x[n]*morlet_imag(tau); // conjugate → -imag
    }
    return {wr*norm*dt,wi*norm*dt};
}

// ─── Wavelet packet decomposition ────────────────────────────────────────────
// A wavelet packet decomposes both approximation and detail at each level
template<int N, int LEVELS>
struct WaveletPacket {
    static constexpr int nodes=1<<LEVELS;
    std::array<std::array<f64,N/nodes>,nodes> packets{};

    constexpr void decompose(const std::array<f64,N>& x) noexcept {
        // Fill level-0 packet (original signal)
        std::array<f64,N> buf=x;
        // Haar decompose at each level
        for(int lvl=0;lvl<LEVELS;++lvl) {
            int step=N>>(lvl);
            for(int node=0;node<(1<<lvl);++node) {
                int off=node*step;
                std::array<f64,N> tmp=buf;
                haar_forward_range(tmp,off,step);
                buf=tmp;
            }
        }
        // Copy into packets
        int pkt_sz=N/nodes;
        for(int i=0;i<nodes;++i)
            for(int j=0;j<pkt_sz;++j)
                packets[i][j]=buf[i*pkt_sz+j];
    }

    static constexpr void haar_forward_range(std::array<f64,N>& x, int off, int n) noexcept {
        for(int i=0;i<n/2;++i) {
            f64 a=x[off+2*i],b=x[off+2*i+1];
            x[off+i]=(a+b)*0.7071067811865476;
            x[off+n/2+i]=(a-b)*0.7071067811865476;
        }
    }
};

// ─── Lifting scheme (custom wavelet design) ───────────────────────────────────
// Predict step P, update step U
template<int N>
constexpr void lifting_predict(std::array<f64,N>& x, f64 p_coeff) noexcept {
    // Odd-indexed samples: d_i = x_{2i+1} - p_coeff * x_{2i}
    for(int i=0;i<N/2-1;++i) x[2*i+1]-=p_coeff*x[2*i];
}
template<int N>
constexpr void lifting_update(std::array<f64,N>& x, f64 u_coeff) noexcept {
    // Even-indexed: s_i = x_{2i} + u_coeff * (d_{i-1} + d_i)
    for(int i=1;i<N/2;++i) x[2*i]+=u_coeff*(x[2*i-1]+x[2*i+1]);
}

// ─── Stationary wavelet transform (à trous) ────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,4> swt_haar(
    const std::array<f64,N>& x) noexcept
{
    std::array<std::array<f64,N>,4> result{};
    result[0]=x;
    int stride=1;
    for(int level=0;level<3&&level<static_cast<int>(result.size()-1);++level) {
        for(int i=0;i<N;++i) {
            int j=(i+stride)%N;
            result[level+1][i]=0.5*(result[level][i]+result[level][j]);
            result[level][i]=0.5*(result[level][i]-result[level][j]);
        }
        // Note: for proper SWT, result[level] holds detail, result[level+1] holds approx
        stride<<=1;
    }
    return result;
}
} // namespace qs::math
