#pragma once
// qs_math/signal/fft.hpp — Cooley-Tukey FFT (radix-2, in-place), real FFT,
//   convolution, correlation, power spectrum, STFT, DCT, DST (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <span>
#include <algorithm>
#include <numeric>

namespace qs::math::signal {

using Cx64 = std::complex<f64>;
using CxVec = std::vector<Cx64>;

// ─── Bit reversal permutation ─────────────────────────────────────────────
[[nodiscard]] inline u32 bit_reverse(u32 x, int bits) noexcept {
    u32 r=0;
    for(int i=0;i<bits;++i){r=(r<<1)|(x&1);x>>=1;}
    return r;
}

inline void bit_reverse_permute(CxVec& a) noexcept {
    int n=int(a.size()), bits=0;
    while((1<<bits)<n) ++bits;
    for(int i=0;i<n;++i){
        int j=int(bit_reverse(u32(i),bits));
        if(j>i) std::swap(a[i],a[j]);
    }
}

// ─── Cooley-Tukey FFT (in-place, radix-2, DIT) ────────────────────────────
inline void fft_inplace(CxVec& a, bool inverse=false) noexcept {
    int n=int(a.size());
    bit_reverse_permute(a);
    for(int len=2;len<=n;len<<=1){
        f64 ang=TWO_PI/len*(inverse?1:-1);
        Cx64 wlen={std::cos(ang),std::sin(ang)};
        for(int i=0;i<n;i+=len){
            Cx64 w={1,0};
            for(int j=0;j<len/2;++j){
                Cx64 u=a[i+j], v=a[i+j+len/2]*w;
                a[i+j]=u+v; a[i+j+len/2]=u-v;
                w*=wlen;
            }
        }
    }
    if(inverse){f64 inv=1.0/n; for(auto& c:a) c*=inv;}
}

[[nodiscard]] inline CxVec fft(CxVec a) noexcept {
    fft_inplace(a,false); return a;
}
[[nodiscard]] inline CxVec ifft(CxVec a) noexcept {
    fft_inplace(a,true); return a;
}

// ─── Next power of two ────────────────────────────────────────────────────
[[nodiscard]] constexpr int next_pow2(int n) noexcept {
    int p=1; while(p<n) p<<=1; return p;
}

// ─── Real FFT (exploits conjugate symmetry) ───────────────────────────────
// Pack real[0..N-1] into complex[0..N/2-1], FFT, unpack
[[nodiscard]] inline CxVec rfft(const std::vector<f64>& x) noexcept {
    int n=int(x.size()); int m=n/2;
    CxVec a(m);
    for(int i=0;i<m;++i) a[i]={x[2*i],x[2*i+1]};
    fft_inplace(a);
    CxVec out(m+1);
    out[0]={(a[0].real()+a[0].imag()),0};
    out[m]={(a[0].real()-a[0].imag()),0};
    for(int k=1;k<m;++k){
        f64 ang=-PI*k/m;
        Cx64 tw={std::cos(ang),std::sin(ang)};
        Cx64 ev=0.5*(a[k]+std::conj(a[m-k]));
        Cx64 od=Cx64(0,-0.5)*(a[k]-std::conj(a[m-k]));
        out[k]=ev+tw*od;
    }
    return out;
}

[[nodiscard]] inline std::vector<f64> irfft(CxVec X, int n) noexcept {
    int m=n/2;
    CxVec a(m);
    for(int k=1;k<m;++k){
        f64 ang=PI*k/m;
        Cx64 tw={std::cos(ang),std::sin(ang)};
        Cx64 ev=0.5*(X[k]+std::conj(X[m-k]));
        Cx64 od=Cx64(0,0.5)*(X[k]-std::conj(X[m-k]));
        a[k]=ev+tw*od;
    }
    a[0]={0.5*(X[0].real()+X[m].real()), 0.5*(X[0].real()-X[m].real())};
    fft_inplace(a,true);
    std::vector<f64> out(n);
    for(int i=0;i<m;++i){out[2*i]=a[i].real();out[2*i+1]=a[i].imag();}
    return out;
}

// ─── Linear convolution via FFT ───────────────────────────────────────────
[[nodiscard]] inline std::vector<f64> convolve(
    const std::vector<f64>& a, const std::vector<f64>& b) noexcept
{
    int na=int(a.size()), nb=int(b.size());
    int n=next_pow2(na+nb-1);
    CxVec fa(n,{0,0}), fb(n,{0,0});
    for(int i=0;i<na;++i) fa[i]={a[i],0};
    for(int i=0;i<nb;++i) fb[i]={b[i],0};
    fft_inplace(fa); fft_inplace(fb);
    for(int i=0;i<n;++i) fa[i]*=fb[i];
    fft_inplace(fa,true);
    std::vector<f64> out(na+nb-1);
    for(int i=0;i<na+nb-1;++i) out[i]=fa[i].real();
    return out;
}

// Cross-correlation
[[nodiscard]] inline std::vector<f64> correlate(
    const std::vector<f64>& a, const std::vector<f64>& b) noexcept
{
    std::vector<f64> br(b.rbegin(),b.rend());
    return convolve(a,br);
}

// Autocorrelation
[[nodiscard]] inline std::vector<f64> autocorrelate(const std::vector<f64>& a) noexcept {
    return correlate(a,a);
}

// ─── Power spectral density ───────────────────────────────────────────────
[[nodiscard]] inline std::vector<f64> power_spectrum(const std::vector<f64>& x) noexcept {
    auto X=rfft(x);
    std::vector<f64> psd(X.size());
    f64 scale=1.0/(x.size()*x.size());
    for(int i=0;i<int(X.size());++i) psd[i]=std::norm(X[i])*scale;
    return psd;
}

// Welch's method (averaged periodogram)
[[nodiscard]] inline std::vector<f64> welch_psd(
    const std::vector<f64>& x, int seg_len, int overlap=0) noexcept
{
    int n=int(x.size()), step=seg_len-overlap;
    int n_seg=(n-seg_len)/step+1;
    if(n_seg<=0) return power_spectrum(x);
    std::vector<f64> psd(seg_len/2+1, 0.0);
    // Hann window
    std::vector<f64> win(seg_len);
    f64 wsum=0;
    for(int i=0;i<seg_len;++i){win[i]=0.5*(1-std::cos(TWO_PI*i/(seg_len-1)));wsum+=win[i]*win[i];}
    for(int s=0;s<n_seg;++s){
        std::vector<f64> seg(seg_len);
        for(int i=0;i<seg_len;++i) seg[i]=x[s*step+i]*win[i];
        auto p=power_spectrum(seg);
        for(int i=0;i<int(psd.size());++i) psd[i]+=p[i];
    }
    f64 scale=1.0/(n_seg*wsum/seg_len);
    for(auto& v:psd) v*=scale;
    return psd;
}

// ─── Short-Time Fourier Transform ─────────────────────────────────────────
struct STFTResult {
    std::vector<CxVec> frames; // [frame][frequency bin]
    int hop, n_fft, n_frames;
};

[[nodiscard]] inline STFTResult stft(
    const std::vector<f64>& x, int n_fft=512, int hop=128) noexcept
{
    // Hann window
    std::vector<f64> win(n_fft);
    for(int i=0;i<n_fft;++i) win[i]=0.5*(1-std::cos(TWO_PI*i/(n_fft-1)));
    int n=int(x.size());
    STFTResult r; r.hop=hop; r.n_fft=n_fft;
    // Zero-pad x on both sides by n_fft/2
    std::vector<f64> xp(n+n_fft,0);
    for(int i=0;i<n;++i) xp[i+n_fft/2]=x[i];
    r.n_frames=(n+hop-1)/hop;
    r.frames.resize(r.n_frames);
    for(int f=0;f<r.n_frames;++f){
        CxVec frame(n_fft,{0,0});
        int start=f*hop;
        for(int i=0;i<n_fft&&start+i<int(xp.size());++i)
            frame[i]={xp[start+i]*win[i],0};
        fft_inplace(frame);
        // Keep only positive frequencies
        r.frames[f]=CxVec(frame.begin(),frame.begin()+n_fft/2+1);
    }
    return r;
}

// Overlap-add reconstruction
[[nodiscard]] inline std::vector<f64> istft(const STFTResult& s) noexcept {
    int n_fft=s.n_fft, hop=s.hop;
    std::vector<f64> win(n_fft);
    for(int i=0;i<n_fft;++i) win[i]=0.5*(1-std::cos(TWO_PI*i/(n_fft-1)));
    int total=s.n_frames*hop+n_fft;
    std::vector<f64> out(total,0), norm(total,0);
    for(int f=0;f<s.n_frames;++f){
        CxVec frame(n_fft,{0,0});
        for(int i=0;i<=n_fft/2;++i) frame[i]=s.frames[f][i];
        for(int i=1;i<n_fft/2;++i) frame[n_fft-i]=std::conj(s.frames[f][i]);
        fft_inplace(frame,true);
        int start=f*hop;
        for(int i=0;i<n_fft;++i){
            out[start+i]+=frame[i].real()*win[i];
            norm[start+i]+=win[i]*win[i];
        }
    }
    for(int i=0;i<total;++i) if(norm[i]>1e-12) out[i]/=norm[i];
    return std::vector<f64>(out.begin()+n_fft/2, out.begin()+n_fft/2+(s.n_frames-1)*hop);
}

// ─── DCT-II (used in JPEG, audio codecs) ──────────────────────────────────
[[nodiscard]] inline std::vector<f64> dct2(const std::vector<f64>& x) noexcept {
    int n=int(x.size());
    std::vector<f64> y(n);
    for(int k=0;k<n;++k){
        f64 s=0;
        for(int i=0;i<n;++i) s+=x[i]*std::cos(PI*k*(2*i+1)/(2*n));
        y[k]=s*2;
    }
    return y;
}

// DCT-II via FFT (O(n log n))
[[nodiscard]] inline std::vector<f64> dct2_fast(const std::vector<f64>& x) noexcept {
    int n=int(x.size());
    // Reorder: v[k] = x[2k] for k<n/2, v[n-1-k] = x[2k+1] for k<n/2
    std::vector<f64> v(n);
    for(int k=0;k<n/2;++k){v[k]=x[2*k];v[n-1-k]=x[2*k+1];}
    auto V=rfft(v);
    std::vector<f64> y(n);
    for(int k=0;k<n;++k){
        f64 ang=-PI*k/(2*n);
        Cx64 tw={std::cos(ang),std::sin(ang)};
        Cx64 Vk=(k<int(V.size()))?V[k]:std::conj(V[n-k]);
        y[k]=(Vk*tw).real()*2;
    }
    return y;
}

// IDCT-II (inverse DCT-II)
[[nodiscard]] inline std::vector<f64> idct2(const std::vector<f64>& X) noexcept {
    int n=int(X.size());
    std::vector<f64> y(n);
    for(int i=0;i<n;++i){
        f64 s=0.5*X[0];
        for(int k=1;k<n;++k) s+=X[k]*std::cos(PI*k*(2*i+1)/(2*n));
        y[i]=s/n*2;
    }
    return y;
}

// ─── Frequency-domain filtering ───────────────────────────────────────────
// Apply spectral mask H[k] to signal x
[[nodiscard]] inline std::vector<f64> spectral_filter(
    const std::vector<f64>& x, const std::vector<f64>& H) noexcept
{
    int n=next_pow2(int(x.size()));
    CxVec fx(n,{0,0});
    for(int i=0;i<int(x.size());++i) fx[i]={x[i],0};
    fft_inplace(fx);
    for(int i=0;i<n;++i){
        f64 h=(i<int(H.size()))?H[i]:H[n-i<int(H.size())?n-i:0];
        fx[i]*=h;
    }
    fft_inplace(fx,true);
    std::vector<f64> y(x.size());
    for(int i=0;i<int(x.size());++i) y[i]=fx[i].real();
    return y;
}

// ─── Chirp Z-transform (Bluestein) ───────────────────────────────────────
// Evaluates DFT on arbitrary spiral contour (generalizes FFT)
[[nodiscard]] inline CxVec chirp_z(const CxVec& x, Cx64 W, Cx64 A, int M) noexcept {
    int n=int(x.size());
    int L=next_pow2(n+M-1);
    std::vector<Cx64> yn(L,{0,0}), hn(L,{0,0});
    for(int k=0;k<n;++k){
        Cx64 Wk=std::pow(W,Cx64(-0.5*k*k))*std::pow(A,Cx64(-k));
        yn[k]=x[k]*Wk;
    }
    for(int k=0;k<=M-1;++k) hn[k]=std::pow(W,Cx64(0.5*k*k));
    for(int k=L-n+1;k<L;++k) hn[k]=std::pow(W,Cx64(0.5*(L-k)*(L-k)));
    fft_inplace(yn); fft_inplace(hn);
    for(int i=0;i<L;++i) yn[i]*=hn[i];
    fft_inplace(yn,true);
    CxVec out(M);
    for(int k=0;k<M;++k)
        out[k]=yn[k]*std::pow(W,Cx64(0.5*k*k));
    return out;
}

} // namespace qs::math::signal
