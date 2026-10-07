#pragma once
// qs_math/fourier_math.hpp — Fourier series, DFT/FFT, convolution theorem (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <functional>
#include <numbers>

namespace qs {

using Complex = std::complex<f64>;
using CArray  = std::vector<Complex>;

static constexpr f64 TWO_PI = 2.0 * std::numbers::pi_v<f64>;

// ─── Cooley-Tukey FFT (in-place, radix-2, N must be power of 2) ──────────────
inline void fft_inplace(CArray& a, bool inverse=false) {
    u32 N=u32(a.size());
    assert((N&(N-1))==0 && "FFT size must be power of 2");
    // Bit-reversal permutation
    for(u32 i=1,j=0;i<N;++i) {
        u32 bit=N>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j) std::swap(a[i],a[j]);
    }
    // Cooley-Tukey butterfly
    for(u32 len=2;len<=N;len<<=1) {
        f64 ang=TWO_PI/len*(inverse?1:-1);
        Complex wlen(std::cos(ang),std::sin(ang));
        for(u32 i=0;i<N;i+=len) {
            Complex w(1,0);
            for(u32 j=0;j<len/2;++j) {
                Complex u_=a[i+j], v=a[i+j+len/2]*w;
                a[i+j]=u_+v; a[i+j+len/2]=u_-v;
                w*=wlen;
            }
        }
    }
    if(inverse) for(auto& c:a) c/=f64(N);
}

// FFT of real signal — returns N/2+1 complex frequency bins
[[nodiscard]] inline CArray fft_real(const std::vector<f64>& x) {
    u32 N=u32(x.size());
    CArray a(N);
    for(u32 i=0;i<N;++i) a[i]=x[i];
    fft_inplace(a);
    a.resize(N/2+1);
    return a;
}

// Inverse FFT to real signal (from N/2+1 bins)
[[nodiscard]] inline std::vector<f64> ifft_real(CArray bins, u32 N) {
    bins.resize(N);
    // Mirror conjugate
    for(u32 i=N/2+1;i<N;++i) bins[i]=std::conj(bins[N-i]);
    fft_inplace(bins,true);
    std::vector<f64> r(N);
    for(u32 i=0;i<N;++i) r[i]=bins[i].real();
    return r;
}

// Zero-pad to next power of 2
[[nodiscard]] inline u32 next_pow2(u32 n) {
    u32 p=1; while(p<n) p<<=1; return p;
}

// ─── DFT (arbitrary N, O(N²) — use for small N or verification) ──────────────
[[nodiscard]] inline CArray dft(const std::vector<f64>& x) {
    u32 N=u32(x.size());
    CArray X(N);
    for(u32 k=0;k<N;++k) {
        for(u32 n=0;n<N;++n)
            X[k]+=x[n]*Complex(std::cos(-TWO_PI*k*n/N),std::sin(-TWO_PI*k*n/N));
    }
    return X;
}

// ─── Convolution via FFT (linear, zero-padded) ────────────────────────────────
[[nodiscard]] inline std::vector<f64> convolve(const std::vector<f64>& a, const std::vector<f64>& b) {
    u32 n=next_pow2(u32(a.size()+b.size()-1));
    CArray fa(n),fb(n);
    for(u32 i=0;i<u32(a.size());++i) fa[i]=a[i];
    for(u32 i=0;i<u32(b.size());++i) fb[i]=b[i];
    fft_inplace(fa); fft_inplace(fb);
    for(u32 i=0;i<n;++i) fa[i]*=fb[i];
    fft_inplace(fa,true);
    std::vector<f64> result(a.size()+b.size()-1);
    for(u32 i=0;i<u32(result.size());++i) result[i]=fa[i].real();
    return result;
}

// Cross-correlation
[[nodiscard]] inline std::vector<f64> correlate(const std::vector<f64>& a, const std::vector<f64>& b) {
    std::vector<f64> b_rev(b.rbegin(),b.rend());
    return convolve(a,b_rev);
}

// ─── Fourier Series ───────────────────────────────────────────────────────────
struct FourierSeries {
    f64 a0{0};
    std::vector<f64> an, bn; // cosine and sine coefficients
    f64 period{TWO_PI};

    // Compute coefficients from samples (N evenly spaced over period)
    static FourierSeries from_samples(const std::vector<f64>& x, u32 n_terms, f64 T=TWO_PI) {
        u32 N=u32(x.size());
        FourierSeries fs; fs.period=T; fs.an.resize(n_terms); fs.bn.resize(n_terms);
        fs.a0=0; for(auto v:x) fs.a0+=v; fs.a0/=N;
        for(u32 k=1;k<=n_terms;++k) {
            f64 ac=0,bc=0;
            for(u32 n=0;n<N;++n) {
                f64 ang=TWO_PI*k*n/N;
                ac+=x[n]*std::cos(ang); bc+=x[n]*std::sin(ang);
            }
            fs.an[k-1]=2*ac/N; fs.bn[k-1]=2*bc/N;
        }
        return fs;
    }

    // Evaluate at t
    [[nodiscard]] f64 eval(f64 t) const {
        f64 s=a0;
        for(u32 k=0;k<u32(an.size());++k) {
            f64 ang=TWO_PI*(k+1)*t/period;
            s+=an[k]*std::cos(ang)+bn[k]*std::sin(ang);
        }
        return s;
    }

    // Power spectrum |c_k|^2
    [[nodiscard]] std::vector<f64> power_spectrum() const {
        std::vector<f64> ps(an.size());
        for(u32 k=0;k<u32(an.size());++k) ps[k]=0.5*(an[k]*an[k]+bn[k]*bn[k]);
        return ps;
    }

    // Integrate f on [0,T]
    [[nodiscard]] f64 integrate(f64 a=0,f64 b=-1) const {
        if(b<0) b=period;
        return a0*(b-a); // DC term only (AC terms integrate to 0 over full period)
    }
};

// ─── Short-Time Fourier Transform (STFT) ─────────────────────────────────────
struct STFTResult {
    std::vector<CArray> frames;     // [frame][freq_bin]
    u32 hop_size{0};
    u32 win_size{0};
};

[[nodiscard]] inline STFTResult stft(const std::vector<f64>& x, u32 win=512, u32 hop=128) {
    u32 N=u32(x.size());
    STFTResult res; res.hop_size=hop; res.win_size=win;
    // Hann window
    std::vector<f64> window(win);
    for(u32 i=0;i<win;++i) window[i]=0.5*(1-std::cos(TWO_PI*i/(win-1)));
    for(u32 start=0;start+win<=N;start+=hop) {
        CArray frame(win);
        for(u32 i=0;i<win;++i) frame[i]=x[start+i]*window[i];
        fft_inplace(frame);
        res.frames.push_back(frame);
    }
    return res;
}

// ─── Convolution theorem: multiplication in freq domain = convolution in time ─
// Circular convolution (no zero-padding)
[[nodiscard]] inline std::vector<f64> circular_convolve(const std::vector<f64>& a, const std::vector<f64>& b) {
    assert(a.size()==b.size());
    u32 N=next_pow2(u32(a.size())); // promote to pow2
    CArray fa(N,0),fb(N,0);
    for(u32 i=0;i<u32(a.size());++i) fa[i]=a[i];
    for(u32 i=0;i<u32(b.size());++i) fb[i]=b[i];
    fft_inplace(fa); fft_inplace(fb);
    for(u32 i=0;i<N;++i) fa[i]*=fb[i];
    fft_inplace(fa,true);
    std::vector<f64> r(a.size());
    for(u32 i=0;i<u32(r.size());++i) r[i]=fa[i].real();
    return r;
}

// ─── Frequency domain filtering ──────────────────────────────────────────────
[[nodiscard]] inline std::vector<f64> lowpass_filter(const std::vector<f64>& x, f64 cutoff_ratio) {
    u32 N=next_pow2(u32(x.size()));
    CArray X(N);
    for(u32 i=0;i<u32(x.size());++i) X[i]=x[i];
    fft_inplace(X);
    u32 cutoff=u32(cutoff_ratio*N);
    for(u32 i=cutoff;i<N-cutoff;++i) X[i]=0;
    fft_inplace(X,true);
    std::vector<f64> r(x.size());
    for(u32 i=0;i<u32(r.size());++i) r[i]=X[i].real();
    return r;
}

// Parseval's theorem: energy in time = energy in frequency
[[nodiscard]] inline f64 parseval_check(const std::vector<f64>& x, const CArray& X) {
    f64 et=0,ef=0;
    for(auto v:x) et+=v*v;
    for(auto c:X) ef+=std::norm(c);
    return std::abs(et-ef/u32(X.size())); // should be ~0
}

// ─── Chirp Z-Transform (CZT) — generalised DFT on a spiral contour ───────────
[[nodiscard]] inline CArray czt(const std::vector<f64>& x, u32 M, Complex W, Complex A) {
    u32 N=u32(x.size());
    u32 L=next_pow2(N+M-1);
    // yn = x_n * A^{-n} * W^{n^2/2}
    CArray yn(L,0);
    for(u32 n=0;n<N;++n) {
        f64 n2=f64(n)*n*0.5;
        yn[n]=x[n]*std::pow(A,(f64)(-i32(n)))*std::pow(W,n2);
    }
    // Filter h_n = W^{-n^2/2}
    CArray hn(L,0);
    for(u32 n=0;n<M;++n) hn[n]=std::pow(W,-f64(n)*n*0.5);
    for(u32 n=1;n<N;++n) hn[L-n]=std::pow(W,-f64(n)*n*0.5);
    fft_inplace(yn); fft_inplace(hn);
    for(u32 i=0;i<L;++i) yn[i]*=hn[i];
    fft_inplace(yn,true);
    CArray result(M);
    for(u32 k=0;k<M;++k) result[k]=std::pow(W,f64(k)*k*0.5)*yn[k];
    return result;
}

} // namespace qs
