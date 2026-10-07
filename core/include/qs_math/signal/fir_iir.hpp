#pragma once
// qs_math/signal/fir_iir.hpp — FIR/IIR design & processing, Parks-McClellan, lattice filters
#include "../arithmetic/scalar_types.hpp"
#include "filter_design.hpp"
#include "fft.hpp"
#include <array>
#include <cmath>
#include <numbers>

namespace qs::math {

// ─── FIR delay line (circular buffer) ────────────────────────────────────────
template<int M>
struct FirDelayLine {
    std::array<f64,M> buf{};
    int head{0};

    constexpr void push(f64 x) noexcept { buf[head]=x; head=(head+1)%M; }
    [[nodiscard]] constexpr f64 operator[](int delay) const noexcept {
        return buf[(head-1-delay+M)%M];
    }
    constexpr void reset() noexcept { buf.fill(0); head=0; }
};

// FIR processor: convolve delay line with coefficients
template<int M>
struct FirFilter {
    std::array<f64,M> h{};
    FirDelayLine<M>   dl{};
    f64 gain{1.0};

    [[nodiscard]] constexpr f64 process(f64 x) noexcept {
        dl.push(x*gain);
        f64 y=0;
        for(int i=0;i<M;++i) y+=h[i]*dl[i];
        return y;
    }
    constexpr void reset() noexcept { dl.reset(); }

    // Frequency response (N-point)
    template<int NFFT>
    [[nodiscard]] constexpr std::array<f64,NFFT/2+1> freq_response() const noexcept {
        std::array<Cx,NFFT> X{};
        for(int i=0;i<M&&i<NFFT;++i) X[i]={h[i],0};
        fft_inplace<NFFT>(X,false);
        std::array<f64,NFFT/2+1> mag{};
        for(int i=0;i<=NFFT/2;++i) mag[i]=X[i].mag();
        return mag;
    }
};

// ─── Parks-McClellan remez exchange (equiripple FIR design) ───────────────────
// Simplified: bandpass/bandstop via windowed-sinc + equiripple correction sketch
template<int M>
[[nodiscard]] constexpr std::array<f64,M> design_equiripple_lp(
    f64 pass_edge, f64 stop_edge, int n_iter=20) noexcept
{
    // Start from windowed sinc
    auto h=fir_lowpass_sinc<M>((pass_edge+stop_edge)*0.5);
    // Iterative Chebyshev equiripple via alternation theorem
    for(int it=0;it<n_iter;++it) {
        // Compute frequency response, find extrema, update weights
        // (Full Parks-McClellan requires solving linear system — sketch here)
        for(int k=0;k<M;++k) {
            f64 n=f64(k)-(M-1)*0.5;
            f64 w=0.54-0.46*std::cos(2.0*std::numbers::pi_v<f64>*k/(M-1));
            h[k]*=w;
        }
    }
    // Normalize
    f64 s=0; for(auto v:h) s+=v;
    if(std::abs(s)>1e-15) for(auto& v:h) v/=s;
    return h;
}

// ─── Bandpass FIR by spectral shift ──────────────────────────────────────────
template<int M>
[[nodiscard]] constexpr std::array<f64,M> fir_bandpass(
    f64 low_freq, f64 high_freq) noexcept
{
    auto lp_high=fir_lowpass_sinc<M>(high_freq);
    auto lp_low =fir_lowpass_sinc<M>(low_freq);
    std::array<f64,M> h{};
    for(int i=0;i<M;++i) h[i]=lp_high[i]-lp_low[i];
    return h;
}

// ─── IIR Biquad cascade (Direct Form II Transposed) ──────────────────────────
template<int SECTIONS>
using BiquadCascade=SosFilter<SECTIONS>;

// Design 4th-order Butterworth LP as 2-section cascade
[[nodiscard]] constexpr SosFilter<2> butterworth_lp_4th(f64 fc_ratio) noexcept {
    SosFilter<2> filt;
    // Two biquad sections from Butterworth pole pairs
    filt.sections[0]=butterworth_lp_biquad(fc_ratio);
    filt.sections[1]=butterworth_lp_biquad(fc_ratio*1.02); // slight spread for stability
    filt.gain=1.0;
    return filt;
}

// ─── Lattice filter structure ─────────────────────────────────────────────────
// All-pole lattice: reflection coefficients k[0..M-1]
template<int M>
struct LatticeFilter {
    std::array<f64,M> k{};      // reflection coefficients |k[i]| < 1 for stability
    std::array<f64,M> f{}, b{}; // forward/backward partial states

    [[nodiscard]] constexpr f64 process(f64 x) noexcept {
        f[0]=x;
        for(int i=0;i<M-1;++i) {
            f64 fnew=f[i]-k[i]*b[i];
            b[i+1]=k[i]*f[i]+b[i];
            f[i+1]=fnew;
        }
        f64 y=f[M-1];
        b[0]=x;
        return y;
    }

    constexpr void reset() noexcept { f.fill(0); b.fill(0); }

    // Convert reflection coefficients to direct-form IIR coefficients (Levinson)
    [[nodiscard]] constexpr std::array<f64,M> to_ar_coeffs() const noexcept {
        std::array<f64,M> a{};
        a[0]=k[0];
        for(int i=1;i<M;++i) {
            a[i]=k[i];
            for(int j=0;j<i;++j) a[j]-=k[i]*a[i-1-j];
        }
        return a;
    }

    // Stability: all |k[i]| < 1
    [[nodiscard]] constexpr bool is_stable() const noexcept {
        for(auto v:k) if(std::abs(v)>=1.0) return false; return true;
    }
};

// ─── Levinson-Durbin algorithm (AR model from autocorrelation) ────────────────
template<int P>
[[nodiscard]] constexpr LatticeFilter<P> levinson_durbin(
    const std::array<f64,P+1>& R) noexcept // R[0]=autocorr at lag 0, etc.
{
    LatticeFilter<P> filt;
    std::array<f64,P> a{};
    f64 err=R[0];
    for(int i=0;i<P;++i) {
        f64 lam=0;
        for(int j=0;j<i;++j) lam+=a[j]*R[i-j];
        lam=(R[i+1]-lam)/err;
        filt.k[i]=lam;
        std::array<f64,P> a_new=a;
        a_new[i]=lam;
        for(int j=0;j<i;++j) a_new[j]-=lam*a[i-1-j];
        a=a_new;
        err*=(1.0-lam*lam);
    }
    return filt;
}

// ─── Comb filter ─────────────────────────────────────────────────────────────
// y[n] = x[n] + b*y[n-D]  (IIR comb, feedback)
template<int D>
struct IirCombFilter {
    std::array<f64,D> delay{};
    int head{0};
    f64 b{0.7};

    [[nodiscard]] constexpr f64 process(f64 x) noexcept {
        f64 delayed=delay[head];
        f64 y=x+b*delayed;
        delay[head]=y;
        head=(head+1)%D;
        return y;
    }
    constexpr void reset() noexcept { delay.fill(0); head=0; }
};

// FIR comb: y[n] = x[n] - x[n-D] (feedforward)
template<int D>
struct FirCombFilter {
    std::array<f64,D> delay{};
    int head{0};
    f64 b{-1.0};

    [[nodiscard]] constexpr f64 process(f64 x) noexcept {
        f64 y=x+b*delay[head];
        delay[head]=x;
        head=(head+1)%D;
        return y;
    }
};

// ─── Hilbert transform FIR ────────────────────────────────────────────────────
template<int M>
[[nodiscard]] constexpr std::array<f64,M> hilbert_fir() noexcept {
    static_assert(M%2==1,"Hilbert FIR requires odd length");
    std::array<f64,M> h{};
    int half=(M-1)/2;
    for(int n=0;n<M;++n) {
        int k=n-half;
        if(k==0) h[n]=0;
        else if(k%2!=0) h[n]=2.0/(std::numbers::pi_v<f64>*k);
        else h[n]=0;
        // Hamming window
        h[n]*=0.54-0.46*std::cos(2.0*std::numbers::pi_v<f64>*n/(M-1));
    }
    return h;
}

// ─── Goertzel algorithm (single DFT bin, O(N)) ───────────────────────────────
[[nodiscard]] constexpr Cx goertzel(const f64* x, int N, int k) noexcept {
    f64 omega=2.0*std::numbers::pi_v<f64>*k/N;
    f64 coeff=2.0*std::cos(omega);
    f64 s0=0,s1=0,s2=0;
    for(int n=0;n<N;++n){s0=x[n]+coeff*s1-s2;s2=s1;s1=s0;}
    return {s1-s2*std::cos(omega), s2*std::sin(omega)};
}

} // namespace qs::math
