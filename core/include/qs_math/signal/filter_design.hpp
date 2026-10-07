#pragma once
// qs_math/signal/filter_design.hpp — IIR/FIR filter design: Butterworth, Chebyshev I/II,
//   Bessel, window-based FIR, bilinear transform, Parks-McClellan remez (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <array>

namespace qs::math::signal {

using Cx64 = std::complex<f64>;
using CxVec = std::vector<Cx64>;

// ─── Biquad section (Direct Form II transposed) ───────────────────────────
struct Biquad {
    f64 b0,b1,b2; // numerator
    f64 a1,a2;    // denominator (a0=1 normalized)
    f64 s1=0,s2=0; // state

    [[nodiscard]] f64 process(f64 x) noexcept {
        f64 y = b0*x + s1;
        s1 = b1*x - a1*y + s2;
        s2 = b2*x - a2*y;
        return y;
    }
    void reset() noexcept { s1=s2=0; }

    // Frequency response at normalized omega [0, pi]
    [[nodiscard]] Cx64 response(f64 omega) const noexcept {
        Cx64 z = std::exp(Cx64{0, omega});
        Cx64 z2 = z*z;
        Cx64 num = Cx64(b0) + Cx64(b1)/z + Cx64(b2)/z2;
        Cx64 den = Cx64(1)  + Cx64(a1)/z + Cx64(a2)/z2;
        return num/den;
    }
};

// ─── Biquad cascade ───────────────────────────────────────────────────────
struct BiquadCascade {
    std::vector<Biquad> sections;

    [[nodiscard]] f64 process(f64 x) noexcept {
        for (auto& bq : sections) x = bq.process(x);
        return x;
    }
    void reset() noexcept { for (auto& bq : sections) bq.reset(); }

    [[nodiscard]] Cx64 response(f64 omega) const noexcept {
        Cx64 H{1,0};
        for (auto& bq : sections) H *= bq.response(omega);
        return H;
    }
    [[nodiscard]] f64 magnitude_db(f64 omega) const noexcept {
        return 20.0*std::log10(std::abs(response(omega))+1e-300);
    }
};

// ─── Bilinear transform (analog -> digital) ───────────────────────────────
// Maps s -> (2/T)*(z-1)/(z+1), with pre-warping
[[nodiscard]] inline f64 prewarp(f64 fc, f64 fs) noexcept {
    return (2.0*fs) * std::tan(PI*fc/fs);
}

// Convert analog biquad (s-domain) to digital (z-domain) via bilinear
// Analog: H(s) = (b0_a*s^2 + b1_a*s + b2_a) / (a0_a*s^2 + a1_a*s + a2_a)
[[nodiscard]] inline Biquad bilinear_biquad(
    f64 b0a, f64 b1a, f64 b2a,
    f64 a0a, f64 a1a, f64 a2a,
    f64 K) noexcept  // K = 2*fs (pre-warped)
{
    f64 K2=K*K;
    f64 b0=b0a*K2+b1a*K+b2a, b1=2.0*(b2a-b0a*K2), b2=b0a*K2-b1a*K+b2a;
    f64 a0=a0a*K2+a1a*K+a2a, a1=2.0*(a2a-a0a*K2), a2=a0a*K2-a1a*K+a2a;
    return {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0};
}

// ─── Butterworth filter design ────────────────────────────────────────────
// Analog prototype poles for order N, unit cutoff
[[nodiscard]] inline CxVec butterworth_poles(int N) noexcept {
    CxVec poles;
    for (int k = 1; k <= N; ++k) {
        f64 angle = PI*(2.0*k+N-1.0)/(2.0*N);
        poles.push_back({std::cos(angle), std::sin(angle)});
    }
    return poles;
}

// Butterworth lowpass biquad cascade, fc normalized [0,1) relative to Nyquist
[[nodiscard]] inline BiquadCascade butterworth_lowpass(int order, f64 fc, f64 fs) noexcept {
    BiquadCascade filt;
    f64 Wc = prewarp(fc, fs);
    f64 K  = Wc; // already in rad/s
    CxVec poles = butterworth_poles(order);
    // Group into biquad sections (complex conjugate pairs)
    int i = 0;
    if (order % 2 == 1) {
        // Real pole: s + |pole| -> single pole section
        f64 p = -poles[(order-1)/2].real() * Wc;
        f64 K2=(2.0*fs);
        // H(s) = Wc/(s+Wc) -> Biquad (b2_a=0, b0_a=0, b1_a=0 -> first order)
        f64 a0=K2+p, a1=-(K2-p);
        filt.sections.push_back({p/a0, p/a0, 0, a1/a0, 0});
        i = 1;
    }
    for (; i < order; i += 2) {
        Cx64 p = poles[i] * Wc;
        f64 om2 = (p*std::conj(p)).real();
        f64 bw  = -2.0*p.real();
        // Analog LP: 1/(s^2 + bw*s + om2)
        filt.sections.push_back(bilinear_biquad(
            0,0,om2, 1,bw,om2, 2.0*fs));
    }
    return filt;
}

// Butterworth highpass
[[nodiscard]] inline BiquadCascade butterworth_highpass(int order, f64 fc, f64 fs) noexcept {
    BiquadCascade filt;
    f64 Wc = prewarp(fc, fs);
    CxVec poles = butterworth_poles(order);
    int i = 0;
    if (order % 2 == 1) {
        f64 p = -poles[(order-1)/2].real() * Wc;
        f64 K2 = 2.0*fs;
        f64 a0=K2+p;
        filt.sections.push_back({K2/a0, -K2/a0, 0, -(K2-p)/a0, 0});
        i=1;
    }
    for (; i < order; i += 2) {
        Cx64 p = poles[i] * Wc;
        f64 om2 = (p*std::conj(p)).real();
        f64 bw  = -2.0*p.real();
        // Analog HP: s^2/(s^2 + bw*s + om2)
        filt.sections.push_back(bilinear_biquad(
            1,0,0, 1,bw,om2, 2.0*fs));
    }
    return filt;
}

// ─── Chebyshev Type I filter ──────────────────────────────────────────────
// eps: ripple parameter, eps = sqrt(10^(Rp_dB/10) - 1)
[[nodiscard]] inline CxVec cheby1_poles(int N, f64 ripple_db) noexcept {
    f64 eps = std::sqrt(std::pow(10.0, ripple_db/10.0) - 1.0);
    f64 mu  = std::asinh(1.0/eps) / f64(N);
    CxVec poles;
    for (int k = 1; k <= N; ++k) {
        f64 angle = PI*(2.0*k-1.0)/(2.0*N);
        poles.push_back({-std::sinh(mu)*std::sin(angle),
                          std::cosh(mu)*std::cos(angle)});
    }
    return poles;
}

[[nodiscard]] inline BiquadCascade chebyshev1_lowpass(int order, f64 fc, f64 fs, f64 ripple_db=3.0) noexcept {
    BiquadCascade filt;
    f64 Wc = prewarp(fc, fs);
    CxVec poles = cheby1_poles(order, ripple_db);
    for (int i = 0; i < order; i += 2) {
        Cx64 p = poles[i] * Wc;
        f64 om2 = (p*std::conj(p)).real();
        f64 bw  = -2.0*p.real();
        filt.sections.push_back(bilinear_biquad(0,0,om2, 1,bw,om2, 2.0*fs));
    }
    return filt;
}

// ─── Chebyshev Type II (inverse Chebyshev) ───────────────────────────────
[[nodiscard]] inline BiquadCascade chebyshev2_lowpass(int order, f64 fc, f64 fs, f64 stopband_db=40.0) noexcept {
    BiquadCascade filt;
    f64 Wc = prewarp(fc, fs);
    f64 eps = std::sqrt(1.0 / (std::pow(10.0, stopband_db/10.0) - 1.0));
    f64 mu  = std::asinh(1.0/eps) / f64(order);
    for (int k = 1; k <= order/2; ++k) {
        f64 angle = PI*(2.0*k-1.0)/(2.0*order);
        Cx64 p_proto = {-std::sinh(mu)*std::sin(angle), std::cosh(mu)*std::cos(angle)};
        // Invert: zeros at j*pi*(2k-1)/(2N), poles = Wc/conjugate(p_proto)
        Cx64 z = {0.0, 1.0/std::cos(angle)};
        Cx64 p = Cx64(Wc) / std::conj(p_proto);
        f64 om2_p = (p*std::conj(p)).real();
        f64 bw_p  = -2.0*p.real();
        f64 om2_z = (z*std::conj(z)).real() * Wc*Wc;
        filt.sections.push_back(bilinear_biquad(1,0,om2_z, 1,bw_p,om2_p, 2.0*fs));
    }
    return filt;
}

// ─── Bessel filter (maximally flat group delay) ───────────────────────────
// Bessel polynomial coefficients up to order 8
[[nodiscard]] inline CxVec bessel_poles_normalized(int N) noexcept {
    // Pre-computed normalized Bessel poles (unit group delay at w=0)
    static const std::vector<std::vector<Cx64>> precomp = {
        {},
        {{-1.0, 0.0}},
        {{-1.5, 0.866025}},
        {{-2.3222, 1.8008}, {-3.6778,0}},
        {{-2.8962,1.9754},{-2.1042,2.6575}},
        {{-3.6467,1.9424},{-3.3520,3.5710},{-4.6484,0}},
        {{-4.2484,1.8680},{-3.7357,3.5835},{-3.0185,5.3426}},
        {{-4.9580,1.7605},{-4.0580,3.5813},{-3.6143,5.3414},{-5.7178,0}},
        {{-5.5878,1.6516},{-4.4895,3.5462},{-3.9373,5.3263},{-4.5642,7.0907}},
    };
    if (N <= 0 || N >= (int)precomp.size()) return {};
    CxVec poles;
    auto& pp = precomp[N];
    for (auto& p : pp) {
        poles.push_back(p);
        if (p.imag() != 0.0) poles.push_back(std::conj(p));
    }
    return poles;
}

// ─── FIR filter design: windowed sinc ────────────────────────────────────
enum class WindowType { Rectangular, Hanning, Hamming, Blackman, Kaiser };

[[nodiscard]] inline f64 window_sample(WindowType w, int n, int N) noexcept {
    f64 x = 2.0*PI*f64(n)/f64(N-1);
    switch(w){
        case WindowType::Hanning:    return 0.5*(1.0-std::cos(x));
        case WindowType::Hamming:    return 0.54-0.46*std::cos(x);
        case WindowType::Blackman:   return 0.42-0.5*std::cos(x)+0.08*std::cos(2*x);
        case WindowType::Kaiser:     return 1.0; // simplified, see kaiser_window()
        default:                     return 1.0;
    }
}

// Kaiser window with parameter beta
[[nodiscard]] inline std::vector<f64> kaiser_window(int N, f64 beta) noexcept {
    auto I0 = [](f64 x) -> f64 {
        f64 s=1,t=1; for(int k=1;k<20;++k){t*=x*x/(4.0*k*k);s+=t;} return s;
    };
    f64 denom = I0(beta);
    std::vector<f64> w(N);
    for (int n=0;n<N;++n){
        f64 a=2.0*n/f64(N-1)-1.0;
        w[n]=I0(beta*std::sqrt(1.0-a*a))/denom;
    }
    return w;
}

// Windowed-sinc lowpass FIR (length N, cutoff fc/fs normalized)
[[nodiscard]] inline std::vector<f64> fir_lowpass(int N, f64 fc, WindowType wt=WindowType::Hamming) noexcept {
    std::vector<f64> h(N);
    int M = N-1;
    f64 fc2 = 2.0*fc; // normalized cutoff [0,1]
    for (int n=0;n<N;++n){
        f64 i=n-M/2.0;
        f64 sinc = (std::abs(i)<1e-12) ? fc2 : std::sin(PI*fc2*i)/(PI*i);
        h[n]=sinc*window_sample(wt,n,N);
    }
    // Normalize to unity gain at DC
    f64 sum=0; for(auto v:h) sum+=v;
    if(std::abs(sum)>1e-12) for(auto& v:h) v/=sum;
    return h;
}

// Highpass FIR by spectral inversion
[[nodiscard]] inline std::vector<f64> fir_highpass(int N, f64 fc, WindowType wt=WindowType::Hamming) noexcept {
    auto h = fir_lowpass(N, fc, wt);
    for(int n=0;n<N;++n) h[n] = -h[n];
    h[N/2] += 1.0;
    return h;
}

// Bandpass FIR (subtract two lowpass)
[[nodiscard]] inline std::vector<f64> fir_bandpass(int N, f64 fc_lo, f64 fc_hi, WindowType wt=WindowType::Hamming) noexcept {
    auto h_hi = fir_lowpass(N, fc_hi, wt);
    auto h_lo = fir_lowpass(N, fc_lo, wt);
    for(int n=0;n<N;++n) h_hi[n]-=h_lo[n];
    return h_hi;
}

// ─── FIR filter application ───────────────────────────────────────────────
struct FIRFilter {
    std::vector<f64> coeffs;
    std::vector<f64> buf;
    int pos = 0;

    FIRFilter(std::vector<f64> h) : coeffs(std::move(h)), buf(coeffs.size(), 0.0) {}

    [[nodiscard]] f64 process(f64 x) noexcept {
        buf[pos] = x;
        f64 y = 0.0;
        int n = int(coeffs.size());
        for (int i=0;i<n;++i) y += coeffs[i] * buf[(pos-i+n)%n];
        pos = (pos+1)%n;
        return y;
    }

    // Process a block
    void process(const f64* in, f64* out, int len) noexcept {
        for(int i=0;i<len;++i) out[i]=process(in[i]);
    }
};

// ─── Frequency response evaluation ───────────────────────────────────────
[[nodiscard]] inline std::vector<Cx64> fir_frequency_response(
    const std::vector<f64>& h, int n_freqs=512) noexcept
{
    std::vector<Cx64> H(n_freqs);
    for (int k=0;k<n_freqs;++k){
        f64 omega = PI*k/f64(n_freqs);
        Cx64 sum{0,0};
        for (int n=0;n<(int)h.size();++n)
            sum += Cx64(h[n]) * std::exp(Cx64{0,-omega*n});
        H[k]=sum;
    }
    return H;
}

// Group delay: -d/domega (arg(H(omega)))
[[nodiscard]] inline std::vector<f64> group_delay(const std::vector<Cx64>& H) noexcept {
    int N = int(H.size());
    std::vector<f64> gd(N);
    for(int k=1;k<N-1;++k){
        f64 a1=std::arg(H[k+1]), a0=std::arg(H[k-1]);
        gd[k]=-0.5*(a1-a0)*f64(N)/PI;
    }
    gd[0]=gd[1]; gd[N-1]=gd[N-2];
    return gd;
}

} // namespace qs::math::signal
