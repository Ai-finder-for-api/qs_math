#pragma once
// qs_math/engineering/signal_processing.hpp — Resampling, modulation, spread spectrum, OFDM
#include "../arithmetic/scalar_types.hpp"
#include "../signal/fft.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Resampling ───────────────────────────────────────────────────────────────
// Polyphase upsampling by integer L
template<int N, int L, int KS>
[[nodiscard]] constexpr std::array<f64,N*L> upsample_polyphase(
    const std::array<f64,N>& x, const std::array<f64,KS>& proto_lpf) noexcept
{
    std::array<f64,N*L> y{};
    for(int i=0;i<N;++i) for(int k=0;k<KS;++k) {
        int phase=k%L, tap=k/L;
        int out_idx=i*L+phase-tap;
        if(out_idx>=0&&out_idx<N*L) y[out_idx]+=x[i]*proto_lpf[k]*L;
    }
    return y;
}

// Linear interpolation resampling
template<int NIN, int NOUT>
[[nodiscard]] constexpr std::array<f64,NOUT> resample_linear(
    const std::array<f64,NIN>& x) noexcept
{
    std::array<f64,NOUT> y{};
    f64 step=f64(NIN-1)/(NOUT-1);
    for(int i=0;i<NOUT;++i) {
        f64 pos=i*step;
        int lo=int(pos),hi=lo+1<NIN?lo+1:lo;
        f64 t=pos-lo;
        y[i]=x[lo]*(1-t)+x[hi]*t;
    }
    return y;
}

// ─── Amplitude Modulation (AM) ────────────────────────────────────────────────
[[nodiscard]] constexpr f64 am_modulate(f64 message, f64 carrier_amp,
                                          f64 fc, f64 t, f64 m_index=1.0) noexcept {
    return carrier_amp*(1+m_index*message)*std::cos(2*3.14159265358979323846*fc*t);
}
// AM demodulation (envelope detection step)
[[nodiscard]] constexpr f64 am_envelope(f64 prev_env, f64 x, f64 alpha=0.1) noexcept {
    return alpha*std::abs(x)+(1-alpha)*prev_env;
}

// ─── Frequency Modulation (FM) ────────────────────────────────────────────────
// FM: s(t) = Ac*cos(2π*fc*t + 2π*kf*∫m(τ)dτ)
struct FMModulator {
    f64 fc{1000},kf{100},Ac{1},phase{0};
    [[nodiscard]] constexpr f64 sample(f64 message, f64 dt) noexcept {
        phase+=2*3.14159265358979323846*(fc+kf*message)*dt;
        return Ac*std::cos(phase);
    }
};

// FM demodulation (discriminator)
[[nodiscard]] constexpr f64 fm_discriminate(f64 x_re, f64 x_im,
                                              f64 prev_re, f64 prev_im, f64 dt) noexcept {
    // Instantaneous frequency: d/dt[arg(x)] = (x_re*dx_im/dt - x_im*dx_re/dt)/|x|²
    f64 dx_re=(x_re-prev_re)/dt, dx_im=(x_im-prev_im)/dt;
    f64 mag2=x_re*x_re+x_im*x_im;
    return mag2>1e-15?(x_re*dx_im-x_im*dx_re)/mag2:0;
}

// ─── Phase-Locked Loop (PLL) ──────────────────────────────────────────────────
struct PLL {
    f64 freq{0},phase{0},bandwidth{100};
    f64 alpha,beta; // loop filter coefficients
    constexpr PLL(f64 bw,f64 damping=0.707) noexcept {
        alpha=2*damping*2*3.14159265358979323846*bw;
        beta=(2*3.14159265358979323846*bw)*(2*3.14159265358979323846*bw);
    }
    [[nodiscard]] constexpr std::pair<f64,f64> update(f64 in_re, f64 in_im, f64 dt) noexcept {
        // NCO output
        f64 nco_re=std::cos(phase), nco_im=-std::sin(phase);
        // Phase detector (cross product)
        f64 error=in_re*nco_im-in_im*nco_re;
        // Loop filter
        freq+=beta*error*dt;
        phase+=(freq+alpha*error)*dt;
        return {nco_re,nco_im};
    }
};

// ─── OFDM ─────────────────────────────────────────────────────────────────────
// OFDM modulation: IFFT of subcarrier symbols → time domain
template<int N_SC, int CP_LEN>
[[nodiscard]] constexpr std::array<Cx,N_SC+CP_LEN> ofdm_modulate(
    const std::array<Cx,N_SC>& symbols) noexcept
{
    std::array<Cx,N_SC> td=symbols;
    fft_inplace<N_SC>(td,true); // IFFT
    std::array<Cx,N_SC+CP_LEN> out{};
    // Cyclic prefix
    for(int i=0;i<CP_LEN;++i) out[i]=td[N_SC-CP_LEN+i];
    for(int i=0;i<N_SC;++i)   out[CP_LEN+i]=td[i];
    return out;
}

template<int N_SC, int CP_LEN>
[[nodiscard]] constexpr std::array<Cx,N_SC> ofdm_demodulate(
    const std::array<Cx,N_SC+CP_LEN>& rx) noexcept
{
    std::array<Cx,N_SC> fd{};
    for(int i=0;i<N_SC;++i) fd[i]=rx[CP_LEN+i];
    fft_inplace<N_SC>(fd,false); // FFT
    return fd;
}

// ─── Spread spectrum ──────────────────────────────────────────────────────────
// PN sequence (maximal length, shift register)
template<int N>
[[nodiscard]] constexpr std::array<int,N> pn_sequence(u32 poly, u32 seed=1) noexcept {
    std::array<int,N> pn{};
    u32 reg=seed;
    for(int i=0;i<N;++i) {
        pn[i]=(reg&1)?1:-1;
        u32 feedback=0;
        u32 tmp=reg&poly;
        while(tmp){feedback^=tmp&1;tmp>>=1;}
        reg=(reg>>1)|(feedback<<(32-__builtin_clz(poly)-1));
    }
    return pn;
}

// BPSK spreading
template<int N, int SF>
[[nodiscard]] constexpr std::array<f64,N*SF> spread_bpsk(
    const std::array<f64,N>& data,
    const std::array<int,SF>& chips) noexcept
{
    std::array<f64,N*SF> out{};
    for(int i=0;i<N;++i) for(int k=0;k<SF;++k) out[i*SF+k]=data[i]*chips[k];
    return out;
}

// ─── Matched filter ───────────────────────────────────────────────────────────
template<int N, int T>
[[nodiscard]] constexpr std::array<f64,N+T-1> matched_filter(
    const std::array<f64,N>& x, const std::array<f64,T>& templ) noexcept
{
    std::array<f64,N+T-1> y{};
    for(int i=0;i<N;++i) for(int k=0;k<T;++k) y[i+k]+=x[i]*templ[T-1-k];
    return y;
}

// ─── Noise figures ────────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 snr_db(f64 signal_power, f64 noise_power) noexcept {
    return 10*std::log10(signal_power/(noise_power+1e-30));
}
[[nodiscard]] constexpr f64 noise_figure_db(f64 noise_factor) noexcept {
    return 10*std::log10(noise_factor);
}
[[nodiscard]] constexpr f64 friis_cascade_nf(f64 nf1, f64 nf2, f64 gain1) noexcept {
    return nf1+(nf2-1)/gain1; // noise factors (linear)
}
[[nodiscard]] constexpr f64 thermal_noise_power(f64 T, f64 bandwidth) noexcept {
    return 1.380649e-23*T*bandwidth; // kTB
}
// Shannon capacity: C = B*log2(1+SNR)
[[nodiscard]] constexpr f64 shannon_capacity(f64 bandwidth, f64 snr_linear) noexcept {
    return bandwidth*std::log2(1+snr_linear);
}
// Eb/N0 from SNR: Eb/N0 = SNR * B/Rb
[[nodiscard]] constexpr f64 ebn0_from_snr(f64 snr, f64 bandwidth, f64 bit_rate) noexcept {
    return snr*bandwidth/bit_rate;
}

// ─── Window-based spectral analysis ───────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 spectral_centroid(
    const std::array<f64,N/2+1>& magnitude, f64 sample_rate) noexcept
{
    f64 num=0,den=0;
    for(int k=0;k<=N/2;++k) {
        f64 freq=f64(k)*sample_rate/N;
        num+=magnitude[k]*freq;
        den+=magnitude[k];
    }
    return den>1e-15?num/den:0;
}
template<int N>
[[nodiscard]] constexpr f64 spectral_flatness(const std::array<f64,N>& X) noexcept {
    f64 log_sum=0,arith=0;
    for(auto v:X) if(v>1e-12){log_sum+=std::log(v);arith+=v;}
    return std::exp(log_sum/N)/(arith/N+1e-15);
}
template<int N>
[[nodiscard]] constexpr f64 spectral_rolloff(
    const std::array<f64,N>& mag, f64 threshold=0.85) noexcept
{
    f64 total=0; for(auto v:mag) total+=v;
    f64 cum=0;
    for(int k=0;k<N;++k){cum+=mag[k];if(cum>=threshold*total) return f64(k);}
    return f64(N-1);
}
} // namespace qs::math
