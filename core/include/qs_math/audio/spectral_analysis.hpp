#pragma once
// qs_math/spectral_analysis.hpp — STFT, CQT, mel spectrogram, MFCC (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <span>
#include <array>

namespace qs {

using CX32 = std::complex<f32>;

// ─── Window functions ─────────────────────────────────────────────────────────

enum class WindowType { Rectangular, Hann, Hamming, Blackman, Kaiser, FlatTop };

inline void apply_window(std::span<f32> buf, WindowType type, f32 beta=8.6f) {
    u32 N = u32(buf.size());
    for (u32 n=0;n<N;++n) {
        f32 w = 1.f;
        f32 t = f32(n)/f32(N-1);
        switch(type) {
        case WindowType::Hann:
            w = 0.5f*(1.f-std::cos(2.f*k_pi*t)); break;
        case WindowType::Hamming:
            w = 0.54f-0.46f*std::cos(2.f*k_pi*t); break;
        case WindowType::Blackman:
            w = 0.42f-0.5f*std::cos(2.f*k_pi*t)+0.08f*std::cos(4.f*k_pi*t); break;
        case WindowType::FlatTop:
            w = 0.21557895f-0.41663158f*std::cos(2.f*k_pi*t)
               +0.27726316f*std::cos(4.f*k_pi*t)-0.08357895f*std::cos(6.f*k_pi*t)
               +0.00694737f*std::cos(8.f*k_pi*t); break;
        case WindowType::Kaiser: {
            // Approximate I0 (modified Bessel) via series
            auto i0=[](f32 x)->f32{
                f32 s=1,t2=1; for(int k=1;k<=12;++k){t2*=(x/2.f/f32(k));t2*=t2;s+=t2;}
                return s;
            };
            f32 r = 1.f - (2.f*f32(n)/f32(N-1)-1.f)*(2.f*f32(n)/f32(N-1)-1.f);
            w = i0(beta*std::sqrt(std::max(0.f,r)))/i0(beta);
        } break;
        default: break;
        }
        buf[n] *= w;
    }
}

// ─── STFT helpers ─────────────────────────────────────────────────────────────

struct STFTParams {
    u32 fft_size{1024};
    u32 hop_size{256};
    WindowType window{WindowType::Hann};
    u32 sample_rate{44100};
};

// Simple DFT for one frame (use power-of-2 fft in practice)
inline void compute_dft_frame(
    std::span<const f32> frame, std::span<CX32> out)
{
    u32 N = u32(frame.size());
    u32 M = std::min(N/2+1, u32(out.size()));
    for (u32 k=0;k<M;++k) {
        CX32 s{0,0};
        for (u32 n=0;n<N;++n) {
            f32 ang = -2.f*k_pi*f32(k)*f32(n)/f32(N);
            s += CX32{frame[n]*std::cos(ang), frame[n]*std::sin(ang)};
        }
        out[k] = s;
    }
}

// Magnitude spectrum in dBFS
inline void magnitude_db(std::span<const CX32> spectrum, std::span<f32> mag_db, f32 ref=1.f) {
    for (u32 i=0;i<u32(std::min(spectrum.size(),mag_db.size()));++i) {
        f32 m = std::abs(spectrum[i]) / ref;
        mag_db[i] = m > 1e-10f ? 20.f*std::log10(m) : -200.f;
    }
}

// Power spectrum
inline void power_spectrum(std::span<const CX32> spectrum, std::span<f32> power) {
    for (u32 i=0;i<u32(std::min(spectrum.size(),power.size()));++i)
        power[i] = std::norm(spectrum[i]);
}

// ─── Mel filter bank ──────────────────────────────────────────────────────────

// Hz <-> Mel conversions (O'Shaughnessy formula)
[[nodiscard]] inline f32 hz_to_mel(f32 hz) { return 2595.f*std::log10(1.f+hz/700.f); }
[[nodiscard]] inline f32 mel_to_hz(f32 mel) { return 700.f*(std::pow(10.f,mel/2595.f)-1.f); }

struct MelFilterBank {
    u32 num_filters{};
    u32 fft_bins{};
    u32 sample_rate{};
    std::vector<std::vector<f32>> filters; // [num_filters][fft_bins]

    void build(u32 n_filters, u32 fft_sz, u32 sr,
               f32 f_min=0.f, f32 f_max=0.f)
    {
        num_filters = n_filters;
        fft_bins    = fft_sz/2+1;
        sample_rate = sr;
        if (f_max <= 0.f) f_max = f32(sr)/2.f;
        filters.assign(n_filters, std::vector<f32>(fft_bins, 0.f));

        f32 mel_min = hz_to_mel(f_min);
        f32 mel_max = hz_to_mel(f_max);
        std::vector<f32> mel_points(n_filters+2);
        for (u32 i=0;i<n_filters+2;++i)
            mel_points[i] = mel_to_hz(mel_min + f32(i)*(mel_max-mel_min)/f32(n_filters+1));

        for (u32 m=0;m<n_filters;++m) {
            f32 f_lo = mel_points[m];
            f32 f_pk = mel_points[m+1];
            f32 f_hi = mel_points[m+2];
            for (u32 k=0;k<fft_bins;++k) {
                f32 f = f32(k)*f32(sr)/f32(fft_sz);
                if (f>=f_lo && f<=f_pk)
                    filters[m][k] = (f-f_lo)/(f_pk-f_lo);
                else if (f>f_pk && f<=f_hi)
                    filters[m][k] = (f_hi-f)/(f_hi-f_pk);
            }
        }
    }

    // Apply filter bank to power spectrum -> mel energies
    void apply(std::span<const f32> power, std::span<f32> mel_energy) const {
        for (u32 m=0;m<num_filters&&m<u32(mel_energy.size());++m) {
            f32 e=0.f;
            for (u32 k=0;k<fft_bins&&k<u32(power.size());++k)
                e += power[k]*filters[m][k];
            mel_energy[m] = e;
        }
    }
};

// ─── MFCC ─────────────────────────────────────────────────────────────────────

// DCT-II of log mel energies -> MFCCs
inline void compute_mfcc(
    std::span<const f32> log_mel, std::span<f32> mfcc)
{
    u32 M = u32(log_mel.size());
    u32 K = u32(mfcc.size());
    f32 scale = std::sqrt(2.f/f32(M));
    for (u32 k=0;k<K;++k) {
        f32 s=0.f;
        for (u32 m=0;m<M;++m)
            s += log_mel[m]*std::cos(k_pi/f32(M)*(f32(m)+0.5f)*f32(k));
        mfcc[k] = scale*s;
    }
}

// Full pipeline: power spectrum -> mel -> log -> DCT -> MFCCs
inline void compute_mfcc_from_power(
    std::span<const f32> power_spectrum_in,
    const MelFilterBank& mel_fb, u32 num_coeffs,
    std::span<f32> mfcc_out)
{
    std::vector<f32> mel_e(mel_fb.num_filters);
    mel_fb.apply(power_spectrum_in, mel_e);
    for (auto& v : mel_e) v = v > 1e-10f ? std::log(v) : -23.f;
    std::vector<f32> mfcc(num_coeffs);
    compute_mfcc(mel_e, mfcc);
    for (u32 i=0;i<num_coeffs&&i<u32(mfcc_out.size());++i) mfcc_out[i]=mfcc[i];
}

// ─── CQT (Constant-Q Transform) ───────────────────────────────────────────────

struct CQTParams {
    u32 num_bins{84};      // e.g. 7 octaves * 12 bins
    f32 f_min{32.7f};      // C1
    u32 bins_per_octave{12};
    u32 sample_rate{44100};
    f32 q_factor{1.f};
};

[[nodiscard]] inline f32 cqt_center_freq(u32 k, const CQTParams& p) {
    return p.f_min * std::pow(2.f, f32(k)/f32(p.bins_per_octave));
}

[[nodiscard]] inline f32 cqt_window_len(u32 k, const CQTParams& p) {
    f32 fk = cqt_center_freq(k, p);
    f32 Q = p.q_factor * (std::pow(2.f, 1.f/f32(p.bins_per_octave))-1.f);
    return f32(p.sample_rate) * Q / fk;
}

// Naive CQT for one bin (expensive, use sparse kernel in production)
[[nodiscard]] inline CX32 cqt_bin(
    std::span<const f32> signal, u32 k, const CQTParams& p)
{
    f32 fk = cqt_center_freq(k, p);
    u32 Nk = u32(cqt_window_len(k, p));
    Nk = std::min(Nk, u32(signal.size()));
    CX32 X{0,0};
    for (u32 n=0;n<Nk;++n) {
        // Hamming window
        f32 w = 0.54f-0.46f*std::cos(2.f*k_pi*f32(n)/f32(Nk-1));
        f32 ang = -2.f*k_pi*fk*f32(n)/f32(p.sample_rate);
        X += CX32{signal[n]*w*std::cos(ang), signal[n]*w*std::sin(ang)};
    }
    return X * (1.f/f32(Nk));
}

// ─── Spectral descriptors ─────────────────────────────────────────────────────

[[nodiscard]] inline f32 spectral_centroid(std::span<const f32> mag, u32 sr, u32 fft_sz) {
    f32 num=0,den=0;
    for (u32 k=0;k<u32(mag.size());++k) {
        f32 f = f32(k)*f32(sr)/f32(fft_sz);
        num += f*mag[k]; den += mag[k];
    }
    return den > 0.f ? num/den : 0.f;
}

[[nodiscard]] inline f32 spectral_bandwidth(std::span<const f32> mag, f32 centroid, u32 sr, u32 fft_sz) {
    f32 num=0,den=0;
    for (u32 k=0;k<u32(mag.size());++k) {
        f32 f = f32(k)*f32(sr)/f32(fft_sz);
        f32 d = f-centroid;
        num += d*d*mag[k]; den += mag[k];
    }
    return den > 0.f ? std::sqrt(num/den) : 0.f;
}

[[nodiscard]] inline f32 spectral_rolloff(std::span<const f32> power, f32 threshold, u32 sr, u32 fft_sz) {
    f32 total = 0.f;
    for (auto v : power) total += v;
    f32 target = total * threshold, acc = 0.f;
    for (u32 k=0;k<u32(power.size());++k) {
        acc += power[k];
        if (acc >= target) return f32(k)*f32(sr)/f32(fft_sz);
    }
    return f32(sr)/2.f;
}

[[nodiscard]] inline f32 spectral_flatness(std::span<const f32> power) {
    f32 log_sum=0.f, lin_sum=0.f;
    u32 N = u32(power.size());
    for (auto v : power) {
        f32 p = std::max(v, 1e-10f);
        log_sum += std::log(p); lin_sum += p;
    }
    return N>0 ? std::exp(log_sum/f32(N))/(lin_sum/f32(N)) : 0.f;
}

} // namespace qs
