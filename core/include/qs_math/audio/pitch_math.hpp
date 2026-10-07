#pragma once
// qs_math/pitch_math.hpp — YIN pitch detection, autocorrelation, pitch shifting, phase vocoder (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>
#include <optional>
#include <span>

namespace qs {

using Complex32 = std::complex<f32>;

// ─── Autocorrelation ──────────────────────────────────────────────────────────

// Biased autocorrelation via direct method
inline void autocorrelate(std::span<const f32> x, std::span<f32> r) {
    u32 N = u32(x.size());
    u32 M = u32(r.size());
    for (u32 lag = 0; lag < M; ++lag) {
        f32 sum = 0.f;
        for (u32 n = 0; n + lag < N; ++n)
            sum += x[n] * x[n+lag];
        r[lag] = sum / f32(N);
    }
}

// ─── YIN pitch detector (de Cheveigné & Kawahara 2002) ───────────────────────

struct YINResult {
    f32 frequency{};
    f32 probability{};  // 1 - minimum dip value
    bool voiced{};
};

// Step 1: difference function d(tau)
inline void yin_difference(std::span<const f32> x, std::span<f32> d) {
    u32 W = u32(d.size());
    for (u32 tau = 0; tau < W; ++tau) {
        f32 sum = 0.f;
        for (u32 j = 0; j < W; ++j) {
            f32 diff = x[j] - (j+tau < x.size() ? x[j+tau] : 0.f);
            sum += diff*diff;
        }
        d[tau] = sum;
    }
}

// Step 2: cumulative mean normalized difference
inline void yin_cmnd(std::span<f32> d) {
    d[0] = 1.f;
    f32 running_sum = 0.f;
    for (u32 tau = 1; tau < u32(d.size()); ++tau) {
        running_sum += d[tau];
        d[tau] = d[tau] * f32(tau) / running_sum;
    }
}

// Step 3: absolute threshold + parabolic interpolation
[[nodiscard]] inline std::optional<f32> yin_threshold(
    std::span<const f32> cmnd, f32 threshold=0.1f)
{
    u32 N = u32(cmnd.size());
    for (u32 tau = 2; tau < N-1; ++tau) {
        if (cmnd[tau] < threshold) {
            // Find local minimum via parabolic interpolation
            while (tau+1 < N-1 && cmnd[tau+1] < cmnd[tau]) ++tau;
            f32 s0=cmnd[tau-1], s1=cmnd[tau], s2=cmnd[tau+1];
            f32 denom = 2.f*(s0-2.f*s1+s2);
            if (std::abs(denom) < 1e-10f) return f32(tau);
            return f32(tau) - (s2-s0)/denom * 0.5f;
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline YINResult yin_detect(
    std::span<const f32> frame, u32 sample_rate,
    f32 threshold=0.1f, u32 min_freq=60, u32 max_freq=2000)
{
    u32 W = u32(frame.size())/2;
    std::vector<f32> d(W);
    yin_difference(frame, d);
    yin_cmnd(d);

    u32 tau_min = sample_rate/max_freq;
    u32 tau_max = std::min(W-2, sample_rate/min_freq);

    std::span<f32> search{d.data()+tau_min, tau_max-tau_min+1};
    // Offset the search
    auto result = yin_threshold(search, threshold);

    if (!result) {
        return {0.f, 0.f, false};
    }

    f32 tau = *result + f32(tau_min);
    f32 freq = f32(sample_rate) / tau;
    f32 prob = 1.f - d[u32(std::round(tau))];
    return {freq, std::clamp(prob,0.f,1.f), true};
}

// ─── AMDF pitch detector ──────────────────────────────────────────────────────

[[nodiscard]] inline f32 amdf_pitch(
    std::span<const f32> x, u32 sample_rate, u32 tau_min=20, u32 tau_max=500)
{
    tau_max = std::min(tau_max, u32(x.size())-1);
    f32 min_val = std::numeric_limits<f32>::max();
    u32 best_tau = tau_min;
    for (u32 tau = tau_min; tau <= tau_max; ++tau) {
        f32 sum = 0.f;
        u32 count = u32(x.size()) - tau;
        for (u32 n = 0; n < count; ++n)
            sum += std::abs(x[n] - x[n+tau]);
        f32 amdf = sum / f32(count);
        if (amdf < min_val) { min_val = amdf; best_tau = tau; }
    }
    return f32(sample_rate) / f32(best_tau);
}

// ─── Phase Vocoder ────────────────────────────────────────────────────────────

struct PhaseVocoderParams {
    u32 fft_size{2048};
    u32 hop_size{512};
    f32 pitch_shift{1.0f};   // ratio (2.0 = octave up)
    f32 time_stretch{1.0f};  // ratio (2.0 = twice as slow)
    u32 sample_rate{44100};
};

// Phase accumulator for phase vocoder output synthesis
struct PVChannel {
    std::vector<f32> phase_acc;
    std::vector<f32> prev_phase;
    u32 fft_size{};

    void init(u32 fft_sz) {
        fft_size = fft_sz;
        phase_acc.assign(fft_sz/2+1, 0.f);
        prev_phase.assign(fft_sz/2+1, 0.f);
    }

    // Compute true frequency from phase difference
    [[nodiscard]] f32 true_freq(u32 bin, f32 phase, f32 prev, u32 hop, u32 sr) const {
        f32 expected = 2.f*k_pi*f32(bin)*f32(hop)/f32(fft_size);
        f32 delta = phase - prev - expected;
        // Wrap to [-pi, pi]
        while (delta >  k_pi) delta -= 2.f*k_pi;
        while (delta < -k_pi) delta += 2.f*k_pi;
        return (f32(bin) + delta/(2.f*k_pi)) * f32(sr)/f32(fft_size);
    }
};

// Simple pitch shift via resampling (no phase vocoder — fast approximation)
inline std::vector<f32> pitch_shift_resample(
    std::span<const f32> x, f32 ratio)
{
    if (std::abs(ratio-1.f) < 1e-6f) return std::vector<f32>(x.begin(),x.end());
    u32 out_len = u32(f32(x.size()) / ratio);
    std::vector<f32> out(out_len);
    for (u32 i = 0; i < out_len; ++i) {
        f32 src_idx = f32(i) * ratio;
        u32 lo = u32(src_idx);
        f32 frac = src_idx - f32(lo);
        u32 hi = std::min(lo+1, u32(x.size())-1);
        out[i] = x[lo]*(1.f-frac) + x[hi]*frac;
    }
    return out;
}

// ─── Vibrato / Tremolo ────────────────────────────────────────────────────────

struct VibratoParams {
    f32 rate{5.0f};        // Hz
    f32 depth{0.005f};     // delay modulation depth in seconds
    u32 max_delay_samps{1000};
};

struct VibratoState {
    std::vector<f32> delay_buf;
    u32 write_pos{};
    f32 lfo_phase{};
    VibratoParams p;

    void init(const VibratoParams& params) {
        p = params;
        delay_buf.assign(p.max_delay_samps, 0.f);
    }

    f32 process(f32 sample, u32 sample_rate) {
        delay_buf[write_pos] = sample;
        f32 lfo = std::sin(lfo_phase) * 0.5f + 0.5f;
        lfo_phase += 2.f*k_pi*p.rate/f32(sample_rate);
        if (lfo_phase > 2.f*k_pi) lfo_phase -= 2.f*k_pi;
        f32 delay = lfo * p.depth * f32(sample_rate);
        f32 read_pos = f32(write_pos) - delay;
        while (read_pos < 0.f) read_pos += f32(p.max_delay_samps);
        u32 lo = u32(read_pos) % p.max_delay_samps;
        u32 hi = (lo+1)%p.max_delay_samps;
        f32 frac = read_pos - std::floor(read_pos);
        write_pos = (write_pos+1)%p.max_delay_samps;
        return delay_buf[lo]*(1.f-frac)+delay_buf[hi]*frac;
    }
};

// ─── Chroma / pitch class features ───────────────────────────────────────────

// Map frequency to pitch class (0..11 = C..B)
[[nodiscard]] inline u32 freq_to_pitch_class(f32 freq_hz, f32 a4=440.f) {
    if (freq_hz <= 0.f) return 0;
    f32 midi = 69.f + 12.f*std::log2(freq_hz/a4);
    i32 semitone = i32(std::round(midi)) % 12;
    return u32((semitone+12)%12);
}

[[nodiscard]] inline f32 midi_to_freq(f32 midi, f32 a4=440.f) {
    return a4 * std::pow(2.f, (midi-69.f)/12.f);
}
[[nodiscard]] inline f32 freq_to_midi(f32 freq, f32 a4=440.f) {
    return 69.f + 12.f*std::log2(freq/a4);
}
[[nodiscard]] inline f32 cents_deviation(f32 freq, f32 a4=440.f) {
    f32 midi = freq_to_midi(freq, a4);
    return (midi - std::round(midi)) * 100.f;
}

} // namespace qs
