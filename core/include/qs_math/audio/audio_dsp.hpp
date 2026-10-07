#pragma once
// qs_math/audio/audio_dsp.hpp — Room acoustics, reverb math, psychoacoustics, synthesis
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Psychoacoustic constants ─────────────────────────────────────────────────
inline constexpr f64 k_hearing_threshold_spl = 0.0;    // dB SPL (0 dB re 20µPa)
inline constexpr f64 k_pain_threshold_spl    = 130.0;  // dB SPL
inline constexpr f64 k_ref_pressure          = 20e-6;  // Pa (reference)
inline constexpr f64 k_a4_freq              = 440.0;   // Hz

// ─── dB conversions ───────────────────────────────────────────────────────────
[[nodiscard]] constexpr f64 linear_to_db(f64 linear) noexcept {
    return 20*std::log10(std::max(linear,1e-100));
}
[[nodiscard]] constexpr f64 db_to_linear(f64 db) noexcept {
    return std::pow(10.0,db/20.0);
}
[[nodiscard]] constexpr f64 power_to_db(f64 power) noexcept {
    return 10*std::log10(std::max(power,1e-100));
}
[[nodiscard]] constexpr f64 spl_from_pressure(f64 p_pa) noexcept {
    return 20*std::log10(std::abs(p_pa)/k_ref_pressure+1e-100);
}

// ─── Musical pitch ────────────────────────────────────────────────────────────
// MIDI note → frequency: f = 440 * 2^{(note-69)/12}
[[nodiscard]] constexpr f64 midi_to_hz(int midi_note) noexcept {
    return k_a4_freq*std::pow(2.0,(midi_note-69)/12.0);
}
[[nodiscard]] constexpr f64 hz_to_midi(f64 freq) noexcept {
    return 69+12*std::log2(freq/k_a4_freq);
}
[[nodiscard]] constexpr f64 semitones_to_ratio(f64 semitones) noexcept {
    return std::pow(2.0,semitones/12.0);
}
[[nodiscard]] constexpr f64 cents_to_ratio(f64 cents) noexcept {
    return std::pow(2.0,cents/1200.0);
}

// ─── Equal loudness (Fletcher-Munson, approximation) ─────────────────────────
// Returns SPL needed for equal loudness at given frequency (phon level)
[[nodiscard]] constexpr f64 equal_loudness_spl(f64 freq_hz, f64 phon) noexcept {
    // Simple approximation based on 1kHz reference
    f64 T=3.64*std::pow(freq_hz*0.001,-0.8)-6.5*std::exp(-0.6*(freq_hz*0.001-3.3)*(freq_hz*0.001-3.3))
          +1e-3*std::pow(freq_hz*0.001,4);
    return phon+T-T; // simplified: just return phon for now, fuller impl needs ISO226 table
}

// Critical bandwidth (Bark scale)
[[nodiscard]] constexpr f64 hz_to_bark(f64 f) noexcept {
    return 13*std::atan(0.00076*f)+3.5*std::atan((f/7500.0)*(f/7500.0));
}
[[nodiscard]] constexpr f64 bark_bandwidth_hz(f64 bark) noexcept {
    return 52548.0/(bark*bark-52.56*bark+690.39);
}
[[nodiscard]] constexpr f64 hz_to_mel(f64 f) noexcept {
    return 2595*std::log10(1+f/700.0);
}
[[nodiscard]] constexpr f64 mel_to_hz(f64 m) noexcept {
    return 700*(std::pow(10.0,m/2595.0)-1);
}

// ─── Room acoustics ───────────────────────────────────────────────────────────
// Sabine's formula: T60 = 0.161 * V / (S * α_avg)
[[nodiscard]] constexpr f64 t60_sabine(f64 volume, f64 surface_area, f64 avg_absorption) noexcept {
    return 0.161*volume/(surface_area*avg_absorption+1e-12);
}
// Eyring's formula (better for high absorption)
[[nodiscard]] constexpr f64 t60_eyring(f64 volume, f64 surface_area, f64 avg_absorption) noexcept {
    return -0.161*volume/(surface_area*std::log(1-avg_absorption+1e-12));
}
// Schroeder frequency: room modes → diffuse field
[[nodiscard]] constexpr f64 schroeder_freq(f64 t60, f64 volume) noexcept {
    return 2000*std::sqrt(t60/volume);
}
// Image source method: distance to image source for single wall
[[nodiscard]] constexpr f64 image_source_distance(
    f64 sx, f64 sy, f64 rx, f64 ry, f64 wall_x) noexcept
{
    f64 img_x=2*wall_x-sx;
    f64 dx=img_x-rx,dy=sy-ry;
    return std::sqrt(dx*dx+dy*dy);
}
// Air absorption coefficient (Stokes equation approximation)
[[nodiscard]] constexpr f64 air_absorption_db_per_m(f64 freq, f64 temp_c=20, f64 humidity=0.5) noexcept {
    // Simplified: proportional to f²
    f64 f2=freq*freq*1e-6;
    return 0.0185*f2*(1+0.015*(temp_c-20));
}

// ─── Reverb algorithms ────────────────────────────────────────────────────────
// Schroeder all-pass (comment: T60 from g and delay length)
[[nodiscard]] constexpr f64 allpass_t60(f64 g, int delay_samples, f64 sample_rate) noexcept {
    return -3*delay_samples/(sample_rate*std::log10(std::abs(g)+1e-15));
}
// Feedback delay network: mixing matrix (Hadamard)
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N> hadamard_fdn() noexcept {
    static_assert((N&(N-1))==0,"N must be power of 2");
    std::array<std::array<f64,N>,N> H{};
    // Build recursively
    H[0][0]=1;
    int sz=1;
    while(sz<N) {
        for(int i=0;i<sz;++i) for(int j=0;j<sz;++j) {
            H[i+sz][j]=H[i][j]; H[i][j+sz]=H[i][j];
            H[i+sz][j+sz]=-H[i][j];
        }
        sz<<=1;
    }
    f64 inv=1.0/std::sqrt(f64(N));
    for(auto& r:H) for(auto& v:r) v*=inv;
    return H;
}

// ─── Wavetable synthesis ──────────────────────────────────────────────────────
template<int TABLE_SIZE>
struct WavetableOsc {
    std::array<f64,TABLE_SIZE> table{};
    f64 phase{0},phase_inc{0};

    constexpr void set_frequency(f64 freq, f64 sample_rate) noexcept {
        phase_inc=freq*TABLE_SIZE/sample_rate;
    }
    [[nodiscard]] constexpr f64 next_sample() noexcept {
        int idx=int(phase)%TABLE_SIZE;
        f64 frac=phase-int(phase);
        f64 s=table[idx]*(1-frac)+table[(idx+1)%TABLE_SIZE]*frac;
        phase+=phase_inc; if(phase>=TABLE_SIZE) phase-=TABLE_SIZE;
        return s;
    }
    // Fill with sine wave
    constexpr void fill_sine() noexcept {
        for(int i=0;i<TABLE_SIZE;++i)
            table[i]=std::sin(2*3.14159265358979323846*i/TABLE_SIZE);
    }
    // Fill with band-limited sawtooth (additive synthesis)
    constexpr void fill_bandlimited_saw(f64 sample_rate, f64 base_freq) noexcept {
        table.fill(0);
        int max_harmonics=int(sample_rate/(2*base_freq));
        for(int k=1;k<=max_harmonics;++k)
            for(int i=0;i<TABLE_SIZE;++i)
                table[i]+=std::sin(2*3.14159265358979323846*k*i/TABLE_SIZE)*(k%2==0?-1:1)/k;
    }
};

// ─── Envelope (ADSR) ──────────────────────────────────────────────────────────
struct ADSREnvelope {
    f64 attack_time{0.01},decay_time{0.1},sustain_level{0.7},release_time{0.3};
    f64 sample_rate{44100};
    f64 level{0},target{0},rate{0};
    enum class State{Idle,Attack,Decay,Sustain,Release}state{State::Idle};

    [[nodiscard]] constexpr f64 next_sample() noexcept {
        switch(state) {
            case State::Attack:
                level+=1.0/(attack_time*sample_rate);
                if(level>=1.0){level=1.0;target=sustain_level;rate=(1-sustain_level)/(decay_time*sample_rate);state=State::Decay;}
                break;
            case State::Decay:
                level-=rate;
                if(level<=sustain_level){level=sustain_level;state=State::Sustain;}
                break;
            case State::Sustain: break;
            case State::Release:
                level-=level/(release_time*sample_rate);
                if(level<1e-6){level=0;state=State::Idle;}
                break;
            default: break;
        }
        return level;
    }
    constexpr void note_on() noexcept { state=State::Attack; }
    constexpr void note_off() noexcept { if(state!=State::Idle)state=State::Release; }
};

// ─── Spectral envelope (LPC) ──────────────────────────────────────────────────
// Autocorrelation of signal (lag 0..P)
template<int N, int P>
[[nodiscard]] constexpr std::array<f64,P+1> autocorrelation(
    const std::array<f64,N>& x) noexcept
{
    std::array<f64,P+1> r{};
    for(int lag=0;lag<=P;++lag)
        for(int i=0;i<N-lag;++i) r[lag]+=x[i]*x[i+lag];
    return r;
}
} // namespace qs::math
