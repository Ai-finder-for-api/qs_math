#pragma once
// audio_math.hpp — Audio signal math: filters, FFT helpers, spatialization,
//                  HRTF approximations, dynamic range, envelope, pitch,
//                  psychoacoustics, procedural audio helpers
#include "vec3.hpp"
#include "vec2.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "constants.hpp"

namespace qs::math::audio {

static constexpr f32 SPEED_OF_SOUND = 343.0f; // m/s at 20°C
static constexpr f32 REF_PRESSURE   = 20e-6f;  // 20 μPa (threshold of hearing)
static constexpr f32 A4_FREQ        = 440.0f;

// ---------------------------------------------------------------------------
// dB / linear conversions
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 linear_to_db(f32 amplitude) noexcept {
    return 20.f*log10(max(amplitude,EPS_F32));
}
[[nodiscard]] constexpr f32 db_to_linear(f32 db) noexcept {
    return pow(10.f,db/20.f);
}
[[nodiscard]] constexpr f32 power_to_db(f32 power) noexcept {
    return 10.f*log10(max(power,EPS_F32));
}
[[nodiscard]] constexpr f32 spl_db(f32 pressure) noexcept {
    return 20.f*log10(pressure/REF_PRESSURE);
}

// ---------------------------------------------------------------------------
// Pitch / Frequency
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 midi_to_freq(i32 midi_note) noexcept {
    return A4_FREQ*pow(2.f,f32(midi_note-69)/12.f);
}
[[nodiscard]] constexpr f32 freq_to_midi(f32 freq) noexcept {
    return 69.f+12.f*log2(freq/A4_FREQ);
}
[[nodiscard]] constexpr f32 semitones_to_ratio(f32 semitones) noexcept {
    return pow(2.f,semitones/12.f);
}
[[nodiscard]] constexpr f32 cents_to_ratio(f32 cents) noexcept {
    return pow(2.f,cents/1200.f);
}
[[nodiscard]] constexpr f32 pitch_shift(f32 freq, f32 semitones) noexcept {
    return freq*semitones_to_ratio(semitones);
}

// Bark scale (critical bands)
[[nodiscard]] constexpr f32 hz_to_bark(f32 hz) noexcept {
    return 13.f*atan(0.00076f*hz)+3.5f*atan(pow(hz/7500.f,2.f));
}
// Mel scale
[[nodiscard]] constexpr f32 hz_to_mel(f32 hz) noexcept {
    return 2595.f*log10(1.f+hz/700.f);
}
[[nodiscard]] constexpr f32 mel_to_hz(f32 mel) noexcept {
    return 700.f*(pow(10.f,mel/2595.f)-1.f);
}

// ---------------------------------------------------------------------------
// Biquad filter coefficients (RBJ Audio EQ cookbook)
// ---------------------------------------------------------------------------
struct BiquadCoeffs { f32 b0,b1,b2,a1,a2; };
struct BiquadState  { f32 z1{0.f},z2{0.f}; };

// Low-pass
[[nodiscard]] inline BiquadCoeffs biquad_lpf(f32 freq, f32 sample_rate, f32 Q=0.707f) noexcept {
    f32 w=TWO_PI*freq/sample_rate, sinw=sin(w), cosw=cos(w);
    f32 a=sinw/(2.f*Q), inv=1.f/(1.f+a);
    return {(1.f-cosw)*0.5f*inv,(1.f-cosw)*inv,(1.f-cosw)*0.5f*inv,-2.f*cosw*inv,(1.f-a)*inv};
}

// High-pass
[[nodiscard]] inline BiquadCoeffs biquad_hpf(f32 freq, f32 sample_rate, f32 Q=0.707f) noexcept {
    f32 w=TWO_PI*freq/sample_rate, sinw=sin(w), cosw=cos(w);
    f32 a=sinw/(2.f*Q), inv=1.f/(1.f+a);
    return {(1.f+cosw)*0.5f*inv,-(1.f+cosw)*inv,(1.f+cosw)*0.5f*inv,-2.f*cosw*inv,(1.f-a)*inv};
}

// Band-pass
[[nodiscard]] inline BiquadCoeffs biquad_bpf(f32 freq, f32 sample_rate, f32 Q=1.f) noexcept {
    f32 w=TWO_PI*freq/sample_rate, sinw=sin(w), cosw=cos(w);
    f32 a=sinw/(2.f*Q), inv=1.f/(1.f+a);
    return {Q*a*inv,0.f,-Q*a*inv,-2.f*cosw*inv,(1.f-a)*inv};
}

// Peaking EQ
[[nodiscard]] inline BiquadCoeffs biquad_peak(
    f32 freq, f32 sample_rate, f32 Q, f32 gain_db) noexcept
{
    f32 A=pow(10.f,gain_db/40.f);
    f32 w=TWO_PI*freq/sample_rate, sinw=sin(w), cosw=cos(w);
    f32 a=sinw/(2.f*Q);
    f32 inv=1.f/(1.f+a/A);
    return {(1.f+a*A)*inv,-2.f*cosw*inv,(1.f-a*A)*inv,-2.f*cosw*inv,(1.f-a/A)*inv};
}

// Process one sample through biquad
[[nodiscard]] constexpr f32 biquad_process(
    BiquadCoeffs& c, BiquadState& s, f32 x) noexcept
{
    f32 y=c.b0*x+s.z1;
    s.z1=c.b1*x-c.a1*y+s.z2;
    s.z2=c.b2*x-c.a2*y;
    return y;
}

// ---------------------------------------------------------------------------
// Envelope (ADSR)
// ---------------------------------------------------------------------------
struct ADSREnvelope {
    f32 attack{0.01f},decay{0.1f},sustain{0.7f},release{0.2f};
    f32 time{0.f};
    bool gate{false};
    f32  release_start{0.f};
    f32  release_level{0.f};

    [[nodiscard]] f32 advance(f32 dt) noexcept {
        time+=dt;
        if(gate){
            if(time<attack) return time/attack;
            if(time<attack+decay) return 1.f-(1.f-sustain)*(time-attack)/decay;
            return sustain;
        } else {
            if(time>release) return 0.f;
            return release_level*(1.f-time/release);
        }
    }

    void note_on()  noexcept { gate=true; time=0.f; }
    void note_off() noexcept {
        gate=false; time=0.f;
        release_level=sustain;
    }
};

// ---------------------------------------------------------------------------
// 3D Audio spatialization
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 distance_attenuation(f32 dist, f32 ref_dist=1.f, f32 rolloff=1.f) noexcept {
    return ref_dist/max(ref_dist,dist*rolloff);
}

[[nodiscard]] constexpr f32 doppler_factor(
    Vec3f listener_pos, Vec3f listener_vel,
    Vec3f source_pos,   Vec3f source_vel,
    f32 speed_of_sound=SPEED_OF_SOUND) noexcept
{
    Vec3f dir=(listener_pos-source_pos).normalized();
    f32 vl=dot(listener_vel,dir);
    f32 vs=dot(source_vel, dir);
    return (speed_of_sound+vl)/(speed_of_sound+vs+EPS_F32);
}

// Panning law (constant power)
[[nodiscard]] inline Vec2f constant_power_pan(f32 pan) noexcept {
    f32 angle=pan*QUARTER_PI;
    return {cos(angle),sin(angle)};
}

// ITD (interaural time difference) in seconds
[[nodiscard]] inline f32 itd(Vec3f source_dir, f32 head_radius=0.0875f) noexcept {
    f32 azimuth=atan2(source_dir.x,source_dir.z);
    return (head_radius/SPEED_OF_SOUND)*(azimuth+sin(azimuth));
}

// ILD (interaural level difference) in dB
[[nodiscard]] inline f32 ild_db(Vec3f source_dir) noexcept {
    return 6.f*dot(source_dir,Vec3f{1.f,0.f,0.f});
}

// ---------------------------------------------------------------------------
// Compressor / Limiter
// ---------------------------------------------------------------------------
struct Compressor {
    f32 threshold{-20.f}; // dBFS
    f32 ratio{4.f};
    f32 attack{0.001f};   // seconds
    f32 release{0.1f};
    f32 makeup{0.f};      // dB
    f32 envelope{0.f};

    [[nodiscard]] f32 process(f32 sample, f32 dt) noexcept {
        f32 level=linear_to_db(abs(sample));
        f32 target=level>threshold?threshold+(level-threshold)/ratio:level;
        f32 gain_db=target-level;
        f32 tc=gain_db<envelope?attack:release;
        f32 coeff=exp(-1.f/(sample_rate_from_tc(tc,dt)));
        envelope=coeff*envelope+(1.f-coeff)*gain_db;
        return sample*db_to_linear(envelope+makeup);
    }
private:
    static constexpr f32 sample_rate_from_tc(f32 tc, f32 dt) noexcept {
        return max(tc/dt,EPS_F32);
    }
};

// ---------------------------------------------------------------------------
// Reverb helpers (Schroeder / Moorer)
// ---------------------------------------------------------------------------
struct CombFilter {
    static constexpr i32 MAX_DELAY=8192;
    f32 buf[MAX_DELAY]{}; i32 pos{0}; i32 delay_samples{0}; f32 feedback{0.f};
    void set(i32 delay, f32 fb) noexcept { delay_samples=min(delay,MAX_DELAY-1); feedback=fb; }
    [[nodiscard]] f32 process(f32 x) noexcept {
        f32 out=buf[pos]; buf[pos]=x+out*feedback;
        pos=(pos+1)%delay_samples; return out;
    }
};

struct AllpassFilter {
    static constexpr i32 MAX_DELAY=2048;
    f32 buf[MAX_DELAY]{}; i32 pos{0}; i32 delay_samples{0}; f32 coeff{0.5f};
    void set(i32 delay, f32 c) noexcept { delay_samples=min(delay,MAX_DELAY-1); coeff=c; }
    [[nodiscard]] f32 process(f32 x) noexcept {
        f32 d=buf[pos]; f32 out=-coeff*x+d;
        buf[pos]=x+coeff*d; pos=(pos+1)%delay_samples; return out;
    }
};

// ---------------------------------------------------------------------------
// Wavetable oscillator helpers
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 sine_osc(f32 phase) noexcept { return sin(phase*TWO_PI); }
[[nodiscard]] constexpr f32 square_osc(f32 phase, f32 pw=0.5f) noexcept {
    return phase<pw?1.f:-1.f;
}
[[nodiscard]] constexpr f32 saw_osc(f32 phase) noexcept { return 2.f*phase-1.f; }
[[nodiscard]] constexpr f32 triangle_osc(f32 phase) noexcept {
    return phase<0.5f?4.f*phase-1.f:3.f-4.f*phase;
}
[[nodiscard]] constexpr f32 pulse_osc(f32 phase, f32 duty) noexcept {
    return phase<duty?1.f:-1.f;
}

// PolyBLEP anti-aliasing for naive oscillators
[[nodiscard]] constexpr f32 polyblep(f32 t, f32 dt) noexcept {
    if(t<dt)  return  t/dt*2.f-1.f-(t/dt)*(t/dt);
    if(t>1.f-dt) { t=(t-1.f)/dt; return (t+1.f)*(t+1.f)+t*dt/dt; }
    return 0.f;
}

} // namespace qs::math::audio
