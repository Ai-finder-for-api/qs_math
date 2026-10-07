#pragma once
// qs_math/signal/audio_dsp.hpp — Audio DSP: pitch detection, spectral analysis,
//   resampling, audio effects (reverb, chorus, flanger, compressor, limiter),
//   psychoacoustics, ADSR envelope, wavetable synthesis (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <span>

namespace qs::math::signal {

// ─── ADSR Envelope ───────────────────────────────────────────────────────
struct ADSR {
    f32 attack_s=0.01f, decay_s=0.1f, sustain=0.7f, release_s=0.3f;
    f32 sample_rate=44100.f;

    enum class Phase { Attack, Decay, Sustain, Release, Done };

    struct State {
        Phase phase=Phase::Done;
        f32   level=0.f, counter=0.f;
    };

    void note_on(State& s) const noexcept {
        s.phase=Phase::Attack; s.counter=0.f;
    }
    void note_off(State& s) const noexcept {
        if(s.phase!=Phase::Done){ s.phase=Phase::Release; s.counter=0.f; }
    }

    [[nodiscard]] f32 tick(State& s) const noexcept {
        f32 dt=1.f/sample_rate;
        switch(s.phase){
            case Phase::Attack:
                s.level=s.counter/attack_s;
                s.counter+=dt;
                if(s.counter>=attack_s){s.level=1.f;s.phase=Phase::Decay;s.counter=0.f;}
                break;
            case Phase::Decay:
                s.level=1.f-(1.f-sustain)*s.counter/decay_s;
                s.counter+=dt;
                if(s.counter>=decay_s){s.level=sustain;s.phase=Phase::Sustain;}
                break;
            case Phase::Sustain:
                s.level=sustain; break;
            case Phase::Release:
                s.level=sustain*(1.f-s.counter/release_s);
                s.counter+=dt;
                if(s.counter>=release_s){s.level=0.f;s.phase=Phase::Done;}
                break;
            case Phase::Done:
                s.level=0.f; break;
        }
        return s.level;
    }
};

// ─── Wavetable synthesis ──────────────────────────────────────────────────
struct Wavetable {
    std::vector<f32> table;
    int size;

    static Wavetable sine(int n=2048) noexcept {
        Wavetable wt; wt.size=n; wt.table.resize(n);
        for(int i=0;i<n;++i) wt.table[i]=std::sin(f32(TWO_PI)*i/n);
        return wt;
    }
    static Wavetable square(int n=2048) noexcept {
        Wavetable wt; wt.size=n; wt.table.resize(n);
        for(int i=0;i<n;++i) wt.table[i]=i<n/2?1.f:-1.f;
        return wt;
    }
    static Wavetable sawtooth(int n=2048) noexcept {
        Wavetable wt; wt.size=n; wt.table.resize(n);
        for(int i=0;i<n;++i) wt.table[i]=2.f*i/n-1.f;
        return wt;
    }
    static Wavetable triangle(int n=2048) noexcept {
        Wavetable wt; wt.size=n; wt.table.resize(n);
        for(int i=0;i<n;++i) wt.table[i]=(i<n/2)?4.f*i/n-1.f:3.f-4.f*i/n;
        return wt;
    }

    [[nodiscard]] f32 sample(f32 phase) const noexcept {
        // phase in [0,1)
        f32 idx=phase*size;
        int i=int(idx)%size;
        f32 frac=idx-i;
        return table[i]*(1.f-frac)+table[(i+1)%size]*frac;
    }
};

struct WavetableOsc {
    Wavetable wt;
    f32 phase=0.f, freq=440.f, sample_rate=44100.f;

    [[nodiscard]] f32 tick() noexcept {
        f32 s=wt.sample(phase);
        phase+=freq/sample_rate;
        if(phase>=1.f) phase-=1.f;
        return s;
    }
    void set_freq(f32 f) noexcept { freq=f; }
    void reset() noexcept { phase=0.f; }
};

// ─── Pitch detection (YIN algorithm) ──────────────────────────────────────
[[nodiscard]] inline std::vector<f32> yin_difference(std::span<const f32> x, int W) noexcept {
    int n=int(x.size());
    std::vector<f32> d(W,0.f);
    for(int tau=1;tau<W;++tau){
        for(int j=0;j<W;++j){
            f32 delta=x[j]-x[j+tau];
            d[tau]+=delta*delta;
        }
    }
    return d;
}

[[nodiscard]] inline std::vector<f32> yin_cmnd(const std::vector<f32>& d) noexcept {
    int W=int(d.size());
    std::vector<f32> cmnd(W,0.f);
    cmnd[0]=1.f;
    f32 sum=0;
    for(int tau=1;tau<W;++tau){
        sum+=d[tau];
        cmnd[tau]=(sum>0.f)?d[tau]*tau/sum:1.f;
    }
    return cmnd;
}

[[nodiscard]] inline f32 yin_pitch(
    std::span<const f32> x, f32 sample_rate, f32 threshold=0.1f) noexcept
{
    int W=int(x.size())/2;
    auto d=yin_difference(x,W);
    auto cmnd=yin_cmnd(d);
    // Find first dip below threshold
    for(int tau=2;tau<W-1;++tau){
        if(cmnd[tau]<threshold&&cmnd[tau]<cmnd[tau-1]&&cmnd[tau]<cmnd[tau+1]){
            // Parabolic interpolation
            f32 s0=cmnd[tau-1],s1=cmnd[tau],s2=cmnd[tau+1];
            f32 ftau=tau+0.5f*(s0-s2)/(s0-2*s1+s2);
            return sample_rate/ftau;
        }
    }
    return -1.f; // no pitch found
}

// ─── Autocorrelation pitch (simpler) ─────────────────────────────────────
[[nodiscard]] inline f32 autocorr_pitch(
    std::span<const f32> x, f32 sr, f32 lo_hz=80.f, f32 hi_hz=1000.f) noexcept
{
    int n=int(x.size());
    int lag_lo=int(sr/hi_hz), lag_hi=int(sr/lo_hz);
    lag_hi=std::min(lag_hi,n/2);
    f32 best_r=-1e30f; int best_lag=lag_lo;
    for(int lag=lag_lo;lag<lag_hi;++lag){
        f32 r=0;
        for(int i=0;i<n-lag;++i) r+=x[i]*x[i+lag];
        if(r>best_r){best_r=r;best_lag=lag;}
    }
    return sr/f32(best_lag);
}

// ─── Dynamics processing ──────────────────────────────────────────────────
struct Compressor {
    f32 threshold_db=-12.f, ratio=4.f, knee_db=3.f;
    f32 attack_s=0.003f, release_s=0.1f;
    f32 makeup_db=0.f, sample_rate=44100.f;
    f32 gain_db=0.f; // state

    [[nodiscard]] f32 process(f32 x) noexcept {
        f32 input_db=x!=0.f?20.f*std::log10(std::abs(x)):-144.f;
        // Compute gain reduction
        f32 gc=0.f;
        f32 overshoot=input_db-threshold_db;
        if(2*overshoot<-knee_db) gc=0.f;
        else if(2*std::abs(overshoot)<knee_db)
            gc=(1.f/ratio-1.f)*(overshoot+knee_db*0.5f)*(overshoot+knee_db*0.5f)/(2.f*knee_db);
        else gc=overshoot*(1.f/ratio-1.f);
        // Smooth gain
        f32 target=gc;
        f32 tc=target<gain_db?attack_s:release_s;
        f32 alpha=std::exp(-1.f/(tc*sample_rate));
        gain_db=alpha*gain_db+(1.f-alpha)*target;
        return x*std::pow(10.f,(gain_db+makeup_db)/20.f);
    }
};

struct Limiter {
    f32 threshold_db=0.f, release_s=0.1f, sample_rate=44100.f;
    f32 gain=1.f;

    [[nodiscard]] f32 process(f32 x) noexcept {
        f32 threshold=std::pow(10.f,threshold_db/20.f);
        f32 peak=std::abs(x)*gain;
        if(peak>threshold) gain*=threshold/peak;
        f32 alpha=std::exp(-1.f/(release_s*sample_rate));
        gain=std::min(1.f,gain+(1.f-alpha)*(1.f-gain));
        return x*gain;
    }
};

struct NoiseGate {
    f32 threshold_db=-40.f, ratio=100.f, attack_s=0.001f, release_s=0.05f, sample_rate=44100.f;
    f32 level=0.f;

    [[nodiscard]] f32 process(f32 x) noexcept {
        f32 env=std::abs(x);
        f32 tc=env>level?attack_s:release_s;
        f32 alpha=std::exp(-1.f/(tc*sample_rate));
        level=alpha*level+(1.f-alpha)*env;
        f32 threshold=std::pow(10.f,threshold_db/20.f);
        f32 gain=level>threshold?1.f:level/threshold/ratio;
        return x*gain;
    }
};

// ─── Delay line ───────────────────────────────────────────────────────────
struct DelayLine {
    std::vector<f32> buf;
    int write_pos=0;

    DelayLine(int max_delay_samples) : buf(max_delay_samples,0.f) {}

    void push(f32 x) noexcept { buf[write_pos]=x; write_pos=(write_pos+1)%int(buf.size()); }

    [[nodiscard]] f32 read(int delay_samples) const noexcept {
        int idx=(write_pos-delay_samples+int(buf.size()))%int(buf.size());
        return buf[idx];
    }

    [[nodiscard]] f32 read_linear(f32 delay_samples) const noexcept {
        int di=int(delay_samples); f32 frac=delay_samples-di;
        return read(di)*(1.f-frac)+read(di+1)*frac;
    }
};

// ─── Reverb (Schroeder) ───────────────────────────────────────────────────
struct SchroederReverb {
    static constexpr int N_COMB=4, N_AP=2;
    std::array<DelayLine,N_COMB> combs;
    std::array<DelayLine,N_AP>   allpasses;
    f32 feedback=0.84f, ap_gain=0.7f;

    SchroederReverb(f32 sr=44100.f) :
        combs{DelayLine(int(0.0297f*sr)),DelayLine(int(0.0371f*sr)),
              DelayLine(int(0.0411f*sr)),DelayLine(int(0.0437f*sr))},
        allpasses{DelayLine(int(0.005f*sr)),DelayLine(int(0.0017f*sr))} {}

    [[nodiscard]] f32 process(f32 x) noexcept {
        // Parallel comb filters
        f32 sum=0.f;
        for(auto& c:combs){
            f32 delayed=c.read(int(c.buf.size())-1);
            c.push(x+feedback*delayed);
            sum+=delayed;
        }
        sum/=N_COMB;
        // Series allpass filters
        for(auto& ap:allpasses){
            f32 delayed=ap.read(int(ap.buf.size())-1);
            f32 out=-ap_gain*sum+delayed;
            ap.push(sum+ap_gain*out);
            sum=out;
        }
        return sum;
    }
};

// ─── Chorus / Flanger ─────────────────────────────────────────────────────
struct Chorus {
    DelayLine delay;
    f32 rate_hz=0.5f, depth_ms=15.f, mix=0.5f;
    f32 lfo_phase=0.f, sample_rate=44100.f;

    Chorus(f32 sr=44100.f) : delay(int(50e-3f*sr)+4), sample_rate(sr) {}

    [[nodiscard]] f32 process(f32 x) noexcept {
        f32 lfo=std::sin(f32(TWO_PI)*lfo_phase);
        lfo_phase+=rate_hz/sample_rate;
        if(lfo_phase>=1.f) lfo_phase-=1.f;
        f32 delay_s=(depth_ms*0.001f)*(0.5f+0.5f*lfo);
        f32 delay_samp=delay_s*sample_rate;
        delay.push(x);
        f32 wet=delay.read_linear(delay_samp+1.f);
        return x*(1.f-mix)+wet*mix;
    }
};

struct Flanger {
    DelayLine delay;
    f32 rate_hz=0.25f, depth_ms=5.f, feedback=0.7f, mix=0.5f;
    f32 lfo_phase=0.f, sample_rate=44100.f;

    Flanger(f32 sr=44100.f) : delay(int(20e-3f*sr)+4), sample_rate(sr) {}

    [[nodiscard]] f32 process(f32 x) noexcept {
        f32 lfo=0.5f+0.5f*std::sin(f32(TWO_PI)*lfo_phase);
        lfo_phase+=rate_hz/sample_rate;
        if(lfo_phase>=1.f) lfo_phase-=1.f;
        f32 delay_samp=lfo*depth_ms*0.001f*sample_rate;
        f32 wet=delay.read_linear(delay_samp+1.f);
        delay.push(x+feedback*wet);
        return x*(1.f-mix)+wet*mix;
    }
};

// ─── Psychoacoustics ──────────────────────────────────────────────────────
// A-weighting filter coefficient (approximate)
[[nodiscard]] inline f32 a_weight(f32 freq_hz) noexcept {
    f32 f2=freq_hz*freq_hz;
    f32 f4=f2*f2;
    f32 ra=12194.217f*12194.217f*f4;
    f32 rb=(f2+20.598997f*20.598997f)*(f2+12194.217f*12194.217f);
    f32 rc=std::sqrt((f2+107.65265f*107.65265f)*(f2+737.86223f*737.86223f));
    if(rb*rc<1e-30f) return 0.f;
    return ra/(rb*rc)*1.0589254f; // +2.0 dB normalization
}

// Equal loudness contour (ISO 226, simplified Fletcher-Munson)
[[nodiscard]] inline f32 equal_loudness_phon(f32 freq_hz, f32 spl_db) noexcept {
    // Approximate: loudness in phon at given SPL and frequency
    f32 af=a_weight(freq_hz);
    if(af<1e-10f) return 0.f;
    f32 af_db=20.f*std::log10(af);
    return spl_db+af_db;
}

// Bark scale (psychoacoustic frequency scale)
[[nodiscard]] inline f32 hz_to_bark(f32 f) noexcept {
    return 13.f*std::atan(0.00076f*f)+3.5f*std::atan((f/7500.f)*(f/7500.f));
}
[[nodiscard]] inline f32 bark_to_hz(f32 b) noexcept {
    // Approximate inverse
    return 600.f*std::sinh(b/6.f);
}

// Mel scale
[[nodiscard]] inline f32 hz_to_mel(f32 f) noexcept { return 2595.f*std::log10(1.f+f/700.f); }
[[nodiscard]] inline f32 mel_to_hz(f32 m) noexcept { return 700.f*(std::pow(10.f,m/2595.f)-1.f); }

// ─── Mel filterbank ───────────────────────────────────────────────────────
[[nodiscard]] inline std::vector<std::vector<f32>> mel_filterbank(
    int n_filters, int n_fft, f32 sample_rate, f32 lo_hz=80.f, f32 hi_hz=8000.f) noexcept
{
    f32 lo_mel=hz_to_mel(lo_hz), hi_mel=hz_to_mel(hi_hz);
    std::vector<f32> mel_points(n_filters+2);
    for(int i=0;i<n_filters+2;++i)
        mel_points[i]=mel_to_hz(lo_mel+f32(i)*(hi_mel-lo_mel)/(n_filters+1));
    std::vector<int> fft_bins(n_filters+2);
    for(int i=0;i<n_filters+2;++i)
        fft_bins[i]=int(std::floor((n_fft+1)*mel_points[i]/sample_rate));
    std::vector<std::vector<f32>> fb(n_filters,std::vector<f32>(n_fft/2+1,0.f));
    for(int m=1;m<=n_filters;++m){
        for(int k=fft_bins[m-1];k<fft_bins[m];++k)
            fb[m-1][k]=(k-fft_bins[m-1])/f32(fft_bins[m]-fft_bins[m-1]);
        for(int k=fft_bins[m];k<fft_bins[m+1];++k)
            fb[m-1][k]=(fft_bins[m+1]-k)/f32(fft_bins[m+1]-fft_bins[m]);
    }
    return fb;
}

// ─── Resampling (linear) ─────────────────────────────────────────────────
[[nodiscard]] inline std::vector<f32> resample_linear(
    std::span<const f32> x, f32 ratio) noexcept
{
    int n=int(x.size()), m=int(n/ratio);
    std::vector<f32> y(m);
    for(int i=0;i<m;++i){
        f32 src=f32(i)*ratio;
        int lo=int(src); f32 frac=src-lo;
        int hi=std::min(lo+1,n-1);
        y[i]=x[lo]*(1.f-frac)+x[hi]*frac;
    }
    return y;
}

// Sinc interpolation (windowed)
[[nodiscard]] inline f32 sinc_interp(std::span<const f32> x, f32 t, int taps=32) noexcept {
    f32 sum=0,wsum=0;
    int center=int(t);
    for(int k=center-taps/2;k<=center+taps/2;++k){
        if(k<0||k>=(int)x.size()) continue;
        f32 dt=t-k;
        f32 sinc_val=(std::abs(dt)<1e-6f)?1.f:std::sin(f32(PI)*dt)/(f32(PI)*dt);
        // Hann window
        f32 w=0.5f+0.5f*std::cos(f32(PI)*dt/(taps/2.f));
        sum+=x[k]*sinc_val*w; wsum+=std::abs(sinc_val*w);
    }
    return wsum>1e-10f?sum:0.f;
}

} // namespace qs::math::signal
