#pragma once
// qs_math/signal_graph_math.hpp — Audio graph nodes, filter chaining, routing matrices (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <memory>
#include <span>
#include <array>
#include <functional>

namespace qs {

// ─── Audio graph node ─────────────────────────────────────────────────────────

struct AudioBus {
    std::vector<f32> samples;
    u32 channels{2};
    u32 sample_rate{44100};
    u32 block_size{256};

    void resize(u32 bs) {
        block_size=bs;
        samples.assign(bs*channels,0.f);
    }
    void zero() { std::fill(samples.begin(),samples.end(),0.f); }
    [[nodiscard]] f32* channel(u32 ch) { return samples.data()+ch*block_size; }
};

// ─── Biquad filter (Direct Form II transposed) ────────────────────────────────

struct BiquadCoeffs {
    f32 b0{1},b1{0},b2{0},a1{0},a2{0};
};

struct BiquadState { f32 s1{0},s2{0}; };

[[nodiscard]] inline f32 biquad_process(f32 x, BiquadCoeffs& c, BiquadState& s) {
    f32 y=c.b0*x+s.s1;
    s.s1=c.b1*x-c.a1*y+s.s2;
    s.s2=c.b2*x-c.a2*y;
    return y;
}

inline void biquad_block(std::span<const f32> in, std::span<f32> out,
                          BiquadCoeffs& c, BiquadState& s)
{
    for(u32 i=0;i<u32(std::min(in.size(),out.size()));++i)
        out[i]=biquad_process(in[i],c,s);
}

// ─── Biquad design ────────────────────────────────────────────────────────────

[[nodiscard]] inline BiquadCoeffs biquad_lpf(f32 fc, f32 Q, f32 sr) {
    f32 w0=k_tau*fc/sr, cw=std::cos(w0), sw=std::sin(w0);
    f32 alpha=sw/(2.f*Q);
    f32 a0=1.f+alpha;
    return {(1.f-cw)*0.5f/a0,(1.f-cw)/a0,(1.f-cw)*0.5f/a0,-2.f*cw/a0,(1.f-alpha)/a0};
}

[[nodiscard]] inline BiquadCoeffs biquad_hpf(f32 fc, f32 Q, f32 sr) {
    f32 w0=k_tau*fc/sr, cw=std::cos(w0), sw=std::sin(w0);
    f32 alpha=sw/(2.f*Q);
    f32 a0=1.f+alpha;
    return {(1.f+cw)*0.5f/a0,-(1.f+cw)/a0,(1.f+cw)*0.5f/a0,-2.f*cw/a0,(1.f-alpha)/a0};
}

[[nodiscard]] inline BiquadCoeffs biquad_bpf(f32 fc, f32 Q, f32 sr) {
    f32 w0=k_tau*fc/sr, sw=std::sin(w0), cw=std::cos(w0);
    f32 alpha=sw/(2.f*Q), a0=1.f+alpha;
    return {Q*alpha/a0,0.f,-Q*alpha/a0,-2.f*cw/a0,(1.f-alpha)/a0};
}

[[nodiscard]] inline BiquadCoeffs biquad_peaking_eq(f32 fc, f32 Q, f32 dBgain, f32 sr) {
    f32 A=std::pow(10.f,dBgain/40.f);
    f32 w0=k_tau*fc/sr, cw=std::cos(w0), sw=std::sin(w0);
    f32 alpha=sw/(2.f*Q), a0=1.f+alpha/A;
    return {(1.f+alpha*A)/a0,-2.f*cw/a0,(1.f-alpha*A)/a0,-2.f*cw/a0,(1.f-alpha/A)/a0};
}

[[nodiscard]] inline BiquadCoeffs biquad_notch(f32 fc, f32 Q, f32 sr) {
    f32 w0=k_tau*fc/sr, cw=std::cos(w0), sw=std::sin(w0);
    f32 alpha=sw/(2.f*Q), a0=1.f+alpha;
    return {1.f/a0,-2.f*cw/a0,1.f/a0,-2.f*cw/a0,(1.f-alpha)/a0};
}

[[nodiscard]] inline BiquadCoeffs biquad_allpass(f32 fc, f32 Q, f32 sr) {
    f32 w0=k_tau*fc/sr, cw=std::cos(w0), sw=std::sin(w0);
    f32 alpha=sw/(2.f*Q), a0=1.f+alpha;
    return {(1.f-alpha)/a0,-2.f*cw/a0,(1.f+alpha)/a0,-2.f*cw/a0,(1.f-alpha)/a0};
}

// ─── Gain / pan matrix ────────────────────────────────────────────────────────

// Equal-power stereo panning
[[nodiscard]] inline std::pair<f32,f32> pan_equal_power(f32 pan) {
    // pan in [-1, 1]
    f32 angle=(pan+1.f)*0.25f*k_pi; // [0, pi/2]
    return {std::cos(angle), std::sin(angle)};
}

// N-channel routing matrix multiply: out[O] = M[O*I] * in[I]
inline void route(std::span<const f32> in, std::span<f32> out, std::span<const f32> M, u32 I, u32 O) {
    for(u32 o=0;o<O;++o){f32 s=0.f;for(u32 i=0;i<I;++i)s+=M[o*I+i]*in[i];out[o]=s;}
}

// ─── Dynamics processing ──────────────────────────────────────────────────────

struct CompressorState {
    f32 envelope{0.f};
    f32 gain_db{0.f};
};

struct CompressorParams {
    f32 threshold_db{-20.f};
    f32 ratio{4.f};
    f32 attack_ms{10.f};
    f32 release_ms{100.f};
    f32 makeup_db{0.f};
    f32 knee_db{6.f};
};

[[nodiscard]] inline f32 compress_sample(
    f32 x, CompressorState& s, const CompressorParams& p, u32 sr)
{
    f32 att=std::exp(-2.2f/(p.attack_ms*0.001f*f32(sr)));
    f32 rel=std::exp(-2.2f/(p.release_ms*0.001f*f32(sr)));
    f32 level=std::abs(x);
    s.envelope=level>s.envelope?att*s.envelope+(1.f-att)*level:rel*s.envelope+(1.f-rel)*level;
    f32 level_db=s.envelope>1e-7f?20.f*std::log10(s.envelope):-140.f;
    f32 over=level_db-p.threshold_db;
    f32 gr=0.f;
    if(over>p.knee_db*0.5f) gr=(1.f/p.ratio-1.f)*over;
    else if(over>-p.knee_db*0.5f) gr=(1.f/p.ratio-1.f)*(over+p.knee_db*0.5f)*(over+p.knee_db*0.5f)/(2.f*p.knee_db);
    s.gain_db=gr+p.makeup_db;
    return x*std::pow(10.f,s.gain_db/20.f);
}

// ─── Delay line ───────────────────────────────────────────────────────────────

struct DelayLine {
    std::vector<f32> buf;
    u32 write_pos{};

    void init(u32 max_samples) { buf.assign(max_samples,0.f); }

    void push(f32 x) { buf[write_pos]=x; write_pos=(write_pos+1)%u32(buf.size()); }

    [[nodiscard]] f32 read(u32 delay_samples) const {
        u32 read_pos=(write_pos+u32(buf.size())-delay_samples-1)%u32(buf.size());
        return buf[read_pos];
    }

    [[nodiscard]] f32 read_frac(f32 delay_samples) const {
        f32 rd=f32(write_pos)+f32(buf.size())-delay_samples-1.f;
        u32 i=u32(rd)%u32(buf.size()), j=(i+1)%u32(buf.size());
        f32 frac=rd-std::floor(rd);
        return buf[i]*(1.f-frac)+buf[j]*frac;
    }
};

// ─── Signal metrics ───────────────────────────────────────────────────────────

[[nodiscard]] inline f32 rms(std::span<const f32> x) {
    f32 s=0.f; for(auto v:x) s+=v*v;
    return std::sqrt(s/f32(x.size()+1));
}

[[nodiscard]] inline f32 peak(std::span<const f32> x) {
    f32 p=0.f; for(auto v:x) p=std::max(p,std::abs(v)); return p;
}

[[nodiscard]] inline f32 crest_factor(std::span<const f32> x) {
    f32 r=rms(x); return r>0.f?peak(x)/r:0.f;
}

[[nodiscard]] inline f32 loudness_k_weighted(std::span<const f32> x, u32 sr) {
    // Simplified K-weighting: high-shelf + high-pass
    BiquadCoeffs hs=biquad_peaking_eq(1500.f,0.7f,4.f,f32(sr));
    BiquadCoeffs hp=biquad_hpf(60.f,0.5f,f32(sr));
    BiquadState s1,s2;
    f32 sum=0.f;
    for(f32 v:x){
        v=biquad_process(v,hs,s1);
        v=biquad_process(v,hp,s2);
        sum+=v*v;
    }
    return -0.691f+10.f*std::log10(sum/f32(x.size()+1));
}

} // namespace qs
