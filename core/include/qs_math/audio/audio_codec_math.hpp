#pragma once
// qs_math/audio_codec_math.hpp — Sample format conversion, LUFS loudness, psychoacoustic masking (C++20)
#include "scalar_types.hpp"
#include "audio_math.hpp"
#include "signal_processing.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <numbers>

namespace qs {

// ─── Sample format conversion ─────────────────────────────────────────────────
// 16-bit PCM ↔ float
[[nodiscard]] inline f32 pcm16_to_float(i16 s) { return f32(s)/32768.f; }
[[nodiscard]] inline i16 float_to_pcm16(f32 s) { return i16(std::clamp(i32(s*32768.f),-32768,32767)); }

// 24-bit PCM (packed in i32) ↔ float
[[nodiscard]] inline f32 pcm24_to_float(i32 s) { return f32(s)/8388608.f; }
[[nodiscard]] inline i32 float_to_pcm24(f32 s) { return i32(std::clamp(f64(s)*8388608.0,-8388608.0,8388607.0)); }

// 32-bit integer PCM ↔ float
[[nodiscard]] inline f32 pcm32_to_float(i32 s) { return f32(f64(s)/2147483648.0); }
[[nodiscard]] inline i32 float_to_pcm32(f32 s) { return i32(std::clamp(f64(s)*2147483648.0,-2147483648.0,2147483647.0)); }

// μ-law (G.711) encode/decode
[[nodiscard]] inline u8 ulaw_encode(i16 pcm) {
    const i16 BIAS=0x84;
    bool sign=pcm<0; if(sign) pcm=-pcm;
    pcm=std::min(i16(32767),i16(pcm+BIAS));
    u8 exp=7; for(;exp>0&&(pcm&(1<<(exp+3)))==0;--exp);
    u8 mantissa=(pcm>>(exp+3))&0x0F;
    return ~((sign?0x80u:0u)|(u8(exp)<<4)|mantissa);
}
[[nodiscard]] inline i16 ulaw_decode(u8 u) {
    u=~u;
    bool sign=(u&0x80)!=0;
    u8 exp=(u>>4)&0x07;
    i16 mantissa=(u&0x0F)<<3;
    i16 val=i16(((mantissa+0x84)<<exp)-0x84);
    return sign?-val:val;
}

// A-law (G.711) encode/decode
[[nodiscard]] inline u8 alaw_encode(i16 pcm) {
    bool sign=pcm>=0; if(!sign) pcm=-pcm-1;
    u8 exp=0; i16 p=pcm;
    if(p>=256){ for(exp=1;exp<8&&p>=512;++exp) p>>=1; }
    u8 mantissa=(exp==0)?(u8(pcm>>1)&0x0F):(u8(p)&0x0F);
    return (u8((sign?0x80u:0u)|(u8(exp)<<4)|mantissa))^0x55u;
}
[[nodiscard]] inline i16 alaw_decode(u8 a) {
    a^=0x55u;
    bool sign=(a&0x80)!=0;
    u8 exp=(a>>4)&0x07;
    i16 mantissa=(a&0x0F)<<1|1;
    i16 val=exp==0?mantissa:i16((mantissa+32)<<(exp-1));
    return sign?val:-val-1;
}

// Dithering: TPDF (triangular probability density function)
[[nodiscard]] inline f32 tpdf_dither(f32 sample, f32 bit_depth=16, u32& seed=*(u32*)nullptr) {
    // Simpler: just add small noise
    static u32 s=12345;
    s=s*1664525u+1013904223u;
    f32 r1=(s>>9|0x3F800000u)-1.f;
    s=s*1664525u+1013904223u;
    f32 r2=(s>>9|0x3F800000u)-1.f;
    f32 quant=2.f/std::pow(2.f,bit_depth);
    return sample+(r1-r2)*quant*0.5f;
}

// ─── Resampling ───────────────────────────────────────────────────────────────
// Sinc resampling (windowed)
[[nodiscard]] inline f64 sinc_window(f64 x, u32 taps=8) {
    if(std::abs(x)<1e-10) return 1;
    f64 pi_x=std::numbers::pi_v<f64>*x;
    f64 sinc=std::sin(pi_x)/pi_x;
    // Kaiser window approximation
    f64 t=x/taps;
    f64 window=std::abs(t)<=1?std::sin(std::numbers::pi_v<f64>*t+1e-10)/(std::numbers::pi_v<f64>*t+1e-10):0;
    return sinc*window;
}

[[nodiscard]] inline std::vector<f32> resample_sinc(const std::vector<f32>& input,
    f64 src_rate, f64 dst_rate, u32 taps=32)
{
    f64 ratio=dst_rate/src_rate;
    u32 n_out=u32(input.size()*ratio);
    std::vector<f32> output(n_out);
    f64 cutoff=std::min(1.0,ratio); // anti-alias
    for(u32 i=0;i<n_out;++i) {
        f64 src_pos=i/ratio;
        f64 sum=0,norm=0;
        for(i32 k=-i32(taps);k<=i32(taps);++k) {
            f64 sx=src_pos+k;
            if(sx<0||sx>=input.size()) continue;
            f64 w=sinc_window((src_pos-sx)*cutoff,taps);
            sum+=input[u32(sx)]*w; norm+=w;
        }
        output[i]=f32(norm>1e-10?sum/norm:0);
    }
    return output;
}

// ─── LUFS loudness (EBU R128 / ITU-R BS.1770) ─────────────────────────────────
// K-weighting filter coefficients (pre-filter stage 1 and stage 2)
struct KWeightFilter {
    // Stage 1: high-shelf pre-filter
    f64 b0s1{1.53512485958697},b1s1{-2.69169618940638},b2s1{1.19839281085285};
    f64 a1s1{-1.69065929318241},a2s1{0.73248077421585};
    // Stage 2: high-pass filter
    f64 b0s2{1.0},b1s2{-2.0},b2s2{1.0};
    f64 a1s2{-1.99004745483398},a2s2{0.99007225036621};

    // Apply to a single channel
    [[nodiscard]] std::vector<f64> apply(const std::vector<f64>& x) const {
        u32 n=u32(x.size());
        // Stage 1
        std::vector<f64> y1(n,0); f64 z1[2]={0,0};
        for(u32 i=0;i<n;++i) {
            f64 w=x[i]-a1s1*z1[0]-a2s1*z1[1];
            y1[i]=b0s1*w+b1s1*z1[0]+b2s1*z1[1];
            z1[1]=z1[0]; z1[0]=w;
        }
        // Stage 2
        std::vector<f64> y2(n,0); f64 z2[2]={0,0};
        for(u32 i=0;i<n;++i) {
            f64 w=y1[i]-a1s2*z2[0]-a2s2*z2[1];
            y2[i]=b0s2*w+b1s2*z2[0]+b2s2*z2[1];
            z2[1]=z2[0]; z2[0]=w;
        }
        return y2;
    }
};

struct LUFSResult {
    f64 momentary{0};     // 400ms window, LUFS
    f64 short_term{0};    // 3s window, LUFS
    f64 integrated{0};    // gated integrated, LUFS
    f64 lra{0};           // loudness range, LU
    f64 peak_dbfs{0};     // true peak, dBFS
};

[[nodiscard]] inline LUFSResult compute_lufs(const std::vector<std::vector<f64>>& channels,
    f64 sample_rate=48000)
{
    LUFSResult result;
    u32 nch=u32(channels.size());
    if(nch==0) return result;
    u32 n=u32(channels[0].size());
    // K-weight each channel
    KWeightFilter filt;
    std::vector<std::vector<f64>> kw(nch);
    for(u32 ch=0;ch<nch;++ch) kw[ch]=filt.apply(channels[ch]);
    // Mean square per channel (with channel gain: L,R=1, C=1, LS,RS=1.41)
    std::vector<f64> gain={1,1,1,1.41,1.41};
    // Block processing: 400ms blocks with 75% overlap
    u32 block_size=u32(sample_rate*0.4);
    u32 hop=u32(block_size*0.25);
    std::vector<f64> block_lk;
    for(u32 start=0;start+block_size<=n;start+=hop) {
        f64 power=0;
        for(u32 ch=0;ch<nch;++ch) {
            f64 g=ch<u32(gain.size())?gain[ch]:1;
            f64 ms=0; for(u32 i=start;i<start+block_size;++i) ms+=kw[ch][i]*kw[ch][i];
            power+=g*ms/block_size;
        }
        f64 lk=-0.691+10*std::log10(power+1e-30);
        block_lk.push_back(lk);
    }
    if(!block_lk.empty()) result.momentary=*std::max_element(block_lk.begin(),block_lk.end());
    // Gating: absolute gate -70 LUFS, relative gate -10 LU
    std::vector<f64> gated;
    for(auto lk:block_lk) if(lk>-70) gated.push_back(lk);
    if(!gated.empty()) {
        f64 gate1=0; for(auto v:gated) gate1+=std::pow(10,(v+0.691)/10); gate1/=gated.size();
        f64 threshold2=-10+10*std::log10(gate1)-0.691;
        std::vector<f64> gated2;
        for(auto lk:gated) if(lk>threshold2) gated2.push_back(lk);
        if(!gated2.empty()){
            f64 intg=0; for(auto v:gated2) intg+=std::pow(10,(v+0.691)/10); intg/=gated2.size();
            result.integrated=-0.691+10*std::log10(intg);
        }
    }
    // Short-term: 3s window
    u32 st_size=u32(sample_rate*3);
    if(n>=st_size) {
        f64 power=0;
        for(u32 ch=0;ch<nch;++ch){
            f64 g=ch<u32(gain.size())?gain[ch]:1; f64 ms=0;
            for(u32 i=n-st_size;i<n;++i) ms+=kw[ch][i]*kw[ch][i];
            power+=g*ms/st_size;
        }
        result.short_term=-0.691+10*std::log10(power+1e-30);
    }
    // True peak (upsampled)
    for(u32 ch=0;ch<nch;++ch) for(auto v:channels[ch]) result.peak_dbfs=std::max(result.peak_dbfs,std::abs(v));
    result.peak_dbfs=20*std::log10(result.peak_dbfs+1e-30);
    return result;
}

// ─── Psychoacoustic masking ───────────────────────────────────────────────────

// Bark scale conversion
[[nodiscard]] inline f64 hz_to_bark(f64 hz) {
    return 13*std::atan(0.00076*hz)+3.5*std::atan(std::pow(hz/7500,2));
}
[[nodiscard]] inline f64 bark_to_hz(f64 bark) {
    // Inverse (approximate)
    return 600*std::sinh(bark/6);
}

// Masking threshold estimation (simplified ISO 532B model)
[[nodiscard]] inline std::vector<f64> masking_threshold(
    const std::vector<f64>& spectrum_db, // dB per FFT bin
    f64 sample_rate, u32 n_fft)
{
    u32 n_bins=u32(spectrum_db.size());
    std::vector<f64> threshold(n_bins,-100);
    // Absolute threshold of hearing (ATH)
    for(u32 i=1;i<n_bins;++i) {
        f64 f=f64(i)*sample_rate/n_fft;
        f64 ath=3.64*std::pow(f/1000,-0.8)-6.5*std::exp(-0.6*(f/1000-3.3)*(f/1000-3.3))+1e-3*std::pow(f/1000,4);
        threshold[i]=std::max(threshold[i],ath);
    }
    // Simultaneous masking: each tone masks frequencies within spreading function
    for(u32 i=1;i<n_bins;++i) {
        f64 level=spectrum_db[i];
        if(level<-50) continue;
        f64 bark_i=hz_to_bark(f64(i)*sample_rate/n_fft);
        for(u32 j=1;j<n_bins;++j) {
            f64 bark_j=hz_to_bark(f64(j)*sample_rate/n_fft);
            f64 dz=bark_j-bark_i;
            // Spreading function: simplified empirical
            f64 spread;
            if(dz>=-3&&dz<=8) spread=level-15.811389+7.5*(dz+0.474)-17.5*std::sqrt(1+(dz+0.474)*(dz+0.474));
            else if(dz<-3)     spread=level+27+dz*(-8);
            else               spread=level-100;
            threshold[j]=std::max(threshold[j],spread);
        }
    }
    return threshold;
}

// Signal-to-Mask Ratio (SMR): spectrum_db[i] - masking_threshold[i]
[[nodiscard]] inline std::vector<f64> signal_to_mask_ratio(
    const std::vector<f64>& spectrum_db,
    const std::vector<f64>& threshold)
{
    u32 n=u32(std::min(spectrum_db.size(),threshold.size()));
    std::vector<f64> smr(n);
    for(u32 i=0;i<n;++i) smr[i]=spectrum_db[i]-threshold[i];
    return smr;
}

// Perceptual Entropy (PE) estimate: bits needed to code audio
[[nodiscard]] inline f64 perceptual_entropy(const std::vector<f64>& smr, f64 bin_bandwidth_hz) {
    f64 pe=0;
    for(auto s:smr) if(s>0) pe+=s*bin_bandwidth_hz/3322; // empirical constant
    return pe;
}

// ─── Codec-specific helpers ───────────────────────────────────────────────────

// MP3 MDCT (Modified DCT) window: 36-point, 18-output
[[nodiscard]] inline std::array<f32,18> mdct_18(const std::array<f32,36>& input) {
    std::array<f32,18> out{};
    for(u32 k=0;k<18;++k) {
        f32 sum=0;
        for(u32 n=0;n<36;++n)
            sum+=input[n]*std::cos(f32(std::numbers::pi)/72*(2*n+1+18)*(2*k+1));
        out[k]=sum;
    }
    return out;
}

// AAC TNS (Temporal Noise Shaping) predictor order selection
[[nodiscard]] inline u32 tns_order(f64 snr_db, u32 max_order=20) {
    // Higher SNR → higher predictor order
    return std::min(max_order,u32(snr_db*0.5));
}

// Opus frame size selection (optimal for latency vs quality)
[[nodiscard]] inline u32 opus_frame_size_samples(f64 bitrate_kbps, f64 sample_rate=48000) {
    if(bitrate_kbps<8) return u32(sample_rate*0.04);  // 40ms
    if(bitrate_kbps<24) return u32(sample_rate*0.02); // 20ms
    return u32(sample_rate*0.01);                      // 10ms
}

// Vorbis floor curve (piecewise linear spectral envelope)
[[nodiscard]] inline std::vector<f64> vorbis_floor_curve(
    const std::vector<u32>& x_points, const std::vector<f64>& y_points, u32 n_bins)
{
    std::vector<f64> floor(n_bins);
    for(u32 i=0;i<n_bins;++i) {
        // Find surrounding points
        u32 lo=0,hi=u32(x_points.size())-1;
        for(u32 j=0;j+1<u32(x_points.size());++j) if(x_points[j]<=i&&x_points[j+1]>i){lo=j;hi=j+1;break;}
        f64 t=x_points[hi]>x_points[lo]?f64(i-x_points[lo])/(x_points[hi]-x_points[lo]):0;
        floor[i]=y_points[lo]*(1-t)+y_points[hi]*t;
    }
    return floor;
}

} // namespace qs
