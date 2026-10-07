#pragma once
// signal_processing.hpp — Signal processing: FIR/IIR filters, windowing,
//                         FFT (Cooley-Tukey), resampling, convolution,
//                         wavelet transform, pitch detection, spectral analysis
#include "vec2.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "constants.hpp"

namespace qs::math::dsp {

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 window_hann(i32 n, i32 N) noexcept {
    return 0.5f*(1.f-cos(TWO_PI*n/(N-1)));
}
[[nodiscard]] constexpr f32 window_hamming(i32 n, i32 N) noexcept {
    return 0.54f-0.46f*cos(TWO_PI*n/(N-1));
}
[[nodiscard]] constexpr f32 window_blackman(i32 n, i32 N) noexcept {
    f32 a=TWO_PI*n/(N-1);
    return 0.42f-0.5f*cos(a)+0.08f*cos(2.f*a);
}
[[nodiscard]] constexpr f32 window_kaiser(i32 n, i32 N, f32 beta=5.f) noexcept {
    f32 t=2.f*n/(N-1)-1.f;
    // Zeroth-order modified Bessel I0 approximation
    f32 x=beta*sqrt(max(0.f,1.f-t*t));
    f32 sum=1.f,term=1.f;
    for(i32 k=1;k<=10;++k){
        term*=(x*x)/(4.f*k*k);
        sum+=term;
    }
    f32 i0_beta=1.f;
    f32 xb=beta; f32 termb=1.f;
    for(i32 k=1;k<=10;++k){ termb*=(xb*xb)/(4.f*k*k); i0_beta+=termb; }
    return sum/i0_beta;
}
[[nodiscard]] constexpr f32 window_flat_top(i32 n, i32 N) noexcept {
    f32 a=TWO_PI*n/(N-1);
    return 0.21557895f-0.41663158f*cos(a)+0.277263158f*cos(2.f*a)
          -0.083578947f*cos(3.f*a)+0.006947368f*cos(4.f*a);
}

// Apply window function to a buffer
inline void apply_window(f32* buf, i32 N, f32(*window_fn)(i32,i32)) noexcept {
    for(i32 i=0;i<N;++i) buf[i]*=window_fn(i,N);
}

// ---------------------------------------------------------------------------
// In-place DFT (Cooley-Tukey FFT, power-of-2 size)
// ---------------------------------------------------------------------------
inline void fft_inplace(Vec2f* x, i32 N, bool inverse=false) noexcept {
    // Bit-reversal
    for(i32 i=1,j=0;i<N;++i){
        i32 bit=N>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j){ Vec2f t=x[i];x[i]=x[j];x[j]=t; }
    }
    // Butterfly
    for(i32 len=2;len<=N;len<<=1){
        f32 ang=TWO_PI/len*(inverse?1.f:-1.f);
        Vec2f wlen{cos(ang),sin(ang)};
        for(i32 i=0;i<N;i+=len){
            Vec2f w{1.f,0.f};
            for(i32 j=0;j<len/2;++j){
                Vec2f u=x[i+j];
                Vec2f v{w.x*x[i+j+len/2].x-w.y*x[i+j+len/2].y,
                        w.x*x[i+j+len/2].y+w.y*x[i+j+len/2].x};
                x[i+j]={u.x+v.x,u.y+v.y};
                x[i+j+len/2]={u.x-v.x,u.y-v.y};
                w={w.x*wlen.x-w.y*wlen.y,w.x*wlen.y+w.y*wlen.x};
            }
        }
    }
    if(inverse){ f32 inv=1.f/N; for(i32 i=0;i<N;++i){x[i].x*=inv;x[i].y*=inv;} }
}

// Real FFT (real input, N/2+1 complex output)
inline void rfft(const f32* real_in, Vec2f* complex_out, i32 N) noexcept {
    constexpr i32 MAX_N=4096;
    Vec2f buf[MAX_N];
    i32 n=min(N,MAX_N);
    for(i32 i=0;i<n;++i) buf[i]={real_in[i],0.f};
    fft_inplace(buf,n);
    for(i32 i=0;i<=n/2;++i) complex_out[i]=buf[i];
}

// Power spectrum from FFT output
inline void power_spectrum(const Vec2f* fft_out, f32* power, i32 N_half) noexcept {
    for(i32 i=0;i<N_half;++i)
        power[i]=(fft_out[i].x*fft_out[i].x+fft_out[i].y*fft_out[i].y)/(N_half*N_half);
}

// Magnitude spectrum in dB
inline void magnitude_db(const Vec2f* fft_out, f32* mag_db, i32 N_half) noexcept {
    for(i32 i=0;i<N_half;++i){
        f32 mag=sqrt(fft_out[i].x*fft_out[i].x+fft_out[i].y*fft_out[i].y);
        mag_db[i]=20.f*log10(max(mag,EPS_F32));
    }
}

// ---------------------------------------------------------------------------
// FIR filter design (windowed sinc)
// ---------------------------------------------------------------------------
inline void design_lowpass_fir(
    f32* coeffs, i32 taps, f32 cutoff_normalized, // 0..1 (1=Nyquist)
    f32(*window_fn)(i32,i32)=window_hann) noexcept
{
    i32 M=taps-1;
    f32 fc=cutoff_normalized*PI;
    for(i32 i=0;i<taps;++i){
        i32 n=i-M/2;
        f32 h=(n==0)?2.f*cutoff_normalized:sin(fc*n)/(PI*n);
        coeffs[i]=h*window_fn(i,taps);
    }
    // Normalize
    f32 sum=0.f; for(i32 i=0;i<taps;++i) sum+=coeffs[i];
    if(sum>EPS_F32) for(i32 i=0;i<taps;++i) coeffs[i]/=sum;
}

// Apply FIR filter (direct convolution)
inline void apply_fir(const f32* input, f32* output, i32 n_samples,
                      const f32* coeffs, i32 n_taps) noexcept
{
    for(i32 i=0;i<n_samples;++i){
        f32 acc=0.f;
        for(i32 k=0;k<n_taps;++k){
            i32 j=i-k;
            if(j>=0) acc+=coeffs[k]*input[j];
        }
        output[i]=acc;
    }
}

// ---------------------------------------------------------------------------
// Resampling (linear interpolation)
// ---------------------------------------------------------------------------
inline void resample_linear(
    const f32* input, i32 in_rate,
    f32* output, i32 out_rate, i32 out_samples) noexcept
{
    f64 ratio=f64(in_rate)/out_rate;
    for(i32 i=0;i<out_samples;++i){
        f64 pos=i*ratio;
        i32 idx=(i32)pos;
        f32 frac=f32(pos-idx);
        f32 a=input[idx], b=(idx+1<in_rate)?input[idx+1]:0.f;
        output[i]=a+frac*(b-a);
    }
}

// ---------------------------------------------------------------------------
// Pitch detection (YIN algorithm, simplified)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 yin_pitch(const f32* buf, i32 N, f32 sample_rate,
                                    f32 threshold=0.1f) noexcept
{
    constexpr i32 MAX_N=2048;
    i32 n=min(N,MAX_N);
    f32 d[MAX_N]={};
    // Difference function
    for(i32 tau=1;tau<n/2;++tau){
        for(i32 j=0;j<n/2;++j){
            f32 diff=buf[j]-buf[j+tau];
            d[tau]+=diff*diff;
        }
    }
    // Cumulative mean normalized
    f32 cmnd[MAX_N]={};
    cmnd[0]=1.f;
    f32 sum=0.f;
    for(i32 tau=1;tau<n/2;++tau){
        sum+=d[tau];
        cmnd[tau]=sum>EPS_F32?d[tau]*tau/sum:1.f;
    }
    // Find first minimum below threshold
    for(i32 tau=2;tau<n/2-1;++tau){
        if(cmnd[tau]<threshold&&cmnd[tau]<cmnd[tau-1]&&cmnd[tau]<cmnd[tau+1]){
            // Parabolic interpolation
            f32 x=cmnd[tau-1],y=cmnd[tau],z=cmnd[tau+1];
            f32 refined=tau+(z-x)/(2.f*(2.f*y-x-z));
            return sample_rate/refined;
        }
    }
    return 0.f; // no pitch found
}

// ---------------------------------------------------------------------------
// Spectral centroid (brightness estimate)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 spectral_centroid(const f32* magnitudes, i32 N, f32 sample_rate) noexcept {
    f32 num=0.f, denom=0.f;
    for(i32 i=0;i<N;++i){
        f32 freq=i*sample_rate/(2.f*N);
        num+=freq*magnitudes[i]; denom+=magnitudes[i];
    }
    return denom>EPS_F32?num/denom:0.f;
}

// Spectral flux (change between frames)
[[nodiscard]] inline f32 spectral_flux(const f32* curr, const f32* prev, i32 N) noexcept {
    f32 flux=0.f;
    for(i32 i=0;i<N;++i){ f32 d=curr[i]-prev[i]; if(d>0.f) flux+=d*d; }
    return sqrt(flux);
}

// MFCC (Mel-frequency cepstral coefficients) — first step: mel filterbank
inline void apply_mel_filterbank(
    const f32* spectrum, i32 fft_size, f32 sample_rate,
    f32* mel_energies, i32 n_mels,
    f32 f_min=80.f, f32 f_max=8000.f) noexcept
{
    auto hz_to_mel_=[](f32 hz)->f32{ return 2595.f*log10(1.f+hz/700.f); };
    auto mel_to_hz_=[](f32 mel)->f32{ return 700.f*(pow(10.f,mel/2595.f)-1.f); };
    f32 mel_min=hz_to_mel_(f_min), mel_max=hz_to_mel_(f_max);
    for(i32 m=0;m<n_mels;++m){
        f32 mel_center=mel_min+(m+1)*(mel_max-mel_min)/(n_mels+1);
        f32 mel_left  =mel_min+m*(mel_max-mel_min)/(n_mels+1);
        f32 mel_right =mel_min+(m+2)*(mel_max-mel_min)/(n_mels+1);
        f32 f_center=mel_to_hz_(mel_center);
        f32 f_left  =mel_to_hz_(mel_left);
        f32 f_right =mel_to_hz_(mel_right);
        mel_energies[m]=0.f;
        for(i32 k=0;k<fft_size/2;++k){
            f32 freq=k*sample_rate/fft_size;
            f32 weight=0.f;
            if(freq>=f_left&&freq<=f_center) weight=(freq-f_left)/(f_center-f_left);
            else if(freq>f_center&&freq<=f_right) weight=(f_right-freq)/(f_right-f_center);
            mel_energies[m]+=weight*spectrum[k];
        }
    }
}

} // namespace qs::math::dsp
