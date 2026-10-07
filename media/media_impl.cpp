
// qs_math/media/media_impl.cpp (C++20)
#include "include/qs_math_media.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
namespace qs::math::media {
constexpr f64 PI = 3.14159265358979323846;
constexpr f64 TAU = 2.0*PI;

void fft(std::vector<Cf64>& a, bool inv) noexcept {
    i32 n=(i32)a.size();
    // Bit-reversal permutation
    for(i32 i=1,j=0;i<n;++i){
        i32 bit=n>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit; if(i<j) std::swap(a[i],a[j]);
    }
    for(i32 len=2;len<=n;len<<=1){
        const f64 ang=TAU/len*(inv?-1:1);
        Cf64 wlen(std::cos(ang),std::sin(ang));
        for(i32 i=0;i<n;i+=len){
            Cf64 w(1,0);
            for(i32 j=0;j<len/2;++j,w*=wlen){
                Cf64 u=a[i+j],v=a[i+j+len/2]*w;
                a[i+j]=u+v; a[i+j+len/2]=u-v;
            }
        }
    }
    if(inv){f64 inv_n=1.0/n;for(auto&x:a)x*=inv_n;}
}

void rfft(const f64* in, i32 n, std::vector<Cf64>& out) noexcept {
    std::vector<Cf64> a(n);
    for(i32 i=0;i<n;++i) a[i]={in[i],0};
    fft(a,false);
    out.assign(a.begin(),a.begin()+n/2+1);
}

void irfft(const std::vector<Cf64>& in, i32 n, std::vector<f64>& out) noexcept {
    std::vector<Cf64> a(n);
    i32 h=n/2+1;
    for(i32 i=0;i<h;++i) a[i]=in[i];
    for(i32 i=h;i<n;++i) a[i]=std::conj(in[n-i]);
    fft(a,true);
    out.resize(n);
    for(i32 i=0;i<n;++i) out[i]=a[i].real();
}

void apply_window(f64* d, i32 n, Window w, f64 beta) noexcept {
    for(i32 i=0;i<n;++i){
        f64 x=(f64)i/(n-1);
        f64 coef=1;
        switch(w){
        case Window::Hann:      coef=0.5*(1-std::cos(TAU*x)); break;
        case Window::Hamming:   coef=0.54-0.46*std::cos(TAU*x); break;
        case Window::Blackman:  coef=0.42-0.5*std::cos(TAU*x)+0.08*std::cos(4*PI*x); break;
        case Window::Triangular:coef=1-std::abs(2*x-1); break;
        case Window::FlatTop:   coef=0.21557895-0.41663158*std::cos(TAU*x)
                                     +0.277263158*std::cos(4*PI*x)
                                     -0.083578947*std::cos(6*PI*x)
                                     +0.006947368*std::cos(8*PI*x); break;
        case Window::Kaiser: {
            auto I0=[](f64 z)->f64{f64 s=1,t=1;for(i32 k=1;k<30;++k){t*=z/(2*k);t*=z/(2*k);s+=t*t;}return s;};
            f64 r=2*x-1; coef=I0(beta*std::sqrt(1-r*r))/I0(beta); break;
        }
        default: coef=1; break;
        }
        d[i]*=coef;
    }
}

std::vector<f64> window_coeffs(i32 n, Window w, f64 beta) noexcept {
    std::vector<f64> v(n,1.0);
    apply_window(v.data(),n,w,beta);
    return v;
}

std::vector<f64> psd(const f64* data, i32 n, Window w) noexcept {
    std::vector<f64> buf(data,data+n);
    apply_window(buf.data(),n,w);
    std::vector<Cf64> spec;
    rfft(buf.data(),n,spec);
    std::vector<f64> p(spec.size());
    for(i32 i=0;i<(i32)spec.size();++i) p[i]=std::norm(spec[i])/n;
    return p;
}

std::vector<f64> convolve(const std::vector<f64>& a, const std::vector<f64>& b) noexcept {
    i32 na=(i32)a.size(),nb=(i32)b.size(),nc=na+nb-1;
    // FFT-based convolution
    i32 fft_n=1; while(fft_n<nc) fft_n<<=1;
    std::vector<Cf64> A(fft_n),B(fft_n);
    for(i32 i=0;i<na;++i)A[i]={a[i],0};
    for(i32 i=0;i<nb;++i)B[i]={b[i],0};
    fft(A,false);fft(B,false);
    for(i32 i=0;i<fft_n;++i)A[i]*=B[i];
    fft(A,true);
    std::vector<f64> res(nc);
    for(i32 i=0;i<nc;++i)res[i]=A[i].real();
    return res;
}

std::vector<f64> cross_correlate(const std::vector<f64>& a, const std::vector<f64>& b) noexcept {
    std::vector<f64> br(b.rbegin(),b.rend());
    return convolve(a,br);
}

static f64 biquad_tan(f64 fc, f64 fs) noexcept { return std::tan(PI*fc/fs); }

BiquadCoeffs biquad_lpf(f64 fc, f64 fs, f64 Q) noexcept {
    const f64 K=biquad_tan(fc,fs),K2=K*K,norm=1+K/Q+K2;
    return {K2/norm, 2*K2/norm, K2/norm, 2*(K2-1)/norm, (1-K/Q+K2)/norm};
}
BiquadCoeffs biquad_hpf(f64 fc, f64 fs, f64 Q) noexcept {
    const f64 K=biquad_tan(fc,fs),K2=K*K,norm=1+K/Q+K2;
    return {1/norm,-2/norm,1/norm,2*(K2-1)/norm,(1-K/Q+K2)/norm};
}
BiquadCoeffs biquad_bpf(f64 fc, f64 fs, f64 Q) noexcept {
    const f64 K=biquad_tan(fc,fs),K2=K*K,norm=1+K/Q+K2;
    return {K/Q/norm,0,-K/Q/norm,2*(K2-1)/norm,(1-K/Q+K2)/norm};
}
BiquadCoeffs biquad_peak(f64 fc, f64 fs, f64 gain_db, f64 Q) noexcept {
    const f64 A=std::pow(10.0,gain_db/40.0),K=biquad_tan(fc,fs),K2=K*K;
    if(gain_db>=0){
        const f64 norm=1+K/Q+K2;
        return {(1+A*K/Q+K2)/norm,2*(K2-1)/norm,(1-A*K/Q+K2)/norm,2*(K2-1)/norm,(1-K/Q+K2)/norm};
    }
    const f64 norm=1+K/(A*Q)+K2;
    return {(1+K/Q+K2)/norm,2*(K2-1)/norm,(1-K/Q+K2)/norm,2*(K2-1)/norm,(1-K/(A*Q)+K2)/norm};
}
f64 biquad_process(const BiquadCoeffs& c, BiquadState& s, f64 x) noexcept {
    const f64 y=c.b0*x+s.z1;
    s.z1=c.b1*x-c.a1*y+s.z2;
    s.z2=c.b2*x-c.a2*y;
    return y;
}
void biquad_process_block(const BiquadCoeffs& c, BiquadState& s,
    const f64* in, f64* out, i32 n) noexcept {
    for(i32 i=0;i<n;++i) out[i]=biquad_process(c,s,in[i]);
}
void resample_sinc(const f64* in, i32 ni, f64* out, i32 no, i32 K) noexcept {
    const f64 ratio=(f64)no/ni;
    for(i32 j=0;j<no;++j){
        const f64 t=(f64)j/ratio;
        f64 sum=0,weight=0;
        for(i32 k=-K/2;k<=K/2;++k){
            const i32 idx=(i32)(t+k);
            if(idx<0||idx>=ni) continue;
            const f64 x=PI*(t+k-std::floor(t+k)-k);
            const f64 sinc=std::abs(x)<1e-10?1.0:std::sin(x)/x;
            const f64 win=0.5*(1+std::cos(TAU*k/K));
            sum+=in[idx]*sinc*win; weight+=sinc*win;
        }
        out[j]=weight<1e-10?0:sum/weight;
    }
}
std::vector<std::vector<f64>> mel_filterbank(i32 M,i32 N,f64 fs,f64 fmin,f64 fmax) noexcept {
    const f64 mmin=hz_to_mel(fmin),mmax=hz_to_mel(fmax);
    std::vector<f64> mels(M+2);
    for(i32 i=0;i<=M+1;++i) mels[i]=mmin+(mmax-mmin)*i/(M+1);
    std::vector<f64> freqs(N/2+1);
    for(i32 i=0;i<=(i32)freqs.size()-1;++i) freqs[i]=i*fs/N;
    std::vector<std::vector<f64>> fb(M,std::vector<f64>(N/2+1,0));
    for(i32 m=0;m<M;++m){
        const f64 fl=mel_to_hz(mels[m]),fc=mel_to_hz(mels[m+1]),fr=mel_to_hz(mels[m+2]);
        for(i32 k=0;k<(i32)freqs.size();++k){
            if(freqs[k]>=fl&&freqs[k]<=fc) fb[m][k]=(freqs[k]-fl)/(fc-fl);
            else if(freqs[k]>fc&&freqs[k]<=fr) fb[m][k]=(fr-freqs[k])/(fr-fc);
        }
    }
    return fb;
}
} // namespace qs::math::media
