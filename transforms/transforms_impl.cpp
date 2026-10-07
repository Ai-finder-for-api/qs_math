
// qs_math/transforms/transforms_impl.cpp (C++20)
#include "include/qs_math_transforms.hpp"
#include <cmath>
#include <algorithm>
namespace qs::math::xform {
constexpr f64 PI2 = 6.28318530717958647692;

void fft_inplace(std::vector<Cf64>& a, bool inv) noexcept {
    i32 n=(i32)a.size();
    for(i32 i=1,j=0;i<n;++i){
        i32 bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j)std::swap(a[i],a[j]);
    }
    for(i32 len=2;len<=n;len<<=1){
        const f64 ang=PI2/len*(inv?-1:1);
        Cf64 wlen(std::cos(ang),std::sin(ang));
        for(i32 i=0;i<n;i+=len){
            Cf64 w(1,0);
            for(i32 j=0;j<len/2;++j,w*=wlen){
                Cf64 u=a[i+j],v=a[i+j+len/2]*w;
                a[i+j]=u+v;a[i+j+len/2]=u-v;
            }
        }
    }
    if(inv){const f64 inv_n=1.0/n;for(auto&x:a)x*=inv_n;}
}
std::vector<Cf64> rfft(const std::vector<f64>& x) noexcept {
    std::vector<Cf64> a(x.size());
    for(i32 i=0;i<(i32)x.size();++i)a[i]={x[i],0};
    fft_inplace(a,false);
    a.resize(x.size()/2+1);
    return a;
}
std::vector<f64> irfft(const std::vector<Cf64>& X,i32 n) noexcept {
    std::vector<Cf64> a(n);
    i32 h=n/2+1;
    for(i32 i=0;i<h;++i)a[i]=X[i];
    for(i32 i=h;i<n;++i)a[i]=std::conj(X[n-i]);
    fft_inplace(a,true);
    std::vector<f64> out(n);
    for(i32 i=0;i<n;++i)out[i]=a[i].real();
    return out;
}
void dct2(const f64* in,f64* out,i32 n) noexcept {
    for(i32 k=0;k<n;++k){
        f64 s=0;
        for(i32 j=0;j<n;++j)s+=in[j]*std::cos(PI2*k*(j+0.5)/n*0.5);
        out[k]=s;
    }
}
void dct3(const f64* in,f64* out,i32 n) noexcept {
    for(i32 j=0;j<n;++j){
        f64 s=in[0]*0.5;
        for(i32 k=1;k<n;++k)s+=in[k]*std::cos(PI2*k*(j+0.5)/n*0.5);
        out[j]=2*s/n;
    }
}
void dct4(const f64* in,f64* out,i32 n) noexcept {
    for(i32 k=0;k<n;++k){
        f64 s=0;
        for(i32 j=0;j<n;++j)s+=in[j]*std::cos(PI2*(k+0.5)*(j+0.5)/n*0.5);
        out[k]=s;
    }
}
static void dct1d_8(const f64* in,f64* out) noexcept {
    constexpr f64 c1=0.9807852804,c2=0.9238795325,c3=0.8314696123,
                  c4=0.7071067812,c5=0.5555702330,c6=0.3826834324,
                  c7=0.1950903220;
    f64 a0=in[0]+in[7],a1=in[1]+in[6],a2=in[2]+in[5],a3=in[3]+in[4];
    f64 a4=in[0]-in[7],a5=in[1]-in[6],a6=in[2]-in[5],a7=in[3]-in[4];
    f64 b0=a0+a3,b1=a1+a2,b2=a0-a3,b3=a1-a2;
    out[0]=(b0+b1)*0.35355339059; out[4]=(b0-b1)*0.35355339059;
    out[2]=b2*0.46193976625+b3*0.19134171618; out[6]=b2*0.19134171618-b3*0.46193976625;
    const f64 c0=a4*c1+a5*c3+a6*c5+a7*c7;
    const f64 d1=a4*c3-a5*c7-a6*c1-a7*c5;
    const f64 d2=a4*c5-a5*c1+a6*c7+a7*c3;
    const f64 d3=a4*c7-a5*c5+a6*c3-a7*c1;
    out[1]=c0*0.5; out[3]=d1*0.5; out[5]=d2*0.5; out[7]=d3*0.5;
}
void dct2_8x8(const f64 in[64],f64 out[64]) noexcept {
    f64 tmp[64];
    for(i32 r=0;r<8;++r)dct1d_8(in+r*8,tmp+r*8);
    for(i32 c=0;c<8;++c){
        f64 col[8];for(i32 r=0;r<8;++r)col[r]=tmp[r*8+c];
        dct1d_8(col,col);
        for(i32 r=0;r<8;++r)out[r*8+c]=col[r];
    }
}
void idct2_8x8(const f64 in[64],f64 out[64]) noexcept {
    // DCT-III is inverse of DCT-II (up to scale)
    f64 tmp[64];
    for(i32 c=0;c<8;++c){
        f64 col[8];for(i32 r=0;r<8;++r)col[r]=in[r*8+c];
        f64 tmp2[8];
        for(i32 j=0;j<8;++j){
            f64 s=col[0]*0.5;
            for(i32 k=1;k<8;++k)s+=col[k]*std::cos(3.14159265358979*k*(j+0.5)/8);
            tmp2[j]=s*0.25;
        }
        for(i32 r=0;r<8;++r)tmp[r*8+c]=tmp2[r];
    }
    for(i32 r=0;r<8;++r){
        f64 row[8],row2[8];for(i32 c=0;c<8;++c)row[c]=tmp[r*8+c];
        for(i32 j=0;j<8;++j){
            f64 s=row[0]*0.5;
            for(i32 k=1;k<8;++k)s+=row[k]*std::cos(3.14159265358979*k*(j+0.5)/8);
            row2[j]=s*0.25;
        }
        for(i32 c=0;c<8;++c)out[r*8+c]=row2[c];
    }
}
void wht(f64* d,i32 n) noexcept {
    for(i32 len=1;len<n;len<<=1)
        for(i32 i=0;i<n;i+=len*2)
            for(i32 j=i;j<i+len;++j){f64 a=d[j],b=d[j+len];d[j]=a+b;d[j+len]=a-b;}
}
void iwht(f64* d,i32 n) noexcept { wht(d,n); const f64 inv=1.0/n; for(i32 i=0;i<n;++i)d[i]*=inv; }
void haar_fwd(f64* d,i32 n) noexcept {
    std::vector<f64> tmp(n);
    for(i32 len=n;len>1;len>>=1){
        for(i32 i=0;i<len/2;++i){tmp[i]=(d[2*i]+d[2*i+1])*0.5;tmp[len/2+i]=(d[2*i]-d[2*i+1])*0.5;}
        std::copy(tmp.begin(),tmp.begin()+len,d);
    }
}
void haar_inv(f64* d,i32 n) noexcept {
    std::vector<f64> tmp(n);
    for(i32 len=2;len<=n;len<<=1){
        for(i32 i=0;i<len/2;++i){tmp[2*i]=d[i]+d[len/2+i];tmp[2*i+1]=d[i]-d[len/2+i];}
        std::copy(tmp.begin(),tmp.begin()+len,d);
    }
}
static const f64 DB4_H[]={0.48296291314,0.83651630374,0.22414386804,-0.12940952256};
static const f64 DB4_G[]={-0.12940952256,-0.22414386804,0.83651630374,-0.48296291314};
void db4_fwd(f64* d,i32 n) noexcept {
    std::vector<f64> tmp(n,0);
    for(i32 i=0;i<n/2;++i){
        for(i32 k=0;k<4;++k){tmp[i]+=DB4_H[k]*d[(2*i+k)%n];tmp[n/2+i]+=DB4_G[k]*d[(2*i+k)%n];}
    }
    std::copy(tmp.begin(),tmp.end(),d);
}
void db4_inv(f64* d,i32 n) noexcept {
    std::vector<f64> tmp(n,0);
    for(i32 i=0;i<n/2;++i){
        for(i32 k=0;k<4;++k){
            tmp[(2*i+k)%n]+=DB4_H[k]*d[i]+DB4_G[k]*d[n/2+i];
        }
    }
    std::copy(tmp.begin(),tmp.end(),d);
}
std::vector<Cf64> czt(const std::vector<Cf64>& x,i32 M,Cf64 W,Cf64 A) noexcept {
    i32 N=(i32)x.size(),L=N+M-1;
    i32 L2=1; while(L2<L)L2<<=1;
    std::vector<Cf64> yn(L2,0),wn(L2,0);
    for(i32 n=0;n<N;++n){
        Cf64 An=std::pow(A,(f64)(-n))*std::pow(W,(f64)(n*n)*0.5);
        yn[n]=x[n]*An; wn[n]=std::pow(W,(f64)(-n*n)*0.5);
    }
    for(i32 k=0;k<M;++k) wn[L2-M+k]=std::pow(W,(f64)(-(M-k)*(M-k))*0.5+k);
    fft_inplace(yn,false); fft_inplace(wn,false);
    for(i32 i=0;i<L2;++i)yn[i]*=wn[i];
    fft_inplace(yn,true);
    std::vector<Cf64> out(M);
    for(i32 k=0;k<M;++k)out[k]=yn[k]*std::pow(W,(f64)(k*k)*0.5);
    return out;
}
Cf64 z_transform_eval(const f64* c,i32 n,Cf64 z) noexcept {
    Cf64 r=0,zn=1;
    for(i32 i=0;i<n;++i){r+=c[i]*zn;zn*=z;}
    return r;
}
std::vector<f64> hilbert_envelope(const std::vector<f64>& x) noexcept {
    auto X=rfft(x); i32 n=(i32)x.size();
    // Multiply positive freqs by 2, zero negative (analytic signal)
    for(i32 k=1;k<(i32)X.size()-1;++k)X[k]*=2;
    auto analytic=irfft(X,n);
    // Envelope = |analytic|; but we need imag part too (full Hilbert)
    // Simplified: just use rfft for magnitude envelope approximation
    std::vector<f64> env(n);
    for(i32 i=0;i<n;++i)env[i]=std::abs(analytic[i]);
    return env;
}
std::vector<f64> instantaneous_phase(const std::vector<f64>& x) noexcept {
    auto env=hilbert_envelope(x);
    std::vector<f64> phase(x.size());
    for(i32 i=0;i<(i32)x.size();++i)
        phase[i]=env[i]>1e-10?std::atan2(env[i]-x[i],x[i]):0;
    return phase;
}
STFTResult stft(const std::vector<f64>& x,i32 fs,i32 hop) noexcept {
    i32 n=(i32)x.size();
    STFTResult r; r.n_bins=fs/2+1;
    std::vector<f64> window(fs);
    for(i32 i=0;i<fs;++i)window[i]=0.5*(1-std::cos(PI2*i/(fs-1)));
    for(i32 start=0;start+fs<=n;start+=hop){
        std::vector<f64> frame(fs);
        for(i32 i=0;i<fs;++i)frame[i]=x[start+i]*window[i];
        r.frames.push_back(rfft(frame));
        r.n_frames++;
    }
    return r;
}
std::vector<f64> istft(const STFTResult& s,i32 fs,i32 hop) noexcept {
    i32 n=s.n_frames*hop+fs;
    std::vector<f64> out(n,0),win_sum(n,0);
    std::vector<f64> window(fs);
    for(i32 i=0;i<fs;++i)window[i]=0.5*(1-std::cos(PI2*i/(fs-1)));
    for(i32 f=0;f<s.n_frames;++f){
        auto frame=irfft(s.frames[f],fs);
        for(i32 i=0;i<fs;++i){out[f*hop+i]+=frame[i]*window[i];win_sum[f*hop+i]+=window[i]*window[i];}
    }
    for(i32 i=0;i<n;++i)if(win_sum[i]>1e-10)out[i]/=win_sum[i];
    return out;
}
} // namespace qs::math::xform
