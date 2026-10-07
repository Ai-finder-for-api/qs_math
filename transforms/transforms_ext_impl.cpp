
// qs_math/transforms/transforms_ext_impl.cpp (C++20)
// Extended transforms: Radon, Abel, Hankel, Mellin, Z-transforms,
// wavelets, filterbank, cepstrum, chirplet.
#include "include/qs_math_transforms.hpp"
#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
namespace qs::math::xform {
using namespace qs::math;
constexpr f64 PI_D = 3.14159265358979323846;

// ─── Radon transform (2D → sinogram) ─────────────────────────────────────────
// Projects image along angles theta[0..n_angles-1]
std::vector<std::vector<f64>> radon_transform(
    const f64* image, i32 img_w, i32 img_h,
    const f64* thetas, i32 n_angles) noexcept
{
    const i32 diag=(i32)(std::ceil(std::sqrt((f64)(img_w*img_w+img_h*img_h))));
    const i32 n_rho=2*diag+1;
    std::vector<std::vector<f64>> sinogram(n_angles,std::vector<f64>(n_rho,0));

    const f64 cx=img_w*0.5, cy=img_h*0.5;
    for(i32 ai=0;ai<n_angles;++ai){
        const f64 th=thetas[ai];
        const f64 ct=std::cos(th), st=std::sin(th);
        for(i32 ri=0;ri<n_rho;++ri){
            const f64 rho=ri-diag;
            // Line: x*cos(th)+y*sin(th) = rho
            // Parameterise by perpendicular
            f64 sum=0;
            for(i32 t=-diag;t<=diag;++t){
                const f64 x=rho*ct-t*st+cx;
                const f64 y=rho*st+t*ct+cy;
                const i32 ix=(i32)std::round(x),iy=(i32)std::round(y);
                if(ix>=0&&ix<img_w&&iy>=0&&iy<img_h)
                    sum+=image[iy*img_w+ix];
            }
            sinogram[ai][ri]=sum;
        }
    }
    return sinogram;
}

// ─── Filtered backprojection (inverse Radon) ──────────────────────────────────
std::vector<f64> filtered_backprojection(
    const std::vector<std::vector<f64>>& sinogram,
    i32 out_w, i32 out_h) noexcept
{
    const i32 n_angles=(i32)sinogram.size();
    if(!n_angles) return {};
    const i32 n_rho=(i32)sinogram[0].size();
    const i32 diag=(n_rho-1)/2;
    std::vector<f64> image(out_w*out_h,0);

    // Ram-Lak filter in frequency domain
    const i32 fft_n=1; // next power of 2 ≥ n_rho
    i32 fn=1; while(fn<n_rho*2)fn<<=1;
    std::vector<f64> filt(fn/2+1);
    for(i32 k=0;k<=fn/2;++k) filt[k]=(f64)k/fn; // |ω| filter

    std::vector<std::vector<f64>> filtered=sinogram;
    for(i32 ai=0;ai<n_angles;++ai){
        // FFT the projection
        std::vector<Cf64> proj(fn,0);
        for(i32 i=0;i<n_rho;++i) proj[i]={sinogram[ai][i],0};
        fft_inplace(proj,false);
        // Apply filter
        for(i32 k=0;k<=fn/2;++k){proj[k]*=filt[k];if(k>0&&k<fn/2)proj[fn-k]=std::conj(proj[k])*filt[k];}
        fft_inplace(proj,true);
        for(i32 i=0;i<n_rho;++i) filtered[ai][i]=proj[i].real();
    }

    // Backproject
    const f64 cx=out_w*0.5, cy=out_h*0.5;
    const f64 dtheta=PI_D/n_angles;
    for(i32 iy=0;iy<out_h;++iy)
        for(i32 ix=0;ix<out_w;++ix){
            const f64 x=ix-cx, y=iy-cy;
            for(i32 ai=0;ai<n_angles;++ai){
                const f64 th=ai*dtheta;
                const f64 rho=x*std::cos(th)+y*std::sin(th);
                const i32 ri=(i32)std::round(rho)+diag;
                if(ri>=0&&ri<n_rho) image[iy*out_w+ix]+=filtered[ai][ri]*dtheta;
            }
        }
    return image;
}

// ─── Abel transform (1D, symmetric) ──────────────────────────────────────────
// Forward: F(y) = 2 ∫_y^∞ f(r)*r/sqrt(r²-y²) dr
std::vector<f64> abel_forward(const std::vector<f64>& f_r, f64 dr) noexcept {
    const i32 n=(i32)f_r.size();
    std::vector<f64> F(n,0);
    for(i32 j=0;j<n;++j){
        const f64 y=j*dr;
        for(i32 i=j+1;i<n;++i){
            const f64 r=i*dr;
            const f64 sq=r*r-y*y;
            if(sq>1e-14) F[j]+=2*f_r[i]*r/std::sqrt(sq)*dr;
        }
    }
    return F;
}

// ─── Hankel transform (0th order, via quadrature) ────────────────────────────
std::vector<Cf64> hankel_transform(const std::vector<f64>& f, f64 dr, i32 n_k) noexcept {
    const i32 n=(i32)f.size();
    std::vector<Cf64> H(n_k);
    for(i32 ki=0;ki<n_k;++ki){
        const f64 k=ki*dr;
        Cf64 sum=0;
        for(i32 ri=0;ri<n;++ri){
            const f64 r=ri*dr;
            // J0(k*r) approximation via cosine series
            const f64 kr=k*r;
            const f64 J0=kr<1e-7?1.0:std::cyl_bessel_j(0,kr);
            sum+=f[ri]*J0*r*dr;
        }
        H[ki]=sum*2*PI_D;
    }
    return H;
}

// ─── Mellin transform ─────────────────────────────────────────────────────────
// M{f}(s) = ∫_0^∞ x^(s-1) f(x) dx
std::vector<Cf64> mellin_transform(const std::vector<f64>& f, f64 dx,
    const std::vector<f64>& s_values) noexcept
{
    const i32 n=(i32)f.size(), ns=(i32)s_values.size();
    std::vector<Cf64> M(ns);
    for(i32 si=0;si<ns;++si){
        const f64 s=s_values[si]-1; // s-1 exponent
        Cf64 sum=0;
        for(i32 i=1;i<n;++i){
            const f64 x=i*dx;
            sum+=f[i]*std::pow(x,s)*dx;
        }
        M[si]=sum;
    }
    return M;
}

// ─── Cepstrum (real) ──────────────────────────────────────────────────────────
std::vector<f64> real_cepstrum(const std::vector<f64>& x) noexcept {
    auto X=rfft(x);
    const i32 n=(i32)x.size();
    std::vector<f64> log_mag(X.size());
    for(i32 i=0;i<(i32)X.size();++i)
        log_mag[i]=std::log(std::max(1e-14,std::abs(X[i])));
    // IFFT of log magnitude (symmetric extension)
    std::vector<Cf64> log_X(n,0);
    log_X[0]={log_mag[0],0};
    for(i32 i=1;i<(i32)X.size()-1;++i){log_X[i]={log_mag[i],0};log_X[n-i]={log_mag[i],0};}
    if((i32)X.size()>1) log_X[X.size()-1]={log_mag.back(),0};
    fft_inplace(log_X,true);
    std::vector<f64> ceps(n);
    for(i32 i=0;i<n;++i) ceps[i]=log_X[i].real();
    return ceps;
}

// ─── Constant-Q transform (CQT) ───────────────────────────────────────────────
struct CQTResult {
    std::vector<std::vector<Cf64>> coeffs; // [freq_bin][time_frame]
    std::vector<f64> freqs;
    i32 n_bins{0};
};

CQTResult cqt(const std::vector<f64>& x, f64 fs, f64 f_min=27.5,
              i32 n_bins_per_octave=12, i32 n_octaves=7) noexcept
{
    CQTResult r;
    r.n_bins=n_bins_per_octave*n_octaves;
    r.freqs.resize(r.n_bins);
    r.coeffs.resize(r.n_bins);
    const i32 n=(i32)x.size();
    const f64 Q=1.0/(std::pow(2,1.0/n_bins_per_octave)-1);

    for(i32 k=0;k<r.n_bins;++k){
        const f64 fk=f_min*std::pow(2,(f64)k/n_bins_per_octave);
        r.freqs[k]=fk;
        const i32 Nk=(i32)(Q*fs/fk);
        // Hamming window kernel
        std::vector<Cf64> kernel(Nk);
        for(i32 j=0;j<Nk;++j){
            const f64 w=0.54-0.46*std::cos(2*PI_D*j/(Nk-1));
            const f64 phase=2*PI_D*Q*j/Nk;
            kernel[j]=Cf64{w*std::cos(phase),-w*std::sin(phase)}*(1.0/Nk);
        }
        // Convolve signal with kernel (hop size = Nk/4)
        const i32 hop=std::max(1,Nk/4);
        const i32 n_frames=(n-Nk)/hop+1;
        r.coeffs[k].resize(n_frames);
        for(i32 t=0;t<n_frames;++t){
            Cf64 sum=0;
            for(i32 j=0;j<Nk&&t*hop+j<n;++j) sum+=x[t*hop+j]*kernel[j];
            r.coeffs[k][t]=sum;
        }
    }
    return r;
}

// ─── MFCC (Mel-frequency cepstral coefficients) ───────────────────────────────
std::vector<std::vector<f64>> mfcc(
    const std::vector<f64>& x, f64 fs, i32 n_mfcc=13,
    i32 frame_size=512, i32 hop=256, i32 n_mels=40) noexcept
{
    // 1. STFT
    auto s=stft(x,frame_size,hop);
    const i32 n_frames=s.n_frames;
    const i32 n_bins=s.n_bins;

    // 2. Mel filterbank
    const f64 f_min=80,f_max=fs/2;
    const f64 mel_min=2595*std::log10(1+f_min/700);
    const f64 mel_max=2595*std::log10(1+f_max/700);
    std::vector<f64> mel_pts(n_mels+2);
    for(i32 m=0;m<=n_mels+1;++m){
        const f64 mel=mel_min+(mel_max-mel_min)*m/(n_mels+1);
        mel_pts[m]=700*(std::pow(10,mel/2595)-1);
    }
    // Build filterbank weights
    std::vector<std::vector<f64>> fb(n_mels,std::vector<f64>(n_bins,0));
    for(i32 m=0;m<n_mels;++m)
        for(i32 k=0;k<n_bins;++k){
            const f64 fk=k*(fs/frame_size);
            if(fk>=mel_pts[m]&&fk<=mel_pts[m+1])
                fb[m][k]=(fk-mel_pts[m])/(mel_pts[m+1]-mel_pts[m]);
            else if(fk>mel_pts[m+1]&&fk<=mel_pts[m+2])
                fb[m][k]=(mel_pts[m+2]-fk)/(mel_pts[m+2]-mel_pts[m+1]);
        }

    // 3. Compute MFCCs
    std::vector<std::vector<f64>> mfccs(n_frames,std::vector<f64>(n_mfcc,0));
    for(i32 t=0;t<n_frames;++t){
        // Power spectrum
        std::vector<f64> power(n_bins);
        for(i32 k=0;k<n_bins;++k) power[k]=std::norm(s.frames[t][k]);
        // Apply mel filterbank
        std::vector<f64> mel_energy(n_mels);
        for(i32 m=0;m<n_mels;++m){
            f64 e=0; for(i32 k=0;k<n_bins;++k)e+=fb[m][k]*power[k];
            mel_energy[m]=std::log(std::max(1e-10,e));
        }
        // DCT to get MFCCs
        for(i32 c=0;c<n_mfcc;++c){
            f64 s2=0;
            for(i32 m=0;m<n_mels;++m)
                s2+=mel_energy[m]*std::cos(PI_D*c*(m+0.5)/n_mels);
            mfccs[t][c]=s2*std::sqrt(2.0/n_mels);
        }
    }
    return mfccs;
}

// ─── Wavelet scattering (simplified, 1 layer) ─────────────────────────────────
std::vector<f64> wavelet_scattering_l1(const std::vector<f64>& x,
    i32 n_wavelets=8, i32 frame_size=512, i32 hop=128) noexcept
{
    const i32 n=(i32)x.size();
    std::vector<f64> result;
    // Morlet wavelets at dyadic scales
    for(i32 j=0;j<n_wavelets;++j){
        const f64 scale=std::pow(2,j);
        const i32 support=std::min((i32)(4*scale)+1,frame_size);
        std::vector<Cf64> psi(support);
        for(i32 k=0;k<support;++k){
            const f64 t=(k-support*0.5)/scale;
            psi[k]=Cf64{std::exp(-0.5*t*t)*std::cos(5*t),
                        std::exp(-0.5*t*t)*std::sin(5*t)};
        }
        // Convolve |x| with psi, take modulus
        for(i32 start=0;start+support<n;start+=hop){
            Cf64 sum=0;
            for(i32 k=0;k<support;++k) sum+=x[start+k]*psi[k];
            result.push_back(std::abs(sum)/support);
        }
    }
    return result;
}

} // namespace qs::math::xform
