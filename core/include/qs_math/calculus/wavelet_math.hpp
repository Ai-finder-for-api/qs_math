#pragma once
// qs_math/wavelet_math.hpp — Haar, Daubechies, CWT/DWT helpers (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>
#include <numbers>

namespace qs {

// ─── 1D Discrete Wavelet Transform (DWT) — lifting scheme ────────────────────

// Haar DWT: simplest wavelet (piecewise constant)
inline void haar_dwt(std::vector<f64>& x) {
    u32 N=u32(x.size());
    std::vector<f64> tmp(N);
    while(N>1) {
        u32 half=N/2;
        for(u32 i=0;i<half;++i) {
            tmp[i]=(x[2*i]+x[2*i+1])*std::numbers::sqrt2_v<f64>/2; // approx
            tmp[i+half]=(x[2*i]-x[2*i+1])*std::numbers::sqrt2_v<f64>/2; // detail
        }
        for(u32 i=0;i<N;++i) x[i]=tmp[i];
        N=half;
    }
}

inline void haar_idwt(std::vector<f64>& x) {
    u32 N=u32(x.size());
    u32 level=1;
    while(level<=N/2) level<<=1; level>>=1;
    std::vector<f64> tmp(N);
    while(level>=2) {
        u32 half=level/2;
        for(u32 i=0;i<half;++i) {
            tmp[2*i]  =(x[i]+x[i+half])/std::numbers::sqrt2_v<f64>;
            tmp[2*i+1]=(x[i]-x[i+half])/std::numbers::sqrt2_v<f64>;
        }
        for(u32 i=0;i<level;++i) x[i]=tmp[i];
        level=half;
    }
}

// ─── Daubechies wavelet filter bank ──────────────────────────────────────────
struct WaveletFilter {
    std::vector<f64> lo_d, hi_d; // decomposition
    std::vector<f64> lo_r, hi_r; // reconstruction

    // Daubechies-4 (db2): 4-tap filter
    static WaveletFilter db4() {
        static constexpr f64 s=std::numbers::sqrt3_v<f64>;
        static constexpr f64 D=4*std::numbers::sqrt2_v<f64>;
        WaveletFilter f;
        f.lo_d={(1+s)/D,(3+s)/D,(3-s)/D,(1-s)/D};
        f.hi_d={-(1-s)/D,(3-s)/D,-(3+s)/D,(1+s)/D};
        f.lo_r={f.lo_d[3],f.lo_d[2],f.lo_d[1],f.lo_d[0]};
        f.hi_r={-f.hi_d[3],f.hi_d[2],-f.hi_d[1],f.hi_d[0]};
        return f;
    }
    // Daubechies-8 (db4): 8-tap
    static WaveletFilter db8() {
        WaveletFilter f;
        f.lo_d={0.2303778133088965,0.7148465705529156,0.6308807679298587,
                -0.02798376941685985,-0.18703481171909309,0.030841381835560764,
                0.032883011666885253,-0.010597401785069033};
        f.hi_d.resize(8);
        for(u32 i=0;i<8;++i) f.hi_d[i]=(i%2==0?-1.0:1.0)*f.lo_d[7-i];
        f.lo_r=std::vector<f64>(f.lo_d.rbegin(),f.lo_d.rend());
        f.hi_r=std::vector<f64>(f.hi_d.rbegin(),f.hi_d.rend());
        return f;
    }
    // Daubechies-2 (Haar)
    static WaveletFilter haar() {
        WaveletFilter f;
        f64 r=1.0/std::numbers::sqrt2_v<f64>;
        f.lo_d={r,r}; f.hi_d={r,-r};
        f.lo_r={r,r}; f.hi_r={-r,r};
        return f;
    }
    // Symmlet-8 (sym4)
    static WaveletFilter sym8() {
        WaveletFilter f;
        f.lo_d={-0.07576571478950477,-0.02963552764599851,0.4976186676332881,
                 0.8037387518052163, 0.2978577956051718,-0.09921954357685224,
                -0.012603967262261656,0.03222310060404270};
        f.hi_d.resize(8);
        for(u32 i=0;i<8;++i) f.hi_d[i]=(i%2==0?1.0:-1.0)*f.lo_d[7-i];
        f.lo_r=std::vector<f64>(f.lo_d.rbegin(),f.lo_d.rend());
        f.hi_r=std::vector<f64>(f.hi_d.rbegin(),f.hi_d.rend());
        return f;
    }
};

// Convolution + downsampling (analysis)
[[nodiscard]] inline std::vector<f64> conv_down(const std::vector<f64>& x, const std::vector<f64>& h) {
    u32 N=u32(x.size()), M=u32(h.size());
    u32 out=(N+M-1)/2;
    std::vector<f64> y(out,0);
    for(u32 i=0;i<out;++i) {
        u32 n=2*i;
        for(u32 k=0;k<M;++k) {
            u32 idx=n+k;
            if(idx<N) y[i]+=h[k]*x[idx];
            else if(idx<N+M) y[i]+=h[k]*x[(N-1)-(idx-N)]; // symmetric extension
        }
    }
    return y;
}

// Upsampling + convolution (synthesis)
[[nodiscard]] inline std::vector<f64> up_conv(const std::vector<f64>& x, const std::vector<f64>& h, u32 out_len) {
    u32 N=u32(x.size()), M=u32(h.size());
    std::vector<f64> y(out_len,0);
    for(u32 i=0;i<N;++i) {
        u32 n=2*i;
        for(u32 k=0;k<M;++k)
            if(n+k<out_len) y[n+k]+=x[i]*h[k];
    }
    return y;
}

// Single-level DWT decomposition
struct DWTLevel { std::vector<f64> approx, detail; };

[[nodiscard]] inline DWTLevel dwt1d(const std::vector<f64>& x, const WaveletFilter& filt) {
    return {conv_down(x,filt.lo_d), conv_down(x,filt.hi_d)};
}

// Single-level IDWT
[[nodiscard]] inline std::vector<f64> idwt1d(const DWTLevel& dl, const WaveletFilter& filt, u32 N) {
    auto lo=up_conv(dl.approx,filt.lo_r,N);
    auto hi=up_conv(dl.detail,filt.hi_r,N);
    std::vector<f64> r(N);
    for(u32 i=0;i<N;++i) r[i]=lo[i]+hi[i];
    return r;
}

// Multi-level wavelet packet decomposition
struct WaveletPacket {
    std::vector<std::vector<f64>> coeffs; // [level][samples]
    u32 levels{0};

    static WaveletPacket decompose(const std::vector<f64>& x, const WaveletFilter& filt, u32 n_levels) {
        WaveletPacket wp; wp.levels=n_levels;
        wp.coeffs.push_back(x);
        std::vector<f64> approx=x;
        for(u32 l=0;l<n_levels;++l) {
            auto [a,d]=dwt1d(approx,filt);
            wp.coeffs.push_back(d);
            approx=a;
            if(approx.size()<2) break;
        }
        wp.coeffs.push_back(approx);
        std::reverse(wp.coeffs.begin(),wp.coeffs.end()); // coarsest first
        return wp;
    }

    // Soft thresholding (denoising)
    void soft_threshold(f64 t) {
        for(u32 l=1;l<u32(coeffs.size());++l) // skip DC
            for(auto& c:coeffs[l]) c=std::copysign(std::max(0.0,std::abs(c)-t),c);
    }

    // Energy per level
    [[nodiscard]] std::vector<f64> energy() const {
        std::vector<f64> e(coeffs.size());
        for(u32 l=0;l<u32(coeffs.size());++l) for(auto v:coeffs[l]) e[l]+=v*v;
        return e;
    }
};

// ─── Continuous Wavelet Transform (CWT) ──────────────────────────────────────
// Mother wavelets

// Morlet wavelet: ψ(t) = e^{iω0t} e^{-t²/2}  (ω0=6 typical)
[[nodiscard]] inline std::complex<f64> morlet(f64 t, f64 omega0=6.0) {
    return std::exp(std::complex<f64>(0,omega0*t)) * std::exp(-t*t*0.5);
}

// Mexican Hat (Ricker): ψ(t) = (1-t²)e^{-t²/2}
[[nodiscard]] inline f64 ricker_wavelet(f64 t) {
    return (1-t*t)*std::exp(-t*t*0.5)*2.0/std::numbers::sqrt3_v<f64>/std::pow(std::numbers::pi_v<f64>,0.25);
}

// CWT scalogram using Morlet
struct ScalogramResult {
    std::vector<f64>  scales;
    std::vector<std::vector<f64>> power; // [scale][time]
};

[[nodiscard]] inline ScalogramResult cwt_morlet(const std::vector<f64>& x,
    const std::vector<f64>& scales, f64 dt=1.0, f64 omega0=6.0)
{
    u32 N=u32(x.size());
    ScalogramResult res; res.scales=scales;
    res.power.resize(scales.size(),std::vector<f64>(N,0));
    for(u32 si=0;si<u32(scales.size());++si) {
        f64 s=scales[si];
        // Compute wavelet at this scale via convolution
        for(u32 t=0;t<N;++t) {
            std::complex<f64> W(0,0);
            for(u32 n=0;n<N;++n) {
                f64 eta=(f64(n)-f64(t))*dt/s;
                if(std::abs(eta)>8) continue;
                W+=x[n]*std::conj(morlet(eta,omega0))*std::sqrt(dt/s);
            }
            res.power[si][t]=std::norm(W);
        }
    }
    return res;
}

// ─── Wavelet denoising (VisuShrink) ──────────────────────────────────────────
[[nodiscard]] inline std::vector<f64> wavelet_denoise(
    const std::vector<f64>& noisy, const WaveletFilter& filt, u32 levels=4)
{
    auto wp=WaveletPacket::decompose(noisy,filt,levels);
    // Estimate noise from finest detail
    auto& finest=wp.coeffs.back();
    f64 sigma=0; for(auto v:finest) sigma+=std::abs(v); sigma/=finest.size();
    sigma/=0.6745; // MAD estimator
    f64 threshold=sigma*std::sqrt(2*std::log(f64(noisy.size())));
    wp.soft_threshold(threshold);
    // Reconstruct (simplified: sum all levels)
    std::vector<f64> result=wp.coeffs[0]; // start from coarsest
    for(u32 l=1;l<u32(wp.coeffs.size());++l) {
        auto detail=wp.coeffs[l];
        DWTLevel dl{result,detail};
        u32 target_len=l+1<u32(wp.coeffs.size())?u32(wp.coeffs[l+1].size())*2:u32(noisy.size());
        result=idwt1d(dl,filt,std::min(target_len,u32(noisy.size())));
    }
    result.resize(noisy.size(),0);
    return result;
}

} // namespace qs
