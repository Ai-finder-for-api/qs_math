#pragma once
// qs_math/signal/z_transform.hpp — Z-transform, transfer functions, stability analysis
#include "../arithmetic/scalar_types.hpp"
#include "fft.hpp"
#include <cmath>
#include <numbers>
#include <array>

namespace qs::math {

// ─── Transfer function H(z) = B(z)/A(z) ──────────────────────────────────────
template<int NB, int NA>
struct TransferFunction {
    std::array<f64,NB> b{}; // numerator coefficients b[0]+b[1]z^-1+...
    std::array<f64,NA> a{}; // denominator a[0]=1 (monic)

    // Evaluate H(e^{jw}) at digital frequency w (0..pi)
    [[nodiscard]] constexpr Cx eval(f64 w) const noexcept {
        Cx num{}, den{};
        for(int k=0;k<NB;++k) num=num+Cx{b[k],0}*cx_polar(1.0,-w*k);
        den={1,0};
        for(int k=1;k<NA;++k) den=den+Cx{a[k],0}*cx_polar(1.0,-w*k);
        // H = num/den
        f64 d2=den.mag2();
        if(d2<1e-30) return {};
        return Cx{(num*den.conj()).re/d2,(num*den.conj()).im/d2};
    }

    // Magnitude response at w
    [[nodiscard]] constexpr f64 magnitude(f64 w) const noexcept { return eval(w).mag(); }

    // Phase response at w
    [[nodiscard]] constexpr f64 phase(f64 w) const noexcept {
        auto h=eval(w);
        return std::atan2(h.im,h.re);
    }

    // Group delay = -d(phase)/dw (finite difference)
    [[nodiscard]] constexpr f64 group_delay(f64 w, f64 dw=1e-5) const noexcept {
        return -(phase(w+dw)-phase(w-dw))/(2.0*dw);
    }
};

// ─── Pole-zero representation ─────────────────────────────────────────────────
template<int N_POLES, int N_ZEROS>
struct PoleZero {
    std::array<Cx,N_POLES> poles{};
    std::array<Cx,N_ZEROS> zeros{};
    f64 gain{1.0};

    // Evaluate H(z) at z
    [[nodiscard]] constexpr Cx eval(Cx z) const noexcept {
        Cx num={gain,0}, den={1,0};
        for(auto& zr: zeros) num=num*(z-zr);
        for(auto& p : poles) den=den*(z-p);
        f64 d2=den.mag2();
        if(d2<1e-30) return {};
        return Cx{(num*den.conj()).re/d2,(num*den.conj()).im/d2};
    }

    // Stability: all poles inside unit circle
    [[nodiscard]] constexpr bool is_stable() const noexcept {
        for(auto& p: poles) if(p.mag()>=1.0) return false;
        return true;
    }

    // Minimum phase: all zeros inside unit circle
    [[nodiscard]] constexpr bool is_minimum_phase() const noexcept {
        for(auto& z: zeros) if(z.mag()>=1.0) return false;
        return true;
    }
};

// ─── Jury stability criterion ─────────────────────────────────────────────────
// Polynomial a[0] + a[1]z + ... + a[N]z^N (z-domain)
// Returns true if all roots inside unit circle
template<int N>
[[nodiscard]] constexpr bool jury_stable(std::array<f64,N+1> a) noexcept {
    // Necessary: a[0]>0, a[N]<|a[0]| pattern, etc.
    // Build Jury table row by row
    for(int k=N;k>=2;--k) {
        f64 an=a[k], a0=a[0];
        if(std::abs(an)>=std::abs(a0)) return false;
        f64 r=an/a0;
        std::array<f64,N+1> b{};
        for(int i=0;i<k;++i) b[i]=a[i]-r*a[k-i];
        for(int i=0;i<=k;++i) a[i]=b[i];
    }
    return a[0]>0;
}

// ─── Bilinear transform of transfer function ──────────────────────────────────
// Maps s-domain H_a(s) to z-domain H_d(z) via s=2/T*(z-1)/(z+1)
// Evaluates H_d(z) = H_a(2/T*(z-1)/(z+1))
[[nodiscard]] constexpr Cx bilinear_eval(
    const Cx* poles_s, int n_poles, const Cx* zeros_s, int n_zeros,
    f64 gain_s, Cx z, f64 T) noexcept
{
    Cx s={2.0/T,0}*(z-Cx{1,0}); // s = 2/T*(z-1)
    // divide by (z+1) properly
    Cx denom_z=z+Cx{1,0};
    // For each factor (s-p) we get (2/T*(z-1)/(z+1)-p)
    // = (2/T*(z-1)-p*(z+1)) / (z+1)
    // numerator product / denominator product
    Cx num={gain_s,0}, den={1,0};
    for(int i=0;i<n_zeros;++i) {
        Cx zi=Cx{2.0/T,0}*(z-Cx{1,0})-(zeros_s[i]*denom_z);
        num=num*zi;
    }
    for(int i=0;i<n_poles;++i) {
        Cx pi=Cx{2.0/T,0}*(z-Cx{1,0})-(poles_s[i]*denom_z);
        den=den*pi;
    }
    f64 d2=den.mag2(); if(d2<1e-30) return {};
    return Cx{(num*den.conj()).re/d2,(num*den.conj()).im/d2};
}

// ─── Z-transform of common sequences ──────────────────────────────────────────
// Unit impulse: Z{δ[n]} = 1
// Unit step: Z{u[n]} = z/(z-1), |z|>1
// Exponential: Z{a^n*u[n]} = z/(z-a), |z|>|a|
// Ramp: Z{n*u[n]} = z/(z-1)^2

// Evaluate Z-transform of a^n*u[n] at z
[[nodiscard]] constexpr Cx z_exponential(Cx z, Cx a) noexcept {
    // z/(z-a)
    Cx den=z-a; f64 d2=den.mag2(); if(d2<1e-30) return {};
    return Cx{(z*den.conj()).re/d2,(z*den.conj()).im/d2};
}

// Evaluate Z{n*a^n*u[n]} = az/(z-a)^2 at z
[[nodiscard]] constexpr Cx z_ramp_exp(Cx z, Cx a) noexcept {
    Cx den=(z-a)*(z-a); f64 d2=den.mag2(); if(d2<1e-30) return {};
    Cx num=a*z;
    return Cx{(num*den.conj()).re/d2,(num*den.conj()).im/d2};
}

// ─── Impulse response from TF (recursive, causal) ────────────────────────────
template<int NB, int NA, int LEN>
[[nodiscard]] constexpr std::array<f64,LEN> impulse_response(
    const TransferFunction<NB,NA>& tf) noexcept
{
    std::array<f64,LEN> h{};
    std::array<f64,NA>  prev_y{};
    for(int n=0;n<LEN;++n) {
        f64 x=(n==0)?1.0:0.0; // impulse
        f64 y=(n<NB)?tf.b[n]:0.0;
        for(int k=1;k<NA;++k) if(n-k>=0) y-=tf.a[k]*prev_y[(n-k)%NA];
        h[n]=y;
        prev_y[n%NA]=y;
    }
    return h;
}

// ─── Partial fraction decomposition (for simple poles) ───────────────────────
// Given poles p[0..M-1] all distinct, and numerator evaluated at poles,
// compute residues r[k] = N(p[k]) / prod_{j!=k}(p[k]-p[j])
template<int M>
[[nodiscard]] constexpr std::array<Cx,M> partial_fraction_residues(
    const std::array<Cx,M>& poles,
    const std::array<Cx,M>& numerator_at_poles) noexcept
{
    std::array<Cx,M> r{};
    for(int k=0;k<M;++k) {
        Cx denom={1,0};
        for(int j=0;j<M;++j) if(j!=k) denom=denom*(poles[k]-poles[j]);
        f64 d2=denom.mag2(); if(d2<1e-30) continue;
        r[k]=Cx{(numerator_at_poles[k]*denom.conj()).re/d2,
                (numerator_at_poles[k]*denom.conj()).im/d2};
    }
    return r;
}

// ─── Frequency response on N-point grid ───────────────────────────────────────
template<int NB, int NA, int NFFT>
[[nodiscard]] constexpr std::array<f64,NFFT/2+1> freqz(
    const TransferFunction<NB,NA>& tf) noexcept
{
    std::array<f64,NFFT/2+1> mag{};
    for(int k=0;k<=NFFT/2;++k) {
        f64 w=std::numbers::pi_v<f64>*k/(NFFT/2);
        mag[k]=tf.magnitude(w);
    }
    return mag;
}

// ─── Cepstrum: IFFT(log|FFT(x)|) ─────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> real_cepstrum(
    const std::array<f64,N>& x) noexcept
{
    auto X=rfft<N>(x);
    // log magnitude spectrum
    std::array<Cx,N> logX{};
    for(int k=0;k<N/2+1;++k) {
        f64 mag=X[k].mag(); if(mag<1e-12) mag=1e-12;
        logX[k]={std::log(mag),0};
        if(k>0&&k<N/2) logX[N-k]={std::log(mag),0};
    }
    fft_inplace<N>(logX,true);
    std::array<f64,N> c{};
    for(int i=0;i<N;++i) c[i]=logX[i].re;
    return c;
}

} // namespace qs::math
