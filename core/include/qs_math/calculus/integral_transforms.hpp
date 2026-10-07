#pragma once
// qs_math/calculus/integral_transforms.hpp — Laplace, Mellin, Hankel, Hilbert transforms
#include "../arithmetic/scalar_types.hpp"
#include "../signal/fft.hpp"
#include <array>
#include <cmath>
#include <numbers>

namespace qs::math {

// ─── Laplace transform (numerical, single-sided) ──────────────────────────────
// F(s) = ∫_0^∞ f(t) e^{-st} dt
template<typename F>
[[nodiscard]] constexpr Cx laplace_numerical(F&& f, Cx s, f64 T=20, int n=1000) noexcept {
    Cx result{};
    f64 dt=T/n;
    for(int k=0;k<n;++k) {
        f64 t=(k+0.5)*dt;
        f64 fv=f(t);
        f64 exp_r=std::exp(-s.re*t);
        result.re+=fv*exp_r*std::cos(s.im*t)*dt;
        result.im+=fv*exp_r*(-std::sin(s.im*t))*dt;
    }
    return result;
}

// Common Laplace transform pairs
// L{1} = 1/s
[[nodiscard]] constexpr Cx laplace_unit_step(Cx s) noexcept { return cx_div({1,0},s); }
// L{t^n} = n!/s^{n+1}
[[nodiscard]] constexpr Cx laplace_power(Cx s, int n) noexcept {
    f64 fact=1; for(int k=1;k<=n;++k) fact*=k;
    Cx sn={1,0}; for(int k=0;k<=n;++k) sn=sn*s;
    return cx_div({fact,0},sn);
}
// L{e^{at}} = 1/(s-a)
[[nodiscard]] constexpr Cx laplace_exp(Cx s, f64 a) noexcept {
    return cx_div({1,0},s-Cx{a,0});
}
// L{sin(ωt)} = ω/(s²+ω²)
[[nodiscard]] constexpr Cx laplace_sin(Cx s, f64 omega) noexcept {
    return cx_div({0,omega},s*s+Cx{omega*omega,0});
}
// L{cos(ωt)} = s/(s²+ω²)
[[nodiscard]] constexpr Cx laplace_cos(Cx s, f64 omega) noexcept {
    return cx_div(s,s*s+Cx{omega*omega,0});
}

// ─── Inverse Laplace (Talbot's method) ────────────────────────────────────────
template<typename LF>
[[nodiscard]] constexpr f64 inverse_laplace_talbot(LF&& F, f64 t, int N=24) noexcept {
    // Talbot's contour deformation
    f64 r=2*N/(5*t);
    f64 result=0;
    for(int k=0;k<N;++k) {
        f64 theta=(k==0)?0:k*3.14159265358979323846/N;
        Cx delta;
        if(k==0){delta={r,0};}
        else{
            f64 cot=std::cos(theta)/std::sin(theta);
            delta=Cx{r*theta*(cot+1),r*theta*(1-theta*cot)};
        }
        Cx sigma=Cx{r,0}; if(k>0)sigma=delta;
        f64 dphi=(k==0)?1:0; // simplified
        // Full Talbot has specific weight function
        auto Fv=F(sigma);
        result+=std::exp(sigma.re*t)*Fv.re;
    }
    return result*2.0/(5*t);
}

// ─── Mellin transform: M{f}(s) = ∫_0^∞ x^{s-1} f(x) dx ──────────────────────
template<typename F>
[[nodiscard]] constexpr Cx mellin_transform(F&& f, Cx s, f64 x_max=100, int n=500) noexcept {
    Cx result{};
    f64 dx=x_max/n;
    for(int k=1;k<n;++k) {
        f64 x=k*dx;
        f64 fv=f(x);
        // x^{s-1} = exp((s-1)*log(x))
        f64 log_x=std::log(x);
        f64 re=(s.re-1)*log_x, im=s.im*log_x;
        result.re+=fv*std::exp(re)*std::cos(im)*dx;
        result.im+=fv*std::exp(re)*std::sin(im)*dx;
    }
    return result;
}

// ─── Hankel transform (Fourier-Bessel) ────────────────────────────────────────
// H_ν{f}(k) = ∫_0^∞ f(r) J_ν(kr) r dr
// Bessel function J_0 (power series)
[[nodiscard]] constexpr f64 bessel_J0(f64 x) noexcept {
    f64 s=1,t=1;
    f64 x2=x*x*0.25;
    for(int k=1;k<=15;++k){t*=-x2/(k*k);s+=t;}
    return s;
}
[[nodiscard]] constexpr f64 bessel_J1(f64 x) noexcept {
    f64 s=0.5*x,t=0.5*x;
    f64 x2=x*x*0.25;
    for(int k=1;k<=15;++k){t*=-x2/(k*(k+1));s+=t;}
    return s;
}
[[nodiscard]] constexpr f64 bessel_Jn(int n, f64 x) noexcept {
    if(n==0) return bessel_J0(x);
    if(n==1) return bessel_J1(x);
    // Recurrence: J_{n+1}(x) = (2n/x)*J_n(x) - J_{n-1}(x)
    f64 J0=bessel_J0(x),J1=bessel_J1(x);
    for(int k=1;k<n;++k){f64 J2=(2*k/x)*J1-J0;J0=J1;J1=J2;}
    return J1;
}

template<typename F>
[[nodiscard]] constexpr f64 hankel_transform(F&& f, int nu, f64 k,
                                               f64 r_max=20, int n=500) noexcept {
    f64 dr=r_max/n,result=0;
    for(int i=1;i<n;++i){
        f64 r=i*dr;
        result+=f(r)*bessel_Jn(nu,k*r)*r*dr;
    }
    return result;
}

// ─── Hilbert transform (via FFT) ─────────────────────────────────────────────
// H{f}(t) = (1/π) PV ∫ f(τ)/(t-τ) dτ
// In freq domain: H(F)(ω) = -i*sgn(ω)*F(ω)
template<int N>
[[nodiscard]] constexpr std::array<f64,N> hilbert_transform(
    const std::array<f64,N>& x) noexcept
{
    std::array<Cx,N> X{};
    for(int i=0;i<N;++i) X[i]={x[i],0};
    fft_inplace<N>(X,false);
    // Multiply by -i*sgn(k)
    for(int k=1;k<N/2;++k){f64 re=X[k].re;X[k].re=X[k].im;X[k].im=-re;}
    for(int k=N/2+1;k<N;++k){f64 re=X[k].re;X[k].re=-X[k].im;X[k].im=re;}
    X[0]={0,0}; X[N/2]={0,0};
    fft_inplace<N>(X,true);
    std::array<f64,N> h{};
    for(int i=0;i<N;++i) h[i]=X[i].re;
    return h;
}

// Analytic signal: x_a(t) = x(t) + i*H{x}(t)
template<int N>
[[nodiscard]] constexpr std::array<Cx,N> analytic_signal(
    const std::array<f64,N>& x) noexcept
{
    auto h=hilbert_transform<N>(x);
    std::array<Cx,N> xa{};
    for(int i=0;i<N;++i) xa[i]={x[i],h[i]};
    return xa;
}

// Instantaneous frequency from analytic signal
template<int N>
[[nodiscard]] constexpr std::array<f64,N> instantaneous_frequency(
    const std::array<Cx,N>& xa, f64 sample_rate) noexcept
{
    std::array<f64,N> fi{};
    for(int i=1;i<N-1;++i) {
        f64 phi_next=std::atan2(xa[i+1].im,xa[i+1].re);
        f64 phi_prev=std::atan2(xa[i-1].im,xa[i-1].re);
        fi[i]=(phi_next-phi_prev)*sample_rate/(4*std::numbers::pi_v<f64>);
    }
    fi[0]=fi[1]; fi[N-1]=fi[N-2];
    return fi;
}

// ─── Fractional Fourier transform (FrFT) ─────────────────────────────────────
// Rotation by angle α in time-frequency plane
// F_α{f}(u) = ∫ K_α(u,t) f(t) dt
// Kernel: K_α(u,t) = exp(iπ(u²cot α - 2ut/sin α + t²cot α)) / sqrt(1-i*cot α)
template<int N>
[[nodiscard]] constexpr std::array<Cx,N> frft(
    const std::array<Cx,N>& f, f64 alpha, f64 dt=1.0) noexcept
{
    if(std::abs(std::fmod(alpha,std::numbers::pi_v<f64>))<1e-6) return f; // identity
    std::array<Cx,N> result{};
    f64 cot=std::cos(alpha)/std::sin(alpha);
    f64 inv_sin=1.0/std::sin(alpha);
    f64 T=N*dt;
    f64 du=T/N;
    for(int m=0;m<N;++m) {
        f64 u=(m-N/2)*du;
        Cx sum{};
        for(int n=0;n<N;++n) {
            f64 t=(n-N/2)*dt;
            f64 phase=std::numbers::pi_v<f64>*(u*u*cot-2*u*t*inv_sin+t*t*cot);
            sum=sum+f[n]*Cx{std::cos(phase),std::sin(phase)};
        }
        result[m]=sum*dt;
    }
    return result;
}
} // namespace qs::math
