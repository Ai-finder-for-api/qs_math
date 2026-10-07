#pragma once
// qs_math/signal/windowing.hpp — All standard window functions + DPSS
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <numbers>
#include <array>

namespace qs::math {

// ─── Window functions (N = length, normalized 0..1) ──────────────────────────

template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_rect() noexcept {
    std::array<f64,N> w{}; w.fill(1.0); return w;
}

template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_hann() noexcept {
    std::array<f64,N> w{};
    for(int n=0;n<N;++n) w[n]=0.5*(1.0-std::cos(2.0*std::numbers::pi_v<f64>*n/(N-1)));
    return w;
}

template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_hamming() noexcept {
    std::array<f64,N> w{};
    for(int n=0;n<N;++n) w[n]=0.54-0.46*std::cos(2.0*std::numbers::pi_v<f64>*n/(N-1));
    return w;
}

template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_blackman() noexcept {
    std::array<f64,N> w{};
    constexpr f64 pi=std::numbers::pi_v<f64>;
    for(int n=0;n<N;++n)
        w[n]=0.42-0.5*std::cos(2.0*pi*n/(N-1))+0.08*std::cos(4.0*pi*n/(N-1));
    return w;
}

template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_blackman_harris() noexcept {
    std::array<f64,N> w{};
    constexpr f64 pi=std::numbers::pi_v<f64>;
    for(int n=0;n<N;++n) {
        f64 x=2.0*pi*n/(N-1);
        w[n]=0.35875-0.48829*std::cos(x)+0.14128*std::cos(2*x)-0.01168*std::cos(3*x);
    }
    return w;
}

template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_flattop() noexcept {
    std::array<f64,N> w{};
    constexpr f64 pi=std::numbers::pi_v<f64>;
    for(int n=0;n<N;++n) {
        f64 x=2.0*pi*n/(N-1);
        w[n]=0.21557895-0.41663158*std::cos(x)+0.27726316*std::cos(2*x)
            -0.08357895*std::cos(3*x)+0.00694737*std::cos(4*x);
    }
    return w;
}

// Bartlett (triangular) window
template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_bartlett() noexcept {
    std::array<f64,N> w{};
    for(int n=0;n<N;++n)
        w[n]=1.0-std::abs(2.0*n/(N-1)-1.0);
    return w;
}

// Nuttall window (4-term, continuous first derivative)
template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_nuttall() noexcept {
    std::array<f64,N> w{};
    constexpr f64 pi=std::numbers::pi_v<f64>;
    for(int n=0;n<N;++n) {
        f64 x=2.0*pi*n/(N-1);
        w[n]=0.355768-0.487396*std::cos(x)+0.144232*std::cos(2*x)-0.012604*std::cos(3*x);
    }
    return w;
}

// Kaiser window (beta controls sidelobe level vs main lobe width)
// Uses modified Bessel function I0
[[nodiscard]] constexpr f64 bessel_i0(f64 x) noexcept {
    // Series: I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)^2
    f64 s=1.0, t=1.0, half_x=x*0.5;
    for(int k=1;k<=20;++k) {
        t*=half_x/k; s+=t*t;
    }
    return s;
}

template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_kaiser(f64 beta) noexcept {
    std::array<f64,N> w{};
    f64 inv_i0=1.0/bessel_i0(beta);
    for(int n=0;n<N;++n) {
        f64 t=2.0*n/(N-1)-1.0;
        w[n]=bessel_i0(beta*std::sqrt(1.0-t*t))*inv_i0;
    }
    return w;
}

// Gaussian window
template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_gaussian(f64 sigma) noexcept {
    std::array<f64,N> w{};
    for(int n=0;n<N;++n) {
        f64 t=(n-(N-1)/2.0)/(sigma*(N-1)/2.0);
        w[n]=std::exp(-0.5*t*t);
    }
    return w;
}

// Tukey (cosine-tapered) window: alpha=0 → rect, alpha=1 → Hann
template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_tukey(f64 alpha) noexcept {
    std::array<f64,N> w{};
    constexpr f64 pi=std::numbers::pi_v<f64>;
    int taper=int(alpha*(N-1)/2.0);
    for(int n=0;n<N;++n) {
        if(n<taper)
            w[n]=0.5*(1.0-std::cos(pi*n/taper));
        else if(n<=N-1-taper)
            w[n]=1.0;
        else
            w[n]=0.5*(1.0-std::cos(pi*(N-1-n)/taper));
    }
    return w;
}

// Parzen (de la Vallée-Poussin) window
template<int N>
[[nodiscard]] constexpr std::array<f64,N> window_parzen() noexcept {
    std::array<f64,N> w{};
    f64 L=N-1;
    for(int n=0;n<N;++n) {
        f64 t=std::abs(2.0*n/L-1.0);
        if(t<=0.5) w[n]=1.0-6.0*t*t+6.0*t*t*t;
        else       w[n]=2.0*(1.0-t)*(1.0-t)*(1.0-t);
    }
    return w;
}

// ─── Window metrics ──────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 window_coherent_gain(const std::array<f64,N>& w) noexcept {
    f64 s=0; for(auto v:w) s+=v; return s/N;
}

template<int N>
[[nodiscard]] constexpr f64 window_power_gain(const std::array<f64,N>& w) noexcept {
    f64 s=0; for(auto v:w) s+=v*v; return s/N;
}

// ─── Zero-phase windowing (symmetric application) ─────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> apply_window(
    const std::array<f64,N>& x, const std::array<f64,N>& w) noexcept
{
    std::array<f64,N> out{};
    for(int i=0;i<N;++i) out[i]=x[i]*w[i];
    return out;
}

// ─── Optimal beta for Kaiser from desired attenuation ─────────────────────────
[[nodiscard]] constexpr f64 kaiser_beta_from_attenuation(f64 As_db) noexcept {
    if(As_db>50.0) return 0.1102*(As_db-8.7);
    if(As_db>21.0) return 0.5842*std::pow(As_db-21.0,0.4)+0.07886*(As_db-21.0);
    return 0.0;
}

// Minimum Kaiser order from transition band width and attenuation
[[nodiscard]] constexpr int kaiser_order(f64 delta_f, f64 As_db) noexcept {
    return int(std::ceil((As_db-8.0)/(2.285*2.0*std::numbers::pi_v<f64>*delta_f)));
}

} // namespace qs::math
