#pragma once
// qs_math/analysis/measure_theory.hpp — Sigma-algebras, Lebesgue measure, L^p spaces, integration
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>
#include <numeric>

namespace qs::math {

// ─── Measurable set representation (indicator via predicate) ──────────────────
// A measurable set E ⊂ R^N given as a predicate
// Lebesgue measure μ(E) approximated by Monte Carlo or grid integration

// Lebesgue measure of E ∩ [a,b]^N by grid approximation
template<int N, typename Pred>
[[nodiscard]] constexpr f64 lebesgue_measure_box(
    Pred&& indicator,
    const std::array<f64,N>& a, const std::array<f64,N>& b,
    int grid_pts=50) noexcept
{
    f64 vol=1; for(int i=0;i<N;++i) vol*=(b[i]-a[i]);
    f64 h=1.0/grid_pts;
    i64 count=0, total=1;
    for(int i=0;i<N;++i) total*=grid_pts;
    for(i64 idx=0;idx<total;++idx) {
        std::array<f64,N> pt;
        i64 tmp=idx;
        for(int d=0;d<N;++d){
            int k=tmp%grid_pts; tmp/=grid_pts;
            pt[d]=a[d]+(k+0.5)*(b[d]-a[d])/grid_pts;
        }
        if(indicator(pt)) ++count;
    }
    return vol*f64(count)/f64(total);
}

// ─── Outer measure (Carathéodory construction via intervals) ──────────────────
// For a function on a finite set of points, compute the "outer measure" mass
template<int N>
[[nodiscard]] constexpr f64 counting_measure(const bool* membership, int size) noexcept {
    f64 s=0; for(int i=0;i<size;++i) if(membership[i]) ++s; return s;
}

// ─── Lebesgue integral (via Riemann sum fallback, converges to Lebesgue for a.e. cont. f) ──
template<typename F>
[[nodiscard]] constexpr f64 lebesgue_integral_1d(F&& f, f64 a, f64 b, int n=1000) noexcept {
    f64 h=(b-a)/n, s=0;
    for(int i=0;i<n;++i) {
        f64 fv=f(a+(i+0.5)*h);
        if(std::isfinite(fv)) s+=fv;
    }
    return h*s;
}

// ─── L^p norm of a function on [a,b] ─────────────────────────────────────────
template<typename F>
[[nodiscard]] constexpr f64 lp_norm(F&& f, f64 a, f64 b, f64 p, int n=1000) noexcept {
    f64 h=(b-a)/n, s=0;
    for(int i=0;i<n;++i) {
        f64 fv=std::abs(f(a+(i+0.5)*h));
        if(std::isfinite(fv)) s+=std::pow(fv,p);
    }
    return std::pow(h*s, 1.0/p);
}

// L^inf norm (essential supremum via grid)
template<typename F>
[[nodiscard]] constexpr f64 linf_norm(F&& f, f64 a, f64 b, int n=1000) noexcept {
    f64 m=0;
    for(int i=0;i<n;++i) {
        f64 fv=std::abs(f(a+(i+0.5)*(b-a)/n));
        if(std::isfinite(fv)&&fv>m) m=fv;
    }
    return m;
}

// ─── Almost-everywhere equality check (sample-based) ─────────────────────────
template<typename F, typename G>
[[nodiscard]] constexpr bool ae_equal(F&& f, G&& g, f64 a, f64 b,
                                       f64 tol=1e-9, int n=1000) noexcept {
    int failures=0;
    for(int i=0;i<n;++i) {
        f64 x=a+(i+0.5)*(b-a)/n;
        if(std::abs(f(x)-g(x))>tol) ++failures;
    }
    return f64(failures)/n < 0.01; // a.e.: less than 1% failures
}

// ─── Product measure (Fubini's theorem) ───────────────────────────────────────
// ∫∫ f(x,y) dxdy = ∫ (∫ f(x,y) dy) dx
template<typename F>
[[nodiscard]] constexpr f64 product_measure_integral(
    F&& f, f64 ax, f64 bx, f64 ay, f64 by,
    int nx=100, int ny=100) noexcept
{
    f64 hx=(bx-ax)/nx, hy=(by-ay)/ny, s=0;
    for(int i=0;i<nx;++i) {
        f64 x=ax+(i+0.5)*hx, inner=0;
        for(int j=0;j<ny;++j) {
            f64 y=ay+(j+0.5)*hy, fv=f(x,y);
            if(std::isfinite(fv)) inner+=fv;
        }
        s+=inner*hy;
    }
    return hx*s;
}

// ─── Radon-Nikodym derivative (discrete) ─────────────────────────────────────
// Given measures μ, ν on finite set, compute dμ/dν at each point
template<int N>
[[nodiscard]] constexpr std::array<f64,N> radon_nikodym(
    const std::array<f64,N>& mu, const std::array<f64,N>& nu) noexcept
{
    std::array<f64,N> dmu_dnu{};
    for(int i=0;i<N;++i)
        dmu_dnu[i]=(nu[i]>1e-15)?mu[i]/nu[i]:0.0;
    return dmu_dnu;
}

// ─── Signed measure (Hahn decomposition) ──────────────────────────────────────
// A signed measure can be decomposed into positive and negative parts
template<int N>
struct SignedMeasure {
    std::array<f64,N> values{};

    [[nodiscard]] constexpr std::array<f64,N> positive_part() const noexcept {
        std::array<f64,N> p{};
        for(int i=0;i<N;++i) p[i]=std::max(0.0,values[i]);
        return p;
    }
    [[nodiscard]] constexpr std::array<f64,N> negative_part() const noexcept {
        std::array<f64,N> n{};
        for(int i=0;i<N;++i) n[i]=std::max(0.0,-values[i]);
        return n;
    }
    [[nodiscard]] constexpr f64 total_variation() const noexcept {
        f64 s=0; for(auto v:values) s+=std::abs(v); return s;
    }
};

// ─── Probability measure properties ───────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr bool is_probability_measure(const std::array<f64,N>& p) noexcept {
    f64 s=0; for(auto v:p){if(v<0) return false;s+=v;}
    return std::abs(s-1.0)<1e-9;
}

template<int N>
[[nodiscard]] constexpr f64 entropy(const std::array<f64,N>& p) noexcept {
    f64 H=0;
    for(auto v:p) if(v>1e-15) H-=v*std::log(v);
    return H;
}

template<int N>
[[nodiscard]] constexpr f64 kl_divergence(const std::array<f64,N>& p,
                                            const std::array<f64,N>& q) noexcept {
    f64 kl=0;
    for(int i=0;i<N;++i)
        if(p[i]>1e-15&&q[i]>1e-15) kl+=p[i]*std::log(p[i]/q[i]);
    return kl;
}

// ─── Convolution of measures (finite, discrete) ───────────────────────────────
template<int M, int N>
[[nodiscard]] constexpr std::array<f64,M+N-1> measure_convolution(
    const std::array<f64,M>& f, const std::array<f64,N>& g) noexcept
{
    std::array<f64,M+N-1> h{};
    for(int i=0;i<M;++i) for(int j=0;j<N;++j) h[i+j]+=f[i]*g[j];
    return h;
}

// ─── Fourier transform of a measure (characteristic function) ─────────────────
template<int N>
[[nodiscard]] constexpr Cx characteristic_fn(
    const std::array<f64,N>& weights, const std::array<f64,N>& atoms,
    f64 t) noexcept
{
    // φ(t) = E[e^{itX}] = sum_k w_k * e^{it*x_k}
    Cx phi{};
    for(int k=0;k<N;++k) phi=phi+cx_polar(weights[k],t*atoms[k]);
    return phi;
}

} // namespace qs::math
