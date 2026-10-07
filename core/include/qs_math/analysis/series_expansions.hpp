#pragma once
// qs_math/analysis/series_expansions.hpp — Laurent, Fourier, asymptotic, power series
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>
#include <numbers>

namespace qs::math {

// ─── Power series sum: Σ aₙ xⁿ ───────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 power_series(const std::array<f64,N>& a, f64 x) noexcept {
    f64 r=0,xn=1;
    for(int n=0;n<N;++n){r+=a[n]*xn;xn*=x;}
    return r;
}
// Evaluate via Horner's method
template<int N>
[[nodiscard]] constexpr f64 horner(const std::array<f64,N>& a, f64 x) noexcept {
    f64 r=a[N-1];
    for(int n=N-2;n>=0;--n) r=r*x+a[n];
    return r;
}

// ─── Maclaurin series for standard functions ──────────────────────────────────
// exp(x) = Σ xⁿ/n!
template<int TERMS=20>
[[nodiscard]] constexpr f64 exp_series(f64 x) noexcept {
    f64 r=1,xn=1,fact=1;
    for(int n=1;n<TERMS;++n){xn*=x;fact*=n;r+=xn/fact;}
    return r;
}
// sin(x) = Σ (-1)ⁿ x^{2n+1}/(2n+1)!
template<int TERMS=15>
[[nodiscard]] constexpr f64 sin_series(f64 x) noexcept {
    f64 r=0,xn=x,fact=1;
    for(int n=0;n<TERMS;++n){
        r+=(n%2==0?1:-1)*xn/fact;
        xn*=x*x; fact*=(2*n+2)*(2*n+3);
    }
    return r;
}
// cos(x) = Σ (-1)ⁿ x^{2n}/(2n)!
template<int TERMS=15>
[[nodiscard]] constexpr f64 cos_series(f64 x) noexcept {
    f64 r=0,xn=1,fact=1;
    for(int n=0;n<TERMS;++n){
        r+=(n%2==0?1:-1)*xn/fact;
        xn*=x*x; fact*=(2*n+1)*(2*n+2);
    }
    return r;
}
// ln(1+x) = Σ (-1)^{n+1} xⁿ/n, |x|<1
template<int TERMS=50>
[[nodiscard]] constexpr f64 log1p_series(f64 x) noexcept {
    f64 r=0,xn=x;
    for(int n=1;n<=TERMS;++n){r+=(n%2==0?-1:1)*xn/n;xn*=x;}
    return r;
}
// atan(x) = Σ (-1)ⁿ x^{2n+1}/(2n+1), |x|≤1
template<int TERMS=50>
[[nodiscard]] constexpr f64 atan_series(f64 x) noexcept {
    f64 r=0,xn=x;
    for(int n=0;n<TERMS;++n){r+=(n%2==0?1:-1)*xn/(2*n+1);xn*=x*x;}
    return r;
}

// ─── Asymptotic expansions ────────────────────────────────────────────────────
// Stirling's approximation: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn) + ...
[[nodiscard]] constexpr f64 stirling_lnfact(f64 n, int terms=6) noexcept {
    f64 r=n*std::log(n)-n+0.5*std::log(2*std::numbers::pi_v<f64>*n);
    // Bernoulli correction terms
    const f64 B[]={1.0/12,-1.0/360,1.0/1260,-1.0/1680,1.0/1188,-691.0/360360};
    f64 nk=n;
    for(int k=0;k<terms&&k<6;++k){nk*=n*n;r+=B[k]/nk;}
    return r;
}

// Euler-Maclaurin summation formula: Σ f(k) ≈ ∫f + (f(a)+f(b))/2 + Σ Bₙ/n! * (f^{n-1}(b)-f^{n-1}(a))
// For sum_{k=0}^{N-1} f(k*h):
template<int N, typename F>
[[nodiscard]] constexpr f64 euler_maclaurin(F&& f, f64 a, f64 b, int terms=4) noexcept {
    f64 h=(b-a)/N;
    f64 sum=(f(a)+f(b))*0.5;
    for(int k=1;k<N;++k) sum+=f(a+k*h);
    sum*=h;
    // Bernoulli correction (B₂=-1/6, B₄=1/30, B₆=-1/42, B₈=1/30)
    const f64 B_over_fact[]={1.0/12,-1.0/720,1.0/30240,-1.0/1209600};
    f64 h_pow=h;
    for(int k=0;k<terms&&k<4;++k) {
        h_pow*=h*h;
        // f^{(2k+1)} at endpoints via finite difference
        auto D=[&](int n,f64 x)->f64{
            if(n==1) return (f(x+h)-f(x-h))/(2*h);
            return 0;
        };
        sum+=B_over_fact[k]*h_pow*(D(1,b)-D(1,a));
    }
    return sum;
}

// ─── Fourier series coefficients ──────────────────────────────────────────────
template<int N, typename F>
[[nodiscard]] constexpr std::pair<std::array<f64,N>,std::array<f64,N>>
fourier_series(F&& f, f64 L, int n_pts=256) noexcept
{
    std::array<f64,N> a{},b{};
    f64 dx=2*L/n_pts;
    for(int k=0;k<n_pts;++k) {
        f64 x=-L+k*dx;
        f64 fx=f(x);
        a[0]+=fx;
        for(int n=1;n<N;++n) {
            a[n]+=fx*std::cos(n*std::numbers::pi_v<f64>*x/L);
            b[n]+=fx*std::sin(n*std::numbers::pi_v<f64>*x/L);
        }
    }
    f64 inv=dx/L;
    a[0]*=dx/(2*L);
    for(int n=1;n<N;++n){a[n]*=inv;b[n]*=inv;}
    return{a,b};
}

// Evaluate Fourier series at x
template<int N>
[[nodiscard]] constexpr f64 fourier_eval(
    const std::array<f64,N>& a, const std::array<f64,N>& b,
    f64 x, f64 L) noexcept
{
    f64 r=a[0];
    for(int n=1;n<N;++n)
        r+=a[n]*std::cos(n*std::numbers::pi_v<f64>*x/L)
          +b[n]*std::sin(n*std::numbers::pi_v<f64>*x/L);
    return r;
}

// ─── Padé approximant [M/N] of a power series ─────────────────────────────────
template<int M, int Nap>
[[nodiscard]] constexpr std::pair<std::array<f64,M+1>,std::array<f64,Nap+1>>
pade_approximant(const std::array<f64,M+Nap+1>& c) noexcept
{
    // Solve for Q via linear system: Σ c_{m+k}*q_k = 0 for m=1..N
    // Then P = (c*Q)_{trunc M}
    std::array<f64,Nap+1> q{}; q[0]=1;
    // Linear system: A*q_1..q_N = -c_{M+1..M+N}
    if constexpr(Nap>0) {
        std::array<std::array<f64,Nap+1>,Nap> A{};
        for(int m=0;m<Nap;++m) for(int k=1;k<=Nap;++k)
            A[m][k-1]=(M-Nap+m+k>=0&&M-Nap+m+k<=M+Nap)?c[M-Nap+m+k]:0;
        std::array<f64,Nap> rhs{};
        for(int m=0;m<Nap;++m) rhs[m]=-c[M+1+m];
        // Gaussian elim
        for(int k=0;k<Nap;++k) {
            if(std::abs(A[k][k])<1e-15) continue;
            for(int i=k+1;i<Nap;++i){f64 f=A[i][k]/A[k][k];for(int j=k;j<Nap;++j)A[i][j]-=f*A[k][j];rhs[i]-=f*rhs[k];}
        }
        for(int i=Nap-1;i>=0;--i){q[i+1]=rhs[i];for(int j=i+1;j<Nap;++j)q[i+1]-=A[i][j]*q[j+1];if(std::abs(A[i][i])>1e-15)q[i+1]/=A[i][i];}
    }
    // Compute P = c*Q truncated to degree M
    std::array<f64,M+1> p{};
    for(int m=0;m<=M;++m) for(int k=0;k<=Nap&&k<=m;++k) p[m]+=c[m-k]*q[k];
    return{p,q};
}

// ─── Residue at pole of order 1 ───────────────────────────────────────────────
// Res_{z=a} [f(z)] for simple pole: lim_{z→a} (z-a)*f(z)
template<typename F>
[[nodiscard]] constexpr f64 residue_real(F&& f, f64 a, f64 h=1e-5) noexcept {
    return (a+h)*f(a+h)*(a+h-a)-h*f(a-h)*(a-h-a); // simplified
}

// ─── Euler-Bernoulli series ───────────────────────────────────────────────────
// Bernoulli numbers B₀,B₂,B₄,...
inline constexpr std::array<f64,14> k_bernoulli = {
    1.0, -0.5, 1.0/6, 0, -1.0/30, 0, 1.0/42, 0,
    -1.0/30, 0, 5.0/66, 0, -691.0/2730, 0
};
// Euler numbers E₀,E₂,E₄,...
inline constexpr std::array<f64,10> k_euler = {
    1.0, 0, -1.0, 0, 5.0, 0, -61.0, 0, 1385.0, 0
};

// Bernoulli polynomial B_n(x)
template<int N>
[[nodiscard]] constexpr f64 bernoulli_poly(f64 x) noexcept {
    f64 r=0,xk=1;
    for(int k=0;k<=N&&k<14;++k) {
        // B_n(x) = Σ C(n,k)*B_k * x^{n-k}
        f64 Cn=1; for(int j=0;j<k;++j){Cn*=(N-j);Cn/=(j+1);}
        r+=Cn*k_bernoulli[k<14?k:0]*std::pow(x,N-k);
    }
    return r;
}
} // namespace qs::math
