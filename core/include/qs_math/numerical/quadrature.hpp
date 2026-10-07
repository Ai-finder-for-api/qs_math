#pragma once
// qs_math/numerical/quadrature.hpp — Gauss-Legendre, Gauss-Hermite, adaptive Simpson, Romberg
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

// ─── Gauss-Legendre nodes and weights (N-point, hardcoded for common orders) ──
template<int N> struct GaussLegendre;

template<> struct GaussLegendre<2> {
    static constexpr std::array<f64,2> nodes   = {-0.5773502691896257, 0.5773502691896257};
    static constexpr std::array<f64,2> weights = { 1.0, 1.0};
};
template<> struct GaussLegendre<3> {
    static constexpr std::array<f64,3> nodes   = {-0.7745966692414834,0.0,0.7745966692414834};
    static constexpr std::array<f64,3> weights = {0.5555555555555556,0.8888888888888888,0.5555555555555556};
};
template<> struct GaussLegendre<4> {
    static constexpr std::array<f64,4> nodes   = {-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526};
    static constexpr std::array<f64,4> weights = {0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538};
};
template<> struct GaussLegendre<5> {
    static constexpr std::array<f64,5> nodes   = {-0.9061798459366370,-0.5384693101056831,0.0,0.5384693101056831,0.9061798459366370};
    static constexpr std::array<f64,5> weights = {0.2369268850561891,0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891};
};
template<> struct GaussLegendre<8> {
    static constexpr std::array<f64,8> nodes   = {-0.9602898564975363,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363};
    static constexpr std::array<f64,8> weights = {0.1012285362903763,0.2223810344533745,0.3137066458778873,0.3626837833783620,0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763};
};

// Integrate f on [a,b] using N-point Gauss-Legendre
template<int N, typename F>
[[nodiscard]] constexpr f64 gauss_legendre(F&& f, f64 a, f64 b) noexcept {
    using GL = GaussLegendre<N>;
    f64 half=(b-a)*0.5, mid=(a+b)*0.5, s=0;
    for(int i=0;i<N;++i) s+=GL::weights[i]*f(mid+half*GL::nodes[i]);
    return half*s;
}

// ─── Gauss-Hermite (for integrals with e^{-x^2} weight) ──────────────────────
template<> struct GaussLegendre<6> { // reuse struct for Hermite 5-point
    static constexpr std::array<f64,5> nodes   = {-2.0201828704560856,-0.9585724646138185,0.0,0.9585724646138185,2.0201828704560856};
    static constexpr std::array<f64,5> weights = {0.019953242059045923,0.39361932315224975,0.9453087204829418,0.39361932315224975,0.019953242059045923};
};

// Integrate f(x)*exp(-x^2) dx from -inf to inf using 5-point Gauss-Hermite
template<typename F>
[[nodiscard]] constexpr f64 gauss_hermite5(F&& f) noexcept {
    using GH = GaussLegendre<6>;
    f64 s=0;
    for(int i=0;i<5;++i) s+=GH::weights[i]*f(GH::nodes[i]);
    return s;
}

// ─── Composite Simpson's rule ─────────────────────────────────────────────────
template<typename F>
[[nodiscard]] constexpr f64 simpson(F&& f, f64 a, f64 b, int n=100) noexcept {
    if(n%2!=0) ++n;
    f64 h=(b-a)/n, s=f(a)+f(b);
    for(int i=1;i<n;++i) s+=f(a+i*h)*(i%2==0?2.0:4.0);
    return h/3.0*s;
}

// ─── Adaptive Simpson ─────────────────────────────────────────────────────────
template<typename F>
[[nodiscard]] f64 adaptive_simpson_rec(F&& f, f64 a, f64 b,
                                        f64 tol, f64 whole, int depth) noexcept {
    f64 m=(a+b)*0.5;
    f64 left =simpson(f,a,m,2);
    f64 right=simpson(f,m,b,2);
    if(depth>50||std::abs(left+right-whole)<15*tol) return left+right+(left+right-whole)/15.0;
    return adaptive_simpson_rec(f,a,m,tol*0.5,left,depth+1)
          +adaptive_simpson_rec(f,m,b,tol*0.5,right,depth+1);
}

template<typename F>
[[nodiscard]] f64 adaptive_simpson(F&& f, f64 a, f64 b, f64 tol=1e-8) noexcept {
    f64 whole=simpson(f,a,b,2);
    return adaptive_simpson_rec(f,a,b,tol,whole,0);
}

// ─── Romberg integration ──────────────────────────────────────────────────────
template<typename F>
[[nodiscard]] constexpr f64 romberg(F&& f, f64 a, f64 b,
                                     int max_order=10, f64 tol=1e-12) noexcept {
    std::array<std::array<f64,12>,12> R{};
    f64 h=b-a;
    R[0][0]=0.5*h*(f(a)+f(b));
    for(int i=1;i<max_order;++i) {
        h*=0.5;
        f64 sum=0;
        int n=1<<(i-1);
        for(int k=0;k<n;++k) sum+=f(a+(2*k+1)*h);
        R[i][0]=0.5*R[i-1][0]+h*sum;
        for(int j=1;j<=i;++j) {
            f64 p=1<<(2*j);
            R[i][j]=(p*R[i][j-1]-R[i-1][j-1])/(p-1);
        }
        if(i>1&&std::abs(R[i][i]-R[i-1][i-1])<tol) return R[i][i];
    }
    return R[max_order-1][max_order-1];
}

// ─── Gauss-Kronrod G7K15 (embedded pair for error estimate) ──────────────────
[[nodiscard]] constexpr f64 gauss_kronrod_g7k15(auto&& f, f64 a, f64 b,
                                                  f64* err=nullptr) noexcept {
    // 15-point Kronrod nodes and weights
    constexpr f64 xgk[15]={
        0.9914553711208126,0.9491079123427585,0.8648644233597123,
        0.7415311855993945,0.5860872354676911,0.4058451513773972,
        0.2077849550078985,0.0,
       -0.2077849550078985,-0.4058451513773972,-0.5860872354676911,
       -0.7415311855993945,-0.8648644233597123,-0.9491079123427585,
       -0.9914553711208126
    };
    constexpr f64 wgk[15]={
        0.02293532201052922,0.06309209262997856,0.10479001032225018,
        0.14065325971552592,0.16900472663926790,0.19035057806478541,
        0.20443294007529889,0.20948214108472783,
        0.20443294007529889,0.19035057806478541,0.16900472663926790,
        0.14065325971552592,0.10479001032225018,0.06309209262997856,
        0.02293532201052922
    };
    constexpr f64 wg7[7]={
        0.12948496616886423,0.27970539148927664,0.38183005050511894,
        0.41795918367346939,
        0.38183005050511894,0.27970539148927664,0.12948496616886423
    };
    f64 mid=(a+b)*0.5, half=(b-a)*0.5;
    f64 resk=0, resg=0;
    for(int j=0;j<15;++j) { f64 fv=f(mid+half*xgk[j]); resk+=wgk[j]*fv; }
    for(int j=0;j<7;++j)  resg+=wg7[j]*f(mid+half*xgk[2*j+1]);
    resk*=half; resg*=half;
    if(err) *err=std::abs(resk-resg);
    return resk;
}

// ─── Double integral (iterated Gauss-Legendre) ────────────────────────────────
template<int N, typename F, typename XLo, typename XHi, typename YLo, typename YHi>
[[nodiscard]] constexpr f64 double_integral(F&& f,
    XLo&& xlo, XHi&& xhi, YLo&& ylo, YHi&& yhi) noexcept {
    return gauss_legendre<N>([&](f64 x){
        return gauss_legendre<N>([&](f64 y){ return f(x,y); }, ylo(x), yhi(x));
    }, xlo(0.0), xhi(0.0));
}

// ─── Trapezoidal rule (composite) ─────────────────────────────────────────────
template<typename F>
[[nodiscard]] constexpr f64 trapezoid(F&& f, f64 a, f64 b, int n=1000) noexcept {
    f64 h=(b-a)/n, s=0.5*(f(a)+f(b));
    for(int i=1;i<n;++i) s+=f(a+i*h);
    return h*s;
}

} // namespace qs::math
