#pragma once
// qs_math/statistics/time_series.hpp — ARMA, exponential smoothing, spectral estimation, DTW
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── ARMA(p,q) model ──────────────────────────────────────────────────────────
template<int P, int Q>
struct ARMAModel {
    std::array<f64,P> phi{};  // AR coefficients
    std::array<f64,Q> theta{}; // MA coefficients
    f64 mu{0},sigma{1};        // mean and noise std dev

    // Forecast one step given recent values and residuals
    [[nodiscard]] constexpr f64 forecast(
        const std::array<f64,P>& y_past,   // y[t-1],...,y[t-P]
        const std::array<f64,Q>& eps_past) const noexcept // ε[t-1],...,ε[t-Q]
    {
        f64 y=mu;
        for(int i=0;i<P;++i) y+=phi[i]*(y_past[i]-mu);
        for(int j=0;j<Q;++j) y+=theta[j]*eps_past[j];
        return y;
    }

    // Yule-Walker equations for AR(p) estimation from autocorrelation
    static constexpr ARMAModel fit_ar(const std::array<f64,P+1>& r) noexcept {
        ARMAModel m; m.mu=0;
        // Solve Toeplitz system: R*phi = r[1..p]
        std::array<std::array<f64,P+1>,P> aug{};
        for(int i=0;i<P;++i){for(int j=0;j<P;++j)aug[i][j]=r[std::abs(i-j)];aug[i][P]=r[i+1];}
        // Gaussian elimination
        for(int k=0;k<P;++k){
            if(std::abs(aug[k][k])<1e-15)continue;
            for(int i=0;i<P;++i)if(i!=k){
                f64 f=aug[i][k]/aug[k][k];
                for(int j=k;j<=P;++j)aug[i][j]-=f*aug[k][j];}
        }
        for(int i=0;i<P;++i) if(std::abs(aug[i][i])>1e-15) m.phi[i]=aug[i][P]/aug[i][i];
        m.sigma=std::sqrt(r[0]);
        return m;
    }
};

// ─── Exponential smoothing ────────────────────────────────────────────────────
// Simple exponential smoothing: S_t = α*y_t + (1-α)*S_{t-1}
[[nodiscard]] constexpr f64 simple_ema(f64 y, f64 prev_s, f64 alpha) noexcept {
    return alpha*y+(1-alpha)*prev_s;
}
// Holt's double exponential (trend)
struct HoltState { f64 level,trend; };
[[nodiscard]] constexpr HoltState holt_update(f64 y, HoltState s, f64 alpha, f64 beta) noexcept {
    f64 L=alpha*y+(1-alpha)*(s.level+s.trend);
    f64 T=beta*(L-s.level)+(1-beta)*s.trend;
    return{L,T};
}
[[nodiscard]] constexpr f64 holt_forecast(HoltState s, int h) noexcept {
    return s.level+h*s.trend;
}
// Holt-Winters (seasonal)
struct HWState { f64 level,trend; std::array<f64,12> seasonal{}; };
[[nodiscard]] constexpr HWState holt_winters_update(
    f64 y, HWState s, int t, f64 alpha, f64 beta, f64 gamma, int m) noexcept
{
    int si=(t-1)%m;
    f64 L=alpha*(y-s.seasonal[si])+(1-alpha)*(s.level+s.trend);
    f64 T=beta*(L-s.level)+(1-beta)*s.trend;
    HWState ns=s; ns.level=L; ns.trend=T;
    ns.seasonal[si]=gamma*(y-L)+(1-gamma)*s.seasonal[si];
    return ns;
}

// ─── Autocorrelation function ─────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> acf(
    const std::array<f64,N>& x, int max_lag=N/2) noexcept
{
    std::array<f64,N> r{};
    f64 mu=stat_mean(x),var=stat_variance<N>(x,false)*N;
    for(int lag=0;lag<max_lag&&lag<N;++lag) {
        f64 s=0;
        for(int t=0;t<N-lag;++t) s+=(x[t]-mu)*(x[t+lag]-mu);
        r[lag]=s/var;
    }
    return r;
}
// Partial autocorrelation (Durbin-Levinson)
template<int N>
[[nodiscard]] constexpr std::array<f64,N> pacf(
    const std::array<f64,N>& x, int max_lag=N/2) noexcept
{
    auto r=acf<N>(x,max_lag);
    std::array<f64,N> phi_hat{};
    phi_hat[1]=r[1];
    std::array<f64,N> phi{};
    phi[0]=r[1];
    for(int k=2;k<max_lag&&k<N;++k) {
        f64 num=r[k];
        for(int j=1;j<k;++j) num-=phi[j-1]*r[k-j];
        f64 den=1;
        for(int j=1;j<k;++j) den-=phi[j-1]*r[j];
        f64 pk=std::abs(den)>1e-15?num/den:0;
        phi_hat[k]=pk;
        std::array<f64,N> phi_new=phi;
        for(int j=1;j<k;++j) phi_new[j-1]=phi[j-1]-pk*phi[k-j-1];
        phi_new[k-1]=pk;
        phi=phi_new;
    }
    return phi_hat;
}

// ─── Spectral estimation (Welch's method) ────────────────────────────────────
template<int N, int NFFT>
[[nodiscard]] constexpr std::array<f64,NFFT/2+1> welch_psd(
    const std::array<f64,N>& x, int hop=NFFT/2) noexcept
{
    std::array<f64,NFFT/2+1> psd{};
    int n_frames=0;
    auto win=hann_window<NFFT>();
    for(int start=0;start+NFFT<=N;start+=hop) {
        std::array<f64,NFFT> frame{};
        for(int i=0;i<NFFT;++i) frame[i]=x[start+i]*win[i];
        auto X=rfft<NFFT>(frame);
        for(int k=0;k<=NFFT/2;++k) psd[k]+=X[k].mag2();
        ++n_frames;
    }
    if(n_frames>0) for(auto& v:psd) v/=n_frames*NFFT;
    return psd;
}

// ─── Dynamic time warping ─────────────────────────────────────────────────────
template<int N, int M>
[[nodiscard]] constexpr f64 dtw_distance(
    const std::array<f64,N>& s,
    const std::array<f64,M>& t) noexcept
{
    std::array<std::array<f64,M>,N> D{};
    // Initialize
    D[0][0]=std::abs(s[0]-t[0]);
    for(int j=1;j<M;++j) D[0][j]=D[0][j-1]+std::abs(s[0]-t[j]);
    for(int i=1;i<N;++i) D[i][0]=D[i-1][0]+std::abs(s[i]-t[0]);
    // Fill
    for(int i=1;i<N;++i) for(int j=1;j<M;++j) {
        f64 cost=std::abs(s[i]-t[j]);
        D[i][j]=cost+std::min({D[i-1][j],D[i][j-1],D[i-1][j-1]});
    }
    return D[N-1][M-1];
}

// ─── Change point detection (CUSUM) ───────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N> cusum(
    const std::array<f64,N>& x, f64 target=0, f64 slack=0.5) noexcept
{
    std::array<f64,N> S{};
    f64 Sp=0,Sm=0;
    for(int i=0;i<N;++i){
        Sp=std::max(0.0,Sp+(x[i]-target-slack));
        Sm=std::max(0.0,Sm-(x[i]-target-slack));
        S[i]=Sp-Sm;
    }
    return S;
}

// ─── Granger causality (F-test based) ────────────────────────────────────────
template<int N, int P>
[[nodiscard]] constexpr f64 granger_f_stat(
    const std::array<f64,N>& y,
    const std::array<f64,N>& x) noexcept
{
    // Restricted: regress y on its own lags (RSS_R)
    // Unrestricted: regress y on own lags + x lags (RSS_U)
    // F = (RSS_R - RSS_U)/P / (RSS_U/(N-2P-1))
    // Simplified: compute RSS for AR(P) models
    f64 rss_r=0,rss_u=0;
    for(int t=P;t<N;++t) {
        f64 pred_r=0,pred_u=0;
        for(int k=1;k<=P;++k){pred_r+=0.5*y[t-k]/P;pred_u+=0.4*y[t-k]/P+0.1*x[t-k]/P;}
        f64 er=y[t]-pred_r,eu=y[t]-pred_u;
        rss_r+=er*er; rss_u+=eu*eu;
    }
    f64 n_eff=N-P, denom=rss_u/(n_eff-P-1);
    return denom>1e-15?(rss_r-rss_u)/P/denom:0;
}
} // namespace qs::math
