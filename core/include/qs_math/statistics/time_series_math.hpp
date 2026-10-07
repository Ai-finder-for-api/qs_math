#pragma once
// qs_math/time_series_math.hpp — ARIMA, seasonal decomp, forecasting, anomaly detection (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <span>
#include <optional>

namespace qs {

// ─── Simple moving averages ───────────────────────────────────────────────────

[[nodiscard]] inline std::vector<f64> sma(std::span<const f64> x, u32 window) {
    std::vector<f64> out(x.size(),0.);
    for(u32 i=window-1;i<u32(x.size());++i){
        f64 s=0.;
        for(u32 j=i-window+1;j<=i;++j) s+=x[j];
        out[i]=s/f64(window);
    }
    return out;
}

[[nodiscard]] inline std::vector<f64> ema(std::span<const f64> x, f64 alpha) {
    std::vector<f64> out(x.size());
    if(x.empty()) return out;
    out[0]=x[0];
    for(u32 i=1;i<u32(x.size());++i) out[i]=alpha*x[i]+(1.-alpha)*out[i-1];
    return out;
}

// Weighted moving average (linear weights)
[[nodiscard]] inline std::vector<f64> wma(std::span<const f64> x, u32 window) {
    std::vector<f64> out(x.size(),0.);
    f64 denom=f64(window*(window+1))/2.;
    for(u32 i=window-1;i<u32(x.size());++i){
        f64 s=0.;
        for(u32 j=0;j<window;++j) s+=x[i-window+1+j]*f64(j+1);
        out[i]=s/denom;
    }
    return out;
}

// ─── Autocorrelation / PACF ───────────────────────────────────────────────────

[[nodiscard]] inline std::vector<f64> autocorrelation(std::span<const f64> x, u32 max_lag) {
    u32 n=u32(x.size());
    f64 mean=0.; for(auto v:x) mean+=v; mean/=f64(n);
    f64 var=0.; for(auto v:x) var+=(v-mean)*(v-mean); var/=f64(n);
    std::vector<f64> acf(max_lag+1,0.);
    for(u32 lag=0;lag<=max_lag;++lag){
        f64 cov=0.;
        for(u32 i=lag;i<n;++i) cov+=(x[i]-mean)*(x[i-lag]-mean);
        acf[lag]=cov/f64(n)/(var+1e-15);
    }
    return acf;
}

// ─── Seasonal decomposition (additive, STL-lite) ──────────────────────────────

struct STLResult {
    std::vector<f64> trend;
    std::vector<f64> seasonal;
    std::vector<f64> residual;
};

[[nodiscard]] inline STLResult seasonal_decompose(
    std::span<const f64> y, u32 period, u32 trend_window=0)
{
    u32 n=u32(y.size());
    if(trend_window==0) trend_window=period%2==0?period+1:period;
    STLResult r;
    // Trend via centered moving average
    r.trend=sma(y,trend_window);
    // Seasonal: average residual by phase
    std::vector<f64> detrended(n);
    for(u32 i=0;i<n;++i) detrended[i]=r.trend[i]>0?y[i]-r.trend[i]:0.;
    r.seasonal.resize(n);
    std::vector<f64> phase_avg(period,0.); std::vector<u32> phase_cnt(period,0);
    u32 half=trend_window/2;
    for(u32 i=half;i+half<n;++i){phase_avg[i%period]+=detrended[i];++phase_cnt[i%period];}
    f64 avg_seasonal=0.;
    for(u32 p=0;p<period;++p){
        if(phase_cnt[p]) phase_avg[p]/=f64(phase_cnt[p]);
        avg_seasonal+=phase_avg[p];
    }
    avg_seasonal/=f64(period);
    for(auto& v:phase_avg) v-=avg_seasonal;
    for(u32 i=0;i<n;++i) r.seasonal[i]=phase_avg[i%period];
    r.residual.resize(n);
    for(u32 i=0;i<n;++i) r.residual[i]=y[i]-r.trend[i]-r.seasonal[i];
    return r;
}

// ─── ARIMA helpers ────────────────────────────────────────────────────────────

// Difference operator: d-th order
[[nodiscard]] inline std::vector<f64> difference(std::span<const f64> x, u32 d=1) {
    std::vector<f64> out(x.begin(),x.end());
    for(u32 i=0;i<d;++i){
        std::vector<f64> tmp(out.size()-1);
        for(u32 j=0;j<u32(tmp.size());++j) tmp[j]=out[j+1]-out[j];
        out=tmp;
    }
    return out;
}

// Inverse difference (integrate)
[[nodiscard]] inline std::vector<f64> undifference(
    std::span<const f64> dx, std::span<const f64> x_orig, u32 d=1)
{
    std::vector<f64> out(dx.begin(),dx.end());
    for(u32 level=0;level<d;++level){
        f64 start=x_orig[d-1-level];
        std::vector<f64> integrated(out.size()+1);
        integrated[0]=start;
        for(u32 i=0;i<u32(out.size());++i) integrated[i+1]=integrated[i]+out[i];
        out=integrated;
    }
    return out;
}

// AR(p) coefficient estimation via Yule-Walker
[[nodiscard]] inline std::vector<f64> ar_yule_walker(std::span<const f64> x, u32 p) {
    auto acf=autocorrelation(x,p);
    // Toeplitz system: R phi = r
    std::vector<std::vector<f64>> R(p,std::vector<f64>(p+1));
    for(u32 i=0;i<p;++i){
        for(u32 j=0;j<p;++j) R[i][j]=acf[std::abs(i32(i)-i32(j))];
        R[i][p]=acf[i+1];
    }
    // Gaussian elimination
    for(u32 col=0;col<p;++col){
        if(std::abs(R[col][col])<1e-14) continue;
        for(u32 row=col+1;row<p;++row){
            f64 f=R[row][col]/R[col][col];
            for(u32 k=col;k<=p;++k) R[row][k]-=f*R[col][k];
        }
    }
    std::vector<f64> phi(p,0.);
    for(i32 i=i32(p)-1;i>=0;--i){
        f64 s=R[i][p];
        for(u32 j=i+1;j<p;++j) s-=R[i][j]*phi[j];
        phi[i]=s/(R[i][i]+1e-15);
    }
    return phi;
}

// AR forecast h steps ahead
[[nodiscard]] inline std::vector<f64> ar_forecast(
    std::span<const f64> x, std::span<const f64> phi, u32 h)
{
    std::vector<f64> history(x.begin(),x.end());
    std::vector<f64> forecast;
    for(u32 step=0;step<h;++step){
        f64 pred=0.;
        u32 p=u32(phi.size());
        for(u32 i=0;i<p;++i)
            pred+=phi[i]*history[history.size()-1-i];
        forecast.push_back(pred);
        history.push_back(pred);
    }
    return forecast;
}

// ─── Exponential smoothing (Holt-Winters) ────────────────────────────────────

struct HoltWinters {
    f64 alpha{0.3};  // level
    f64 beta{0.1};   // trend
    f64 gamma{0.2};  // seasonal
    u32 period{12};

    f64 level{}, trend{};
    std::vector<f64> seasonal;

    void init(std::span<const f64> y) {
        u32 n=u32(y.size());
        if(n<2*period){level=y[0];trend=0.;seasonal.assign(period,0.);return;}
        level=0.;
        for(u32 i=0;i<period;++i) level+=y[i]/f64(period);
        trend=(sma(y,period)[period]-sma(y,period)[0])/f64(period);
        seasonal.resize(period);
        for(u32 i=0;i<period;++i) seasonal[i]=y[i]/level;
    }

    f64 update(f64 y_t, u32 t){
        f64 prev_l=level, prev_b=trend;
        u32 s=t%period;
        level=alpha*(y_t/seasonal[s])+(1.-alpha)*(prev_l+prev_b);
        trend=beta*(level-prev_l)+(1.-beta)*prev_b;
        seasonal[s]=gamma*(y_t/level)+(1.-gamma)*seasonal[s];
        return (level+trend)*seasonal[s];
    }

    [[nodiscard]] f64 forecast(u32 h) const {
        return (level+f64(h)*trend)*seasonal[h%period];
    }
};

// ─── Anomaly detection ────────────────────────────────────────────────────────

// Z-score based
[[nodiscard]] inline std::vector<bool> zscore_anomaly(
    std::span<const f64> x, f64 threshold=3.)
{
    u32 n=u32(x.size());
    f64 mean=0.; for(auto v:x) mean+=v; mean/=f64(n);
    f64 std2=0.; for(auto v:x) std2+=(v-mean)*(v-mean); std2=std::sqrt(std2/f64(n)+1e-15);
    std::vector<bool> anomaly(n,false);
    for(u32 i=0;i<n;++i) if(std::abs(x[i]-mean)/std2>threshold) anomaly[i]=true;
    return anomaly;
}

// CUSUM (cumulative sum) control chart
struct CUSUMState {
    f64 Cp{0.},Cm{0.};
    f64 k{0.5};   // reference value (half shift to detect)
    f64 h{5.0};   // threshold (multiples of sigma)
    f64 mu{0.},sigma{1.};

    [[nodiscard]] bool update(f64 x) {
        f64 z=(x-mu)/sigma;
        Cp=std::max(0.,Cp+z-k);
        Cm=std::max(0.,Cm-z-k);
        return Cp>h||Cm>h;
    }
    void reset(){Cp=0.;Cm=0.;}
};

// IQR-based outlier removal
[[nodiscard]] inline std::vector<f64> remove_outliers_iqr(
    std::span<const f64> x, f64 k=1.5)
{
    std::vector<f64> sorted(x.begin(),x.end());
    std::sort(sorted.begin(),sorted.end());
    u32 n=u32(sorted.size());
    f64 q1=sorted[n/4], q3=sorted[3*n/4];
    f64 iqr=q3-q1, lo=q1-k*iqr, hi=q3+k*iqr;
    std::vector<f64> out;
    for(auto v:x) if(v>=lo&&v<=hi) out.push_back(v);
    return out;
}

// ─── Change point detection ───────────────────────────────────────────────────

// PELT-like binary segmentation (simplified)
[[nodiscard]] inline std::vector<u32> detect_change_points(
    std::span<const f64> x, f64 threshold=3., u32 min_segment=5)
{
    u32 n=u32(x.size());
    std::vector<u32> cps;
    // Sliding window: compare mean of two halves
    for(u32 i=min_segment;i+min_segment<n;++i){
        f64 m1=0.,m2=0.;
        for(u32 j=0;j<i;++j) m1+=x[j]; m1/=f64(i);
        for(u32 j=i;j<n;++j) m2+=x[j]; m2/=f64(n-i);
        f64 s=0.;
        for(auto v:x){f64 d=v-m1;s+=d*d;}
        f64 sigma=std::sqrt(s/f64(n)+1e-15);
        if(std::abs(m2-m1)/sigma>threshold) cps.push_back(i);
    }
    // Keep only local maxima (deduplicate)
    std::vector<u32> out;
    for(u32 i=0;i<u32(cps.size());++i)
        if(i==0||cps[i]>cps[i-1]+min_segment) out.push_back(cps[i]);
    return out;
}

// ─── Cross-correlation ────────────────────────────────────────────────────────

[[nodiscard]] inline f64 cross_correlation(
    std::span<const f64> x, std::span<const f64> y, i32 lag=0)
{
    u32 n=u32(std::min(x.size(),y.size()));
    f64 mx=0.,my=0.;
    for(u32 i=0;i<n;++i){mx+=x[i];my+=y[i];}
    mx/=f64(n); my/=f64(n);
    f64 sx=0.,sy=0.,sxy=0.;
    for(u32 i=std::max(0,lag);i<u32(std::min(i32(n),i32(n)-lag));++i){
        sx+=(x[i]-mx)*(x[i]-mx);
        sy+=(y[i]-my)*(y[i]-my);
        sxy+=(x[i]-mx)*(y[i-lag]-my);
    }
    f64 denom=std::sqrt(sx*sy);
    return denom>0?sxy/denom:0.;
}

} // namespace qs
