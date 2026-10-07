#pragma once
// qs_math/statistics/descriptive.hpp — Descriptive statistics: mean, variance,
//   skewness, kurtosis, quantiles, histogram, covariance, correlation,
//   robust statistics, kernel density estimation (C++20)
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <span>
#include <array>

namespace qs::math::statistics {

// ─── Basic moments ────────────────────────────────────────────────────────
[[nodiscard]] inline f64 mean(std::span<const f64> x) noexcept {
    if(x.empty()) return 0.0;
    f64 s=0; for(auto v:x) s+=v; return s/f64(x.size());
}

[[nodiscard]] inline f64 variance(std::span<const f64> x, bool population=false) noexcept {
    if(x.size()<2) return 0.0;
    f64 m=mean(x), s=0;
    for(auto v:x){f64 d=v-m;s+=d*d;}
    return s/(population?f64(x.size()):f64(x.size()-1));
}

[[nodiscard]] inline f64 std_dev(std::span<const f64> x, bool population=false) noexcept {
    return std::sqrt(variance(x,population));
}

[[nodiscard]] inline f64 skewness(std::span<const f64> x) noexcept {
    int n=int(x.size()); if(n<3) return 0.0;
    f64 m=mean(x), s=std_dev(x,true);
    if(s<1e-300) return 0.0;
    f64 sum=0; for(auto v:x){f64 z=(v-m)/s;sum+=z*z*z;}
    return sum/n;
}

[[nodiscard]] inline f64 kurtosis(std::span<const f64> x, bool excess=true) noexcept {
    int n=int(x.size()); if(n<4) return 0.0;
    f64 m=mean(x), s=std_dev(x,true);
    if(s<1e-300) return 0.0;
    f64 sum=0; for(auto v:x){f64 z=(v-m)/s;sum+=z*z*z*z;}
    f64 k=sum/n;
    return excess?k-3.0:k;
}

// ─── Welford's online variance (numerically stable) ───────────────────────
struct WelfordAccum {
    f64 mean=0,M2=0; int n=0;
    void add(f64 x) noexcept {
        ++n; f64 delta=x-mean; mean+=delta/n; M2+=delta*(x-mean);
    }
    [[nodiscard]] f64 variance() const noexcept { return n>1?M2/(n-1):0; }
    [[nodiscard]] f64 std_dev()  const noexcept { return std::sqrt(variance()); }
    [[nodiscard]] int  count()   const noexcept { return n; }
};

// ─── Quantiles ────────────────────────────────────────────────────────────
[[nodiscard]] inline f64 quantile(std::vector<f64> x, f64 p) noexcept {
    if(x.empty()) return 0.0;
    std::sort(x.begin(),x.end());
    f64 idx=p*(x.size()-1);
    int lo=int(idx); f64 frac=idx-lo;
    if(lo>=int(x.size())-1) return x.back();
    return x[lo]+frac*(x[lo+1]-x[lo]);
}

[[nodiscard]] inline f64 median(std::vector<f64> x) noexcept { return quantile(std::move(x),0.5); }
[[nodiscard]] inline f64 iqr   (std::vector<f64> x) noexcept {
    auto s=x; // sorted once inside quantile - ok here
    return quantile(x,0.75)-quantile(std::move(x),0.25);
}

// Five-number summary
struct FiveNumber { f64 min,q1,median,q3,max; };
[[nodiscard]] inline FiveNumber five_number_summary(std::vector<f64> x) noexcept {
    if(x.empty()) return {};
    std::sort(x.begin(),x.end());
    return {x.front(), quantile(x,0.25), quantile(x,0.5), quantile(x,0.75), x.back()};
}

// ─── Histogram ───────────────────────────────────────────────────────────
struct Histogram {
    std::vector<f64> edges; // n_bins+1 edges
    std::vector<u64> counts;
    int n_bins;

    [[nodiscard]] int bin_of(f64 v) const noexcept {
        if(v<edges.front()||v>edges.back()) return -1;
        int lo=0,hi=n_bins-1;
        while(lo<hi){int m=(lo+hi)/2;v>=edges[m+1]?lo=m+1:hi=m;}
        return lo;
    }
    [[nodiscard]] f64 density(int bin) const noexcept {
        if(bin<0||bin>=n_bins) return 0.0;
        f64 total=0; for(auto c:counts) total+=c;
        return counts[bin]/(total*(edges[bin+1]-edges[bin]));
    }
};

[[nodiscard]] inline Histogram histogram(
    std::span<const f64> x, int n_bins, f64 lo=0, f64 hi=0) noexcept
{
    if(x.empty()) return {};
    Histogram h; h.n_bins=n_bins;
    if(lo==hi){
        lo=*std::min_element(x.begin(),x.end());
        hi=*std::max_element(x.begin(),x.end());
        if(lo==hi){lo-=1;hi+=1;}
    }
    h.edges.resize(n_bins+1);
    for(int i=0;i<=n_bins;++i) h.edges[i]=lo+f64(i)*(hi-lo)/n_bins;
    h.counts.assign(n_bins,0);
    for(auto v:x){int b=h.bin_of(v);if(b>=0) ++h.counts[b];}
    return h;
}

// Freedman-Diaconis rule for optimal bin width
[[nodiscard]] inline int fd_bins(std::vector<f64> x) noexcept {
    f64 iqr_val=iqr(x);
    if(iqr_val<1e-300) return 10;
    f64 bw=2.0*iqr_val*std::pow(f64(x.size()),-1.0/3.0);
    f64 range=*std::max_element(x.begin(),x.end())-*std::min_element(x.begin(),x.end());
    return std::max(1,int(std::ceil(range/bw)));
}

// ─── Covariance & correlation ─────────────────────────────────────────────
[[nodiscard]] inline f64 covariance(std::span<const f64> x, std::span<const f64> y) noexcept {
    int n=std::min(x.size(),y.size()); if(n<2) return 0;
    f64 mx=mean(x.subspan(0,n)), my=mean(y.subspan(0,n)), s=0;
    for(int i=0;i<n;++i) s+=(x[i]-mx)*(y[i]-my);
    return s/(n-1);
}

[[nodiscard]] inline f64 pearson_r(std::span<const f64> x, std::span<const f64> y) noexcept {
    f64 cov=covariance(x,y);
    int n=std::min(x.size(),y.size());
    f64 sx=std_dev(x.subspan(0,n)), sy=std_dev(y.subspan(0,n));
    return (sx*sy>1e-300)?cov/(sx*sy):0.0;
}

// Spearman rank correlation
[[nodiscard]] inline f64 spearman_r(std::vector<f64> x, std::vector<f64> y) noexcept {
    int n=std::min(x.size(),y.size());
    // Compute ranks
    auto rank_vec=[](std::vector<f64> v)->std::vector<f64>{
        int m=int(v.size());
        std::vector<int> idx(m); std::iota(idx.begin(),idx.end(),0);
        std::sort(idx.begin(),idx.end(),[&](int a,int b){return v[a]<v[b];});
        std::vector<f64> r(m);
        for(int i=0;i<m;) {
            int j=i;
            while(j<m&&v[idx[j]]==v[idx[i]]) ++j;
            f64 avg_rank=0.5*(i+j-1);
            for(int k=i;k<j;++k) r[idx[k]]=avg_rank;
            i=j;
        }
        return r;
    };
    auto rx=rank_vec(x); auto ry=rank_vec(y);
    return pearson_r(std::span(rx.data(),n),std::span(ry.data(),n));
}

// Covariance matrix (n variables, m observations each)
[[nodiscard]] inline std::vector<f64> cov_matrix(
    const std::vector<std::vector<f64>>& X) noexcept
{
    int p=int(X.size()); if(p==0) return {};
    int n=int(X[0].size());
    std::vector<f64> mu(p), C(p*p,0.0);
    for(int i=0;i<p;++i) mu[i]=mean(X[i]);
    for(int i=0;i<p;++i) for(int j=i;j<p;++j){
        f64 s=0;
        for(int k=0;k<n;++k) s+=(X[i][k]-mu[i])*(X[j][k]-mu[j]);
        C[i*p+j]=C[j*p+i]=s/(n-1);
    }
    return C;
}

// ─── Robust statistics ────────────────────────────────────────────────────
// Median Absolute Deviation (MAD)
[[nodiscard]] inline f64 mad(std::vector<f64> x) noexcept {
    f64 m=median(x);
    std::vector<f64> dev(x.size());
    for(int i=0;i<int(x.size());++i) dev[i]=std::abs(x[i]-m);
    return median(dev);
}

// Winsorized mean (trim p fraction from each tail)
[[nodiscard]] inline f64 winsorized_mean(std::vector<f64> x, f64 p=0.1) noexcept {
    std::sort(x.begin(),x.end());
    int n=int(x.size()), k=int(n*p);
    for(int i=0;i<k;++i) x[i]=x[k];
    for(int i=n-k;i<n;++i) x[i]=x[n-k-1];
    return mean(x);
}

// Trimmed mean
[[nodiscard]] inline f64 trimmed_mean(std::vector<f64> x, f64 p=0.1) noexcept {
    std::sort(x.begin(),x.end());
    int n=int(x.size()), k=int(n*p);
    f64 s=0; for(int i=k;i<n-k;++i) s+=x[i];
    return s/f64(n-2*k);
}

// ─── Kernel density estimation ────────────────────────────────────────────
enum class KernelType { Gaussian, Epanechnikov, Triangular, Uniform };

[[nodiscard]] inline f64 kernel_eval(KernelType kt, f64 u) noexcept {
    switch(kt){
        case KernelType::Epanechnikov:
            return std::abs(u)<=1.0?0.75*(1.0-u*u):0.0;
        case KernelType::Triangular:
            return std::abs(u)<=1.0?(1.0-std::abs(u)):0.0;
        case KernelType::Uniform:
            return std::abs(u)<=1.0?0.5:0.0;
        default: // Gaussian
            return std::exp(-0.5*u*u)*0.3989422803; // 1/sqrt(2pi)
    }
}

// Silverman's rule of thumb bandwidth
[[nodiscard]] inline f64 silverman_bw(std::span<const f64> x) noexcept {
    int n=int(x.size()); if(n<2) return 1.0;
    f64 s=std_dev(x);
    std::vector<f64> tmp(x.begin(),x.end());
    f64 iq=iqr(tmp);
    f64 A=std::min(s,iq/1.34);
    return 1.06*A*std::pow(f64(n),-0.2);
}

[[nodiscard]] inline f64 kde(
    std::span<const f64> data, f64 x,
    f64 h=-1.0, KernelType kt=KernelType::Gaussian) noexcept
{
    if(h<=0) h=silverman_bw(data);
    f64 sum=0;
    for(auto xi:data) sum+=kernel_eval(kt,(x-xi)/h);
    return sum/(h*data.size());
}

[[nodiscard]] inline std::vector<f64> kde_grid(
    std::span<const f64> data, int n_pts=200,
    f64 h=-1.0, KernelType kt=KernelType::Gaussian) noexcept
{
    if(data.empty()) return {};
    f64 lo=*std::min_element(data.begin(),data.end());
    f64 hi=*std::max_element(data.begin(),data.end());
    if(h<=0) h=silverman_bw(data);
    lo-=3*h; hi+=3*h;
    std::vector<f64> grid(n_pts);
    for(int i=0;i<n_pts;++i){
        f64 x=lo+f64(i)*(hi-lo)/(n_pts-1);
        grid[i]=kde(data,x,h,kt);
    }
    return grid;
}

// ─── Running statistics ───────────────────────────────────────────────────
struct RunningStats {
    f64 mn=1e300,mx=-1e300,sum=0,sum2=0; int n=0;
    void add(f64 v) noexcept {
        mn=std::min(mn,v); mx=std::max(mx,v);
        sum+=v; sum2+=v*v; ++n;
    }
    [[nodiscard]] f64 mean()     const noexcept { return n?sum/n:0; }
    [[nodiscard]] f64 variance() const noexcept { return n>1?(sum2-sum*sum/n)/(n-1):0; }
    [[nodiscard]] f64 std_dev()  const noexcept { return std::sqrt(variance()); }
    [[nodiscard]] f64 range()    const noexcept { return mx-mn; }
    void reset() noexcept { *this={}; }
};

// ─── Chi-squared goodness of fit ─────────────────────────────────────────
[[nodiscard]] inline f64 chi_squared_stat(
    const std::vector<f64>& observed,
    const std::vector<f64>& expected) noexcept
{
    f64 s=0;
    for(int i=0;i<int(observed.size());++i)
        if(expected[i]>0) s+=(observed[i]-expected[i])*(observed[i]-expected[i])/expected[i];
    return s;
}

// Incomplete gamma (chi-squared p-value approximation, regularized)
[[nodiscard]] inline f64 chi_squared_pvalue(f64 chi2, int df) noexcept {
    // Simple normal approximation for large df
    f64 z=(std::pow(chi2/df,1.0/3.0)-1.0+2.0/(9.0*df))/std::sqrt(2.0/(9.0*df));
    return 0.5*std::erfc(z/std::sqrt(2.0));
}

// ─── Z-score normalization ────────────────────────────────────────────────
[[nodiscard]] inline std::vector<f64> zscore(std::vector<f64> x) noexcept {
    f64 m=mean(x), s=std_dev(x);
    if(s<1e-300) return std::vector<f64>(x.size(),0.0);
    for(auto& v:x) v=(v-m)/s;
    return x;
}

// Min-max normalization
[[nodiscard]] inline std::vector<f64> minmax_normalize(std::vector<f64> x) noexcept {
    if(x.empty()) return {};
    f64 lo=*std::min_element(x.begin(),x.end());
    f64 hi=*std::max_element(x.begin(),x.end());
    f64 range=hi-lo;
    if(range<1e-300) return std::vector<f64>(x.size(),0.0);
    for(auto& v:x) v=(v-lo)/range;
    return x;
}

} // namespace qs::math::statistics
