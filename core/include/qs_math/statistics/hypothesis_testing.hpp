#pragma once
// qs_math/statistics/hypothesis_testing.hpp — t-test, chi-square, ANOVA, bootstrap, permutation
#include "../arithmetic/scalar_types.hpp"
#include "distributions.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Descriptive statistics ───────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 stat_mean(const std::array<f64,N>& x) noexcept {
    f64 s=0; for(auto v:x)s+=v; return s/N;
}
template<int N>
[[nodiscard]] constexpr f64 stat_variance(const std::array<f64,N>& x, bool sample=true) noexcept {
    f64 mu=stat_mean(x),s=0;
    for(auto v:x)s+=(v-mu)*(v-mu);
    return s/(sample?N-1:N);
}
template<int N>
[[nodiscard]] constexpr f64 stat_std(const std::array<f64,N>& x, bool sample=true) noexcept {
    return std::sqrt(stat_variance<N>(x,sample));
}
template<int N>
[[nodiscard]] constexpr f64 stat_median(std::array<f64,N> x) noexcept {
    // Insertion sort
    for(int i=1;i<N;++i)for(int j=i;j>0&&x[j]<x[j-1];--j)std::swap(x[j],x[j-1]);
    return N%2==0?(x[N/2-1]+x[N/2])*0.5:x[N/2];
}
template<int N>
[[nodiscard]] constexpr f64 stat_skewness(const std::array<f64,N>& x) noexcept {
    f64 mu=stat_mean(x),s=stat_std<N>(x);
    if(s<1e-15)return 0;
    f64 sum=0; for(auto v:x){f64 z=(v-mu)/s;sum+=z*z*z;}
    return sum*N/((N-1.0)*(N-2.0));
}
template<int N>
[[nodiscard]] constexpr f64 stat_kurtosis(const std::array<f64,N>& x) noexcept {
    f64 mu=stat_mean(x),s2=stat_variance<N>(x,false);
    if(s2<1e-15)return 0;
    f64 sum=0; for(auto v:x){f64 d=v-mu;sum+=d*d*d*d;}
    return sum/(N*s2*s2)-3;
}

// ─── One-sample t-test ────────────────────────────────────────────────────────
template<int N>
struct TTestResult { f64 t_stat,p_value,df; bool reject; };

template<int N>
[[nodiscard]] constexpr TTestResult<N> one_sample_t_test(
    const std::array<f64,N>& x, f64 mu0=0, f64 alpha=0.05) noexcept
{
    f64 xbar=stat_mean(x), s=stat_std<N>(x);
    f64 se=s/std::sqrt(f64(N));
    f64 t=(se>1e-15)?(xbar-mu0)/se:0;
    f64 df=N-1;
    // Two-tailed p-value approximation
    f64 p=2*(1-t_cdf(std::abs(t),df));
    return{t,p,df,p<alpha};
}

// Two-sample t-test (Welch's)
template<int N1, int N2>
[[nodiscard]] constexpr TTestResult<N1> two_sample_t_test(
    const std::array<f64,N1>& x, const std::array<f64,N2>& y,
    f64 alpha=0.05) noexcept
{
    f64 x1=stat_mean(x),x2=stat_mean(y);
    f64 s1=stat_variance<N1>(x),s2=stat_variance<N2>(y);
    f64 se=std::sqrt(s1/N1+s2/N2);
    f64 t=(se>1e-15)?(x1-x2)/se:0;
    // Welch-Satterthwaite df
    f64 num=(s1/N1+s2/N2)*(s1/N1+s2/N2);
    f64 den=(s1/N1)*(s1/N1)/(N1-1)+(s2/N2)*(s2/N2)/(N2-1);
    f64 df=den>1e-15?num/den:1;
    f64 p=2*(1-t_cdf(std::abs(t),df));
    return{t,p,df,p<alpha};
}

// Paired t-test
template<int N>
[[nodiscard]] constexpr TTestResult<N> paired_t_test(
    const std::array<f64,N>& x, const std::array<f64,N>& y, f64 alpha=0.05) noexcept
{
    std::array<f64,N> d{};
    for(int i=0;i<N;++i) d[i]=x[i]-y[i];
    return one_sample_t_test<N>(d,0,alpha);
}

// ─── Chi-squared goodness of fit ──────────────────────────────────────────────
template<int K>
[[nodiscard]] constexpr std::pair<f64,f64> chi2_gof(
    const std::array<f64,K>& observed,
    const std::array<f64,K>& expected) noexcept
{
    f64 chi2=0;
    for(int i=0;i<K;++i) if(expected[i]>1e-15)
        chi2+=(observed[i]-expected[i])*(observed[i]-expected[i])/expected[i];
    f64 df=K-1;
    // P-value: 1-CDF(chi2,df)
    f64 p=1-std::exp(-chi2*0.5)*std::pow(chi2*0.5,(df*0.5-1)); // rough approx
    return{chi2,p};
}

// ─── Pearson correlation ──────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 pearson_r(
    const std::array<f64,N>& x, const std::array<f64,N>& y) noexcept
{
    f64 mx=stat_mean(x),my=stat_mean(y);
    f64 num=0,dx2=0,dy2=0;
    for(int i=0;i<N;++i){
        f64 dxi=x[i]-mx,dyi=y[i]-my;
        num+=dxi*dyi;dx2+=dxi*dxi;dy2+=dyi*dyi;
    }
    f64 d=std::sqrt(dx2*dy2);
    return d>1e-15?num/d:0;
}

// Spearman rank correlation
template<int N>
[[nodiscard]] constexpr f64 spearman_r(
    const std::array<f64,N>& x, const std::array<f64,N>& y) noexcept
{
    // Rank x and y
    std::array<f64,N> rx{},ry{};
    for(int i=0;i<N;++i){
        f64 rank=1;
        for(int j=0;j<N;++j) if(x[j]<x[i])++rank;
        rx[i]=rank;
    }
    for(int i=0;i<N;++i){
        f64 rank=1;
        for(int j=0;j<N;++j) if(y[j]<y[i])++rank;
        ry[i]=rank;
    }
    return pearson_r<N>(rx,ry);
}

// ─── One-way ANOVA ────────────────────────────────────────────────────────────
template<int K, int N>
[[nodiscard]] constexpr std::pair<f64,f64> one_way_anova(
    const std::array<std::array<f64,N>,K>& groups) noexcept
{
    // Grand mean
    f64 grand_mean=0;
    for(auto& g:groups) for(auto v:g) grand_mean+=v;
    grand_mean/=(K*N);
    // SS_between
    f64 ssb=0;
    for(auto& g:groups){f64 gm=stat_mean(g);ssb+=N*(gm-grand_mean)*(gm-grand_mean);}
    // SS_within
    f64 ssw=0;
    for(auto& g:groups){f64 gm=stat_mean(g);for(auto v:g)ssw+=(v-gm)*(v-gm);}
    f64 df_b=K-1,df_w=K*(N-1);
    f64 msb=ssb/df_b, msw=ssw/df_w;
    f64 F=msw>1e-15?msb/msw:0;
    // P-value approx (not exact without F CDF)
    f64 p=std::exp(-0.5*F); // placeholder: needs proper F distribution CDF
    return{F,p};
}

// ─── Bootstrap confidence interval ───────────────────────────────────────────
template<int N, int B=1000>
[[nodiscard]] constexpr std::pair<f64,f64> bootstrap_ci(
    const std::array<f64,N>& x, f64 level=0.95, u64 seed=42) noexcept
{
    Xorshift64 rng{seed};
    std::array<f64,B> boot_means{};
    for(int b=0;b<B;++b){
        f64 s=0;
        for(int i=0;i<N;++i) s+=x[rng.next()%N];
        boot_means[b]=s/N;
    }
    // Sort
    for(int i=1;i<B;++i) for(int j=i;j>0&&boot_means[j]<boot_means[j-1];--j)
        std::swap(boot_means[j],boot_means[j-1]);
    int lo=int((1-level)*0.5*B), hi=int((1-(1-level)*0.5)*B);
    return{boot_means[lo],boot_means[hi]};
}

// ─── Multiple testing corrections ────────────────────────────────────────────
// Bonferroni: alpha_adjusted = alpha / m
[[nodiscard]] constexpr f64 bonferroni(f64 alpha, int m) noexcept { return alpha/m; }
// Holm-Bonferroni: sort p-values, adjust threshold
template<int M>
[[nodiscard]] constexpr std::array<bool,M> holm_bonferroni(
    std::array<f64,M> p_values, f64 alpha=0.05) noexcept
{
    // Sort with index
    std::array<int,M> idx{}; for(int i=0;i<M;++i) idx[i]=i;
    for(int i=1;i<M;++i) for(int j=i;j>0&&p_values[idx[j]]<p_values[idx[j-1]];--j)
        std::swap(idx[j],idx[j-1]);
    std::array<bool,M> reject{};
    for(int i=0;i<M;++i){
        if(p_values[idx[i]]<=alpha/(M-i)){
            reject[idx[i]]=true;
        } else break;
    }
    return reject;
}
} // namespace qs::math
