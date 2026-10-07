
// qs_math/statistics/statistics_more_impl.cpp (C++20)
// More stats: Gaussian mixture, EM algorithm, robust estimators, hypothesis tests.
#include "include/qs_math_statistics.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::stats {

// ─── Gaussian mixture model (EM algorithm) ───────────────────────────────────
struct GMMResult {
    std::vector<f64> weights;
    std::vector<std::vector<f64>> means; // each: d-dimensional
    std::vector<std::vector<f64>> variances; // diagonal covariance
    f64 log_likelihood;
    i32 iters;
};

GMMResult gmm_em(const f64* X, i32 n, i32 d, i32 K,
    i32 max_iter=200, f64 tol=1e-6, u64 seed=42) noexcept
{
    GMMResult r;
    r.weights.assign(K,1.0/K);
    r.means.resize(K,std::vector<f64>(d));
    r.variances.resize(K,std::vector<f64>(d,1.0));

    // K-means init
    auto km=kmeans(X,n,d,K,100,seed);
    for(i32 k=0;k<K;++k) r.means[k]=km.centroids[k];

    // Compute variance from initial assignment
    for(i32 k=0;k<K;++k){
        std::fill(r.variances[k].begin(),r.variances[k].end(),0);
        i32 cnt=0;
        for(i32 i=0;i<n;++i) if(km.labels[i]==k){
            for(i32 j=0;j<d;++j)r.variances[k][j]+=(X[i*d+j]-r.means[k][j])*(X[i*d+j]-r.means[k][j]);
            ++cnt;
        }
        for(auto& v:r.variances[k]) v=std::max(1e-4,cnt>0?v/cnt:1.0);
    }

    // EM iterations
    std::vector<std::vector<f64>> resp(n,std::vector<f64>(K));
    r.log_likelihood=-1e300;

    for(r.iters=0;r.iters<max_iter;++r.iters){
        // E-step: compute responsibilities
        f64 ll=0;
        for(i32 i=0;i<n;++i){
            f64 sum_r=0;
            for(i32 k=0;k<K;++k){
                f64 log_p=std::log(std::max(1e-300,r.weights[k]));
                for(i32 j=0;j<d;++j){
                    const f64 dv=X[i*d+j]-r.means[k][j];
                    log_p-=0.5*(dv*dv/r.variances[k][j]+std::log(2*M_PI*r.variances[k][j]));
                }
                resp[i][k]=std::exp(log_p);
                sum_r+=resp[i][k];
            }
            ll+=std::log(std::max(1e-300,sum_r));
            for(i32 k=0;k<K;++k) resp[i][k]/=std::max(sum_r,1e-300);
        }

        // M-step
        std::vector<f64> Nk(K,0);
        for(i32 i=0;i<n;++i) for(i32 k=0;k<K;++k) Nk[k]+=resp[i][k];

        for(i32 k=0;k<K;++k){
            r.weights[k]=Nk[k]/n;
            std::fill(r.means[k].begin(),r.means[k].end(),0);
            for(i32 i=0;i<n;++i)
                for(i32 j=0;j<d;++j) r.means[k][j]+=resp[i][k]*X[i*d+j]/std::max(Nk[k],1e-14);
            std::fill(r.variances[k].begin(),r.variances[k].end(),0);
            for(i32 i=0;i<n;++i)
                for(i32 j=0;j<d;++j){
                    const f64 dv=X[i*d+j]-r.means[k][j];
                    r.variances[k][j]+=resp[i][k]*dv*dv/std::max(Nk[k],1e-14);
                }
            for(auto& v:r.variances[k]) v=std::max(1e-4,v);
        }

        if(std::abs(ll-r.log_likelihood)<tol){r.log_likelihood=ll;break;}
        r.log_likelihood=ll;
    }
    return r;
}

// ─── Robust estimators ────────────────────────────────────────────────────────
// Median absolute deviation (MAD)
[[nodiscard]] f64 mad(const f64* data, i32 n) noexcept {
    std::vector<f64> tmp(data,data+n);
    std::nth_element(tmp.begin(),tmp.begin()+n/2,tmp.end());
    const f64 med=tmp[n/2];
    for(auto& v:tmp) v=std::abs(v-med);
    std::nth_element(tmp.begin(),tmp.begin()+n/2,tmp.end());
    return tmp[n/2]*1.4826; // normalised to be consistent with std dev
}

// Huber M-estimator (robust mean)
[[nodiscard]] f64 huber_m_est(const f64* data, i32 n,
    f64 k=1.345, i32 max_iter=50) noexcept
{
    f64 mu=0; for(i32 i=0;i<n;++i)mu+=data[i]/n;
    const f64 s=mad(data,n)+1e-14;
    for(i32 iter=0;iter<max_iter;++iter){
        f64 num=0,den=0;
        for(i32 i=0;i<n;++i){
            const f64 r=(data[i]-mu)/s;
            const f64 w=std::abs(r)<=k?1.0:k/std::abs(r);
            num+=w*data[i]; den+=w;
        }
        const f64 mu_new=den>1e-14?num/den:mu;
        if(std::abs(mu_new-mu)<1e-10*s){mu=mu_new;break;}
        mu=mu_new;
    }
    return mu;
}

// Theil-Sen slope estimator (robust linear regression)
[[nodiscard]] std::pair<f64,f64> theil_sen(const f64* x, const f64* y, i32 n) noexcept {
    std::vector<f64> slopes;
    slopes.reserve(n*(n-1)/2);
    for(i32 i=0;i<n;++i)
        for(i32 j=i+1;j<n;++j)
            if(std::abs(x[j]-x[i])>1e-14)
                slopes.push_back((y[j]-y[i])/(x[j]-x[i]));
    std::nth_element(slopes.begin(),slopes.begin()+slopes.size()/2,slopes.end());
    const f64 slope=slopes[slopes.size()/2];
    // Intercept via median of residuals
    std::vector<f64> intercepts(n);
    for(i32 i=0;i<n;++i) intercepts[i]=y[i]-slope*x[i];
    std::nth_element(intercepts.begin(),intercepts.begin()+n/2,intercepts.end());
    return{slope,intercepts[n/2]};
}

// ─── Kolmogorov-Smirnov test (1-sample, vs normal) ───────────────────────────
struct KSResult { f64 statistic, p_value; bool rejected; };

[[nodiscard]] KSResult ks_test_normal(const f64* data, i32 n) noexcept {
    // Lilliefors test variant
    f64 mu=0,sigma=0;
    for(i32 i=0;i<n;++i) mu+=data[i]/n;
    for(i32 i=0;i<n;++i) sigma+=(data[i]-mu)*(data[i]-mu);
    sigma=std::sqrt(sigma/(n-1));

    std::vector<f64> sorted(data,data+n);
    std::sort(sorted.begin(),sorted.end());

    f64 D=0;
    for(i32 i=0;i<n;++i){
        const f64 F=(1+std::erf((sorted[i]-mu)/(sigma*1.41421356237)))*0.5;
        const f64 Fn=(f64)(i+1)/n;
        const f64 Fn_=i>0?(f64)i/n:0;
        D=std::max(D,std::max(std::abs(F-Fn),std::abs(F-Fn_)));
    }
    // Approximate p-value (Kolmogorov distribution)
    const f64 t=D*std::sqrt(n);
    const f64 p=2*std::exp(-2*t*t); // asymptotic
    return{D,std::min(1.0,p),p<0.05};
}

// ─── Anderson-Darling test ────────────────────────────────────────────────────
[[nodiscard]] f64 anderson_darling_statistic(const f64* data, i32 n) noexcept {
    f64 mu=0,sigma=0;
    for(i32 i=0;i<n;++i)mu+=data[i]/n;
    for(i32 i=0;i<n;++i)sigma+=(data[i]-mu)*(data[i]-mu);
    sigma=std::sqrt(sigma/(n-1));
    std::vector<f64> s(data,data+n);
    std::sort(s.begin(),s.end());
    f64 A=0;
    for(i32 i=0;i<n;++i){
        const f64 F=(1+std::erf((s[i]-mu)/(sigma*1.41421356237)))*0.5;
        const f64 F2=(1+std::erf((s[n-1-i]-mu)/(sigma*1.41421356237)))*0.5;
        A+=(2*i+1)*(std::log(std::max(1e-14,F))+std::log(std::max(1e-14,1-F2)));
    }
    return -n-A/n;
}

// ─── False discovery rate (Benjamini-Hochberg) ────────────────────────────────
std::vector<f64> bh_correction(std::vector<f64> pvals) noexcept {
    const i32 m=(i32)pvals.size();
    std::vector<i32> idx(m); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](i32 a,i32 b){return pvals[a]<pvals[b];});
    std::vector<f64> adj(m);
    f64 prev=1;
    for(i32 i=m-1;i>=0;--i){
        const f64 p=pvals[idx[i]]*m/(i+1);
        adj[idx[i]]=prev=std::min(prev,std::min(1.0,p));
    }
    return adj;
}

// ─── Bayesian information criterion ───────────────────────────────────────────
[[nodiscard]] f64 bic(f64 log_likelihood, i32 n_params, i32 n_samples) noexcept {
    return -2*log_likelihood+n_params*std::log(n_samples);
}
[[nodiscard]] f64 aic(f64 log_likelihood, i32 n_params) noexcept {
    return -2*log_likelihood+2*n_params;
}
[[nodiscard]] f64 aicc(f64 log_likelihood, i32 n_params, i32 n_samples) noexcept {
    if(n_samples-n_params-1<=0) return aic(log_likelihood,n_params);
    return aic(log_likelihood,n_params)+2.0*n_params*(n_params+1)/(n_samples-n_params-1);
}

// ─── Cross-validation (k-fold) ────────────────────────────────────────────────
[[nodiscard]] f64 k_fold_cv(
    std::function<f64(const f64*,const f64*,i32,const f64*,i32)> model_cv,
    const f64* X, const f64* y, i32 n, i32 k=5, u64 seed=0) noexcept
{
    // model_cv(X_train, y_train, n_train, X_test, n_test) → MSE on test
    std::vector<i32> idx(n); std::iota(idx.begin(),idx.end(),0);
    u64 rng=seed;
    for(i32 i=n-1;i>0;--i){
        rng=rng*6364136223846793005ULL+1442695040888963407ULL;
        i32 j=(i32)((rng>>33)%(i+1));
        std::swap(idx[i],idx[j]);
    }
    f64 total_mse=0;
    for(i32 fold=0;fold<k;++fold){
        const i32 test_start=fold*n/k, test_end=(fold+1)*n/k;
        const i32 n_test=test_end-test_start, n_train=n-n_test;
        std::vector<f64> X_tr(n_train),y_tr(n_train),X_te(n_test);
        i32 ti=0,vi=0;
        for(i32 i=0;i<n;++i){
            if(i>=test_start&&i<test_end){X_te[vi]=X[idx[i]];}
            else{X_tr[ti]=X[idx[i]];y_tr[ti]=y[idx[i]];++ti;}
            if(i>=test_start&&i<test_end)++vi;
        }
        total_mse+=model_cv(X_tr.data(),y_tr.data(),n_train,X_te.data(),n_test);
    }
    return total_mse/k;
}

} // namespace qs::math::stats
