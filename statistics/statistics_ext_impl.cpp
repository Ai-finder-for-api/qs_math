
// qs_math/statistics/statistics_ext_impl.cpp (C++20)
#include "include/qs_math_statistics.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::stats {

static f64 mean_v(const f64* d,i32 n){f64 s=0;for(i32 i=0;i<n;++i)s+=d[i];return s/n;}
static f64 var_v(const f64* d,i32 n,bool pop=false){
    f64 m=mean_v(d,n),s=0;for(i32 i=0;i<n;++i)s+=(d[i]-m)*(d[i]-m);return s/(pop?n:n-1);}
static f64 normal_cdf_approx(f64 z){return 0.5*(1+std::erf(z/1.41421356237));}

BootstrapCI bootstrap_ci(const f64* data,i32 n,
    std::function<f64(const f64*,i32)> stat,i32 B,f64 alpha,u64 seed) noexcept
{
    u64 rng=seed;
    auto randu=[&]()->f64{rng=rng*6364136223846793005ULL+1442695040888963407ULL;return(rng>>11)*(1.0/9007199254740992.0);};
    std::vector<f64> boot_stats(B);
    std::vector<f64> sample(n);
    for(i32 b=0;b<B;++b){
        for(i32 i=0;i<n;++i)sample[i]=data[(i32)(randu()*n)];
        boot_stats[b]=stat(sample.data(),n);
    }
    std::sort(boot_stats.begin(),boot_stats.end());
    const f64 lo=boot_stats[(i32)(alpha*0.5*B)];
    const f64 hi=boot_stats[(i32)((1-alpha*0.5)*B)];
    const f64 m=stat(data,n);
    return{2*m-hi,2*m-lo,m};
}

ANOVAResult one_way_anova(const std::vector<std::vector<f64>>& g) noexcept {
    i32 k=(i32)g.size(),N=0;
    f64 grand_sum=0;
    for(auto&v:g){N+=(i32)v.size();for(auto x:v)grand_sum+=x;}
    const f64 grand_mean=grand_sum/N;
    f64 SSB=0,SSW=0;
    for(auto&v:g){
        const f64 gm=mean_v(v.data(),(i32)v.size());
        SSB+=v.size()*(gm-grand_mean)*(gm-grand_mean);
        for(auto x:v)SSW+=(x-gm)*(x-gm);
    }
    const f64 dfB=k-1,dfW=N-k;
    if(dfW<1||SSW<1e-14)return{0,1,false};
    const f64 F=(SSB/dfB)/(SSW/dfW);
    // F CDF approximation via beta distribution
    const f64 x=dfB*F/(dfB*F+dfW);
    // Regularized incomplete beta (simplified)
    const f64 p=1-x; // very rough — proper impl needs inc_beta
    return{F,p,p<0.05};
}

MWResult mann_whitney(const f64* a,i32 na,const f64* b,i32 nb,f64 alpha) noexcept {
    // Compute U statistic
    f64 U=0;
    for(i32 i=0;i<na;++i)for(i32 j=0;j<nb;++j)
        if(a[i]>b[j])U+=1;else if(a[i]==b[j])U+=0.5;
    const f64 mu=(f64)na*nb/2;
    const f64 sigma=std::sqrt((f64)na*nb*(na+nb+1)/12.0);
    const f64 z=(U-mu)/sigma;
    const f64 p=2*(1-normal_cdf_approx(std::abs(z)));
    return{U,p,p<alpha};
}

ARModel fit_ar(const f64* ts,i32 n,i32 p) noexcept {
    assert(n>p+1);
    // OLS via Yule-Walker equations
    std::vector<f64> r(p+1,0);
    const f64 mu=mean_v(ts,n);
    for(i32 lag=0;lag<=p;++lag)
        for(i32 i=lag;i<n;++i)r[lag]+=(ts[i]-mu)*(ts[i-lag]-mu);
    for(auto&v:r)v/=(n-1);
    // Solve Toeplitz system Rp * phi = r1..p (Durbin-Levinson simplified)
    std::vector<f64> phi(p,0);
    f64 err=r[0];
    for(i32 k=0;k<p;++k){
        f64 lambda=r[k+1];
        for(i32 j=0;j<k;++j)lambda-=phi[j]*r[k-j];
        const f64 alpha2=lambda/err;
        std::vector<f64> phi2=phi;
        for(i32 j=0;j<k;++j)phi2[j]-=alpha2*phi[k-1-j];
        phi2[k]=alpha2;
        phi=phi2;
        err*=(1-alpha2*alpha2);
    }
    const f64 intercept=mu*(1-std::accumulate(phi.begin(),phi.end(),0.0));
    return{phi,intercept};
}
f64 ARModel::predict(const f64* history,i32 p) const noexcept {
    f64 v=intercept;
    for(i32 i=0;i<p;++i)v+=coefs[i]*history[p-1-i];
    return v;
}
std::vector<f64> moving_average(const f64* ts,i32 n,i32 w) noexcept {
    std::vector<f64> out(n,0);
    for(i32 i=0;i<n;++i){
        i32 cnt=0; f64 s=0;
        for(i32 j=std::max(0,i-w/2);j<=std::min(n-1,i+w/2);++j){s+=ts[j];++cnt;}
        out[i]=s/cnt;
    }
    return out;
}
std::vector<f64> exponential_smoothing(const f64* ts,i32 n,f64 alpha) noexcept {
    std::vector<f64> out(n);
    out[0]=ts[0];
    for(i32 i=1;i<n;++i)out[i]=alpha*ts[i]+(1-alpha)*out[i-1];
    return out;
}
std::vector<f64> holt_winters(const f64* ts,i32 n,f64 alpha,f64 beta,f64 gamma,i32 period) noexcept {
    std::vector<f64> out(n),level(n),trend(n),seasonal(period,1.0);
    level[0]=ts[0]; trend[0]=0;
    for(i32 i=0;i<std::min(period,n);++i)seasonal[i]=ts[i]/(ts[0]+1e-14);
    for(i32 i=1;i<n;++i){
        const i32 s=((i-1)%period);
        const f64 L=alpha*(ts[i]/seasonal[s])+(1-alpha)*(level[i-1]+trend[i-1]);
        const f64 T=beta*(L-level[i-1])+(1-beta)*trend[i-1];
        seasonal[s]=gamma*(ts[i]/L)+(1-gamma)*seasonal[s];
        out[i]=(L+T)*seasonal[s];
        level[i]=L; trend[i]=T;
    }
    return out;
}
PCAResult pca(const f64* X,i32 n,i32 d,i32 k) noexcept {
    // Center data
    std::vector<f64> mean(d,0);
    for(i32 i=0;i<n;++i)for(i32 j=0;j<d;++j)mean[j]+=X[i*d+j]/n;
    std::vector<f64> Xc(n*d);
    for(i32 i=0;i<n;++i)for(i32 j=0;j<d;++j)Xc[i*d+j]=X[i*d+j]-mean[j];
    // Covariance matrix d×d
    std::vector<f64> cov(d*d,0);
    for(i32 i=0;i<n;++i)for(i32 p2=0;p2<d;++p2)for(i32 q=0;q<d;++q)
        cov[p2*d+q]+=Xc[i*d+p2]*Xc[i*d+q]/(n-1);
    // Power iteration for k components
    PCAResult r;
    std::vector<f64> C=cov;
    for(i32 comp=0;comp<k&&comp<d;++comp){
        std::vector<f64> v(d,1.0/d); // init
        f64 lam=0;
        for(i32 it=0;it<200;++it){
            std::vector<f64> Cv(d,0);
            for(i32 i=0;i<d;++i)for(i32 j=0;j<d;++j)Cv[i]+=C[i*d+j]*v[j];
            f64 nm=0;for(auto x:Cv)nm+=x*x;nm=std::sqrt(nm);
            if(nm<1e-14)break;
            const f64 lam2=0;(void)lam2;
            lam=nm;
            for(i32 i=0;i<d;++i)v[i]=Cv[i]/nm;
        }
        r.eigenvalues.push_back(lam);
        r.components.push_back(v);
        // Deflate
        for(i32 i=0;i<d;++i)for(i32 j=0;j<d;++j)C[i*d+j]-=lam*v[i]*v[j];
    }
    const f64 total=std::accumulate(r.eigenvalues.begin(),r.eigenvalues.end(),0.0);
    for(auto lam:r.eigenvalues)r.explained_variance_ratio.push_back(total>0?lam/total:0);
    return r;
}
KMeansResult kmeans(const f64* X,i32 n,i32 d,i32 k,i32 max_iter,u64 seed) noexcept {
    u64 rng=seed;
    auto randi=[&](i32 m)->i32{rng=rng*6364136223846793005ULL+1;return(i32)((rng>>33)%(u32)m);};
    std::vector<std::vector<f64>> centroids(k,std::vector<f64>(d));
    // K-means++ init
    std::vector<i32> chosen;
    chosen.push_back(randi(n));
    for(i32 c=1;c<k;++c){
        std::vector<f64> dist2(n,1e20);
        for(i32 i=0;i<n;++i)for(i32 ci:chosen){
            f64 s=0;for(i32 j=0;j<d;++j)s+=(X[i*d+j]-X[ci*d+j])*(X[i*d+j]-X[ci*d+j]);
            dist2[i]=std::min(dist2[i],s);}
        f64 tot=std::accumulate(dist2.begin(),dist2.end(),0.0);
        f64 r=(rng>>11)*(1.0/9007199254740992.0)*tot; rng=rng*6364136223846793005ULL+1;
        f64 acc=0; i32 sel=0;
        for(i32 i=0;i<n;++i){acc+=dist2[i];if(acc>=r){sel=i;break;}}
        chosen.push_back(sel);
    }
    for(i32 c=0;c<k;++c)for(i32 j=0;j<d;++j)centroids[c][j]=X[chosen[c]*d+j];
    std::vector<i32> labels(n,-1);
    for(i32 it=0;it<max_iter;++it){
        bool changed=false;
        for(i32 i=0;i<n;++i){
            f64 best=1e20;i32 bl=-1;
            for(i32 c=0;c<k;++c){f64 s=0;for(i32 j=0;j<d;++j)s+=(X[i*d+j]-centroids[c][j])*(X[i*d+j]-centroids[c][j]);if(s<best){best=s;bl=c;}}
            if(bl!=labels[i]){labels[i]=bl;changed=true;}
        }
        if(!changed)break;
        for(auto&c:centroids)std::fill(c.begin(),c.end(),0);
        std::vector<i32> cnt(k,0);
        for(i32 i=0;i<n;++i){for(i32 j=0;j<d;++j)centroids[labels[i]][j]+=X[i*d+j];cnt[labels[i]]++;}
        for(i32 c=0;c<k;++c)if(cnt[c]>0)for(auto&v:centroids[c])v/=cnt[c];
    }
    f64 inertia=0;
    for(i32 i=0;i<n;++i){for(i32 j=0;j<d;++j)inertia+=(X[i*d+j]-centroids[labels[i]][j])*(X[i*d+j]-centroids[labels[i]][j]);}
    return{labels,centroids,inertia};
}
std::vector<f64> correlation_matrix(const f64* X,i32 n,i32 d) noexcept {
    std::vector<f64> mu(d,0),sigma(d,0),C(d*d,0);
    for(i32 i=0;i<n;++i)for(i32 j=0;j<d;++j)mu[j]+=X[i*d+j]/n;
    for(i32 i=0;i<n;++i)for(i32 j=0;j<d;++j)sigma[j]+=(X[i*d+j]-mu[j])*(X[i*d+j]-mu[j]);
    for(i32 j=0;j<d;++j)sigma[j]=std::sqrt(sigma[j]/(n-1));
    for(i32 p2=0;p2<d;++p2)for(i32 q=0;q<d;++q){
        f64 s=0;for(i32 i=0;i<n;++i)s+=(X[i*d+p2]-mu[p2])*(X[i*d+q]-mu[q]);
        C[p2*d+q]=sigma[p2]*sigma[q]>1e-14?s/((n-1)*sigma[p2]*sigma[q]):0;
    }
    return C;
}
void zscore_normalize(f64* X,i32 n,i32 d) noexcept {
    for(i32 j=0;j<d;++j){
        f64 mu=0;for(i32 i=0;i<n;++i)mu+=X[i*d+j]/n;
        f64 sg=0;for(i32 i=0;i<n;++i)sg+=(X[i*d+j]-mu)*(X[i*d+j]-mu);
        sg=std::sqrt(sg/(n-1));if(sg<1e-14)sg=1;
        for(i32 i=0;i<n;++i)X[i*d+j]=(X[i*d+j]-mu)/sg;
    }
}
} // namespace qs::math::stats
