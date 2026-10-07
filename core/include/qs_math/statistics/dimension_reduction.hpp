#pragma once
// qs_math/dimension_reduction.hpp — PCA, t-SNE perplexity, UMAP distances, MDS (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <span>
#include <array>
#include <random>

namespace qs {

// ─── PCA ──────────────────────────────────────────────────────────────────────

struct PCAResult {
    std::vector<f64> mean;
    std::vector<f64> components;   // [n_components x n_features] row-major
    std::vector<f64> explained_var;
    u32 n_features{}, n_components{};
};

// Center data: X -= mean
inline void center_data(std::span<f64> X, u32 n_samples, u32 n_features,
                         std::vector<f64>& mean)
{
    mean.assign(n_features,0.);
    for(u32 i=0;i<n_samples;++i) for(u32 j=0;j<n_features;++j) mean[j]+=X[i*n_features+j];
    for(auto&m:mean) m/=f64(n_samples);
    for(u32 i=0;i<n_samples;++i) for(u32 j=0;j<n_features;++j) X[i*n_features+j]-=mean[j];
}

// Covariance matrix C = X^T X / (n-1)
[[nodiscard]] inline std::vector<f64> covariance_matrix(
    std::span<const f64> X, u32 n_samples, u32 n_features)
{
    std::vector<f64> C(n_features*n_features,0.);
    for(u32 i=0;i<n_features;++i)
    for(u32 j=0;j<n_features;++j){
        f64 s=0.;
        for(u32 k=0;k<n_samples;++k) s+=X[k*n_features+i]*X[k*n_features+j];
        C[i*n_features+j]=s/f64(n_samples-1);
    }
    return C;
}

// Power iteration for top-k eigenvectors
[[nodiscard]] inline PCAResult pca_power_iter(
    std::vector<f64> X, u32 n_samples, u32 n_features,
    u32 n_components=2, u32 max_iter=100, u64 seed=42)
{
    PCAResult res;
    res.n_features=n_features; res.n_components=n_components;
    center_data(X,n_samples,n_features,res.mean);
    auto C=covariance_matrix(X,n_samples,n_features);

    std::mt19937 rng(u32(seed));
    std::normal_distribution<f64> dist;
    res.components.resize(n_components*n_features);
    res.explained_var.resize(n_components);

    std::vector<f64> deflated=C;
    for(u32 c=0;c<n_components;++c){
        std::vector<f64> v(n_features);
        for(auto&x:v) x=dist(rng);
        f64 norm=0.; for(auto x:v) norm+=x*x; norm=std::sqrt(norm);
        for(auto&x:v) x/=norm;
        for(u32 iter=0;iter<max_iter;++iter){
            std::vector<f64> vn(n_features,0.);
            for(u32 i=0;i<n_features;++i) for(u32 j=0;j<n_features;++j) vn[i]+=deflated[i*n_features+j]*v[j];
            // Gram-Schmidt orthogonalize against previous components
            for(u32 p=0;p<c;++p){
                f64 dot=0.; for(u32 i=0;i<n_features;++i) dot+=vn[i]*res.components[p*n_features+i];
                for(u32 i=0;i<n_features;++i) vn[i]-=dot*res.components[p*n_features+i];
            }
            norm=0.; for(auto x:vn) norm+=x*x; norm=std::sqrt(norm+1e-15);
            for(u32 i=0;i<n_features;++i) v[i]=vn[i]/norm;
        }
        // Eigenvalue = v^T C v
        std::vector<f64> Cv(n_features,0.);
        for(u32 i=0;i<n_features;++i) for(u32 j=0;j<n_features;++j) Cv[i]+=deflated[i*n_features+j]*v[j];
        f64 eigenval=0.; for(u32 i=0;i<n_features;++i) eigenval+=v[i]*Cv[i];
        res.explained_var[c]=eigenval;
        for(u32 i=0;i<n_features;++i) res.components[c*n_features+i]=v[i];
        // Deflate
        for(u32 i=0;i<n_features;++i) for(u32 j=0;j<n_features;++j)
            deflated[i*n_features+j]-=eigenval*v[i]*v[j];
    }
    return res;
}

// Project data onto PCA components
[[nodiscard]] inline std::vector<f64> pca_transform(
    std::span<const f64> X, u32 n_samples, const PCAResult& pca)
{
    u32 D=pca.n_features, K=pca.n_components;
    std::vector<f64> out(n_samples*K,0.);
    for(u32 i=0;i<n_samples;++i)
    for(u32 k=0;k<K;++k)
    for(u32 j=0;j<D;++j)
        out[i*K+k]+=(X[i*D+j]-pca.mean[j])*pca.components[k*D+j];
    return out;
}

// ─── t-SNE perplexity search ──────────────────────────────────────────────────

// Binary search for bandwidth sigma such that perplexity(P_i) = target_perp
[[nodiscard]] inline f64 tsne_sigma_for_perplexity(
    std::span<const f64> distances_sq, // distances^2 from point i to all others
    f64 target_perp, u32 max_iter=50, f64 tol=1e-5)
{
    f64 sigma=1., lo=0., hi=1e10f;
    for(u32 iter=0;iter<max_iter;++iter){
        // Compute P_j|i = exp(-d^2/2sigma^2) / Z
        f64 Z=0., H=0.;
        for(f64 d2:distances_sq){f64 p=std::exp(-d2/(2.*sigma*sigma));Z+=p;}
        if(Z<1e-15) {sigma*=2.; continue;}
        for(f64 d2:distances_sq){
            f64 p=std::exp(-d2/(2.*sigma*sigma))/Z;
            if(p>1e-15) H-=p*std::log2(p);
        }
        f64 perp=std::pow(2.,H);
        if(std::abs(perp-target_perp)<tol) break;
        if(perp<target_perp){lo=sigma;sigma=(hi>1e9)?sigma*2:0.5*(sigma+hi);}
        else{hi=sigma;sigma=0.5*(lo+sigma);}
    }
    return sigma;
}

// t-SNE conditional probabilities P(j|i)
[[nodiscard]] inline std::vector<f64> tsne_conditional_probs(
    std::span<const f64> X, u32 n, u32 d, f64 perplexity=30.)
{
    std::vector<f64> P(n*n,0.);
    for(u32 i=0;i<n;++i){
        std::vector<f64> dists(n,0.);
        for(u32 j=0;j<n;++j){
            if(i==j) continue;
            for(u32 k=0;k<d;++k){f64 diff=X[i*d+k]-X[j*d+k];dists[j]+=diff*diff;}
        }
        std::vector<f64> dists_noself;
        for(u32 j=0;j<n;++j) if(j!=i) dists_noself.push_back(dists[j]);
        f64 sigma=tsne_sigma_for_perplexity(dists_noself,perplexity);
        f64 Z=0.;
        for(u32 j=0;j<n;++j){if(j==i)continue;P[i*n+j]=std::exp(-dists[j]/(2.*sigma*sigma));Z+=P[i*n+j];}
        if(Z>0.) for(u32 j=0;j<n;++j) P[i*n+j]/=Z;
    }
    return P;
}

// ─── Classical MDS ────────────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<f64> classical_mds(
    std::span<const f64> D, u32 n, u32 target_dim=2)
{
    // Double-centering: B = -0.5 * J * D^2 * J, J = I - 1/n * 11^T
    std::vector<f64> B(n*n,0.);
    std::vector<f64> row_mean(n,0.), col_mean(n,0.); f64 grand=0.;
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j){
        f64 d2=D[i*n+j]*D[i*n+j];
        row_mean[i]+=d2; col_mean[j]+=d2; grand+=d2;}
    for(auto&r:row_mean)r/=f64(n); for(auto&c:col_mean)c/=f64(n); grand/=f64(n*n);
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
        B[i*n+j]=-0.5*(D[i*n+j]*D[i*n+j]-row_mean[i]-col_mean[j]+grand);
    // PCA on B to get embedding
    std::vector<f64> flat(n*n);for(u32 i=0;i<n*n;++i)flat[i]=B[i];
    auto pca=pca_power_iter(flat,n,n,target_dim);
    return pca_transform(flat,n,pca);
}

// ─── Isomap geodesic distance ─────────────────────────────────────────────────

// Floyd-Warshall for geodesic distances (small graphs)
[[nodiscard]] inline std::vector<f64> floyd_warshall(
    std::span<const f64> W, u32 n)
{
    std::vector<f64> D(W.begin(),W.end());
    for(u32 k=0;k<n;++k) for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
        if(D[i*n+k]+D[k*n+j]<D[i*n+j]) D[i*n+j]=D[i*n+k]+D[k*n+j];
    return D;
}

} // namespace qs
