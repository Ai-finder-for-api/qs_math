#pragma once
// qs_math/regression_math.hpp — Linear, polynomial, logistic, RANSAC, Gaussian process (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <span>
#include <optional>
#include <random>
#include <functional>

namespace qs {

// ─── Simple linear regression (y = mx + b) ────────────────────────────────────

struct LinearReg { f64 slope{}, intercept{}, r_sq{}; };

[[nodiscard]] inline LinearReg linear_regression(
    std::span<const f64> x, std::span<const f64> y)
{
    u32 n=u32(std::min(x.size(),y.size()));
    if(n<2) return {};
    f64 sx=0,sy=0,sxy=0,sxx=0;
    for(u32 i=0;i<n;++i){sx+=x[i];sy+=y[i];sxy+=x[i]*y[i];sxx+=x[i]*x[i];}
    f64 d=f64(n)*sxx-sx*sx;
    if(std::abs(d)<1e-15) return {};
    f64 m=(f64(n)*sxy-sx*sy)/d;
    f64 b=(sy-m*sx)/f64(n);
    // R^2
    f64 y_mean=sy/f64(n),ss_res=0,ss_tot=0;
    for(u32 i=0;i<n;++i){
        f64 yi_pred=m*x[i]+b;
        ss_res+=(y[i]-yi_pred)*(y[i]-yi_pred);
        ss_tot+=(y[i]-y_mean)*(y[i]-y_mean);
    }
    return {m,b,ss_tot>0?1.-ss_res/ss_tot:0.};
}

// ─── Polynomial regression ────────────────────────────────────────────────────

// Fit degree-d polynomial via normal equations (small d, moderate n)
[[nodiscard]] inline std::vector<f64> polynomial_regression(
    std::span<const f64> x, std::span<const f64> y, u32 degree)
{
    u32 n=u32(std::min(x.size(),y.size()));
    u32 d=degree+1;
    // Build Vandermonde matrix AtA and Aty
    std::vector<f64> AtA(d*d,0.), Aty(d,0.);
    for(u32 i=0;i<n;++i){
        std::vector<f64> xpow(d,1.);
        for(u32 k=1;k<d;++k) xpow[k]=xpow[k-1]*x[i];
        for(u32 j=0;j<d;++j){
            Aty[j]+=xpow[j]*y[i];
            for(u32 k=0;k<d;++k) AtA[j*d+k]+=xpow[j]*xpow[k];
        }
    }
    // Gaussian elimination
    std::vector<std::vector<f64>> M(d,std::vector<f64>(d+1));
    for(u32 i=0;i<d;++i){for(u32 j=0;j<d;++j)M[i][j]=AtA[i*d+j];M[i][d]=Aty[i];}
    for(u32 col=0;col<d;++col){
        u32 pivot=col;
        for(u32 row=col+1;row<d;++row) if(std::abs(M[row][col])>std::abs(M[pivot][col])) pivot=row;
        std::swap(M[col],M[pivot]);
        if(std::abs(M[col][col])<1e-14) continue;
        for(u32 row=col+1;row<d;++row){
            f64 f=M[row][col]/M[col][col];
            for(u32 k=col;k<=d;++k) M[row][k]-=f*M[col][k];
        }
    }
    std::vector<f64> coeffs(d,0.);
    for(i32 i=i32(d)-1;i>=0;--i){
        f64 s=M[i][d];
        for(u32 j=i+1;j<d;++j) s-=M[i][j]*coeffs[j];
        coeffs[i]=s/(M[i][i]+1e-15);
    }
    return coeffs;
}

[[nodiscard]] inline f64 poly_predict(std::span<const f64> coeffs, f64 x) {
    f64 y=0.,xpow=1.;
    for(f64 c:coeffs){y+=c*xpow;xpow*=x;}
    return y;
}

// ─── Ridge regression (L2 regularised) ───────────────────────────────────────

[[nodiscard]] inline std::vector<f64> ridge_regression(
    std::span<const f64> X, // n x p row-major
    std::span<const f64> y,
    u32 n, u32 p, f64 lambda=1.)
{
    // (X^T X + lambda I) beta = X^T y
    std::vector<f64> AtA(p*p,0.),Aty(p,0.);
    for(u32 i=0;i<n;++i){
        for(u32 j=0;j<p;++j){
            Aty[j]+=X[i*p+j]*y[i];
            for(u32 k=0;k<p;++k) AtA[j*p+k]+=X[i*p+j]*X[i*p+k];
        }
    }
    for(u32 j=0;j<p;++j) AtA[j*p+j]+=lambda;
    // Solve via Cholesky (simplified: Gaussian)
    std::vector<std::vector<f64>> M(p,std::vector<f64>(p+1));
    for(u32 i=0;i<p;++i){for(u32 j=0;j<p;++j)M[i][j]=AtA[i*p+j];M[i][p]=Aty[i];}
    for(u32 col=0;col<p;++col){
        if(std::abs(M[col][col])<1e-14) continue;
        for(u32 row=col+1;row<p;++row){
            f64 f=M[row][col]/M[col][col];
            for(u32 k=col;k<=p;++k) M[row][k]-=f*M[col][k];
        }
    }
    std::vector<f64> beta(p,0.);
    for(i32 i=i32(p)-1;i>=0;--i){
        f64 s=M[i][p];
        for(u32 j=i+1;j<p;++j) s-=M[i][j]*beta[j];
        beta[i]=s/(M[i][i]+1e-15);
    }
    return beta;
}

// ─── Logistic regression ──────────────────────────────────────────────────────

[[nodiscard]] inline f64 logistic(f64 z) { return 1./(1.+std::exp(-z)); }

// Gradient of log-likelihood: X^T (y - sigma(Xw))
[[nodiscard]] inline std::vector<f64> logistic_grad(
    std::span<const f64> X, std::span<const f64> y,
    std::span<const f64> w, u32 n, u32 p)
{
    std::vector<f64> g(p,0.);
    for(u32 i=0;i<n;++i){
        f64 z=0.;
        for(u32 j=0;j<p;++j) z+=X[i*p+j]*w[j];
        f64 err=y[i]-logistic(z);
        for(u32 j=0;j<p;++j) g[j]+=X[i*p+j]*err;
    }
    return g;
}

[[nodiscard]] inline std::vector<f64> logistic_sgd(
    std::span<const f64> X, std::span<const f64> y,
    u32 n, u32 p, f64 lr=0.01, u32 epochs=100, u64 seed=0)
{
    std::vector<f64> w(p,0.);
    std::mt19937_64 rng(seed);
    for(u32 e=0;e<epochs;++e){
        u32 i=u32(rng()%n);
        f64 z=0.;
        for(u32 j=0;j<p;++j) z+=X[i*p+j]*w[j];
        f64 err=y[i]-logistic(z);
        for(u32 j=0;j<p;++j) w[j]+=lr*X[i*p+j]*err;
    }
    return w;
}

// ─── RANSAC ───────────────────────────────────────────────────────────────────

struct RANSACResult {
    std::vector<bool> inliers;
    std::vector<f64>  model; // fitted model parameters
    u32 num_inliers{};
};

// RANSAC for linear regression
[[nodiscard]] inline RANSACResult ransac_linear(
    std::span<const f64> x, std::span<const f64> y,
    f64 threshold=0.1, u32 min_samples=2,
    u32 max_iter=100, f64 inlier_ratio=0.9,
    u64 seed=42)
{
    u32 n=u32(std::min(x.size(),y.size()));
    std::mt19937 rng(u32(seed));
    RANSACResult best;
    best.inliers.assign(n,false);

    for(u32 iter=0;iter<max_iter;++iter){
        // Sample min_samples points
        std::vector<u32> idx(n); std::iota(idx.begin(),idx.end(),0);
        std::shuffle(idx.begin(),idx.end(),rng);
        idx.resize(min_samples);

        std::vector<f64> sx(min_samples),sy(min_samples);
        for(u32 i=0;i<min_samples;++i){sx[i]=x[idx[i]];sy[i]=y[idx[i]];}
        auto lr=linear_regression(sx,sy);

        std::vector<bool> inliers(n);
        u32 count=0;
        for(u32 i=0;i<n;++i){
            f64 pred=lr.slope*x[i]+lr.intercept;
            if(std::abs(y[i]-pred)<threshold){inliers[i]=true;++count;}
        }
        if(count>best.num_inliers){
            best.inliers=inliers; best.num_inliers=count;
            best.model={lr.slope,lr.intercept};
        }
        if(count>=u32(f64(n)*inlier_ratio)) break;
    }
    // Refit on inliers
    std::vector<f64> ix,iy;
    for(u32 i=0;i<n;++i) if(best.inliers[i]){ix.push_back(x[i]);iy.push_back(y[i]);}
    if(!ix.empty()){
        auto lr=linear_regression(ix,iy);
        best.model={lr.slope,lr.intercept};
    }
    return best;
}

// ─── Gaussian process regression ─────────────────────────────────────────────

// Squared exponential kernel: k(x,x') = exp(-||x-x'||^2 / (2*l^2))
[[nodiscard]] inline f64 rbf_kernel(f64 xa, f64 xb, f64 length_scale=1., f64 signal_var=1.) {
    f64 d=(xa-xb)/length_scale;
    return signal_var*std::exp(-0.5*d*d);
}

// Simple 1D GP prediction (posterior mean and variance)
struct GPResult { f64 mean{}, variance{}; };

[[nodiscard]] inline GPResult gp_predict_1d(
    std::span<const f64> x_train, std::span<const f64> y_train,
    f64 x_test, f64 length_scale=1., f64 signal_var=1., f64 noise_var=0.01)
{
    u32 n=u32(std::min(x_train.size(),y_train.size()));
    // K = kernel matrix + noise*I
    std::vector<f64> K(n*n),k_star(n);
    for(u32 i=0;i<n;++i){
        k_star[i]=rbf_kernel(x_train[i],x_test,length_scale,signal_var);
        for(u32 j=0;j<n;++j) K[i*n+j]=rbf_kernel(x_train[i],x_train[j],length_scale,signal_var)+(i==j?noise_var:0.);
    }
    // Solve K alpha = y via Cholesky (simplified: Gaussian elim)
    std::vector<std::vector<f64>> M(n,std::vector<f64>(n+1));
    for(u32 i=0;i<n;++i){for(u32 j=0;j<n;++j)M[i][j]=K[i*n+j];M[i][n]=y_train[i];}
    for(u32 col=0;col<n;++col){
        if(std::abs(M[col][col])<1e-14) continue;
        for(u32 row=col+1;row<n;++row){
            f64 f=M[row][col]/M[col][col];
            for(u32 k=col;k<=n;++k) M[row][k]-=f*M[col][k];
        }
    }
    std::vector<f64> alpha(n,0.);
    for(i32 i=i32(n)-1;i>=0;--i){
        f64 s=M[i][n];
        for(u32 j=i+1;j<n;++j) s-=M[i][j]*alpha[j];
        alpha[i]=s/(M[i][i]+1e-15);
    }
    f64 mean=0.;
    for(u32 i=0;i<n;++i) mean+=k_star[i]*alpha[i];
    // Posterior variance (simplified: k** - k*^T K^-1 k*)
    f64 kss=rbf_kernel(x_test,x_test,length_scale,signal_var)+noise_var;
    // Solve K v = k_star
    std::vector<f64> v=k_star;
    f64 var=kss;
    for(u32 i=0;i<n;++i) var-=k_star[i]*v[i]/(K[i*n+i]+1e-15);
    return {mean,std::max(0.,var)};
}

} // namespace qs
