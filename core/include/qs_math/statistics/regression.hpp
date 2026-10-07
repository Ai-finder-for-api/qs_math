#pragma once
// qs_math/statistics/regression.hpp — Linear, ridge, LASSO, logistic, polynomial regression
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Ordinary least squares (closed form: β=(X'X)^{-1}X'y) ───────────────────
template<int N, int P>
[[nodiscard]] constexpr std::array<f64,P> ols(
    const std::array<std::array<f64,P>,N>& X,
    const std::array<f64,N>& y) noexcept
{
    // X'X
    std::array<std::array<f64,P>,P> XtX{};
    for(int i=0;i<P;++i) for(int j=0;j<P;++j)
        for(int k=0;k<N;++k) XtX[i][j]+=X[k][i]*X[k][j];
    // X'y
    std::array<f64,P> Xty{};
    for(int i=0;i<P;++i) for(int k=0;k<N;++k) Xty[i]+=X[k][i]*y[k];
    // Solve XtX*beta=Xty via Cholesky or Gaussian elimination
    // Gaussian elimination
    std::array<std::array<f64,P+1>,P> aug{};
    for(int i=0;i<P;++i){for(int j=0;j<P;++j)aug[i][j]=XtX[i][j];aug[i][P]=Xty[i];}
    for(int k=0;k<P;++k) {
        int p=k; for(int i=k+1;i<P;++i) if(std::abs(aug[i][k])>std::abs(aug[p][k]))p=i;
        std::swap(aug[p],aug[k]);
        if(std::abs(aug[k][k])<1e-15) continue;
        for(int i=0;i<P;++i) if(i!=k) {
            f64 f=aug[i][k]/aug[k][k];
            for(int j=k;j<=P;++j) aug[i][j]-=f*aug[k][j];
        }
    }
    std::array<f64,P> beta{};
    for(int i=0;i<P;++i) if(std::abs(aug[i][i])>1e-15) beta[i]=aug[i][P]/aug[i][i];
    return beta;
}

// ─── Ridge regression (L2 regularization): β=(X'X+λI)^{-1}X'y ───────────────
template<int N, int P>
[[nodiscard]] constexpr std::array<f64,P> ridge(
    const std::array<std::array<f64,P>,N>& X,
    const std::array<f64,N>& y, f64 lambda) noexcept
{
    std::array<std::array<f64,P>,P> XtX{};
    for(int i=0;i<P;++i) for(int j=0;j<P;++j)
        for(int k=0;k<N;++k) XtX[i][j]+=X[k][i]*X[k][j];
    for(int i=0;i<P;++i) XtX[i][i]+=lambda; // add ridge penalty
    std::array<f64,P> Xty{};
    for(int i=0;i<P;++i) for(int k=0;k<N;++k) Xty[i]+=X[k][i]*y[k];
    std::array<std::array<f64,P+1>,P> aug{};
    for(int i=0;i<P;++i){for(int j=0;j<P;++j)aug[i][j]=XtX[i][j];aug[i][P]=Xty[i];}
    for(int k=0;k<P;++k) {
        if(std::abs(aug[k][k])<1e-15) continue;
        for(int i=0;i<P;++i) if(i!=k) {
            f64 f=aug[i][k]/aug[k][k];
            for(int j=k;j<=P;++j) aug[i][j]-=f*aug[k][j];
        }
    }
    std::array<f64,P> beta{};
    for(int i=0;i<P;++i) if(std::abs(aug[i][i])>1e-15) beta[i]=aug[i][P]/aug[i][i];
    return beta;
}

// ─── LASSO (L1 regularization) via coordinate descent ────────────────────────
template<int N, int P>
[[nodiscard]] constexpr std::array<f64,P> lasso(
    const std::array<std::array<f64,P>,N>& X,
    const std::array<f64,N>& y, f64 lambda,
    int max_iter=1000, f64 tol=1e-6) noexcept
{
    std::array<f64,P> beta{};
    // Precompute X'X diagonal and X'y
    std::array<f64,P> XtX_diag{},Xty{};
    for(int j=0;j<P;++j) {
        for(int i=0;i<N;++i){XtX_diag[j]+=X[i][j]*X[i][j];Xty[j]+=X[i][j]*y[i];}
    }
    for(int iter=0;iter<max_iter;++iter) {
        f64 max_delta=0;
        for(int j=0;j<P;++j) {
            // Compute partial residual
            f64 rho=0;
            for(int i=0;i<N;++i) {
                f64 r=y[i]; for(int k=0;k<P;++k) if(k!=j) r-=X[i][k]*beta[k];
                rho+=X[i][j]*r;
            }
            f64 beta_new=0;
            if(XtX_diag[j]>1e-15) {
                f64 z=rho/XtX_diag[j];
                f64 lam_n=lambda/XtX_diag[j];
                beta_new=z>lam_n?z-lam_n:(z<-lam_n?z+lam_n:0);
            }
            max_delta=std::max(max_delta,std::abs(beta_new-beta[j]));
            beta[j]=beta_new;
        }
        if(max_delta<tol) break;
    }
    return beta;
}

// ─── Logistic regression (binary) ────────────────────────────────────────────
template<int N, int P>
[[nodiscard]] constexpr std::array<f64,P> logistic_regression(
    const std::array<std::array<f64,P>,N>& X,
    const std::array<f64,N>& y, f64 lr=0.01,
    int max_iter=1000, f64 tol=1e-6) noexcept
{
    std::array<f64,P> beta{};
    for(int iter=0;iter<max_iter;++iter) {
        std::array<f64,P> grad{};
        for(int i=0;i<N;++i) {
            f64 lin=0; for(int j=0;j<P;++j) lin+=X[i][j]*beta[j];
            f64 p=1.0/(1+std::exp(-lin));
            f64 err=p-y[i];
            for(int j=0;j<P;++j) grad[j]+=err*X[i][j];
        }
        f64 gnorm=0;
        for(int j=0;j<P;++j){grad[j]/=N;beta[j]-=lr*grad[j];gnorm+=grad[j]*grad[j];}
        if(gnorm<tol*tol) break;
    }
    return beta;
}

// ─── Polynomial regression basis expansion ────────────────────────────────────
template<int N, int DEG>
[[nodiscard]] constexpr std::array<std::array<f64,DEG+1>,N> poly_basis(
    const std::array<f64,N>& x) noexcept
{
    std::array<std::array<f64,DEG+1>,N> X{};
    for(int i=0;i<N;++i){X[i][0]=1;for(int d=1;d<=DEG;++d)X[i][d]=X[i][d-1]*x[i];}
    return X;
}

// ─── R-squared ────────────────────────────────────────────────────────────────
template<int N, int P>
[[nodiscard]] constexpr f64 r_squared(
    const std::array<std::array<f64,P>,N>& X,
    const std::array<f64,N>& y,
    const std::array<f64,P>& beta) noexcept
{
    f64 y_mean=0; for(auto v:y) y_mean+=v; y_mean/=N;
    f64 SS_tot=0,SS_res=0;
    for(int i=0;i<N;++i) {
        f64 pred=0; for(int j=0;j<P;++j) pred+=X[i][j]*beta[j];
        SS_tot+=(y[i]-y_mean)*(y[i]-y_mean);
        SS_res+=(y[i]-pred)*(y[i]-pred);
    }
    return SS_tot>1e-15?1-SS_res/SS_tot:0;
}

// ─── Cross-validation (k-fold) ────────────────────────────────────────────────
template<int N, int P, int K>
[[nodiscard]] constexpr f64 kfold_cv_mse(
    const std::array<std::array<f64,P>,N>& X,
    const std::array<f64,N>& y, f64 lambda=0) noexcept
{
    f64 total_mse=0;
    int fold_size=N/K;
    for(int fold=0;fold<K;++fold) {
        // Train on all except fold
        int train_n=N-fold_size;
        std::array<std::array<f64,P>,N> X_train{};
        std::array<f64,N> y_train{};
        int ti=0;
        for(int i=0;i<N;++i) if(i<fold*fold_size||i>=(fold+1)*fold_size)
            {X_train[ti]=X[i];y_train[ti++]=y[i];}
        // Fit (ridge for robustness)
        std::array<std::array<f64,P>,P> XtX{};
        for(int i=0;i<ti;++i) for(int a=0;a<P;++a) for(int b=0;b<P;++b) XtX[a][b]+=X_train[i][a]*X_train[i][b];
        std::array<f64,P> Xty2{};
        for(int i=0;i<ti;++i) for(int a=0;a<P;++a) Xty2[a]+=X_train[i][a]*y_train[i];
        for(int a=0;a<P;++a) XtX[a][a]+=lambda;
        std::array<f64,P> beta{};
        // Simple diag approx for speed
        for(int a=0;a<P;++a) if(XtX[a][a]>1e-15) beta[a]=Xty2[a]/XtX[a][a];
        // Eval on fold
        f64 mse=0;
        for(int i=fold*fold_size;i<(fold+1)*fold_size&&i<N;++i) {
            f64 pred=0; for(int j=0;j<P;++j) pred+=X[i][j]*beta[j];
            f64 d=y[i]-pred; mse+=d*d;
        }
        total_mse+=mse/fold_size;
    }
    return total_mse/K;
}

// ─── Gaussian process regression (exact, small N) ────────────────────────────
// RBF kernel: k(x,x') = σ²*exp(-|x-x'|²/(2l²))
template<int N>
[[nodiscard]] constexpr f64 rbf_kernel(f64 x1, f64 x2, f64 sigma=1.0, f64 l=1.0) noexcept {
    f64 d=x1-x2; return sigma*sigma*std::exp(-d*d/(2*l*l));
}
template<int N>
[[nodiscard]] constexpr std::pair<f64,f64> gp_predict(
    const std::array<f64,N>& x_train, const std::array<f64,N>& y_train,
    f64 x_star, f64 noise=0.01, f64 sigma=1.0, f64 l=1.0) noexcept
{
    // K matrix
    std::array<std::array<f64,N>,N> K{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) K[i][j]=rbf_kernel<N>(x_train[i],x_train[j],sigma,l);
    for(int i=0;i<N;++i) K[i][i]+=noise*noise;
    // k_star vector
    std::array<f64,N> k_star{};
    for(int i=0;i<N;++i) k_star[i]=rbf_kernel<N>(x_star,x_train[i],sigma,l);
    // Solve K*alpha=y via Gaussian elim
    std::array<std::array<f64,N+1>,N> aug{};
    for(int i=0;i<N;++i){for(int j=0;j<N;++j)aug[i][j]=K[i][j];aug[i][N]=y_train[i];}
    for(int k=0;k<N;++k) {
        if(std::abs(aug[k][k])<1e-15) continue;
        for(int i=0;i<N;++i) if(i!=k) {
            f64 f=aug[i][k]/aug[k][k];
            for(int j=k;j<=N;++j) aug[i][j]-=f*aug[k][j];
        }
    }
    std::array<f64,N> alpha{};
    for(int i=0;i<N;++i) if(std::abs(aug[i][i])>1e-15) alpha[i]=aug[i][N]/aug[i][i];
    // Mean: k_star' * alpha
    f64 mean=0; for(int i=0;i<N;++i) mean+=k_star[i]*alpha[i];
    // Variance: k_ss - k_star'*K^{-1}*k_star
    // Solve K*v=k_star
    for(int i=0;i<N;++i) aug[i][N]=k_star[i];
    for(int k=0;k<N;++k) if(std::abs(aug[k][k])>1e-15)
        for(int i=0;i<N;++i) if(i!=k) {
            f64 f=aug[i][k]/aug[k][k];
            for(int j=k;j<=N;++j) aug[i][j]-=f*aug[k][j];
        }
    std::array<f64,N> v{};
    for(int i=0;i<N;++i) if(std::abs(aug[i][i])>1e-15) v[i]=aug[i][N]/aug[i][i];
    f64 var=rbf_kernel<N>(x_star,x_star,sigma,l);
    for(int i=0;i<N;++i) var-=k_star[i]*v[i];
    return{mean,std::max(0.0,var)};
}
} // namespace qs::math
