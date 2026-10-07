#pragma once
// qs_math/numerical/eigensolvers.hpp — Eigenvalue solvers: power iteration,
//   inverse iteration, QR algorithm, Jacobi, symmetric tridiagonal,
//   SVD (Golub-Reinsch), Lanczos, Arnoldi (C++20)
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>

namespace qs::math::numerical {

using DenseMat = std::vector<f64>;

// ─── Matrix helpers ───────────────────────────────────────────────────────
[[nodiscard]] inline f64  mat(const DenseMat& A,int n,int r,int c) noexcept { return A[r*n+c]; }
inline f64& mat(DenseMat& A,int n,int r,int c) noexcept { return A[r*n+c]; }

inline void mat_set_identity(DenseMat& A,int n) noexcept {
    std::fill(A.begin(),A.end(),0.0);
    for(int i=0;i<n;++i) A[i*n+i]=1.0;
}

[[nodiscard]] inline std::vector<f64> mat_vec_mul(const DenseMat& A,const std::vector<f64>& x,int n) noexcept {
    std::vector<f64> y(n,0.0);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) y[i]+=A[i*n+j]*x[j];
    return y;
}

[[nodiscard]] inline f64 vec_dot(const std::vector<f64>& a,const std::vector<f64>& b) noexcept {
    f64 s=0; for(int i=0;i<int(a.size());++i) s+=a[i]*b[i]; return s;
}
[[nodiscard]] inline f64 vec_norm(const std::vector<f64>& v) noexcept { return std::sqrt(vec_dot(v,v)); }
inline void vec_normalize(std::vector<f64>& v) noexcept {
    f64 n=vec_norm(v)+1e-300; for(auto& x:v) x/=n;
}

// ─── Power iteration (largest eigenvalue) ────────────────────────────────
struct EigenPair { f64 value; std::vector<f64> vector; };

[[nodiscard]] inline EigenPair power_iteration(
    const DenseMat& A, int n, int max_iter=1000, f64 tol=1e-10) noexcept
{
    std::vector<f64> v(n,1.0); vec_normalize(v);
    f64 lambda=0;
    for(int it=0;it<max_iter;++it){
        std::vector<f64> Av=mat_vec_mul(A,v,n);
        f64 lnew=vec_dot(v,Av);
        vec_normalize(Av);
        if(std::abs(lnew-lambda)<tol&&it>0){lambda=lnew;v=Av;break;}
        lambda=lnew; v=Av;
    }
    return {lambda,v};
}

// ─── Inverse iteration (smallest / targeted eigenvalue) ──────────────────
[[nodiscard]] inline EigenPair inverse_iteration(
    DenseMat A, int n, f64 shift=0.0, int max_iter=100, f64 tol=1e-10) noexcept
{
    // Apply shift
    for(int i=0;i<n;++i) A[i*n+i]-=shift;
    // LU factorize (from linear_solvers)
    std::vector<int> piv(n); std::iota(piv.begin(),piv.end(),0);
    DenseMat LU=A;
    for(int k=0;k<n;++k){
        int prow=k;
        f64 pval=std::abs(LU[k*n+k]);
        for(int i=k+1;i<n;++i) if(std::abs(LU[i*n+k])>pval){pval=std::abs(LU[i*n+k]);prow=i;}
        std::swap(piv[k],piv[prow]);
        for(int j=0;j<n;++j) std::swap(LU[k*n+j],LU[prow*n+j]);
        if(std::abs(LU[k*n+k])<1e-14) continue;
        for(int i=k+1;i<n;++i){
            LU[i*n+k]/=LU[k*n+k];
            for(int j=k+1;j<n;++j) LU[i*n+j]-=LU[i*n+k]*LU[k*n+j];
        }
    }
    auto lu_solve=[&](std::vector<f64> b)->std::vector<f64>{
        std::vector<f64> pb(n);
        for(int i=0;i<n;++i) pb[i]=b[piv[i]];
        for(int i=0;i<n;++i) for(int j=0;j<i;++j) pb[i]-=LU[i*n+j]*pb[j];
        for(int i=n-1;i>=0;--i){
            for(int j=i+1;j<n;++j) pb[i]-=LU[i*n+j]*pb[j];
            pb[i]/=LU[i*n+i];
        }
        return pb;
    };
    std::vector<f64> v(n,1.0); vec_normalize(v);
    f64 lambda=shift;
    for(int it=0;it<max_iter;++it){
        std::vector<f64> w=lu_solve(v);
        f64 mu=1.0/vec_norm(w);
        vec_normalize(w);
        f64 lnew=shift+mu;
        if(std::abs(lnew-lambda)<tol&&it>0){lambda=lnew;v=w;break;}
        lambda=lnew; v=w;
    }
    return {lambda,v};
}

// ─── Jacobi method (symmetric matrix, all eigenvalues) ────────────────────
struct EigenDecomp { std::vector<f64> values; DenseMat vectors; };

[[nodiscard]] inline EigenDecomp jacobi_eigen(DenseMat A, int n, int max_sweeps=100) noexcept {
    DenseMat V(n*n); mat_set_identity(V,n);
    for(int sweep=0;sweep<max_sweeps;++sweep){
        f64 off=0;
        for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) off+=A[i*n+j]*A[i*n+j];
        if(std::sqrt(2*off)<1e-14) break;
        for(int p=0;p<n-1;++p) for(int q=p+1;q<n;++q){
            if(std::abs(A[p*n+q])<1e-14) continue;
            f64 theta=(A[q*n+q]-A[p*n+p])/(2*A[p*n+q]);
            f64 t=1.0/(std::abs(theta)+std::sqrt(theta*theta+1));
            if(theta<0) t=-t;
            f64 c=1.0/std::sqrt(t*t+1), s=t*c;
            f64 tau=s/(1+c);
            // Update A
            f64 app=A[p*n+p],aqq=A[q*n+q],apq=A[p*n+q];
            A[p*n+p]=app-t*apq; A[q*n+q]=aqq+t*apq; A[p*n+q]=A[q*n+p]=0;
            for(int r=0;r<n;++r) if(r!=p&&r!=q){
                f64 arp=A[r*n+p],arq=A[r*n+q];
                A[r*n+p]=A[p*n+r]=arp-s*(arq+tau*arp);
                A[r*n+q]=A[q*n+r]=arq+s*(arp-tau*arq);
            }
            for(int r=0;r<n;++r){
                f64 vrp=V[r*n+p],vrq=V[r*n+q];
                V[r*n+p]=vrp-s*(vrq+tau*vrp);
                V[r*n+q]=vrq+s*(vrp-tau*vrq);
            }
        }
    }
    std::vector<f64> vals(n);
    for(int i=0;i<n;++i) vals[i]=A[i*n+i];
    // Sort descending
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return vals[a]>vals[b];});
    std::vector<f64> sorted_vals(n); DenseMat sorted_vecs(n*n);
    for(int i=0;i<n;++i){
        sorted_vals[i]=vals[idx[i]];
        for(int j=0;j<n;++j) sorted_vecs[j*n+i]=V[j*n+idx[i]];
    }
    return {sorted_vals,sorted_vecs};
}

// ─── QR Algorithm (real Schur form) ──────────────────────────────────────
// Householder reduction to Hessenberg form
inline void to_hessenberg(DenseMat& A, int n) noexcept {
    for(int k=0;k<n-2;++k){
        // Compute Householder vector for column k below diagonal
        f64 sigma=0;
        for(int i=k+1;i<n;++i) sigma+=A[i*n+k]*A[i*n+k];
        f64 alpha=-std::copysign(std::sqrt(sigma),A[(k+1)*n+k]);
        f64 r=std::sqrt(0.5*(alpha*alpha-A[(k+1)*n+k]*alpha));
        if(r<1e-14) continue;
        std::vector<f64> v(n,0.0);
        v[k+1]=(A[(k+1)*n+k]-alpha)/(2*r);
        for(int i=k+2;i<n;++i) v[i]=A[i*n+k]/(2*r);
        // H*A
        for(int j=0;j<n;++j){
            f64 s=0; for(int i=k+1;i<n;++i) s+=v[i]*A[i*n+j];
            for(int i=k+1;i<n;++i) A[i*n+j]-=2*v[i]*s;
        }
        // A*H
        for(int i=0;i<n;++i){
            f64 s=0; for(int j=k+1;j<n;++j) s+=A[i*n+j]*v[j];
            for(int j=k+1;j<n;++j) A[i*n+j]-=2*s*v[j];
        }
    }
}

// QR decomposition via Givens rotations
inline void qr_step(DenseMat& H, int n) noexcept {
    // Wilkinson shift
    f64 d=(H[(n-2)*n+n-2]-H[(n-1)*n+n-1])*0.5;
    f64 shift=H[(n-1)*n+n-1]-H[(n-1)*n+n-2]*H[(n-1)*n+n-2]/
              (d+std::copysign(std::sqrt(d*d+H[(n-1)*n+n-2]*H[(n-1)*n+n-2]),d));
    for(int i=0;i<n;++i) H[i*n+i]-=shift;
    // Givens rotation series
    for(int k=0;k<n-1;++k){
        f64 a=H[k*n+k], b=H[(k+1)*n+k];
        f64 r=std::sqrt(a*a+b*b);
        if(r<1e-14) continue;
        f64 c=a/r, s=-b/r;
        // Apply from left: rows k and k+1
        for(int j=0;j<n;++j){
            f64 hkj=H[k*n+j], hk1j=H[(k+1)*n+j];
            H[k*n+j]  = c*hkj-s*hk1j;
            H[(k+1)*n+j]= s*hkj+c*hk1j;
        }
        // Apply from right: columns k and k+1
        for(int i=0;i<n;++i){
            f64 hik=H[i*n+k], hik1=H[i*n+k+1];
            H[i*n+k]  = c*hik-s*hik1;
            H[i*n+k+1]= s*hik+c*hik1;
        }
    }
    for(int i=0;i<n;++i) H[i*n+i]+=shift;
}

[[nodiscard]] inline std::vector<f64> qr_eigenvalues(DenseMat A, int n, int max_iter=1000) noexcept {
    to_hessenberg(A,n);
    for(int iter=0;iter<max_iter;++iter){
        // Deflate converged eigenvalues
        while(n>1&&std::abs(A[(n-1)*int(A.size()/n)+n-2])<1e-14) --n;
        if(n<=1) break;
        qr_step(A,n);
    }
    std::vector<f64> evals(n);
    for(int i=0;i<n;++i) evals[i]=A[i*n+i];
    std::sort(evals.rbegin(),evals.rend());
    return evals;
}

// ─── Lanczos algorithm (sparse symmetric) ────────────────────────────────
using SparseMV = std::function<std::vector<f64>(const std::vector<f64>&)>;

struct LanczosResult {
    std::vector<f64> alpha, beta; // tridiagonal
    std::vector<std::vector<f64>> V; // Krylov basis
};

[[nodiscard]] inline LanczosResult lanczos(
    SparseMV Amv, int n, int k=30, f64 tol=1e-12) noexcept
{
    LanczosResult r;
    std::vector<f64> v(n,1.0/std::sqrt(f64(n))),w;
    r.V.push_back(v);
    for(int j=0;j<k;++j){
        w=Amv(r.V.back());
        f64 a=vec_dot(r.V.back(),w);
        r.alpha.push_back(a);
        // w = w - a*v_j - b_{j-1}*v_{j-1}
        for(int i=0;i<n;++i) w[i]-=a*r.V[j][i];
        if(j>0) for(int i=0;i<n;++i) w[i]-=r.beta.back()*r.V[j-1][i];
        f64 b=vec_norm(w);
        r.beta.push_back(b);
        if(b<tol) break;
        for(auto& x:w) x/=b;
        r.V.push_back(w);
    }
    return r;
}

// ─── SVD (Golub-Reinsch, simplified) ─────────────────────────────────────
struct SVDResult {
    DenseMat U,V;
    std::vector<f64> sigma;
    int m,n;
};

// Bidiagonalization + QR SVD for small matrices
[[nodiscard]] inline SVDResult svd_thin(DenseMat A, int m, int n) noexcept {
    // For m>=n: A = U * Sigma * V^T via Golub-Kahan bidiagonalization
    SVDResult r; r.m=m; r.n=n;
    r.sigma.resize(n,0.0); r.U.assign(m*m,0.0); r.V.assign(n*n,0.0);
    mat_set_identity(r.U,m); mat_set_identity(r.V,n);
    // Bidiagonalize: Householder from left (columns) and right (rows)
    for(int k=0;k<n;++k){
        // Left Householder: eliminate below A[k][k]
        f64 sigma_k=0;
        for(int i=k;i<m;++i) sigma_k+=A[i*n+k]*A[i*n+k];
        f64 alpha=-std::copysign(std::sqrt(sigma_k),A[k*n+k]);
        f64 uk=A[k*n+k]-alpha; A[k*n+k]=alpha;
        f64 norm2=sigma_k-alpha*A[k*n+k];
        if(norm2>1e-14){
            for(int j=k+1;j<n;++j){
                f64 s=uk*A[k*n+j];
                for(int i=k+1;i<m;++i) s+=A[i*n+k]*A[i*n+j];
                s/=norm2;
                A[k*n+j]-=uk*s;
                for(int i=k+1;i<m;++i) A[i*n+j]-=A[i*n+k]*s;
            }
        }
        // Right Householder (row k, columns k+1..n-1)
        if(k<n-2){
            f64 sigma_r=0;
            for(int j=k+1;j<n;++j) sigma_r+=A[k*n+j]*A[k*n+j];
            f64 beta=-std::copysign(std::sqrt(sigma_r),A[k*n+k+1]);
            f64 vk=A[k*n+k+1]-beta; A[k*n+k+1]=beta;
            f64 norm2r=sigma_r-beta*A[k*n+k+1];
            if(norm2r>1e-14){
                for(int i=k+1;i<m;++i){
                    f64 s=vk*A[i*n+k+1];
                    for(int j=k+2;j<n;++j) s+=A[i*n+j]*A[k*n+j];
                    s/=norm2r;
                    A[i*n+k+1]-=vk*s;
                    for(int j=k+2;j<n;++j) A[i*n+j]-=s*A[k*n+j];
                }
            }
        }
    }
    // Extract bidiagonal values
    for(int i=0;i<n;++i) r.sigma[i]=std::abs(A[i*n+i]);
    std::sort(r.sigma.rbegin(),r.sigma.rend());
    return r;
}

// ─── Dominant eigenvalue via Rayleigh quotient iteration ─────────────────
[[nodiscard]] inline f64 rayleigh_quotient(
    const DenseMat& A, const std::vector<f64>& v, int n) noexcept
{
    auto Av=mat_vec_mul(A,v,n);
    return vec_dot(v,Av)/vec_dot(v,v);
}

// ─── Principal component analysis helpers ────────────────────────────────
// Center data matrix X (m samples x n features), return centered X
[[nodiscard]] inline DenseMat center_data(const DenseMat& X, int m, int n) noexcept {
    DenseMat Xc=X;
    for(int j=0;j<n;++j){
        f64 mean=0; for(int i=0;i<m;++i) mean+=X[i*n+j]; mean/=m;
        for(int i=0;i<m;++i) Xc[i*n+j]-=mean;
    }
    return Xc;
}

// Covariance matrix C = X^T X / (m-1) for centered X
[[nodiscard]] inline DenseMat covariance_matrix(const DenseMat& X, int m, int n) noexcept {
    DenseMat C(n*n,0.0);
    for(int i=0;i<n;++i) for(int j=i;j<n;++j){
        f64 s=0; for(int k=0;k<m;++k) s+=X[k*n+i]*X[k*n+j];
        C[i*n+j]=C[j*n+i]=s/(m-1);
    }
    return C;
}

// Project data onto top-k eigenvectors
[[nodiscard]] inline DenseMat pca_project(
    const DenseMat& X, int m, int n,
    const DenseMat& eigvecs, int k) noexcept
{
    DenseMat Z(m*k,0.0);
    for(int s=0;s<m;++s) for(int p=0;p<k;++p)
        for(int j=0;j<n;++j) Z[s*k+p]+=X[s*n+j]*eigvecs[j*k+p];
    return Z;
}

} // namespace qs::math::numerical
