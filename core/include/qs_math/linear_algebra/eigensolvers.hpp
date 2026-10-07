#pragma once
// qs_math/eigensolvers.hpp — Full Jacobi/QR algorithm for N×N, inverse iteration (C++20)
#include "scalar_types.hpp"
#include "linear_algebra.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>

namespace qs {

struct EigenResult {
    std::vector<f64> eigenvalues;
    DenseMatrix<f64> eigenvectors; // columns = eigenvectors
    u32 iterations{0};
    bool converged{false};
};

// ─── Jacobi eigenvalue algorithm (symmetric matrices) ─────────────────────────
// Repeatedly applies Givens rotations to zero off-diagonal elements
template<typename T>
[[nodiscard]] EigenResult jacobi_eigen(DenseMatrix<T> A, u32 max_iter=500, f64 tol=1e-10) {
    u32 n=A.rows; assert(A.cols==n);
    DenseMatrix<f64> Ad(n,n); for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) Ad.at(i,j)=f64(A.at(i,j));
    auto V=DenseMatrix<f64>::identity(n);
    EigenResult res;
    for(u32 iter=0;iter<max_iter;++iter) {
        // Find max off-diagonal element
        f64 max_off=0; u32 p=0,q=1;
        for(u32 i=0;i<n;++i) for(u32 j=i+1;j<n;++j) {
            f64 v=std::abs(Ad.at(i,j));
            if(v>max_off){max_off=v;p=i;q=j;}
        }
        if(max_off<tol){res.converged=true;res.iterations=iter;break;}
        // Compute Givens rotation angle
        f64 theta=0.5*(Ad.at(q,q)-Ad.at(p,p))/(Ad.at(p,q)+1e-300);
        f64 t=std::copysign(1.0/(std::abs(theta)+std::sqrt(1.0+theta*theta)),theta);
        f64 c=1.0/std::sqrt(1.0+t*t), s=t*c;
        // Apply rotation A' = G^T A G
        // Update rows p and q
        for(u32 i=0;i<n;++i) {
            if(i==p||i==q) continue;
            f64 aip=Ad.at(i,p), aiq=Ad.at(i,q);
            Ad.at(i,p)=Ad.at(p,i)=c*aip-s*aiq;
            Ad.at(i,q)=Ad.at(q,i)=s*aip+c*aiq;
        }
        f64 app=Ad.at(p,p),aqq=Ad.at(q,q),apq=Ad.at(p,q);
        Ad.at(p,p)=c*c*app-2*s*c*apq+s*s*aqq;
        Ad.at(q,q)=s*s*app+2*s*c*apq+c*c*aqq;
        Ad.at(p,q)=Ad.at(q,p)=0.0;
        // Accumulate eigenvectors
        for(u32 i=0;i<n;++i) {
            f64 vip=V.at(i,p), viq=V.at(i,q);
            V.at(i,p)=c*vip-s*viq;
            V.at(i,q)=s*vip+c*viq;
        }
        res.iterations=iter+1;
    }
    res.eigenvalues.resize(n);
    for(u32 i=0;i<n;++i) res.eigenvalues[i]=Ad.at(i,i);
    res.eigenvectors=V;
    // Sort by ascending eigenvalue
    std::vector<u32> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](u32 a,u32 b){return res.eigenvalues[a]<res.eigenvalues[b];});
    std::vector<f64> ev_sorted(n);
    DenseMatrix<f64> V_sorted(n,n);
    for(u32 i=0;i<n;++i){ ev_sorted[i]=res.eigenvalues[idx[i]]; for(u32 j=0;j<n;++j) V_sorted.at(j,i)=V.at(j,idx[i]); }
    res.eigenvalues=ev_sorted; res.eigenvectors=V_sorted;
    return res;
}

// ─── QR algorithm with Householder (general matrices, real eigenvalues) ────────
// Produces Schur decomposition; eigenvalues on diagonal of T
template<typename T>
[[nodiscard]] EigenResult qr_eigen(DenseMatrix<T> A, u32 max_iter=1000, f64 tol=1e-10) {
    u32 n=A.rows; assert(A.cols==n);
    DenseMatrix<f64> H(n,n);
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) H.at(i,j)=f64(A.at(i,j));
    EigenResult res; res.eigenvectors=DenseMatrix<f64>::identity(n);

    // Hessenberg reduction (Householder)
    for(u32 k=0;k+2<n;++k) {
        std::vector<f64> x(n-k-1);
        for(u32 i=0;i<u32(x.size());++i) x[i]=H.at(i+k+1,k);
        f64 xn=0; for(auto v:x) xn+=v*v; xn=std::sqrt(xn);
        if(xn<1e-14) continue;
        x[0]+=std::copysign(xn,x[0]);
        f64 xn2=0; for(auto v:x) xn2+=v*v;
        // Apply Householder: H = (I - 2vv^T/|v|^2) H (I - 2vv^T/|v|^2)
        for(u32 j=k;j<n;++j) {
            f64 dot=0; for(u32 i=0;i<u32(x.size());++i) dot+=x[i]*H.at(i+k+1,j);
            dot*=2.0/xn2;
            for(u32 i=0;i<u32(x.size());++i) H.at(i+k+1,j)-=dot*x[i];
        }
        for(u32 i=0;i<n;++i) {
            f64 dot=0; for(u32 j=0;j<u32(x.size());++j) dot+=H.at(i,j+k+1)*x[j];
            dot*=2.0/xn2;
            for(u32 j=0;j<u32(x.size());++j) H.at(i,j+k+1)-=dot*x[j];
        }
    }
    // QR iteration with Wilkinson shift
    u32 m=n;
    for(u32 iter=0;iter<max_iter&&m>1;++iter) {
        u32 p=m-1;
        // Wilkinson shift
        f64 d=(H.at(p-1,p-1)-H.at(p,p))*0.5;
        f64 shift=H.at(p,p)+d-std::copysign(std::sqrt(d*d+H.at(p,p-1)*H.at(p-1,p)),d);
        for(u32 i=0;i<m;++i) H.at(i,i)-=shift;
        // Single QR step via Givens
        for(u32 k=0;k+1<m;++k) {
            f64 a=H.at(k,k),b=H.at(k+1,k);
            f64 r=std::sqrt(a*a+b*b);
            if(r<1e-14) continue;
            f64 c=a/r,s=b/r;
            for(u32 j=k;j<m;++j) {
                f64 h0=H.at(k,j),h1=H.at(k+1,j);
                H.at(k,j)=c*h0+s*h1; H.at(k+1,j)=-s*h0+c*h1;
            }
            for(u32 i=0;i<=k+1&&i<m;++i) {
                f64 h0=H.at(i,k),h1=H.at(i,k+1);
                H.at(i,k)=c*h0+s*h1; H.at(i,k+1)=-s*h0+c*h1;
            }
        }
        for(u32 i=0;i<m;++i) H.at(i,i)+=shift;
        // Deflate converged eigenvalue
        if(std::abs(H.at(m-1,m-2))<tol*(std::abs(H.at(m-2,m-2))+std::abs(H.at(m-1,m-1)))) --m;
        res.iterations=iter+1;
    }
    res.eigenvalues.resize(n);
    for(u32 i=0;i<n;++i) res.eigenvalues[i]=H.at(i,i);
    res.converged=(m<=1);
    return res;
}

// ─── Inverse iteration (power method for specific eigenvalue) ─────────────────
template<typename T>
[[nodiscard]] std::pair<f64,std::vector<f64>> inverse_iteration(
    const DenseMatrix<T>& A, f64 mu=0.0, u32 max_iter=100, f64 tol=1e-10)
{
    u32 n=A.rows;
    // A - μI
    DenseMatrix<f64> B(n,n); for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) B.at(i,j)=f64(A.at(i,j));
    for(u32 i=0;i<n;++i) B.at(i,i)-=mu;
    // Random start vector
    std::vector<f64> x(n); for(u32 i=0;i<n;++i) x[i]=f64((i+1));
    f64 xn=vec_norm(x); for(auto& v:x) v/=xn;
    f64 lambda=mu;
    for(u32 it=0;it<max_iter;++it) {
        auto x_new_opt=solve(B,x);
        if(!x_new_opt) break;
        auto& x_new=*x_new_opt;
        f64 nm=vec_norm(x_new); if(nm<1e-30) break;
        for(auto& v:x_new) v/=nm;
        lambda=mu+1.0/nm;
        f64 diff=0; for(u32 i=0;i<n;++i) diff+=std::abs(x_new[i]-x[i]);
        x=x_new;
        if(diff<tol) break;
    }
    return {lambda,x};
}

// ─── Power iteration (dominant eigenvalue) ────────────────────────────────────
template<typename T>
[[nodiscard]] std::pair<f64,std::vector<f64>> power_iteration(
    const DenseMatrix<T>& A, u32 max_iter=500, f64 tol=1e-10)
{
    u32 n=A.rows;
    std::vector<f64> x(n,1.0/std::sqrt(f64(n)));
    f64 lambda=0;
    for(u32 it=0;it<max_iter;++it) {
        std::vector<f64> y(n,0);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) y[i]+=f64(A.at(i,j))*x[j];
        f64 lambda_new=vec_dot(x,y);
        f64 yn=vec_norm(y); if(yn<1e-30) break;
        for(auto& v:y) v/=yn;
        if(std::abs(lambda_new-lambda)<tol){lambda=lambda_new;x=y;break;}
        lambda=lambda_new; x=y;
    }
    return {lambda,x};
}

// Symmetric tridiagonal eigenvalues via bisection (fast for banded matrices)
[[nodiscard]] inline std::vector<f64> tridiag_eigen_bisection(
    const std::vector<f64>& diag, const std::vector<f64>& offdiag, f64 tol=1e-12)
{
    u32 n=u32(diag.size());
    // Gershgorin bounds
    f64 lo=diag[0]-std::abs(offdiag[0]);
    f64 hi=diag[0]+std::abs(offdiag[0]);
    for(u32 i=1;i<n-1;++i){ lo=std::min(lo,diag[i]-std::abs(offdiag[i-1])-std::abs(offdiag[i])); hi=std::max(hi,diag[i]+std::abs(offdiag[i-1])+std::abs(offdiag[i])); }
    lo=std::min(lo,diag[n-1]-std::abs(offdiag[n-2]));
    hi=std::max(hi,diag[n-1]+std::abs(offdiag[n-2]));
    lo-=1.0; hi+=1.0;
    // Count eigenvalues ≤ x via Sturm sequence
    auto sturm=[&](f64 x)->u32{
        u32 cnt=0; f64 d=diag[0]-x;
        if(d<0) ++cnt;
        for(u32 i=1;i<n;++i){
            f64 prev_d=d;
            d=(diag[i]-x)-offdiag[i-1]*offdiag[i-1]/(prev_d<1e-300?1e-300:prev_d);
            if(d<0) ++cnt;
        }
        return cnt;
    };
    std::vector<f64> evals;
    for(u32 k=0;k<n;++k) {
        f64 a=lo,b=hi;
        while(b-a>tol) {
            f64 m=0.5*(a+b);
            if(sturm(m)<=k) a=m; else b=m;
        }
        evals.push_back(0.5*(a+b));
    }
    return evals;
}

} // namespace qs
