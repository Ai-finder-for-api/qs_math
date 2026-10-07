#pragma once
// qs_math/numerical/linear_solvers.hpp — Dense & sparse linear solvers:
//   Gaussian elimination, LU, Cholesky, QR (Householder), LDLT, CG, GMRES,
//   iterative refinement, least squares (C++20)
#include "../arithmetic/scalar_types.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <span>

namespace qs::math::numerical {

// ─── Dense matrix helpers (row-major, n×n) ───────────────────────────────
using DenseMat = std::vector<f64>;

[[nodiscard]] inline f64& mat_at(DenseMat& A, int n, int r, int c) noexcept { return A[r*n+c]; }
[[nodiscard]] inline f64  mat_at(const DenseMat& A, int n, int r, int c) noexcept { return A[r*n+c]; }

inline void mat_swap_rows(DenseMat& A, int n, int r1, int r2) noexcept {
    for (int j=0;j<n;++j) std::swap(A[r1*n+j], A[r2*n+j]);
}

[[nodiscard]] inline DenseMat mat_identity(int n) noexcept {
    DenseMat I(n*n, 0.0);
    for (int i=0;i<n;++i) I[i*n+i]=1.0;
    return I;
}

[[nodiscard]] inline DenseMat mat_multiply(const DenseMat& A, const DenseMat& B, int n) noexcept {
    DenseMat C(n*n, 0.0);
    for (int i=0;i<n;++i)
        for (int k=0;k<n;++k)
            if (A[i*n+k]!=0.0)
                for (int j=0;j<n;++j)
                    C[i*n+j]+=A[i*n+k]*B[k*n+j];
    return C;
}

[[nodiscard]] inline std::vector<f64> mat_vec(const DenseMat& A, const std::vector<f64>& x, int n) noexcept {
    std::vector<f64> y(n,0.0);
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            y[i]+=A[i*n+j]*x[j];
    return y;
}

// ─── LU decomposition with partial pivoting ───────────────────────────────
struct LUResult {
    DenseMat LU;          // in-place L/U
    std::vector<int> piv; // pivot vector
    int n;
    bool singular;
};

[[nodiscard]] inline LUResult lu_factor(DenseMat A, int n) noexcept {
    LUResult r; r.n=n; r.LU=std::move(A);
    r.piv.resize(n); std::iota(r.piv.begin(),r.piv.end(),0);
    r.singular=false;
    auto& LU=r.LU;
    for (int k=0;k<n;++k){
        // Find pivot
        int prow=k;
        f64 pval=std::abs(LU[k*n+k]);
        for (int i=k+1;i<n;++i)
            if (std::abs(LU[i*n+k])>pval){pval=std::abs(LU[i*n+k]);prow=i;}
        if (pval<1e-14){r.singular=true;}
        std::swap(r.piv[k],r.piv[prow]);
        mat_swap_rows(LU,n,k,prow);
        f64 diag=LU[k*n+k];
        if (std::abs(diag)<1e-300) continue;
        for (int i=k+1;i<n;++i){
            LU[i*n+k]/=diag;
            for (int j=k+1;j<n;++j)
                LU[i*n+j]-=LU[i*n+k]*LU[k*n+j];
        }
    }
    return r;
}

[[nodiscard]] inline std::vector<f64> lu_solve(const LUResult& lu, std::vector<f64> b) noexcept {
    int n=lu.n; auto& LU=lu.LU;
    // Apply pivot
    std::vector<f64> pb(n);
    for (int i=0;i<n;++i) pb[i]=b[lu.piv[i]];
    // Forward sub (L)
    for (int i=0;i<n;++i)
        for (int j=0;j<i;++j) pb[i]-=LU[i*n+j]*pb[j];
    // Back sub (U)
    for (int i=n-1;i>=0;--i){
        for (int j=i+1;j<n;++j) pb[i]-=LU[i*n+j]*pb[j];
        pb[i]/=LU[i*n+i];
    }
    return pb;
}

[[nodiscard]] inline DenseMat lu_inverse(const LUResult& lu) noexcept {
    int n=lu.n; DenseMat inv(n*n);
    for (int j=0;j<n;++j){
        std::vector<f64> ej(n,0.0); ej[j]=1.0;
        auto col=lu_solve(lu,ej);
        for (int i=0;i<n;++i) inv[i*n+j]=col[i];
    }
    return inv;
}

[[nodiscard]] inline f64 lu_determinant(const LUResult& lu) noexcept {
    f64 det=1.0;
    for (int i=0;i<lu.n;++i) det*=lu.LU[i*lu.n+i];
    // Count pivot inversions
    int swaps=0;
    std::vector<int> p=lu.piv;
    for (int i=0;i<lu.n;++i)
        while (p[i]!=i){std::swap(p[i],p[p[i]]);++swaps;}
    return (swaps%2==0)?det:-det;
}

// ─── Cholesky decomposition (symmetric positive definite) ─────────────────
struct CholeskyResult {
    DenseMat L; // lower triangular
    int n;
    bool ok;
};

[[nodiscard]] inline CholeskyResult cholesky(DenseMat A, int n) noexcept {
    CholeskyResult r; r.n=n; r.L.assign(n*n,0.0); r.ok=true;
    for (int j=0;j<n;++j){
        f64 s=A[j*n+j];
        for (int k=0;k<j;++k) s-=r.L[j*n+k]*r.L[j*n+k];
        if (s<0.0){r.ok=false;return r;}
        r.L[j*n+j]=std::sqrt(s);
        for (int i=j+1;i<n;++i){
            f64 t=A[i*n+j];
            for (int k=0;k<j;++k) t-=r.L[i*n+k]*r.L[j*n+k];
            r.L[i*n+j]=t/r.L[j*n+j];
        }
    }
    return r;
}

[[nodiscard]] inline std::vector<f64> cholesky_solve(const CholeskyResult& ch, std::vector<f64> b) noexcept {
    int n=ch.n; auto& L=ch.L;
    // Forward sub L*y=b
    for (int i=0;i<n;++i){
        for (int j=0;j<i;++j) b[i]-=L[i*n+j]*b[j];
        b[i]/=L[i*n+i];
    }
    // Back sub L^T*x=y
    for (int i=n-1;i>=0;--i){
        for (int j=i+1;j<n;++j) b[i]-=L[j*n+i]*b[j];
        b[i]/=L[i*n+i];
    }
    return b;
}

// ─── LDLT decomposition ───────────────────────────────────────────────────
struct LDLTResult {
    std::vector<f64> L; // unit lower triangular (stored as n×n)
    std::vector<f64> D; // diagonal
    int n; bool ok;
};

[[nodiscard]] inline LDLTResult ldlt(DenseMat A, int n) noexcept {
    LDLTResult r; r.n=n; r.ok=true;
    r.L=mat_identity(n); r.D.assign(n,0.0);
    std::vector<f64> v(n);
    for (int j=0;j<n;++j){
        for (int i=0;i<j;++i) v[i]=r.L[j*n+i]*r.D[i];
        r.D[j]=A[j*n+j];
        for (int i=0;i<j;++i) r.D[j]-=r.L[j*n+i]*v[i];
        if (std::abs(r.D[j])<1e-300){r.ok=false;return r;}
        for (int i=j+1;i<n;++i){
            f64 s=A[i*n+j];
            for (int k=0;k<j;++k) s-=r.L[i*n+k]*v[k];
            r.L[i*n+j]=s/r.D[j];
        }
    }
    return r;
}

// ─── Householder QR decomposition ─────────────────────────────────────────
struct QRResult {
    DenseMat Q, R; // m×m and m×n
    int m, n;
};

[[nodiscard]] inline QRResult qr_householder(DenseMat A, int m, int n) noexcept {
    QRResult r; r.m=m; r.n=n;
    r.Q=mat_identity(m);
    r.R=std::move(A);
    std::vector<f64> v(m);

    for (int k=0;k<std::min(m-1,n);++k){
        // Compute Householder vector for column k
        f64 sigma=0.0;
        for (int i=k;i<m;++i) sigma+=r.R[i*n+k]*r.R[i*n+k];
        f64 alpha=std::sqrt(sigma)*(r.R[k*n+k]>0?-1.0:1.0);
        for (int i=0;i<m;++i) v[i]=(i==k)?(r.R[k*n+k]-alpha):((i>k)?r.R[i*n+k]:0.0);
        f64 vTv=0.0; for(int i=k;i<m;++i) vTv+=v[i]*v[i];
        if (vTv<1e-30) continue;
        f64 tau=2.0/vTv;
        // Apply H=I-tau*v*v^T to R (columns k..n-1)
        for (int j=k;j<n;++j){
            f64 vTr=0.0; for(int i=k;i<m;++i) vTr+=v[i]*r.R[i*n+j];
            for(int i=k;i<m;++i) r.R[i*n+j]-=tau*v[i]*vTr;
        }
        // Apply H to Q (columns 0..m-1)
        for (int j=0;j<m;++j){
            f64 vTq=0.0; for(int i=k;i<m;++i) vTq+=v[i]*r.Q[i*m+j];
            for(int i=k;i<m;++i) r.Q[i*m+j]-=tau*v[i]*vTq;
        }
    }
    // Q is currently the product of Householder reflections applied to I; transpose to get Q
    DenseMat Qt(m*m);
    for(int i=0;i<m;++i) for(int j=0;j<m;++j) Qt[i*m+j]=r.Q[j*m+i];
    r.Q=std::move(Qt);
    return r;
}

// Solve min||Ax-b|| via QR (least squares)
[[nodiscard]] inline std::vector<f64> qr_least_squares(const QRResult& qr, std::vector<f64> b) noexcept {
    int m=qr.m, n=qr.n;
    // c = Q^T b
    std::vector<f64> c(m,0.0);
    for(int i=0;i<m;++i) for(int j=0;j<m;++j) c[i]+=qr.Q[j*m+i]*b[j];
    // Back solve R*x = c[0..n-1]
    std::vector<f64> x(n);
    for(int i=n-1;i>=0;--i){
        x[i]=c[i];
        for(int j=i+1;j<n;++j) x[i]-=qr.R[i*n+j]*x[j];
        x[i]/=qr.R[i*n+i];
    }
    return x;
}

// ─── Conjugate Gradient (symmetric positive definite sparse systems) ───────
using SparseMV = std::function<std::vector<f64>(const std::vector<f64>&)>;

struct IterativeSolveResult {
    std::vector<f64> x;
    int iterations;
    f64 residual_norm;
    bool converged;
};

[[nodiscard]] inline IterativeSolveResult conjugate_gradient(
    SparseMV Amv, std::vector<f64> b, std::vector<f64> x0={},
    f64 tol=1e-10, int max_iter=1000) noexcept
{
    int n=int(b.size());
    std::vector<f64> x=x0.empty()?std::vector<f64>(n,0.0):x0;
    // r = b - A*x
    auto Ax=Amv(x);
    std::vector<f64> r(n),p(n);
    f64 rr=0.0;
    for(int i=0;i<n;++i){r[i]=b[i]-Ax[i];p[i]=r[i];rr+=r[i]*r[i];}
    IterativeSolveResult res;
    for(int it=0;it<max_iter;++it){
        auto Ap=Amv(p);
        f64 pAp=0.0; for(int i=0;i<n;++i) pAp+=p[i]*Ap[i];
        if(std::abs(pAp)<1e-300) break;
        f64 alpha=rr/pAp;
        f64 rr_new=0.0;
        for(int i=0;i<n;++i){x[i]+=alpha*p[i]; r[i]-=alpha*Ap[i]; rr_new+=r[i]*r[i];}
        res.residual_norm=std::sqrt(rr_new);
        if(res.residual_norm<tol){res.converged=true;res.iterations=it+1;break;}
        f64 beta=rr_new/rr; rr=rr_new;
        for(int i=0;i<n;++i) p[i]=r[i]+beta*p[i];
    }
    res.x=std::move(x);
    return res;
}

// ─── GMRES (general sparse, non-symmetric) ────────────────────────────────
[[nodiscard]] inline IterativeSolveResult gmres(
    SparseMV Amv, std::vector<f64> b, std::vector<f64> x0={},
    f64 tol=1e-10, int max_iter=200, int restart=50) noexcept
{
    int n=int(b.size());
    std::vector<f64> x=x0.empty()?std::vector<f64>(n,0.0):x0;
    IterativeSolveResult res; res.converged=false;

    for(int outer=0;outer<max_iter/restart;++outer){
        auto r0=Amv(x);
        f64 beta=0.0;
        for(int i=0;i<n;++i){r0[i]=b[i]-r0[i];beta+=r0[i]*r0[i];}
        beta=std::sqrt(beta);
        if(beta<tol){res.converged=true;break;}

        // Arnoldi basis
        int m=std::min(restart,n);
        std::vector<std::vector<f64>> V(m+1,std::vector<f64>(n,0.0));
        for(int i=0;i<n;++i) V[0][i]=r0[i]/beta;
        std::vector<std::vector<f64>> H(m+1,std::vector<f64>(m,0.0));
        std::vector<f64> g(m+1,0.0); g[0]=beta;
        std::vector<f64> cs(m),sn(m);

        int j=0;
        for(;j<m;++j){
            auto w=Amv(V[j]);
            for(int i=0;i<=j;++i){
                H[i][j]=0.0; for(int k=0;k<n;++k) H[i][j]+=w[k]*V[i][k];
                for(int k=0;k<n;++k) w[k]-=H[i][j]*V[i][k];
            }
            H[j+1][j]=0.0; for(int k=0;k<n;++k) H[j+1][j]+=w[k]*w[k];
            H[j+1][j]=std::sqrt(H[j+1][j]);
            if(H[j+1][j]>1e-300) for(int k=0;k<n;++k) V[j+1][k]=w[k]/H[j+1][j];
            // Apply previous Givens rotations
            for(int i=0;i<j;++i){
                f64 tmp=cs[i]*H[i][j]+sn[i]*H[i+1][j];
                H[i+1][j]=-sn[i]*H[i][j]+cs[i]*H[i+1][j];
                H[i][j]=tmp;
            }
            f64 rho=std::sqrt(H[j][j]*H[j][j]+H[j+1][j]*H[j+1][j]);
            cs[j]=H[j][j]/rho; sn[j]=H[j+1][j]/rho;
            H[j][j]=rho; H[j+1][j]=0.0;
            g[j+1]=-sn[j]*g[j]; g[j]=cs[j]*g[j];
            res.residual_norm=std::abs(g[j+1]);
            if(res.residual_norm<tol){res.converged=true;++j;break;}
        }
        // Back solve upper triangular H[0..j-1][0..j-1]*y=g[0..j-1]
        std::vector<f64> y(j);
        for(int i=j-1;i>=0;--i){
            y[i]=g[i];
            for(int k=i+1;k<j;++k) y[i]-=H[i][k]*y[k];
            y[i]/=H[i][i];
        }
        // x = x + V[:,0..j-1]*y
        for(int i=0;i<j;++i) for(int k=0;k<n;++k) x[k]+=y[i]*V[i][k];
        if(res.converged) break;
    }
    res.x=std::move(x); res.iterations=max_iter;
    return res;
}

// ─── Iterative refinement ─────────────────────────────────────────────────
[[nodiscard]] inline std::vector<f64> iterative_refinement(
    const DenseMat& A, const LUResult& lu,
    std::vector<f64> b, int n, int iters=3) noexcept
{
    auto x=lu_solve(lu,b);
    for(int it=0;it<iters;++it){
        auto r=mat_vec(A,x,n);
        for(int i=0;i<n;++i) r[i]=b[i]-r[i];
        auto dx=lu_solve(lu,r);
        for(int i=0;i<n;++i) x[i]+=dx[i];
    }
    return x;
}

// ─── Gauss-Seidel iteration ───────────────────────────────────────────────
[[nodiscard]] inline IterativeSolveResult gauss_seidel(
    const DenseMat& A, std::vector<f64> b, int n,
    f64 tol=1e-10, int max_iter=1000) noexcept
{
    std::vector<f64> x(n,0.0);
    IterativeSolveResult res; res.converged=false;
    for(int it=0;it<max_iter;++it){
        f64 norm=0.0;
        for(int i=0;i<n;++i){
            f64 s=b[i];
            for(int j=0;j<n;++j) if(j!=i) s-=A[i*n+j]*x[j];
            f64 xi_new=s/A[i*n+i];
            norm+=std::abs(xi_new-x[i]);
            x[i]=xi_new;
        }
        if(norm<tol){res.converged=true;res.iterations=it+1;res.residual_norm=norm;break;}
    }
    res.x=std::move(x); return res;
}

// ─── Tridiagonal solver (Thomas algorithm) ────────────────────────────────
[[nodiscard]] inline std::vector<f64> tridiagonal_solve(
    std::vector<f64> a, // sub-diagonal (n-1)
    std::vector<f64> b, // main diagonal (n)
    std::vector<f64> c, // super-diagonal (n-1)
    std::vector<f64> d) noexcept // RHS (n)
{
    int n=int(b.size());
    std::vector<f64> cp(n-1),dp(n);
    cp[0]=c[0]/b[0]; dp[0]=d[0]/b[0];
    for(int i=1;i<n;++i){
        f64 m=b[i]-a[i-1]*cp[i-1];
        if(i<n-1) cp[i]=c[i]/m;
        dp[i]=(d[i]-a[i-1]*dp[i-1])/m;
    }
    std::vector<f64> x(n); x[n-1]=dp[n-1];
    for(int i=n-2;i>=0;--i) x[i]=dp[i]-cp[i]*x[i+1];
    return x;
}

} // namespace qs::math::numerical
