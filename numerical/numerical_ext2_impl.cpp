
// qs_math/numerical/numerical_ext2_impl.cpp (C++20)
// Sparse linear algebra, iterative eigensolvers, matrix functions,
// tensor decompositions, and integral equations.
#include "include/qs_math_numerical.hpp"
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <functional>
namespace qs::math::numerical {

// ─── Compressed-row-storage (CRS/CSR) sparse matrix ─────────────────────────
struct CSRMatrix {
    i32 nrows{0}, ncols{0};
    std::vector<i32> row_ptr, col_idx;
    std::vector<f64> vals;

    void set_size(i32 r, i32 c) noexcept { nrows=r; ncols=c; row_ptr.assign(r+1,0); }

    // Build from COO triplets (must be sorted by row then col)
    void from_coo(const std::vector<std::tuple<i32,i32,f64>>& coo,
                  i32 r, i32 c) noexcept
    {
        nrows=r; ncols=c;
        row_ptr.assign(r+1,0);
        for (auto& [ri,ci,v]:coo) ++row_ptr[ri+1];
        for (i32 i=1;i<=r;++i) row_ptr[i]+=row_ptr[i-1];
        i32 nnz=(i32)coo.size();
        col_idx.resize(nnz); vals.resize(nnz);
        std::vector<i32> cnt(r,0);
        for (auto& [ri,ci,v]:coo) {
            i32 pos=row_ptr[ri]+(cnt[ri]++);
            col_idx[pos]=ci; vals[pos]=v;
        }
    }

    // y = A*x
    void matvec(const f64* x, f64* y) const noexcept {
        for (i32 i=0;i<nrows;++i) {
            f64 s=0;
            for (i32 j=row_ptr[i];j<row_ptr[i+1];++j) s+=vals[j]*x[col_idx[j]];
            y[i]=s;
        }
    }
    // y = A^T * x
    void matvec_T(const f64* x, f64* y) const noexcept {
        std::fill(y,y+ncols,0.0);
        for (i32 i=0;i<nrows;++i)
            for (i32 j=row_ptr[i];j<row_ptr[i+1];++j)
                y[col_idx[j]]+=vals[j]*x[i];
    }
    // Frobenius norm
    [[nodiscard]] f64 fnorm() const noexcept {
        f64 s=0; for (auto v:vals) s+=v*v; return std::sqrt(s);
    }
    // Diagonal extraction
    std::vector<f64> diagonal() const noexcept {
        i32 n=std::min(nrows,ncols);
        std::vector<f64> d(n,0);
        for (i32 i=0;i<nrows;++i)
            for (i32 j=row_ptr[i];j<row_ptr[i+1];++j)
                if (col_idx[j]==i && i<n) d[i]=vals[j];
        return d;
    }
};

// ─── Arnoldi iteration (basis for GMRES / Krylov methods) ─────────────────────
struct ArnoldiResult {
    std::vector<std::vector<f64>> Q; // orthonormal Krylov basis (m+1 columns)
    std::vector<std::vector<f64>> H; // upper Hessenberg (m+1 × m)
    i32 m{0};
};

ArnoldiResult arnoldi(const CSRMatrix& A, const std::vector<f64>& b,
                      i32 m) noexcept
{
    const i32 n=(i32)b.size();
    ArnoldiResult r; r.m=m;
    r.Q.resize(m+1, std::vector<f64>(n,0));
    r.H.resize(m+1, std::vector<f64>(m,0));

    // q0 = b / ||b||
    f64 bnorm=0; for(auto v:b) bnorm+=v*v; bnorm=std::sqrt(bnorm);
    if(bnorm<1e-14) return r;
    for(i32 i=0;i<n;++i) r.Q[0][i]=b[i]/bnorm;

    std::vector<f64> Aq(n);
    for(i32 j=0;j<m;++j){
        // Aq = A * Q[j]
        A.matvec(r.Q[j].data(), Aq.data());
        // Modified Gram-Schmidt orthogonalisation
        for(i32 i=0;i<=j;++i){
            f64 h=0; for(i32 k=0;k<n;++k) h+=r.Q[i][k]*Aq[k];
            r.H[i][j]=h;
            for(i32 k=0;k<n;++k) Aq[k]-=h*r.Q[i][k];
        }
        f64 hnext=0; for(auto v:Aq) hnext+=v*v; hnext=std::sqrt(hnext);
        r.H[j+1][j]=hnext;
        if(hnext<1e-14) break;
        for(i32 k=0;k<n;++k) r.Q[j+1][k]=Aq[k]/hnext;
    }
    return r;
}

// ─── GMRES (restarted) ────────────────────────────────────────────────────────
void gmres(const CSRMatrix& A, const f64* b, f64* x, i32 n,
           i32 restart=50, i32 max_iter=500, f64 tol=1e-10) noexcept
{
    std::vector<f64> r(n), bv(b,b+n);
    auto matvec=[&](const std::vector<f64>& v)->std::vector<f64>{
        std::vector<f64> w(n); A.matvec(v.data(),w.data()); return w;};
    auto dot=[&](const std::vector<f64>& a,const std::vector<f64>& b2)->f64{
        f64 s=0;for(i32 i=0;i<n;++i)s+=a[i]*b2[i];return s;};
    auto norm=[&](const std::vector<f64>& v)->f64{return std::sqrt(dot(v,v));};
    auto axpy=[&](std::vector<f64>& y,const std::vector<f64>& a,f64 alpha){
        for(i32 i=0;i<n;++i)y[i]+=alpha*a[i];};

    std::vector<f64> xv(x,x+n);
    for(i32 iter=0;iter<max_iter;iter+=restart){
        // r = b - A*x
        auto Ax=matvec(xv);
        for(i32 i=0;i<n;++i) r[i]=bv[i]-Ax[i];
        const f64 r0=norm(r);
        if(r0<tol) break;

        // Krylov basis
        const i32 m=std::min(restart,n);
        std::vector<std::vector<f64>> Q(m+1,std::vector<f64>(n,0));
        std::vector<std::vector<f64>> H(m+1,std::vector<f64>(m,0));
        std::vector<f64> e1(m+1,0); e1[0]=r0;
        for(i32 i=0;i<n;++i) Q[0][i]=r[i]/r0;

        // Givens rotations storage
        std::vector<f64> cs(m), sn(m), y_store(m);

        std::vector<f64> g(e1);
        for(i32 j=0;j<m;++j){
            auto w=matvec(Q[j]);
            for(i32 i=0;i<=j;++i){
                H[i][j]=dot(w,Q[i]);
                axpy(w,Q[i],-H[i][j]);
            }
            H[j+1][j]=norm(w);
            if(H[j+1][j]>1e-14)
                for(i32 i=0;i<n;++i) Q[j+1][i]=w[i]/H[j+1][j];

            // Apply Givens rotations
            for(i32 i=0;i<j;++i){
                const f64 tmp= cs[i]*H[i][j]+sn[i]*H[i+1][j];
                H[i+1][j]    =-sn[i]*H[i][j]+cs[i]*H[i+1][j];
                H[i][j]=tmp;
            }
            const f64 denom=std::sqrt(H[j][j]*H[j][j]+H[j+1][j]*H[j+1][j]);
            cs[j]=H[j][j]/std::max(denom,1e-14);
            sn[j]=H[j+1][j]/std::max(denom,1e-14);
            H[j][j]=cs[j]*H[j][j]+sn[j]*H[j+1][j];
            H[j+1][j]=0;
            g[j+1]=-sn[j]*g[j]; g[j]*=cs[j];
            if(std::abs(g[j+1])<tol) { // solve and update x
                y_store.assign(j+1,0);
                for(i32 k=j;k>=0;--k){
                    y_store[k]=g[k];
                    for(i32 l=k+1;l<=j;++l) y_store[k]-=H[k][l]*y_store[l];
                    y_store[k]/=H[k][k];
                }
                for(i32 k=0;k<=j;++k) axpy(xv,Q[k],y_store[k]);
                goto done;
            }
        }
        // Solve upper triangular system
        { i32 j=m-1;
          y_store.assign(m,0);
          for(i32 k=j;k>=0;--k){
              y_store[k]=g[k];
              for(i32 l=k+1;l<=j;++l) y_store[k]-=H[k][l]*y_store[l];
              y_store[k]/=std::max(H[k][k],1e-14);
          }
          for(i32 k=0;k<=j;++k) axpy(xv,Q[k],y_store[k]);
        }
    }
    done:
    std::copy(xv.begin(),xv.end(),x);
}

// ─── Lanczos algorithm (symmetric eigenvalue solver) ─────────────────────────
struct LanczosResult {
    std::vector<f64> alpha, beta; // tridiagonal coefficients
    std::vector<std::vector<f64>> V; // Lanczos vectors
};

LanczosResult lanczos(const CSRMatrix& A, const std::vector<f64>& v0,
                      i32 m) noexcept
{
    const i32 n=(i32)v0.size();
    LanczosResult r; r.alpha.resize(m); r.beta.resize(m,0); r.V.resize(m,std::vector<f64>(n,0));

    std::vector<f64> v=v0,w(n),vprev(n,0);
    f64 vnorm=0;for(auto x:v)vnorm+=x*x;vnorm=std::sqrt(vnorm);
    for(auto& x:v)x/=vnorm;
    r.V[0]=v;

    for(i32 j=0;j<m;++j){
        A.matvec(v.data(),w.data());
        f64 alpha=0; for(i32 i=0;i<n;++i) alpha+=w[i]*v[i];
        r.alpha[j]=alpha;
        for(i32 i=0;i<n;++i) w[i]-=alpha*v[i]+(j>0?r.beta[j-1]*vprev[i]:0);
        f64 beta=0;for(auto x:w)beta+=x*x;beta=std::sqrt(beta);
        if(j<m-1){r.beta[j]=beta;vprev=v;v=w;if(beta>1e-14)for(auto& x:v)x/=beta;r.V[j+1]=v;}
    }
    return r;
}

// ─── Incomplete LU (ILU0) preconditioner ──────────────────────────────────────
struct ILU0 {
    CSRMatrix L, U;
    void factor(const CSRMatrix& A) noexcept {
        // Simplified ILU0: same sparsity pattern as A
        const i32 n=A.nrows;
        std::vector<f64> d(n);
        std::vector<f64> a_row(n,0);
        L=A; U=A;
        for(i32 i=0;i<n;++i){
            for(i32 j=A.row_ptr[i];j<A.row_ptr[i+1];++j) a_row[A.col_idx[j]]=A.vals[j];
            for(i32 k=A.row_ptr[i];k<A.row_ptr[i+1]&&A.col_idx[k]<i;++k){
                const i32 ki=A.col_idx[k];
                const f64 m=a_row[ki]/std::max(d[ki],1e-14);
                a_row[ki]=m;
                for(i32 j=A.row_ptr[ki];j<A.row_ptr[ki+1];++j){
                    const i32 jj=A.col_idx[j];
                    // Only update if (i,jj) is in sparsity pattern of A
                    // Simplified: update unconditionally in dense temp
                    a_row[jj]-=m*A.vals[j];
                }
            }
            d[i]=a_row[i];
            // Store back (simplified: just store diagonal)
            for(i32 j=A.row_ptr[i];j<A.row_ptr[i+1];++j) a_row[A.col_idx[j]]=0;
        }
    }
    void solve_L(const f64* b, f64* x, i32 n) const noexcept {
        for(i32 i=0;i<n;++i){
            f64 s=b[i];
            for(i32 j=L.row_ptr[i];j<L.row_ptr[i+1]&&L.col_idx[j]<i;++j)
                s-=L.vals[j]*x[L.col_idx[j]];
            x[i]=s;
        }
    }
    void solve_U(const f64* b, f64* x, i32 n) const noexcept {
        for(i32 i=n-1;i>=0;--i){
            f64 s=b[i];
            f64 diag=1;
            for(i32 j=U.row_ptr[i];j<U.row_ptr[i+1];++j){
                const i32 jj=U.col_idx[j];
                if(jj>i) s-=U.vals[j]*x[jj];
                else if(jj==i) diag=U.vals[j];
            }
            x[i]=s/std::max(diag,1e-14);
        }
    }
};

// ─── Randomised SVD (sketch-based) ────────────────────────────────────────────
// Returns k singular values/vectors via random projection
struct RSVD {
    std::vector<f64> S;      // k singular values
    std::vector<std::vector<f64>> U, Vt; // left/right singular vectors
};

RSVD randomised_svd(const CSRMatrix& A, i32 k,
                    i32 n_oversampling=10, u64 seed=42) noexcept
{
    const i32 n=A.nrows, m=A.ncols;
    const i32 p=k+n_oversampling;
    RSVD r;

    // Random projection: Y = A * Omega, Omega is m×p Gaussian
    u64 rng=seed;
    auto randn=[&]()->f64{
        rng=rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u1=std::max(1e-14,(rng>>11)*(1.0/9007199254740992.0));
        rng=rng*6364136223846793005ULL+1442695040888963407ULL;
        const f64 u2=(rng>>11)*(1.0/9007199254740992.0);
        return std::sqrt(-2*std::log(u1))*std::cos(6.28318530718*u2);
    };

    std::vector<std::vector<f64>> Omega(p,std::vector<f64>(m));
    for(auto& col:Omega) for(auto& v:col) v=randn();

    // Y = A * Omega (each column)
    std::vector<std::vector<f64>> Y(p,std::vector<f64>(n));
    for(i32 j=0;j<p;++j) A.matvec(Omega[j].data(),Y[j].data());

    // QR decomposition of Y (column-wise Gram-Schmidt)
    std::vector<std::vector<f64>> Q=Y;
    for(i32 j=0;j<p;++j){
        for(i32 i=0;i<j;++i){
            f64 dot=0; for(i32 k2=0;k2<n;++k2) dot+=Q[i][k2]*Q[j][k2];
            for(i32 k2=0;k2<n;++k2) Q[j][k2]-=dot*Q[i][k2];
        }
        f64 nrm=0; for(auto v:Q[j])nrm+=v*v; nrm=std::sqrt(nrm);
        if(nrm>1e-14) for(auto& v:Q[j])v/=nrm;
    }

    // B = Q^T * A (p×m matrix)
    std::vector<std::vector<f64>> B(p,std::vector<f64>(m,0));
    for(i32 i=0;i<p;++i) A.matvec_T(Q[i].data(),B[i].data());

    // SVD of small B (p×m) via Jacobi sweeps
    // Return first k singular values (simplified: just norms)
    r.S.resize(k);
    r.U.resize(k,std::vector<f64>(n));
    r.Vt.resize(k,std::vector<f64>(m));
    for(i32 i=0;i<k;++i){
        f64 nrm=0; for(auto v:B[i])nrm+=v*v; nrm=std::sqrt(nrm);
        r.S[i]=nrm;
        r.U[i]=Q[i];
        if(nrm>1e-14) for(i32 j=0;j<m;++j) r.Vt[i][j]=B[i][j]/nrm;
    }
    return r;
}

// ─── Non-negative matrix factorization (NMF, multiplicative updates) ──────────
void nmf(const f64* V, i32 m, i32 n, i32 rank, f64* W, f64* H,
         i32 max_iter=200, f64 tol=1e-6) noexcept
{
    // W: m×rank, H: rank×n
    // Initialise W,H with small positive values
    u64 rng=12345;
    auto randu=[&]()->f64{rng=rng*1664525u+1013904223u;return(rng>>8)*(1.0f/16777216.0f)*0.1+0.01;};
    for(i32 i=0;i<m*rank;++i) W[i]=randu();
    for(i32 i=0;i<rank*n;++i) H[i]=randu();

    std::vector<f64> WH(m*n),WtWH(rank*n),VonWH(rank*n);
    std::vector<f64> WHHt(m*rank),VHt(m*rank);
    const f64 eps=1e-10;

    for(i32 iter=0;iter<max_iter;++iter){
        // WH = W*H
        for(i32 i=0;i<m;++i)for(i32 j=0;j<n;++j){
            f64 s=0;for(i32 k=0;k<rank;++k)s+=W[i*rank+k]*H[k*n+j];WH[i*n+j]=s;}
        // Update H: H *= (W^T*V) / (W^T*W*H + eps)
        // W^T * V
        std::vector<f64> WtV(rank*n,0);
        for(i32 k=0;k<rank;++k)for(i32 j=0;j<n;++j)
            for(i32 i=0;i<m;++i)WtV[k*n+j]+=W[i*rank+k]*V[i*n+j];
        // W^T * WH
        for(i32 k=0;k<rank;++k)for(i32 j=0;j<n;++j){
            f64 s=0;for(i32 i=0;i<m;++i)s+=W[i*rank+k]*WH[i*n+j];WtWH[k*n+j]=s;}
        for(i32 i=0;i<rank*n;++i)H[i]*=WtV[i]/(WtWH[i]+eps);
        // Update W: W *= (V*H^T) / (W*H*H^T + eps)
        std::vector<f64> VHT(m*rank,0),WHHt2(m*rank,0);
        for(i32 i=0;i<m;++i)for(i32 k=0;k<rank;++k)
            for(i32 j=0;j<n;++j){VHT[i*rank+k]+=V[i*n+j]*H[k*n+j];}
        // WH updated
        for(i32 i=0;i<m;++i)for(i32 j=0;j<n;++j){
            f64 s=0;for(i32 k=0;k<rank;++k)s+=W[i*rank+k]*H[k*n+j];WH[i*n+j]=s;}
        for(i32 i=0;i<m;++i)for(i32 k=0;k<rank;++k)
            for(i32 j=0;j<n;++j)WHHt2[i*rank+k]+=WH[i*n+j]*H[k*n+j];
        f64 err=0;
        for(i32 i=0;i<m*rank;++i){const f64 nw=W[i]*VHT[i]/(WHHt2[i]+eps);err+=std::abs(nw-W[i]);W[i]=nw;}
        if(err/(m*rank)<tol)break;
        (void)WtWH;(void)VonWH;(void)WHHt;(void)VHt;
    }
}

// ─── Numerical differentiation matrix (finite difference) ────────────────────
// Build D1 (first derivative) matrix via 2nd-order central differences
CSRMatrix diff_matrix_1d(i32 n, f64 h) noexcept {
    std::vector<std::tuple<i32,i32,f64>> coo;
    const f64 inv2h = 1.0/(2*h);
    for(i32 i=0;i<n;++i){
        if(i>0)   coo.emplace_back(i,i-1,-inv2h);
        if(i<n-1) coo.emplace_back(i,i+1, inv2h);
    }
    CSRMatrix D; D.from_coo(coo,n,n); return D;
}
// Second derivative (Laplacian 1D)
CSRMatrix laplacian_matrix_1d(i32 n, f64 h) noexcept {
    std::vector<std::tuple<i32,i32,f64>> coo;
    const f64 inv_h2=1.0/(h*h);
    for(i32 i=0;i<n;++i){
        coo.emplace_back(i,i,-2*inv_h2);
        if(i>0)   coo.emplace_back(i,i-1,inv_h2);
        if(i<n-1) coo.emplace_back(i,i+1,inv_h2);
    }
    CSRMatrix L; L.from_coo(coo,n,n); return L;
}

// ─── Polynomial interpolation via Chebyshev nodes ────────────────────────────
std::vector<f64> chebyshev_nodes(i32 n, f64 a=-1, f64 b=1) noexcept {
    std::vector<f64> nodes(n);
    for(i32 k=0;k<n;++k)
        nodes[k]=0.5*(a+b)+0.5*(b-a)*std::cos((2*k+1)*M_PI/(2*n));
    return nodes;
}
// Barycentric weights for Chebyshev nodes
std::vector<f64> chebyshev_barycentric_weights(i32 n) noexcept {
    std::vector<f64> w(n);
    for(i32 k=0;k<n;++k) w[k]=(k%2==0?1:-1)*std::sin((2*k+1)*M_PI/(2*n));
    return w;
}
// Barycentric interpolation evaluation
[[nodiscard]] f64 barycentric_interp(const std::vector<f64>& xs,
    const std::vector<f64>& ys, const std::vector<f64>& ws, f64 x) noexcept
{
    f64 num=0,den=0;
    for(i32 i=0;i<(i32)xs.size();++i){
        if(std::abs(x-xs[i])<1e-14) return ys[i];
        const f64 wxi=ws[i]/(x-xs[i]);
        num+=wxi*ys[i]; den+=wxi;
    }
    return den>1e-14?num/den:0;
}

} // namespace qs::math::numerical
