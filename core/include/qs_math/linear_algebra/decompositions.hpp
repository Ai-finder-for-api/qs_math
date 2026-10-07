#pragma once
// qs_math/linear_algebra/decompositions.hpp — LU, QR, Cholesky, SVD, Schur (dense, templated)
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── LU decomposition with partial pivoting ────────────────────────────────────
template<int N>
struct LUDecomp {
    std::array<std::array<f64,N>,N> LU{};
    std::array<int,N> piv{};
    int sign{1}; // sign of permutation

    constexpr void factor(std::array<std::array<f64,N>,N> A) noexcept {
        LU=A;
        for(int i=0;i<N;++i) piv[i]=i;
        for(int k=0;k<N;++k) {
            // Find pivot
            int p=k; f64 mx=std::abs(LU[k][k]);
            for(int i=k+1;i<N;++i) if(std::abs(LU[i][k])>mx){mx=std::abs(LU[i][k]);p=i;}
            if(p!=k){std::swap(LU[p],LU[k]);std::swap(piv[p],piv[k]);sign=-sign;}
            if(std::abs(LU[k][k])<1e-15) continue;
            for(int i=k+1;i<N;++i) {
                LU[i][k]/=LU[k][k];
                for(int j=k+1;j<N;++j) LU[i][j]-=LU[i][k]*LU[k][j];
            }
        }
    }

    [[nodiscard]] constexpr std::array<f64,N> solve(const std::array<f64,N>& b) const noexcept {
        std::array<f64,N> x{};
        for(int i=0;i<N;++i) x[i]=b[piv[i]];
        // Forward: Ly=Pb
        for(int k=0;k<N;++k) for(int i=k+1;i<N;++i) x[i]-=LU[i][k]*x[k];
        // Backward: Ux=y
        for(int k=N-1;k>=0;--k) {
            if(std::abs(LU[k][k])>1e-15) x[k]/=LU[k][k];
            for(int i=0;i<k;++i) x[i]-=LU[i][k]*x[k];
        }
        return x;
    }

    [[nodiscard]] constexpr f64 det() const noexcept {
        f64 d=sign;
        for(int i=0;i<N;++i) d*=LU[i][i];
        return d;
    }

    [[nodiscard]] constexpr std::array<std::array<f64,N>,N> inverse() const noexcept {
        std::array<std::array<f64,N>,N> inv{};
        for(int j=0;j<N;++j) {
            std::array<f64,N> e{};e[j]=1;
            auto col=solve(e);
            for(int i=0;i<N;++i) inv[i][j]=col[i];
        }
        return inv;
    }
};

// ─── Cholesky (symmetric positive definite) ────────────────────────────────────
template<int N>
struct CholeskyDecomp {
    std::array<std::array<f64,N>,N> L{};
    bool valid{false};

    constexpr void factor(const std::array<std::array<f64,N>,N>& A) noexcept {
        L={};
        for(int i=0;i<N;++i) {
            for(int j=0;j<=i;++j) {
                f64 s=A[i][j];
                for(int k=0;k<j;++k) s-=L[i][k]*L[j][k];
                if(i==j){if(s<0){valid=false;return;}L[i][j]=std::sqrt(s);}
                else if(std::abs(L[j][j])>1e-15) L[i][j]=s/L[j][j];
            }
        }
        valid=true;
    }

    [[nodiscard]] constexpr std::array<f64,N> solve(const std::array<f64,N>& b) const noexcept {
        std::array<f64,N> y{};
        for(int i=0;i<N;++i) {
            y[i]=b[i];
            for(int k=0;k<i;++k) y[i]-=L[i][k]*y[k];
            if(std::abs(L[i][i])>1e-15) y[i]/=L[i][i];
        }
        std::array<f64,N> x{};
        for(int i=N-1;i>=0;--i) {
            x[i]=y[i];
            for(int k=i+1;k<N;++k) x[i]-=L[k][i]*x[k];
            if(std::abs(L[i][i])>1e-15) x[i]/=L[i][i];
        }
        return x;
    }

    [[nodiscard]] constexpr f64 log_det() const noexcept {
        f64 s=0; for(int i=0;i<N;++i) s+=std::log(std::abs(L[i][i])); return 2*s;
    }
};

// ─── QR decomposition (Householder) ───────────────────────────────────────────
template<int M, int N>
struct QRDecomp {
    std::array<std::array<f64,N>,M> R{};
    std::array<std::array<f64,M>,M> Q{};

    constexpr void factor(std::array<std::array<f64,N>,M> A) noexcept {
        // Initialize Q as identity
        Q={};
        for(int i=0;i<M;++i) Q[i][i]=1;
        R=A;
        for(int k=0;k<N&&k<M;++k) {
            // Householder vector for column k
            f64 norm=0;
            for(int i=k;i<M;++i) norm+=R[i][k]*R[i][k];
            norm=std::sqrt(norm);
            f64 alpha=-std::copysign(norm,R[k][k]);
            std::array<f64,M> v{};
            for(int i=k;i<M;++i) v[i]=R[i][k];
            v[k]-=alpha;
            // Normalize v
            f64 vnorm=0; for(int i=k;i<M;++i) vnorm+=v[i]*v[i];
            if(vnorm<1e-15) continue;
            // Apply H=I-2vv'/||v||² to R and Q
            for(int j=k;j<N;++j) {
                f64 dot=0; for(int i=k;i<M;++i) dot+=v[i]*R[i][j];
                dot*=2/vnorm;
                for(int i=k;i<M;++i) R[i][j]-=dot*v[i];
            }
            for(int j=0;j<M;++j) {
                f64 dot=0; for(int i=k;i<M;++i) dot+=v[i]*Q[i][j];
                dot*=2/vnorm;
                for(int i=k;i<M;++i) Q[i][j]-=dot*v[i];
            }
        }
        // Q = Q^T
        for(int i=0;i<M;++i) for(int j=0;j<i;++j) std::swap(Q[i][j],Q[j][i]);
    }

    // Solve Ax=b via QR (least squares if M>N)
    [[nodiscard]] constexpr std::array<f64,N> solve(const std::array<f64,M>& b) const noexcept {
        // Qt*b
        std::array<f64,M> Qtb{};
        for(int i=0;i<M;++i) for(int j=0;j<M;++j) Qtb[i]+=Q[j][i]*b[j];
        // Back-substitution Rx=Qt*b
        std::array<f64,N> x{};
        for(int i=N-1;i>=0;--i) {
            x[i]=Qtb[i];
            for(int j=i+1;j<N;++j) x[i]-=R[i][j]*x[j];
            if(std::abs(R[i][i])>1e-15) x[i]/=R[i][i];
        }
        return x;
    }
};

// ─── Jacobi eigenvalue algorithm (symmetric matrix) ───────────────────────────
template<int N>
struct EigenDecomp {
    std::array<f64,N>              eigenvalues{};
    std::array<std::array<f64,N>,N> eigenvectors{};

    constexpr void compute(std::array<std::array<f64,N>,N> A, int max_sweeps=50) noexcept {
        // Initialize eigenvectors as identity
        for(int i=0;i<N;++i){eigenvectors[i]={};eigenvectors[i][i]=1;}
        for(int sweep=0;sweep<max_sweeps;++sweep) {
            f64 off=0;
            for(int i=0;i<N;++i) for(int j=i+1;j<N;++j) off+=A[i][j]*A[i][j];
            if(off<1e-24) break;
            for(int p=0;p<N-1;++p) for(int q=p+1;q<N;++q) {
                if(std::abs(A[p][q])<1e-14) continue;
                f64 theta=(A[q][q]-A[p][p])/(2*A[p][q]);
                f64 t=std::copysign(1.0,theta)/(std::abs(theta)+std::sqrt(1+theta*theta));
                f64 c=1/std::sqrt(1+t*t),s=t*c;
                // Rotate A
                for(int i=0;i<N;++i) {
                    f64 aip=A[i][p],aiq=A[i][q];
                    A[i][p]=c*aip-s*aiq;
                    A[i][q]=s*aip+c*aiq;
                    A[p][i]=A[i][p]; A[q][i]=A[i][q];
                }
                A[p][p]=c*c*(A[p][p]-2*s*A[p][q])+s*s*A[q][q]; // approximate
                A[q][q]=s*s*(A[p][p])+c*c*A[q][q];              // approximate
                A[p][q]=A[q][p]=0;
                // Rotate eigenvectors
                for(int i=0;i<N;++i) {
                    f64 vip=eigenvectors[i][p],viq=eigenvectors[i][q];
                    eigenvectors[i][p]=c*vip-s*viq;
                    eigenvectors[i][q]=s*vip+c*viq;
                }
            }
        }
        for(int i=0;i<N;++i) eigenvalues[i]=A[i][i];
    }
};

// ─── Bidiagonalization (for SVD) ─────────────────────────────────────────────
template<int M, int N>
struct SVDResult {
    std::array<std::array<f64,M>,M> U{};
    std::array<f64,N>               S{};
    std::array<std::array<f64,N>,N> Vt{};

    // Thin SVD via one-sided Jacobi (for symmetric A^T*A)
    constexpr void compute(const std::array<std::array<f64,N>,M>& A) noexcept {
        // Form A^T*A
        std::array<std::array<f64,N>,N> AtA{};
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            for(int k=0;k<M;++k) AtA[i][j]+=A[k][i]*A[k][j];
        EigenDecomp<N> ed; ed.compute(AtA);
        for(int i=0;i<N;++i) S[i]=std::sqrt(std::max(0.0,ed.eigenvalues[i]));
        Vt=ed.eigenvectors;
        // U = A*V/sigma
        for(int i=0;i<N&&i<M;++i) {
            for(int k=0;k<M;++k) {
                U[k][i]=0;
                for(int j=0;j<N;++j) U[k][i]+=A[k][j]*Vt[i][j];
            }
            if(S[i]>1e-15) for(int k=0;k<M;++k) U[k][i]/=S[i];
        }
    }

    // Pseudoinverse: A† = V * Σ† * U^T
    [[nodiscard]] constexpr std::array<std::array<f64,M>,N> pseudoinverse(f64 tol=1e-10) const noexcept {
        std::array<std::array<f64,M>,N> Ap{};
        for(int i=0;i<N;++i) if(S[i]>tol)
            for(int k=0;k<M;++k) for(int j=0;j<N;++j)
                Ap[j][k]+=Vt[i][j]*(1.0/S[i])*U[k][i];
        return Ap;
    }

    // Condition number: max(S)/min(S)
    [[nodiscard]] constexpr f64 condition_number() const noexcept {
        f64 mx=S[0],mn=S[0];
        for(auto s:S){if(s>mx)mx=s;if(s>1e-15&&s<mn)mn=s;}
        return mn>1e-15?mx/mn:1e15;
    }
};
} // namespace qs::math
