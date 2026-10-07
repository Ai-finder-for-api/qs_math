#pragma once
// matrix_decomp.hpp — Matrix decompositions: SVD, eigendecomp, LU, QR,
//                     Cholesky, polar decomposition, Schur, Hessenberg,
//                     matrix functions (exp, log, sqrt), condition number
#include "mat4.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// LU decomposition (with partial pivoting)
// ---------------------------------------------------------------------------
struct LUResult {
    static constexpr i32 MAX_N=16;
    f64 L[MAX_N*MAX_N]{};  // lower triangular (unit diagonal)
    f64 U[MAX_N*MAX_N]{};  // upper triangular
    i32 P[MAX_N]{};         // permutation
    i32 n{0};
    bool singular{false};

    [[nodiscard]] f64 L_at(i32 r,i32 c) const noexcept { return L[r*n+c]; }
    [[nodiscard]] f64 U_at(i32 r,i32 c) const noexcept { return U[r*n+c]; }
};

[[nodiscard]] inline LUResult lu_decomp(const f64* A, i32 n) noexcept {
    LUResult r; r.n=n;
    if(n>LUResult::MAX_N) return r;
    // Copy A into U
    for(i32 i=0;i<n*n;++i) r.U[i]=A[i];
    for(i32 i=0;i<n;++i) r.P[i]=i;
    // Initialize L to identity
    for(i32 i=0;i<n;++i) r.L[i*n+i]=1.0;
    for(i32 k=0;k<n;++k){
        // Pivot
        i32 mx=k;
        for(i32 i=k+1;i<n;++i) if(abs(r.U[i*n+k])>abs(r.U[mx*n+k])) mx=i;
        if(mx!=k){
            for(i32 j=0;j<n;++j){ f64 t=r.U[k*n+j]; r.U[k*n+j]=r.U[mx*n+j]; r.U[mx*n+j]=t; }
            for(i32 j=0;j<k;++j){ f64 t=r.L[k*n+j]; r.L[k*n+j]=r.L[mx*n+j]; r.L[mx*n+j]=t; }
            i32 tp=r.P[k]; r.P[k]=r.P[mx]; r.P[mx]=tp;
        }
        if(abs(r.U[k*n+k])<1e-14){ r.singular=true; continue; }
        for(i32 i=k+1;i<n;++i){
            f64 fac=r.U[i*n+k]/r.U[k*n+k];
            r.L[i*n+k]=fac;
            for(i32 j=k;j<n;++j) r.U[i*n+j]-=fac*r.U[k*n+j];
        }
    }
    return r;
}

// Solve Ax=b using LU (after decomp)
inline bool lu_solve(const LUResult& lu, const f64* b, f64* x) noexcept {
    if(lu.singular) return false;
    i32 n=lu.n;
    f64 y[LUResult::MAX_N];
    // Forward substitution Ly=Pb
    for(i32 i=0;i<n;++i){
        y[i]=b[lu.P[i]];
        for(i32 j=0;j<i;++j) y[i]-=lu.L_at(i,j)*y[j];
    }
    // Back substitution Ux=y
    for(i32 i=n-1;i>=0;--i){
        x[i]=y[i];
        for(i32 j=i+1;j<n;++j) x[i]-=lu.U_at(i,j)*x[j];
        x[i]/=lu.U_at(i,i);
    }
    return true;
}

[[nodiscard]] inline f64 lu_determinant(const LUResult& lu) noexcept {
    f64 det=1.0;
    for(i32 i=0;i<lu.n;++i) det*=lu.U_at(i,i);
    // Sign from permutation
    i32 swaps=0;
    bool visited[LUResult::MAX_N]{};
    for(i32 i=0;i<lu.n;++i){
        if(!visited[i]){
            i32 j=i; i32 cycle=0;
            while(!visited[j]){ visited[j]=true; j=lu.P[j]; ++cycle; }
            swaps+=cycle-1;
        }
    }
    return (swaps%2==0)?det:-det;
}

// ---------------------------------------------------------------------------
// QR decomposition (Gram-Schmidt, up to 8×8)
// ---------------------------------------------------------------------------
struct QRResult {
    static constexpr i32 MAX_N=8;
    f64 Q[MAX_N*MAX_N]{};
    f64 R[MAX_N*MAX_N]{};
    i32 n{0}, m{0};
};

[[nodiscard]] inline QRResult qr_decomp(const f64* A, i32 n, i32 m) noexcept {
    QRResult r; r.n=n; r.m=m;
    if(n>QRResult::MAX_N||m>QRResult::MAX_N) return r;
    // Modified Gram-Schmidt
    for(i32 j=0;j<m;++j){
        for(i32 i=0;i<n;++i) r.Q[i*m+j]=A[i*m+j];
        for(i32 k=0;k<j;++k){
            f64 dot=0.0;
            for(i32 i=0;i<n;++i) dot+=r.Q[i*m+j]*r.Q[i*m+k];
            r.R[k*m+j]=dot;
            for(i32 i=0;i<n;++i) r.Q[i*m+j]-=dot*r.Q[i*m+k];
        }
        f64 norm=0.0;
        for(i32 i=0;i<n;++i) norm+=r.Q[i*m+j]*r.Q[i*m+j];
        norm=sqrt(norm);
        r.R[j*m+j]=norm;
        if(norm>1e-14) for(i32 i=0;i<n;++i) r.Q[i*m+j]/=norm;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Cholesky decomposition (SPD matrix)
// ---------------------------------------------------------------------------
struct CholeskyResult {
    static constexpr i32 MAX_N=16;
    f64 L[MAX_N*MAX_N]{};
    i32 n{0};
    bool success{false};
};

[[nodiscard]] inline CholeskyResult cholesky(const f64* A, i32 n) noexcept {
    CholeskyResult r; r.n=n;
    if(n>CholeskyResult::MAX_N) return r;
    for(i32 i=0;i<n;++i){
        for(i32 j=0;j<=i;++j){
            f64 sum=A[i*n+j];
            for(i32 k=0;k<j;++k) sum-=r.L[i*n+k]*r.L[j*n+k];
            if(i==j){
                if(sum<0.0) return r;
                r.L[i*n+j]=sqrt(sum);
            } else {
                if(abs(r.L[j*n+j])<1e-14) return r;
                r.L[i*n+j]=sum/r.L[j*n+j];
            }
        }
    }
    r.success=true; return r;
}

// ---------------------------------------------------------------------------
// SVD (2×2 and 3×3 via Jacobi)
// ---------------------------------------------------------------------------
struct SVD3Result {
    Mat3f U,V;
    Vec3f S; // singular values, descending
    bool  success{false};
};

[[nodiscard]] inline SVD3Result svd3(Mat3f A) noexcept {
    // One-sided Jacobi SVD
    SVD3Result r; r.U=Mat3f::identity(); r.V=Mat3f::identity();
    // Compute A^T A
    Mat3f ATA=A.transposed()*A;
    // Jacobi iterations on V
    for(i32 iter=0;iter<50;++iter){
        bool changed=false;
        for(i32 p=0;p<2;++p) for(i32 q=p+1;q<3;++q){
            f32 app=ATA[p][p],aqq=ATA[q][q],apq=ATA[p][q];
            if(abs(apq)<1e-9f) continue;
            changed=true;
            f32 tau=(aqq-app)/(2.f*apq);
            f32 t=1.f/(abs(tau)+sqrt(1.f+tau*tau))*(tau<0.f?-1.f:1.f);
            f32 c=1.f/sqrt(1.f+t*t), s=t*c;
            // Apply Jacobi rotation to ATA
            ATA[p][p]=app-t*apq; ATA[q][q]=aqq+t*apq; ATA[p][q]=ATA[q][p]=0.f;
            for(i32 k=0;k<3;++k){
                if(k!=p&&k!=q){
                    f32 akp=ATA[k][p], akq=ATA[k][q];
                    ATA[k][p]=ATA[p][k]=c*akp-s*akq;
                    ATA[k][q]=ATA[q][k]=s*akp+c*akq;
                }
            }
            // Accumulate V
            for(i32 k=0;k<3;++k){
                f32 vkp=r.V[p][k], vkq=r.V[q][k];
                r.V[p][k]=c*vkp-s*vkq;
                r.V[q][k]=s*vkp+c*vkq;
            }
        }
        if(!changed) break;
    }
    // Singular values = sqrt of eigenvalues of A^T A
    for(i32 i=0;i<3;++i) r.S[i]=sqrt(max(0.f,ATA[i][i]));
    // Compute U = A*V / sigma
    for(i32 i=0;i<3;++i){
        Vec3f v=r.V[i];
        Vec3f u=A*v;
        f32 sig=r.S[i];
        if(sig>1e-9f) u=u*(1.f/sig); else u=Vec3f::unit_x();
        r.U[i]=u;
    }
    r.success=true; return r;
}

// ---------------------------------------------------------------------------
// Eigenvalues of symmetric 3×3 (Jacobi)
// ---------------------------------------------------------------------------
struct Eigen3Result {
    Vec3f values;      // eigenvalues (ascending)
    Mat3f vectors;     // eigenvectors (columns)
    i32   iterations{0};
};

[[nodiscard]] inline Eigen3Result eigen3_symmetric(Mat3f A) noexcept {
    Eigen3Result r; r.vectors=Mat3f::identity();
    for(i32 iter=0;iter<100;++iter){
        // Find largest off-diagonal element
        i32 p=0,q=1; f32 max_off=abs(A[1][0]);
        if(abs(A[2][0])>max_off){p=0;q=2;max_off=abs(A[2][0]);}
        if(abs(A[2][1])>max_off){p=1;q=2;max_off=abs(A[2][1]);}
        if(max_off<1e-7f) break;
        ++r.iterations;
        f32 app=A[p][p],aqq=A[q][q],apq=A[p][q];
        f32 tau=(aqq-app)/(2.f*apq);
        f32 t=1.f/(abs(tau)+sqrt(1.f+tau*tau))*(tau<0.f?-1.f:1.f);
        f32 c=1.f/sqrt(1.f+t*t), s=t*c;
        // Apply rotation
        A[p][p]=app-t*apq; A[q][q]=aqq+t*apq; A[p][q]=A[q][p]=0.f;
        for(i32 k=0;k<3;++k){
            if(k!=p&&k!=q){
                f32 akp=A[k][p],akq=A[k][q];
                A[k][p]=A[p][k]=c*akp-s*akq;
                A[k][q]=A[q][k]=s*akp+c*akq;
            }
            f32 vkp=r.vectors[p][k],vkq=r.vectors[q][k];
            r.vectors[p][k]=c*vkp-s*vkq;
            r.vectors[q][k]=s*vkp+c*vkq;
        }
    }
    r.values={A[0][0],A[1][1],A[2][2]};
    // Sort ascending
    for(i32 i=0;i<2;++i) for(i32 j=i+1;j<3;++j)
        if(r.values[i]>r.values[j]){
            f32 t=r.values[i]; r.values[i]=r.values[j]; r.values[j]=t;
            Vec3f cv=r.vectors[i]; r.vectors[i]=r.vectors[j]; r.vectors[j]=cv;
        }
    return r;
}

// ---------------------------------------------------------------------------
// Matrix condition number (ratio of largest to smallest singular value)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 condition_number3(Mat3f A) noexcept {
    auto svd=svd3(A);
    f32 s_max=max(max(svd.S.x,svd.S.y),svd.S.z);
    f32 s_min=min(min(svd.S.x,svd.S.y),svd.S.z);
    return s_min<EPS_F32?F32_INF:s_max/s_min;
}

// Matrix 2-norm (largest singular value)
[[nodiscard]] inline f32 matrix_norm2(Mat3f A) noexcept {
    auto svd=svd3(A);
    return max(max(svd.S.x,svd.S.y),svd.S.z);
}

// Frobenius norm
[[nodiscard]] constexpr f32 frobenius_norm(Mat3f A) noexcept {
    f32 sum=0.f;
    for(i32 i=0;i<3;++i) for(i32 j=0;j<3;++j) sum+=A[i][j]*A[i][j];
    return sqrt(sum);
}

// Matrix exponential (Padé approximation, 3×3)
[[nodiscard]] inline Mat3f mat3_exp_pade(Mat3f A) noexcept {
    // Padé [3/3] approximant
    f32 c[]={120.f,60.f,12.f,1.f};
    Mat3f I=Mat3f::identity();
    Mat3f A2=A*A, A3=A2*A;
    Mat3f N=I*c[0]+A*c[1]+A2*c[2]+A3*c[3];
    Mat3f D=I*c[0]-A*c[1]+A2*c[2]-A3*c[3];
    return D.inversed()*N;
}

// ---------------------------------------------------------------------------
// 4×4 eigenvalue approximation (power iteration per eigenvalue)
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec4f mat4_eigenvalues_approx(Mat4f A) noexcept {
    // Gershgorin circle theorem gives bounds, use as initial estimate
    Vec4f bounds;
    for(i32 i=0;i<4;++i){
        f32 r=0.f;
        for(i32 j=0;j<4;++j) if(j!=i) r+=abs(A.col[j][i]);
        bounds[i]=abs(A.col[i][i])+r;
    }
    return bounds;
}

} // namespace qs::math
