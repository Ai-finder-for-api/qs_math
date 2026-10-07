#pragma once
// qs_math/linear_algebra/matrix_functions.hpp — Matrix exp/log/sqrt/pow, Cayley-Hamilton, norms
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Matrix power via repeated squaring ──────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_identity_f() noexcept {
    std::array<std::array<f64,N>,N> I{};
    for(int i=0;i<N;++i) I[i][i]=1.0;
    return I;
}

template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_mul_f(const std::array<std::array<f64,N>,N>& A,
          const std::array<std::array<f64,N>,N>& B) noexcept {
    std::array<std::array<f64,N>,N> C{};
    for(int i=0;i<N;++i) for(int k=0;k<N;++k) if(A[i][k]!=0)
        for(int j=0;j<N;++j) C[i][j]+=A[i][k]*B[k][j];
    return C;
}

template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_add_f(const std::array<std::array<f64,N>,N>& A,
          const std::array<std::array<f64,N>,N>& B) noexcept {
    std::array<std::array<f64,N>,N> C{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) C[i][j]=A[i][j]+B[i][j];
    return C;
}

template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_scale_f(const std::array<std::array<f64,N>,N>& A, f64 s) noexcept {
    std::array<std::array<f64,N>,N> C{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) C[i][j]=A[i][j]*s;
    return C;
}

template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_pow_int(std::array<std::array<f64,N>,N> A, int n) noexcept {
    auto R=mat_identity_f<N>();
    while(n>0){if(n&1)R=mat_mul_f<N>(R,A);A=mat_mul_f<N>(A,A);n>>=1;}
    return R;
}

// ─── Matrix exponential (Padé approximation, scaling & squaring) ──────────────
// expm(A) via [6/6] Padé approximant
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_exp(const std::array<std::array<f64,N>,N>& A) noexcept {
    // Compute 1-norm for scaling
    f64 norm=0;
    for(int j=0;j<N;++j){f64 col=0;for(int i=0;i<N;++i)col+=std::abs(A[i][j]);if(col>norm)norm=col;}
    int s=0; f64 A_scaled_norm=norm;
    while(A_scaled_norm>0.5){A_scaled_norm*=0.5;++s;}

    // Scale A
    f64 scale=std::pow(0.5,s);
    auto As=mat_scale_f<N>(A,scale);

    // Padé [6/6] coefficients
    const f64 c[]={1,0.5,0.12,1.833333e-2,1.992063e-3,1.630435e-4,1.035161e-5};
    auto A2=mat_mul_f<N>(As,As);
    auto A4=mat_mul_f<N>(A2,A2);
    auto A6=mat_mul_f<N>(A2,A4);
    auto I=mat_identity_f<N>();
    // U = A*(c5*A4 + c3*A2 + c1*I), V = c6*A6 + c4*A4 + c2*A2 + c0*I (approx)
    auto U=mat_mul_f<N>(As,mat_add_f<N>(mat_scale_f<N>(A4,c[5]),
           mat_add_f<N>(mat_scale_f<N>(A2,c[3]),mat_scale_f<N>(I,c[1]))));
    auto V=mat_add_f<N>(mat_scale_f<N>(A6,c[6]),
           mat_add_f<N>(mat_scale_f<N>(A4,c[4]),
           mat_add_f<N>(mat_scale_f<N>(A2,c[2]),mat_scale_f<N>(I,c[0]))));
    // P = U+V, Q = V-U → expm = Q^{-1}*P
    auto P=mat_add_f<N>(U,V);
    auto Qm=mat_add_f<N>(V,mat_scale_f<N>(U,-1));
    // Solve Qm*R=P via Gaussian elimination
    std::array<std::array<f64,2*N>,N> aug{};
    for(int i=0;i<N;++i){for(int j=0;j<N;++j)aug[i][j]=Qm[i][j];for(int j=0;j<N;++j)aug[i][N+j]=P[i][j];}
    for(int k=0;k<N;++k){
        int p=k;for(int i=k+1;i<N;++i)if(std::abs(aug[i][k])>std::abs(aug[p][k]))p=i;
        std::swap(aug[p],aug[k]);
        if(std::abs(aug[k][k])<1e-15)continue;
        for(int i=0;i<N;++i)if(i!=k){f64 f=aug[i][k]/aug[k][k];for(int j=k;j<2*N;++j)aug[i][j]-=f*aug[k][j];}
    }
    std::array<std::array<f64,N>,N> R{};
    for(int i=0;i<N;++i)for(int j=0;j<N;++j)if(std::abs(aug[i][i])>1e-15)R[i][j]=aug[i][N+j]/aug[i][i];
    // Repeated squaring: R = R^{2^s}
    for(int i=0;i<s;++i) R=mat_mul_f<N>(R,R);
    return R;
}

// ─── Matrix logarithm (Gregory series, near identity) ─────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_log_near_identity(const std::array<std::array<f64,N>,N>& A, int terms=20) noexcept {
    // log(A) = log(I + (A-I)) via series: X - X²/2 + X³/3 - ...
    auto I=mat_identity_f<N>();
    std::array<std::array<f64,N>,N> X{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) X[i][j]=A[i][j]-I[i][j];
    auto result=X;
    auto Xn=mat_mul_f<N>(X,X);
    for(int n=2;n<=terms;++n){
        result=mat_add_f<N>(result,mat_scale_f<N>(Xn,(n%2==0?-1.0:1.0)/n));
        Xn=mat_mul_f<N>(Xn,X);
    }
    return result;
}

// ─── Matrix square root (Denman-Beavers iteration) ────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
mat_sqrt_db(std::array<std::array<f64,N>,N> A, int iters=20) noexcept {
    auto Y=A, Z=mat_identity_f<N>();
    for(int k=0;k<iters;++k){
        // Y_{k+1} = (Y_k + Z_k^{-1})/2, Z_{k+1} = (Z_k + Y_k^{-1})/2
        // Invert via Gaussian elimination
        auto inv_f=[](std::array<std::array<f64,N>,N> M) {
            auto I=mat_identity_f<N>();
            std::array<std::array<f64,2*N>,N> aug{};
            for(int i=0;i<N;++i){for(int j=0;j<N;++j)aug[i][j]=M[i][j];aug[i][N+i]=1.0;}
            for(int k2=0;k2<N;++k2){
                int p=k2;for(int i=k2+1;i<N;++i)if(std::abs(aug[i][k2])>std::abs(aug[p][k2]))p=i;
                std::swap(aug[p],aug[k2]);
                if(std::abs(aug[k2][k2])<1e-15)continue;
                for(int i=0;i<N;++i)if(i!=k2){f64 f=aug[i][k2]/aug[k2][k2];for(int j=k2;j<2*N;++j)aug[i][j]-=f*aug[k2][j];}
            }
            std::array<std::array<f64,N>,N> inv{};
            for(int i=0;i<N;++i)for(int j=0;j<N;++j)if(std::abs(aug[i][i])>1e-15)inv[i][j]=aug[i][N+j]/aug[i][i];
            return inv;
        };
        auto Zinv=inv_f(Z),Yinv=inv_f(Y);
        auto Yn=mat_scale_f<N>(mat_add_f<N>(Y,Zinv),0.5);
        auto Zn=mat_scale_f<N>(mat_add_f<N>(Z,Yinv),0.5);
        Y=Yn;Z=Zn;
    }
    return Y;
}

// ─── Cayley-Hamilton theorem ───────────────────────────────────────────────────
// Every matrix satisfies its own characteristic polynomial p(A)=0
// Characteristic polynomial of 2×2: λ²-tr(A)λ+det(A)=0
template<int N>
[[nodiscard]] constexpr std::array<f64,N+1> characteristic_poly(
    const std::array<std::array<f64,N>,N>& A) noexcept
{
    // Leverrier-Faddeev algorithm
    std::array<f64,N+1> p{}; p[N]=1;
    auto Mk=mat_identity_f<N>();
    for(int k=1;k<=N;++k){
        Mk=mat_mul_f<N>(A,Mk);
        f64 tr=0; for(int i=0;i<N;++i) tr+=Mk[i][i];
        p[N-k]=-tr/k;
        // M_{k+1} = A*M_k + c_{n-k}*I
        for(int i=0;i<N;++i) Mk[i][i]+=p[N-k];
    }
    return p;
}

// ─── Matrix norms ─────────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 mat_norm_1(const std::array<std::array<f64,N>,N>& A) noexcept {
    f64 mx=0;
    for(int j=0;j<N;++j){f64 s=0;for(int i=0;i<N;++i)s+=std::abs(A[i][j]);if(s>mx)mx=s;}
    return mx;
}
template<int N>
[[nodiscard]] constexpr f64 mat_norm_inf(const std::array<std::array<f64,N>,N>& A) noexcept {
    f64 mx=0;
    for(int i=0;i<N;++i){f64 s=0;for(int j=0;j<N;++j)s+=std::abs(A[i][j]);if(s>mx)mx=s;}
    return mx;
}
template<int N>
[[nodiscard]] constexpr f64 mat_norm_fro(const std::array<std::array<f64,N>,N>& A) noexcept {
    f64 s=0;for(int i=0;i<N;++i)for(int j=0;j<N;++j)s+=A[i][j]*A[i][j];return std::sqrt(s);
}
// Nuclear norm (sum of singular values, approximated via Frobenius for symmetric)
template<int N>
[[nodiscard]] constexpr f64 mat_norm_nuclear_approx(
    const std::array<std::array<f64,N>,N>& A) noexcept
{
    // Lower bound: ||A||_* ≥ ||A||_F
    return mat_norm_fro<N>(A);
}

// ─── Matrix condition number ──────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 condition_number_1(
    const std::array<std::array<f64,N>,N>& A,
    const std::array<std::array<f64,N>,N>& A_inv) noexcept
{
    return mat_norm_1<N>(A)*mat_norm_1<N>(A_inv);
}

// ─── Commutator [A,B] = AB - BA ───────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
commutator(const std::array<std::array<f64,N>,N>& A,
           const std::array<std::array<f64,N>,N>& B) noexcept {
    return mat_add_f<N>(mat_mul_f<N>(A,B),mat_scale_f<N>(mat_mul_f<N>(B,A),-1));
}

// ─── Hadamard product (elementwise multiply) ───────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N>
hadamard(const std::array<std::array<f64,N>,N>& A,
         const std::array<std::array<f64,N>,N>& B) noexcept {
    std::array<std::array<f64,N>,N> C{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) C[i][j]=A[i][j]*B[i][j];
    return C;
}

// ─── Vec operator (column stacking) ───────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<f64,N*N>
vec_op(const std::array<std::array<f64,N>,N>& A) noexcept {
    std::array<f64,N*N> v{};
    for(int j=0;j<N;++j) for(int i=0;i<N;++i) v[j*N+i]=A[i][j];
    return v;
}

// ─── Bilinear form x'Ay ───────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 bilinear_form(
    const std::array<f64,N>& x,
    const std::array<std::array<f64,N>,N>& A,
    const std::array<f64,N>& y) noexcept
{
    f64 s=0;
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) s+=x[i]*A[i][j]*y[j];
    return s;
}
} // namespace qs::math
