#pragma once
// qs_math/tensor/tensor_ops.hpp — Tensor contractions, outer products, rank operations (C++20)
#include "../linear_algebra/mat4.hpp"
#include "../linear_algebra/vec4.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <array>
#include <cstring>

namespace qs::math::tensor {

// ─── Rank-2 tensor (3x3 & 4x4) operations ────────────────────────────────

// Frobenius norm of a 3x3 matrix stored as 9 floats
[[nodiscard]] inline f32 frobenius_norm3(const f32 m[9]) noexcept {
    f32 s = 0.f;
    for (int i = 0; i < 9; ++i) s += m[i]*m[i];
    return std::sqrt(s);
}
[[nodiscard]] inline f32 frobenius_norm4(const Mat4f& m) noexcept {
    f32 s = 0.f;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) { f32 v = m[i][j]; s += v*v; }
    return std::sqrt(s);
}

// Double contraction A:B = sum_ij A_ij * B_ij
[[nodiscard]] inline f32 double_contraction(const Mat4f& A, const Mat4f& B) noexcept {
    f32 s = 0.f;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) s += A[i][j] * B[i][j];
    return s;
}

// Outer product of two Vec3 -> rank-2 tensor (3x3 stored row-major)
inline void outer_product3(const Vec3f& a, const Vec3f& b, f32 out[9]) noexcept {
    out[0]=a.x*b.x; out[1]=a.x*b.y; out[2]=a.x*b.z;
    out[3]=a.y*b.x; out[4]=a.y*b.y; out[5]=a.y*b.z;
    out[6]=a.z*b.x; out[7]=a.z*b.y; out[8]=a.z*b.z;
}

// Symmetric part: S = (A + A^T)/2
inline void symmetric_part3(const f32 A[9], f32 S[9]) noexcept {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S[i*3+j] = 0.5f*(A[i*3+j] + A[j*3+i]);
}

// Skew-symmetric (antisymmetric) part: W = (A - A^T)/2
inline void skew_part3(const f32 A[9], f32 W[9]) noexcept {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            W[i*3+j] = 0.5f*(A[i*3+j] - A[j*3+i]);
}

// Trace: sum of diagonal
[[nodiscard]] inline f32 trace3(const f32 m[9]) noexcept { return m[0]+m[4]+m[8]; }
[[nodiscard]] inline f32 trace4(const Mat4f& m) noexcept { return m[0][0]+m[1][1]+m[2][2]+m[3][3]; }

// Deviator: remove hydrostatic part D = A - (tr(A)/3)*I
inline void deviator3(const f32 A[9], f32 D[9]) noexcept {
    f32 tr3 = trace3(A) / 3.f;
    for (int i = 0; i < 9; ++i) D[i] = A[i];
    D[0] -= tr3; D[4] -= tr3; D[8] -= tr3;
}

// Von Mises equivalent stress from Cauchy stress tensor (symmetric 3x3)
// sigma_s[0..5] = sxx,syy,szz,sxy,syz,sxz
[[nodiscard]] inline f32 von_mises_stress(const f32 s[6]) noexcept {
    f32 ds = s[0]-s[1], dt = s[1]-s[2], du = s[0]-s[2];
    return std::sqrt(0.5f*(ds*ds + dt*dt + du*du) + 3.f*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]));
}

// Principal stresses from symmetric 3x3 (analytic eigenvalues — Cardano)
struct PrincipalStresses { f32 s1, s2, s3; }; // s1 >= s2 >= s3

[[nodiscard]] inline PrincipalStresses principal_stresses(const f32 s[6]) noexcept {
    // s[0]=sxx, s[1]=syy, s[2]=szz, s[3]=sxy, s[4]=syz, s[5]=sxz
    f32 I1 = s[0]+s[1]+s[2];
    f32 I2 = s[0]*s[1]+s[1]*s[2]+s[0]*s[2] - s[3]*s[3]-s[4]*s[4]-s[5]*s[5];
    f32 I3 = s[0]*(s[1]*s[2]-s[4]*s[4]) - s[3]*(s[3]*s[2]-s[4]*s[5]) + s[5]*(s[3]*s[4]-s[1]*s[5]);
    f32 p  = I1/3.f;
    f32 q  = I1*I1/9.f - I2/3.f;
    f32 r  = I1*I1*I1/27.f - I1*I2/6.f + I3/2.f;
    f32 sq = std::sqrt(max(0.f, q));
    f32 phi= (sq > EPS_F32) ? std::acos(clamp(r/(sq*sq*sq), -1.f, 1.f)) / 3.f : 0.f;
    f32 s1 = p + 2.f*sq*std::cos(phi);
    f32 s2 = p + 2.f*sq*std::cos(phi + TWO_PI/3.f);
    f32 s3 = p + 2.f*sq*std::cos(phi + 2.f*TWO_PI/3.f);
    // sort descending
    if(s1<s2) std::swap(s1,s2);
    if(s1<s3) std::swap(s1,s3);
    if(s2<s3) std::swap(s2,s3);
    return {s1,s2,s3};
}

// ─── Rank-4 identity & isotropic tensors ─────────────────────────────────
// Isotropic rank-4 tensor: C_ijkl = lambda*delta_ij*delta_kl + mu*(delta_ik*delta_jl + delta_il*delta_jk)
// Stored as Voigt 6x6 (for symmetric stress/strain)
struct VoigtTensor6 {
    f32 C[6][6] = {};

    static VoigtTensor6 isotropic(f32 lambda, f32 mu) noexcept {
        VoigtTensor6 t;
        f32 l2m = lambda + 2.f*mu;
        t.C[0][0]=l2m; t.C[1][1]=l2m; t.C[2][2]=l2m;
        t.C[0][1]=t.C[1][0]=lambda;
        t.C[0][2]=t.C[2][0]=lambda;
        t.C[1][2]=t.C[2][1]=lambda;
        t.C[3][3]=mu; t.C[4][4]=mu; t.C[5][5]=mu;
        return t;
    }

    // Convert Young's modulus E and Poisson's ratio nu to Lame constants
    static VoigtTensor6 from_E_nu(f32 E, f32 nu) noexcept {
        f32 lambda = E*nu / ((1.f+nu)*(1.f-2.f*nu));
        f32 mu     = E / (2.f*(1.f+nu));
        return isotropic(lambda, mu);
    }

    // Apply: stress = C * strain (Voigt notation)
    void apply(const f32 strain[6], f32 stress[6]) const noexcept {
        for (int i = 0; i < 6; ++i) {
            stress[i] = 0.f;
            for (int j = 0; j < 6; ++j) stress[i] += C[i][j]*strain[j];
        }
    }
};

// ─── Tensor contraction (rank-N, fixed small sizes) ───────────────────────
// Contract last index of A with first index of B: C_ij = A_ik * B_kj
template<int M, int K, int N>
inline void tensor_contract(
    const f32 A[M*K], const f32 B[K*N], f32 C[M*N]) noexcept
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            f32 s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i*K+k]*B[k*N+j];
            C[i*N+j] = s;
        }
}

// ─── Levi-Civita symbol ───────────────────────────────────────────────────
// eps[i][j][k] = +1 (even perm), -1 (odd perm), 0 (repeated)
[[nodiscard]] constexpr int levi_civita(int i, int j, int k) noexcept {
    if (i==j || j==k || i==k) return 0;
    return ((i-j)*(j-k)*(k-i)/2) > 0 ? 1 : -1;
}

// Cross product via Levi-Civita: c_i = eps_ijk * a_j * b_k
[[nodiscard]] inline Vec3f lc_cross(Vec3f a, Vec3f b) noexcept {
    Vec3f c{0,0,0};
    f32 av[3]={a.x,a.y,a.z}, bv[3]={b.x,b.y,b.z}, cv[3]={0,0,0};
    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
                cv[i] += f32(levi_civita(i,j,k))*av[j]*bv[k];
    return {cv[0],cv[1],cv[2]};
}

// ─── Tensor invariants ────────────────────────────────────────────────────
// First, second, third invariants of a symmetric 3x3 tensor (Voigt)
struct TensorInvariants { f32 I1, I2, I3; };

[[nodiscard]] inline TensorInvariants tensor_invariants(const f32 s[6]) noexcept {
    f32 I1 = s[0]+s[1]+s[2];
    f32 I2 = s[0]*s[1]+s[1]*s[2]+s[0]*s[2] - s[3]*s[3]-s[4]*s[4]-s[5]*s[5];
    f32 I3 = s[0]*(s[1]*s[2]-s[4]*s[4])
            -s[3]*(s[3]*s[2]-s[4]*s[5])
            +s[5]*(s[3]*s[4]-s[1]*s[5]);
    return {I1, I2, I3};
}

// ─── Green-Lagrange strain tensor from deformation gradient F (3x3) ───────
// E = (F^T F - I) / 2
inline void green_lagrange_strain(const f32 F[9], f32 E[9]) noexcept {
    // C = F^T * F
    f32 C[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            C[i*3+j] = 0.f;
            for (int k = 0; k < 3; ++k) C[i*3+j] += F[k*3+i]*F[k*3+j];
        }
    for (int i = 0; i < 9; ++i) E[i] = 0.5f*C[i];
    E[0] -= 0.5f; E[4] -= 0.5f; E[8] -= 0.5f; // subtract I/2
}

// ─── Polar decomposition F = R * U (R orthogonal, U symmetric pos-def) ───
// Iterative method (Higham / Björck-Bowie)
inline bool polar_decompose(const f32 F[9], f32 R[9], f32 U[9], int iters=10) noexcept {
    f32 X[9]; std::memcpy(X, F, 36);
    for (int it = 0; it < iters; ++it) {
        // X_next = (X + X^{-T}) / 2  (simplified iteration)
        // Approximate X^{-T} via adjugate / det
        f32 det = X[0]*(X[4]*X[8]-X[5]*X[7])
                 -X[1]*(X[3]*X[8]-X[5]*X[6])
                 +X[2]*(X[3]*X[7]-X[4]*X[6]);
        if (std::abs(det) < 1e-12f) return false;
        f32 inv_det = 1.f/det;
        f32 inv[9];
        inv[0]= (X[4]*X[8]-X[5]*X[7])*inv_det;
        inv[1]=-(X[1]*X[8]-X[2]*X[7])*inv_det;
        inv[2]= (X[1]*X[5]-X[2]*X[4])*inv_det;
        inv[3]=-(X[3]*X[8]-X[5]*X[6])*inv_det;
        inv[4]= (X[0]*X[8]-X[2]*X[6])*inv_det;
        inv[5]=-(X[0]*X[5]-X[2]*X[3])*inv_det;
        inv[6]= (X[3]*X[7]-X[4]*X[6])*inv_det;
        inv[7]=-(X[0]*X[7]-X[1]*X[6])*inv_det;
        inv[8]= (X[0]*X[4]-X[1]*X[3])*inv_det;
        for(int i=0;i<9;++i) X[i] = 0.5f*(X[i]+inv[i]); // inv^T = inv for 3x3 transposed
    }
    std::memcpy(R, X, 36);
    // U = R^T * F
    for(int i=0;i<3;++i)
        for(int j=0;j<3;++j){
            U[i*3+j]=0.f;
            for(int k=0;k<3;++k) U[i*3+j]+=R[k*3+i]*F[k*3+j];
        }
    return true;
}

// ─── Covariant / contravariant index helpers ──────────────────────────────
// Metric tensor operations (for curvilinear coordinates)

// Raise index: v^i = g^ij * v_j  (g_inv is 3x3 inverse metric)
inline void raise_index(const f32 g_inv[9], const f32 v_lo[3], f32 v_hi[3]) noexcept {
    for(int i=0;i<3;++i){
        v_hi[i]=0.f;
        for(int j=0;j<3;++j) v_hi[i]+=g_inv[i*3+j]*v_lo[j];
    }
}
// Lower index: v_i = g_ij * v^j
inline void lower_index(const f32 g[9], const f32 v_hi[3], f32 v_lo[3]) noexcept {
    for(int i=0;i<3;++i){
        v_lo[i]=0.f;
        for(int j=0;j<3;++j) v_lo[i]+=g[i*3+j]*v_hi[j];
    }
}

} // namespace qs::math::tensor
