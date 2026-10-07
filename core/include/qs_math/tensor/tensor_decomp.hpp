#pragma once
// qs_math/tensor/tensor_decomp.hpp — CP/Tucker decomposition, tensor SVD, unfolding
#include "../arithmetic/scalar_types.hpp"
#include "tensor_core.hpp"
#include "../linear_algebra/vec3.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Mode-n unfolding (matricization) ────────────────────────────────────────
// Unfold rank-3 tensor T(I,J,K) along mode n into a matrix
// Mode 0: X_{i, jk} → (I) × (J*K)
// Mode 1: X_{j, ik} → (J) × (I*K)
// Mode 2: X_{k, ij} → (K) × (I*J)
template<typename T, int I, int J, int K>
[[nodiscard]] constexpr Tensor<T,I,J*K> unfold_mode0(const Tensor<T,I,J,K>& A) noexcept {
    Tensor<T,I,J*K> M{};
    for(int i=0;i<I;++i) for(int j=0;j<J;++j) for(int k=0;k<K;++k)
        M(i, j*K+k)=A(i,j,k);
    return M;
}

template<typename T, int I, int J, int K>
[[nodiscard]] constexpr Tensor<T,J,I*K> unfold_mode1(const Tensor<T,I,J,K>& A) noexcept {
    Tensor<T,J,I*K> M{};
    for(int i=0;i<I;++i) for(int j=0;j<J;++j) for(int k=0;k<K;++k)
        M(j, i*K+k)=A(i,j,k);
    return M;
}

template<typename T, int I, int J, int K>
[[nodiscard]] constexpr Tensor<T,K,I*J> unfold_mode2(const Tensor<T,I,J,K>& A) noexcept {
    Tensor<T,K,I*J> M{};
    for(int i=0;i<I;++i) for(int j=0;j<J;++j) for(int k=0;k<K;++k)
        M(k, i*J+j)=A(i,j,k);
    return M;
}

// Fold back: mode-0 unfolded matrix → rank-3 tensor
template<typename T, int I, int J, int K>
[[nodiscard]] constexpr Tensor<T,I,J,K> fold_mode0(const Tensor<T,I,J*K>& M) noexcept {
    Tensor<T,I,J,K> A{};
    for(int i=0;i<I;++i) for(int j=0;j<J;++j) for(int k=0;k<K;++k)
        A(i,j,k)=M(i,j*K+k);
    return A;
}

// ─── CP (CANDECOMP/PARAFAC) rank-R approximation building block ───────────────
// Represents T ≈ sum_{r=0}^{R-1} a_r ⊗ b_r ⊗ c_r
template<typename T, int I, int J, int K, int R>
struct CPDecomp {
    std::array<Tensor<T,I>,R> A{}; // factor matrices columns
    std::array<Tensor<T,J>,R> B{};
    std::array<Tensor<T,K>,R> C{};
    std::array<T,R>           lambdas{}; // scaling

    // Reconstruct tensor from CP decomposition
    [[nodiscard]] constexpr Tensor<T,I,J,K> reconstruct() const noexcept {
        Tensor<T,I,J,K> X{};
        for(int r=0;r<R;++r)
            for(int i=0;i<I;++i) for(int j=0;j<J;++j) for(int k=0;k<K;++k)
                X(i,j,k)+=lambdas[r]*A[r][i]*B[r][j]*C[r][k];
        return X;
    }

    // Frobenius error: ||T - X_hat||_F
    [[nodiscard]] constexpr f64 error(const Tensor<T,I,J,K>& T_orig) const noexcept {
        auto diff=T_orig-reconstruct();
        return frobenius_norm(diff);
    }
};

// ─── Tucker decomposition: T = G ×_1 U ×_2 V ×_3 W ──────────────────────────
template<typename T, int I, int J, int K, int P, int Q, int S>
struct TuckerDecomp {
    Tensor<T,P,Q,S>  G{};  // core tensor
    Tensor<T,I,P>    U{};  // mode-0 factor
    Tensor<T,J,Q>    V{};  // mode-1 factor
    Tensor<T,K,S>    W{};  // mode-2 factor

    [[nodiscard]] constexpr Tensor<T,I,J,K> reconstruct() const noexcept {
        // T_{ijk} = sum_{pqs} G_{pqs} U_{ip} V_{jq} W_{ks}
        Tensor<T,I,J,K> X{};
        for(int i=0;i<I;++i) for(int j=0;j<J;++j) for(int k=0;k<K;++k)
            for(int p=0;p<P;++p) for(int q=0;q<Q;++q) for(int s=0;s<S;++s)
                X(i,j,k)+=G(p,q,s)*U(i,p)*V(j,q)*W(k,s);
        return X;
    }
};

// Mode-n product: T ×_n M → new tensor with nth dimension replaced by M's rows
template<typename T, int I, int J, int K, int P>
[[nodiscard]] constexpr Tensor<T,P,J,K> mode0_product(
    const Tensor<T,I,J,K>& A, const Tensor<T,P,I>& M) noexcept
{
    Tensor<T,P,J,K> B{};
    for(int p=0;p<P;++p) for(int j=0;j<J;++j) for(int k=0;k<K;++k)
        for(int i=0;i<I;++i) B(p,j,k)+=M(p,i)*A(i,j,k);
    return B;
}

// ─── Tensor train (TT) / MPS format ──────────────────────────────────────────
// T[i0,i1,...,iN-1] = G0[i0] * G1[i1] * ... * G_{N-1}[i_{N-1}]
// where G_k is a r_{k-1} × n_k × r_k tensor (r_0=r_N=1)
template<typename T, int N_MODES, int MODE_DIM, int BOND>
struct TensorTrain {
    // Simplified: all modes same dimension, all bonds same
    static constexpr int n=MODE_DIM, r=BOND;
    // cores[k] = r × n × r matrix (rank-3 TT core)
    std::array<Tensor<T,r,n,r>, N_MODES> cores{};

    // Evaluate at multi-index (i0, i1, ..., iN-1) by contracting cores
    [[nodiscard]] constexpr T eval(const int* idx) const noexcept {
        std::array<T,r> vec{};
        // Left boundary: 1×n×r, treat as vector
        for(int b=0;b<r;++b) vec[b]=cores[0](0,idx[0],b);
        for(int k=1;k<N_MODES;++k) {
            std::array<T,r> nvec{};
            for(int b2=0;b2<r;++b2)
                for(int b1=0;b1<r;++b1)
                    nvec[b2]+=vec[b1]*cores[k](b1,idx[k],b2);
            vec=nvec;
        }
        return vec[0]; // right boundary bond dim = 1
    }
};

// ─── Tensor norm and normalization ────────────────────────────────────────────
template<typename T, int... Dims>
[[nodiscard]] constexpr Tensor<T,Dims...> tensor_normalize(
    const Tensor<T,Dims...>& A) noexcept
{
    f64 nrm=frobenius_norm(A);
    if(nrm<1e-15) return A;
    return A*(T(1.0/nrm));
}

// ─── Tensor contractions in Einstein notation (concrete, small) ───────────────
// A_{ij} B_{jk} → C_{ik} (matrix multiply, already in tensor_core)
// A_{ijk} B_{k} → C_{ij} (tensor-vector)
template<typename T, int I, int J, int K>
[[nodiscard]] constexpr Tensor<T,I,J> contract_last(
    const Tensor<T,I,J,K>& A, const Tensor<T,K>& v) noexcept
{
    Tensor<T,I,J> C{};
    for(int i=0;i<I;++i) for(int j=0;j<J;++j)
        for(int k=0;k<K;++k) C(i,j)+=A(i,j,k)*v[k];
    return C;
}

// A_{ij} B_{ij} → scalar (double contraction)
template<typename T, int I, int J>
[[nodiscard]] constexpr T full_contract(
    const Tensor<T,I,J>& A, const Tensor<T,I,J>& B) noexcept
{
    return double_contract(A,B);
}

} // namespace qs::math
