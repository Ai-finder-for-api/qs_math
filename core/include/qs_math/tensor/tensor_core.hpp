#pragma once
// qs_math/tensor/tensor_core.hpp — N-dimensional dense tensor, index ops, reshape, slice
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <numeric>
#include <algorithm>
#include <cstring>

namespace qs::math {

// ─── Static tensor: rank R, shape stored at compile time ─────────────────────
template<typename T, int... Dims>
struct Tensor {
    static constexpr int rank = sizeof...(Dims);
    static constexpr int size = (Dims * ...);
    static constexpr std::array<int,rank> shape = {Dims...};

    std::array<T,size> data{};

    // Flat index from multi-index (row-major)
    template<typename... Idx>
    [[nodiscard]] static constexpr int flat(Idx... idx) noexcept {
        static_assert(sizeof...(idx)==rank);
        const std::array<int,rank> sh={Dims...};
        const std::array<int,rank> ix={int(idx)...};
        int flat=0, stride=1;
        for(int i=rank-1;i>=0;--i){ flat+=ix[i]*stride; stride*=sh[i]; }
        return flat;
    }

    template<typename... Idx>
    [[nodiscard]] constexpr T  operator()(Idx... idx) const noexcept { return data[flat(idx...)]; }
    template<typename... Idx>
    [[nodiscard]] constexpr T& operator()(Idx... idx)       noexcept { return data[flat(idx...)]; }

    [[nodiscard]] constexpr T  operator[](int i) const noexcept { return data[i]; }
    [[nodiscard]] constexpr T& operator[](int i)       noexcept { return data[i]; }

    [[nodiscard]] constexpr Tensor operator+(const Tensor& o) const noexcept {
        Tensor r; for(int i=0;i<size;++i) r.data[i]=data[i]+o.data[i]; return r;
    }
    [[nodiscard]] constexpr Tensor operator-(const Tensor& o) const noexcept {
        Tensor r; for(int i=0;i<size;++i) r.data[i]=data[i]-o.data[i]; return r;
    }
    [[nodiscard]] constexpr Tensor operator*(T s) const noexcept {
        Tensor r; for(int i=0;i<size;++i) r.data[i]=data[i]*s; return r;
    }
    [[nodiscard]] constexpr bool operator==(const Tensor& o) const noexcept {
        for(int i=0;i<size;++i) if(data[i]!=o.data[i]) return false; return true;
    }
};

// ─── Tensor contraction ───────────────────────────────────────────────────────
// Contract rank-2 tensors (matrix multiply): T_{ik} = A_{ij}*B_{jk}
template<typename T, int M, int K, int N>
[[nodiscard]] constexpr Tensor<T,M,N> contract(
    const Tensor<T,M,K>& A, const Tensor<T,K,N>& B) noexcept
{
    Tensor<T,M,N> C{};
    for(int i=0;i<M;++i)
        for(int k=0;k<K;++k) if(A(i,k)!=T{})
            for(int j=0;j<N;++j)
                C(i,j)+=A(i,k)*B(k,j);
    return C;
}

// Outer product: T_{ij} = a_i * b_j
template<typename T, int M, int N>
[[nodiscard]] constexpr Tensor<T,M,N> outer(
    const Tensor<T,M>& a, const Tensor<T,N>& b) noexcept
{
    Tensor<T,M,N> C{};
    for(int i=0;i<M;++i) for(int j=0;j<N;++j) C(i,j)=a[i]*b[j];
    return C;
}

// Trace of a rank-2 tensor
template<typename T, int N>
[[nodiscard]] constexpr T trace(const Tensor<T,N,N>& A) noexcept {
    T s{}; for(int i=0;i<N;++i) s+=A(i,i); return s;
}

// Transpose of rank-2 tensor
template<typename T, int M, int N>
[[nodiscard]] constexpr Tensor<T,N,M> transpose(const Tensor<T,M,N>& A) noexcept {
    Tensor<T,N,M> B{};
    for(int i=0;i<M;++i) for(int j=0;j<N;++j) B(j,i)=A(i,j);
    return B;
}

// ─── Frobenius norm ───────────────────────────────────────────────────────────
template<typename T, int... Dims>
[[nodiscard]] constexpr f64 frobenius_norm(const Tensor<T,Dims...>& A) noexcept {
    f64 s=0; for(auto v:A.data) s+=f64(v)*f64(v);
    return std::sqrt(s);
}

// ─── Kronecker product (tensor product of matrices) ───────────────────────────
template<typename T, int M, int N, int P, int Q>
[[nodiscard]] constexpr Tensor<T,M*P,N*Q> kron(
    const Tensor<T,M,N>& A, const Tensor<T,P,Q>& B) noexcept
{
    Tensor<T,M*P,N*Q> C{};
    for(int i=0;i<M;++i) for(int j=0;j<N;++j)
        for(int p=0;p<P;++p) for(int q=0;q<N;++q)
            C(i*P+p, j*Q+q)=A(i,j)*B(p,q);
    return C;
}

// ─── Einstein summation helpers (index bookkeeping types) ────────────────────
// Tag types for upper/lower indices
template<int I> struct Up{};   // contravariant index
template<int I> struct Dn{};   // covariant index

// ─── Symmetrize / antisymmetrize rank-2 ──────────────────────────────────────
template<typename T, int N>
[[nodiscard]] constexpr Tensor<T,N,N> symmetrize(const Tensor<T,N,N>& A) noexcept {
    Tensor<T,N,N> S{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) S(i,j)=(A(i,j)+A(j,i))*T(0.5);
    return S;
}

template<typename T, int N>
[[nodiscard]] constexpr Tensor<T,N,N> antisymmetrize(const Tensor<T,N,N>& A) noexcept {
    Tensor<T,N,N> S{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) S(i,j)=(A(i,j)-A(j,i))*T(0.5);
    return S;
}

// Symmetric part: S = (T + T^T)/2 for rank-2
// Antisymmetric part: A = (T - T^T)/2

// ─── Contraction on specified axes (rank reduction) ───────────────────────────
// Contract A[M,N] on both axes (double contraction): sum_{ij} A_{ij}*B_{ij}
template<typename T, int M, int N>
[[nodiscard]] constexpr T double_contract(
    const Tensor<T,M,N>& A, const Tensor<T,M,N>& B) noexcept
{
    T s{}; for(int i=0;i<M;++i) for(int j=0;j<N;++j) s+=A(i,j)*B(i,j);
    return s;
}

// ─── Rank-4 tensor (common in elasticity/curvature) ──────────────────────────
template<typename T, int N>
using Tensor4 = Tensor<T,N,N,N,N>;

// Symmetries of Riemann-like rank-4 tensors: R_{ijkl}=-R_{jikl}=-R_{ijlk}=R_{klij}
template<typename T, int N>
[[nodiscard]] constexpr bool has_riemann_symmetry(const Tensor4<T,N>& R) noexcept {
    for(int i=0;i<N;++i) for(int j=0;j<N;++j)
        for(int k=0;k<N;++k) for(int l=0;l<N;++l) {
            if(R(i,j,k,l)+R(j,i,k,l)!=T{}) return false;
            if(R(i,j,k,l)+R(i,j,l,k)!=T{}) return false;
            if(R(i,j,k,l)-R(k,l,i,j)!=T{}) return false;
        }
    return true;
}

} // namespace qs::math
