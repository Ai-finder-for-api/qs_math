#pragma once
// qs_math/tensor_math.hpp — Rank-N tensors, contraction, Kronecker product, Levi-Civita (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "mat4.hpp"
#include <array>
#include <vector>
#include <numeric>
#include <cassert>
#include <cmath>
#include <functional>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// Tensor<T, Rank> — fixed-rank tensor with dynamic shape
// ─────────────────────────────────────────────────────────────────────────────
template<typename T, u32 Rank>
struct Tensor {
    std::array<u32, Rank>  shape{};
    std::vector<T>         data;

    Tensor() = default;
    explicit Tensor(std::array<u32, Rank> s) : shape(s) {
        u32 n = 1; for (auto d : s) n *= d;
        data.assign(n, T(0));
    }

    [[nodiscard]] u32 size() const { return u32(data.size()); }
    [[nodiscard]] bool empty() const { return data.empty(); }

    // Linear index from multi-index (row-major)
    template<typename... Idx>
    [[nodiscard]] u32 linear_index(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank);
        std::array<u32,Rank> ids{u32(idx)...};
        u32 lin = 0, stride = 1;
        for (i32 i = i32(Rank)-1; i >= 0; --i) {
            lin += ids[u32(i)] * stride;
            stride *= shape[u32(i)];
        }
        return lin;
    }

    template<typename... Idx>
    [[nodiscard]] T& at(Idx... idx)       { return data[linear_index(idx...)]; }
    template<typename... Idx>
    [[nodiscard]] T  at(Idx... idx) const { return data[linear_index(idx...)]; }

    // Element access by multi-index array
    [[nodiscard]] T& operator[](std::array<u32,Rank> idx) {
        u32 lin = 0, stride = 1;
        for (i32 i=i32(Rank)-1;i>=0;--i) { lin+=idx[u32(i)]*stride; stride*=shape[u32(i)]; }
        return data[lin];
    }
    [[nodiscard]] T operator[](std::array<u32,Rank> idx) const {
        u32 lin = 0, stride = 1;
        for (i32 i=i32(Rank)-1;i>=0;--i) { lin+=idx[u32(i)]*stride; stride*=shape[u32(i)]; }
        return data[lin];
    }

    // Scalar ops
    [[nodiscard]] Tensor operator*(T s) const { Tensor r=*this; for(auto& v:r.data) v*=s; return r; }
    [[nodiscard]] Tensor operator+(const Tensor& b) const {
        assert(shape==b.shape);
        Tensor r=*this; for(u32 i=0;i<size();++i) r.data[i]+=b.data[i]; return r;
    }
    [[nodiscard]] Tensor operator-(const Tensor& b) const {
        assert(shape==b.shape);
        Tensor r=*this; for(u32 i=0;i<size();++i) r.data[i]-=b.data[i]; return r;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Contraction — trace over two indices
// contract(A, i, j) where A[...,i,...,j,...] with shape[i]==shape[j]
// reduces rank by 2
// ─────────────────────────────────────────────────────────────────────────────
// For rank-2 (matrix): trace
template<typename T>
[[nodiscard]] T trace(const Tensor<T,2>& A) {
    assert(A.shape[0]==A.shape[1]);
    T s = T(0);
    for (u32 i=0;i<A.shape[0];++i) s += A.at(i,i);
    return s;
}

// Full contraction of two rank-1 tensors (dot product)
template<typename T>
[[nodiscard]] T contract(const Tensor<T,1>& a, const Tensor<T,1>& b) {
    assert(a.shape[0]==b.shape[0]);
    T s=T(0); for(u32 i=0;i<a.shape[0];++i) s+=a.data[i]*b.data[i]; return s;
}

// Matrix-vector product via contraction
template<typename T>
[[nodiscard]] Tensor<T,1> contract_mv(const Tensor<T,2>& M, const Tensor<T,1>& v) {
    assert(M.shape[1]==v.shape[0]);
    Tensor<T,1> r({M.shape[0]});
    for(u32 i=0;i<M.shape[0];++i)
        for(u32 j=0;j<M.shape[1];++j)
            r.at(i) += M.at(i,j)*v.at(j);
    return r;
}

// Matrix-matrix product
template<typename T>
[[nodiscard]] Tensor<T,2> contract_mm(const Tensor<T,2>& A, const Tensor<T,2>& B) {
    assert(A.shape[1]==B.shape[0]);
    Tensor<T,2> R({A.shape[0], B.shape[1]});
    for(u32 i=0;i<A.shape[0];++i)
        for(u32 k=0;k<A.shape[1];++k)
            for(u32 j=0;j<B.shape[1];++j)
                R.at(i,j) += A.at(i,k)*B.at(k,j);
    return R;
}

// Outer product: rank-m ⊗ rank-n → rank-(m+n)
template<typename T>
[[nodiscard]] Tensor<T,2> outer(const Tensor<T,1>& a, const Tensor<T,1>& b) {
    Tensor<T,2> R({a.shape[0], b.shape[0]});
    for(u32 i=0;i<a.shape[0];++i)
        for(u32 j=0;j<b.shape[0];++j)
            R.at(i,j) = a.at(i)*b.at(j);
    return R;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kronecker product A ⊗ B  (for rank-2 tensors / matrices)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
[[nodiscard]] Tensor<T,2> kronecker_product(const Tensor<T,2>& A, const Tensor<T,2>& B) {
    u32 m=A.shape[0],n=A.shape[1],p=B.shape[0],q=B.shape[1];
    Tensor<T,2> R({m*p, n*q});
    for(u32 i=0;i<m;++i) for(u32 j=0;j<n;++j)
        for(u32 k=0;k<p;++k) for(u32 l=0;l<q;++l)
            R.at(i*p+k, j*q+l) = A.at(i,j)*B.at(k,l);
    return R;
}

// Kronecker delta δ_ij
template<typename T=f32>
[[nodiscard]] constexpr T kronecker_delta(u32 i, u32 j) { return i==j ? T(1) : T(0); }

// ─────────────────────────────────────────────────────────────────────────────
// Levi-Civita symbol ε_ijk (rank-3, fully antisymmetric)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T=f32>
[[nodiscard]] constexpr T levi_civita(u32 i, u32 j, u32 k) {
    if (i==j||j==k||i==k) return T(0);
    if ((i==0&&j==1&&k==2)||(i==1&&j==2&&k==0)||(i==2&&j==0&&k==1)) return T( 1);
    return T(-1);
}

// 4D Levi-Civita ε_ijkl
template<typename T=f32>
[[nodiscard]] constexpr T levi_civita4(u32 a,u32 b,u32 c,u32 d) {
    // Sign = signature of permutation (0,1,2,3)→(a,b,c,d)
    if(a==b||a==c||a==d||b==c||b==d||c==d) return T(0);
    std::array<u32,4> p{a,b,c,d};
    int sign=1;
    for(u32 i=0;i<4;++i) for(u32 j=i+1;j<4;++j) if(p[i]>p[j]) sign=-sign;
    return T(sign);
}

// Build the full 3×3×3 Levi-Civita tensor
template<typename T=f32>
[[nodiscard]] Tensor<T,3> levi_civita_tensor3() {
    Tensor<T,3> E({3,3,3});
    for(u32 i=0;i<3;++i) for(u32 j=0;j<3;++j) for(u32 k=0;k<3;++k)
        E.at(i,j,k) = levi_civita<T>(i,j,k);
    return E;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank-2 tensor operations common in physics/graphics
// ─────────────────────────────────────────────────────────────────────────────

// Symmetric part: S = (T + T^T) / 2
template<typename T>
[[nodiscard]] Tensor<T,2> symmetric_part(const Tensor<T,2>& A) {
    assert(A.shape[0]==A.shape[1]);
    u32 n=A.shape[0];
    Tensor<T,2> S({n,n});
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
        S.at(i,j) = T(0.5)*(A.at(i,j)+A.at(j,i));
    return S;
}

// Antisymmetric part: A = (T - T^T) / 2
template<typename T>
[[nodiscard]] Tensor<T,2> antisymmetric_part(const Tensor<T,2>& A) {
    assert(A.shape[0]==A.shape[1]);
    u32 n=A.shape[0];
    Tensor<T,2> R({n,n});
    for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
        R.at(i,j) = T(0.5)*(A.at(i,j)-A.at(j,i));
    return R;
}

// Deviator (traceless part): D = T - (tr(T)/n) * I
template<typename T>
[[nodiscard]] Tensor<T,2> deviator(const Tensor<T,2>& A) {
    u32 n=A.shape[0]; assert(A.shape[1]==n);
    T tr = trace(A) / T(n);
    Tensor<T,2> D = A;
    for(u32 i=0;i<n;++i) D.at(i,i) -= tr;
    return D;
}

// Frobenius norm
template<typename T, u32 R>
[[nodiscard]] T frobenius_norm(const Tensor<T,R>& A) {
    T s=T(0); for(auto v:A.data) s+=v*v; return std::sqrt(s);
}

// Tensor from Mat3 / Mat4
template<typename T=f32>
[[nodiscard]] Tensor<T,2> from_mat3(const Mat3<T>& M) {
    Tensor<T,2> A({3,3});
    for(u32 i=0;i<3;++i) for(u32 j=0;j<3;++j) A.at(i,j)=M.m[i][j];
    return A;
}
template<typename T=f32>
[[nodiscard]] Tensor<T,2> from_mat4(const Mat4<T>& M) {
    Tensor<T,2> A({4,4});
    for(u32 i=0;i<4;++i) for(u32 j=0;j<4;++j) A.at(i,j)=M.m[i][j];
    return A;
}

// Inertia tensor of a collection of point masses: I_ij = Σ m_k (|r_k|²δ_ij - r_ki r_kj)
template<typename T=f32>
[[nodiscard]] Tensor<T,2> inertia_tensor(const std::vector<T>& masses, const std::vector<Vec3<T>>& positions) {
    Tensor<T,2> I({3,3});
    for(u32 k=0;k<masses.size();++k) {
        T m=masses[k]; const Vec3<T>& r=positions[k];
        T r2=r.x*r.x+r.y*r.y+r.z*r.z;
        T rx[3]={r.x,r.y,r.z};
        for(u32 i=0;i<3;++i) for(u32 j=0;j<3;++j)
            I.at(i,j) += m*(kronecker_delta<T>(i,j)*r2 - rx[i]*rx[j]);
    }
    return I;
}

} // namespace qs
