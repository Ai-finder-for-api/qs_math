#pragma once
// qs_math/algebra/module_theory.hpp — Modules over rings, free modules, projective, flat
#include "../arithmetic/scalar_types.hpp"
#include "ring_theory.hpp"
#include <array>

namespace qs::math {

// ─── Free module R^n over ring R ─────────────────────────────────────────────
template<typename R, int N>
struct FreeModule {
    std::array<R,N> comps{};

    [[nodiscard]] constexpr FreeModule operator+(const FreeModule& o) const noexcept {
        FreeModule r;
        for(int i=0;i<N;++i) r.comps[i]=comps[i]+o.comps[i];
        return r;
    }
    [[nodiscard]] constexpr FreeModule operator-(const FreeModule& o) const noexcept {
        FreeModule r;
        for(int i=0;i<N;++i) r.comps[i]=comps[i]-o.comps[i];
        return r;
    }
    [[nodiscard]] constexpr FreeModule operator*(R s) const noexcept {
        FreeModule r;
        for(int i=0;i<N;++i) r.comps[i]=comps[i]*s;
        return r;
    }
    [[nodiscard]] constexpr R dot(const FreeModule& o) const noexcept {
        R s=R::zero();
        for(int i=0;i<N;++i) s=s+comps[i]*o.comps[i];
        return s;
    }
    [[nodiscard]] constexpr bool operator==(const FreeModule& o) const noexcept {
        for(int i=0;i<N;++i) if(!(comps[i]==o.comps[i])) return false;
        return true;
    }
    [[nodiscard]] static constexpr FreeModule zero() noexcept {
        FreeModule r;
        for(int i=0;i<N;++i) r.comps[i]=R::zero();
        return r;
    }
    [[nodiscard]] static constexpr FreeModule basis(int k) noexcept {
        FreeModule r=zero();
        r.comps[k]=R::one();
        return r;
    }
};

// ─── Module homomorphism (matrix over ring) ────────────────────────────────────
template<typename R, int M, int N>
struct ModuleMap {
    std::array<std::array<R,N>,M> mat{};

    [[nodiscard]] constexpr FreeModule<R,M> apply(const FreeModule<R,N>& v) const noexcept {
        FreeModule<R,M> result=FreeModule<R,M>::zero();
        for(int i=0;i<M;++i)
            for(int j=0;j<N;++j)
                result.comps[i]=result.comps[i]+mat[i][j]*v.comps[j];
        return result;
    }

    // Compose two maps: (M×K) ∘ (K×N) → (M×N)
    template<int K>
    [[nodiscard]] constexpr ModuleMap<R,M,N> compose(const ModuleMap<R,K,N>& other) const noexcept {
        ModuleMap<R,M,N> result{};
        for(int i=0;i<M;++i)
            for(int j=0;j<N;++j) {
                result.mat[i][j]=R::zero();
                // This would be M×K * K×N, but we fix types for demonstration
                for(int k=0;k<K&&k<N;++k)
                    result.mat[i][j]=result.mat[i][j]+mat[i][k]*other.mat[k][j];
            }
        return result;
    }

    // Kernel (null space) — find elements v such that A*v=0 over Zn via row reduction
    [[nodiscard]] constexpr int rank_mod(i64 p) const noexcept {
        std::array<std::array<i64,N>,M> A{};
        for(int i=0;i<M;++i) for(int j=0;j<N;++j) A[i][j]=mat[i][j].v;
        int rank=0;
        for(int col=0;col<N&&rank<M;++col) {
            int pivot=-1;
            for(int row=rank;row<M;++row) if(A[row][col]%p!=0){pivot=row;break;}
            if(pivot<0) continue;
            std::swap(A[pivot],A[rank]);
            i64 inv=1,base=A[rank][col]%p,e=p-2;
            while(e>0){if(e&1)inv=inv*base%p;base=base*base%p;e>>=1;}
            for(int j=0;j<N;++j) A[rank][j]=A[rank][j]*inv%p;
            for(int row=0;row<M;++row) if(row!=rank&&A[row][col]) {
                i64 f=A[row][col];
                for(int j=0;j<N;++j) A[row][j]=(A[row][j]-f*A[rank][j]%p+p)%p;
            }
            ++rank;
        }
        return rank;
    }
};

// ─── Tensor product of free modules M ⊗_R N ──────────────────────────────────
// M=R^m, N=R^n, M⊗N=R^{mn}, basis e_i⊗f_j ↔ flat index i*n+j
template<typename R, int M, int N>
[[nodiscard]] constexpr FreeModule<R,M*N> tensor_product(
    const FreeModule<R,M>& u, const FreeModule<R,N>& v) noexcept
{
    FreeModule<R,M*N> result=FreeModule<R,M*N>::zero();
    for(int i=0;i<M;++i)
        for(int j=0;j<N;++j)
            result.comps[i*N+j]=u.comps[i]*v.comps[j];
    return result;
}

// ─── Hom module: Hom_R(M,N) ≅ R^{m*n} (for free modules) ─────────────────────
// A linear map M→N is an m×n matrix; composition is matrix multiply

// ─── Exact sequences ──────────────────────────────────────────────────────────
// Check: im(f) = ker(g) for maps f:A→B, g:B→C (on finite Zp modules)
template<typename R, int A, int B, int C>
[[nodiscard]] constexpr bool is_exact(
    const ModuleMap<R,B,A>& f,
    const ModuleMap<R,C,B>& g,
    i64 p) noexcept
{
    // Check g∘f = 0
    for(int j=0;j<A;++j) {
        auto ej=FreeModule<R,A>::basis(j);
        auto fej=f.apply(ej);
        auto gfej=g.apply(fej);
        for(int i=0;i<C;++i)
            if(gfej.comps[i].v%p!=0) return false;
    }
    // im(f) = ker(g) requires rank check: rank(f)+rank(g)=rank(B)
    // (First-isomorphism theorem: B/ker(g) ≅ im(g))
    int rf=f.rank_mod(p), rg=g.rank_mod(p);
    return rf+rg==B; // exact iff rank splits correctly
}

// ─── Projective module test (for Z/nZ modules) ────────────────────────────────
// A module P is projective iff for every surjection M→N, Hom(P,M)→Hom(P,N) is surj.
// Free modules are projective; Z/nZ over Z is not projective (for composite n)
[[nodiscard]] constexpr bool is_projective_Zn_module(i64 n) noexcept {
    // Z/nZ is projective over Z iff n=1 (trivial)
    return n==1;
}

// ─── Injective module test ────────────────────────────────────────────────────
// Q is injective Z-module (divisible); Z/nZ is injective over itself
[[nodiscard]] constexpr bool is_self_injective(i64 n) noexcept {
    // Z/nZ is injective over Z/nZ iff n is square-free
    auto f=factorize(n);
    for(int i=0;i<f.count;++i) if(f.exponents[i]>1) return false;
    return true;
}

// ─── Fitting ideal of a module ────────────────────────────────────────────────
// For M presented by matrix A (m×n over PID R), Fitt_k(M) = ideal of (n-k)-minors
// We compute for 2×2 case (determinant = first Fitting ideal)
template<typename R>
[[nodiscard]] constexpr R fitting_ideal_2x2(const ModuleMap<R,2,2>& A) noexcept {
    // det(A) = a00*a11 - a01*a10
    return A.mat[0][0]*A.mat[1][1]-A.mat[0][1]*A.mat[1][0];
}

// ─── Localization at multiplicative set ───────────────────────────────────────
// S^{-1}Z where S = Z\pZ (integers localized at prime p)
// Elements: a/s where s not divisible by p
struct ZLocalizedAtP {
    i64 num{0}, den{1}, p{2};

    [[nodiscard]] constexpr ZLocalizedAtP operator+(ZLocalizedAtP o) const noexcept {
        return {num*o.den+o.num*den, den*o.den, p};
    }
    [[nodiscard]] constexpr ZLocalizedAtP operator*(ZLocalizedAtP o) const noexcept {
        return {num*o.num, den*o.den, p};
    }
    [[nodiscard]] constexpr bool in_ring() const noexcept {
        return den%p!=0; // denominator coprime to p
    }
};

// ─── Graded module (for polynomial modules) ────────────────────────────────────
// A graded module M = ⊕ M_d; we track degrees
template<int MAXDEG, int RANK>
struct GradedModule {
    // graded_comps[d][r] = component in degree d, rank r
    std::array<std::array<i64,RANK>,MAXDEG+1> comps{};

    [[nodiscard]] constexpr i64 hilbert_function(int d) const noexcept {
        if(d<0||d>MAXDEG) return 0;
        int count=0;
        for(int r=0;r<RANK;++r) if(comps[d][r]) ++count;
        return count;
    }
};

} // namespace qs::math
