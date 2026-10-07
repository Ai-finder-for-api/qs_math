#pragma once
// qs_math/topology/homology.hpp — Simplicial homology, Betti numbers, boundary operators
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bitset>

namespace qs::math {

// ─── Simplex ─────────────────────────────────────────────────────────────────
template<int K>
struct Simplex {
    std::array<int,K+1> verts{}; // K+1 vertices for a K-simplex

    // Orientation sign: (-1)^i for ith face
    [[nodiscard]] constexpr int face_sign(int i) const noexcept {
        return (i%2==0)?1:-1;
    }

    // ith face: K-simplex obtained by removing vertex i
    [[nodiscard]] constexpr Simplex<K-1> face(int i) const noexcept {
        Simplex<K-1> f;
        for(int j=0,k=0;j<=K;++j) if(j!=i) f.verts[k++]=verts[j];
        return f;
    }

    [[nodiscard]] constexpr bool operator==(const Simplex& o) const noexcept {
        for(int i=0;i<=K;++i) if(verts[i]!=o.verts[i]) return false;
        return true;
    }
};

// ─── Simplicial complex (fixed max dimension and simplex count) ───────────────
template<int MAX_DIM, int MAX_SIMPLICES>
struct SimplicialComplex {
    // Simplices stored by dimension — flatten into one array tagged by dim
    struct SimplexEntry {
        int dim{-1};
        std::array<int,MAX_DIM+1> verts{};
        int orientation{1};
    };
    std::array<SimplexEntry,MAX_SIMPLICES> simplices{};
    int n_simplices{0};

    constexpr void add(int dim, const int* verts, int ori=1) noexcept {
        if(n_simplices>=MAX_SIMPLICES) return;
        auto& e=simplices[n_simplices++];
        e.dim=dim; e.orientation=ori;
        for(int i=0;i<=dim;++i) e.verts[i]=verts[i];
    }

    [[nodiscard]] constexpr int count(int dim) const noexcept {
        int c=0; for(int i=0;i<n_simplices;++i) if(simplices[i].dim==dim) ++c; return c;
    }
};

// ─── Boundary matrix (Z_2 coefficients for simplicity) ────────────────────────
// ∂_k: C_k → C_{k-1}
// Entry ∂[i][j] = 1 if (k-1)-simplex i is in boundary of k-simplex j (mod 2)
template<int ROWS, int COLS>
using BoundaryMatrix = std::array<std::array<u8,COLS>,ROWS>;

// Compute rank of Z_2 matrix via Gaussian elimination
template<int M, int N>
[[nodiscard]] constexpr int z2_rank(BoundaryMatrix<M,N> mat) noexcept {
    int rank=0, row=0;
    for(int col=0;col<N&&row<M;++col) {
        // Find pivot
        int pivot=-1;
        for(int r=row;r<M;++r) if(mat[r][col]){pivot=r;break;}
        if(pivot<0) continue;
        // Swap
        if(pivot!=row) std::swap(mat[pivot],mat[row]);
        // Eliminate
        for(int r=0;r<M;++r) if(r!=row&&mat[r][col])
            for(int c=0;c<N;++c) mat[r][c]^=mat[row][c];
        ++rank; ++row;
    }
    return rank;
}

// ─── Euler characteristic χ = Σ (-1)^k * β_k = Σ (-1)^k * #k-simplices ──────
template<int MAX_DIM, int MAX_SIMPLICES>
[[nodiscard]] constexpr i64 euler_characteristic(
    const SimplicialComplex<MAX_DIM,MAX_SIMPLICES>& K) noexcept
{
    i64 chi=0;
    for(int k=0;k<=MAX_DIM;++k) {
        i64 count=K.count(k);
        chi+=(k%2==0)?count:-count;
    }
    return chi;
}

// ─── Persistent homology (Vietoris-Rips style, pointcloud) ───────────────────
// Birth-death pairs for H_0 via union-find
template<int N>
struct PersistentH0 {
    std::array<int,N>  parent{};
    std::array<f64,N>  birth{};
    int n{N};

    constexpr void init() noexcept {
        for(int i=0;i<N;++i){parent[i]=i;birth[i]=0;}
    }

    constexpr int find(int x) noexcept {
        while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}
        return x;
    }

    // Add edge (u,v,distance); returns birth-death pair if merges components
    constexpr std::optional<std::pair<f64,f64>> add_edge(int u, int v, f64 d) noexcept {
        int pu=find(u),pv=find(v);
        if(pu==pv) return std::nullopt;
        // Merge smaller into larger
        if(birth[pu]<birth[pv]){parent[pv]=pu;return {{birth[pv],d}};}
        else                   {parent[pu]=pv;return {{birth[pu],d}};}
    }
};

// ─── Smith normal form (Z coefficients, for integer homology) ─────────────────
template<int M, int N>
[[nodiscard]] constexpr std::array<i64,M<N?M:N> smith_diagonal(
    std::array<std::array<i64,N>,M> D) noexcept
{
    constexpr int K=M<N?M:N;
    std::array<i64,K> diag{};
    for(int k=0;k<K;++k) {
        // Find nonzero element
        bool found=false;
        for(int i=k;i<M&&!found;++i) for(int j=k;j<N&&!found;++j)
            if(D[i][j]){if(i!=k)std::swap(D[i],D[k]);if(j!=k)for(int r=0;r<M;++r)std::swap(D[r][j],D[r][k]);found=true;}
        if(!found) break;
        // Reduce using gcd eliminations
        bool changed=true;
        while(changed){
            changed=false;
            for(int i=k+1;i<M;++i) if(D[i][k]){
                i64 q=D[i][k]/D[k][k];
                for(int j=k;j<N;++j) D[i][j]-=q*D[k][j];
                if(D[i][k]) changed=true;
            }
            for(int j=k+1;j<N;++j) if(D[k][j]){
                i64 q=D[k][j]/D[k][k];
                for(int i=k;i<M;++i) D[i][j]-=q*D[i][k];
                if(D[k][j]) changed=true;
            }
        }
        diag[k]=D[k][k];
    }
    return diag;
}

// ─── Topological data: Betti numbers b_k = dim H_k (mod 2) ───────────────────
// Given ranks of boundary matrices ∂_k and ∂_{k+1}:
// β_k = rank(C_k) - rank(∂_k) - rank(∂_{k+1})
[[nodiscard]] constexpr i64 betti_number(i64 chain_rank, i64 boundary_rank,
                                          i64 next_boundary_rank) noexcept {
    return chain_rank - boundary_rank - next_boundary_rank;
}

// ─── Fundamental polygon identification (surface topology) ────────────────────
// Surface type by Euler characteristic and orientability
enum class SurfaceType { Sphere, Torus, ProjectivePlane, KleinBottle, Other };

[[nodiscard]] constexpr SurfaceType classify_surface(i64 chi, bool orientable) noexcept {
    if(chi==2  && orientable)  return SurfaceType::Sphere;
    if(chi==0  && orientable)  return SurfaceType::Torus;
    if(chi==1  && !orientable) return SurfaceType::ProjectivePlane;
    if(chi==0  && !orientable) return SurfaceType::KleinBottle;
    return SurfaceType::Other;
}

} // namespace qs::math
