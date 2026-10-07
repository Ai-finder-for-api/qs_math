#pragma once
// qs_math/combinatorics/graph_combinatorics.hpp — Chromatic, Tutte, Ramsey, Matroid
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bitset>

namespace qs::math {

// ─── Simple graph (adjacency bitmask, N≤64) ───────────────────────────────────
template<int N>
struct SimpleGraph {
    std::array<u64,N> adj{}; // adj[i] = bitmask of neighbors
    int n{N};

    constexpr void add_edge(int u, int v) noexcept {
        adj[u]|=(1ULL<<v); adj[v]|=(1ULL<<u);
    }
    constexpr void remove_edge(int u, int v) noexcept {
        adj[u]&=~(1ULL<<v); adj[v]&=~(1ULL<<u);
    }
    [[nodiscard]] constexpr bool has_edge(int u, int v) const noexcept {
        return (adj[u]>>v)&1;
    }
    [[nodiscard]] constexpr int degree(int v) const noexcept {
        return __builtin_popcountll(adj[v]);
    }
    [[nodiscard]] constexpr int edge_count() const noexcept {
        int e=0; for(int i=0;i<n;++i) e+=degree(i); return e/2;
    }
    // Complement graph
    [[nodiscard]] constexpr SimpleGraph complement() const noexcept {
        SimpleGraph g; u64 all=(n<64)?(1ULL<<n)-1:~0ULL;
        for(int i=0;i<n;++i) g.adj[i]=(all&~adj[i])&~(1ULL<<i);
        return g;
    }
};

// ─── Chromatic polynomial P(G,k) via deletion-contraction ────────────────────
// P(G,k): number of proper k-colorings of G
template<int N>
[[nodiscard]] constexpr i64 chromatic_polynomial(SimpleGraph<N> G, i64 k) noexcept {
    // Base cases
    int e=G.edge_count();
    if(e==0) return 1; // all isolated: k^n would be computed by caller
    // Find an edge
    int u=-1,v=-1;
    for(int i=0;i<N&&u<0;++i) if(G.adj[i]) for(int j=i+1;j<N;++j) if(G.has_edge(i,j)){u=i;v=j;break;}
    if(u<0) return 1;
    // Delete edge
    SimpleGraph<N> Gd=G; Gd.remove_edge(u,v);
    // Contract edge (merge v into u)
    SimpleGraph<N> Gc=G; Gc.remove_edge(u,v);
    // Redirect v's edges to u
    for(int i=0;i<N;++i) if(i!=u&&i!=v&&Gc.has_edge(v,i)){Gc.add_edge(u,i);Gc.remove_edge(v,i);}
    Gc.adj[v]=0; // isolate v
    return chromatic_polynomial<N>(Gd,k)-chromatic_polynomial<N>(Gc,k);
}

// Chromatic number (minimum k such that P(G,k)>0)
template<int N>
[[nodiscard]] constexpr int chromatic_number(const SimpleGraph<N>& G) noexcept {
    for(int k=1;k<=N;++k) if(chromatic_polynomial(G,k)>0) return k;
    return N;
}

// ─── Independence number (max independent set size) ───────────────────────────
template<int N>
[[nodiscard]] constexpr int independence_number(const SimpleGraph<N>& G) noexcept {
    int best=0;
    // Enumerate all subsets (works for small N)
    for(u64 mask=0;mask<(1ULL<<N);++mask) {
        // Check if mask is independent
        bool ok=true;
        for(int i=0;i<N&&ok;++i) if((mask>>i)&1)
            if(G.adj[i]&mask) ok=false;
        if(ok) { int cnt=__builtin_popcountll(mask); if(cnt>best) best=cnt; }
    }
    return best;
}

// ─── Clique number (max clique = independence in complement) ──────────────────
template<int N>
[[nodiscard]] constexpr int clique_number(const SimpleGraph<N>& G) noexcept {
    return independence_number<N>(G.complement());
}

// ─── Ramsey numbers (exact, small) ───────────────────────────────────────────
// R(s,t): min n such that every 2-coloring of K_n has K_s in red or K_t in blue
inline constexpr std::array<std::array<int,7>,7> k_ramsey_table = {{
    {0,0,0,0,0,0,0},
    {0,1,2,3,4,5,6},
    {0,2,4,6,9,14,18},
    {0,3,6,9,14,18,23},
    {0,4,9,14,18,25,36},
    {0,5,14,18,25,43,48},
    {0,6,18,23,36,48,?}
}}; // Known values
[[nodiscard]] constexpr int ramsey_number(int s, int t) noexcept {
    if(s<=0||t<=0||s>6||t>6) return -1;
    return k_ramsey_table[s][t];
}

// ─── Tutte polynomial T(G;x,y) ───────────────────────────────────────────────
// T(G;x,y) = T(G-e;x,y) + T(G/e;x,y) for non-loop, non-bridge e
// Base: T(empty;x,y)=1, T(bridge;x,y)=x, T(loop;x,y)=y
template<int N>
[[nodiscard]] constexpr f64 tutte_polynomial(SimpleGraph<N> G, f64 x, f64 y) noexcept {
    // Find an edge that is neither bridge nor loop
    int u=-1,v=-1;
    for(int i=0;i<N&&u<0;++i) for(int j=i+1;j<N;++j) if(G.has_edge(i,j)){u=i;v=j;break;}
    if(u<0) return 1; // empty graph
    // Check if it's a bridge (connectivity via BFS)
    G.remove_edge(u,v);
    // BFS from u
    u64 vis=1ULL<<u; u64 q=vis;
    while(q){int s=__builtin_ctzll(q);q&=~(1ULL<<s);
              u64 new_n=G.adj[s]&~vis;vis|=new_n;q|=new_n;}
    bool is_bridge=!((vis>>v)&1);
    G.add_edge(u,v);
    if(is_bridge) {
        G.remove_edge(u,v);
        return x*tutte_polynomial<N>(G,x,y);
    }
    // Delete and contract
    SimpleGraph<N> Gd=G; Gd.remove_edge(u,v);
    SimpleGraph<N> Gc=G; Gc.remove_edge(u,v);
    for(int i=0;i<N;++i) if(i!=u&&i!=v&&Gc.has_edge(v,i)){Gc.add_edge(u,i);Gc.remove_edge(v,i);}
    Gc.adj[v]=0;
    return tutte_polynomial<N>(Gd,x,y)+tutte_polynomial<N>(Gc,x,y);
}

// ─── Matroid (graphic matroid: independent sets = acyclic subsets) ────────────
template<int N>
struct GraphicMatroid {
    const SimpleGraph<N>* G;

    // Is a set of edges (given as edge mask) independent (acyclic)?
    [[nodiscard]] constexpr bool is_independent(u64 edge_mask) const noexcept {
        // Build graph from selected edges and check for cycles via union-find
        std::array<int,N> parent{};
        for(int i=0;i<N;++i) parent[i]=i;
        std::function<int(int)> find=[&](int x){return parent[x]==x?x:parent[x]=find(parent[x]);};
        int eidx=0;
        for(int i=0;i<N&&(u64)eidx<64;++i) for(int j=i+1;j<N;++j) {
            if(G->has_edge(i,j)) {
                if((edge_mask>>eidx)&1) {
                    int pi=find(i),pj=find(j);
                    if(pi==pj) return false; // cycle
                    parent[pi]=pj;
                }
                ++eidx;
            }
        }
        return true;
    }

    // Rank function
    [[nodiscard]] constexpr int rank(u64 edge_mask) const noexcept {
        std::array<int,N> parent{};
        for(int i=0;i<N;++i) parent[i]=i;
        std::function<int(int)> find=[&](int x){return parent[x]==x?x:parent[x]=find(parent[x]);};
        int r=0,eidx=0;
        for(int i=0;i<N;++i) for(int j=i+1;j<N;++j) {
            if(G->has_edge(i,j)) {
                if((edge_mask>>eidx)&1) {
                    int pi=find(i),pj=find(j);
                    if(pi!=pj){parent[pi]=pj;++r;}
                }
                ++eidx;
            }
        }
        return r;
    }
};

// ─── Graph coloring (greedy) ──────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<int,N> greedy_coloring(const SimpleGraph<N>& G) noexcept {
    std::array<int,N> color{}; color.fill(-1);
    for(int v=0;v<N;++v) {
        u64 used=0;
        for(int u=0;u<N;++u) if(G.has_edge(v,u)&&color[u]>=0) used|=(1ULL<<color[u]);
        int c=0; while((used>>c)&1) ++c;
        color[v]=c;
    }
    return color;
}

// ─── Permanents and hafnians ──────────────────────────────────────────────────
// Permanent of N×N matrix (Ryser formula)
template<int N>
[[nodiscard]] constexpr i64 permanent(const std::array<std::array<i64,N>,N>& A) noexcept {
    i64 perm=0;
    for(u64 S=1;S<(1ULL<<N);++S) {
        i64 prod=(__builtin_popcountll(S)%2==0)?-1:1;
        for(int j=0;j<N;++j) {
            i64 row_sum=0;
            for(int i=0;i<N;++i) if((S>>i)&1) row_sum+=A[i][j];
            prod*=row_sum;
        }
        perm+=prod;
    }
    return (N%2==0)?perm:-perm;
}
} // namespace qs::math
