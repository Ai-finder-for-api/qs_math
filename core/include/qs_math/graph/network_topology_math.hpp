#pragma once
// qs_math/network_topology_math.hpp — Graph centrality, spectral, Laplacian, community detection (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <span>
#include <numeric>
#include <array>

namespace qs {

// ─── Graph representation ─────────────────────────────────────────────────────

struct GraphEdge { u32 from{},to{}; f32 weight{1.f}; };

struct Graph {
    u32 n_nodes{};
    std::vector<std::vector<std::pair<u32,f32>>> adj; // adjacency list

    void resize(u32 n) { n_nodes=n; adj.resize(n); }

    void add_edge(u32 u, u32 v, f32 w=1.f, bool directed=false) {
        adj[u].push_back({v,w});
        if(!directed) adj[v].push_back({u,w});
    }

    [[nodiscard]] u32 degree(u32 v) const { return u32(adj[v].size()); }

    [[nodiscard]] f32 weighted_degree(u32 v) const {
        f32 d=0.f; for(auto[nb,w]:adj[v]) d+=w; return d;
    }
};

// ─── BFS / SSSP ───────────────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<f32> dijkstra(const Graph& g, u32 src) {
    std::vector<f32> dist(g.n_nodes, std::numeric_limits<f32>::max());
    dist[src]=0.f;
    std::priority_queue<std::pair<f32,u32>,std::vector<std::pair<f32,u32>>,std::greater<>> pq;
    pq.push({0.f,src});
    while(!pq.empty()){
        auto[d,u]=pq.top(); pq.pop();
        if(d>dist[u]) continue;
        for(auto[v,w]:g.adj[u]) if(dist[u]+w<dist[v]){dist[v]=dist[u]+w;pq.push({dist[v],v});}
    }
    return dist;
}

[[nodiscard]] inline std::vector<u32> bfs_order(const Graph& g, u32 src) {
    std::vector<bool> vis(g.n_nodes,false);
    std::vector<u32> order;
    std::queue<u32> q;
    q.push(src); vis[src]=true;
    while(!q.empty()){
        u32 u=q.front();q.pop();order.push_back(u);
        for(auto[v,w]:g.adj[u]) if(!vis[v]){vis[v]=true;q.push(v);}
    }
    return order;
}

// ─── Centrality measures ──────────────────────────────────────────────────────

// Degree centrality
[[nodiscard]] inline std::vector<f32> degree_centrality(const Graph& g) {
    std::vector<f32> c(g.n_nodes);
    f32 norm=f32(g.n_nodes-1);
    for(u32 v=0;v<g.n_nodes;++v) c[v]=f32(g.degree(v))/(norm+1e-8f);
    return c;
}

// Closeness centrality: 1 / avg shortest path length
[[nodiscard]] inline std::vector<f32> closeness_centrality(const Graph& g) {
    std::vector<f32> c(g.n_nodes,0.f);
    for(u32 v=0;v<g.n_nodes;++v){
        auto dist=dijkstra(g,v);
        f32 sum=0.f; u32 reachable=0;
        for(u32 u=0;u<g.n_nodes;++u) if(u!=v&&dist[u]<1e30f){sum+=dist[u];++reachable;}
        c[v]=reachable>0?f32(reachable)*(f32(reachable)/f32(g.n_nodes-1))/sum:0.f;
    }
    return c;
}

// Betweenness centrality (Brandes 2001 exact)
[[nodiscard]] inline std::vector<f32> betweenness_centrality(const Graph& g) {
    std::vector<f32> cb(g.n_nodes,0.f);
    for(u32 s=0;s<g.n_nodes;++s){
        std::vector<std::vector<u32>> P(g.n_nodes);
        std::vector<f32> sigma(g.n_nodes,0.f); sigma[s]=1.f;
        std::vector<f32> dist2(g.n_nodes,-1.f); dist2[s]=0.f;
        std::queue<u32> Q; Q.push(s);
        std::vector<u32> S;
        while(!Q.empty()){
            u32 v=Q.front();Q.pop();S.push_back(v);
            for(auto[w,wt]:g.adj[v]){
                if(dist2[w]<0.f){dist2[w]=dist2[v]+1.f;Q.push(w);}
                if(dist2[w]==dist2[v]+1.f){sigma[w]+=sigma[v];P[w].push_back(v);}
            }
        }
        std::vector<f32> delta(g.n_nodes,0.f);
        while(!S.empty()){
            u32 w=S.back();S.pop_back();
            for(u32 v:P[w]) delta[v]+=sigma[v]/sigma[w]*(1.f+delta[w]);
            if(w!=s) cb[w]+=delta[w];
        }
    }
    f32 norm=f32((g.n_nodes-1)*(g.n_nodes-2));
    if(norm>0.f) for(auto&v:cb) v/=norm;
    return cb;
}

// PageRank
[[nodiscard]] inline std::vector<f32> pagerank(
    const Graph& g, f32 damping=0.85f, u32 iters=50)
{
    u32 N=g.n_nodes;
    std::vector<f32> rank(N,1.f/f32(N));
    std::vector<f32> new_rank(N);
    for(u32 iter=0;iter<iters;++iter){
        std::fill(new_rank.begin(),new_rank.end(),(1.f-damping)/f32(N));
        for(u32 v=0;v<N;++v){
            f32 deg=f32(g.degree(v));
            if(deg>0.f) for(auto[nb,w]:g.adj[v])
                new_rank[nb]+=damping*rank[v]/deg;
        }
        rank=new_rank;
    }
    return rank;
}

// ─── Spectral analysis ────────────────────────────────────────────────────────

// Graph Laplacian: L = D - A
[[nodiscard]] inline std::vector<f32> laplacian_row(const Graph& g, u32 v) {
    std::vector<f32> row(g.n_nodes,0.f);
    row[v]=f32(g.degree(v));
    for(auto[nb,w]:g.adj[v]) row[nb]=-1.f;
    return row;
}

// Fiedler value approximation via power iteration on L
[[nodiscard]] inline f32 algebraic_connectivity(const Graph& g, u32 iters=50) {
    u32 N=g.n_nodes;
    std::vector<f32> v(N,1.f/std::sqrt(f32(N)));
    // Subtract mean each step (project out zero eigenvalue)
    for(u32 iter=0;iter<iters;++iter){
        std::vector<f32> Lv(N,0.f);
        for(u32 i=0;i<N;++i){Lv[i]+=f32(g.degree(i))*v[i];for(auto[nb,w]:g.adj[i])Lv[i]-=v[nb];}
        f32 mean=0.f; for(auto x:Lv) mean+=x; mean/=f32(N);
        for(auto&x:Lv) x-=mean;
        f32 norm2=0.f; for(auto x:Lv) norm2+=x*x;
        f32 norm=std::sqrt(norm2+1e-12f);
        for(auto&x:Lv) x/=norm;
        v=Lv;
    }
    // Rayleigh quotient
    std::vector<f32> Lv(N,0.f);
    for(u32 i=0;i<N;++i){Lv[i]+=f32(g.degree(i))*v[i];for(auto[nb,w]:g.adj[i])Lv[i]-=v[nb];}
    f32 num=0.f,den=0.f;
    for(u32 i=0;i<N;++i){num+=v[i]*Lv[i];den+=v[i]*v[i];}
    return num/(den+1e-12f);
}

// ─── Community detection (Louvain greedy) ─────────────────────────────────────

[[nodiscard]] inline std::vector<u32> louvain_communities(
    const Graph& g, u32 max_passes=10)
{
    u32 N=g.n_nodes;
    std::vector<u32> community(N);
    std::iota(community.begin(),community.end(),0);
    f32 m=0.f;
    for(u32 v=0;v<N;++v) m+=f32(g.degree(v));
    m*=0.5f;

    for(u32 pass=0;pass<max_passes;++pass){
        bool improved=false;
        for(u32 v=0;v<N;++v){
            u32 best_c=community[v];
            f32 best_dQ=0.f;
            std::unordered_map<u32,f32> c_weight;
            for(auto[nb,w]:g.adj[v]) c_weight[community[nb]]+=w;
            for(auto[c,kc]:c_weight){
                f32 dQ=kc-f32(g.degree(v))*f32(g.degree(v))/(2.f*m+1e-8f);
                if(dQ>best_dQ){best_dQ=dQ;best_c=c;}
            }
            if(best_c!=community[v]){community[v]=best_c;improved=true;}
        }
        if(!improved) break;
    }
    // Renumber communities 0..K
    std::unordered_map<u32,u32> remap; u32 idx=0;
    for(auto&c:community){if(!remap.count(c))remap[c]=idx++;c=remap[c];}
    return community;
}

// ─── Small-world / scale-free metrics ────────────────────────────────────────

[[nodiscard]] inline f32 clustering_coefficient(const Graph& g, u32 v) {
    u32 k=g.degree(v);
    if(k<2) return 0.f;
    u32 triangles=0;
    for(auto[u,wu]:g.adj[v])
    for(auto[w,ww]:g.adj[v]) {
        if(u==w) continue;
        for(auto[nb,wn]:g.adj[u]) if(nb==w){++triangles;break;}
    }
    return f32(triangles)/(f32(k*(k-1)));
}

[[nodiscard]] inline f32 average_clustering(const Graph& g) {
    f32 sum=0.f;
    for(u32 v=0;v<g.n_nodes;++v) sum+=clustering_coefficient(g,v);
    return sum/f32(g.n_nodes);
}

// Power law exponent estimate via MLE (Clauset 2009)
[[nodiscard]] inline f32 power_law_exponent(std::span<const u32> degrees, u32 k_min=1) {
    u32 n=0; f32 sum=0.f;
    for(u32 d:degrees) if(d>=k_min){++n;sum+=std::log(f32(d)/(f32(k_min)-0.5f));}
    return n>0?1.f+f32(n)/sum:2.f;
}

} // namespace qs
