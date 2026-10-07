#pragma once
// graph_theory.hpp — Graph mathematics: adjacency, shortest paths, spanning trees,
//                    graph metrics, topological sort, strongly connected components,
//                    flow networks, matching, graph isomorphism helpers
#include "scalar_types.hpp"
#include "scalar_math.hpp"

namespace qs::math::graph {

static constexpr i32 MAX_VERTS = 512;
static constexpr f32 INF_DIST  = 1e30f;

// ---------------------------------------------------------------------------
// Graph representation
// ---------------------------------------------------------------------------
struct Edge { i32 from, to; f32 weight; };

struct AdjMatrix {
    f32  w[MAX_VERTS][MAX_VERTS];
    i32  n{0};

    constexpr AdjMatrix() noexcept {
        for(i32 i=0;i<MAX_VERTS;++i) for(i32 j=0;j<MAX_VERTS;++j) w[i][j]=(i==j)?0.f:INF_DIST;
    }
    void set_edge(i32 u,i32 v,f32 wt,bool directed=true) noexcept {
        w[u][v]=wt; if(!directed) w[v][u]=wt; n=max(n,max(u,v)+1);
    }
    [[nodiscard]] constexpr bool has_edge(i32 u,i32 v) const noexcept { return w[u][v]<INF_DIST; }
};

struct AdjList {
    struct Neighbor { i32 to; f32 weight; };
    static constexpr i32 MAX_EDGES=4096;
    Neighbor edges[MAX_EDGES];
    i32 head[MAX_VERTS];
    i32 next[MAX_EDGES];
    i32 n{0}, m{0};

    constexpr AdjList() noexcept {
        for(i32 i=0;i<MAX_VERTS;++i) head[i]=-1;
        for(i32 i=0;i<MAX_EDGES;++i) next[i]=-1;
    }
    void add_edge(i32 u,i32 v,f32 w=1.f) noexcept {
        if(m>=MAX_EDGES) return;
        edges[m]={v,w}; next[m]=head[u]; head[u]=m++; n=max(n,max(u,v)+1);
    }
    void add_undirected(i32 u,i32 v,f32 w=1.f) noexcept { add_edge(u,v,w); add_edge(v,u,w); }
};

// ---------------------------------------------------------------------------
// Shortest paths
// ---------------------------------------------------------------------------
// Dijkstra (via simple priority queue simulation, no heap, O(V^2))
inline void dijkstra(const AdjList& g, i32 src, f32* dist, i32* prev) noexcept {
    bool visited[MAX_VERTS]={};
    for(i32 i=0;i<g.n;++i){ dist[i]=INF_DIST; prev[i]=-1; }
    dist[src]=0.f;
    for(i32 iter=0;iter<g.n;++iter){
        // Find minimum unvisited
        i32 u=-1;
        for(i32 i=0;i<g.n;++i) if(!visited[i]&&(u<0||dist[i]<dist[u])) u=i;
        if(u<0||dist[u]==INF_DIST) break;
        visited[u]=true;
        for(i32 e=g.head[u];e>=0;e=g.next[e]){
            i32 v=g.edges[e].to;
            f32 d=dist[u]+g.edges[e].weight;
            if(d<dist[v]){ dist[v]=d; prev[v]=u; }
        }
    }
}

// Bellman-Ford (handles negative weights, detects negative cycles)
[[nodiscard]] inline bool bellman_ford(const Edge* edges, i32 m, i32 n, i32 src,
                                        f32* dist, i32* prev) noexcept
{
    for(i32 i=0;i<n;++i){ dist[i]=INF_DIST; prev[i]=-1; }
    dist[src]=0.f;
    for(i32 iter=0;iter<n-1;++iter)
        for(i32 e=0;e<m;++e){
            if(dist[edges[e].from]<INF_DIST){
                f32 d=dist[edges[e].from]+edges[e].weight;
                if(d<dist[edges[e].to]){ dist[edges[e].to]=d; prev[edges[e].to]=edges[e].from; }
            }
        }
    // Check for negative cycles
    for(i32 e=0;e<m;++e)
        if(dist[edges[e].from]<INF_DIST&&dist[edges[e].from]+edges[e].weight<dist[edges[e].to])
            return false;
    return true;
}

// Floyd-Warshall (all-pairs shortest paths)
inline void floyd_warshall(f32 dist[][MAX_VERTS], i32 n) noexcept {
    for(i32 k=0;k<n;++k) for(i32 i=0;i<n;++i) for(i32 j=0;j<n;++j)
        if(dist[i][k]<INF_DIST&&dist[k][j]<INF_DIST)
            dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
}

// A* (heuristic shortest path)
template<typename HFn>
inline f32 astar(const AdjList& g, i32 src, i32 dst, f32* dist, i32* prev, HFn&& h) noexcept {
    bool visited[MAX_VERTS]={};
    f32  f[MAX_VERTS];
    for(i32 i=0;i<g.n;++i){ dist[i]=INF_DIST; f[i]=INF_DIST; prev[i]=-1; }
    dist[src]=0.f; f[src]=h(src);
    for(i32 iter=0;iter<g.n;++iter){
        i32 u=-1;
        for(i32 i=0;i<g.n;++i) if(!visited[i]&&(u<0||f[i]<f[u])) u=i;
        if(u<0||u==dst) break;
        visited[u]=true;
        for(i32 e=g.head[u];e>=0;e=g.next[e]){
            i32 v=g.edges[e].to;
            f32 d=dist[u]+g.edges[e].weight;
            if(d<dist[v]){ dist[v]=d; prev[v]=u; f[v]=d+h(v); }
        }
    }
    return dist[dst];
}

// ---------------------------------------------------------------------------
// Minimum spanning tree (Kruskal)
// ---------------------------------------------------------------------------
struct UnionFind {
    i32 parent[MAX_VERTS], rank_[MAX_VERTS];
    constexpr UnionFind() noexcept { for(i32 i=0;i<MAX_VERTS;++i){parent[i]=i;rank_[i]=0;} }
    i32 find(i32 x) noexcept { return parent[x]==x?x:(parent[x]=find(parent[x])); }
    bool unite(i32 a,i32 b) noexcept {
        a=find(a);b=find(b);if(a==b)return false;
        if(rank_[a]<rank_[b])swap(a,b);
        parent[b]=a;if(rank_[a]==rank_[b])++rank_[a];return true;
    }
    static void swap(i32& a,i32& b){i32 t=a;a=b;b=t;}
};

inline f32 kruskal_mst(Edge* edges, i32 m, i32 n, Edge* mst_edges, i32& mst_size) noexcept {
    // Sort edges by weight
    for(i32 i=1;i<m;++i){Edge k=edges[i];i32 j=i-1;
        while(j>=0&&edges[j].weight>k.weight){edges[j+1]=edges[j];--j;}edges[j+1]=k;}
    UnionFind uf; mst_size=0; f32 total=0.f;
    for(i32 i=0;i<m&&mst_size<n-1;++i)
        if(uf.unite(edges[i].from,edges[i].to)){ mst_edges[mst_size++]=edges[i]; total+=edges[i].weight; }
    return total;
}

// ---------------------------------------------------------------------------
// Topological sort (Kahn's algorithm)
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool topological_sort(const AdjList& g, i32* order, i32& out_n) noexcept {
    i32 in_degree[MAX_VERTS]={};
    for(i32 u=0;u<g.n;++u) for(i32 e=g.head[u];e>=0;e=g.next[e]) ++in_degree[g.edges[e].to];
    i32 queue[MAX_VERTS]; i32 qh=0,qt=0;
    for(i32 i=0;i<g.n;++i) if(in_degree[i]==0) queue[qt++]=i;
    out_n=0;
    while(qh<qt){
        i32 u=queue[qh++]; order[out_n++]=u;
        for(i32 e=g.head[u];e>=0;e=g.next[e])
            if(--in_degree[g.edges[e].to]==0) queue[qt++]=g.edges[e].to;
    }
    return out_n==g.n; // false = cycle exists
}

// ---------------------------------------------------------------------------
// Strongly connected components (Tarjan's algorithm)
// ---------------------------------------------------------------------------
struct SCCResult {
    i32 comp[MAX_VERTS];  // component id for each vertex
    i32 n_comp{0};
};

struct TarjanState {
    i32 idx[MAX_VERTS],low[MAX_VERTS],stack[MAX_VERTS];
    bool on_stack[MAX_VERTS];
    i32 index{0},sp{0};
};

inline void tarjan_dfs(const AdjList& g, i32 u, TarjanState& s, SCCResult& r) noexcept {
    s.idx[u]=s.low[u]=s.index++;
    s.stack[s.sp++]=u; s.on_stack[u]=true;
    for(i32 e=g.head[u];e>=0;e=g.next[e]){
        i32 v=g.edges[e].to;
        if(s.idx[v]<0){ tarjan_dfs(g,v,s,r); s.low[u]=min(s.low[u],s.low[v]); }
        else if(s.on_stack[v]) s.low[u]=min(s.low[u],s.idx[v]);
    }
    if(s.low[u]==s.idx[u]){
        i32 v; do { v=s.stack[--s.sp]; s.on_stack[v]=false; r.comp[v]=r.n_comp; } while(v!=u);
        ++r.n_comp;
    }
}

inline SCCResult tarjan_scc(const AdjList& g) noexcept {
    SCCResult r; TarjanState s;
    for(i32 i=0;i<MAX_VERTS;++i){ s.idx[i]=-1; s.low[i]=0; s.on_stack[i]=false; r.comp[i]=-1; }
    for(i32 u=0;u<g.n;++u) if(s.idx[u]<0) tarjan_dfs(g,u,s,r);
    return r;
}

// ---------------------------------------------------------------------------
// Maximum bipartite matching (Hopcroft-Karp, simplified)
// ---------------------------------------------------------------------------
[[nodiscard]] inline i32 bipartite_matching(
    const bool* adj, i32 n_left, i32 n_right, i32* match_l, i32* match_r) noexcept
{
    for(i32 i=0;i<n_left;++i) match_l[i]=-1;
    for(i32 j=0;j<n_right;++j) match_r[j]=-1;
    i32 matching=0;
    for(i32 u=0;u<n_left;++u){
        bool visited[MAX_VERTS]={};
        // DFS augmenting path
        std::function<bool(i32)> dfs=[&](i32 v)->bool{
            for(i32 w=0;w<n_right;++w){
                if(!adj[v*n_right+w]||visited[w]) continue;
                visited[w]=true;
                if(match_r[w]<0||dfs(match_r[w])){ match_l[v]=w;match_r[w]=v;return true; }
            }
            return false;
        };
        if(dfs(u)) ++matching;
    }
    return matching;
}

// ---------------------------------------------------------------------------
// Graph metrics
// ---------------------------------------------------------------------------
// Diameter (longest shortest path)
[[nodiscard]] inline f32 graph_diameter(const AdjMatrix& g) noexcept {
    f32 dist[MAX_VERTS][MAX_VERTS];
    for(i32 i=0;i<g.n;++i) for(i32 j=0;j<g.n;++j) dist[i][j]=g.w[i][j];
    floyd_warshall(dist,g.n);
    f32 d=0.f;
    for(i32 i=0;i<g.n;++i) for(i32 j=0;j<g.n;++j)
        if(dist[i][j]<INF_DIST) d=max(d,dist[i][j]);
    return d;
}

// Degree sequence
inline void degree_sequence(const AdjList& g, i32* out_deg, i32* in_deg) noexcept {
    for(i32 i=0;i<g.n;++i) out_deg[i]=in_deg[i]=0;
    for(i32 u=0;u<g.n;++u) for(i32 e=g.head[u];e>=0;e=g.next[e])
        { ++out_deg[u]; ++in_deg[g.edges[e].to]; }
}

// Clustering coefficient of vertex u
[[nodiscard]] inline f32 clustering_coefficient(const AdjMatrix& g, i32 u) noexcept {
    i32 neighbors[MAX_VERTS]; i32 k=0;
    for(i32 v=0;v<g.n;++v) if(v!=u&&g.has_edge(u,v)) neighbors[k++]=v;
    if(k<2) return 0.f;
    i32 triangles=0;
    for(i32 i=0;i<k;++i) for(i32 j=i+1;j<k;++j)
        if(g.has_edge(neighbors[i],neighbors[j])) ++triangles;
    return f32(2*triangles)/f32(k*(k-1));
}

} // namespace qs::math::graph
