#pragma once
// qs_math/graph/graph_algorithms.hpp — Dijkstra, A*, Bellman-Ford, Floyd-Warshall,
//   Kruskal MST, topological sort, SCC (Kosaraju), bipartite check (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <span>
#include <functional>

namespace qs::math::graph {

inline constexpr f32 INF_DIST = std::numeric_limits<f32>::infinity();
inline constexpr u32 NO_VERTEX = 0xFFFFFFFFu;

// ─── Adjacency structures ─────────────────────────────────────────────────
struct WeightedEdge { u32 to; f32 weight; };
struct Edge         { u32 u, v; f32 weight; };

using AdjList = std::vector<std::vector<WeightedEdge>>;

[[nodiscard]] inline AdjList make_adj_list(u32 n_vertices) { return AdjList(n_vertices); }

inline void add_edge(AdjList& g, u32 u, u32 v, f32 w, bool directed=true) {
    g[u].push_back({v, w});
    if (!directed) g[v].push_back({u, w});
}

// ─── Dijkstra (single-source shortest path, non-negative weights) ──────────
struct ShortestPath {
    std::vector<f32> dist;
    std::vector<u32> prev;

    [[nodiscard]] std::vector<u32> path_to(u32 target) const {
        std::vector<u32> p;
        for (u32 v = target; v != NO_VERTEX; v = prev[v]) p.push_back(v);
        std::reverse(p.begin(), p.end());
        return p;
    }
};

[[nodiscard]] inline ShortestPath dijkstra(const AdjList& g, u32 src) {
    u32 n = u32(g.size());
    ShortestPath sp;
    sp.dist.assign(n, INF_DIST);
    sp.prev.assign(n, NO_VERTEX);
    sp.dist[src] = 0.f;

    using P = std::pair<f32,u32>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    pq.push({0.f, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > sp.dist[u]) continue;
        for (auto& e : g[u]) {
            f32 nd = sp.dist[u] + e.weight;
            if (nd < sp.dist[e.to]) {
                sp.dist[e.to] = nd;
                sp.prev[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }
    return sp;
}

// ─── A* (heuristic shortest path) ─────────────────────────────────────────
template<typename Heuristic>
[[nodiscard]] inline std::vector<u32> astar(
    const AdjList& g, u32 src, u32 dst, Heuristic&& h)
{
    u32 n = u32(g.size());
    std::vector<f32> g_cost(n, INF_DIST), f_cost(n, INF_DIST);
    std::vector<u32> prev(n, NO_VERTEX);
    g_cost[src] = 0.f;
    f_cost[src] = h(src, dst);

    using P = std::pair<f32,u32>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> open;
    open.push({f_cost[src], src});

    while (!open.empty()) {
        auto [fc, u] = open.top(); open.pop();
        if (u == dst) {
            std::vector<u32> path;
            for (u32 v = dst; v != NO_VERTEX; v = prev[v]) path.push_back(v);
            std::reverse(path.begin(), path.end());
            return path;
        }
        if (fc > f_cost[u]) continue;
        for (auto& e : g[u]) {
            f32 ng = g_cost[u] + e.weight;
            if (ng < g_cost[e.to]) {
                g_cost[e.to] = ng;
                f_cost[e.to] = ng + h(e.to, dst);
                prev[e.to]   = u;
                open.push({f_cost[e.to], e.to});
            }
        }
    }
    return {}; // no path
}

// ─── Bellman-Ford (handles negative weights, detects negative cycles) ──────
struct BellmanFordResult {
    std::vector<f32> dist;
    std::vector<u32> prev;
    bool has_negative_cycle;
};

[[nodiscard]] inline BellmanFordResult bellman_ford(
    u32 n, const std::vector<Edge>& edges, u32 src)
{
    BellmanFordResult r;
    r.dist.assign(n, INF_DIST);
    r.prev.assign(n, NO_VERTEX);
    r.has_negative_cycle = false;
    r.dist[src] = 0.f;

    for (u32 i = 0; i < n-1; ++i)
        for (auto& e : edges)
            if (r.dist[e.u] < INF_DIST && r.dist[e.u]+e.weight < r.dist[e.v]) {
                r.dist[e.v] = r.dist[e.u]+e.weight;
                r.prev[e.v] = e.u;
            }
    // Check for negative cycles
    for (auto& e : edges)
        if (r.dist[e.u] < INF_DIST && r.dist[e.u]+e.weight < r.dist[e.v])
            r.has_negative_cycle = true;
    return r;
}

// ─── Floyd-Warshall (all-pairs shortest paths) ────────────────────────────
struct AllPairsResult {
    std::vector<std::vector<f32>> dist;
    std::vector<std::vector<u32>> next;
    u32 n;

    [[nodiscard]] std::vector<u32> path(u32 u, u32 v) const {
        if (dist[u][v] >= INF_DIST) return {};
        std::vector<u32> p = {u};
        while (u != v) { u = next[u][v]; p.push_back(u); }
        return p;
    }
};

[[nodiscard]] inline AllPairsResult floyd_warshall(u32 n, const std::vector<Edge>& edges) {
    AllPairsResult r;
    r.n = n;
    r.dist.assign(n, std::vector<f32>(n, INF_DIST));
    r.next.assign(n, std::vector<u32>(n, NO_VERTEX));
    for (u32 i = 0; i < n; ++i) r.dist[i][i] = 0.f;
    for (auto& e : edges) {
        if (e.weight < r.dist[e.u][e.v]) {
            r.dist[e.u][e.v] = e.weight;
            r.next[e.u][e.v] = e.v;
        }
    }
    for (u32 k = 0; k < n; ++k)
        for (u32 i = 0; i < n; ++i)
            for (u32 j = 0; j < n; ++j)
                if (r.dist[i][k] < INF_DIST && r.dist[k][j] < INF_DIST) {
                    f32 nd = r.dist[i][k] + r.dist[k][j];
                    if (nd < r.dist[i][j]) {
                        r.dist[i][j] = nd;
                        r.next[i][j] = r.next[i][k];
                    }
                }
    return r;
}

// ─── Kruskal MST ──────────────────────────────────────────────────────────
struct UnionFind {
    std::vector<u32> parent, rank;
    UnionFind(u32 n) : parent(n), rank(n,0) { for(u32 i=0;i<n;++i) parent[i]=i; }
    u32 find(u32 x) { return parent[x]==x ? x : parent[x]=find(parent[x]); }
    bool unite(u32 a, u32 b) {
        a=find(a); b=find(b); if(a==b) return false;
        if(rank[a]<rank[b]) std::swap(a,b);
        parent[b]=a; if(rank[a]==rank[b]) ++rank[a];
        return true;
    }
};

[[nodiscard]] inline std::vector<Edge> kruskal_mst(u32 n, std::vector<Edge> edges) {
    std::sort(edges.begin(), edges.end(), [](auto& a, auto& b){ return a.weight<b.weight; });
    UnionFind uf(n);
    std::vector<Edge> mst;
    mst.reserve(n-1);
    for (auto& e : edges) {
        if (uf.unite(e.u, e.v)) {
            mst.push_back(e);
            if (mst.size() == n-1) break;
        }
    }
    return mst;
}

// ─── Topological sort (Kahn's algorithm) ──────────────────────────────────
// Returns empty if cycle detected
[[nodiscard]] inline std::vector<u32> topological_sort(u32 n, const AdjList& g) {
    std::vector<u32> indegree(n, 0);
    for (u32 u = 0; u < n; ++u) for (auto& e : g[u]) ++indegree[e.to];
    std::queue<u32> q;
    for (u32 i = 0; i < n; ++i) if (indegree[i]==0) q.push(i);
    std::vector<u32> order;
    while (!q.empty()) {
        u32 u = q.front(); q.pop();
        order.push_back(u);
        for (auto& e : g[u]) if (--indegree[e.to]==0) q.push(e.to);
    }
    return (order.size()==n) ? order : std::vector<u32>{};
}

// ─── Strongly connected components (Kosaraju's algorithm) ─────────────────
[[nodiscard]] inline std::vector<std::vector<u32>> kosaraju_scc(u32 n, const AdjList& g) {
    std::vector<u32> order;
    std::vector<bool> visited(n, false);
    std::function<void(u32)> dfs1 = [&](u32 u) {
        visited[u] = true;
        for (auto& e : g[u]) if (!visited[e.to]) dfs1(e.to);
        order.push_back(u);
    };
    for (u32 i = 0; i < n; ++i) if (!visited[i]) dfs1(i);
    // Build transpose
    AdjList gt(n);
    for (u32 u = 0; u < n; ++u) for (auto& e : g[u]) gt[e.to].push_back({u, e.weight});
    std::fill(visited.begin(), visited.end(), false);
    std::vector<std::vector<u32>> sccs;
    std::function<void(u32, std::vector<u32>&)> dfs2 = [&](u32 u, std::vector<u32>& comp) {
        visited[u] = true; comp.push_back(u);
        for (auto& e : gt[u]) if (!visited[e.to]) dfs2(e.to, comp);
    };
    for (int i = int(order.size())-1; i >= 0; --i) {
        if (!visited[order[i]]) {
            sccs.emplace_back();
            dfs2(order[i], sccs.back());
        }
    }
    return sccs;
}

// ─── Bipartite check (2-coloring via BFS) ────────────────────────────────
[[nodiscard]] inline bool is_bipartite(u32 n, const AdjList& g,
    std::vector<int>* coloring = nullptr)
{
    std::vector<int> color(n, -1);
    for (u32 start = 0; start < n; ++start) {
        if (color[start] != -1) continue;
        color[start] = 0;
        std::queue<u32> q;
        q.push(start);
        while (!q.empty()) {
            u32 u = q.front(); q.pop();
            for (auto& e : g[u]) {
                if (color[e.to] == -1) { color[e.to] = 1-color[u]; q.push(e.to); }
                else if (color[e.to] == color[u]) return false;
            }
        }
    }
    if (coloring) *coloring = color;
    return true;
}

// ─── Graph diameter ───────────────────────────────────────────────────────
// Approximate via double BFS from arbitrary source
[[nodiscard]] inline f32 graph_diameter(const AdjList& g) {
    u32 n = u32(g.size());
    if (n == 0) return 0.f;
    auto sp1 = dijkstra(g, 0);
    u32 farthest = 0;
    for (u32 i = 1; i < n; ++i) if (sp1.dist[i] > sp1.dist[farthest]) farthest=i;
    auto sp2 = dijkstra(g, farthest);
    f32 diam = 0.f;
    for (f32 d : sp2.dist) if (d < INF_DIST) diam = max(diam, d);
    return diam;
}

} // namespace qs::math::graph
