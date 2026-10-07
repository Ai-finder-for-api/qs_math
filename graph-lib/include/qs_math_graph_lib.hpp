
#pragma once
// qs_math/graph-lib/include/qs_math_graph_lib.hpp (C++20)
// Mathematical graph library: weighted directed/undirected graphs,
// shortest paths, MST, topological sort, SCC.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <functional>
namespace qs::math::graph {
using namespace qs::math;
constexpr f32 INF_F = std::numeric_limits<f32>::infinity();

struct Edge { i32 to; f32 weight{1}; };
struct Graph {
    i32 n_verts{0};
    std::vector<std::vector<Edge>> adj;
    explicit Graph(i32 n=0) : n_verts(n), adj(n) {}
    void add_edge(i32 u, i32 v, f32 w=1, bool directed=true) noexcept {
        adj[u].push_back({v,w});
        if(!directed) adj[v].push_back({u,w});
    }
};

// Dijkstra single-source shortest paths
std::vector<f32> dijkstra(const Graph& g, i32 src) noexcept;
// Bellman-Ford (handles negative weights; returns false if negative cycle)
bool bellman_ford(const Graph& g, i32 src, std::vector<f32>& dist) noexcept;
// Floyd-Warshall all-pairs shortest paths (n×n matrix, row-major)
std::vector<f32> floyd_warshall(const Graph& g) noexcept;
// Prim's MST (undirected, returns total weight and parent array)
std::pair<f32,std::vector<i32>> prim_mst(const Graph& g, i32 root=0) noexcept;
// Kruskal's MST
std::pair<f32,std::vector<std::pair<i32,i32>>> kruskal_mst(i32 n,
    std::vector<std::tuple<f32,i32,i32>> edges) noexcept;
// Topological sort (Kahn's algorithm; returns false if cycle)
bool topo_sort(const Graph& g, std::vector<i32>& order) noexcept;
// Tarjan's SCC — returns component id per vertex
std::vector<i32> tarjan_scc(const Graph& g) noexcept;
// BFS/DFS
std::vector<i32> bfs(const Graph& g, i32 src) noexcept;
std::vector<i32> dfs(const Graph& g, i32 src) noexcept;
// Max-flow (Dinic's algorithm)
f32 max_flow(Graph& g, i32 s, i32 t) noexcept;
// Bipartite matching (Hopcroft-Karp)
i32 bipartite_matching(i32 nl, i32 nr,
    const std::vector<std::pair<i32,i32>>& edges) noexcept;
} // namespace qs::math::graph
