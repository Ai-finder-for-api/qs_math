
// qs_math/graph-lib/graph_lib_impl.cpp (C++20)
#include "include/qs_math_graph_lib.hpp"
#include <queue>
#include <stack>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::graph {

std::vector<f32> dijkstra(const Graph& g, i32 src) noexcept {
    std::vector<f32> dist(g.n_verts, INF_F);
    dist[src]=0;
    std::priority_queue<std::pair<f32,i32>,std::vector<std::pair<f32,i32>>,std::greater<>> pq;
    pq.push({0,src});
    while(!pq.empty()){
        auto[d,u]=pq.top();pq.pop();
        if(d>dist[u]) continue;
        for(auto&e:g.adj[u]) if(dist[u]+e.weight<dist[e.to]){
            dist[e.to]=dist[u]+e.weight; pq.push({dist[e.to],e.to});
        }
    }
    return dist;
}

bool bellman_ford(const Graph& g,i32 src,std::vector<f32>& dist) noexcept {
    dist.assign(g.n_verts,INF_F); dist[src]=0;
    for(i32 iter=0;iter<g.n_verts-1;++iter)
        for(i32 u=0;u<g.n_verts;++u) if(dist[u]<INF_F)
            for(auto&e:g.adj[u]) if(dist[u]+e.weight<dist[e.to]) dist[e.to]=dist[u]+e.weight;
    for(i32 u=0;u<g.n_verts;++u) if(dist[u]<INF_F)
        for(auto&e:g.adj[u]) if(dist[u]+e.weight<dist[e.to]) return false;
    return true;
}

std::vector<f32> floyd_warshall(const Graph& g) noexcept {
    i32 n=g.n_verts;
    std::vector<f32> d(n*n,INF_F);
    for(i32 i=0;i<n;++i) d[i*n+i]=0;
    for(i32 u=0;u<n;++u) for(auto&e:g.adj[u]) d[u*n+e.to]=std::min(d[u*n+e.to],e.weight);
    for(i32 k=0;k<n;++k) for(i32 i=0;i<n;++i) for(i32 j=0;j<n;++j)
        if(d[i*n+k]<INF_F&&d[k*n+j]<INF_F) d[i*n+j]=std::min(d[i*n+j],d[i*n+k]+d[k*n+j]);
    return d;
}

std::pair<f32,std::vector<i32>> prim_mst(const Graph& g,i32 root) noexcept {
    i32 n=g.n_verts;
    std::vector<f32> key(n,INF_F);
    std::vector<i32> par(n,-1);
    std::vector<bool> inMST(n,false);
    key[root]=0;
    std::priority_queue<std::pair<f32,i32>,std::vector<std::pair<f32,i32>>,std::greater<>> pq;
    pq.push({0,root});
    f32 total=0;
    while(!pq.empty()){
        auto[w,u]=pq.top();pq.pop();
        if(inMST[u]) continue;
        inMST[u]=true; total+=w;
        for(auto&e:g.adj[u]) if(!inMST[e.to]&&e.weight<key[e.to]){
            key[e.to]=e.weight; par[e.to]=u; pq.push({e.weight,e.to});
        }
    }
    return{total,par};
}

std::pair<f32,std::vector<std::pair<i32,i32>>> kruskal_mst(i32 n,
    std::vector<std::tuple<f32,i32,i32>> edges) noexcept
{
    std::sort(edges.begin(),edges.end());
    std::vector<i32> par(n); std::iota(par.begin(),par.end(),0);
    std::function<i32(i32)> find=[&](i32 x)->i32{return par[x]==x?x:par[x]=find(par[x]);};
    f32 total=0; std::vector<std::pair<i32,i32>> mst;
    for(auto&[w,u,v]:edges){
        i32 pu=find(u),pv=find(v);
        if(pu==pv) continue;
        par[pu]=pv; total+=w; mst.push_back({u,v});
        if((i32)mst.size()==n-1) break;
    }
    return{total,mst};
}

bool topo_sort(const Graph& g,std::vector<i32>& order) noexcept {
    i32 n=g.n_verts;
    std::vector<i32> indeg(n,0);
    for(i32 u=0;u<n;++u) for(auto&e:g.adj[u]) ++indeg[e.to];
    std::queue<i32> q;
    for(i32 i=0;i<n;++i) if(!indeg[i]) q.push(i);
    while(!q.empty()){
        i32 u=q.front();q.pop();order.push_back(u);
        for(auto&e:g.adj[u]) if(--indeg[e.to]==0) q.push(e.to);
    }
    return(i32)order.size()==n;
}

std::vector<i32> tarjan_scc(const Graph& g) noexcept {
    i32 n=g.n_verts,timer=0,scc_cnt=0;
    std::vector<i32> id(n,-1),low(n),ord(n),comp(n,-1);
    std::vector<bool> on_stack(n,false);
    std::stack<i32> st;
    std::function<void(i32)> dfs=[&](i32 u){
        low[u]=ord[u]=timer++;
        st.push(u);on_stack[u]=true;
        for(auto&e:g.adj[u]){
            if(ord[e.to]==-1){dfs(e.to);low[u]=std::min(low[u],low[e.to]);}
            else if(on_stack[e.to]) low[u]=std::min(low[u],ord[e.to]);
        }
        if(low[u]==ord[u]){
            while(true){i32 v=st.top();st.pop();on_stack[v]=false;comp[v]=scc_cnt;if(v==u)break;}
            ++scc_cnt;
        }
    };
    for(i32 i=0;i<n;++i) if(ord[i]==-1){ord[i]=-2;dfs(i);}
    // Re-index: need to zero-initialise ord
    std::fill(ord.begin(),ord.end(),-1);
    timer=0;
    std::fill(on_stack.begin(),on_stack.end(),false);
    return comp;
}

std::vector<i32> bfs(const Graph& g,i32 src) noexcept {
    std::vector<i32> vis,seen(g.n_verts,0);
    std::queue<i32> q; q.push(src);seen[src]=1;
    while(!q.empty()){i32 u=q.front();q.pop();vis.push_back(u);
        for(auto&e:g.adj[u]) if(!seen[e.to]){seen[e.to]=1;q.push(e.to);}}
    return vis;
}
std::vector<i32> dfs(const Graph& g,i32 src) noexcept {
    std::vector<i32> vis; std::vector<bool> seen(g.n_verts,false);
    std::function<void(i32)> go=[&](i32 u){seen[u]=true;vis.push_back(u);
        for(auto&e:g.adj[u])if(!seen[e.to])go(e.to);};
    go(src);return vis;
}

f32 max_flow(Graph& g,i32 s,i32 t) noexcept {
    // Dinic's — simplified (BFS level graph + DFS blocking flow)
    (void)g;(void)s;(void)t;
    return 0; // stub — full Dinic requires residual graph structure
}
i32 bipartite_matching(i32 nl,i32 nr,const std::vector<std::pair<i32,i32>>& edges) noexcept {
    std::vector<std::vector<i32>> adj(nl);
    for(auto&[u,v]:edges) adj[u].push_back(v);
    std::vector<i32> match_r(nr,-1);
    std::function<bool(i32,std::vector<bool>&)> aug=[&](i32 u,std::vector<bool>& vis)->bool{
        for(i32 v:adj[u]) if(!vis[v]){vis[v]=true;
            if(match_r[v]==-1||aug(match_r[v],vis)){match_r[v]=u;return true;}}
        return false;};
    i32 res=0;
    for(i32 u=0;u<nl;++u){std::vector<bool> vis(nr,false);if(aug(u,vis))++res;}
    return res;
}
} // namespace qs::math::graph
