#pragma once
// qs_math/graph_embedding.hpp — Force-directed layout, spectral embedding, TSNE graph (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <unordered_map>
#include <numeric>

namespace qs {

// ─── Graph for layout ─────────────────────────────────────────────────────────

struct LayoutGraph {
    u32 n_nodes{};
    std::vector<std::pair<u32,u32>> edges;
    std::vector<f32> edge_weights;

    void add_edge(u32 u, u32 v, f32 w=1.f){
        edges.push_back({u,v}); edge_weights.push_back(w);}

    [[nodiscard]] std::vector<std::vector<std::pair<u32,f32>>> adjacency() const {
        std::vector<std::vector<std::pair<u32,f32>>> adj(n_nodes);
        for(u32 i=0;i<u32(edges.size());++i){
            adj[edges[i].first].push_back({edges[i].second,edge_weights[i]});
            adj[edges[i].second].push_back({edges[i].first,edge_weights[i]});
        }
        return adj;
    }
};

// ─── Force-directed layout (Fruchterman-Reingold) ────────────────────────────

struct FR2DLayout {
    std::vector<Vec2f> pos;
    u32 n_nodes{};
    f32 area{};
    f32 k{};  // optimal distance

    void init(u32 n, f32 width=1000.f, u64 seed=42){
        n_nodes=n; area=width*width; k=std::sqrt(area/f32(n));
        pos.resize(n); Pcg32 rng{seed};
        for(auto& p:pos) p={width*(rng.next_f32()-0.5f),width*(rng.next_f32()-0.5f)};
    }

    void step(const LayoutGraph& g, f32 temp){
        std::vector<Vec2f> disp(n_nodes,{0.f,0.f});
        // Repulsive forces (all pairs)
        for(u32 u=0;u<n_nodes;++u)
        for(u32 v=u+1;v<n_nodes;++v){
            Vec2f d=pos[u]-pos[v];
            f32 dist=length(d);
            if(dist<1e-4f){d={0.001f,0.f};dist=0.001f;}
            f32 rep=k*k/dist;
            Vec2f f=d/dist*rep;
            disp[u]=disp[u]+f; disp[v]=disp[v]-f;
        }
        // Attractive forces (edges)
        for(u32 e=0;e<u32(g.edges.size());++e){
            u32 u=g.edges[e].first, v=g.edges[e].second;
            Vec2f d=pos[u]-pos[v];
            f32 dist=length(d);
            if(dist<1e-4f) continue;
            f32 att=dist*dist/k;
            Vec2f f=d/dist*att;
            disp[u]=disp[u]-f; disp[v]=disp[v]+f;
        }
        // Apply displacements clamped to temperature
        for(u32 u=0;u<n_nodes;++u){
            f32 dl=length(disp[u]);
            if(dl<1e-6f) continue;
            pos[u]=pos[u]+disp[u]/dl*std::min(dl,temp);
        }
    }

    void run(const LayoutGraph& g, u32 iters=100){
        f32 temp=std::sqrt(area)*0.1f;
        for(u32 i=0;i<iters;++i){
            step(g,temp);
            temp*=0.95f;
        }
    }
};

// ─── Kamada-Kawai layout ──────────────────────────────────────────────────────

struct KKLayout {
    std::vector<Vec2f> pos;
    u32 n{};

    // All-pairs shortest path as ideal distances
    [[nodiscard]] std::vector<f32> compute_distances(const LayoutGraph& g) const {
        u32 N=g.n_nodes;
        std::vector<f32> D(N*N,1e30f);
        for(u32 i=0;i<N;++i) D[i*N+i]=0.f;
        for(u32 e=0;e<u32(g.edges.size());++e){
            u32 u=g.edges[e].first,v=g.edges[e].second;
            f32 w=1.f/g.edge_weights[e];
            D[u*N+v]=D[v*N+u]=w;
        }
        for(u32 k=0;k<N;++k) for(u32 i=0;i<N;++i) for(u32 j=0;j<N;++j)
            D[i*N+j]=std::min(D[i*N+j],D[i*N+k]+D[k*N+j]);
        return D;
    }

    void init(const LayoutGraph& g, u64 seed=42){
        n=g.n_nodes; pos.resize(n);
        Pcg32 rng{seed};
        for(auto& p:pos) p={rng.next_f32()*2.f-1.f,rng.next_f32()*2.f-1.f};
    }

    // One KK step (move node m to minimize energy)
    void step(u32 m, const std::vector<f32>& D, f32 L=1.f, f32 K=1.f){
        u32 N=n;
        f32 dx=0.f,dy=0.f,dxx=0.f,dyy=0.f,dxy=0.f;
        for(u32 i=0;i<N;++i){
            if(i==m) continue;
            f32 dij=D[m*N+i]; if(dij>1e9f) continue;
            f32 lij=L*dij, kij=K/(dij*dij);
            Vec2f d=pos[m]-pos[i]; f32 dist=length(d);
            if(dist<1e-6f) continue;
            dx +=kij*(d.x-lij*d.x/dist);
            dy +=kij*(d.y-lij*d.y/dist);
            dxx+=kij*(1.f-lij*d.y*d.y/(dist*dist*dist));
            dyy+=kij*(1.f-lij*d.x*d.x/(dist*dist*dist));
            dxy+=kij*(lij*d.x*d.y/(dist*dist*dist));
        }
        f32 det=dxx*dyy-dxy*dxy;
        if(std::abs(det)<1e-10f) return;
        f32 ddx=(dx*dyy-dy*dxy)/det;
        f32 ddy=(dy*dxx-dx*dxy)/det;
        pos[m].x-=ddx; pos[m].y-=ddy;
    }
};

// ─── Spectral graph embedding ─────────────────────────────────────────────────

// 2D embedding via Fiedler vector and second Laplacian eigenvector
[[nodiscard]] inline std::vector<Vec2f> spectral_embed_2d(
    const LayoutGraph& g, u32 power_iters=50)
{
    u32 N=g.n_nodes;
    auto adj=g.adjacency();
    // Normalized Laplacian: L = D^{-1/2} (D-A) D^{-1/2}
    std::vector<f32> deg(N,0.f);
    for(u32 u=0;u<N;++u) for(auto[v,w]:adj[u]) deg[u]+=w;

    auto matvec=[&](const std::vector<f32>& x)->std::vector<f32>{
        std::vector<f32> Lx(N,0.f);
        for(u32 u=0;u<N;++u){
            Lx[u]+=x[u]; // D part (normalized = 1 on diagonal)
            for(auto[v,w]:adj[u]){
                f32 norm=std::sqrt(deg[u]*deg[v]);
                if(norm>1e-8f) Lx[u]-=w/norm*x[v];
            }
        }
        return Lx;
    };

    // Power iteration for 2 eigenvectors (deflated)
    Pcg32 rng{42};
    std::vector<std::vector<f32>> evecs(2,std::vector<f32>(N));
    for(auto& ev:evecs) for(auto& v:ev) v=rng.next_f32()*2.f-1.f;

    for(u32 ev_idx=0;ev_idx<2;++ev_idx){
        auto& v=evecs[ev_idx];
        // Deflate previous eigenvectors
        for(u32 prev=0;prev<ev_idx;++prev){
            f64 dot=0.; for(u32 i=0;i<N;++i) dot+=v[i]*evecs[prev][i];
            for(u32 i=0;i<N;++i) v[i]-=f32(dot)*evecs[prev][i];
        }
        for(u32 iter=0;iter<power_iters;++iter){
            // Shift: use (2I - L) to find smallest eigenvalues
            auto Lv=matvec(v);
            std::vector<f32> vn(N);
            for(u32 i=0;i<N;++i) vn[i]=2.f*v[i]-Lv[i];
            // Deflate
            for(u32 prev=0;prev<ev_idx;++prev){
                f64 dot=0.; for(u32 i=0;i<N;++i) dot+=vn[i]*evecs[prev][i];
                for(u32 i=0;i<N;++i) vn[i]-=f32(dot)*evecs[prev][i];
            }
            f64 norm=0.; for(auto x:vn) norm+=x*x; norm=std::sqrt(norm+1e-15);
            for(auto& x:vn) x/=f32(norm);
            v=vn;
        }
    }

    std::vector<Vec2f> pos(N);
    for(u32 i=0;i<N;++i) pos[i]={evecs[0][i],evecs[1][i]};
    return pos;
}

// ─── 3D graph layout ──────────────────────────────────────────────────────────

struct FR3DLayout {
    std::vector<Vec3f> pos;
    u32 n_nodes{};
    f32 k{};

    void init(u32 n, f32 side=10.f, u64 seed=0){
        n_nodes=n; k=side/std::pow(f32(n),1.f/3.f);
        pos.resize(n); Pcg32 rng{seed};
        for(auto& p:pos) p={side*(rng.next_f32()-0.5f),side*(rng.next_f32()-0.5f),side*(rng.next_f32()-0.5f)};
    }

    void step(const LayoutGraph& g, f32 temp){
        std::vector<Vec3f> disp(n_nodes,{});
        for(u32 u=0;u<n_nodes;++u)
        for(u32 v=u+1;v<n_nodes;++v){
            Vec3f d=pos[u]-pos[v]; f32 dist=length(d);
            if(dist<1e-4f){d={0.001f,0,0};dist=0.001f;}
            f32 rep=k*k/dist;
            Vec3f f=d/dist*rep;
            disp[u]=disp[u]+f; disp[v]=disp[v]-f;
        }
        for(u32 e=0;e<u32(g.edges.size());++e){
            u32 u=g.edges[e].first,v=g.edges[e].second;
            Vec3f d=pos[u]-pos[v]; f32 dist=length(d);
            if(dist<1e-4f) continue;
            f32 att=dist*dist/k;
            Vec3f f=d/dist*att;
            disp[u]=disp[u]-f; disp[v]=disp[v]+f;
        }
        for(u32 u=0;u<n_nodes;++u){
            f32 dl=length(disp[u]);
            if(dl<1e-6f) continue;
            pos[u]=pos[u]+disp[u]/dl*std::min(dl,temp);
        }
    }

    void run(const LayoutGraph& g,u32 iters=100){
        f32 temp=k*2.f;
        for(u32 i=0;i<iters;++i){step(g,temp);temp*=0.95f;}
    }
};

// ─── Edge crossing minimization metric ───────────────────────────────────────

[[nodiscard]] inline u32 count_edge_crossings_2d(
    std::span<const Vec2f> pos,
    std::span<const std::pair<u32,u32>> edges)
{
    u32 crossings=0;
    auto seg_intersect=[](Vec2f a,Vec2f b,Vec2f c,Vec2f d)->bool{
        auto cross2d=[](Vec2f p,Vec2f q,Vec2f r)->f32{
            return (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x);};
        f32 d1=cross2d(c,d,a),d2=cross2d(c,d,b);
        f32 d3=cross2d(a,b,c),d4=cross2d(a,b,d);
        if(((d1>0&&d2<0)||(d1<0&&d2>0))&&((d3>0&&d4<0)||(d3<0&&d4>0))) return true;
        return false;};
    for(u32 i=0;i<u32(edges.size());++i)
    for(u32 j=i+1;j<u32(edges.size());++j){
        if(edges[i].first==edges[j].first||edges[i].first==edges[j].second||
           edges[i].second==edges[j].first||edges[i].second==edges[j].second) continue;
        if(seg_intersect(pos[edges[i].first],pos[edges[i].second],
                          pos[edges[j].first],pos[edges[j].second])) ++crossings;
    }
    return crossings;
}

} // namespace qs
