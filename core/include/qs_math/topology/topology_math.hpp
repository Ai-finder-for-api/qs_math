#pragma once
// qs_math/topology_math.hpp — Euler characteristic, genus, homology, simplicial complex (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <span>
#include <array>
#include <numeric>

namespace qs {

// ─── Simplicial complex ───────────────────────────────────────────────────────

struct SimplicialComplex {
    std::vector<u32>              vertices;   // 0-simplices
    std::vector<std::array<u32,2>> edges;     // 1-simplices
    std::vector<std::array<u32,3>> faces;     // 2-simplices
    std::vector<std::array<u32,4>> tetra;     // 3-simplices

    void add_vertex(u32 v) { vertices.push_back(v); }
    void add_edge(u32 a,u32 b){ if(a>b)std::swap(a,b); edges.push_back({a,b}); }
    void add_face(u32 a,u32 b,u32 c){
        std::array<u32,3> t={a,b,c};std::sort(t.begin(),t.end());faces.push_back(t);
        add_edge(t[0],t[1]);add_edge(t[1],t[2]);add_edge(t[0],t[2]);}
    void add_tetra(u32 a,u32 b,u32 c,u32 d){
        std::array<u32,4> t={a,b,c,d};std::sort(t.begin(),t.end());tetra.push_back(t);
        add_face(t[0],t[1],t[2]);add_face(t[0],t[1],t[3]);add_face(t[0],t[2],t[3]);add_face(t[1],t[2],t[3]);}

    // Deduplicate
    void normalize(){
        std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
        std::sort(faces.begin(),faces.end());faces.erase(std::unique(faces.begin(),faces.end()),faces.end());
        std::sort(tetra.begin(),tetra.end());tetra.erase(std::unique(tetra.begin(),tetra.end()),tetra.end());}

    // Euler characteristic: V - E + F - T
    [[nodiscard]] i32 euler_characteristic() const {
        return i32(vertices.size())-i32(edges.size())+i32(faces.size())-i32(tetra.size());}

    // Genus from Euler characteristic (for closed orientable surface): g = (2 - chi) / 2
    [[nodiscard]] i32 genus_from_euler() const {
        return (2-euler_characteristic())/2;}
};

// ─── Mesh topology ────────────────────────────────────────────────────────────

struct MeshTopology {
    u32 n_vertices{}, n_edges{}, n_faces{};
    u32 n_boundary_edges{};
    u32 n_connected_components{};

    [[nodiscard]] i32 euler_characteristic() const {
        return i32(n_vertices)-i32(n_edges)+i32(n_faces);}

    [[nodiscard]] i32 genus() const {
        // For closed orientable surface: chi = 2 - 2g - b  (b=boundaries)
        return (2-euler_characteristic()-i32(n_boundary_edges>0?1:0))/2;}

    [[nodiscard]] bool is_manifold_closed() const { return n_boundary_edges==0; }
    [[nodiscard]] bool is_sphere_topology() const { return euler_characteristic()==2&&n_boundary_edges==0; }
    [[nodiscard]] bool is_disk_topology() const { return euler_characteristic()==1&&n_boundary_edges>0; }
};

// Compute mesh topology from triangle soup
[[nodiscard]] inline MeshTopology compute_mesh_topology(
    u32 n_verts, std::span<const std::array<u32,3>> tris)
{
    MeshTopology topo;
    topo.n_vertices=n_verts;
    topo.n_faces=u32(tris.size());
    std::unordered_map<u64,u32> edge_count;
    auto edge_key=[](u32 a,u32 b)->u64{if(a>b)std::swap(a,b);return u64(a)<<32|b;};
    for(const auto& t:tris) for(u32 e=0;e<3;++e){
        u32 a=t[e],b=t[(e+1)%3];
        ++edge_count[edge_key(a,b)];}
    topo.n_edges=u32(edge_count.size());
    for(auto[k,c]:edge_count) if(c==1) ++topo.n_boundary_edges;
    // Connected components via union-find
    std::vector<u32> parent(n_verts);
    std::iota(parent.begin(),parent.end(),0);
    std::function<u32(u32)> find=[&](u32 x)->u32{return parent[x]==x?x:parent[x]=find(parent[x]);};
    for(const auto& t:tris){
        u32 r=find(t[0]);parent[find(t[1])]=r;parent[find(t[2])]=r;}
    std::unordered_set<u32> roots;
    for(u32 i=0;i<n_verts;++i) roots.insert(find(i));
    topo.n_connected_components=u32(roots.size());
    return topo;
}

// ─── Persistent homology (Vietoris-Rips, 0th order = connected components) ───

struct PersistencePair { f32 birth{}, death{std::numeric_limits<f32>::max()}; u32 dim{}; };

// 0-dimensional persistent homology via union-find on edges sorted by weight
[[nodiscard]] inline std::vector<PersistencePair> persistence_0d(
    u32 n_vertices, std::span<const std::tuple<f32,u32,u32>> weighted_edges)
{
    // Sort edges by weight
    std::vector<std::tuple<f32,u32,u32>> sorted_edges(weighted_edges.begin(),weighted_edges.end());
    std::sort(sorted_edges.begin(),sorted_edges.end());

    std::vector<u32> parent(n_vertices);
    std::vector<f32> birth(n_vertices,0.f);
    std::iota(parent.begin(),parent.end(),0);
    std::function<u32(u32)> find=[&](u32 x)->u32{return parent[x]==x?x:parent[x]=find(parent[x]);};

    std::vector<PersistencePair> pairs;
    for(auto[w,a,b]:sorted_edges){
        u32 ra=find(a),rb=find(b);
        if(ra!=rb){
            // Merge: kill younger component
            f32 ba=birth[ra],bb=birth[rb];
            if(ba<bb){parent[rb]=ra;pairs.push_back({bb,w,0});}
            else{parent[ra]=rb;pairs.push_back({ba,w,0});}
        }
    }
    // Surviving component
    pairs.push_back({0.f,std::numeric_limits<f32>::max(),0});
    return pairs;
}

// Betti numbers from persistence diagram
[[nodiscard]] inline std::array<u32,3> betti_numbers(const SimplicialComplex& sc) {
    // b0 = connected components, b1 = loops, b2 = voids
    // For simplicial complex: use Euler characteristic and face counts
    u32 V=u32(sc.vertices.size()), E=u32(sc.edges.size()), F=u32(sc.faces.size()), T=u32(sc.tetra.size());
    i32 chi=i32(V)-i32(E)+i32(F)-i32(T);
    // Simplified: b0 - b1 + b2 = chi, with estimates
    u32 b0=1; // assume connected
    u32 b2=T>0?1:0;
    u32 b1=u32(std::max(0,i32(b0)+i32(b2)-chi));
    return {b0,b1,b2};
}

// ─── Winding number ───────────────────────────────────────────────────────────

[[nodiscard]] inline f32 winding_number_2d(
    const std::vector<Vec2f>& poly, const Vec2f& p)
{
    f32 angle=0.f;
    u32 N=u32(poly.size());
    for(u32 i=0;i<N;++i){
        Vec2f a=poly[i]-p, b=poly[(i+1)%N]-p;
        angle+=std::atan2(a.x*b.y-a.y*b.x, a.x*b.x+a.y*b.y);
    }
    return angle/(2.f*k_pi);
}

[[nodiscard]] inline i32 winding_number_int(const std::vector<Vec2f>& poly, const Vec2f& p) {
    return i32(std::round(winding_number_2d(poly,p)));
}

// ─── Topological complexity of motion ────────────────────────────────────────

// Rotation number of a 2D curve (total curvature / 2pi)
[[nodiscard]] inline f32 rotation_number(std::span<const Vec2f> curve) {
    f32 total=0.f;
    u32 N=u32(curve.size());
    for(u32 i=0;i<N;++i){
        Vec2f a=curve[(i+N-1)%N],b=curve[i],c=curve[(i+1)%N];
        Vec2f t1=normalize(b-a),t2=normalize(c-b);
        f32 cross=t1.x*t2.y-t1.y*t2.x;
        total+=std::atan2(cross,dot(t1,t2));
    }
    return total/(2.f*k_pi);
}

// Self-intersection count of 2D polyline
[[nodiscard]] inline u32 count_self_intersections(std::span<const Vec2f> poly) {
    u32 count=0, N=u32(poly.size());
    for(u32 i=0;i<N;++i)
    for(u32 j=i+2;j<N;++j){
        if(i==0&&j==N-1) continue;
        Vec2f a=poly[i],b=poly[(i+1)%N],c=poly[j],d=poly[(j+1)%N];
        Vec2f r=b-a,s=d-c;
        f32 denom=r.x*s.y-r.y*s.x;
        if(std::abs(denom)<1e-8f) continue;
        Vec2f ca=c-a;
        f32 t=(ca.x*s.y-ca.y*s.x)/denom;
        f32 u2=(ca.x*r.y-ca.y*r.x)/denom;
        if(t>=0.f&&t<=1.f&&u2>=0.f&&u2<=1.f) ++count;
    }
    return count;
}

} // namespace qs
