#pragma once
// mesh_math_ext.hpp — Extended mesh math: LOD generation, parameterization,
//                      Laplacian smoothing, mesh metrics, UV atlas helpers,
//                      geodesic distance, mesh simplification helpers
#include "mesh_math.hpp"
#include "geometry3d.hpp"
#include "hashing.hpp"

namespace qs::math::mesh {

// ---------------------------------------------------------------------------
// Mesh topology helpers
// ---------------------------------------------------------------------------
// Build half-edge adjacency (returns true if manifold)
inline bool build_adjacency(
    const u32* indices, i32 tri_count, i32 vert_count,
    u32* adj_faces, // adj_faces[tri*3+e] = adjacent face index
    u32* adj_edges) noexcept // adj_edges[tri*3+e] = edge index in adjacent face
{
    // Simple O(n^2) for correctness; production would use hash map
    for(i32 i=0;i<tri_count*3;++i) adj_faces[i]=u32(-1), adj_edges[i]=0;
    for(i32 t1=0;t1<tri_count;++t1) for(i32 e1=0;e1<3;++e1){
        u32 a=indices[t1*3+e1], b=indices[t1*3+(e1+1)%3];
        for(i32 t2=t1+1;t2<tri_count;++t2) for(i32 e2=0;e2<3;++e2){
            u32 c=indices[t2*3+e2], d=indices[t2*3+(e2+1)%3];
            if((a==d&&b==c)){
                adj_faces[t1*3+e1]=t2; adj_edges[t1*3+e1]=e2;
                adj_faces[t2*3+e2]=t1; adj_edges[t2*3+e2]=e1;
            }
        }
    }
    // Check manifold: every edge shared by exactly 2 faces
    for(i32 i=0;i<tri_count*3;++i) if(adj_faces[i]==u32(-1)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Laplacian mesh smoothing
// ---------------------------------------------------------------------------
inline void laplacian_smooth(
    Vec3f* positions, i32 vert_count,
    const u32* indices, i32 tri_count,
    i32 iterations, f32 lambda=0.5f) noexcept
{
    // Build vertex-vertex adjacency
    constexpr i32 MAX_ADJ=32;
    u32 adj[4096*MAX_ADJ]; i32 adj_count[4096]={};
    i32 max_v=min(vert_count,4096);
    for(i32 t=0;t<tri_count;++t) for(i32 e=0;e<3;++e){
        u32 a=indices[t*3+e], b=indices[t*3+(e+1)%3];
        if(int(a)<max_v&&adj_count[a]<MAX_ADJ) adj[a*MAX_ADJ+adj_count[a]++]=b;
        if(int(b)<max_v&&adj_count[b]<MAX_ADJ) adj[b*MAX_ADJ+adj_count[b]++]=a;
    }
    Vec3f tmp[4096];
    for(i32 iter=0;iter<iterations;++iter){
        for(i32 v=0;v<max_v;++v){
            if(adj_count[v]==0){tmp[v]=positions[v];continue;}
            Vec3f centroid{0.f};
            for(i32 j=0;j<adj_count[v];++j) centroid=centroid+positions[adj[v*MAX_ADJ+j]];
            centroid=centroid*(1.f/adj_count[v]);
            tmp[v]=positions[v]+lambda*(centroid-positions[v]);
        }
        for(i32 v=0;v<max_v;++v) positions[v]=tmp[v];
    }
}

// ---------------------------------------------------------------------------
// Mesh metrics
// ---------------------------------------------------------------------------
// Aspect ratio of a triangle (close to 1 = equilateral = good)
[[nodiscard]] constexpr f32 triangle_aspect_ratio(Vec3f a, Vec3f b, Vec3f c) noexcept {
    f32 ab=(b-a).length(), bc=(c-b).length(), ca=(a-c).length();
    f32 s=(ab+bc+ca)*0.5f;
    if(s<EPS_F32) return F32_INF;
    f32 area=triangle_area(a,b,c);
    return (ab*bc*ca)/(8.f*area*(s-ab)*(s-bc)*(s-ca)+EPS_F32);
}

// Skewness (0=perfect, 1=degenerate)
[[nodiscard]] inline f32 triangle_skewness(Vec3f a, Vec3f b, Vec3f c) noexcept {
    f32 ab=(b-a).length(), bc=(c-b).length(), ca=(a-c).length();
    f32 s=(ab+bc+ca)/3.f;
    f32 max_e=max(max(ab,bc),ca);
    f32 min_e=min(min(ab,bc),ca);
    return s>EPS_F32?(max_e-min_e)/(max_e+min_e):0.f;
}

// Mesh quality histogram (counts by aspect ratio ranges)
inline void mesh_quality_stats(
    const Vec3f* positions, const u32* indices, i32 tri_count,
    i32& good, i32& ok, i32& bad, i32& degenerate) noexcept
{
    good=ok=bad=degenerate=0;
    for(i32 t=0;t<tri_count;++t){
        Vec3f a=positions[indices[t*3]],b=positions[indices[t*3+1]],c=positions[indices[t*3+2]];
        f32 ar=triangle_aspect_ratio(a,b,c);
        if(ar>100.f) ++degenerate;
        else if(ar>10.f) ++bad;
        else if(ar>3.f)  ++ok;
        else             ++good;
    }
}

// ---------------------------------------------------------------------------
// UV Atlas helpers
// ---------------------------------------------------------------------------
// Angle-based parameterization seed (LSCM requires a solver; this is ABF seed)
[[nodiscard]] inline Vec2f project_to_uv_plane(
    Vec3f v, Vec3f normal, Vec3f tangent, Vec3f bitangent) noexcept
{
    return {dot(v,tangent), dot(v,bitangent)};
}

// Pack UVs into atlas (greedy shelf packing, simplified)
struct UVIsland {
    Vec2f min_uv, max_uv;
    Vec2f atlas_offset;
    f32   scale{1.f};
};

inline f32 pack_uv_islands(UVIsland* islands, i32 n,
                            f32 atlas_w, f32 atlas_h, f32 padding=0.01f) noexcept
{
    // Simple row-by-row packing
    f32 x=padding, y=padding, row_h=0.f;
    f32 total_area=0.f;
    for(i32 i=0;i<n;++i){
        Vec2f size=islands[i].max_uv-islands[i].min_uv;
        if(x+size.x+padding>atlas_w){ x=padding; y+=row_h+padding; row_h=0.f; }
        if(y+size.y+padding>atlas_h) break; // doesn't fit
        islands[i].atlas_offset={x-islands[i].min_uv.x, y-islands[i].min_uv.y};
        x+=size.x+padding;
        row_h=max(row_h,size.y);
        total_area+=size.x*size.y;
    }
    return total_area/(atlas_w*atlas_h); // fill rate
}

// ---------------------------------------------------------------------------
// Geodesic distance (approximate, Dijkstra on mesh graph)
// ---------------------------------------------------------------------------
inline void geodesic_distance_approx(
    const Vec3f* positions, const u32* indices, i32 tri_count,
    i32 source_vert, i32 vert_count, f32* distances) noexcept
{
    constexpr i32 MAX_ADJ=32;
    constexpr i32 MAX_V=4096;
    i32 max_v=min(vert_count,MAX_V);
    u32 adj[MAX_V*MAX_ADJ]; i32 adj_count[MAX_V]={};
    for(i32 i=0;i<max_v;++i) distances[i]=F32_INF;
    distances[source_vert]=0.f;

    // Build adjacency
    for(i32 t=0;t<tri_count;++t) for(i32 e=0;e<3;++e){
        u32 a=indices[t*3+e], b=indices[t*3+(e+1)%3];
        if(int(a)<max_v&&adj_count[a]<MAX_ADJ) adj[a*MAX_ADJ+adj_count[a]++]=b;
        if(int(b)<max_v&&adj_count[b]<MAX_ADJ) adj[b*MAX_ADJ+adj_count[b]++]=a;
    }

    // Simple Dijkstra (no priority queue — O(V^2), fine for small meshes)
    bool visited[MAX_V]={};
    for(i32 iter=0;iter<max_v;++iter){
        // Find minimum
        i32 u=-1; f32 min_d=F32_INF;
        for(i32 v=0;v<max_v;++v) if(!visited[v]&&distances[v]<min_d){min_d=distances[v];u=v;}
        if(u<0) break;
        visited[u]=true;
        // Relax
        for(i32 j=0;j<adj_count[u];++j){
            i32 v=i32(adj[u*MAX_ADJ+j]);
            f32 d=distances[u]+(positions[u]-positions[v]).length();
            if(d<distances[v]) distances[v]=d;
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh decimation quality metric (quadric error)
// ---------------------------------------------------------------------------
struct QuadricMatrix {
    f64 a[10]{}; // symmetric 4x4, stored as upper triangle
    void add_plane(Vec3f n, Vec3f p) noexcept {
        f64 nx=n.x,ny=n.y,nz=n.z, d=-dot(n,p);
        a[0]+=nx*nx; a[1]+=nx*ny; a[2]+=nx*nz; a[3]+=nx*d;
        a[4]+=ny*ny; a[5]+=ny*nz; a[6]+=ny*d;
        a[7]+=nz*nz; a[8]+=nz*d; a[9]+=d*d;
    }
    f64 eval(Vec3f v) const noexcept {
        f64 x=v.x,y=v.y,z=v.z;
        return x*x*a[0]+2*x*y*a[1]+2*x*z*a[2]+2*x*a[3]
                       +y*y*a[4]+2*y*z*a[5]+2*y*a[6]
                                 +z*z*a[7]+2*z*a[8]+a[9];
    }
    void add(const QuadricMatrix& o) noexcept { for(i32 i=0;i<10;++i) a[i]+=o.a[i]; }
};

} // namespace qs::math::mesh
