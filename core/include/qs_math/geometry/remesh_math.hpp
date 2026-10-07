#pragma once
// qs_math/remesh_math.hpp — Isotropic remeshing, Delaunay refinement, edge collapse criteria (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "geometry3d.hpp"
#include "mesh_math.hpp"
#include "delaunay_math.hpp"
#include "geodesic_math.hpp"
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <cassert>

namespace qs {

// ─── Edge data structures ─────────────────────────────────────────────────────
struct HalfEdge {
    u32 vert{~0u};   // destination vertex
    u32 face{~0u};   // incident face
    u32 next{~0u};   // next half-edge in face
    u32 twin{~0u};   // opposite half-edge
    u32 prev{~0u};   // previous half-edge in face
};

struct HEMesh {
    std::vector<Vec3f>    verts;
    std::vector<HalfEdge> halfedges;
    std::vector<u32>      vert_he;  // one outgoing half-edge per vertex
    std::vector<u32>      face_he;  // one half-edge per face

    [[nodiscard]] u32 twin(u32 he) const { return halfedges[he].twin; }
    [[nodiscard]] u32 next(u32 he) const { return halfedges[he].next; }
    [[nodiscard]] u32 prev(u32 he) const { return halfedges[he].prev; }
    [[nodiscard]] u32 vert(u32 he) const { return halfedges[he].vert; }
    [[nodiscard]] u32 face(u32 he) const { return halfedges[he].face; }
    [[nodiscard]] u32 origin(u32 he) const { return vert(prev(he)); }

    [[nodiscard]] Vec3f edge_midpoint(u32 he) const {
        return (verts[origin(he)]+verts[vert(he)])*0.5f;
    }
    [[nodiscard]] f32 edge_length(u32 he) const {
        return dist(verts[origin(he)],verts[vert(he)]);
    }
    [[nodiscard]] bool is_boundary_edge(u32 he) const { return twin(he)==~0u; }
    [[nodiscard]] Vec3f face_normal(u32 fi) const {
        u32 he=face_he[fi];
        Vec3f a=verts[origin(he)],b=verts[vert(he)],c=verts[vert(next(he))];
        return normalize(cross(b-a,c-a));
    }

    // Build from triangle soup
    static HEMesh from_triangles(const std::vector<Vec3f>& vs,
        const std::vector<std::array<u32,3>>& tris)
    {
        HEMesh m; m.verts=vs;
        u32 nf=u32(tris.size());
        m.halfedges.resize(nf*3);
        m.face_he.resize(nf);
        m.vert_he.assign(u32(vs.size()),~0u);
        std::unordered_map<u64,u32> edge_map;
        auto ek=[](u32 a,u32 b)->u64{return (u64(a)<<32)|b;};
        for(u32 fi=0;fi<nf;++fi) {
            m.face_he[fi]=fi*3;
            for(u32 e=0;e<3;++e) {
                u32 he=fi*3+e;
                m.halfedges[he].vert=tris[fi][(e+1)%3];
                m.halfedges[he].face=fi;
                m.halfedges[he].next=fi*3+(e+1)%3;
                m.halfedges[he].prev=fi*3+(e+2)%3;
                m.halfedges[he].twin=~0u;
                u32 origin=tris[fi][e];
                m.vert_he[origin]=he;
                edge_map[ek(origin,tris[fi][(e+1)%3])]=he;
            }
        }
        // Link twins
        for(u32 fi=0;fi<nf;++fi) for(u32 e=0;e<3;++e) {
            u32 he=fi*3+e;
            u32 o=m.origin(he),d=m.vert(he);
            auto it=edge_map.find(ek(d,o));
            if(it!=edge_map.end()){ m.halfedges[he].twin=it->second; m.halfedges[it->second].twin=he; }
        }
        return m;
    }

    [[nodiscard]] std::vector<std::array<u32,3>> to_triangles() const {
        std::vector<std::array<u32,3>> tris(face_he.size());
        for(u32 fi=0;fi<u32(face_he.size());++fi) {
            u32 he=face_he[fi];
            tris[fi]={origin(he),vert(he),vert(next(he))};
        }
        return tris;
    }
};

// ─── Isotropic remeshing (Botsch & Kobbelt 2004) ─────────────────────────────
// Target edge length h; performs: split long, collapse short, flip non-Delaunay,
// tangential smoothing, back-project to surface

struct RemeshConfig {
    f32 target_edge_length{0};  // 0 = auto (mean edge length)
    u32 iterations{5};
    f32 split_threshold{4.f/3}; // split edges > threshold * h
    f32 collapse_threshold{4.f/5}; // collapse edges < threshold * h
    bool project_to_surface{true};
    bool smooth{true};
    u32  smooth_iters{1};
};

// Compute mean edge length
[[nodiscard]] inline f32 mean_edge_length(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris)
{
    f32 total=0; u32 cnt=0;
    for(const auto& t:tris) for(u32 e=0;e<3;++e){
        total+=dist(verts[t[e]],verts[t[(e+1)%3]]); ++cnt;
    }
    return cnt>0?total/cnt:1.f;
}

// Edge collapse quality metric (quadric error — simplified)
[[nodiscard]] inline f32 edge_collapse_error(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris,
    u32 v0, u32 v1)
{
    // Volume change heuristic: sum of triangle normals weighted by area change
    Vec3f mid=(verts[v0]+verts[v1])*0.5f;
    f32 err=0;
    for(const auto& t:tris) {
        bool has_v0=false,has_v1=false;
        for(u32 v:t){ if(v==v0)has_v0=true; if(v==v1)has_v1=true; }
        if(!has_v0&&!has_v1) continue;
        Vec3f a=verts[t[0]],b=verts[t[1]],c=verts[t[2]];
        Vec3f n=cross(b-a,c-a);
        f32 area=length(n)*0.5f;
        // Error = area * distance of midpoint from triangle plane
        if(area>1e-10f) err+=area*std::abs(dot(normalize(n),mid-a));
    }
    return err;
}

// Delaunay edge flip criterion: flip if opposite angle > 90°
[[nodiscard]] inline bool should_flip_edge(
    const std::vector<Vec3f>& verts, u32 a,u32 b,u32 c,u32 d)
{
    // Edge shared: a-b, opposite vertices c and d
    // Flip if d is inside circumcircle of (a,b,c)
    // In 3D: check if dihedral improves
    Vec3f va=verts[a],vb=verts[b],vc=verts[c],vd=verts[d];
    f32 cos_c=dot(normalize(va-vc),normalize(vb-vc));
    f32 cos_d=dot(normalize(va-vd),normalize(vb-vd));
    // Flip if sum of opposite angles > π (Delaunay condition violated)
    return std::acos(std::clamp(cos_c,-1.f,1.f))+std::acos(std::clamp(cos_d,-1.f,1.f))>f32(std::numbers::pi)+1e-5f;
}

// Tangential Laplacian smoothing (moves vertices along tangent plane)
[[nodiscard]] inline std::vector<Vec3f> tangential_smooth(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris,
    f32 step=0.5f, u32 iters=1)
{
    u32 nv=u32(verts.size());
    std::vector<std::vector<u32>> adj(nv);
    for(const auto& t:tris) for(u32 e=0;e<3;++e){
        adj[t[e]].push_back(t[(e+1)%3]);
        adj[t[(e+1)%3]].push_back(t[e]);
    }
    for(auto& a:adj){ std::sort(a.begin(),a.end()); a.erase(std::unique(a.begin(),a.end()),a.end()); }
    std::vector<Vec3f> v=verts;
    for(u32 it=0;it<iters;++it) {
        std::vector<Vec3f> nv2=v;
        for(u32 i=0;i<nv;++i) {
            if(adj[i].empty()) continue;
            // Compute centroid of one-ring
            Vec3f c{}; for(u32 nb:adj[i]) c=c+v[nb]; c=c*(1.f/adj[i].size());
            // Compute vertex normal (area-weighted average of incident triangle normals)
            Vec3f n{};
            for(u32 nb:adj[i]) {
                // Find triangles containing edge (i,nb)
                for(const auto& t:tris) {
                    bool hi=false,hnb=false;
                    for(u32 vt:t){ if(vt==i)hi=true; if(vt==nb)hnb=true; }
                    if(hi&&hnb){
                        Vec3f e1=v[t[1]]-v[t[0]],e2=v[t[2]]-v[t[0]];
                        n=n+cross(e1,e2);
                    }
                }
            }
            f32 nlen=length(n);
            if(nlen<1e-7f) continue;
            n=n/nlen;
            // Project centroid displacement onto tangent plane
            Vec3f delta=c-v[i];
            delta=delta-n*dot(delta,n);
            nv2[i]=v[i]+delta*step;
        }
        v=nv2;
    }
    return v;
}

// ─── Progressive mesh (edge collapse sequence) ────────────────────────────────
struct CollapseRecord {
    u32 v_removed, v_kept;
    Vec3f v_removed_pos, v_kept_pos_new;
    std::vector<u32> removed_faces;
};

[[nodiscard]] inline std::vector<CollapseRecord> build_collapse_sequence(
    std::vector<Vec3f>& verts,
    std::vector<std::array<u32,3>>& tris,
    u32 target_faces, f32 max_error=0.01f)
{
    std::vector<CollapseRecord> log;
    while(u32(tris.size())>target_faces) {
        // Find best edge to collapse (lowest error)
        f32 best_err=1e30f; i32 best_e=-1;
        std::pair<u32,u32> best_edge;
        for(u32 fi=0;fi<u32(tris.size());++fi) {
            for(u32 e=0;e<3;++e) {
                u32 va=tris[fi][e],vb=tris[fi][(e+1)%3];
                f32 err=edge_collapse_error(verts,tris,va,vb);
                if(err<best_err){ best_err=err; best_e=i32(fi*3+e); best_edge={va,vb}; }
            }
        }
        if(best_e<0||best_err>max_error) break;
        auto [v0,v1]=best_edge;
        // Collapse v1 into v0
        Vec3f mid=(verts[v0]+verts[v1])*0.5f;
        CollapseRecord rec; rec.v_removed=v1; rec.v_kept=v0;
        rec.v_removed_pos=verts[v1]; rec.v_kept_pos_new=mid;
        verts[v0]=mid;
        // Remove degenerate faces and remap v1→v0
        std::vector<std::array<u32,3>> new_tris;
        for(auto& t:tris) {
            bool has_v1=false; for(auto& v:t) if(v==v1){v=v0;has_v1=true;}
            if(t[0]!=t[1]&&t[1]!=t[2]&&t[0]!=t[2]) new_tris.push_back(t);
            else if(has_v1) rec.removed_faces.push_back(u32(&t-tris.data()));
        }
        tris=new_tris;
        log.push_back(rec);
    }
    return log;
}

// ─── Quadric error metrics (Garland-Heckbert) ─────────────────────────────────
struct Quadric {
    f32 A[4][4]{}; // symmetric 4×4 matrix

    Quadric() { std::fill(&A[0][0],&A[0][0]+16,0.f); }

    // Plane quadric: Q = n^T n where n=(a,b,c,d) is the plane equation
    static Quadric from_plane(Vec3f normal, f32 d) {
        Quadric q;
        f32 n[4]={normal.x,normal.y,normal.z,d};
        for(u32 i=0;i<4;++i) for(u32 j=0;j<4;++j) q.A[i][j]=n[i]*n[j];
        return q;
    }

    [[nodiscard]] Quadric operator+(const Quadric& o) const {
        Quadric r;
        for(u32 i=0;i<4;++i) for(u32 j=0;j<4;++j) r.A[i][j]=A[i][j]+o.A[i][j];
        return r;
    }
    Quadric& operator+=(const Quadric& o){ *this=*this+o; return *this; }

    // Error at point v: v^T Q v
    [[nodiscard]] f32 error(Vec3f v) const {
        f32 p[4]={v.x,v.y,v.z,1};
        f32 qv[4]={};
        for(u32 i=0;i<4;++i) for(u32 j=0;j<4;++j) qv[i]+=A[i][j]*p[j];
        f32 e=0; for(u32 i=0;i<4;++i) e+=p[i]*qv[i];
        return e;
    }

    // Optimal contraction target: solve 3×3 linear system
    [[nodiscard]] std::optional<Vec3f> optimal_vertex() const {
        DenseMatrix<f64> M(3,3);
        for(u32 i=0;i<3;++i) for(u32 j=0;j<3;++j) M.at(i,j)=A[i][j];
        std::vector<f64> b={-A[0][3],-A[1][3],-A[2][3]};
        auto sol=solve(M,b);
        if(!sol) return std::nullopt;
        return Vec3f{f32((*sol)[0]),f32((*sol)[1]),f32((*sol)[2])};
    }
};

// Build per-vertex quadrics from mesh
[[nodiscard]] inline std::vector<Quadric> build_quadrics(
    const std::vector<Vec3f>& verts,
    const std::vector<std::array<u32,3>>& tris)
{
    std::vector<Quadric> Q(verts.size());
    for(const auto& t:tris) {
        Vec3f n=normalize(cross(verts[t[1]]-verts[t[0]],verts[t[2]]-verts[t[0]]));
        f32 d=-dot(n,verts[t[0]]);
        auto q=Quadric::from_plane(n,d);
        for(u32 v:t) Q[v]+=q;
    }
    return Q;
}

} // namespace qs
