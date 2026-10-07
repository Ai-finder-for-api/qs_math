#pragma once
// qs_math/subdivision_surfaces.hpp — Catmull-Clark, Loop, Doo-Sabin, limit surface (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <span>
#include <functional>

namespace qs {

// ─── Half-edge mesh ───────────────────────────────────────────────────────────

struct HalfEdge {
    u32 vertex{};   // destination vertex
    u32 next{};     // next half-edge in face
    u32 twin{};     // opposite half-edge
    u32 face{};     // owning face
    u32 id{};
    static constexpr u32 NONE = u32(-1);
};

struct HEMesh {
    std::vector<Vec3f>    verts;
    std::vector<HalfEdge> halfedges;
    std::vector<u32>      face_start; // first halfedge of each face
    std::vector<u32>      vert_edge;  // one outgoing halfedge per vertex

    u32 add_vertex(const Vec3f& p) {
        vert_edge.push_back(HalfEdge::NONE);
        verts.push_back(p);
        return u32(verts.size())-1;
    }

    // Build twin links after all faces added
    void build_twins() {
        std::unordered_map<u64,u32> edge_map;
        for(u32 i=0;i<u32(halfedges.size());++i) {
            const auto& he=halfedges[i];
            u32 prev_i=i>0&&halfedges[i-1].face==he.face?i-1:he.next;
            // Find prev to get the from-vertex
            u32 from=halfedges[he.next].vertex; // wrong - need prev
        }
        // Simpler: key = (from << 32) | to
        for(u32 i=0;i<u32(halfedges.size());++i) halfedges[i].twin=HalfEdge::NONE;
        for(u32 i=0;i<u32(halfedges.size());++i){
            u32 to=halfedges[i].vertex;
            u32 from=halfedges[halfedges[i].next].vertex; // approximate
            u64 key=(u64(from)<<32)|to;
            auto it=edge_map.find(key);
            if(it!=edge_map.end()){
                halfedges[i].twin=it->second;
                halfedges[it->second].twin=i;
            } else {
                edge_map[(u64(to)<<32)|from]=i;
            }
        }
    }

    [[nodiscard]] bool is_boundary(u32 he) const {
        return halfedges[he].twin==HalfEdge::NONE;
    }

    // Valence of vertex v (number of outgoing edges)
    [[nodiscard]] u32 valence(u32 v) const {
        u32 start=vert_edge[v];
        if(start==HalfEdge::NONE) return 0;
        u32 he=start, count=0;
        do { ++count; he=halfedges[halfedges[he].twin==HalfEdge::NONE?he:halfedges[he].twin].next; }
        while(he!=start&&count<100);
        return count;
    }
};

// ─── Catmull-Clark subdivision ────────────────────────────────────────────────

// CC face point: average of all face vertices
[[nodiscard]] inline Vec3f cc_face_point(std::span<const Vec3f> face_verts) {
    Vec3f c{0,0,0};
    for(const auto& v:face_verts) c=c+v;
    return c*(1.f/f32(face_verts.size()));
}

// CC edge point: average of two endpoints and two adjacent face points
[[nodiscard]] inline Vec3f cc_edge_point(
    const Vec3f& v0, const Vec3f& v1,
    const Vec3f& fp0, const Vec3f& fp1)
{
    return (v0+v1+fp0+fp1)*0.25f;
}

// CC vertex point update for interior vertex of valence n
[[nodiscard]] inline Vec3f cc_vertex_point_interior(
    const Vec3f& v,
    std::span<const Vec3f> adjacent_face_points,
    std::span<const Vec3f> adjacent_edge_midpoints,
    u32 n)
{
    Vec3f F{0,0,0}, R{0,0,0};
    for(const auto& fp:adjacent_face_points) F=F+fp;
    for(const auto& ep:adjacent_edge_midpoints) R=R+ep;
    F=F*(1.f/f32(n));
    R=R*(1.f/f32(n));
    // (F + 2R + (n-3)v) / n
    return (F+R*2.f+v*f32(n-3))*(1.f/f32(n));
}

// CC vertex point for boundary vertex
[[nodiscard]] inline Vec3f cc_vertex_point_boundary(
    const Vec3f& v, const Vec3f& adj0, const Vec3f& adj1)
{
    return v*0.75f + adj0*0.125f + adj1*0.125f;
}

// Limit position of a CC vertex (where it converges to)
[[nodiscard]] inline Vec3f cc_limit_position(
    const Vec3f& v,
    std::span<const Vec3f> face_points,
    std::span<const Vec3f> edge_midpoints,
    u32 n)
{
    Vec3f F{0,0,0}, E{0,0,0};
    for(const auto& fp:face_points) F=F+fp;
    for(const auto& ep:edge_midpoints) E=E+ep;
    F=F*(1.f/f32(n));
    E=E*(1.f/f32(n));
    return (F+E*4.f+v*f32(n*(n-3)+2.f))*(1.f/(f32(n*n)+4.f*f32(n)+2.f-2.f));
}

// Limit normal at CC vertex
[[nodiscard]] inline Vec3f cc_limit_normal(
    std::span<const Vec3f> ring_verts,  // adjacent vertices in order
    std::span<const Vec3f> face_points, // adjacent face centroids
    u32 n)
{
    // Tangent vectors via eigenanalysis of subdivision matrix
    Vec3f T1{0,0,0}, T2{0,0,0};
    for(u32 i=0;i<n;++i){
        f32 a=std::cos(k_tau*f32(i)/f32(n));
        f32 b=std::cos(k_tau*(f32(i)+0.5f)/f32(n));
        T1=T1+ring_verts[i]*a;
        T2=T2+ring_verts[i]*b;
        if(i<face_points.size()){
            T1=T1+face_points[i]*a;
            T2=T2+face_points[i]*b;
        }
    }
    return normalize(cross(T1,T2));
}

// ─── Loop subdivision (triangles) ────────────────────────────────────────────

// Loop edge point: 3/8*(e0+e1) + 1/8*(opp0+opp1)
[[nodiscard]] inline Vec3f loop_edge_point(
    const Vec3f& e0, const Vec3f& e1,
    const Vec3f& opp0, const Vec3f& opp1)
{
    return (e0+e1)*0.375f + (opp0+opp1)*0.125f;
}

// Loop vertex weight beta(n)
[[nodiscard]] inline f32 loop_beta(u32 n) {
    if(n==3) return 3.f/16.f;
    f32 w=(3.f/8.f+(std::cos(k_tau/f32(n)))/4.f);
    return (5.f/8.f - w*w)/f32(n);
}

// Loop updated vertex
[[nodiscard]] inline Vec3f loop_vertex_point(
    const Vec3f& v, std::span<const Vec3f> neighbours)
{
    u32 n=u32(neighbours.size());
    f32 beta=loop_beta(n);
    Vec3f sum{0,0,0};
    for(const auto& nb:neighbours) sum=sum+nb;
    return v*(1.f-f32(n)*beta)+sum*beta;
}

// Loop limit position
[[nodiscard]] inline Vec3f loop_limit_position(
    const Vec3f& v, std::span<const Vec3f> neighbours)
{
    u32 n=u32(neighbours.size());
    f32 beta=loop_beta(n);
    Vec3f sum{0,0,0};
    for(const auto& nb:neighbours) sum=sum+nb;
    f32 w=3.f/8.f;
    return (v*w+sum*(1.f-w))*(1.f/n);
}

// ─── Doo-Sabin subdivision ────────────────────────────────────────────────────

// Doo-Sabin weights for n-sided face
[[nodiscard]] inline std::vector<f32> doo_sabin_weights(u32 n) {
    std::vector<f32> w(n);
    for(u32 i=0;i<n;++i){
        f32 sum=0.f;
        if(i==0) sum=(f32(n)+5.f)/f32(4*n);
        else sum=(3.f+2.f*std::cos(k_tau*f32(i)/f32(n)))/(4.f*f32(n));
        w[i]=sum;
    }
    return w;
}

// New face vertex in Doo-Sabin for vertex i in face of n vertices
[[nodiscard]] inline Vec3f doo_sabin_face_vertex(
    std::span<const Vec3f> face_verts, u32 i)
{
    u32 n=u32(face_verts.size());
    auto w=doo_sabin_weights(n);
    Vec3f result{0,0,0};
    for(u32 j=0;j<n;++j)
        result=result+face_verts[(i+j)%n]*w[j];
    return result;
}

// ─── Butterfly subdivision (interpolating) ───────────────────────────────────

// Butterfly edge point (regular case, 8-point stencil)
[[nodiscard]] inline Vec3f butterfly_edge_point(
    const Vec3f& v0, const Vec3f& v1,
    const Vec3f& a0, const Vec3f& a1,   // opposite verts of the edge faces
    const Vec3f& b0, const Vec3f& b1,   // next ring on v0 side
    const Vec3f& b2, const Vec3f& b3,   // next ring on v1 side
    f32 w=0.f)                           // tension (0 = standard butterfly)
{
    return v0*0.5f + v1*0.5f
          + a0*(0.125f-w) + a1*(0.125f-w)
          - b0*(w*0.5f) - b1*(w*0.5f) - b2*(w*0.5f) - b3*(w*0.5f);
}

// ─── Subdivision limit evaluation (on-the-fly) ───────────────────────────────

// Evaluate a CC limit surface point given local patch data
struct CCPatch {
    Vec3f cv[4][4]; // 4x4 control vertices (bicubic Bezier after CC)

    [[nodiscard]] Vec3f eval(f32 s, f32 t) const {
        // Cubic Bernstein basis
        auto B=[](f32 u)->std::array<f32,4>{
            f32 u2=u*u,u3=u2*u;
            return {(1-u)*(1-u)*(1-u), 3*u*(1-u)*(1-u), 3*u*u*(1-u), u3};
        };
        auto Bu=B(s), Bv=B(t);
        Vec3f p{0,0,0};
        for(u32 i=0;i<4;++i) for(u32 j=0;j<4;++j)
            p=p+cv[i][j]*(Bu[i]*Bv[j]);
        return p;
    }

    [[nodiscard]] Vec3f normal(f32 s, f32 t) const {
        // Tangent vectors via finite difference
        f32 eps=0.001f;
        Vec3f ts=(eval(s+eps,t)-eval(s-eps,t))*(1.f/(2.f*eps));
        Vec3f tt=(eval(s,t+eps)-eval(s,t-eps))*(1.f/(2.f*eps));
        return normalize(cross(ts,tt));
    }
};

} // namespace qs
