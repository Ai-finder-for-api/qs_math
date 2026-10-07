#pragma once
// qs_math/mesh_boolean.hpp — Triangle mesh intersection, boolean ops (union/subtract/intersect) (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "geometry3d.hpp"
#include "intersection.hpp"
#include "csg_math.hpp"
#include <vector>
#include <array>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <optional>
#include <functional>

namespace qs {

// ─── Triangle soup ────────────────────────────────────────────────────────────
struct TriMesh {
    std::vector<Vec3f>             verts;
    std::vector<std::array<u32,3>> tris;

    [[nodiscard]] Vec3f tri_normal(u32 i) const {
        auto& t=tris[i];
        return normalize(cross(verts[t[1]]-verts[t[0]],verts[t[2]]-verts[t[0]]));
    }
    [[nodiscard]] Vec3f centroid(u32 i) const {
        auto& t=tris[i]; return (verts[t[0]]+verts[t[1]]+verts[t[2]])*(1.f/3);
    }
    [[nodiscard]] AABB<f32> aabb() const {
        AABB<f32> b; bool first=true;
        for(auto& v:verts){ if(first){b.min=b.max=v;first=false;} b.expand(v); }
        return b;
    }
    [[nodiscard]] u32 add_vertex(Vec3f p) { verts.push_back(p); return u32(verts.size()-1); }
    void add_triangle(u32 a,u32 b,u32 c) { tris.push_back({a,b,c}); }
};

// ─── Mesh-mesh intersection: collect all pairs of intersecting triangles ──────
struct TriTriIntersection {
    u32 tri_a, tri_b;
    Vec3f seg_start, seg_end; // intersection segment
};

// Möller triangle-triangle intersection test
[[nodiscard]] inline bool tri_tri_test(Vec3f a0,Vec3f a1,Vec3f a2,
    Vec3f b0,Vec3f b1,Vec3f b2)
{
    // Plane of B: n2 * (x - b0) = 0
    Vec3f n2=cross(b1-b0,b2-b0);
    f32 d0=dot(n2,a0-b0),d1=dot(n2,a1-b0),d2=dot(n2,a2-b0);
    if(d0*d1>0&&d0*d2>0) return false; // all on same side
    Vec3f n1=cross(a1-a0,a2-a0);
    f32 e0=dot(n1,b0-a0),e1=dot(n1,b1-a0),e2=dot(n1,b2-a0);
    if(e0*e1>0&&e0*e2>0) return false;
    return true;
}

// Compute intersection segment between two coplanar-rejected triangles
[[nodiscard]] inline std::optional<std::pair<Vec3f,Vec3f>> tri_tri_segment(
    Vec3f a0,Vec3f a1,Vec3f a2, Vec3f b0,Vec3f b1,Vec3f b2)
{
    if(!tri_tri_test(a0,a1,a2,b0,b1,b2)) return std::nullopt;
    // Intersection line direction: cross of both normals
    Vec3f n1=cross(a1-a0,a2-a0), n2=cross(b1-b0,b2-b0);
    Vec3f D=cross(n1,n2);
    if(length(D)<1e-7f) return std::nullopt; // coplanar
    D=normalize(D);
    // Project triangle vertices onto line
    auto proj_tri=[&](Vec3f p0,Vec3f p1,Vec3f p2,Vec3f n)->std::pair<f32,f32>{
        f32 d=dot(n,cross(n,p0)); // plane offset
        f32 t0=dot(D,p0),t1=dot(D,p1),t2=dot(D,p2);
        f32 v0=dot(n,p0-p0),v1=dot(n,p1-p0),v2=dot(n,p2-p0);
        // Simplified interval
        f32 mn=std::min({t0,t1,t2}), mx=std::max({t0,t1,t2});
        return {mn,mx};
    };
    auto [a_min,a_max]=proj_tri(a0,a1,a2,n2);
    auto [b_min,b_max]=proj_tri(b0,b1,b2,n1);
    f32 seg_min=std::max(a_min,b_min), seg_max=std::min(a_max,b_max);
    if(seg_min>seg_max) return std::nullopt;
    // Find a point on the intersection line
    // Solve n1·x=d1, n2·x=d2 for a point on line
    f32 d1=dot(n1,a0),d2=dot(n2,b0);
    // Use Cramer (pick two components of D as free)
    Vec3f p_line; // point on intersection line
    f32 ax=std::abs(D.x),ay=std::abs(D.y),az=std::abs(D.z);
    if(ax>=ay&&ax>=az){
        f32 det=n1.y*n2.z-n1.z*n2.y;
        if(std::abs(det)>1e-10f){p_line.y=(d1*n2.z-d2*n1.z)/det;p_line.z=(n1.y*d2-n2.y*d1)/det;}
        p_line.x=(d1-n1.y*p_line.y-n1.z*p_line.z)/(n1.x+1e-20f);
    } else if(ay>=az){
        f32 det=n1.x*n2.z-n1.z*n2.x;
        if(std::abs(det)>1e-10f){p_line.x=(d1*n2.z-d2*n1.z)/det;p_line.z=(n1.x*d2-n2.x*d1)/det;}
        p_line.y=(d1-n1.x*p_line.x-n1.z*p_line.z)/(n1.y+1e-20f);
    } else {
        f32 det=n1.x*n2.y-n1.y*n2.x;
        if(std::abs(det)>1e-10f){p_line.x=(d1*n2.y-d2*n1.y)/det;p_line.y=(n1.x*d2-n2.x*d1)/det;}
        p_line.z=(d1-n1.x*p_line.x-n1.y*p_line.y)/(n1.z+1e-20f);
    }
    return {{p_line+D*seg_min, p_line+D*seg_max}};
}

// Find all intersecting triangle pairs between two meshes (brute force O(n²))
[[nodiscard]] inline std::vector<TriTriIntersection> mesh_intersections(
    const TriMesh& A, const TriMesh& B)
{
    std::vector<TriTriIntersection> result;
    for(u32 i=0;i<u32(A.tris.size());++i) {
        auto [a0,a1,a2]=A.tris[i];
        Vec3f va0=A.verts[a0],va1=A.verts[a1],va2=A.verts[a2];
        for(u32 j=0;j<u32(B.tris.size());++j) {
            auto [b0,b1,b2]=B.tris[j];
            Vec3f vb0=B.verts[b0],vb1=B.verts[b1],vb2=B.verts[b2];
            auto seg=tri_tri_segment(va0,va1,va2,vb0,vb1,vb2);
            if(seg) result.push_back({i,j,seg->first,seg->second});
        }
    }
    return result;
}

// ─── Point-in-mesh test (winding number, O(n)) ────────────────────────────────
[[nodiscard]] inline bool point_in_mesh(const TriMesh& mesh, Vec3f p) {
    f32 total=0;
    for(u32 i=0;i<u32(mesh.tris.size());++i) {
        auto& t=mesh.tris[i];
        Vec3f a=mesh.verts[t[0]]-p, b=mesh.verts[t[1]]-p, c=mesh.verts[t[2]]-p;
        f32 la=length(a),lb=length(b),lc=length(c);
        if(la<1e-7f||lb<1e-7f||lc<1e-7f) continue;
        Vec3f an=a/la,bn=b/lb,cn=c/lc;
        f32 num=dot(an,cross(bn,cn));
        f32 den=1+dot(an,bn)+dot(bn,cn)+dot(cn,an);
        total+=2*std::atan2(num,den);
    }
    return std::abs(total)>f32(std::numbers::pi);
}

// ─── Mesh boolean via BSP (solid classification) ──────────────────────────────
enum class BoolOp { Union, Subtract, Intersect };

[[nodiscard]] inline TriMesh mesh_boolean(const TriMesh& A, const TriMesh& B, BoolOp op) {
    // Convert meshes to BSP polygons
    auto mesh_to_bsp=[](const TriMesh& m)->std::vector<BSPPolygon>{
        std::vector<BSPPolygon> polys;
        for(auto& t:m.tris) {
            BSPPolygon p;
            p.verts={m.verts[t[0]],m.verts[t[1]],m.verts[t[2]]};
            p.normal=normalize(cross(p.verts[1]-p.verts[0],p.verts[2]-p.verts[0]));
            p.d=dot(p.normal,p.verts[0]);
            polys.push_back(p);
        }
        return polys;
    };
    auto bsp_to_mesh=[](const std::vector<BSPPolygon>& polys)->TriMesh{
        TriMesh m;
        for(const auto& p:polys) {
            u32 base=u32(m.verts.size());
            for(auto& v:p.verts) m.verts.push_back(v);
            for(u32 i=1;i+1<u32(p.verts.size());++i) m.tris.push_back({base,base+i,base+i+1});
        }
        return m;
    };
    auto a_polys=mesh_to_bsp(A), b_polys=mesh_to_bsp(B);
    std::vector<BSPPolygon> result;
    switch(op) {
        case BoolOp::Union:     result=csg_bsp_union(a_polys,b_polys);    break;
        case BoolOp::Subtract:  result=csg_bsp_subtract(a_polys,b_polys); break;
        case BoolOp::Intersect: result=csg_bsp_intersect(a_polys,b_polys);break;
    }
    return bsp_to_mesh(result);
}

// ─── Mesh repair helpers ──────────────────────────────────────────────────────
// Remove duplicate vertices within epsilon
inline void merge_vertices(TriMesh& mesh, f32 eps=1e-5f) {
    u32 n=u32(mesh.verts.size());
    std::vector<u32> remap(n);
    std::vector<Vec3f> unique_verts;
    for(u32 i=0;i<n;++i) {
        u32 found=u32(unique_verts.size());
        for(u32 j=0;j<u32(unique_verts.size());++j)
            if(dist_sq(mesh.verts[i],unique_verts[j])<eps*eps){found=j;break;}
        if(found==u32(unique_verts.size())) unique_verts.push_back(mesh.verts[i]);
        remap[i]=found;
    }
    mesh.verts=unique_verts;
    for(auto& t:mesh.tris){ t[0]=remap[t[0]];t[1]=remap[t[1]];t[2]=remap[t[2]]; }
}

// Remove degenerate triangles (zero area)
inline void remove_degenerate(TriMesh& mesh, f32 eps=1e-7f) {
    mesh.tris.erase(std::remove_if(mesh.tris.begin(),mesh.tris.end(),[&](const auto& t){
        Vec3f e1=mesh.verts[t[1]]-mesh.verts[t[0]],e2=mesh.verts[t[2]]-mesh.verts[t[0]];
        return length(cross(e1,e2))<eps;
    }),mesh.tris.end());
}

// Ensure consistent outward winding (flip triangles facing inward relative to centroid)
inline void orient_outward(TriMesh& mesh) {
    Vec3f center{};
    for(auto& v:mesh.verts) center=center+v;
    center=center*(1.f/std::max(1u,u32(mesh.verts.size())));
    for(auto& t:mesh.tris) {
        Vec3f n=cross(mesh.verts[t[1]]-mesh.verts[t[0]],mesh.verts[t[2]]-mesh.verts[t[0]]);
        Vec3f c=(mesh.verts[t[0]]+mesh.verts[t[1]]+mesh.verts[t[2]])*(1.f/3);
        if(dot(n,c-center)<0) std::swap(t[1],t[2]);
    }
}

} // namespace qs
