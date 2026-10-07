#pragma once
// qs_math/procedural_geometry.hpp — Platonic solids, parametric surfaces, extrusion (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace qs {

// ─── Mesh data ────────────────────────────────────────────────────────────────

struct ProcMesh {
    std::vector<Vec3f> positions;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> uvs;
    std::vector<u32>   indices;

    void clear() { positions.clear(); normals.clear(); uvs.clear(); indices.clear(); }

    u32 add_vertex(const Vec3f& p, const Vec3f& n, const Vec2f& uv) {
        positions.push_back(p); normals.push_back(n); uvs.push_back(uv);
        return u32(positions.size())-1;
    }

    void add_triangle(u32 a, u32 b, u32 c) {
        indices.push_back(a); indices.push_back(b); indices.push_back(c);
    }
};

// ─── UV Sphere ────────────────────────────────────────────────────────────────

[[nodiscard]] inline ProcMesh make_sphere(f32 r=1.f, u32 slices=32, u32 stacks=16) {
    ProcMesh m;
    for (u32 j=0;j<=stacks;++j) {
        f32 v = f32(j)/f32(stacks);
        f32 phi = v*k_pi;
        for (u32 i=0;i<=slices;++i) {
            f32 u = f32(i)/f32(slices);
            f32 theta = u*k_tau;
            Vec3f n{std::sin(phi)*std::cos(theta),std::cos(phi),std::sin(phi)*std::sin(theta)};
            m.add_vertex(n*r, n, {u,v});
        }
    }
    for (u32 j=0;j<stacks;++j)
    for (u32 i=0;i<slices;++i) {
        u32 a=(j)*(slices+1)+i, b=a+1, c=(j+1)*(slices+1)+i, d=c+1;
        m.add_triangle(a,b,c); m.add_triangle(b,d,c);
    }
    return m;
}

// ─── Box ─────────────────────────────────────────────────────────────────────

[[nodiscard]] inline ProcMesh make_box(Vec3f half=Vec3f{0.5f,0.5f,0.5f}) {
    ProcMesh m;
    static constexpr Vec3f ns[6]={{0,0,1},{0,0,-1},{0,1,0},{0,-1,0},{1,0,0},{-1,0,0}};
    static constexpr Vec3f us[6]={{1,0,0},{-1,0,0},{1,0,0},{1,0,0},{0,0,-1},{0,0,1}};
    for (u32 f=0;f<6;++f) {
        Vec3f n=ns[f], u2=us[f], v2=cross(n,u2);
        Vec3f c=n*half;  // adjust by face half-extents
        // We scale u/v by the appropriate half-extents
        Vec3f hh{std::abs(dot(u2,half))*u2+std::abs(dot(v2,half))*v2};
        u32 base=u32(m.positions.size());
        m.add_vertex(c+u2*std::abs(dot(u2,half))+v2*std::abs(dot(v2,half)), n, {1,1});
        m.add_vertex(c-u2*std::abs(dot(u2,half))+v2*std::abs(dot(v2,half)), n, {0,1});
        m.add_vertex(c-u2*std::abs(dot(u2,half))-v2*std::abs(dot(v2,half)), n, {0,0});
        m.add_vertex(c+u2*std::abs(dot(u2,half))-v2*std::abs(dot(v2,half)), n, {1,0});
        m.add_triangle(base,base+1,base+2); m.add_triangle(base,base+2,base+3);
    }
    return m;
}

// ─── Cylinder ────────────────────────────────────────────────────────────────

[[nodiscard]] inline ProcMesh make_cylinder(f32 r=0.5f, f32 h=1.f, u32 slices=32) {
    ProcMesh m;
    f32 half_h = h*0.5f;
    for (u32 i=0;i<=slices;++i) {
        f32 u = f32(i)/f32(slices);
        f32 theta = u*k_tau;
        f32 c=std::cos(theta), s=std::sin(theta);
        Vec3f n{c,0,s};
        m.add_vertex({c*r, half_h,s*r}, n, {u,1});
        m.add_vertex({c*r,-half_h,s*r}, n, {u,0});
    }
    for (u32 i=0;i<slices;++i) {
        u32 a=i*2,b=a+1,c=a+2,d=a+3;
        m.add_triangle(a,b,c); m.add_triangle(b,d,c);
    }
    // Caps
    auto add_cap = [&](f32 y, bool flip) {
        u32 center = u32(m.positions.size());
        Vec3f cn{0,flip?-1.f:1.f,0};
        m.add_vertex({0,y,0}, cn, {0.5f,0.5f});
        for (u32 i=0;i<=slices;++i) {
            f32 theta=f32(i)/f32(slices)*k_tau;
            m.add_vertex({std::cos(theta)*r,y,std::sin(theta)*r},cn,
                {std::cos(theta)*0.5f+0.5f, std::sin(theta)*0.5f+0.5f});
        }
        for (u32 i=0;i<slices;++i) {
            u32 a=center+1+i, b=center+1+i+1;
            if (flip) m.add_triangle(center,b,a);
            else      m.add_triangle(center,a,b);
        }
    };
    add_cap( half_h, false);
    add_cap(-half_h, true);
    return m;
}

// ─── Torus ────────────────────────────────────────────────────────────────────

[[nodiscard]] inline ProcMesh make_torus(f32 R=1.f, f32 r=0.3f, u32 slices=32, u32 stacks=16) {
    ProcMesh m;
    for (u32 j=0;j<=stacks;++j) {
        f32 v=f32(j)/f32(stacks), phi=v*k_tau;
        for (u32 i=0;i<=slices;++i) {
            f32 u2=f32(i)/f32(slices), theta=u2*k_tau;
            f32 x=(R+r*std::cos(phi))*std::cos(theta);
            f32 y=r*std::sin(phi);
            f32 z=(R+r*std::cos(phi))*std::sin(theta);
            Vec3f center{R*std::cos(theta),0,R*std::sin(theta)};
            Vec3f n=normalize(Vec3f{x,y,z}-center);
            m.add_vertex({x,y,z}, n, {u2,v});
        }
    }
    for (u32 j=0;j<stacks;++j)
    for (u32 i=0;i<slices;++i) {
        u32 a=j*(slices+1)+i, b=a+1, c=(j+1)*(slices+1)+i, d=c+1;
        m.add_triangle(a,b,c); m.add_triangle(b,d,c);
    }
    return m;
}

// ─── Profile extrusion ────────────────────────────────────────────────────────

[[nodiscard]] inline ProcMesh extrude_profile(
    std::span<const Vec2f> profile,   // 2D cross section (closed)
    std::span<const Vec3f> spine,     // 3D spine path
    bool close_spine=false)
{
    ProcMesh m;
    u32 NP=u32(profile.size()), NS=u32(spine.size());
    if (NS<2||NP<2) return m;
    for (u32 s=0;s<NS;++s) {
        Vec3f tangent = (s<NS-1)
            ? normalize(spine[s+1]-spine[s])
            : normalize(spine[s]-spine[s-1]);
        Vec3f up = (std::abs(tangent.y)<0.9f)?Vec3f{0,1,0}:Vec3f{1,0,0};
        Vec3f right = normalize(cross(tangent,up));
        Vec3f up2   = cross(right,tangent);
        for (u32 p=0;p<NP;++p) {
            Vec3f pt = spine[s] + right*profile[p].x + up2*profile[p].y;
            m.add_vertex(pt, right, {f32(p)/f32(NP-1), f32(s)/f32(NS-1)});
        }
    }
    for (u32 s=0;s<NS-1;++s)
    for (u32 p=0;p<NP-1;++p) {
        u32 a=s*NP+p, b=a+1, c=(s+1)*NP+p, d=c+1;
        m.add_triangle(a,b,c); m.add_triangle(b,d,c);
    }
    return m;
}

// ─── Icosphere ────────────────────────────────────────────────────────────────

[[nodiscard]] inline ProcMesh make_icosphere(f32 r=1.f, u32 subdivisions=2) {
    // Start with icosahedron
    f32 t = (1.f+std::sqrt(5.f))*0.5f;
    std::vector<Vec3f> verts = {
        normalize(Vec3f{-1,t,0}),normalize(Vec3f{1,t,0}),normalize(Vec3f{-1,-t,0}),normalize(Vec3f{1,-t,0}),
        normalize(Vec3f{0,-1,t}),normalize(Vec3f{0,1,t}),normalize(Vec3f{0,-1,-t}),normalize(Vec3f{0,1,-t}),
        normalize(Vec3f{t,0,-1}),normalize(Vec3f{t,0,1}),normalize(Vec3f{-t,0,-1}),normalize(Vec3f{-t,0,1})};
    std::vector<std::array<u32,3>> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},{1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},{4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
    for (u32 s=0;s<subdivisions;++s) {
        std::vector<std::array<u32,3>> new_faces;
        for (auto& f2 : faces) {
            Vec3f a=normalize(verts[f2[0]]+verts[f2[1]]);
            Vec3f b=normalize(verts[f2[1]]+verts[f2[2]]);
            Vec3f c=normalize(verts[f2[2]]+verts[f2[0]]);
            u32 ai=u32(verts.size()); verts.push_back(a);
            u32 bi=ai+1; verts.push_back(b);
            u32 ci=ai+2; verts.push_back(c);
            new_faces.push_back({f2[0],ai,ci}); new_faces.push_back({f2[1],bi,ai});
            new_faces.push_back({f2[2],ci,bi}); new_faces.push_back({ai,bi,ci});
        }
        faces=std::move(new_faces);
    }
    ProcMesh m;
    for (auto& v : verts) m.add_vertex(v*r, v, {std::atan2(v.z,v.x)/k_tau+0.5f, v.y*0.5f+0.5f});
    for (auto& f2 : faces) m.add_triangle(f2[0],f2[1],f2[2]);
    return m;
}

} // namespace qs
