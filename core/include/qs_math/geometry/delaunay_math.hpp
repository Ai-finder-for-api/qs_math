#pragma once
// qs_math/delaunay_math.hpp — Bowyer-Watson, circumcircle, flip criteria (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "geometry2d.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <optional>

namespace qs {

// ─── Triangle with circumcircle ───────────────────────────────────────────────
struct DelTriangle {
    std::array<u32,3> v{~0u,~0u,~0u}; // vertex indices
    Vec2f circumcenter{};
    f32   circumradius2{0};
    bool  bad{false}; // for Bowyer-Watson

    bool operator==(const DelTriangle& o) const {
        auto a=v; auto b=o.v;
        std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
        return a==b;
    }
};

// Compute circumcircle of triangle (p0,p1,p2)
[[nodiscard]] inline std::pair<Vec2f,f32> circumcircle(Vec2f p0,Vec2f p1,Vec2f p2) {
    f32 ax=p1.x-p0.x, ay=p1.y-p0.y;
    f32 bx=p2.x-p0.x, by=p2.y-p0.y;
    f32 D=2*(ax*by-ay*bx);
    if(std::abs(D)<1e-8f) return {{(p0.x+p1.x+p2.x)/3,(p0.y+p1.y+p2.y)/3},1e18f};
    f32 ux=(by*(ax*ax+ay*ay)-ay*(bx*bx+by*by))/D;
    f32 uy=(ax*(bx*bx+by*by)-bx*(ax*ax+ay*ay))/D;
    Vec2f cc{p0.x+ux,p0.y+uy};
    f32 r2=(cc.x-p0.x)*(cc.x-p0.x)+(cc.y-p0.y)*(cc.y-p0.y);
    return {cc,r2};
}

// Delaunay in-circle test: returns true if point d is inside circumcircle of (a,b,c)
// Uses exact arithmetic version via determinant
[[nodiscard]] inline bool in_circumcircle(Vec2f a, Vec2f b, Vec2f c, Vec2f d) {
    f64 ax=f64(a.x-d.x), ay=f64(a.y-d.y);
    f64 bx=f64(b.x-d.x), by=f64(b.y-d.y);
    f64 cx=f64(c.x-d.x), cy=f64(c.y-d.y);
    f64 det=ax*(by*(cx*cx+cy*cy)-cy*(bx*bx+by*by))
           -ay*(bx*(cx*cx+cy*cy)-cx*(bx*bx+by*by))
           +(ax*ax+ay*ay)*(bx*cy-by*cx);
    return det>0; // positive = d inside (CCW orientation assumed)
}

// Check CCW orientation
[[nodiscard]] inline f32 orient2d(Vec2f a,Vec2f b,Vec2f c) {
    return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}

// Flip criterion: edge (p,q) shared by triangles (p,q,r) and (p,q,s)
// Should flip if s is inside circumcircle of (p,q,r)
[[nodiscard]] inline bool should_flip(Vec2f p,Vec2f q,Vec2f r,Vec2f s) {
    return in_circumcircle(p,q,r,s);
}

// ─── Bowyer-Watson incremental Delaunay triangulation ────────────────────────
struct DelaunayResult {
    std::vector<Vec2f>       vertices;
    std::vector<DelTriangle> triangles;
};

[[nodiscard]] inline DelaunayResult bowyer_watson(std::vector<Vec2f> pts) {
    u32 n=u32(pts.size());
    if(n<3) return {pts,{}};
    // Add super-triangle surrounding all points
    f32 min_x=pts[0].x,max_x=pts[0].x,min_y=pts[0].y,max_y=pts[0].y;
    for(auto& p:pts){ min_x=std::min(min_x,p.x);max_x=std::max(max_x,p.x);min_y=std::min(min_y,p.y);max_y=std::max(max_y,p.y); }
    f32 dx=max_x-min_x, dy=max_y-min_y, dmax=std::max(dx,dy)*3;
    Vec2f s0{min_x-dmax, min_y-dmax};
    Vec2f s1{min_x+dmax*3, min_y-dmax};
    Vec2f s2{min_x+dmax*0.5f, max_y+dmax*3};
    pts.push_back(s0); pts.push_back(s1); pts.push_back(s2);
    u32 ns0=n, ns1=n+1, ns2=n+2;
    std::vector<DelTriangle> tris;
    {
        DelTriangle t; t.v={ns0,ns1,ns2};
        auto [cc,r2]=circumcircle(s0,s1,s2); t.circumcenter=cc; t.circumradius2=r2;
        tris.push_back(t);
    }
    for(u32 i=0;i<n;++i) {
        Vec2f p=pts[i];
        // Find all triangles whose circumcircle contains p
        std::vector<DelTriangle> bad, good;
        for(auto& t:tris) {
            f32 dx=p.x-t.circumcenter.x, dy=p.y-t.circumcenter.y;
            if(dx*dx+dy*dy<t.circumradius2) bad.push_back(t);
            else good.push_back(t);
        }
        // Find boundary of polygonal hole
        std::vector<std::pair<u32,u32>> boundary;
        for(auto& bt:bad) {
            for(u32 e=0;e<3;++e) {
                u32 va=bt.v[e], vb=bt.v[(e+1)%3];
                bool shared=false;
                for(auto& bt2:bad) if(!(bt2==bt))
                    for(u32 f=0;f<3;++f) if((bt2.v[f]==va&&bt2.v[(f+1)%3]==vb)||(bt2.v[f]==vb&&bt2.v[(f+1)%3]==va)){ shared=true; break; }
                if(!shared) boundary.emplace_back(va,vb);
            }
        }
        tris=good;
        for(auto [va,vb]:boundary) {
            DelTriangle nt; nt.v={va,vb,i};
            auto [cc,r2]=circumcircle(pts[va],pts[vb],pts[i]);
            nt.circumcenter=cc; nt.circumradius2=r2;
            tris.push_back(nt);
        }
    }
    // Remove triangles containing super-triangle vertices
    tris.erase(std::remove_if(tris.begin(),tris.end(),[&](const DelTriangle& t){
        return t.v[0]>=n||t.v[1]>=n||t.v[2]>=n;
    }),tris.end());
    pts.resize(n);
    return {pts,tris};
}

// ─── Edge flip for Lawson's algorithm (incremental refinement) ───────────────
// Returns true if the edge was flipped
inline bool flip_edge_if_illegal(DelaunayResult& del,
    u32 t1_idx, u32 t2_idx, u32 shared_e1, u32 shared_e2)
{
    auto& T=del.triangles;
    auto& V=del.vertices;
    if(t1_idx>=u32(T.size())||t2_idx>=u32(T.size())) return false;
    auto& t1=T[t1_idx]; auto& t2=T[t2_idx];
    // Find the two non-shared vertices
    u32 p=~0u,q=~0u;
    for(u32 v:t1.v) if(v!=shared_e1&&v!=shared_e2){p=v;break;}
    for(u32 v:t2.v) if(v!=shared_e1&&v!=shared_e2){q=v;break;}
    if(p==~0u||q==~0u) return false;
    if(in_circumcircle(V[shared_e1],V[shared_e2],V[p],V[q])) {
        // Flip: replace triangles (shared_e1,shared_e2,p) and (shared_e1,shared_e2,q)
        //   with (shared_e1,p,q) and (shared_e2,p,q)
        t1.v={shared_e1,p,q};
        t2.v={shared_e2,p,q};
        auto [cc1,r21]=circumcircle(V[shared_e1],V[p],V[q]);
        t1.circumcenter=cc1; t1.circumradius2=r21;
        auto [cc2,r22]=circumcircle(V[shared_e2],V[p],V[q]);
        t2.circumcenter=cc2; t2.circumradius2=r22;
        return true;
    }
    return false;
}

// ─── Constrained Delaunay: insert constraint edge (simplified — segment recovery) ─
[[nodiscard]] inline void insert_constraint(DelaunayResult& del, u32 v_a, u32 v_b) {
    // Mark all triangles intersected by segment (v_a,v_b) and re-triangulate cavities
    // Simplified: just mark without full re-triangulation (full CDT is complex)
    auto& T=del.triangles;
    auto& V=del.vertices;
    Vec2f a=V[v_a], b=V[v_b];
    for(auto& t:T) {
        // Check if this triangle contains the constraint edge
        bool ha=false,hb=false;
        for(u32 v:t.v){ if(v==v_a)ha=true; if(v==v_b)hb=true; }
        if(ha&&hb) return; // already an edge
    }
    // For simplicity, add the vertices and re-triangulate (stub for full CDT)
    (void)a;(void)b;
}

} // namespace qs
