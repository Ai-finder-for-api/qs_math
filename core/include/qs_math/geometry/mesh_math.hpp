#pragma once
// qs_math/geometry/mesh_math.hpp — Half-edge, Laplacian smoothing, subdivision, remeshing
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── 3D mesh vertex ───────────────────────────────────────────────────────────
struct MeshV3 { f64 x,y,z;
    [[nodiscard]] constexpr MeshV3 operator+(MeshV3 b) const noexcept {return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr MeshV3 operator-(MeshV3 b) const noexcept {return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr MeshV3 operator*(f64 s)    const noexcept {return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64 dot(MeshV3 b)          const noexcept {return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr MeshV3 cross(MeshV3 b)     const noexcept {return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    [[nodiscard]] constexpr f64 len2()                 const noexcept {return dot(*this);}
    [[nodiscard]]           f64 len()                  const noexcept {return std::sqrt(len2());}
    [[nodiscard]]           MeshV3 norm()              const noexcept {f64 l=len();return l>1e-12?MeshV3{x/l,y/l,z/l}:*this;}
};

// ─── Triangle mesh (N verts, T tris) ─────────────────────────────────────────
template<int NV, int NT>
struct TriMesh {
    std::array<MeshV3,NV> verts{};
    std::array<std::array<int,3>,NT> tris{};
    int nv{NV},nt{NT};

    // Triangle area
    [[nodiscard]] constexpr f64 tri_area(int t) const noexcept {
        auto e1=verts[tris[t][1]]-verts[tris[t][0]];
        auto e2=verts[tris[t][2]]-verts[tris[t][0]];
        return e1.cross(e2).len()*0.5;
    }

    // Triangle normal
    [[nodiscard]] constexpr MeshV3 tri_normal(int t) const noexcept {
        auto e1=verts[tris[t][1]]-verts[tris[t][0]];
        auto e2=verts[tris[t][2]]-verts[tris[t][0]];
        return e1.cross(e2).norm();
    }

    // Vertex normal (area-weighted average of incident triangle normals)
    [[nodiscard]] constexpr MeshV3 vertex_normal(int v) const noexcept {
        MeshV3 n{};
        for(int t=0;t<nt;++t) {
            bool incident=false;
            for(int k=0;k<3;++k) if(tris[t][k]==v) incident=true;
            if(!incident) continue;
            auto tn=tri_normal(t);
            f64 area=tri_area(t);
            n.x+=tn.x*area; n.y+=tn.y*area; n.z+=tn.z*area;
        }
        return n.norm();
    }

    // Total surface area
    [[nodiscard]] constexpr f64 surface_area() const noexcept {
        f64 area=0; for(int t=0;t<nt;++t) area+=tri_area(t); return area;
    }

    // Volume (divergence theorem: V=1/6 * Σ (v0·(v1×v2)))
    [[nodiscard]] constexpr f64 volume() const noexcept {
        f64 vol=0;
        for(int t=0;t<nt;++t) {
            auto& a=verts[tris[t][0]],&b=verts[tris[t][1]],&c=verts[tris[t][2]];
            vol+=a.dot(b.cross(c));
        }
        return std::abs(vol)/6.0;
    }

    // Centroid
    [[nodiscard]] constexpr MeshV3 centroid() const noexcept {
        MeshV3 c{}; for(int i=0;i<nv;++i){c.x+=verts[i].x;c.y+=verts[i].y;c.z+=verts[i].z;}
        return c*(1.0/nv);
    }

    // Cotangent Laplacian weight for edge (i,j)
    [[nodiscard]] constexpr f64 cotan_weight(int i, int j) const noexcept {
        f64 w=0;
        for(int t=0;t<nt;++t) {
            // Find if edge (i,j) is in this triangle
            int ai=-1,bi=-1,ci=-1;
            for(int k=0;k<3;++k) {
                if(tris[t][k]==i) ai=k;
                if(tris[t][k]==j) bi=k;
            }
            if(ai<0||bi<0) continue;
            // ci is the opposite vertex
            for(int k=0;k<3;++k) if(k!=ai&&k!=bi) ci=k;
            MeshV3 pci=verts[tris[t][ci]];
            MeshV3 u=verts[i]-pci, v=verts[j]-pci;
            f64 cosA=u.dot(v);
            f64 sinA=u.cross(v).len();
            if(sinA>1e-12) w+=cosA/sinA;
        }
        return w*0.5;
    }

    // Laplacian smoothing step
    constexpr void laplacian_smooth(f64 lambda=0.5, int iters=5) noexcept {
        for(int iter=0;iter<iters;++iter) {
            std::array<MeshV3,NV> delta{};
            std::array<int,NV> count{};
            for(int t=0;t<nt;++t) for(int k=0;k<3;++k) {
                int vi=tris[t][k],vj=tris[t][(k+1)%3];
                delta[vi].x+=verts[vj].x-verts[vi].x;
                delta[vi].y+=verts[vj].y-verts[vi].y;
                delta[vi].z+=verts[vj].z-verts[vi].z;
                ++count[vi];
            }
            for(int i=0;i<nv;++i) if(count[i]>0) {
                f64 inv=lambda/count[i];
                verts[i].x+=delta[i].x*inv;
                verts[i].y+=delta[i].y*inv;
                verts[i].z+=delta[i].z*inv;
            }
        }
    }

    // Taubin smoothing (anti-shrinkage: alternate + and - steps)
    constexpr void taubin_smooth(f64 lambda=0.5, f64 mu=-0.53, int iters=10) noexcept {
        for(int iter=0;iter<iters;++iter) {
            laplacian_smooth(lambda,1);
            laplacian_smooth(mu,1);
        }
    }
};

// ─── Loop subdivision (one step) ─────────────────────────────────────────────
// Loop weights for even vertices: β = (1/n)(5/8 - (3/8+1/4*cos(2π/n))²)
[[nodiscard]] constexpr f64 loop_beta(int n) noexcept {
    if(n==3) return 3.0/16;
    f64 c=std::cos(2*3.14159265358979323846/n);
    f64 t=3.0/8+c/4; return (5.0/8-t*t)/n;
}

// ─── Mesh quality metrics ─────────────────────────────────────────────────────
// Aspect ratio of a triangle (circumradius/inradius * 2 = 1 for equilateral)
[[nodiscard]] constexpr f64 triangle_aspect_ratio(MeshV3 a, MeshV3 b, MeshV3 c) noexcept {
    f64 la=(b-a).len(),lb=(c-b).len(),lc=(a-c).len();
    f64 s=(la+lb+lc)*0.5;
    f64 area2=s*(s-la)*(s-lb)*(s-lc); if(area2<0) area2=0;
    f64 area=std::sqrt(area2);
    if(area<1e-12) return 1e9;
    f64 R=la*lb*lc/(4*area); // circumradius
    f64 r=area/s;             // inradius
    return R/r;
}
// Minimum angle of triangle
[[nodiscard]] constexpr f64 triangle_min_angle(MeshV3 a, MeshV3 b, MeshV3 c) noexcept {
    f64 la=(b-a).len(),lb=(c-b).len(),lc=(a-c).len();
    // Law of cosines
    auto safe_acos=[](f64 x){return std::acos(x<-1?-1:(x>1?1:x));};
    f64 A=safe_acos((lb*lb+lc*lc-la*la)/(2*lb*lc));
    f64 B=safe_acos((la*la+lc*lc-lb*lb)/(2*la*lc));
    f64 C=3.14159265358979323846-A-B;
    return std::min({A,B,C});
}

// ─── AABB tree node (for ray-mesh intersection) ───────────────────────────────
struct AABBNode {
    MeshV3 lo{1e9,1e9,1e9},hi{-1e9,-1e9,-1e9};
    int tri_idx{-1},left{-1},right{-1};

    constexpr void expand(MeshV3 p) noexcept {
        lo.x=lo.x<p.x?lo.x:p.x; lo.y=lo.y<p.y?lo.y:p.y; lo.z=lo.z<p.z?lo.z:p.z;
        hi.x=hi.x>p.x?hi.x:p.x; hi.y=hi.y>p.y?hi.y:p.y; hi.z=hi.z>p.z?hi.z:p.z;
    }
    [[nodiscard]] constexpr MeshV3 center() const noexcept {
        return{(lo.x+hi.x)*0.5,(lo.y+hi.y)*0.5,(lo.z+hi.z)*0.5};
    }
    [[nodiscard]] constexpr f64 surface_area() const noexcept {
        f64 dx=hi.x-lo.x,dy=hi.y-lo.y,dz=hi.z-lo.z;
        return 2*(dx*dy+dy*dz+dz*dx);
    }
};

// ─── Winding number for point-in-mesh test ────────────────────────────────────
template<int NV, int NT>
[[nodiscard]] constexpr f64 mesh_winding_number(const TriMesh<NV,NT>& mesh, MeshV3 q) noexcept {
    f64 omega=0;
    for(int t=0;t<mesh.nt;++t) {
        MeshV3 a=mesh.verts[mesh.tris[t][0]]-q;
        MeshV3 b=mesh.verts[mesh.tris[t][1]]-q;
        MeshV3 c=mesh.verts[mesh.tris[t][2]]-q;
        f64 la=a.len(),lb=b.len(),lc=c.len();
        if(la<1e-12||lb<1e-12||lc<1e-12) continue;
        // Solid angle via van Oosterom formula
        f64 num=a.dot(b.cross(c));
        f64 den=la*lb*lc+a.dot(b)*lc+b.dot(c)*la+c.dot(a)*lb;
        omega+=2*std::atan2(num,den);
    }
    return omega/(4*3.14159265358979323846);
}
} // namespace qs::math
