#pragma once
// qs_math/geometry/convex_hull.hpp — 2D/3D convex hull, GJK, EPA, Minkowski sum
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

struct Pt2 { f64 x,y;
    [[nodiscard]] constexpr Pt2 operator-(Pt2 b) const noexcept {return{x-b.x,y-b.y};}
    [[nodiscard]] constexpr Pt2 operator+(Pt2 b) const noexcept {return{x+b.x,y+b.y};}
    [[nodiscard]] constexpr f64 cross(Pt2 b) const noexcept {return x*b.y-y*b.x;}
    [[nodiscard]] constexpr f64 dot(Pt2 b)   const noexcept {return x*b.x+y*b.y;}
    [[nodiscard]] constexpr f64 len2()        const noexcept {return dot(*this);}
    [[nodiscard]]           f64 len()         const noexcept {return std::sqrt(len2());}
};
struct Pt3 { f64 x,y,z;
    [[nodiscard]] constexpr Pt3 operator-(Pt3 b) const noexcept {return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr Pt3 operator+(Pt3 b) const noexcept {return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr Pt3 operator*(f64 s) const noexcept {return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64 dot(Pt3 b)   const noexcept {return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr Pt3 cross(Pt3 b) const noexcept {return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    [[nodiscard]] constexpr f64 len2()       const noexcept {return dot(*this);}
    [[nodiscard]]           f64 len()        const noexcept {return std::sqrt(len2());}
    [[nodiscard]] Pt3 normalized() const noexcept {f64 l=len();return l>1e-12?Pt3{x/l,y/l,z/l}:*this;}
};

// ─── 2D Convex Hull (Graham scan, O(n log n)) ─────────────────────────────────
template<int N>
[[nodiscard]] constexpr int convex_hull_2d(const std::array<Pt2,N>& pts,
                                             std::array<int,N>& hull) noexcept {
    // Find bottom-most point
    int bot=0;
    for(int i=1;i<N;++i) if(pts[i].y<pts[bot].y||(pts[i].y==pts[bot].y&&pts[i].x<pts[bot].x)) bot=i;
    // Sort by polar angle w.r.t. pts[bot]
    std::array<int,N> idx{};
    for(int i=0;i<N;++i) idx[i]=i;
    std::swap(idx[0],idx[bot]);
    Pt2 pivot=pts[idx[0]];
    // Insertion sort by angle
    for(int i=2;i<N;++i) {
        int j=i;
        while(j>1) {
            Pt2 a=pts[idx[j-1]]-pivot,b=pts[idx[j]]-pivot;
            f64 cr=a.cross(b);
            if(cr<0||(std::abs(cr)<1e-12&&a.len2()>b.len2())) {std::swap(idx[j-1],idx[j]);--j;}
            else break;
        }
    }
    // Graham scan
    int sz=0;
    for(int i=0;i<N;++i) {
        while(sz>=2) {
            Pt2 a=pts[hull[sz-1]]-pts[hull[sz-2]];
            Pt2 b=pts[idx[i]]-pts[hull[sz-2]];
            if(a.cross(b)<=0) --sz; else break;
        }
        hull[sz++]=idx[i];
    }
    return sz;
}

// ─── 2D Convex hull area ──────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 hull_area(const std::array<Pt2,N>& pts,
                                       const std::array<int,N>& hull, int sz) noexcept {
    f64 area=0;
    for(int i=0;i<sz;++i) {
        int j=(i+1)%sz;
        area+=pts[hull[i]].x*pts[hull[j]].y-pts[hull[j]].x*pts[hull[i]].y;
    }
    return std::abs(area)*0.5;
}

// ─── 3D convex hull face normal (for triangular face) ─────────────────────────
[[nodiscard]] constexpr Pt3 tri_normal(Pt3 a, Pt3 b, Pt3 c) noexcept {
    return (b-a).cross(c-a);
}
[[nodiscard]] constexpr bool above_plane(Pt3 p, Pt3 a, Pt3 b, Pt3 c) noexcept {
    return tri_normal(a,b,c).dot(p-a)>0;
}

// ─── GJK (Gilbert-Johnson-Keerthi) collision detection ────────────────────────
// Support function: furthest point in direction d
template<int N>
[[nodiscard]] constexpr Pt3 support(const std::array<Pt3,N>& shape, Pt3 d) noexcept {
    f64 best=shape[0].dot(d); int idx=0;
    for(int i=1;i<N;++i){f64 v=shape[i].dot(d);if(v>best){best=v;idx=i;}}
    return shape[idx];
}

// Minkowski difference support
template<int NA, int NB>
[[nodiscard]] constexpr Pt3 mink_support(const std::array<Pt3,NA>& A,
                                           const std::array<Pt3,NB>& B, Pt3 d) noexcept {
    Pt3 nd={-d.x,-d.y,-d.z};
    Pt3 sa=support(A,d), sb=support(B,nd);
    return sa-sb;
}

struct GjkSimplex {
    Pt3 pts[4]{}; int n{0};
    constexpr void push(Pt3 p) noexcept {for(int i=n;i>0;--i)pts[i]=pts[i-1];pts[0]=p;if(n<4)++n;}
};

// Closest point on line segment to origin
[[nodiscard]] constexpr Pt3 closest_on_segment(Pt3 a, Pt3 b) noexcept {
    Pt3 ab=b-a; f64 t=-a.dot(ab)/(ab.dot(ab)+1e-30);
    t=t<0?0:(t>1?1:t);
    return a+ab*t;
}

// GJK iteration (simplified, 2D simplex only for robustness in header)
template<int NA, int NB>
[[nodiscard]] constexpr bool gjk_intersects(const std::array<Pt3,NA>& A,
                                              const std::array<Pt3,NB>& B) noexcept {
    Pt3 d={1,0,0};
    GjkSimplex s;
    s.push(mink_support(A,B,d));
    d={-s.pts[0].x,-s.pts[0].y,-s.pts[0].z};
    for(int iter=0;iter<64;++iter) {
        Pt3 a=mink_support(A,B,d);
        if(a.dot(d)<0) return false;
        s.push(a);
        // Update simplex (line case)
        if(s.n==2) {
            Pt3 ab=s.pts[0]-s.pts[1], ao={-s.pts[1].x,-s.pts[1].y,-s.pts[1].z};
            if(ab.dot(ao)>0) d=ab.cross(ao).cross(ab);
            else {s.pts[0]=s.pts[1];s.n=1;d=ao;}
        } else {
            // Triangle/tetrahedron case — origin test
            Pt3 ab=s.pts[1]-s.pts[0],ac=s.pts[2]-s.pts[0];
            Pt3 ao={-s.pts[0].x,-s.pts[0].y,-s.pts[0].z};
            Pt3 abc=ab.cross(ac);
            if(abc.cross(ac).dot(ao)>0){s.pts[1]=s.pts[2];s.n=2;d=ac.cross(ao).cross(ac);}
            else if(ab.cross(abc).dot(ao)>0){s.n=2;d=ab.cross(ao).cross(ab);}
            else if(abc.dot(ao)>0) d=abc;
            else {std::swap(s.pts[1],s.pts[2]);d={-abc.x,-abc.y,-abc.z};}
        }
        if(d.len2()<1e-20) return true;
    }
    return true;
}

// ─── Minkowski sum of two convex polygons (2D, simple merge) ─────────────────
template<int NA, int NB, int NC>
[[nodiscard]] constexpr int minkowski_sum_2d(const std::array<Pt2,NA>& A,
                                               const std::array<Pt2,NB>& B,
                                               std::array<Pt2,NC>& C) noexcept {
    // Both polygons assumed CCW; output is CCW Minkowski sum
    int ia=0,ib=0,sz=0;
    // Find bottom-most points
    for(int i=1;i<NA;++i) if(A[i].y<A[ia].y||(A[i].y==A[ia].y&&A[i].x<A[ia].x)) ia=i;
    for(int i=1;i<NB;++i) if(B[i].y<B[ib].y||(B[i].y==B[ib].y&&B[i].x<B[ib].x)) ib=i;
    for(int step=0;step<NA+NB&&sz<NC;++step) {
        C[sz++]={A[ia].x+B[ib].x,A[ia].y+B[ib].y};
        Pt2 ea=A[(ia+1)%NA]-A[ia], eb=B[(ib+1)%NB]-B[ib];
        f64 cr=ea.cross(eb);
        if(cr>0) ia=(ia+1)%NA;
        else if(cr<0) ib=(ib+1)%NB;
        else {ia=(ia+1)%NA;ib=(ib+1)%NB;}
    }
    return sz;
}

// ─── Separating Axis Theorem (SAT) for OBBs ──────────────────────────────────
struct OBB { Pt3 center,ax,ay,az; f64 ex,ey,ez; };

[[nodiscard]] constexpr f64 obb_project(const OBB& box, Pt3 axis) noexcept {
    return std::abs(box.ax.dot(axis))*box.ex
          +std::abs(box.ay.dot(axis))*box.ey
          +std::abs(box.az.dot(axis))*box.ez;
}

[[nodiscard]] constexpr bool obb_overlap_on_axis(const OBB& A, const OBB& B, Pt3 axis) noexcept {
    if(axis.len2()<1e-20) return true;
    f64 d=std::abs((B.center-A.center).dot(axis));
    return d<obb_project(A,axis)+obb_project(B,axis);
}

[[nodiscard]] constexpr bool obb_intersects(const OBB& A, const OBB& B) noexcept {
    std::array<Pt3,15> axes={
        A.ax,A.ay,A.az,B.ax,B.ay,B.az,
        A.ax.cross(B.ax),A.ax.cross(B.ay),A.ax.cross(B.az),
        A.ay.cross(B.ax),A.ay.cross(B.ay),A.ay.cross(B.az),
        A.az.cross(B.ax),A.az.cross(B.ay),A.az.cross(B.az)
    };
    for(auto& ax:axes) if(!obb_overlap_on_axis(A,B,ax)) return false;
    return true;
}
} // namespace qs::math
