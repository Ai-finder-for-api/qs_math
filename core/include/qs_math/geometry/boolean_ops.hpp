#pragma once
// qs_math/geometry/boolean_ops.hpp — Polygon clipping, polygon boolean, sweep line
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>
#include <optional>

namespace qs::math {

// ─── 2D point ────────────────────────────────────────────────────────────────
struct BP2 { f64 x,y;
    [[nodiscard]] constexpr BP2 operator+(BP2 b) const noexcept {return{x+b.x,y+b.y};}
    [[nodiscard]] constexpr BP2 operator-(BP2 b) const noexcept {return{x-b.x,y-b.y};}
    [[nodiscard]] constexpr BP2 operator*(f64 s) const noexcept {return{x*s,y*s};}
    [[nodiscard]] constexpr f64  cross(BP2 b) const noexcept {return x*b.y-y*b.x;}
    [[nodiscard]] constexpr f64  dot(BP2 b)   const noexcept {return x*b.x+y*b.y;}
};

// ─── Sutherland-Hodgman polygon clipping ─────────────────────────────────────
// Clip polygon against half-plane: all points with (p-a)×(b-a) ≥ 0
template<int NIN, int NOUT>
[[nodiscard]] constexpr int clip_polygon_halfplane(
    const std::array<BP2,NIN>& poly, int n_in,
    std::array<BP2,NOUT>& out,
    BP2 a, BP2 b) noexcept
{
    int n_out=0;
    BP2 edge=b-a;
    auto inside=[&](BP2 p){return (p-a).cross(edge)>=0;};
    auto intersect=[&](BP2 p1, BP2 p2) -> BP2 {
        BP2 d=p2-p1, e=edge;
        f64 t=((a-p1).cross(e))/(d.cross(e)+1e-15);
        return p1+d*t;
    };
    for(int i=0;i<n_in&&n_out<NOUT;++i) {
        BP2 cur=poly[i],next=poly[(i+1)%n_in];
        bool ci=inside(cur),ni=inside(next);
        if(ci)     out[n_out++]=cur;
        if(ci!=ni) out[n_out++]=intersect(cur,next);
    }
    return n_out;
}

// Clip convex polygon against convex clip polygon (Sutherland-Hodgman)
template<int NP, int NC, int NOUT=NP+NC>
[[nodiscard]] constexpr int sutherland_hodgman(
    const std::array<BP2,NP>& subject, int ns,
    const std::array<BP2,NC>& clip,     int nc,
    std::array<BP2,NOUT>& out) noexcept
{
    std::array<BP2,NOUT> buf1{},buf2{};
    for(int i=0;i<ns;++i) buf1[i]=subject[i];
    int n=ns;
    for(int i=0;i<nc&&n>0;++i) {
        int m=(i+1)%nc;
        n=clip_polygon_halfplane<NOUT,NOUT>(buf1,n,buf2,clip[i],clip[m]);
        buf1=buf2;
    }
    for(int i=0;i<n;++i) out[i]=buf1[i];
    return n;
}

// ─── Line segment intersection ────────────────────────────────────────────────
[[nodiscard]] constexpr std::optional<BP2> segment_intersect(
    BP2 p1, BP2 p2, BP2 q1, BP2 q2) noexcept
{
    BP2 d=p2-p1, e=q2-q1;
    f64 denom=d.cross(e);
    if(std::abs(denom)<1e-12) return std::nullopt;
    f64 t=(q1-p1).cross(e)/denom;
    f64 u=(q1-p1).cross(d)/denom;
    if(t<0||t>1||u<0||u>1) return std::nullopt;
    return p1+d*t;
}

// ─── Point in polygon (ray casting) ──────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr bool point_in_polygon(
    const std::array<BP2,N>& poly, int n, BP2 p) noexcept
{
    bool inside=false;
    for(int i=0,j=n-1;i<n;j=i++) {
        if(((poly[i].y>p.y)!=(poly[j].y>p.y))&&
           (p.x<(poly[j].x-poly[i].x)*(p.y-poly[i].y)/(poly[j].y-poly[i].y)+poly[i].x))
            inside=!inside;
    }
    return inside;
}

// ─── Polygon area (shoelace) ──────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr f64 polygon_area(
    const std::array<BP2,N>& poly, int n) noexcept
{
    f64 area=0;
    for(int i=0,j=n-1;i<n;j=i++) area+=poly[j].cross(poly[i]);
    return std::abs(area)*0.5;
}

// ─── Polygon centroid ────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr BP2 polygon_centroid(
    const std::array<BP2,N>& poly, int n) noexcept
{
    BP2 c{}; f64 area=0;
    for(int i=0,j=n-1;i<n;j=i++) {
        f64 cross=poly[j].cross(poly[i]);
        area+=cross;
        c.x+=(poly[j].x+poly[i].x)*cross;
        c.y+=(poly[j].y+poly[i].y)*cross;
    }
    area*=0.5; f64 inv=1.0/(6*area+1e-15);
    return{c.x*inv,c.y*inv};
}

// ─── Minkowski sum boundary (convex polygons) ─────────────────────────────────
// Merge sorted edge vectors
template<int NA, int NB>
[[nodiscard]] constexpr int minkowski_boundary(
    const std::array<BP2,NA>& A, int na,
    const std::array<BP2,NB>& B, int nb,
    std::array<BP2,NA+NB>& result) noexcept
{
    // Find bottom-most vertices
    int ia=0,ib=0;
    for(int i=1;i<na;++i) if(A[i].y<A[ia].y||(A[i].y==A[ia].y&&A[i].x<A[ia].x))ia=i;
    for(int i=1;i<nb;++i) if(B[i].y<B[ib].y||(B[i].y==B[ib].y&&B[i].x<B[ib].x))ib=i;
    int n=0;
    result[n++]={A[ia].x+B[ib].x,A[ia].y+B[ib].y};
    for(int step=0;step<na+nb&&n<na+nb;++step) {
        BP2 ea=A[(ia+1)%na]-A[ia], eb=B[(ib+1)%nb]-B[ib];
        f64 cr=ea.cross(eb);
        BP2 edge;
        if(cr>0){edge=ea;ia=(ia+1)%na;}
        else if(cr<0){edge=eb;ib=(ib+1)%nb;}
        else{edge=ea+eb;ia=(ia+1)%na;ib=(ib+1)%nb;}
        result[n]={result[n-1].x+edge.x,result[n-1].y+edge.y};
        ++n;
    }
    return n;
}

// ─── Polygon offset (inward/outward) ─────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<BP2,N> polygon_offset(
    const std::array<BP2,N>& poly, int n, f64 delta) noexcept
{
    std::array<BP2,N> result{};
    for(int i=0;i<n;++i) {
        int prev=(i-1+n)%n, next=(i+1)%n;
        BP2 d1=poly[i]-poly[prev],d2=poly[next]-poly[i];
        f64 l1=std::sqrt(d1.dot(d1)),l2=std::sqrt(d2.dot(d2));
        if(l1<1e-12||l2<1e-12){result[i]=poly[i];continue;}
        BP2 n1={-d1.y/l1,d1.x/l1},n2={-d2.y/l2,d2.x/l2};
        // Bisector
        BP2 bis={(n1.x+n2.x),(n1.y+n2.y)};
        f64 blen=std::sqrt(bis.dot(bis));
        if(blen<1e-12){bis=n1;}else{bis.x/=blen;bis.y/=blen;}
        f64 cos_half=bis.dot(n1); if(std::abs(cos_half)<1e-6)cos_half=1e-6;
        f64 miter=delta/cos_half;
        result[i]={poly[i].x+bis.x*miter,poly[i].y+bis.y*miter};
    }
    return result;
}

// ─── Rectangle/AABB overlap ──────────────────────────────────────────────────
struct AABB2 { f64 min_x,min_y,max_x,max_y; };
[[nodiscard]] constexpr bool aabb_overlap(AABB2 a, AABB2 b) noexcept {
    return a.min_x<b.max_x&&a.max_x>b.min_x&&a.min_y<b.max_y&&a.max_y>b.min_y;
}
[[nodiscard]] constexpr AABB2 aabb_union(AABB2 a, AABB2 b) noexcept {
    return{std::min(a.min_x,b.min_x),std::min(a.min_y,b.min_y),
           std::max(a.max_x,b.max_x),std::max(a.max_y,b.max_y)};
}
[[nodiscard]] constexpr AABB2 aabb_intersect(AABB2 a, AABB2 b) noexcept {
    return{std::max(a.min_x,b.min_x),std::max(a.min_y,b.min_y),
           std::min(a.max_x,b.max_x),std::min(a.max_y,b.max_y)};
}
[[nodiscard]] constexpr f64 aabb_area(AABB2 a) noexcept {
    return std::max(0.0,a.max_x-a.min_x)*std::max(0.0,a.max_y-a.min_y);
}
} // namespace qs::math
