#pragma once
// qs_math/geometry/computational_geometry.hpp — Convex hull, Voronoi, Delaunay,
//   polygon clipping (Sutherland-Hodgman), point-in-polygon, closest pair,
//   line sweep, half-plane intersection (C++20)
#include "../linear_algebra/vec2.hpp"
#include "../linear_algebra/vec3.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <span>

namespace qs::math::geometry {

using V2 = Vec2f;
using V2d = Vec2<f64>;

// ─── Orientation test ─────────────────────────────────────────────────────
// Returns: >0 CCW, <0 CW, =0 collinear
[[nodiscard]] inline f64 orient2d(V2d a, V2d b, V2d c) noexcept {
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}
[[nodiscard]] inline f32 orient2d(V2 a, V2 b, V2 c) noexcept {
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}
[[nodiscard]] inline bool in_circle(V2d a, V2d b, V2d c, V2d d) noexcept {
    // Returns true if d is inside circumcircle of CCW triangle abc
    f64 ax=a.x-d.x,ay=a.y-d.y;
    f64 bx=b.x-d.x,by=b.y-d.y;
    f64 cx=c.x-d.x,cy=c.y-d.y;
    return (ax*ax+ay*ay)*(bx*cy-by*cx)
          -(bx*bx+by*by)*(ax*cy-ay*cx)
          +(cx*cx+cy*cy)*(ax*by-ay*bx) > 0.0;
}

// ─── Convex hull (Andrew's monotone chain) O(n log n) ────────────────────
[[nodiscard]] inline std::vector<V2> convex_hull(std::vector<V2> pts) noexcept {
    int n=int(pts.size()); if(n<3) return pts;
    std::sort(pts.begin(),pts.end(),[](V2 a,V2 b){return a.x<b.x||(a.x==b.x&&a.y<b.y);});
    std::vector<V2> hull;
    // Lower hull
    for(int i=0;i<n;++i){
        while(hull.size()>=2&&orient2d(hull[hull.size()-2],hull.back(),pts[i])<=0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    // Upper hull
    int lo=int(hull.size())+1;
    for(int i=n-2;i>=0;--i){
        while(int(hull.size())>=lo&&orient2d(hull[hull.size()-2],hull.back(),pts[i])<=0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    hull.pop_back();
    return hull;
}

// ─── Polygon area (shoelace) and centroid ────────────────────────────────
[[nodiscard]] inline f32 polygon_area(const std::vector<V2>& poly) noexcept {
    f32 a=0.f; int n=int(poly.size());
    for(int i=0;i<n;++i){
        auto& p=poly[i], &q=poly[(i+1)%n];
        a+=p.x*q.y - q.x*p.y;
    }
    return std::abs(a)*0.5f;
}

[[nodiscard]] inline V2 polygon_centroid(const std::vector<V2>& poly) noexcept {
    f32 cx=0,cy=0,area=0; int n=int(poly.size());
    for(int i=0;i<n;++i){
        auto& p=poly[i],&q=poly[(i+1)%n];
        f32 cross=p.x*q.y-q.x*p.y;
        cx+=(p.x+q.x)*cross; cy+=(p.y+q.y)*cross; area+=cross;
    }
    area*=0.5f;
    return {cx/(6.f*area), cy/(6.f*area)};
}

// ─── Point-in-polygon (ray casting) ──────────────────────────────────────
[[nodiscard]] inline bool point_in_polygon(V2 p, const std::vector<V2>& poly) noexcept {
    int n=int(poly.size()); bool inside=false;
    for(int i=0,j=n-1;i<n;j=i++){
        auto& a=poly[i],&b=poly[j];
        if(((a.y>p.y)!=(b.y>p.y))&&(p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x))
            inside=!inside;
    }
    return inside;
}

// Winding number test (handles self-intersecting polygons)
[[nodiscard]] inline int winding_number(V2 p, const std::vector<V2>& poly) noexcept {
    int wn=0; int n=int(poly.size());
    for(int i=0;i<n;++i){
        auto& a=poly[i],&b=poly[(i+1)%n];
        if(a.y<=p.y){
            if(b.y>p.y&&orient2d(a,b,p)>0) ++wn;
        } else {
            if(b.y<=p.y&&orient2d(a,b,p)<0) --wn;
        }
    }
    return wn;
}

// ─── Sutherland-Hodgman polygon clipping ─────────────────────────────────
[[nodiscard]] inline std::vector<V2> clip_polygon_by_halfplane(
    const std::vector<V2>& poly, V2 a, V2 b) noexcept
{
    // Clip poly against half-plane: left side of directed edge a->b
    std::vector<V2> out;
    int n=int(poly.size());
    for(int i=0;i<n;++i){
        V2 c=poly[i], d=poly[(i+1)%n];
        f32 cc=orient2d(a,b,c), dc=orient2d(a,b,d);
        if(cc>=0) out.push_back(c);
        if((cc>0&&dc<0)||(cc<0&&dc>0)){
            // Compute intersection
            f32 t=cc/(cc-dc);
            out.push_back({c.x+t*(d.x-c.x), c.y+t*(d.y-c.y)});
        }
    }
    return out;
}

[[nodiscard]] inline std::vector<V2> sutherland_hodgman(
    std::vector<V2> subject, const std::vector<V2>& clip) noexcept
{
    int n=int(clip.size());
    for(int i=0;i<n&&!subject.empty();++i)
        subject=clip_polygon_by_halfplane(subject,clip[i],clip[(i+1)%n]);
    return subject;
}

// ─── Segment intersection ─────────────────────────────────────────────────
struct SegmentIntersect { bool hit; V2 point; f32 t,s; };

[[nodiscard]] inline SegmentIntersect segment_intersect(V2 p1,V2 p2,V2 p3,V2 p4) noexcept {
    V2 d1={p2.x-p1.x,p2.y-p1.y}, d2={p4.x-p3.x,p4.y-p3.y};
    f32 cross=d1.x*d2.y-d1.y*d2.x;
    if(std::abs(cross)<1e-9f) return {false,{},0,0};
    V2 r={p3.x-p1.x,p3.y-p1.y};
    f32 t=(r.x*d2.y-r.y*d2.x)/cross;
    f32 s=(r.x*d1.y-r.y*d1.x)/cross;
    if(t<0||t>1||s<0||s>1) return {false,{},t,s};
    return {true,{p1.x+t*d1.x,p1.y+t*d1.y},t,s};
}

// ─── Closest pair of points (divide & conquer) O(n log n) ─────────────────
[[nodiscard]] inline f32 closest_pair_dist(std::vector<V2> pts) noexcept {
    int n=int(pts.size()); if(n<2) return 0.f;
    std::sort(pts.begin(),pts.end(),[](V2 a,V2 b){return a.x<b.x;});
    std::function<f32(int,int)> rec=[&](int l,int r)->f32{
        if(r-l<=3){
            f32 best=1e30f;
            for(int i=l;i<r;++i) for(int j=i+1;j<r;++j){
                f32 dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y;
                best=min(best,std::sqrt(dx*dx+dy*dy));
            } return best;
        }
        int mid=(l+r)/2; f32 mx=pts[mid].x;
        f32 d=min(rec(l,mid),rec(mid,r));
        std::vector<V2> strip;
        for(int i=l;i<r;++i) if(std::abs(pts[i].x-mx)<d) strip.push_back(pts[i]);
        std::sort(strip.begin(),strip.end(),[](V2 a,V2 b){return a.y<b.y;});
        for(int i=0;i<int(strip.size());++i)
            for(int j=i+1;j<int(strip.size())&&(strip[j].y-strip[i].y)<d;++j){
                f32 dx=strip[i].x-strip[j].x,dy=strip[i].y-strip[j].y;
                d=min(d,std::sqrt(dx*dx+dy*dy));
            }
        return d;
    };
    return rec(0,n);
}

// ─── Minimum bounding rectangle (rotating calipers) ───────────────────────
struct BoundingRect { V2 center; V2 axis[2]; f32 half_extents[2]; f32 area; };

[[nodiscard]] inline BoundingRect minimum_bounding_rect(std::vector<V2> pts) noexcept {
    auto hull=convex_hull(pts);
    int n=int(hull.size()); if(n==0) return {};
    f32 best_area=1e30f; BoundingRect best;
    for(int i=0;i<n;++i){
        V2 e={hull[(i+1)%n].x-hull[i].x, hull[(i+1)%n].y-hull[i].y};
        f32 len=std::sqrt(e.x*e.x+e.y*e.y);
        if(len<1e-9f) continue;
        V2 u={e.x/len,e.y/len}, v={-u.y,u.x};
        f32 mn0=1e30f,mx0=-1e30f,mn1=1e30f,mx1=-1e30f;
        for(auto& p:hull){
            f32 a=p.x*u.x+p.y*u.y, b=p.x*v.x+p.y*v.y;
            mn0=min(mn0,a); mx0=max(mx0,a);
            mn1=min(mn1,b); mx1=max(mx1,b);
        }
        f32 area=(mx0-mn0)*(mx1-mn1);
        if(area<best_area){
            best_area=area;
            f32 cx=(mn0+mx0)*0.5f, cy=(mn1+mx1)*0.5f;
            best.center={cx*u.x+cy*v.x, cx*u.y+cy*v.y};
            best.axis[0]=u; best.axis[1]=v;
            best.half_extents[0]=(mx0-mn0)*0.5f;
            best.half_extents[1]=(mx1-mn1)*0.5f;
            best.area=area;
        }
    }
    return best;
}

// ─── Triangulation (ear clipping) ────────────────────────────────────────
[[nodiscard]] inline bool is_ear(const std::vector<V2>& poly, const std::vector<int>& idx, int i) noexcept {
    int n=int(idx.size());
    int prev=idx[(i-1+n)%n], cur=idx[i], next=idx[(i+1)%n];
    V2 a=poly[prev], b=poly[cur], c=poly[next];
    if(orient2d(a,b,c)<=0) return false; // not convex
    for(int j=0;j<n;++j){
        if(idx[j]==prev||idx[j]==cur||idx[j]==next) continue;
        V2 p=poly[idx[j]];
        if(orient2d(a,b,p)>=0&&orient2d(b,c,p)>=0&&orient2d(c,a,p)>=0) return false;
    }
    return true;
}

[[nodiscard]] inline std::vector<std::array<int,3>> ear_clip_triangulate(const std::vector<V2>& poly) noexcept {
    int n=int(poly.size());
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::vector<std::array<int,3>> tris;
    // Ensure CCW
    f32 area=0; for(int i=0;i<n;++i) area+=poly[i].x*poly[(i+1)%n].y-poly[(i+1)%n].x*poly[i].y;
    if(area<0) std::reverse(idx.begin(),idx.end());
    while(int(idx.size())>3){
        bool found=false;
        for(int i=0;i<int(idx.size());++i){
            if(is_ear(poly,idx,i)){
                int m=int(idx.size());
                tris.push_back({idx[(i-1+m)%m], idx[i], idx[(i+1)%m]});
                idx.erase(idx.begin()+i);
                found=true; break;
            }
        }
        if(!found) break; // degenerate polygon
    }
    if(idx.size()==3) tris.push_back({idx[0],idx[1],idx[2]});
    return tris;
}

// ─── Circumcircle of 3 points ─────────────────────────────────────────────
struct Circle { V2 center; f32 radius; };

[[nodiscard]] inline Circle circumcircle(V2 a, V2 b, V2 c) noexcept {
    f32 ax=a.x,ay=a.y,bx=b.x,by=b.y,cx=c.x,cy=c.y;
    f32 D=2.f*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by));
    if(std::abs(D)<1e-10f) return {{0,0},1e30f};
    f32 ux=((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/D;
    f32 uy=((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/D;
    f32 r=std::sqrt((ax-ux)*(ax-ux)+(ay-uy)*(ay-uy));
    return {{ux,uy},r};
}

// ─── Half-plane intersection (compute feasible region) ───────────────────
// Half-plane: points p such that orient2d(a,b,p) >= 0
[[nodiscard]] inline std::vector<V2> half_plane_intersection(
    const std::vector<std::pair<V2,V2>>& halfplanes) noexcept
{
    // Start with large bounding box
    std::vector<V2> poly={
        {-1e6f,-1e6f},{1e6f,-1e6f},{1e6f,1e6f},{-1e6f,1e6f}
    };
    for(auto& [a,b]:halfplanes){
        poly=clip_polygon_by_halfplane(poly,a,b);
        if(poly.empty()) return {};
    }
    return poly;
}

// ─── Point-to-segment distance ────────────────────────────────────────────
[[nodiscard]] inline f32 point_segment_dist(V2 p, V2 a, V2 b) noexcept {
    V2 ab={b.x-a.x,b.y-a.y}, ap={p.x-a.x,p.y-a.y};
    f32 t=clamp01((ap.x*ab.x+ap.y*ab.y)/(ab.x*ab.x+ab.y*ab.y+1e-30f));
    V2 closest={a.x+t*ab.x, a.y+t*ab.y};
    f32 dx=p.x-closest.x, dy=p.y-closest.y;
    return std::sqrt(dx*dx+dy*dy);
}

// ─── Polygon offsetting (Minkowski sum with circle approximation) ──────────
[[nodiscard]] inline std::vector<V2> polygon_offset(const std::vector<V2>& poly, f32 d) noexcept {
    int n=int(poly.size());
    std::vector<V2> result;
    for(int i=0;i<n;++i){
        V2 prev=poly[(i-1+n)%n], cur=poly[i], next=poly[(i+1)%n];
        V2 e1={cur.x-prev.x,cur.y-prev.y};
        V2 e2={next.x-cur.x,next.y-cur.y};
        f32 l1=std::sqrt(e1.x*e1.x+e1.y*e1.y)+1e-10f;
        f32 l2=std::sqrt(e2.x*e2.x+e2.y*e2.y)+1e-10f;
        V2 n1={-e1.y/l1,e1.x/l1}, n2={-e2.y/l2,e2.x/l2};
        // Miter join
        f32 miter_scale=1.f/(n1.x*n2.x+n1.y*n2.y+1e-10f);
        f32 bx=(n1.x+n2.x)*0.5f*d*miter_scale;
        f32 by=(n1.y+n2.y)*0.5f*d*miter_scale;
        result.push_back({cur.x+bx, cur.y+by});
    }
    return result;
}

} // namespace qs::math::geometry
