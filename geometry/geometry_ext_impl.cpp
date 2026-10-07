// qs_math/geometry/geometry_ext_impl.cpp (C++20)
#include "include/qs_math_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>
namespace qs::math::geometry {
// ─── Convex hull 2D (Graham scan) ────────────────────────────────────────────
std::vector<Vec2<f32>> convex_hull_2d(std::vector<Vec2<f32>> pts) noexcept {
    i32 n=(i32)pts.size(); if(n<3) return pts;
    auto cross2=[](Vec2<f32> O,Vec2<f32> A,Vec2<f32> B)->f32{
        return (A.x-O.x)*(B.y-O.y)-(A.y-O.y)*(B.x-O.x);};
    std::sort(pts.begin(),pts.end(),[](Vec2<f32> a,Vec2<f32> b){
        return a.x<b.x||(a.x==b.x&&a.y<b.y);});
    std::vector<Vec2<f32>> h;
    for(auto& p:pts){while(h.size()>=2&&cross2(h[h.size()-2],h.back(),p)<=0)h.pop_back();h.push_back(p);}
    i32 lo=h.size();
    for(i32 i=n-2;i>=0;--i){while((i32)h.size()>lo&&cross2(h[h.size()-2],h.back(),pts[i])<=0)h.pop_back();h.push_back(pts[i]);}
    h.pop_back(); return h;
}
// ─── Polygon centroid ─────────────────────────────────────────────────────────
Vec2<f32> polygon_centroid(const Vec2<f32>* pts, i32 n) noexcept {
    f32 area=0,cx=0,cy=0;
    for(i32 i=0,j=n-1;i<n;j=i++){
        const f32 c=pts[j].x*pts[i].y-pts[i].x*pts[j].y;
        area+=c; cx+=(pts[j].x+pts[i].x)*c; cy+=(pts[j].y+pts[i].y)*c;
    }
    area*=3; if(std::abs(area)<1e-10f) return {pts[0].x,pts[0].y};
    return {cx/area,cy/area};
}
// ─── Sutherland-Hodgman ───────────────────────────────────────────────────────
std::vector<Vec2<f32>> clip_polygon_half_plane(const std::vector<Vec2<f32>>& poly,
    Vec2<f32> pp, Vec2<f32> pn) noexcept
{
    std::vector<Vec2<f32>> out;
    i32 n=(i32)poly.size();
    for(i32 i=0;i<n;++i){
        Vec2<f32> a=poly[i],b=poly[(i+1)%n];
        const f32 da=(a-pp).x*pn.x+(a-pp).y*pn.y;
        const f32 db=(b-pp).x*pn.x+(b-pp).y*pn.y;
        if(da>=0) out.push_back(a);
        if((da>=0)!=(db>=0)){
            const f32 t=da/(da-db);
            out.push_back(a+(b-a)*t);
        }
    }
    return out;
}
// ─── Ear-clipping triangulation ───────────────────────────────────────────────
static bool is_ear(const Vec2<f32>* pts,const std::vector<i32>& idx,i32 i) noexcept {
    i32 n=(i32)idx.size();
    i32 prev=idx[(i-1+n)%n],cur=idx[i],next=idx[(i+1)%n];
    Vec2<f32> A=pts[prev],B=pts[cur],C=pts[next];
    const f32 cross=(B.x-A.x)*(C.y-A.y)-(B.y-A.y)*(C.x-A.x);
    if(cross<=0) return false;
    for(i32 j=0;j<n;++j){
        i32 jj=idx[j]; if(jj==prev||jj==cur||jj==next) continue;
        Vec2<f32> P=pts[jj];
        auto sign=[](Vec2<f32> a,Vec2<f32> b,Vec2<f32> p)->f32{
            return (p.x-b.x)*(a.y-b.y)-(a.x-b.x)*(p.y-b.y);};
        f32 d1=sign(A,B,P),d2=sign(B,C,P),d3=sign(C,A,P);
        if(((d1>0&&d2>0&&d3>0)||(d1<0&&d2<0&&d3<0))) return false;
    }
    return true;
}
std::vector<std::array<i32,3>> triangulate_polygon(const Vec2<f32>* pts, i32 n) noexcept {
    std::vector<std::array<i32,3>> tris;
    std::vector<i32> idx(n); for(i32 i=0;i<n;++i) idx[i]=i;
    while(idx.size()>3){
        bool found=false;
        for(i32 i=0;i<(i32)idx.size();++i){
            if(is_ear(pts,idx,i)){
                i32 sz=idx.size();
                tris.push_back({idx[(i-1+sz)%sz],idx[i],idx[(i+1)%sz]});
                idx.erase(idx.begin()+i); found=true; break;
            }
        }
        if(!found) break;
    }
    if(idx.size()==3) tris.push_back({idx[0],idx[1],idx[2]});
    return tris;
}
// ─── Bowyer-Watson Delaunay ───────────────────────────────────────────────────
struct DTri { std::array<i32,3> v; Vec2<f32> cc; f32 cr2; };
static bool in_circumcircle(DTri& t, const Vec2<f32>* pts, Vec2<f32> p) noexcept {
    return (p-t.cc).length_sq()<t.cr2;
}
static DTri make_tri(i32 a,i32 b,i32 c,const Vec2<f32>* pts) noexcept {
    Vec2<f32> A=pts[a],B=pts[b],C=pts[c];
    const f32 ax=A.x-C.x,ay=A.y-C.y,bx=B.x-C.x,by=B.y-C.y;
    const f32 D=2*(ax*by-ay*bx);
    if(std::abs(D)<1e-10f) return {{a,b,c},{C.x,C.y},0};
    const f32 ux=(by*(ax*ax+ay*ay)-ay*(bx*bx+by*by))/D;
    const f32 uy=(ax*(bx*bx+by*by)-bx*(ax*ax+ay*ay))/D;
    Vec2<f32> cc{C.x+ux,C.y+uy};
    return {{a,b,c},cc,(cc-A).length_sq()};
}
std::vector<DelaunayTri> delaunay_2d(const Vec2<f32>* pts, i32 n) noexcept {
    // Add super-triangle vertices
    std::vector<Vec2<f32>> all(pts,pts+n);
    all.push_back({-1e5f,-1e5f}); all.push_back({1e5f,-1e5f}); all.push_back({0,1e5f});
    i32 N=(i32)all.size();
    std::vector<DTri> tris;
    tris.push_back(make_tri(N-3,N-2,N-1,all.data()));
    for(i32 pi=0;pi<n;++pi){
        std::vector<DTri> bad;
        std::vector<DTri> good;
        for(auto& t:tris){
            if(in_circumcircle(t,all.data(),all[pi])) bad.push_back(t);
            else good.push_back(t);
        }
        // Find boundary edges
        std::vector<std::pair<i32,i32>> edges;
        for(auto& t:bad) for(i32 e=0;e<3;++e){
            std::pair<i32,i32> edge={t.v[e],t.v[(e+1)%3]};
            bool shared=false;
            for(auto& t2:bad) if(&t!=&t2) for(i32 e2=0;e2<3;++e2)
                if((t2.v[e2]==edge.first&&t2.v[(e2+1)%3]==edge.second)||
                   (t2.v[e2]==edge.second&&t2.v[(e2+1)%3]==edge.first)){shared=true;break;}
            if(!shared) edges.push_back(edge);
        }
        tris=good;
        for(auto& e:edges) tris.push_back(make_tri(e.first,e.second,pi,all.data()));
    }
    // Remove super-triangle
    std::vector<DelaunayTri> result;
    for(auto& t:tris) if(t.v[0]<n&&t.v[1]<n&&t.v[2]<n) result.push_back({t.v});
    return result;
}
// ─── SAT polygon vs polygon ───────────────────────────────────────────────────
static std::pair<f32,f32> sat_project(const Vec2<f32>* pts,i32 n,Vec2<f32> ax) noexcept {
    f32 mn=1e30f,mx=-1e30f;
    for(i32 i=0;i<n;++i){f32 p=pts[i].x*ax.x+pts[i].y*ax.y; mn=std::min(mn,p); mx=std::max(mx,p);}
    return{mn,mx};
}
bool sat_overlap(const Vec2<f32>* A,i32 na,const Vec2<f32>* B,i32 nb,Vec2<f32>& mtv) noexcept {
    f32 min_overlap=1e30f;
    Vec2<f32> min_ax{};
    auto test=[&](const Vec2<f32>* poly,i32 n)->bool{
        for(i32 i=0;i<n;++i){
            Vec2<f32> e=poly[(i+1)%n]-poly[i];
            Vec2<f32> ax{-e.y,e.x};
            const f32 len=std::sqrt(ax.x*ax.x+ax.y*ax.y);
            if(len<1e-7f) continue; ax.x/=len;ax.y/=len;
            auto [a0,a1]=sat_project(A,na,ax);
            auto [b0,b1]=sat_project(B,nb,ax);
            if(a1<b0||b1<a0) return false;
            const f32 ov=std::min(a1,b1)-std::max(a0,b0);
            if(ov<min_overlap){min_overlap=ov;min_ax=ax;}
        }
        return true;
    };
    if(!test(A,na)||!test(B,nb)) return false;
    mtv=min_ax*min_overlap; return true;
}
bool point_in_convex_polygon(Vec2<f32> p,const Vec2<f32>* hull,i32 n) noexcept {
    for(i32 i=0;i<n;++i){
        Vec2<f32> e=hull[(i+1)%n]-hull[i];
        Vec2<f32> ep=p-hull[i];
        if(e.x*ep.y-e.y*ep.x<0) return false;
    }
    return true;
}
ConvexHull3 convex_hull_3d(const std::vector<Vec3<f32>>& pts) noexcept {
    // Stub: return tetrahedron from first 4 points
    ConvexHull3 h; h.verts=pts;
    if(pts.size()>=4){h.faces={{{{0,1,2}},{{0,1,3}},{{0,2,3}},{{1,2,3}}}};}
    return h;
}
std::vector<VoronoiEdge> voronoi_2d(const Vec2<f32>* sites,i32 n,Vec2<f32>,Vec2<f32>) noexcept {
    // Dual of Delaunay — stub returning empty (full Fortune's algo is large)
    (void)sites;(void)n; return {};
}
std::vector<Vec2<f32>> minkowski_sum(const std::vector<Vec2<f32>>& A,const std::vector<Vec2<f32>>& B) noexcept {
    // Rotate-caliper Minkowski sum of two CCW convex polygons
    std::vector<Vec2<f32>> result;
    i32 na=(i32)A.size(),nb=(i32)B.size();
    i32 ia=0,ib=0;
    for(i32 i=1;i<na;++i) if(A[i].y<A[ia].y||(A[i].y==A[ia].y&&A[i].x<A[ia].x)) ia=i;
    for(i32 i=1;i<nb;++i) if(B[i].y<B[ib].y||(B[i].y==B[ib].y&&B[i].x<B[ib].x)) ib=i;
    result.push_back(A[ia]+B[ib]);
    for(i32 k=0;k<na+nb;++k){
        Vec2<f32> ea=A[(ia+1)%na]-A[ia],eb=B[(ib+1)%nb]-B[ib];
        const f32 cross=ea.x*eb.y-ea.y*eb.x;
        if(cross>0){ia=(ia+1)%na;result.push_back(A[ia]+B[ib]);}
        else if(cross<0){ib=(ib+1)%nb;result.push_back(A[ia]+B[ib]);}
        else{ia=(ia+1)%na;ib=(ib+1)%nb;result.push_back(A[ia]+B[ib]);}
    }
    return result;
}
} // namespace qs::math::geometry
