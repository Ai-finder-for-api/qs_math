
// qs_math/uv/uv_impl.cpp (C++20)
#include "include/qs_math_uv.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
namespace qs::math::uv {
constexpr f32 PI = 3.14159265358979323846f;

Vec2<f32> project_planar_xy(Vec3<f32> p) noexcept { return{p.x,p.y}; }
Vec2<f32> project_planar_xz(Vec3<f32> p) noexcept { return{p.x,p.z}; }
Vec2<f32> project_cylindrical(Vec3<f32> p, Vec3<f32> axis) noexcept {
    // Project onto plane perp to axis, then compute angle
    const Vec3<f32> ax=axis.normalized();
    const f32 ht=p.dot(ax);
    const Vec3<f32> r=p-ax*ht;
    const f32 phi=std::atan2(r.y,r.x);
    return{(phi+PI)/(2*PI), ht};
}
Vec2<f32> project_spherical(Vec3<f32> dir) noexcept {
    const Vec3<f32> d=dir.normalized();
    return{(std::atan2(d.z,d.x)+PI)/(2*PI),
           std::acos(std::max(-1.0f,std::min(1.0f,d.y)))/PI};
}
Vec2<f32> project_cube(Vec3<f32> dir, i32& face) noexcept {
    const f32 ax=std::abs(dir.x),ay=std::abs(dir.y),az=std::abs(dir.z);
    Vec2<f32> uv;
    if(ax>=ay&&ax>=az){
        face=dir.x>0?0:1;
        uv={dir.x>0?-dir.z/ax:dir.z/ax, dir.y/ax};
    }else if(ay>=az){
        face=dir.y>0?2:3;
        uv={dir.x/ay, dir.y>0?-dir.z/ay:dir.z/ay};
    }else{
        face=dir.z>0?4:5;
        uv={dir.z>0?dir.x/az:-dir.x/az, dir.y/az};
    }
    return uv*0.5f+Vec2<f32>{0.5f,0.5f};
}
Vec3<f32> uv_to_bary(Vec2<f32> uv,Vec2<f32> u0,Vec2<f32> u1,Vec2<f32> u2) noexcept {
    const Vec2<f32> v0=u1-u0,v1=u2-u0,v2=uv-u0;
    const f32 d00=v0.x*v0.x+v0.y*v0.y,d01=v0.x*v1.x+v0.y*v1.y;
    const f32 d11=v1.x*v1.x+v1.y*v1.y,d20=v2.x*v0.x+v2.y*v0.y,d21=v2.x*v1.x+v2.y*v1.y;
    const f32 denom=d00*d11-d01*d01;
    if(std::abs(denom)<1e-10f)return{1.0f/3,1.0f/3,1.0f/3};
    const f32 v=( d11*d20-d01*d21)/denom;
    const f32 w=( d00*d21-d01*d20)/denom;
    return{1-v-w,v,w};
}
Vec2<f32> bary_to_uv(Vec3<f32> b,Vec2<f32> u0,Vec2<f32> u1,Vec2<f32> u2) noexcept {
    return u0*b.x+u1*b.y+u2*b.z;
}
f32 uv_area_distortion(Vec3<f32> p0,Vec3<f32> p1,Vec3<f32> p2,
    Vec2<f32> uv0,Vec2<f32> uv1,Vec2<f32> uv2) noexcept
{
    const f32 a3d=(p1-p0).cross(p2-p0).length()*0.5f;
    const Vec2<f32> e1=uv1-uv0,e2=uv2-uv0;
    const f32 a2d=std::abs(e1.x*e2.y-e1.y*e2.x)*0.5f;
    if(a3d<1e-10f||a2d<1e-10f)return 0;
    const f32 r=a2d/a3d;
    return r>1?r:1.0f/r;
}
f32 uv_angle_distortion(Vec3<f32> p0,Vec3<f32> p1,Vec3<f32> p2,
    Vec2<f32> uv0,Vec2<f32> uv1,Vec2<f32> uv2) noexcept
{
    // Compare angles in 3D vs 2D at each vertex, return max difference
    auto angle3=[](Vec3<f32> a,Vec3<f32> b,Vec3<f32> c)->f32{
        Vec3<f32> u=b-a,v=c-a; f32 d=u.dot(v)/(u.length()*v.length()+1e-10f);
        return std::acos(std::max(-1.0f,std::min(1.0f,d)));};
    auto angle2=[](Vec2<f32> a,Vec2<f32> b,Vec2<f32> c)->f32{
        Vec2<f32> u=b-a,v=c-a;
        f32 d=(u.x*v.x+u.y*v.y)/(std::sqrt(u.x*u.x+u.y*u.y)*std::sqrt(v.x*v.x+v.y*v.y)+1e-10f);
        return std::acos(std::max(-1.0f,std::min(1.0f,d)));};
    const f32 d0=std::abs(angle3(p0,p1,p2)-angle2(uv0,uv1,uv2));
    const f32 d1=std::abs(angle3(p1,p2,p0)-angle2(uv1,uv2,uv0));
    const f32 d2=std::abs(angle3(p2,p0,p1)-angle2(uv2,uv0,uv1));
    return std::max({d0,d1,d2});
}
UVChart lscm_unwrap(const Vec3<f32>* verts,i32 nv,
    const i32* faces,i32 nf) noexcept
{
    UVChart chart;
    chart.uvs.resize(nv,{0,0});
    chart.vertex_ids.resize(nv);
    std::iota(chart.vertex_ids.begin(),chart.vertex_ids.end(),0);
    // Pin two vertices (first and last on boundary, simplified)
    chart.uvs[0]={0,0}; chart.uvs[nv>1?1:0]={1,0};
    // Compute initial planar projection as approximation
    // (Full LSCM requires solving a sparse linear system)
    Vec3<f32> avg_n{};
    for(i32 t=0;t<nf;++t){
        const Vec3<f32> e1=verts[faces[t*3+1]]-verts[faces[t*3]];
        const Vec3<f32> e2=verts[faces[t*3+2]]-verts[faces[t*3]];
        avg_n=avg_n+e1.cross(e2);
    }
    avg_n=avg_n.normalized();
    // Build local tangent frame
    Vec3<f32> up=std::abs(avg_n.z)<0.9f?Vec3<f32>{0,0,1}:Vec3<f32>{1,0,0};
    Vec3<f32> T=(up.cross(avg_n)).normalized(), B=avg_n.cross(T);
    // Project all verts onto plane
    Vec2<f32> mn{1e9f,1e9f},mx{-1e9f,-1e9f};
    for(i32 i=0;i<nv;++i){
        chart.uvs[i]={verts[i].dot(T),verts[i].dot(B)};
        mn.x=std::min(mn.x,chart.uvs[i].x); mn.y=std::min(mn.y,chart.uvs[i].y);
        mx.x=std::max(mx.x,chart.uvs[i].x); mx.y=std::max(mx.y,chart.uvs[i].y);
    }
    const f32 rng=std::max(mx.x-mn.x,mx.y-mn.y)+1e-7f;
    for(auto&uv:chart.uvs) uv=(uv-mn)*(1.0f/rng);
    // Compute areas
    for(i32 t=0;t<nf;++t){
        const i32 a=faces[t*3],b=faces[t*3+1],c=faces[t*3+2];
        chart.surface_area_3d+=(verts[b]-verts[a]).cross(verts[c]-verts[a]).length()*0.5f;
        const Vec2<f32> e1=chart.uvs[b]-chart.uvs[a],e2=chart.uvs[c]-chart.uvs[a];
        chart.surface_area_uv+=std::abs(e1.x*e2.y-e1.y*e2.x)*0.5f;
    }
    return chart;
}
void pack_islands(std::vector<UVIsland>& islands, f32 padding) noexcept {
    // Shelf packing
    std::sort(islands.begin(),islands.end(),[](const UVIsland& a,const UVIsland& b){
        return (a.max.y-a.min.y)>(b.max.y-b.min.y);});
    f32 x=padding,y=padding,shelf_h=0;
    for(auto& isl:islands){
        const f32 w=isl.max.x-isl.min.x,h=isl.max.y-isl.min.y;
        if(x+w+padding>1.0f){x=padding;y+=shelf_h+padding;shelf_h=0;}
        // Translate island UVs
        const Vec2<f32> offset{x-isl.min.x,y-isl.min.y};
        if(isl.uvs)for(auto& uv:*isl.uvs)uv=uv+offset;
        x+=w+padding; shelf_h=std::max(shelf_h,h);
    }
}
void relax_uvs(Vec2<f32>* uvs,i32 nv,const i32* faces,i32 nf,i32 iters) noexcept {
    std::vector<std::vector<i32>> adj(nv);
    for(i32 t=0;t<nf;++t){
        const i32 a=faces[t*3],b=faces[t*3+1],c=faces[t*3+2];
        adj[a].push_back(b);adj[a].push_back(c);
        adj[b].push_back(a);adj[b].push_back(c);
        adj[c].push_back(a);adj[c].push_back(b);
    }
    for(auto& a:adj){std::sort(a.begin(),a.end());a.erase(std::unique(a.begin(),a.end()),a.end());}
    std::vector<Vec2<f32>> tmp(nv);
    for(i32 it=0;it<iters;++it){
        for(i32 i=0;i<nv;++i){
            if(adj[i].empty()){tmp[i]=uvs[i];continue;}
            Vec2<f32> avg{};
            for(i32 j:adj[i])avg=avg+uvs[j];
            tmp[i]=(uvs[i]+avg*(1.0f/adj[i].size()))*0.5f;
        }
        std::copy(tmp.begin(),tmp.end(),uvs);
    }
}
TangentFrame compute_tangent_frame(Vec3<f32> p0,Vec3<f32> p1,Vec3<f32> p2,
    Vec2<f32> uv0,Vec2<f32> uv1,Vec2<f32> uv2) noexcept
{
    const Vec3<f32> e1=p1-p0,e2=p2-p0;
    const Vec2<f32> duv1=uv1-uv0,duv2=uv2-uv0;
    const f32 r=duv1.x*duv2.y-duv2.x*duv1.y;
    const f32 inv=std::abs(r)>1e-10f?1.0f/r:0;
    const Vec3<f32> T=(e1*duv2.y-e2*duv1.y)*inv;
    const Vec3<f32> B=(e2*duv1.x-e1*duv2.x)*inv;
    const Vec3<f32> N=e1.cross(e2).normalized();
    return{T.normalized(),B.normalized(),N};
}
} // namespace qs::math::uv
