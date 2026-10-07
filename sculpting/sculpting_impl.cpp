
// qs_math/sculpting/sculpting_impl.cpp (C++20)
#include "include/qs_math_sculpting.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cassert>
namespace qs::math::sculpt {

f32 brush_falloff(f32 t, Falloff f) noexcept {
    t=std::max(0.0f,std::min(1.0f,t));
    switch(f){
    case Falloff::Smooth:    return t*t*(3-2*t);
    case Falloff::Constant:  return 1.0f;
    case Falloff::Linear:    return 1-t;
    case Falloff::Sharp:     { f32 t2=1-t; return t2*t2*t2*t2; }
    case Falloff::Sphere:    return std::sqrt(std::max(0.0f,1-t*t));
    case Falloff::Root:      return std::sqrt(1-t);
    default: return 0;
    }
}
f32 brush_draw(Vec3<f32> p,Vec3<f32> c,Vec3<f32> n,f32 r,f32 s,Falloff f) noexcept {
    // Project to local 2D distance in brush plane
    Vec3<f32> d=p-c;
    const f32 depth=d.dot(n); // signed distance along normal
    Vec3<f32> dp=d-n*depth;
    const f32 dist=dp.length()/r;
    if(dist>1) return 0;
    return s*brush_falloff(dist,f);
}
Vec3<f32> brush_smooth(const Vec3<f32>* v,const i32* nb,i32 n,f32 s) noexcept {
    Vec3<f32> avg{};
    for(i32 i=0;i<n;++i) avg=avg+v[nb[i]];
    avg=avg*(1.0f/n);
    return (avg-v[0])*s;
}
Vec3<f32> brush_flatten(Vec3<f32> p,Vec3<f32> pp,Vec3<f32> pn,f32 s) noexcept {
    const f32 d=pn.dot(p-pp);
    return pn*(-d*s);
}
Vec3<f32> brush_pinch(Vec3<f32> p,Vec3<f32> c,f32 s) noexcept {
    return (c-p)*s;
}
Vec3<f32> brush_crease(Vec3<f32> p,Vec3<f32> c,Vec3<f32> n,f32 s,f32 r) noexcept {
    const Vec3<f32> d=p-c;
    const f32 dn=d.dot(n), dist=(d-n*dn).length()/r;
    if(dist>1) return {};
    const f32 w=brush_falloff(dist);
    // Pinch toward center laterally, push along normal
    return (c-p-n*dn)*w*s;
}

// ─── Loop subdivision ─────────────────────────────────────────────────────────
SubdivResult loop_subdivide(const Vec3<f32>* verts,i32 nv,
    const i32* faces_in,i32 nf) noexcept
{
    SubdivResult r;
    r.verts.assign(verts,verts+nv);
    // Edge midpoint map
    using Edge2=std::pair<i32,i32>;
    std::unordered_map<u64,i32> edge_mid;
    auto ek=[](i32 a,i32 b)->u64{return a<b?(u64)a<<32|(u32)b:(u64)b<<32|(u32)a;};
    auto get_mid=[&](i32 a,i32 b)->i32{
        u64 k=ek(a,b);
        auto it=edge_mid.find(k);
        if(it!=edge_mid.end())return it->second;
        i32 idx=(i32)r.verts.size();
        r.verts.push_back((verts[a]+verts[b])*0.5f);
        return edge_mid[k]=idx;
    };
    // Collect neighbour info for even-vertex update
    std::vector<std::vector<i32>> adj(nv);
    for(i32 t=0;t<nf;++t){
        const i32 a=faces_in[t*3],b=faces_in[t*3+1],c=faces_in[t*3+2];
        adj[a].push_back(b);adj[a].push_back(c);
        adj[b].push_back(a);adj[b].push_back(c);
        adj[c].push_back(a);adj[c].push_back(b);
    }
    // Update even vertices (Loop weights)
    for(i32 i=0;i<nv;++i){
        std::sort(adj[i].begin(),adj[i].end());adj[i].erase(std::unique(adj[i].begin(),adj[i].end()),adj[i].end());
        const i32 n=adj[i].size();
        if(n==0){continue;}
        const f32 beta=n==3?3.0f/16:(3.0f/(8*n));
        Vec3<f32> nb_sum{};
        for(auto j:adj[i])nb_sum=nb_sum+verts[j];
        r.verts[i]=verts[i]*(1-n*beta)+nb_sum*beta;
    }
    // Build new faces
    for(i32 t=0;t<nf;++t){
        const i32 a=faces_in[t*3],b=faces_in[t*3+1],c=faces_in[t*3+2];
        const i32 ab=get_mid(a,b),bc=get_mid(b,c),ca=get_mid(c,a);
        r.faces.push_back({a,ab,ca});
        r.faces.push_back({ab,b,bc});
        r.faces.push_back({ca,bc,c});
        r.faces.push_back({ab,bc,ca});
    }
    return r;
}

// ─── Catmull-Clark limit surface ──────────────────────────────────────────────
Vec3<f32> catmull_clark_limit(Vec3<f32> v,const Vec3<f32>* ring,i32 n) noexcept {
    // Limit position: (n-2)/n*v + (1/n^2)*sum(ring)
    Vec3<f32> sum{};
    for(i32 i=0;i<n;++i)sum=sum+ring[i];
    const f32 nf=(f32)n;
    return v*((nf-2)/nf)+sum*(1.0f/(nf*nf));
}
Vec3<f32> catmull_clark_limit_normal(Vec3<f32> v,const Vec3<f32>* ring,i32 n) noexcept {
    // Tangent vectors via eigenvalue of the subdivision matrix
    Vec3<f32> t1{},t2{};
    for(i32 i=0;i<n;++i){
        const f32 a1=2*3.14159265f*i/n, a2=2*3.14159265f*i/n+3.14159265f/n;
        t1=t1+ring[i]*std::cos(a1);
        t2=t2+ring[i]*std::cos(a2);
    }
    return t1.cross(t2).normalized();
    (void)v;
}
// ─── Voxel SDF ────────────────────────────────────────────────────────────────
void VoxelGrid::compute_sdf(const Vec3<f32>* verts,i32 nv,
    const i32* faces,i32 nf) noexcept
{
    sdf.assign(nx*ny*nz,1e10f);
    for(i32 z=0;z<nz;++z)for(i32 y=0;y<ny;++y)for(i32 x=0;x<nx;++x){
        Vec3<f32> p=origin+Vec3<f32>{x*voxel_size,y*voxel_size,z*voxel_size};
        f32 best=1e10f;
        for(i32 t=0;t<nf;++t){
            const Vec3<f32> a=verts[faces[t*3]],b=verts[faces[t*3+1]],c=verts[faces[t*3+2]];
            // Point-triangle distance
            const Vec3<f32> ab=b-a,ac=c-a,ap=p-a;
            const f32 d1=ab.dot(ap),d2=ac.dot(ap);
            if(d1<=0&&d2<=0){const f32 d=(p-a).length();if(d<std::abs(best))best=d;continue;}
            const Vec3<f32> bp=p-b; const f32 d3=ab.dot(bp),d4=ac.dot(bp);
            if(d3>=0&&d4<=d3){const f32 d=(p-b).length();if(d<std::abs(best))best=d;continue;}
            const Vec3<f32> cp=p-c; const f32 d5=ab.dot(cp),d6=ac.dot(cp);
            if(d6>=0&&d5<=d6){const f32 d=(p-c).length();if(d<std::abs(best))best=d;continue;}
            const f32 vc=d1*d4-d3*d2;
            if(vc<=0&&d1>=0&&d3<=0){const f32 v=d1/(d1-d3);const f32 d=(p-a-ab*v).length();if(d<std::abs(best))best=d;continue;}
            const f32 vb=d5*d2-d1*d6;
            if(vb<=0&&d2>=0&&d6<=0){const f32 w=d2/(d2-d6);const f32 d=(p-a-ac*w).length();if(d<std::abs(best))best=d;continue;}
            const f32 va=d3*d6-d5*d4;
            if(va<=0){const f32 w3=(d4-d3)/((d4-d3)+(d5-d6));const f32 d=(p-b-(c-b)*w3).length();if(d<std::abs(best))best=d;continue;}
            const f32 denom=1.0f/(va+vb+vc),vf=vb*denom,wf=vc*denom;
            Vec3<f32> closest=a+ab*vf+ac*wf;
            const f32 d=(p-closest).length();
            if(d<std::abs(best))best=d;
        }
        sdf[z*ny*nx+y*nx+x]=best;
    }
    (void)nv;
}
Vec3<f32> VoxelGrid::gradient_at(i32 x,i32 y,i32 z) const noexcept {
    auto s=[&](i32 xi,i32 yi,i32 zi)->f32{
        if(xi<0||xi>=nx||yi<0||yi>=ny||zi<0||zi>=nz)return 0;
        return sdf[zi*ny*nx+yi*nx+xi];};
    return Vec3<f32>{s(x+1,y,z)-s(x-1,y,z),s(x,y+1,z)-s(x,y-1,z),s(x,y,z+1)-s(x,y,z-1)}*(0.5f/voxel_size);
}
f32 VoxelGrid::sample(Vec3<f32> p) const noexcept {
    Vec3<f32> lp=(p-origin)*(1.0f/voxel_size);
    const i32 xi=(i32)lp.x,yi=(i32)lp.y,zi=(i32)lp.z;
    if(xi<0||xi>=nx-1||yi<0||yi>=ny-1||zi<0||zi>=nz-1)return 1e10f;
    const f32 fx=lp.x-xi,fy=lp.y-yi,fz=lp.z-zi;
    auto s=[&](i32 a,i32 b,i32 c)->f32{return sdf[c*ny*nx+b*nx+a];};
    return s(xi,yi,zi)*(1-fx)*(1-fy)*(1-fz)+s(xi+1,yi,zi)*fx*(1-fy)*(1-fz)+
           s(xi,yi+1,zi)*(1-fx)*fy*(1-fz)+s(xi+1,yi+1,zi)*fx*fy*(1-fz)+
           s(xi,yi,zi+1)*(1-fx)*(1-fy)*fz+s(xi+1,yi,zi+1)*fx*(1-fy)*fz+
           s(xi,yi+1,zi+1)*(1-fx)*fy*fz+s(xi+1,yi+1,zi+1)*fx*fy*fz;
}
} // namespace qs::math::sculpt
