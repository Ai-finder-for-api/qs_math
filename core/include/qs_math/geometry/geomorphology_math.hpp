#pragma once
// qs_math/geomorphology_math.hpp — Drainage networks, watershed, tectonic uplift (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <span>
#include <array>

namespace qs {

// ─── Digital elevation model ──────────────────────────────────────────────────

struct DEM {
    std::vector<f32> h;
    u32 nx{},ny{};
    f32 cell_size{1.f};

    void resize(u32 x,u32 y,f32 cs=1.f){nx=x;ny=y;cell_size=cs;h.assign(x*y,0.f);}
    [[nodiscard]] f32& at(u32 x,u32 y)      {return h[y*nx+x];}
    [[nodiscard]] f32  at(u32 x,u32 y) const{return h[y*nx+x];}
    [[nodiscard]] u32 size() const{return nx*ny;}

    // D8 flow direction: returns (dx,dy) of steepest descent neighbour
    [[nodiscard]] std::pair<i32,i32> flow_dir(u32 x,u32 y) const {
        f32 min_h=at(x,y); i32 bx=0,by=0;
        for(i32 dy=-1;dy<=1;++dy) for(i32 dx=-1;dx<=1;++dx){
            if(dx==0&&dy==0) continue;
            u32 nx2=u32(i32(x)+dx),ny2=u32(i32(y)+dy);
            if(nx2>=nx||ny2>=ny) continue;
            f32 slope=(at(x,y)-at(nx2,ny2))/std::sqrt(f32(dx*dx+dy*dy));
            if(slope>0&&at(nx2,ny2)<min_h){min_h=at(nx2,ny2);bx=dx;by=dy;}
        }
        return {bx,by};
    }
};

// ─── Flow accumulation (D8) ───────────────────────────────────────────────────

[[nodiscard]] inline std::vector<u32> flow_accumulation(const DEM& dem) {
    u32 N=dem.nx*dem.ny;
    std::vector<u32> acc(N,1);
    // Topological sort by elevation (low to high)
    std::vector<u32> order(N);
    for(u32 i=0;i<N;++i) order[i]=i;
    std::sort(order.begin(),order.end(),[&](u32 a,u32 b){return dem.h[a]>dem.h[b];});
    // Propagate accumulation downslope
    for(u32 idx:order){
        u32 x=idx%dem.nx, y=idx/dem.nx;
        auto[dx,dy]=dem.flow_dir(x,y);
        if(dx==0&&dy==0) continue;
        u32 nx2=u32(i32(x)+dx),ny2=u32(i32(y)+dy);
        acc[ny2*dem.nx+nx2]+=acc[idx];
    }
    return acc;
}

// ─── Watershed delineation ────────────────────────────────────────────────────

// Priority-flood algorithm for DEM depression filling
inline void fill_depressions(DEM& dem) {
    u32 nx=dem.nx,ny=dem.ny;
    std::vector<bool> closed(nx*ny,false);
    using P=std::pair<f32,u32>;
    std::priority_queue<P,std::vector<P>,std::greater<P>> pq;
    // Add border cells
    for(u32 x=0;x<nx;++x){pq.push({dem.at(x,0),x});closed[x]=true;pq.push({dem.at(x,ny-1),x+(ny-1)*nx});closed[x+(ny-1)*nx]=true;}
    for(u32 y=1;y<ny-1;++y){pq.push({dem.at(0,y),y*nx});closed[y*nx]=true;pq.push({dem.at(nx-1,y),(nx-1)+y*nx});closed[(nx-1)+y*nx]=true;}
    while(!pq.empty()){
        auto[h,idx]=pq.top();pq.pop();
        u32 x=idx%nx,y=idx/nx;
        for(i32 dy=-1;dy<=1;++dy)for(i32 dx=-1;dx<=1;++dx){
            if(!dx&&!dy) continue;
            u32 nx2=u32(i32(x)+dx),ny2=u32(i32(y)+dy);
            if(nx2>=nx||ny2>=ny) continue;
            u32 nidx=ny2*nx+nx2;
            if(closed[nidx]) continue;
            closed[nidx]=true;
            dem.at(nx2,ny2)=std::max(dem.at(nx2,ny2),h);
            pq.push({dem.at(nx2,ny2),nidx});
        }
    }
}

// Label watersheds by pour points
[[nodiscard]] inline std::vector<u32> watershed_labels(
    const DEM& dem, const std::vector<u32>& acc, u32 min_acc=100)
{
    u32 N=dem.nx*dem.ny;
    std::vector<u32> labels(N,0);
    u32 label=1;
    // Find drainage outlets (local acc maxima above threshold that flow off edge)
    for(u32 y=0;y<dem.ny;++y)
    for(u32 x=0;x<dem.nx;++x){
        u32 idx=y*dem.nx+x;
        if(acc[idx]<min_acc) continue;
        auto[dx,dy]=dem.flow_dir(x,y);
        if(dx==0&&dy==0){labels[idx]=label++;continue;}
        u32 nx2=u32(i32(x)+dx),ny2=u32(i32(y)+dy);
        if(nx2>=dem.nx||ny2>=dem.ny) labels[idx]=label++;
    }
    return labels;
}

// ─── Stream power incision (SPM) ─────────────────────────────────────────────

struct StreamPowerParams {
    f32 uplift{0.001f};      // m/yr
    f32 K{0.0005f};          // erodibility
    f32 m{0.5f};             // area exponent
    f32 n{1.0f};             // slope exponent
    f32 dt{100.f};           // years per step
};

// One step of stream power erosion
inline void stream_power_step(DEM& dem, const std::vector<u32>& acc, const StreamPowerParams& p) {
    u32 nx=dem.nx,ny=dem.ny;
    f32 cell_area=dem.cell_size*dem.cell_size;
    std::vector<f32> dh(nx*ny,0.f);
    for(u32 y=1;y<ny-1;++y)
    for(u32 x=1;x<nx-1;++x){
        u32 idx=y*nx+x;
        auto[dx,dy]=dem.flow_dir(x,y);
        if(!dx&&!dy) continue;
        u32 nx2=u32(i32(x)+dx),ny2=u32(i32(y)+dy);
        f32 slope=(dem.at(x,y)-dem.at(nx2,ny2))/(dem.cell_size*std::sqrt(f32(dx*dx+dy*dy)));
        slope=std::max(0.f,slope);
        f32 A=f32(acc[idx])*cell_area;
        f32 E=p.K*std::pow(A,p.m)*std::pow(slope,p.n);
        dh[idx]=(p.uplift-E)*p.dt;
    }
    for(u32 i=0;i<nx*ny;++i) dem.h[i]=std::max(0.f,dem.h[i]+dh[i]);
}

// ─── Tectonic uplift patterns ─────────────────────────────────────────────────

[[nodiscard]] inline std::vector<f32> dome_uplift(u32 nx, u32 ny, f32 rate=0.001f) {
    std::vector<f32> u(nx*ny);
    f32 cx=f32(nx)*0.5f,cy=f32(ny)*0.5f,r=std::min(cx,cy);
    for(u32 y=0;y<ny;++y)
    for(u32 x=0;x<nx;++x){
        f32 d=std::sqrt((f32(x)-cx)*(f32(x)-cx)+(f32(y)-cy)*(f32(y)-cy));
        u[y*nx+x]=rate*std::max(0.f,1.f-d/r);
    }
    return u;
}

[[nodiscard]] inline std::vector<f32> rift_uplift(u32 nx, u32 ny, f32 rate=0.001f) {
    std::vector<f32> u(nx*ny);
    f32 cx=f32(nx)*0.5f;
    for(u32 y=0;y<ny;++y)
    for(u32 x=0;x<nx;++x){
        f32 d=std::abs(f32(x)-cx)/cx;
        u[y*nx+x]=rate*(d>0.3f?1.f:0.f); // rift flanks
    }
    return u;
}

// ─── Hillslope diffusion ──────────────────────────────────────────────────────

// Linear hillslope diffusion: dh/dt = K * laplacian(h)
inline void hillslope_diffuse(DEM& dem, f32 K=0.01f, f32 dt=100.f) {
    u32 nx=dem.nx,ny=dem.ny;
    std::vector<f32> hn=dem.h;
    f32 inv_cs2=1.f/(dem.cell_size*dem.cell_size);
    for(u32 y=1;y<ny-1;++y)
    for(u32 x=1;x<nx-1;++x){
        f32 lapl=(dem.at(x+1,y)+dem.at(x-1,y)+dem.at(x,y+1)+dem.at(x,y-1)-4.f*dem.at(x,y))*inv_cs2;
        hn[y*nx+x]=dem.h[y*nx+x]+K*lapl*dt;
    }
    dem.h=hn;
}

// ─── Drainage basin metrics ───────────────────────────────────────────────────

[[nodiscard]] inline f32 hypsometric_integral(const DEM& dem) {
    if(dem.h.empty()) return 0.f;
    f32 hmin=*std::min_element(dem.h.begin(),dem.h.end());
    f32 hmax=*std::max_element(dem.h.begin(),dem.h.end());
    if(hmax<=hmin) return 0.f;
    f32 sum=0.f;
    for(f32 v:dem.h) sum+=(v-hmin)/(hmax-hmin);
    return sum/f32(dem.h.size());
}

[[nodiscard]] inline f32 relief_ratio(const DEM& dem) {
    f32 hmin=*std::min_element(dem.h.begin(),dem.h.end());
    f32 hmax=*std::max_element(dem.h.begin(),dem.h.end());
    f32 len=f32(std::max(dem.nx,dem.ny))*dem.cell_size;
    return (hmax-hmin)/len;
}

// Ruggedness index (high = rough terrain)
[[nodiscard]] inline f32 terrain_ruggedness_index(const DEM& dem, u32 x, u32 y) {
    f32 c=dem.at(x,y); f32 sum=0.f; u32 n=0;
    for(i32 dy=-1;dy<=1;++dy)for(i32 dx=-1;dx<=1;++dx){
        if(!dx&&!dy) continue;
        u32 nx2=u32(i32(x)+dx),ny2=u32(i32(y)+dy);
        if(nx2>=dem.nx||ny2>=dem.ny) continue;
        f32 d=dem.at(nx2,ny2)-c; sum+=d*d; ++n;
    }
    return n>0?std::sqrt(sum/f32(n)):0.f;
}

} // namespace qs
