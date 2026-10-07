
// qs_math/procedural/procedural_impl.cpp (C++20)
#include "include/qs_math_procedural.hpp"
#include <cmath>
#include <stack>
#include <queue>
#include <algorithm>
#include <numeric>
#include <random>
namespace qs::math::proc {
static f32 perlin_p2(f32 x,f32 y,u32 seed) noexcept {
    auto h2=[](i32 a,i32 b,u32 s)->u32{u32 h=(u32)a*1619u^(u32)b*31337u^s;h^=h>>16;h*=0x45d9f3b;return h^(h>>16);};
    auto gf=[](u32 h)->f32{return (h&0xffffff)*(1.0f/16777216.0f)*2-1;};
    i32 xi=(i32)std::floor(x),yi=(i32)std::floor(y);
    f32 fx=x-xi,fy=y-yi;
    auto fade=[](f32 t){return t*t*t*(t*(t*6-15)+10);};
    auto grad=[&](i32 ix,i32 iy,f32 rx,f32 ry)->f32{
        u32 hh=h2(ix,iy,seed);
        return gf(hh)*rx+gf(hh>>8)*ry;};
    f32 ux=fade(fx),uy=fade(fy);
    return grad(xi,yi,fx,fy)*(1-ux)*(1-uy)+grad(xi+1,yi,fx-1,fy)*ux*(1-uy)+
           grad(xi,yi+1,fx,fy-1)*(1-ux)*uy+grad(xi+1,yi+1,fx-1,fy-1)*ux*uy;
}
void Heightmap::normalise() noexcept {
    f32 mn=*std::min_element(data.begin(),data.end());
    f32 mx=*std::max_element(data.begin(),data.end());
    f32 range=mx-mn; if(range<1e-7f)return;
    for(auto&v:data)v=(v-mn)/range;
}
void Heightmap::erode_thermal(f32 talus,i32 iters) noexcept {
    for(i32 it=0;it<iters;++it)
        for(i32 y=1;y<h-1;++y)for(i32 x=1;x<w-1;++x){
            f32 c=at(x,y),dmax=0; i32 bx=x,by=y;
            for(auto[dx,dy]:std::initializer_list<std::pair<i32,i32>>{{1,0},{-1,0},{0,1},{0,-1}}){
                f32 d=c-at(x+dx,y+dy);
                if(d>dmax){dmax=d;bx=x+dx;by=y+dy;}
            }
            if(dmax>talus){at(x,y)-=dmax*0.5f;at(bx,by)+=dmax*0.5f;}
        }
}
void Heightmap::erode_hydraulic(i32 iters) noexcept {
    std::vector<f32> water(w*h,0.01f),sediment(w*h,0);
    for(i32 it=0;it<iters;++it){
        for(i32 y=1;y<h-1;++y)for(i32 x=1;x<w-1;++x){
            const f32 total=at(x,y)+water[y*w+x];
            f32 avg=0; i32 cnt=0;
            for(auto[dx,dy]:std::initializer_list<std::pair<i32,i32>>{{1,0},{-1,0},{0,1},{0,-1}}){
                avg+=at(x+dx,y+dy)+water[(y+dy)*w+(x+dx)]; cnt++;
            }
            avg/=cnt;
            const f32 delta=total-avg;
            if(delta>0){at(x,y)-=delta*0.1f;water[y*w+x]-=delta*0.1f;}
        }
    }
}
Vec3<f32> Heightmap::normal_at(i32 x,i32 y,f32 cs) const noexcept {
    const f32 hx1=x>0?at(x-1,y):at(x,y),hx2=x<w-1?at(x+1,y):at(x,y);
    const f32 hy1=y>0?at(x,y-1):at(x,y),hy2=y<h-1?at(x,y+1):at(x,y);
    Vec3<f32> n{(hx1-hx2)/(2*cs),1.0f,(hy1-hy2)/(2*cs)};
    const f32 len=n.length(); return len>1e-7f?n/len:Vec3<f32>{0,1,0};
}
void gen_heightmap_fbm(Heightmap& hm,i32 oct,f32 lac,f32 gain,u32 seed) noexcept {
    for(i32 y=0;y<hm.h;++y)for(i32 x=0;x<hm.w;++x){
        f32 fx=(f32)x/hm.w,fy=(f32)y/hm.h,v=0,amp=0.5f,freq=1;
        for(i32 i=0;i<oct;++i){v+=amp*perlin_p2(fx*freq,fy*freq,seed+i);amp*=gain;freq*=lac;}
        hm.at(x,y)=v;
    }
    hm.normalise();
}
void gen_heightmap_diamond_square(Heightmap& hm,f32 roughness,u32 seed) noexcept {
    // Size should be 2^n+1
    const i32 n=hm.w-1;
    u32 rng=seed;
    auto rf=[&](f32 scale)->f32{rng=rng*1664525u+1013904223u;return((rng>>8)*(1.0f/16777216.0f)*2-1)*scale;};
    hm.at(0,0)=hm.at(n,0)=hm.at(0,n)=hm.at(n,n)=0.5f;
    for(i32 step=n;step>1;step/=2){
        f32 scale=roughness*step;
        // Diamond
        for(i32 y=0;y<n;y+=step)for(i32 x=0;x<n;x+=step){
            const i32 h2=step/2;
            hm.at(x+h2,y+h2)=(hm.at(x,y)+hm.at(x+step,y)+hm.at(x,y+step)+hm.at(x+step,y+step))*0.25f+rf(scale);
        }
        // Square
        for(i32 y=0;y<=n;y+=step/2)for(i32 x=(y+step/2)%step;x<=n;x+=step){
            f32 sum=0; i32 cnt=0;
            if(x-step/2>=0){sum+=hm.at(x-step/2,y);cnt++;}
            if(x+step/2<=n){sum+=hm.at(x+step/2,y);cnt++;}
            if(y-step/2>=0){sum+=hm.at(x,y-step/2);cnt++;}
            if(y+step/2<=n){sum+=hm.at(x,y+step/2);cnt++;}
            hm.at(x,y)=sum/cnt+rf(scale);
        }
    }
    hm.normalise();
}
std::string LSystem::iterate(i32 n) const noexcept {
    std::string cur=axiom;
    for(i32 i=0;i<n;++i){
        std::string next; next.reserve(cur.size()*3);
        for(char c:cur){auto it=rules.find(c);next+=it!=rules.end()?it->second:std::string(1,c);}
        cur=std::move(next);
    }
    return cur;
}
std::vector<Vec3<f32>> lsystem_to_lines(const std::string& s,f32 step,f32 angle_deg,Vec3<f32> start) noexcept {
    std::vector<Vec3<f32>> lines;
    const f32 angle=angle_deg*3.14159265f/180;
    Vec3<f32> pos=start,dir{0,1,0};
    std::stack<std::pair<Vec3<f32>,Vec3<f32>>> stk;
    for(char c:s){
        switch(c){
        case 'F':case 'G':{Vec3<f32> np=pos+dir*step;lines.push_back(pos);lines.push_back(np);pos=np;}break;
        case '+':dir={dir.x*std::cos(angle)-dir.y*std::sin(angle),dir.x*std::sin(angle)+dir.y*std::cos(angle),dir.z};break;
        case '-':dir={dir.x*std::cos(-angle)-dir.y*std::sin(-angle),dir.x*std::sin(-angle)+dir.y*std::cos(-angle),dir.z};break;
        case '[':stk.push({pos,dir});break;
        case ']':if(!stk.empty()){pos=stk.top().first;dir=stk.top().second;stk.pop();}break;
        }
    }
    return lines;
}
void Maze::generate(u32 seed) noexcept {
    std::vector<bool> visited(w*h,false);
    u32 rng=seed;
    auto rn=[&](i32 n)->i32{rng=rng*1664525u+1013904223u;return(i32)((rng>>8)%n);};
    std::stack<Vec2<i32>> st;
    st.push({0,0}); visited[0]=true;
    const i32 dx[]={0,0,1,-1},dy[]={1,-1,0,0};
    const u8 bits[]={1,2,4,8},opp[]={1,0,3,2};
    while(!st.empty()){
        Vec2<i32> cur=st.top();
        std::vector<i32> dirs={0,1,2,3};
        // shuffle
        for(i32 i=3;i>0;--i)std::swap(dirs[i],dirs[rn(i+1)]);
        bool moved=false;
        for(i32 d:dirs){
            i32 nx=cur.x+dx[d],ny=cur.y+dy[d];
            if(nx<0||nx>=w||ny<0||ny>=h||visited[ny*w+nx])continue;
            cells[cur.y*w+cur.x]|=bits[d];
            cells[ny*w+nx]|=bits[opp[d]];
            visited[ny*w+nx]=true;
            st.push({nx,ny}); moved=true; break;
        }
        if(!moved)st.pop();
    }
}
bool Maze::can_go(i32 x,i32 y,i32 dir)const noexcept{return(cells[y*w+x]>>(dir))&1;}
std::vector<Vec2<i32>> Maze::solve(Vec2<i32> start,Vec2<i32> end)const noexcept{
    std::vector<Vec2<i32>> from(w*h,{-1,-1});
    std::queue<Vec2<i32>> q;q.push(start);from[start.y*w+start.x]=start;
    const i32 dx[]={0,0,1,-1},dy[]={1,-1,0,0};
    while(!q.empty()){
        Vec2<i32> c=q.front();q.pop();
        if(c.x==end.x&&c.y==end.y){
            std::vector<Vec2<i32>> path;
            for(Vec2<i32> p=end;!(p.x==start.x&&p.y==start.y);p=from[p.y*w+p.x])path.push_back(p);
            path.push_back(start);std::reverse(path.begin(),path.end());return path;}
        for(i32 d=0;d<4;++d){
            if(!can_go(c.x,c.y,d))continue;
            Vec2<i32> n{c.x+dx[d],c.y+dy[d]};
            if(from[n.y*w+n.x].x==-1){from[n.y*w+n.x]=c;q.push(n);}
        }
    }
    return {};
}
std::vector<Vec2<f32>> poisson_disk(f32 W,f32 H,f32 r,i32 tries,u32 seed) noexcept {
    const f32 cell=r/std::sqrt(2.0f);
    const i32 gw=(i32)std::ceil(W/cell),gh=(i32)std::ceil(H/cell);
    std::vector<i32> grid(gw*gh,-1);
    std::vector<Vec2<f32>> pts,active;
    u32 rng=seed;
    auto randu=[&]()->f32{rng=rng*1664525u+1013904223u;return(rng>>8)*(1.0f/16777216.0f);};
    auto add=[&](Vec2<f32> p){
        pts.push_back(p);active.push_back(p);
        const i32 gx=(i32)(p.x/cell),gy=(i32)(p.y/cell);
        grid[gy*gw+gx]=(i32)pts.size()-1;};
    add({randu()*W,randu()*H});
    while(!active.empty()){
        const i32 ri=(i32)(randu()*active.size());
        Vec2<f32> origin=active[ri];
        bool found=false;
        for(i32 t=0;t<tries;++t){
            const f32 a=randu()*6.2831853f,d=r+randu()*r;
            Vec2<f32> c{origin.x+d*std::cos(a),origin.y+d*std::sin(a)};
            if(c.x<0||c.x>=W||c.y<0||c.y>=H)continue;
            const i32 gx=(i32)(c.x/cell),gy=(i32)(c.y/cell);
            bool ok=true;
            for(i32 dy=-2;dy<=2&&ok;++dy)for(i32 dx=-2;dx<=2&&ok;++dx){
                const i32 nx=gx+dx,ny=gy+dy;
                if(nx<0||nx>=gw||ny<0||ny>=gh)continue;
                const i32 idx=grid[ny*gw+nx];
                if(idx<0)continue;
                if((pts[idx]-c).length_sq()<r*r)ok=false;
            }
            if(ok){add(c);found=true;break;}
        }
        if(!found)active.erase(active.begin()+ri);
    }
    return pts;
}
} // namespace qs::math::proc
