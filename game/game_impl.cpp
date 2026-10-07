// qs_math/game/game_impl.cpp (C++20)
#include "include/qs_math_game.hpp"
#include <qs_math/linear_algebra/mat4.hpp>
#include <cmath>
#include <queue>
namespace qs::math::game {

// ─── Camera2D ────────────────────────────────────────────────────────────────
Vec2<f32> Camera2D::world_to_screen(Vec2<f32> w, Vec2<f32> vs) const noexcept {
    Vec2<f32> d=w-position;
    const f32 c=std::cos(-rotation),s=std::sin(-rotation);
    Vec2<f32> r{d.x*c-d.y*s,d.x*s+d.y*c};
    return r*zoom+vs*0.5f;
}
Vec2<f32> Camera2D::screen_to_world(Vec2<f32> sc, Vec2<f32> vs) const noexcept {
    Vec2<f32> r=(sc-vs*0.5f)*(1.0f/zoom);
    const f32 c=std::cos(rotation),s=std::sin(rotation);
    return Vec2<f32>{r.x*c-r.y*s,r.x*s+r.y*c}+position;
}
Mat4<f32> Camera2D::view_matrix() const noexcept {
    // 2D view: translate + rotate + scale
    return Mat4<f32>::trs(
        Vec3<f32>{-position.x,-position.y,0},
        Vec3<f32>{0,0,rotation},
        Vec3<f32>{zoom,zoom,1});
}

// ─── Bresenham line ───────────────────────────────────────────────────────────
void bresenham_line(Vec2<i32> a, Vec2<i32> b, std::vector<Vec2<i32>>& out) noexcept {
    i32 dx=std::abs(b.x-a.x),dy=std::abs(b.y-a.y);
    i32 sx=a.x<b.x?1:-1, sy=a.y<b.y?1:-1;
    i32 err=dx-dy;
    for(;;){
        out.push_back(a);
        if(a.x==b.x&&a.y==b.y) break;
        i32 e2=2*err;
        if(e2>-dy){err-=dy;a.x+=sx;}
        if(e2< dx){err+=dx;a.y+=sy;}
    }
}

// ─── Flood fill (BFS, 4-connected) ────────────────────────────────────────────
void flood_fill(bool* grid,i32 w,i32 h,Vec2<i32> start,bool fv,std::vector<Vec2<i32>>& out) noexcept {
    if(start.x<0||start.x>=w||start.y<0||start.y>=h) return;
    const bool orig=grid[start.y*w+start.x];
    if(orig==fv) return;
    std::queue<Vec2<i32>> q; q.push(start);
    grid[start.y*w+start.x]=fv;
    const i32 dx4[4]={1,-1,0,0},dy4[4]={0,0,1,-1};
    while(!q.empty()){
        auto cur=q.front(); q.pop(); out.push_back(cur);
        for(i32 d=0;d<4;++d){
            Vec2<i32> nb{cur.x+dx4[d],cur.y+dy4[d]};
            if(nb.x<0||nb.x>=w||nb.y<0||nb.y>=h) continue;
            if(grid[nb.y*w+nb.x]==fv) continue;
            grid[nb.y*w+nb.x]=fv; q.push(nb);
        }
    }
}

// ─── Screenshake offset ───────────────────────────────────────────────────────
Vec2<f32> ScreenShake::offset(f32 seed) const noexcept {
    const f32 t2=trauma*trauma;
    // Simple hash-based pseudo-random using seed
    auto hash=[](f32 s)->f32{
        const i32 n=(i32)(s*127.1f+311.7f);
        const i32 m=n*n*n*15731;
        return (m&0x7fffffff)/(f32)0x7fffffff*2-1;
    };
    return { hash(seed)*max_offset*t2, hash(seed+1.37f)*max_offset*t2 };
}

// ─── A* pathfinding on uniform grid ──────────────────────────────────────────
std::vector<Vec2<i32>> astar(const bool* walkable, i32 w, i32 h,
    Vec2<i32> start, Vec2<i32> goal) noexcept
{
    struct Node{Vec2<i32>p;f32 g,f;i32 parent;};
    std::vector<f32> g(w*h,1e30f), f(w*h,1e30f);
    std::vector<i32> from(w*h,-1);
    std::vector<bool> closed(w*h,false);
    auto idx=[&](Vec2<i32> t){return t.y*w+t.x;};
    auto valid=[&](Vec2<i32> t){return t.x>=0&&t.x<w&&t.y>=0&&t.y<h&&walkable[idx(t)];};
    auto h_fn=[&](Vec2<i32> a){return heuristic_euclidean(a,goal);};
    g[idx(start)]=0; f[idx(start)]=h_fn(start);
    // Min-heap by f
    auto cmp=[&](i32 a,i32 b){return f[a]>f[b];};
    std::vector<i32> open; open.push_back(idx(start));
    const i32 dx8[8]={1,-1,0,0,1,-1,1,-1};
    const i32 dy8[8]={0,0,1,-1,1,-1,-1,1};
    const f32 dc8[8]={1,1,1,1,1.41f,1.41f,1.41f,1.41f};
    while(!open.empty()){
        std::pop_heap(open.begin(),open.end(),cmp);
        i32 ci=open.back(); open.pop_back();
        if(closed[ci]) continue;
        closed[ci]=true;
        Vec2<i32> cur{ci%w,ci/w};
        if(cur.x==goal.x&&cur.y==goal.y){
            std::vector<Vec2<i32>> path;
            for(i32 i=ci;i!=-1;i=from[i]) path.push_back({i%w,i/w});
            std::reverse(path.begin(),path.end()); return path;
        }
        for(i32 d=0;d<8;++d){
            Vec2<i32> nb{cur.x+dx8[d],cur.y+dy8[d]};
            if(!valid(nb)) continue;
            i32 ni=idx(nb);
            if(closed[ni]) continue;
            const f32 ng=g[ci]+dc8[d];
            if(ng<g[ni]){
                g[ni]=ng; f[ni]=ng+h_fn(nb); from[ni]=ci;
                open.push_back(ni); std::push_heap(open.begin(),open.end(),cmp);
            }
        }
    }
    return {};
}
} // namespace qs::math::game
