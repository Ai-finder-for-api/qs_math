#pragma once
// qs_math/pathfinding_math.hpp — A* heuristics, NavMesh math, steering, potential fields (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "geometry2d.hpp"
#include "geometry3d.hpp"
#include <vector>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <functional>
#include <algorithm>
#include <cassert>
#include <optional>
#include <numbers>

namespace qs {

// ─── A* heuristics ───────────────────────────────────────────────────────────
[[nodiscard]] inline f32 heuristic_manhattan(Vec2f a, Vec2f b) {
    return std::abs(a.x-b.x)+std::abs(a.y-b.y);
}
[[nodiscard]] inline f32 heuristic_euclidean(Vec2f a, Vec2f b) {
    return dist(a,b);
}
[[nodiscard]] inline f32 heuristic_chebyshev(Vec2f a, Vec2f b) {
    return std::max(std::abs(a.x-b.x),std::abs(a.y-b.y));
}
[[nodiscard]] inline f32 heuristic_octile(Vec2f a, Vec2f b) {
    f32 dx=std::abs(a.x-b.x), dy=std::abs(a.y-b.y);
    return dx+dy+(std::numbers::sqrt2_v<f32>-2)*std::min(dx,dy);
}
// Weighted heuristic (inadmissible but faster): h_w = w * h, w > 1
[[nodiscard]] inline f32 heuristic_weighted(Vec2f a, Vec2f b, f32 w=1.5f) {
    return w*heuristic_euclidean(a,b);
}

// ─── Grid A* ─────────────────────────────────────────────────────────────────
struct GridNode {
    Vec2i pos;
    f32 g{0},f{0};
    bool operator>(const GridNode& o) const { return f>o.f; }
};

[[nodiscard]] inline std::vector<Vec2i> astar_grid(
    u32 width, u32 height,
    const std::function<bool(Vec2i)>& walkable,
    Vec2i start, Vec2i goal,
    bool allow_diagonal=true)
{
    auto idx=[&](Vec2i p){ return p.y*i32(width)+p.x; };
    auto valid=[&](Vec2i p){ return p.x>=0&&p.x<i32(width)&&p.y>=0&&p.y<i32(height)&&walkable(p); };
    u32 N=width*height;
    std::vector<f32> g_score(N,1e30f); g_score[u32(idx(start))]=0;
    std::vector<i32> came_from(N,-1);
    std::priority_queue<GridNode,std::vector<GridNode>,std::greater<GridNode>> open;
    open.push({start,0,heuristic_octile(Vec2f(f32(start.x),f32(start.y)),Vec2f(f32(goal.x),f32(goal.y)))});
    static constexpr Vec2i dirs4[4]={{1,0},{-1,0},{0,1},{0,-1}};
    static constexpr Vec2i dirs8[8]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    while(!open.empty()) {
        auto cur=open.top(); open.pop();
        if(cur.pos==goal) {
            // Reconstruct path
            std::vector<Vec2i> path;
            i32 ci=idx(goal);
            while(ci>=0){ path.push_back({ci%i32(width),ci/i32(width)}); ci=came_from[u32(ci)]; }
            std::reverse(path.begin(),path.end());
            return path;
        }
        u32 n_dirs=allow_diagonal?8:4;
        for(u32 d=0;d<n_dirs;++d) {
            Vec2i nb={cur.pos.x+(allow_diagonal?dirs8[d]:dirs4[d]).x,
                      cur.pos.y+(allow_diagonal?dirs8[d]:dirs4[d]).y};
            if(!valid(nb)) continue;
            f32 move_cost=(d>=4)?std::numbers::sqrt2_v<f32>:1.f;
            f32 ng=g_score[u32(idx(cur.pos))]+move_cost;
            u32 nbi=u32(idx(nb));
            if(ng<g_score[nbi]) {
                g_score[nbi]=ng;
                came_from[nbi]=idx(cur.pos);
                f32 h=heuristic_octile(Vec2f(f32(nb.x),f32(nb.y)),Vec2f(f32(goal.x),f32(goal.y)));
                open.push({nb,ng,ng+h});
            }
        }
    }
    return {}; // no path
}

// ─── Theta* (any-angle pathfinding) ──────────────────────────────────────────
// Line-of-sight check on grid
[[nodiscard]] inline bool line_of_sight(const std::function<bool(Vec2i)>& walkable,
    Vec2i s, Vec2i e)
{
    i32 dx=std::abs(e.x-s.x), dy=std::abs(e.y-s.y);
    i32 sx=s.x<e.x?1:-1, sy=s.y<e.y?1:-1;
    i32 x=s.x,y=s.y,err=dx-dy;
    while(x!=e.x||y!=e.y) {
        if(!walkable({x,y})) return false;
        i32 e2=2*err;
        if(e2>-dy){err-=dy;x+=sx;}
        if(e2<dx){err+=dx;y+=sy;}
    }
    return walkable(e);
}

// ─── NavMesh helpers ──────────────────────────────────────────────────────────
struct NavPoly {
    std::vector<Vec3f> verts;
    std::vector<u32>   neighbours; // adjacent polygon indices
    Vec3f centroid() const {
        Vec3f c{}; for(auto& v:verts) c=c+v; return c*(1.f/verts.size());
    }
    f32 area() const {
        f32 a=0;
        for(u32 i=1;i+1<u32(verts.size());++i) {
            Vec3f e1=verts[i]-verts[0],e2=verts[i+1]-verts[0];
            a+=length(cross(e1,e2))*0.5f;
        }
        return a;
    }
};

struct NavMesh {
    std::vector<NavPoly> polys;

    // Find nearest polygon to point p
    [[nodiscard]] u32 nearest_poly(Vec3f p) const {
        u32 best=0; f32 bd=1e30f;
        for(u32 i=0;i<u32(polys.size());++i) {
            f32 d=dist_sq(polys[i].centroid(),p);
            if(d<bd){bd=d;best=i;}
        }
        return best;
    }

    // A* on polygon graph
    [[nodiscard]] std::vector<u32> astar_polys(u32 from, u32 to) const {
        u32 N=u32(polys.size());
        std::vector<f32> g(N,1e30f); g[from]=0;
        std::vector<i32> prev(N,-1);
        using P=std::pair<f32,u32>;
        std::priority_queue<P,std::vector<P>,std::greater<P>> open;
        open.push({0,from});
        Vec3f goal=polys[to].centroid();
        while(!open.empty()) {
            auto [cost,cur]=open.top(); open.pop();
            if(cur==to) {
                std::vector<u32> path;
                for(i32 c=i32(to);c>=0;c=prev[u32(c)]) path.push_back(u32(c));
                std::reverse(path.begin(),path.end());
                return path;
            }
            for(u32 nb:polys[cur].neighbours) {
                f32 d=dist(polys[cur].centroid(),polys[nb].centroid());
                f32 ng=g[cur]+d;
                if(ng<g[nb]){g[nb]=ng;prev[nb]=i32(cur);open.push({ng+dist(polys[nb].centroid(),goal),nb});}
            }
        }
        return {};
    }

    // Simple funnel algorithm (string pulling) for polygon path
    [[nodiscard]] std::vector<Vec3f> funnel(const std::vector<u32>& poly_path, Vec3f start, Vec3f end) const {
        if(poly_path.empty()) return {start,end};
        std::vector<Vec3f> waypoints={start};
        // Simplified: just use centroids
        for(u32 pi:poly_path) waypoints.push_back(polys[pi].centroid());
        waypoints.push_back(end);
        return waypoints;
    }
};

// ─── Steering behaviours ──────────────────────────────────────────────────────
struct SteeringAgent {
    Vec3f pos{}, vel{}, acc{};
    f32 max_speed{5.f}, max_force{1.f}, mass{1.f};

    void apply_force(Vec3f force) {
        if(length(force)>max_force) force=normalize(force)*max_force;
        acc=acc+force*(1.f/mass);
    }
    void update(f32 dt) { vel=vel+acc*dt; if(length(vel)>max_speed) vel=normalize(vel)*max_speed; pos=pos+vel*dt; acc={}; }

    [[nodiscard]] Vec3f seek(Vec3f target) const {
        Vec3f desired=normalize(target-pos)*max_speed;
        return desired-vel;
    }
    [[nodiscard]] Vec3f flee(Vec3f threat) const { return -seek(threat); }
    [[nodiscard]] Vec3f arrive(Vec3f target, f32 slow_radius=2.f) const {
        Vec3f to=target-pos; f32 d=length(to);
        if(d<1e-5f) return -vel;
        f32 speed=d<slow_radius?max_speed*d/slow_radius:max_speed;
        return normalize(to)*speed-vel;
    }
    [[nodiscard]] Vec3f wander(Vec3f& wander_target, f32 dt, PCG32& rng) const {
        f32 wander_dist=2.f, wander_radius=1.f, wander_jitter=0.5f;
        Vec3f jitter{rng.next_gaussian()*wander_jitter,0,rng.next_gaussian()*wander_jitter};
        wander_target=wander_target+jitter;
        if(length(wander_target)>1e-5f) wander_target=normalize(wander_target)*wander_radius;
        Vec3f circle_center=normalize(vel)*wander_dist;
        return seek(pos+circle_center+wander_target);
    }
    [[nodiscard]] Vec3f pursue(const SteeringAgent& target) const {
        f32 T=dist(pos,target.pos)/max_speed;
        return seek(target.pos+target.vel*T);
    }
    [[nodiscard]] Vec3f evade(const SteeringAgent& threat) const {
        f32 T=dist(pos,threat.pos)/max_speed;
        return flee(threat.pos+threat.vel*T);
    }
    [[nodiscard]] Vec3f avoid_obstacle(Vec3f obstacle, f32 radius) const {
        Vec3f to=pos-obstacle;
        f32 d=length(to);
        if(d>radius*3) return {};
        return normalize(to)*(max_force*(1-d/(radius*3)));
    }
    [[nodiscard]] Vec3f separate(const std::vector<SteeringAgent>& others, f32 desired_sep=2.f) const {
        Vec3f steer{}; u32 cnt=0;
        for(const auto& o:others) {
            f32 d=dist(pos,o.pos);
            if(d>0&&d<desired_sep){ steer=steer+normalize(pos-o.pos)*(1.f/d); ++cnt; }
        }
        if(cnt>0) steer=steer*(1.f/cnt);
        return length(steer)>1e-5f?normalize(steer)*max_speed-vel:Vec3f{};
    }
    [[nodiscard]] Vec3f align(const std::vector<SteeringAgent>& neighbours) const {
        Vec3f avg_vel{}; u32 cnt=0;
        for(const auto& n:neighbours) if(dist(pos,n.pos)<5.f){ avg_vel=avg_vel+n.vel; ++cnt; }
        if(cnt==0) return {};
        avg_vel=avg_vel*(1.f/cnt);
        return normalize(avg_vel)*max_speed-vel;
    }
    [[nodiscard]] Vec3f cohere(const std::vector<SteeringAgent>& neighbours) const {
        Vec3f center{}; u32 cnt=0;
        for(const auto& n:neighbours) if(dist(pos,n.pos)<5.f){ center=center+n.pos; ++cnt; }
        if(cnt==0) return {};
        return seek(center*(1.f/cnt));
    }
    // Flocking = separation + alignment + cohesion
    [[nodiscard]] Vec3f flock(const std::vector<SteeringAgent>& all) const {
        return separate(all)*1.5f+align(all)*1.0f+cohere(all)*1.0f;
    }
};

// ─── Potential fields ─────────────────────────────────────────────────────────
using PotentialField2D = std::function<Vec2f(Vec2f)>;

// Attractive potential towards goal
[[nodiscard]] inline Vec2f attractive_potential(Vec2f pos, Vec2f goal, f32 ka=1.f, f32 threshold=5.f) {
    Vec2f to=goal-pos; f32 d=length(to);
    if(d<1e-5f) return {};
    if(d<threshold) return to*ka; // parabolic near goal
    return normalize(to)*(ka*threshold); // conic far away
}

// Repulsive potential from obstacle
[[nodiscard]] inline Vec2f repulsive_potential(Vec2f pos, Vec2f obs, f32 influence=3.f, f32 kr=1.f) {
    Vec2f to=pos-obs; f32 d=length(to);
    if(d>influence||d<1e-5f) return {};
    f32 scale=kr*(1/d-1/influence)*(1/(d*d));
    return normalize(to)*scale;
}

// Combined navigation potential
[[nodiscard]] inline Vec2f navigation_potential(Vec2f pos, Vec2f goal,
    const std::vector<Vec2f>& obstacles,
    f32 ka=1.f, f32 kr=2.f, f32 influence=2.f)
{
    Vec2f F=attractive_potential(pos,goal,ka);
    for(const auto& obs:obstacles) F=F+repulsive_potential(pos,obs,influence,kr);
    return F;
}

// ─── Jump Point Search (JPS) helper ──────────────────────────────────────────
// Forced neighbour detection for diagonal movement
[[nodiscard]] inline bool has_forced_neighbour(const std::function<bool(Vec2i)>& walkable,
    Vec2i cur, Vec2i dir)
{
    i32 dx=dir.x, dy=dir.y;
    if(dx!=0&&dy==0) { // horizontal
        return (!walkable({cur.x,cur.y+1})&&walkable({cur.x+dx,cur.y+1}))||
               (!walkable({cur.x,cur.y-1})&&walkable({cur.x+dx,cur.y-1}));
    }
    if(dx==0&&dy!=0) { // vertical
        return (!walkable({cur.x+1,cur.y})&&walkable({cur.x+1,cur.y+dy}))||
               (!walkable({cur.x-1,cur.y})&&walkable({cur.x-1,cur.y+dy}));
    }
    // diagonal
    return (!walkable({cur.x-dx,cur.y})&&walkable({cur.x-dx,cur.y+dy}))||
           (!walkable({cur.x,cur.y-dy})&&walkable({cur.x+dx,cur.y-dy}));
}

} // namespace qs
