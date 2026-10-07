#pragma once
// qs_math/navigation_mesh.hpp — NavMesh polygon ops, A*, funnel, string pull (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <span>
#include <optional>
#include <array>

namespace qs {

// ─── NavMesh polygon ─────────────────────────────────────────────────────────

struct NavPoly {
    std::vector<Vec3f> verts;      // polygon vertices (convex)
    Vec3f centroid{};
    f32   area{};
    u32   id{};
    std::vector<u32> neighbours;   // adjacent poly ids
    u32   flags{0};                // walkable, water, etc.
};

[[nodiscard]] inline Vec3f poly_centroid(std::span<const Vec3f> verts) {
    Vec3f c{0,0,0};
    for(const auto& v:verts) c=c+v;
    return c*(1.f/f32(verts.size()));
}

[[nodiscard]] inline bool point_in_poly_xz(const Vec3f& p, std::span<const Vec3f> verts) {
    u32 N=u32(verts.size());
    bool inside=false;
    for(u32 i=0,j=N-1;i<N;j=i++) {
        f32 xi=verts[i].x,zi=verts[i].z;
        f32 xj=verts[j].x,zj=verts[j].z;
        if(((zi>p.z)!=(zj>p.z))&&(p.x<(xj-xi)*(p.z-zi)/(zj-zi)+xi))
            inside=!inside;
    }
    return inside;
}

// Find which poly contains point
[[nodiscard]] inline std::optional<u32> find_poly(
    const Vec3f& p, std::span<const NavPoly> polys)
{
    for(const auto& poly:polys)
        if(point_in_poly_xz(p,poly.verts)) return poly.id;
    return std::nullopt;
}

// Project point onto polygon surface (XZ search, Y from plane)
[[nodiscard]] inline f32 poly_height_at(const Vec3f& p, const NavPoly& poly) {
    // Compute barycentric Y from triangle fan
    u32 N=u32(poly.verts.size());
    for(u32 i=1;i<N-1;++i) {
        const Vec3f& a=poly.verts[0];
        const Vec3f& b=poly.verts[i];
        const Vec3f& c=poly.verts[i+1];
        Vec3f v0=c-a,v1=b-a,v2=Vec3f{p.x,a.y,p.z}-a;
        f32 d00=dot(v0,v0),d01=dot(v0,v1),d11=dot(v1,v1);
        f32 d20=dot(v2,v0),d21=dot(v2,v1);
        f32 denom=d00*d11-d01*d01;
        if(std::abs(denom)<1e-8f) continue;
        f32 w=(d11*d20-d01*d21)/denom;
        f32 v=(d00*d21-d01*d20)/denom;
        f32 u2=1.f-v-w;
        if(u2>=0&&v>=0&&w>=0) return a.y*u2+b.y*v+c.y*w;
    }
    return poly.centroid.y;
}

// ─── Portal edge (shared edge between two polys) ──────────────────────────────

struct Portal {
    Vec3f left{}, right{}; // portal vertices
    u32 from_poly{}, to_poly{};
};

[[nodiscard]] inline std::optional<Portal> find_portal(
    const NavPoly& a, const NavPoly& b)
{
    // Find the shared edge
    u32 NA=u32(a.verts.size()),NB=u32(b.verts.size());
    for(u32 i=0;i<NA;++i)
    for(u32 j=0;j<NB;++j) {
        Vec3f ai=a.verts[i],   ai2=a.verts[(i+1)%NA];
        Vec3f bj=b.verts[j],   bj2=b.verts[(j+1)%NB];
        if(length_sq(ai-bj)<0.001f&&length_sq(ai2-bj2)<0.001f)
            return Portal{ai,ai2,a.id,b.id};
        if(length_sq(ai-bj2)<0.001f&&length_sq(ai2-bj)<0.001f)
            return Portal{ai2,ai,a.id,b.id};
    }
    return std::nullopt;
}

// ─── A* pathfinding on NavMesh ────────────────────────────────────────────────

struct NavNode {
    u32 poly_id{};
    f32 g{}, h{}, f{};
    u32 parent{u32(-1)};
};

[[nodiscard]] inline std::vector<u32> navmesh_astar(
    u32 start_poly, u32 goal_poly,
    std::span<const NavPoly> polys)
{
    auto heuristic=[&](u32 a, u32 b)->f32{
        return length(polys[a].centroid-polys[b].centroid);};

    std::unordered_map<u32,NavNode> nodes;
    std::priority_queue<std::pair<f32,u32>,std::vector<std::pair<f32,u32>>,std::greater<>> open;
    std::unordered_map<u32,bool> closed;

    nodes[start_poly]={start_poly,0.f,heuristic(start_poly,goal_poly)};
    nodes[start_poly].f=nodes[start_poly].h;
    open.push({nodes[start_poly].f,start_poly});

    while(!open.empty()){
        auto[f,curr]=open.top(); open.pop();
        if(curr==goal_poly){
            // Reconstruct path
            std::vector<u32> path;
            u32 node=goal_poly;
            while(node!=u32(-1)){path.push_back(node);node=nodes[node].parent;}
            std::reverse(path.begin(),path.end());
            return path;
        }
        if(closed[curr]) continue;
        closed[curr]=true;
        for(u32 nb:polys[curr].neighbours){
            if(closed[nb]) continue;
            f32 edge_cost=length(polys[curr].centroid-polys[nb].centroid);
            f32 g2=nodes[curr].g+edge_cost;
            if(!nodes.count(nb)||g2<nodes[nb].g){
                nodes[nb]={nb,g2,heuristic(nb,goal_poly),nb,curr};
                nodes[nb].f=g2+nodes[nb].h;
                open.push({nodes[nb].f,nb});
            }
        }
    }
    return {}; // no path
}

// ─── Funnel algorithm (string pulling) ───────────────────────────────────────

// Simple string-pull on portal sequence to get smooth path
[[nodiscard]] inline std::vector<Vec3f> funnel_path(
    const Vec3f& start, const Vec3f& goal,
    std::span<const Portal> portals)
{
    std::vector<Vec3f> path;
    path.push_back(start);

    Vec3f apex=start;
    Vec3f left_leg=start, right_leg=start;
    u32 left_idx=0, right_idx=0;

    auto cross2d=[](Vec3f a,Vec3f b,Vec3f c)->f32{
        return (b.x-a.x)*(c.z-a.z)-(b.z-a.z)*(c.x-a.x);};

    for(u32 i=0;i<u32(portals.size());++i){
        Vec3f pl=portals[i].left, pr=portals[i].right;
        // Tighten right leg
        if(cross2d(apex,right_leg,pr)<=0.f){
            if(cross2d(apex,left_leg,pr)>0.f){
                path.push_back(left_leg);
                apex=left_leg; right_leg=apex; left_leg=apex;
                left_idx=i; i=left_idx; continue;
            }
            right_leg=pr; right_idx=i;
        }
        // Tighten left leg
        if(cross2d(apex,left_leg,pl)>=0.f){
            if(cross2d(apex,right_leg,pl)<0.f){
                path.push_back(right_leg);
                apex=right_leg; right_leg=apex; left_leg=apex;
                right_idx=i; i=right_idx; continue;
            }
            left_leg=pl; left_idx=i;
        }
    }
    path.push_back(goal);
    return path;
}

// ─── Crowd simulation (local avoidance) ──────────────────────────────────────

struct CrowdAgent {
    Vec3f pos{};
    Vec3f vel{};
    Vec3f desired_vel{};
    f32   radius{0.3f};
    f32   max_speed{3.5f};
    u32   id{};
};

// RVO (Reciprocal Velocity Obstacle) velocity candidate selection
[[nodiscard]] inline Vec3f rvo_desired_velocity(
    const CrowdAgent& agent,
    std::span<const CrowdAgent> others,
    f32 time_horizon=2.f)
{
    Vec3f pref=agent.desired_vel;
    Vec3f best=pref;
    f32 best_cost=std::numeric_limits<f32>::max();

    // Sample candidate velocities
    u32 N_candidates=32;
    for(u32 c=0;c<N_candidates;++c){
        f32 ang=k_tau*f32(c)/f32(N_candidates);
        f32 speed=agent.max_speed*(f32(c%4)+1.f)/4.f;
        Vec3f cand{std::cos(ang)*speed,0,std::sin(ang)*speed};

        // Check all others
        bool valid=true;
        for(const auto& other:others){
            if(other.id==agent.id) continue;
            Vec3f rel_pos=other.pos-agent.pos;
            Vec3f rel_vel=cand-other.vel; // RVO: half the combined VO
            f32 combined_r=agent.radius+other.radius;
            // Collision time
            f32 a=dot(rel_vel,rel_vel);
            f32 b=dot(rel_pos,rel_vel);
            f32 c2=dot(rel_pos,rel_pos)-combined_r*combined_r;
            f32 discr=b*b-a*c2;
            if(discr>0.f&&a>1e-6f){
                f32 t=(-b-std::sqrt(discr))/a;
                if(t>0.f&&t<time_horizon){valid=false;break;}
            }
        }
        f32 cost=length_sq(cand-pref)+(valid?0.f:1e6f);
        if(cost<best_cost){best_cost=cost;best=cand;}
    }
    return best;
}

} // namespace qs
