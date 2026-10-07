#pragma once
// qs_math/stochastic_geometry.hpp — Poisson point process, coverage, random graphs (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <functional>

namespace qs {

// ─── Poisson point process ────────────────────────────────────────────────────

// Homogeneous PPP: expected N = lambda * area
[[nodiscard]] inline std::vector<Vec2f> poisson_process_2d(
    f32 lambda, const Vec2f& lo, const Vec2f& hi, u64 seed=42)
{
    Pcg32 rng{seed};
    f32 area=(hi.x-lo.x)*(hi.y-lo.y);
    u32 n=u32(std::round(lambda*area+std::sqrt(lambda*area)*(rng.next_f32()*2.f-1.f)));
    std::vector<Vec2f> pts;
    pts.reserve(n);
    for(u32 i=0;i<n;++i)
        pts.push_back({lo.x+(hi.x-lo.x)*rng.next_f32(), lo.y+(hi.y-lo.y)*rng.next_f32()});
    return pts;
}

// Inhomogeneous PPP via thinning: keep point with probability p(x,y)
[[nodiscard]] inline std::vector<Vec2f> inhomogeneous_ppp_2d(
    f32 lambda_max, const Vec2f& lo, const Vec2f& hi,
    const std::function<f32(Vec2f)>& intensity_fn,
    u64 seed=42)
{
    auto candidates=poisson_process_2d(lambda_max,lo,hi,seed);
    Pcg32 rng{seed^0xDEAD};
    std::vector<Vec2f> pts;
    for(const auto& p:candidates)
        if(rng.next_f32()<intensity_fn(p)/lambda_max) pts.push_back(p);
    return pts;
}

// Cluster process (Matérn): parent PPP, each parent spawns children in disk
[[nodiscard]] inline std::vector<Vec2f> cluster_process_2d(
    f32 lambda_parent, f32 lambda_child, f32 cluster_radius,
    const Vec2f& lo, const Vec2f& hi, u64 seed=42)
{
    Pcg32 rng{seed};
    auto parents=poisson_process_2d(lambda_parent,lo,hi,seed^1);
    f32 area=k_pi*cluster_radius*cluster_radius;
    std::vector<Vec2f> pts;
    for(const auto& par:parents){
        u32 nc=u32(std::round(lambda_child*area*(0.5f+rng.next_f32())));
        for(u32 i=0;i<nc;++i){
            f32 r=cluster_radius*std::sqrt(rng.next_f32());
            f32 theta=k_tau*rng.next_f32();
            Vec2f c{par.x+r*std::cos(theta), par.y+r*std::sin(theta)};
            if(c.x>=lo.x&&c.x<=hi.x&&c.y>=lo.y&&c.y<=hi.y) pts.push_back(c);
        }
    }
    return pts;
}

// ─── Coverage / contact distributions ────────────────────────────────────────

// Empty space function F(r): P(no point in ball of radius r centered at arbitrary loc)
[[nodiscard]] inline f32 empty_space_function(
    const std::vector<Vec2f>& pts, const Vec2f& query, f32 r)
{
    for(const auto& p:pts) if(length_sq(p-query)<=r*r) return 0.f;
    return 1.f;
}

// Nearest-neighbour distance distribution G(r): estimated over sample
[[nodiscard]] inline std::vector<std::pair<f32,f32>> nn_distance_cdf(
    const std::vector<Vec2f>& pts, u32 n_bins=50, f32 max_r=1.f)
{
    u32 N=u32(pts.size());
    std::vector<f32> dists(N, std::numeric_limits<f32>::max());
    for(u32 i=0;i<N;++i) for(u32 j=0;j<N;++j) {
        if(i==j) continue;
        dists[i]=std::min(dists[i],length(pts[i]-pts[j]));
    }
    std::vector<std::pair<f32,f32>> cdf;
    for(u32 b=0;b<=n_bins;++b){
        f32 r=max_r*f32(b)/f32(n_bins);
        u32 count=0;
        for(f32 d:dists) if(d<=r) ++count;
        cdf.push_back({r,f32(count)/f32(N)});
    }
    return cdf;
}

// ─── Ripley's K function (spatial clustering measure) ────────────────────────

[[nodiscard]] inline f32 ripley_K(
    const std::vector<Vec2f>& pts, f32 r, f32 area)
{
    u32 N=u32(pts.size());
    if(N<2) return 0.f;
    u32 count=0;
    for(u32 i=0;i<N;++i) for(u32 j=0;j<N;++j) {
        if(i==j) continue;
        if(length_sq(pts[i]-pts[j])<=r*r) ++count;
    }
    f32 lambda=f32(N)/area;
    return f32(count)/(f32(N)*lambda);
}

// Linearized L function: L(r) = sqrt(K(r)/pi) - r; >0 = clustering, <0 = regularity
[[nodiscard]] inline f32 ripley_L(const std::vector<Vec2f>& pts, f32 r, f32 area) {
    return std::sqrt(ripley_K(pts,r,area)/k_pi)-r;
}

// ─── Random geometric graph ───────────────────────────────────────────────────

struct RandomGeoGraph {
    std::vector<Vec2f> nodes;
    std::vector<std::pair<u32,u32>> edges;

    void build(const std::vector<Vec2f>& pts, f32 connection_radius) {
        nodes=pts;
        edges.clear();
        u32 N=u32(pts.size());
        for(u32 i=0;i<N;++i) for(u32 j=i+1;j<N;++j)
            if(length(pts[i]-pts[j])<=connection_radius) edges.push_back({i,j});
    }

    [[nodiscard]] u32 degree(u32 v) const {
        u32 d=0;
        for(const auto&[u,w]:edges) if(u==v||w==v) ++d;
        return d;
    }

    [[nodiscard]] f32 connectivity() const {
        if(nodes.empty()) return 0.f;
        // Simple: fraction of nodes with at least one edge
        std::vector<bool> connected(nodes.size(),false);
        for(const auto&[u,v]:edges){connected[u]=true;connected[v]=true;}
        u32 cnt=0; for(bool c:connected) if(c) ++cnt;
        return f32(cnt)/f32(nodes.size());
    }
};

// ─── Germ-grain model ─────────────────────────────────────────────────────────

struct Grain {
    Vec2f center{};
    f32   radius{};
};

[[nodiscard]] inline std::vector<Grain> boolean_model(
    f32 lambda, f32 mean_radius, f32 radius_var,
    const Vec2f& lo, const Vec2f& hi, u64 seed=42)
{
    Pcg32 rng{seed};
    auto centers=poisson_process_2d(lambda,lo,hi,seed^42);
    std::vector<Grain> grains;
    for(const auto& c:centers){
        f32 r=std::max(0.01f,mean_radius+(rng.next_f32()*2.f-1.f)*std::sqrt(radius_var));
        grains.push_back({c,r});
    }
    return grains;
}

// Coverage probability at point p (fraction of point p covered by grains)
[[nodiscard]] inline bool boolean_model_covers(
    const Vec2f& p, const std::vector<Grain>& grains)
{
    for(const auto& g:grains) if(length_sq(p-g.center)<=g.radius*g.radius) return true;
    return false;
}

// Area fraction covered (Monte Carlo)
[[nodiscard]] inline f32 area_fraction(
    const std::vector<Grain>& grains, const Vec2f& lo, const Vec2f& hi,
    u32 samples=10000, u64 seed=0)
{
    Pcg32 rng{seed};
    u32 covered=0;
    for(u32 i=0;i<samples;++i){
        Vec2f p{lo.x+(hi.x-lo.x)*rng.next_f32(), lo.y+(hi.y-lo.y)*rng.next_f32()};
        if(boolean_model_covers(p,grains)) ++covered;
    }
    return f32(covered)/f32(samples);
}

// ─── Strauss process (repulsive PPP) ─────────────────────────────────────────

// Metropolis-Hastings sampler for Strauss process
[[nodiscard]] inline std::vector<Vec2f> strauss_process(
    f32 beta, f32 gamma, f32 r, const Vec2f& lo, const Vec2f& hi,
    u32 n_iter=10000, u64 seed=42)
{
    Pcg32 rng{seed};
    std::vector<Vec2f> pts;
    f32 area=(hi.x-lo.x)*(hi.y-lo.y);

    for(u32 iter=0;iter<n_iter;++iter){
        bool add=rng.next_f32()<0.5f;
        if(add||pts.empty()){
            // Propose adding a point
            Vec2f p{lo.x+(hi.x-lo.x)*rng.next_f32(), lo.y+(hi.y-lo.y)*rng.next_f32()};
            u32 c=0; for(const auto& q:pts) if(length_sq(p-q)<r*r) ++c;
            f32 ratio=beta*area*std::pow(gamma,f32(c))/f32(pts.size()+1);
            if(rng.next_f32()<std::min(1.f,ratio)) pts.push_back(p);
        } else {
            // Propose removing a random point
            u32 idx=u32(rng.next_f32()*f32(pts.size()));
            Vec2f p=pts[idx];
            u32 c=0; for(u32 j=0;j<u32(pts.size());++j) if(j!=idx&&length_sq(p-pts[j])<r*r) ++c;
            f32 ratio=f32(pts.size())/(beta*area*std::pow(gamma,f32(c)));
            if(rng.next_f32()<std::min(1.f,ratio)){pts.erase(pts.begin()+idx);}
        }
    }
    return pts;
}

} // namespace qs
