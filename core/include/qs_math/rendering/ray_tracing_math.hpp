#pragma once
// qs_math/ray_tracing_math.hpp — BVH traversal, path tracing estimators, MIS weights (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "geometry3d.hpp"
#include "intersection.hpp"
#include "random.hpp"
#include "material_math.hpp"
#include <vector>
#include <cmath>
#include <optional>
#include <numbers>
#include <functional>
#include <algorithm>

namespace qs {

// ─── Radiometric quantities ───────────────────────────────────────────────────

// Radiance, irradiance, BRDF units all in SI
// Convert between radiance and irradiance: E = ∫ L cos(θ) dω

// Cosine-weighted hemisphere sampling
[[nodiscard]] inline Vec3f cosine_hemisphere_sample(Vec2f u) {
    f32 phi=2*f32(std::numbers::pi)*u.x;
    f32 r=std::sqrt(u.y);
    return {r*std::cos(phi), r*std::sin(phi), std::sqrt(std::max(0.f,1-u.y))};
}
[[nodiscard]] inline f32 cosine_hemisphere_pdf(f32 cos_theta) {
    return std::max(0.f,cos_theta)/f32(std::numbers::pi);
}

// Uniform hemisphere sampling
[[nodiscard]] inline Vec3f uniform_hemisphere_sample(Vec2f u) {
    f32 phi=2*f32(std::numbers::pi)*u.x;
    f32 cos_theta=1-u.y, sin_theta=std::sqrt(std::max(0.f,1-cos_theta*cos_theta));
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}
[[nodiscard]] inline f32 uniform_hemisphere_pdf() { return 1.f/(2*f32(std::numbers::pi)); }

// Uniform sphere sampling
[[nodiscard]] inline Vec3f uniform_sphere_sample(Vec2f u) {
    f32 phi=2*f32(std::numbers::pi)*u.x;
    f32 cos_theta=1-2*u.y, sin_theta=std::sqrt(std::max(0.f,1-cos_theta*cos_theta));
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}
[[nodiscard]] inline f32 uniform_sphere_pdf() { return 1.f/(4*f32(std::numbers::pi)); }

// GGX importance sampling (already in vec3_ext.hpp — here with full PDF)
[[nodiscard]] inline std::pair<Vec3f,f32> ggx_sample(Vec2f u, f32 roughness, Vec3f N) {
    f32 a=roughness*roughness;
    f32 phi=2*f32(std::numbers::pi)*u.x;
    f32 cos_theta2=(1-u.y)/(1+(a*a-1)*u.y);
    f32 cos_theta=std::sqrt(std::max(0.f,cos_theta2));
    f32 sin_theta=std::sqrt(std::max(0.f,1-cos_theta2));
    // Half-vector in tangent space
    Vec3f H_local{sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
    // Build TBN
    Vec3f T,B;
    if(std::abs(N.x)>0.9f) T=normalize(cross(N,Vec3f{0,1,0}));
    else T=normalize(cross(N,Vec3f{1,0,0}));
    B=cross(N,T);
    Vec3f H=T*H_local.x+B*H_local.y+N*H_local.z;
    // PDF = D(H) cos(θ_H) / (4 dot(H,V)) — evaluated for wi below
    f32 D_GGX=a*a/(f32(std::numbers::pi)*(cos_theta2*(a*a-1)+1)*(cos_theta2*(a*a-1)+1));
    f32 pdf=D_GGX*cos_theta; // caller must divide by 4|H·V| when converting to wi pdf
    return {normalize(H), pdf};
}

// ─── Multiple Importance Sampling (MIS) weights ────────────────────────────────

// Balance heuristic: w_i = n_i p_i / Σ_j n_j p_j
[[nodiscard]] inline f32 mis_balance(f32 pdf_a, f32 pdf_b, f32 n_a=1, f32 n_b=1) {
    f32 denom=n_a*pdf_a+n_b*pdf_b;
    return denom>1e-20f?n_a*pdf_a/denom:0;
}

// Power heuristic (β=2 recommended by Veach)
[[nodiscard]] inline f32 mis_power(f32 pdf_a, f32 pdf_b, f32 beta=2.f, f32 n_a=1, f32 n_b=1) {
    f32 a=std::pow(n_a*pdf_a,beta), b=std::pow(n_b*pdf_b,beta);
    return (a+b)>1e-20f?a/(a+b):0;
}

// Maximum heuristic
[[nodiscard]] inline f32 mis_maximum(f32 pdf_a, f32 pdf_b) {
    return pdf_a>pdf_b?1.f:0.f;
}

// ─── Path tracing estimators ──────────────────────────────────────────────────

// Monte Carlo estimator for direct lighting with MIS
struct DirectSample {
    Vec3f Le{};       // emitted radiance
    Vec3f wi{};       // incident direction
    f32   pdf{0};     // pdf of wi
    f32   dist{0};    // distance to light
};

// Russian roulette continuation probability
[[nodiscard]] inline f32 rr_probability(Vec3f throughput, u32 depth, u32 min_depth=3) {
    if(depth<min_depth) return 1.f;
    return std::min(0.95f,std::max(throughput.x,std::max(throughput.y,throughput.z)));
}

// Throughput update after bounce: T *= f_r * cos(θ) / pdf
[[nodiscard]] inline Vec3f update_throughput(Vec3f T, Vec3f brdf, f32 cos_theta, f32 pdf) {
    if(pdf<1e-10f) return {};
    return T*brdf*(cos_theta/pdf);
}

// Compute solid angle subtended by a sphere as seen from point
[[nodiscard]] inline f32 sphere_solid_angle(f32 r, f32 dist) {
    if(dist<=r) return 4*f32(std::numbers::pi);
    f32 sin_max=r/dist;
    return 2*f32(std::numbers::pi)*(1-std::sqrt(std::max(0.f,1-sin_max*sin_max)));
}

// Sample a point on a sphere light
[[nodiscard]] inline std::pair<Vec3f,f32> sphere_light_sample(Vec3f center, f32 radius, Vec3f ref_pos, Vec2f u) {
    Vec3f dir=center-ref_pos;
    f32 dist=length(dir);
    if(dist<=radius) {
        Vec3f p=uniform_sphere_sample(u)*radius+center;
        return {p, 1.f/(4*f32(std::numbers::pi)*radius*radius)};
    }
    dir=dir*(1/dist);
    // Sample cone subtended by sphere
    f32 cos_max=std::sqrt(std::max(0.f,1-(radius/dist)*(radius/dist)));
    f32 phi=2*f32(std::numbers::pi)*u.x;
    f32 cos_theta=1-u.y*(1-cos_max);
    f32 sin_theta=std::sqrt(std::max(0.f,1-cos_theta*cos_theta));
    // Build frame around dir
    Vec3f T,B;
    if(std::abs(dir.x)>0.9f) T=normalize(cross(dir,{0,1,0}));
    else T=normalize(cross(dir,{1,0,0}));
    B=cross(dir,T);
    Vec3f wi=T*(sin_theta*std::cos(phi))+B*(sin_theta*std::sin(phi))+dir*cos_theta;
    // Intersection with sphere
    f32 pdf=1.f/(2*f32(std::numbers::pi)*(1-cos_max));
    // Find point on sphere surface along wi from ref_pos
    Vec3f to_center=center-ref_pos;
    f32 tc=dot(to_center,wi);
    f32 disc=radius*radius-dot(to_center,to_center)+tc*tc;
    Vec3f hit=ref_pos+wi*(tc-std::sqrt(std::max(0.f,disc)));
    return {hit,pdf};
}

// ─── BVH traversal helpers ────────────────────────────────────────────────────
// (Full BVH is in bounding_volume.hpp; here are traversal and packet ray utilities)

// Ray packet for SIMD BVH traversal
struct RayPacket4 {
    f32 ox[4],oy[4],oz[4]; // origins
    f32 dx[4],dy[4],dz[4]; // directions
    f32 tmin[4],tmax[4];
};

// Slab test for ray vs AABB (SIMD-friendly structure)
[[nodiscard]] inline bool ray_aabb_scalar(Vec3f ro, Vec3f rd_inv, Vec3f mn, Vec3f mx, f32 tmin, f32 tmax) {
    f32 tx0=(mn.x-ro.x)*rd_inv.x, tx1=(mx.x-ro.x)*rd_inv.x;
    f32 ty0=(mn.y-ro.y)*rd_inv.y, ty1=(mx.y-ro.y)*rd_inv.y;
    f32 tz0=(mn.z-ro.z)*rd_inv.z, tz1=(mx.z-ro.z)*rd_inv.z;
    tmin=std::max({tmin,std::min(tx0,tx1),std::min(ty0,ty1),std::min(tz0,tz1)});
    tmax=std::min({tmax,std::max(tx0,tx1),std::max(ty0,ty1),std::max(tz0,tz1)});
    return tmin<=tmax;
}

// Surface area of AABB for SAH cost computation
[[nodiscard]] inline f32 aabb_surface_area(Vec3f mn, Vec3f mx) {
    Vec3f d=mx-mn;
    return 2*(d.x*d.y+d.y*d.z+d.z*d.x);
}

// SAH cost for splitting N primitives into left/right sets with given surface areas
[[nodiscard]] inline f32 sah_cost(f32 SA_node, f32 SA_left, f32 SA_right,
    u32 N_left, u32 N_right, f32 Ct=1.f, f32 Ci=1.f)
{
    return Ct+Ci*(SA_left*N_left+SA_right*N_right)/SA_node;
}

// Compute ray direction inverse (for slab test)
[[nodiscard]] inline Vec3f ray_inv_dir(Vec3f d) {
    return {1.f/(d.x+1e-20f),1.f/(d.y+1e-20f),1.f/(d.z+1e-20f)};
}

// ─── Monte Carlo estimators ───────────────────────────────────────────────────

// Irradiance estimator: E ≈ (1/N) Σ f(ωi) cos(θi) / p(ωi)
[[nodiscard]] inline Vec3f mc_irradiance_estimate(
    const std::function<Vec3f(Vec3f)>& Li, // incoming radiance function
    Vec3f N, u32 samples, PCG32& rng)
{
    Vec3f acc{};
    for(u32 i=0;i<samples;++i) {
        Vec2f u{rng.next_float(),rng.next_float()};
        Vec3f wi_local=cosine_hemisphere_sample(u);
        // Transform to world space (simple: N=(0,0,1))
        Vec3f wi=wi_local; // assume N=up for simplicity
        f32 cos_theta=wi.z;
        f32 pdf=cosine_hemisphere_pdf(cos_theta);
        if(pdf>1e-10f) acc=acc+Li(wi)*(cos_theta/pdf);
    }
    return acc*(1.f/samples);
}

// One-sample estimator: f(x)/p(x) — unbiased
[[nodiscard]] inline Vec3f one_sample_estimator(Vec3f f, f32 pdf) {
    return pdf>1e-10f?f*(1.f/pdf):Vec3f{};
}

// Stratified 2D sample in [0,1)²
[[nodiscard]] inline Vec2f stratified_sample(u32 i, u32 N, PCG32& rng) {
    u32 nx=u32(std::sqrt(f64(N))); nx=std::max(nx,1u);
    u32 iy=i/nx, ix=i%nx;
    return {(ix+rng.next_float())/nx, (iy+rng.next_float())/(N/nx)};
}

// ─── Photon mapping helpers ───────────────────────────────────────────────────
struct Photon {
    Vec3f pos, dir, power;
    u8 flags{0};
};

// Build kd-tree for photon map (returns sorted photon list by x)
inline void photon_map_sort_x(std::vector<Photon>& photons) {
    std::sort(photons.begin(),photons.end(),[](const Photon& a,const Photon& b){return a.pos.x<b.pos.x;});
}

// Cone filter for photon density estimation
[[nodiscard]] inline f32 photon_cone_filter(f32 r, f32 r_max, f32 k=1.1f) {
    return std::max(0.f,1-r/(k*r_max));
}
// Gaussian filter
[[nodiscard]] inline f32 photon_gaussian_filter(f32 r2, f32 r_max2, f32 alpha=0.918f) {
    return alpha*(1-std::exp(-1.953f*r2/(2*r_max2)));
}

// ─── Spectral rendering helpers ───────────────────────────────────────────────

// Hero wavelength sampling: pick λ_hero uniformly, derive 4 wavelengths with stratification
[[nodiscard]] inline std::array<f32,4> hero_wavelength_sample(f32 u, f32 lambda_min=380, f32 lambda_max=780) {
    f32 range=lambda_max-lambda_min;
    f32 base=lambda_min+u*range;
    return {base, std::fmod(base+range*0.25f-lambda_min,range)+lambda_min,
            std::fmod(base+range*0.50f-lambda_min,range)+lambda_min,
            std::fmod(base+range*0.75f-lambda_min,range)+lambda_min};
}

} // namespace qs
