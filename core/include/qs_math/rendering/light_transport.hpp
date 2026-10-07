#pragma once
// qs_math/rendering/light_transport.hpp — Light transport: path tracing math,
//   Monte Carlo estimators, importance sampling, MIS (multiple importance sampling),
//   photon mapping, radiosity, participating media, phase functions (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include "../arithmetic/random.hpp"
#include <cmath>
#include <vector>

namespace qs::math::render {

// ─── Hemisphere sampling ──────────────────────────────────────────────────
// Uniform hemisphere sample given (u1,u2) ∈ [0,1)^2
[[nodiscard]] inline Vec3f sample_hemisphere_uniform(f32 u1, f32 u2) noexcept {
    f32 cos_theta = u1;
    f32 sin_theta = std::sqrt(max(0.f, 1.f - cos_theta*cos_theta));
    f32 phi = f32(TWO_PI)*u2;
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}
[[nodiscard]] constexpr f32 pdf_hemisphere_uniform() noexcept { return f32(INV_TWO_PI); }

// Cosine-weighted hemisphere (importance sampling for Lambertian)
[[nodiscard]] inline Vec3f sample_hemisphere_cosine(f32 u1, f32 u2) noexcept {
    f32 phi = f32(TWO_PI)*u2;
    f32 sin_theta = std::sqrt(u1);
    f32 cos_theta = std::sqrt(max(0.f, 1.f-u1));
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}
[[nodiscard]] inline f32 pdf_cosine(f32 cos_theta) noexcept {
    return max(0.f, cos_theta) * f32(INV_PI);
}

// Full sphere uniform
[[nodiscard]] inline Vec3f sample_sphere_uniform(f32 u1, f32 u2) noexcept {
    f32 cos_theta = 1.f - 2.f*u1;
    f32 sin_theta = std::sqrt(max(0.f, 1.f - cos_theta*cos_theta));
    f32 phi = f32(TWO_PI)*u2;
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}
[[nodiscard]] constexpr f32 pdf_sphere_uniform() noexcept { return f32(0.25*INV_PI); }

// ─── GGX importance sampling (see pbr_math.hpp for VNDF) ─────────────────
[[nodiscard]] inline Vec3f sample_ggx(f32 u1, f32 u2, f32 alpha) noexcept {
    f32 phi = f32(TWO_PI)*u1;
    f32 cos2 = (1.f-u2) / (1.f+(alpha*alpha-1.f)*u2);
    f32 cos_theta = std::sqrt(max(0.f,cos2));
    f32 sin_theta = std::sqrt(max(0.f,1.f-cos2));
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}

// ─── Disk & triangle sampling ─────────────────────────────────────────────
// Concentric disk (low distortion)
[[nodiscard]] inline Vec2f sample_disk_concentric(f32 u1, f32 u2) noexcept {
    f32 sx=2*u1-1, sy=2*u2-1;
    if(sx==0&&sy==0) return {0,0};
    f32 r,theta;
    if(std::abs(sx)>std::abs(sy)){r=sx;theta=f32(PI/4)*(sy/sx);}
    else{r=sy;theta=f32(PI/2)-f32(PI/4)*(sx/sy);}
    return {r*std::cos(theta), r*std::sin(theta)};
}

// Triangle (barycentric)
[[nodiscard]] inline Vec2f sample_triangle(f32 u1, f32 u2) noexcept {
    f32 su1=std::sqrt(u1);
    return {1.f-su1, u2*su1};
}

// Cone sampling (within half-angle theta_max)
[[nodiscard]] inline Vec3f sample_cone(f32 u1, f32 u2, f32 cos_theta_max) noexcept {
    f32 cos_theta = lerp(cos_theta_max, 1.f, u1);
    f32 sin_theta = std::sqrt(max(0.f,1.f-cos_theta*cos_theta));
    f32 phi = f32(TWO_PI)*u2;
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}
[[nodiscard]] inline f32 pdf_cone(f32 cos_theta_max) noexcept {
    return f32(INV_TWO_PI) / (1.f-cos_theta_max);
}

// ─── Local frame (TBN from normal) ────────────────────────────────────────
struct LocalFrame {
    Vec3f T, B, N;
    LocalFrame() = default;
    explicit LocalFrame(Vec3f n) noexcept : N(n) {
        Vec3f up = (std::abs(n.x) > 0.9f) ? Vec3f{0,1,0} : Vec3f{1,0,0};
        T = cross(up, n).normalized();
        B = cross(n, T);
    }
    [[nodiscard]] Vec3f to_world(Vec3f v) const noexcept {
        return T*v.x + B*v.y + N*v.z;
    }
    [[nodiscard]] Vec3f to_local(Vec3f v) const noexcept {
        return {dot(v,T), dot(v,B), dot(v,N)};
    }
};

// ─── Multiple Importance Sampling ─────────────────────────────────────────
// Balance heuristic: w_i = n_i * p_i / sum_j n_j * p_j
[[nodiscard]] inline f32 mis_balance(f32 n1_p1, f32 n2_p2) noexcept {
    return n1_p1 / (n1_p1 + n2_p2 + 1e-30f);
}

// Power heuristic (beta=2 default)
[[nodiscard]] inline f32 mis_power(f32 p1, f32 p2, f32 beta=2.f) noexcept {
    f32 a=std::pow(p1,beta), b=std::pow(p2,beta);
    return a / (a + b + 1e-30f);
}

// ─── Path contribution estimator ─────────────────────────────────────────
struct PathVertex {
    Vec3f pos, normal, wo; // outgoing direction (toward eye)
    Vec3f throughput;      // accumulated weight
    f32   pdf;             // PDF that generated this vertex
    bool  is_specular;
};

// Russian roulette continuation probability
[[nodiscard]] inline f32 russian_roulette_prob(Vec3f throughput, f32 min_prob=0.05f) noexcept {
    f32 max_ch = max({throughput.x, throughput.y, throughput.z});
    return clamp(max_ch, min_prob, 1.f);
}

// ─── Phase functions (participating media) ───────────────────────────────
// Henyey-Greenstein phase function
[[nodiscard]] inline f32 henyey_greenstein(f32 cos_theta, f32 g) noexcept {
    f32 g2=g*g;
    f32 denom=1+g2-2*g*cos_theta;
    return f32(INV_FOUR_PI)*(1-g2)/(denom*std::sqrt(denom));
}

[[nodiscard]] inline Vec3f sample_hg(f32 u1, f32 u2, f32 g) noexcept {
    f32 cos_theta;
    if(std::abs(g)<1e-3f){
        cos_theta=1-2*u1;
    } else {
        f32 sqr=(1-g*g)/(1-g+2*g*u1);
        cos_theta=(1+g*g-sqr*sqr)/(2*g);
    }
    f32 sin_theta=std::sqrt(max(0.f,1-cos_theta*cos_theta));
    f32 phi=f32(TWO_PI)*u2;
    return {sin_theta*std::cos(phi), sin_theta*std::sin(phi), cos_theta};
}

// Rayleigh phase function
[[nodiscard]] inline f32 rayleigh_phase(f32 cos_theta) noexcept {
    return 3.f/(16.f*f32(PI))*(1+cos_theta*cos_theta);
}

// Mie scattering (Lorenz-Mie approximation — Cornette-Shanks)
[[nodiscard]] inline f32 cornette_shanks(f32 cos_theta, f32 g) noexcept {
    f32 g2=g*g;
    f32 num=3*(1-g2)*(1+cos_theta*cos_theta);
    f32 den=2*(2+g2)*std::pow(1+g2-2*g*cos_theta,1.5f);
    return f32(INV_FOUR_PI)*num/den;
}

// ─── Transmittance (Beer-Lambert) ────────────────────────────────────────
[[nodiscard]] inline Vec3f beer_lambert(Vec3f sigma_t, f32 t) noexcept {
    return {std::exp(-sigma_t.x*t), std::exp(-sigma_t.y*t), std::exp(-sigma_t.z*t)};
}

// Sample free-flight distance in homogeneous medium
[[nodiscard]] inline f32 sample_exponential(f32 sigma_t, f32 u) noexcept {
    return -std::log(max(1e-10f,1.f-u))/sigma_t;
}

// Woodcock tracking (null scattering, heterogeneous media)
// Returns sampled t or INF if escaped
template<typename DensityFn>
[[nodiscard]] inline f32 woodcock_tracking(
    Vec3f org, Vec3f dir, f32 sigma_t_max, f32 t_max,
    DensityFn&& density, PCG32& rng) noexcept
{
    f32 t=0;
    while(true){
        t+=sample_exponential(sigma_t_max, rng.next_f32());
        if(t>=t_max) return 1e30f;
        Vec3f p=org+dir*t;
        f32 sigma_at_p=density(p);
        if(rng.next_f32()<sigma_at_p/sigma_t_max) return t;
    }
}

// ─── Radiosity (form factors) ─────────────────────────────────────────────
// Analytic form factor between two parallel rectangles (special case)
[[nodiscard]] inline f64 form_factor_parallel_rect(f64 a,f64 b,f64 c) noexcept {
    // a,b = half-widths of rectangles, c = separation
    // From Howell's "Catalog of Radiation Configuration Factors"
    f64 X=a/c, Y=b/c;
    auto F=[](f64 x)->f64{ return std::sqrt(x*x+1)*std::atan(x)-x; };
    return (2.0/(PI*X*Y))*(F(X)+F(Y)-std::atan(std::sqrt(X*X+Y*Y)));
}

// Point-to-differential area form factor: cos(theta_i)*cos(theta_j)/(pi*r^2)
[[nodiscard]] inline f32 form_factor_point_to_area(
    Vec3f p, Vec3f n_p, Vec3f q, Vec3f n_q) noexcept
{
    Vec3f r=q-p; f32 d=r.length();
    if(d<1e-6f) return 0.f;
    Vec3f rn=r*(1.f/d);
    f32 cos_p=max(0.f,dot(rn,n_p));
    f32 cos_q=max(0.f,-dot(rn,n_q));
    return cos_p*cos_q/(f32(PI)*d*d);
}

// ─── Photon map (kNN density estimation helper) ────────────────────────────
struct Photon {
    Vec3f pos, power, dir;
};

// Photon density estimate at surface point p with normal n
[[nodiscard]] inline Vec3f estimate_irradiance(
    const std::vector<Photon>& photons,
    std::span<const int> nearby_ids,
    Vec3f p, Vec3f n, f32 radius) noexcept
{
    Vec3f sum{0,0,0};
    f32 inv_area = 1.f/(f32(PI)*radius*radius);
    for(int id:nearby_ids){
        const Photon& ph=photons[id];
        f32 cos_i=dot(n,-ph.dir);
        if(cos_i>0) sum=sum+ph.power*cos_i;
    }
    return sum*inv_area;
}

// ─── Emission sampling (area lights) ──────────────────────────────────────
// Sample point on triangle light source
struct LightSample {
    Vec3f pos, normal;
    f32   pdf; // per unit area
};

[[nodiscard]] inline LightSample sample_triangle_light(
    Vec3f v0, Vec3f v1, Vec3f v2, f32 u1, f32 u2) noexcept
{
    Vec2f bary = sample_triangle(u1,u2);
    Vec3f p = v0*(1.f-bary.x-bary.y) + v1*bary.x + v2*bary.y;
    Vec3f n = cross(v1-v0, v2-v0);
    f32 area = n.length()*0.5f;
    return {p, n*(1.f/(area*2.f+1e-10f)), area>1e-10f?1.f/area:0.f};
}

// ─── Low-discrepancy sequences ────────────────────────────────────────────
// Radical inverse base b
[[nodiscard]] inline f64 radical_inverse(u64 n, u64 base) noexcept {
    f64 val=0, invBase=1.0/base, invBi=invBase;
    while(n>0){ val+=f64(n%base)*invBi; n/=base; invBi*=invBase; }
    return val;
}

// Halton sequence (base 2, base 3 for 2D)
[[nodiscard]] inline Vec2f halton2d(u64 n) noexcept {
    return {f32(radical_inverse(n,2)), f32(radical_inverse(n,3))};
}

// Sobol sequence (first 2 dimensions)
[[nodiscard]] inline Vec2f sobol2d(u64 n) noexcept {
    // Direction numbers for dim 1 (Van der Corput) and dim 2
    u64 x=0, y=0, v1=0x80000000u, v2=0x80000000u;
    for(u64 i=n;i;i>>=1,v1>>=1,v2>>=1) if(i&1){x^=v1;y^=v2;}
    return {f32(x)*2.3283064365e-10f, f32(y)*2.3283064365e-10f};
}

// Blue noise via Stochastic Best Candidate (simplified 2D)
// Returns n points in [0,1)^2 with good blue-noise properties
[[nodiscard]] inline std::vector<Vec2f> blue_noise_2d(int n, u32 seed=42) noexcept {
    PCG32 rng(seed);
    std::vector<Vec2f> pts; pts.reserve(n);
    for(int i=0;i<n;++i){
        Vec2f best{rng.next_f32(),rng.next_f32()};
        f32 best_dist=0;
        int candidates=max(1,int(std::sqrt(f32(n))));
        for(int c=0;c<candidates;++c){
            Vec2f cand{rng.next_f32(),rng.next_f32()};
            f32 min_d=1e30f;
            for(auto& p:pts){
                f32 dx=cand.x-p.x,dy=cand.y-p.y;
                min_d=min(min_d,dx*dx+dy*dy);
            }
            if(min_d>best_dist){best_dist=min_d;best=cand;}
        }
        pts.push_back(best);
    }
    return pts;
}

} // namespace qs::math::render
