#pragma once
// qs_math/rendering/path_tracing.hpp — Ray, intersection, sampling, MIS, denoising math
#include "../arithmetic/scalar_types.hpp"
#include "pbr_math.hpp"
#include <cmath>
#include <array>
#include <optional>

namespace qs::math {

// ─── Ray ─────────────────────────────────────────────────────────────────────
struct Ray { R3 origin,dir; f32 tmin{1e-4f},tmax{1e9f}; };

[[nodiscard]] constexpr R3 ray_at(const Ray& r, f32 t) noexcept {
    return r.origin+r.dir*t;
}

// ─── Intersection records ─────────────────────────────────────────────────────
struct HitRecord {
    f32 t{-1}; R3 p,n; bool front{true};
    [[nodiscard]] constexpr bool valid() const noexcept {return t>0;}
};

// ─── Sphere intersection ──────────────────────────────────────────────────────
[[nodiscard]] constexpr std::optional<HitRecord> intersect_sphere(
    const Ray& r, R3 center, f32 radius) noexcept
{
    R3 oc=r.origin-center;
    f32 a=r.dir.dot(r.dir), hb=oc.dot(r.dir), c=oc.dot(oc)-radius*radius;
    f32 disc=hb*hb-a*c;
    if(disc<0) return std::nullopt;
    f32 sq=std::sqrt(disc);
    f32 t=(-hb-sq)/a;
    if(t<r.tmin||t>r.tmax){t=(-hb+sq)/a;if(t<r.tmin||t>r.tmax) return std::nullopt;}
    R3 p=ray_at(r,t);
    R3 n=(p-center)*(1.0f/radius);
    bool front=r.dir.dot(n)<0;
    return HitRecord{t,p,front?n:R3{-n.x,-n.y,-n.z},front};
}

// ─── Triangle intersection (Möller-Trumbore) ──────────────────────────────────
[[nodiscard]] constexpr std::optional<HitRecord> intersect_triangle(
    const Ray& r, R3 A, R3 B, R3 C) noexcept
{
    R3 e1=B-A, e2=C-A;
    R3 h={r.dir.y*e2.z-r.dir.z*e2.y, r.dir.z*e2.x-r.dir.x*e2.z, r.dir.x*e2.y-r.dir.y*e2.x};
    f32 det=e1.dot(h);
    if(std::abs(det)<1e-7f) return std::nullopt;
    f32 inv=1/det;
    R3 s=r.origin-A;
    f32 u=s.dot(h)*inv;
    if(u<0||u>1) return std::nullopt;
    R3 q={s.y*e1.z-s.z*e1.y,s.z*e1.x-s.x*e1.z,s.x*e1.y-s.y*e1.x};
    f32 v=r.dir.dot(q)*inv;
    if(v<0||u+v>1) return std::nullopt;
    f32 t=e2.dot(q)*inv;
    if(t<r.tmin||t>r.tmax) return std::nullopt;
    R3 n={e1.y*e2.z-e1.z*e2.y,e1.z*e2.x-e1.x*e2.z,e1.x*e2.y-e1.y*e2.x};
    n=n.norm(); bool front=r.dir.dot(n)<0;
    return HitRecord{t,ray_at(r,t),front?n:R3{-n.x,-n.y,-n.z},front};
}

// AABB intersection (slab method)
[[nodiscard]] constexpr bool intersect_aabb(const Ray& r,R3 lo,R3 hi) noexcept {
    auto invd=[](f32 d){return std::abs(d)>1e-7f?1.0f/d:1e30f;};
    f32 idx=invd(r.dir.x),idy=invd(r.dir.y),idz=invd(r.dir.z);
    f32 tx0=(lo.x-r.origin.x)*idx,tx1=(hi.x-r.origin.x)*idx;
    f32 ty0=(lo.y-r.origin.y)*idy,ty1=(hi.y-r.origin.y)*idy;
    f32 tz0=(lo.z-r.origin.z)*idz,tz1=(hi.z-r.origin.z)*idz;
    if(tx0>tx1)std::swap(tx0,tx1);if(ty0>ty1)std::swap(ty0,ty1);if(tz0>tz1)std::swap(tz0,tz1);
    f32 tmin=std::max({tx0,ty0,tz0,r.tmin});
    f32 tmax=std::min({tx1,ty1,tz1,r.tmax});
    return tmin<=tmax;
}

// ─── Sampling utilities ───────────────────────────────────────────────────────
// Uniform sphere sampling
[[nodiscard]] constexpr R3 sample_sphere(f32 xi1,f32 xi2) noexcept {
    f32 cos_theta=1-2*xi1, sin_theta=std::sqrt(1-cos_theta*cos_theta);
    f32 phi=2*k_pi_f*xi2;
    return {sin_theta*std::cos(phi),cos_theta,sin_theta*std::sin(phi)};
}
[[nodiscard]] constexpr f32 pdf_sphere() noexcept {return 1.0f/(4*k_pi_f);}

// Uniform disk sampling (Shirley concentric)
[[nodiscard]] constexpr std::pair<f32,f32> sample_disk(f32 xi1,f32 xi2) noexcept {
    f32 a=2*xi1-1,b=2*xi2-1;
    if(a==0&&b==0) return {0,0};
    f32 r,phi;
    if(std::abs(a)>std::abs(b)){r=a;phi=k_pi_f*0.25f*b/a;}
    else{r=b;phi=k_pi_f*(0.5f-0.25f*a/b);}
    return {r*std::cos(phi),r*std::sin(phi)};
}

// Orthonormal basis from normal
[[nodiscard]] constexpr std::pair<R3,R3> build_onb(R3 n) noexcept {
    R3 up=std::abs(n.x)<0.9f?R3{1,0,0}:R3{0,1,0};
    R3 t={n.y*up.z-n.z*up.y,n.z*up.x-n.x*up.z,n.x*up.y-n.y*up.x};
    t=t.norm();
    R3 b={n.y*t.z-n.z*t.y,n.z*t.x-n.x*t.z,n.x*t.y-n.y*t.x};
    return {t,b};
}

// Transform from local to world (n=normal, t=tangent, b=bitangent)
[[nodiscard]] constexpr R3 local_to_world(R3 v, R3 n, R3 t, R3 b) noexcept {
    return t*v.x+n*v.y+b*v.z;
}

// ─── Multiple importance sampling weight (balance heuristic) ─────────────────
[[nodiscard]] constexpr f32 mis_balance(f32 pdf_a, f32 pdf_b) noexcept {
    return pdf_a/(pdf_a+pdf_b+1e-7f);
}
// Power heuristic (β=2)
[[nodiscard]] constexpr f32 mis_power(f32 pdf_a, f32 pdf_b) noexcept {
    f32 a2=pdf_a*pdf_a, b2=pdf_b*pdf_b;
    return a2/(a2+b2+1e-7f);
}

// ─── Russian roulette ─────────────────────────────────────────────────────────
[[nodiscard]] constexpr f32 russian_roulette_prob(R3 throughput) noexcept {
    return std::min(0.95f,std::max(throughput.x,std::max(throughput.y,throughput.z)));
}

// ─── Spectral upsampling (Hable/Karis tone mapping) ───────────────────────────
[[nodiscard]] constexpr f32 aces_film(f32 x) noexcept {
    return (x*(2.51f*x+0.03f))/(x*(2.43f*x+0.59f)+0.14f);
}
[[nodiscard]] constexpr R3 aces_film_rgb(R3 c) noexcept {
    return {aces_film(c.x),aces_film(c.y),aces_film(c.z)};
}
[[nodiscard]] constexpr f32 reinhard(f32 x) noexcept {return x/(1+x);}
[[nodiscard]] constexpr R3 reinhard_rgb(R3 c) noexcept {
    return {reinhard(c.x),reinhard(c.y),reinhard(c.z)};
}
// Reinhard-Jodie (luminance-based)
[[nodiscard]] constexpr R3 reinhard_jodie(R3 c) noexcept {
    f32 lum=0.2126f*c.x+0.7152f*c.y+0.0722f*c.z;
    R3 tv=R3{c.x/(1+c.x),c.y/(1+c.y),c.z/(1+c.z)};
    f32 l=reinhard(lum);
    return {c.x*(l/lum)*(1-tv.x)+tv.x,c.y*(l/lum)*(1-tv.y)+tv.y,c.z*(l/lum)*(1-tv.z)+tv.z};
}

// ─── Light sampling ───────────────────────────────────────────────────────────
// Area light: power = emittance * area * π (for diffuse)
[[nodiscard]] constexpr f32 area_light_power(f32 emittance, f32 area) noexcept {
    return emittance*area*k_pi_f;
}
// Solid angle subtended by sphere light
[[nodiscard]] constexpr f32 sphere_light_solid_angle(f32 radius, f32 dist) noexcept {
    if(dist<=radius) return 2*k_pi_f;
    f32 sin2=radius*radius/(dist*dist);
    return 2*k_pi_f*(1-std::sqrt(1-sin2));
}
// Point light attenuation (physically based: 1/r²)
[[nodiscard]] constexpr f32 point_light_attenuation(f32 dist) noexcept {
    return 1.0f/(dist*dist+1e-4f);
}

// ─── Denoising math (A-Trous wavelet filter weights) ─────────────────────────
[[nodiscard]] constexpr f32 atrous_weight(f32 depth_diff,f32 normal_diff,
                                            f32 color_diff, f32 phi_d=1.f,
                                            f32 phi_n=0.5f,f32 phi_c=0.5f) noexcept {
    return std::exp(-depth_diff/phi_d-normal_diff/phi_n-color_diff/phi_c);
}
// SVGF temporal blend alpha
[[nodiscard]] constexpr f32 svgf_alpha(f32 history_len, f32 alpha_min=0.05f) noexcept {
    return std::max(alpha_min,1.0f/history_len);
}
} // namespace qs::math
