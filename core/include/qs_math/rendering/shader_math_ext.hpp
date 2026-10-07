#pragma once
// qs_math/shader_math_ext.hpp — Parallax, detail maps, triplanar, hash noise GLSL analogs (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <cmath>
#include <algorithm>
#include <array>

namespace qs {

// ─── UV transformations ───────────────────────────────────────────────────────

[[nodiscard]] inline Vec2f uv_rotate(Vec2f uv, f32 angle) {
    f32 c=std::cos(angle), s=std::sin(angle);
    Vec2f c2=uv-Vec2f{0.5f,0.5f};
    return Vec2f{c*c2.x-s*c2.y+0.5f, s*c2.x+c*c2.y+0.5f};
}

[[nodiscard]] inline Vec2f uv_tile(Vec2f uv, Vec2f scale) {
    return {uv.x*scale.x,uv.y*scale.y};
}

[[nodiscard]] inline Vec2f uv_distort(Vec2f uv, f32 strength, f32 time) {
    f32 dx=std::sin(uv.y*10.f+time)*strength;
    f32 dy=std::cos(uv.x*10.f+time)*strength;
    return {uv.x+dx,uv.y+dy};
}

// ─── Triplanar mapping ────────────────────────────────────────────────────────

struct TriplanarWeights {
    f32 x{},y{},z{};
};

[[nodiscard]] inline TriplanarWeights triplanar_weights(const Vec3f& normal, f32 blend_sharpness=4.f) {
    Vec3f w{std::abs(normal.x),std::abs(normal.y),std::abs(normal.z)};
    w.x=std::pow(w.x,blend_sharpness);
    w.y=std::pow(w.y,blend_sharpness);
    w.z=std::pow(w.z,blend_sharpness);
    f32 sum=w.x+w.y+w.z+1e-8f;
    return {w.x/sum,w.y/sum,w.z/sum};
}

template<typename T, typename SampleFn>
[[nodiscard]] T triplanar_sample(
    const Vec3f& world_pos, const Vec3f& normal,
    SampleFn sample, // sample(Vec2f uv) -> T
    f32 scale=1.f, f32 blend=4.f)
{
    auto w=triplanar_weights(normal,blend);
    T tx=sample(Vec2f{world_pos.z*scale,world_pos.y*scale});
    T ty=sample(Vec2f{world_pos.x*scale,world_pos.z*scale});
    T tz=sample(Vec2f{world_pos.x*scale,world_pos.y*scale});
    return tx*w.x+ty*w.y+tz*w.z;
}

// ─── Detail normals ───────────────────────────────────────────────────────────

// Blend normals: base + detail (UDN method)
[[nodiscard]] inline Vec3f blend_normals_udn(const Vec3f& base, const Vec3f& detail) {
    return normalize(Vec3f{base.x+detail.x, base.y+detail.y, base.z*detail.z});
}

// Whiteout blending (preserves both normals better)
[[nodiscard]] inline Vec3f blend_normals_whiteout(const Vec3f& n1, const Vec3f& n2) {
    return normalize(Vec3f{n1.x+n2.x, n1.y+n2.y, n1.z*n2.z});
}

// ─── Procedural hash / noise (shader-style, no lookup tables) ─────────────────

[[nodiscard]] inline f32 hash11(f32 p) {
    p=std::fmod(p*127.1f,289.f);
    return std::fmod((p*p*p+p*p)*17.f,1.f);
}

[[nodiscard]] inline f32 hash21(Vec2f p) {
    Vec2f q{std::fmod(p.x*127.1f+p.y*311.7f,289.f),
             std::fmod(p.x*269.5f+p.y*183.3f,289.f)};
    return std::fmod((std::sin(q.x*q.x*q.y+q.x*q.y)*43758.5453f),1.f);
}

[[nodiscard]] inline Vec2f hash22(Vec2f p) {
    Vec2f q{std::fmod(dot(p,Vec2f{127.1f,311.7f}),289.f),
             std::fmod(dot(p,Vec2f{269.5f,183.3f}),289.f)};
    f32 a=std::sin(q.x*q.x*q.y+q.x*q.y)*43758.5453f;
    f32 b=std::sin(q.y*q.x*q.x+q.y*q.x)*43758.5453f;
    return {a-std::floor(a),b-std::floor(b)};
}

[[nodiscard]] inline f32 hash31(const Vec3f& p) {
    Vec3f q{std::fmod(dot(p,Vec3f{127.1f,311.7f,74.7f}),289.f),
             std::fmod(dot(p,Vec3f{269.5f,183.3f,246.1f}),289.f),
             std::fmod(dot(p,Vec3f{113.5f,271.9f,124.6f}),289.f)};
    return std::fmod(std::sin(dot(q,Vec3f{q.y,q.z,q.x}))*43758.5453f,1.f);
}

// ─── Value noise ──────────────────────────────────────────────────────────────

[[nodiscard]] inline f32 value_noise_2d(Vec2f uv) {
    Vec2f i{std::floor(uv.x),std::floor(uv.y)};
    Vec2f f{uv.x-i.x,uv.y-i.y};
    Vec2f u{f.x*f.x*(3.f-2.f*f.x),f.y*f.y*(3.f-2.f*f.y)};
    f32 a=hash21(i);
    f32 b=hash21({i.x+1.f,i.y});
    f32 c=hash21({i.x,i.y+1.f});
    f32 d=hash21({i.x+1.f,i.y+1.f});
    return a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y;
}

// ─── Cellular / Worley noise ──────────────────────────────────────────────────

[[nodiscard]] inline std::pair<f32,f32> worley_noise_2d(Vec2f uv, f32 scale=1.f) {
    Vec2f suv=Vec2f{uv.x*scale,uv.y*scale};
    Vec2f cell{std::floor(suv.x),std::floor(suv.y)};
    f32 d1=1e10f,d2=1e10f;
    for(i32 dy=-1;dy<=1;++dy)
    for(i32 dx=-1;dx<=1;++dx){
        Vec2f nc{cell.x+f32(dx),cell.y+f32(dy)};
        Vec2f jitter=hash22(nc);
        Vec2f feature{nc.x+jitter.x,nc.y+jitter.y};
        f32 d=length(suv-feature);
        if(d<d1){d2=d1;d1=d;}
        else if(d<d2) d2=d;
    }
    return {d1,d2};
}

// ─── Distance functions for SDF shaders ──────────────────────────────────────

[[nodiscard]] inline f32 sdf_roundbox_2d(Vec2f p, Vec2f b, f32 r) {
    Vec2f q{std::abs(p.x)-b.x,std::abs(p.y)-b.y};
    return length(Vec2f{std::max(q.x,0.f),std::max(q.y,0.f)})
          +std::min(std::max(q.x,q.y),0.f)-r;
}

[[nodiscard]] inline f32 sdf_segment_2d(Vec2f p, Vec2f a, Vec2f b) {
    Vec2f pa=Vec2f{p.x-a.x,p.y-a.y},ba=Vec2f{b.x-a.x,b.y-a.y};
    f32 h=std::clamp(dot(pa,ba)/dot(ba,ba),0.f,1.f);
    Vec2f d{pa.x-ba.x*h,pa.y-ba.y*h};
    return std::sqrt(d.x*d.x+d.y*d.y);
}

[[nodiscard]] inline f32 sdf_polygon_2d(const Vec2f* verts, u32 n, Vec2f p) {
    f32 d=dot(Vec2f{p.x-verts[0].x,p.y-verts[0].y},Vec2f{p.x-verts[0].x,p.y-verts[0].y});
    f32 s=1.f;
    for(u32 i=0,j=n-1;i<n;j=i++){
        Vec2f e=Vec2f{verts[j].x-verts[i].x,verts[j].y-verts[i].y};
        Vec2f w=Vec2f{p.x-verts[i].x,p.y-verts[i].y};
        Vec2f b2=Vec2f{w.x-e.x*std::clamp(dot(w,e)/dot(e,e),0.f,1.f),
                       w.y-e.y*std::clamp(dot(w,e)/dot(e,e),0.f,1.f)};
        d=std::min(d,dot(b2,b2));
        bool c1=p.y>=verts[i].y,c2=p.y<verts[j].y,c3=e.x*w.y>e.y*w.x;
        if((c1&&c2&&c3)||(!c1&&!c2&&!c3)) s=-s;
    }
    return s*std::sqrt(d);
}

// ─── PBR utility functions ────────────────────────────────────────────────────

// Specular BRDF lobe parameters
[[nodiscard]] inline f32 roughness_to_alpha(f32 roughness) { return roughness*roughness; }
[[nodiscard]] inline f32 alpha_to_roughness(f32 alpha) { return std::sqrt(alpha); }

// Metallic-roughness to specular-glossiness
[[nodiscard]] inline Vec3f metallic_to_specular(const Vec3f& albedo, f32 metallic) {
    static constexpr Vec3f DIELECTRIC_F0{0.04f,0.04f,0.04f};
    Vec3f metallic_f0=albedo;
    return DIELECTRIC_F0*(1.f-metallic)+metallic_f0*metallic;
}

[[nodiscard]] inline Vec3f specular_to_albedo(const Vec3f& diffuse, const Vec3f& specular) {
    f32 max_spec=std::max({specular.x,specular.y,specular.z});
    if(max_spec>0.04f){
        // Metallic
        return {diffuse.x+specular.x,diffuse.y+specular.y,diffuse.z+specular.z};
    }
    return diffuse;
}

// Anisotropic roughness
[[nodiscard]] inline std::pair<f32,f32> anisotropic_roughness(
    f32 roughness, f32 anisotropy)
{
    f32 r=roughness*roughness;
    f32 at=r*(1.f+anisotropy);
    f32 ab=r*(1.f-anisotropy);
    return {at,ab};
}

// ─── Halftone / dithering patterns ───────────────────────────────────────────

[[nodiscard]] inline f32 bayer8x8(Vec2f screen_pos) {
    static constexpr f32 bayer8[64]={
        0,32, 8,40, 2,34,10,42,
       48,16,56,24,50,18,58,26,
       12,44, 4,36,14,46, 6,38,
       60,28,52,20,62,30,54,22,
        3,35,11,43, 1,33, 9,41,
       51,19,59,27,49,17,57,25,
       15,47, 7,39,13,45, 5,37,
       63,31,55,23,61,29,53,21};
    u32 x=u32(screen_pos.x)&7, y=u32(screen_pos.y)&7;
    return (bayer8[y*8+x]+0.5f)/64.f;
}

[[nodiscard]] inline f32 ordered_dither(f32 value, Vec2f screen_pos) {
    return value>bayer8x8(screen_pos)?1.f:0.f;
}

// Blue noise approximation via Vlachos (used in temporal AA)
[[nodiscard]] inline Vec2f blue_noise_offset(Vec2f screen_pos, u32 frame) {
    f32 r1=std::fmod(0.75487766624f*f32(frame)+0.5f,1.f);
    f32 r2=std::fmod(0.56984029099f*f32(frame)+0.5f,1.f);
    return {r1,r2};
}

} // namespace qs
