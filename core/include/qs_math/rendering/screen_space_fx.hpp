#pragma once
// qs_math/screen_space_fx.hpp — SSR, SSSR, bloom, lens flare, chromatic aberration (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>

namespace qs {

// ─── Screen-space reflections ─────────────────────────────────────────────────

struct SSRParams {
    u32  max_steps{64};
    f32  step_size{0.05f};
    f32  thickness{0.1f};
    f32  max_distance{20.f};
    f32  fade_start{0.8f};   // fade out near screen edge
    f32  roughness_cutoff{0.4f};
    bool use_binary_search{true};
    u32  binary_steps{8};
};

// Ray march in screen space; returns UV if hit, else nullopt
[[nodiscard]] inline std::optional<Vec2f> ssr_march(
    const Vec3f& ray_origin_vs,  // view-space origin
    const Vec3f& ray_dir_vs,     // view-space direction
    const Mat4f& projection,
    const SSRParams& p,
    auto depth_sample_fn)        // depth_sample_fn(Vec2f uv) -> f32 (view-space depth)
{
    Vec3f pos=ray_origin_vs;
    f32 step=p.step_size;

    for(u32 i=0;i<p.max_steps;++i){
        pos=pos+ray_dir_vs*step;
        step*=1.05f; // exponential step growth

        // Project to screen
        Vec4f clip=projection*Vec4f{pos.x,pos.y,pos.z,1.f};
        if(clip.w<0.001f) break;
        Vec2f uv{clip.x/clip.w*0.5f+0.5f, clip.y/clip.w*0.5f+0.5f};
        if(uv.x<0||uv.x>1||uv.y<0||uv.y>1) break;

        f32 scene_depth=depth_sample_fn(uv);
        f32 ray_depth=pos.z;

        if(ray_depth < scene_depth && ray_depth > scene_depth-p.thickness){
            if(!p.use_binary_search) return uv;
            // Binary search refinement
            Vec3f lo=pos-ray_dir_vs*step, hi=pos;
            for(u32 b=0;b<p.binary_steps;++b){
                Vec3f mid=(lo+hi)*0.5f;
                Vec4f mc=projection*Vec4f{mid.x,mid.y,mid.z,1.f};
                Vec2f muv{mc.x/mc.w*0.5f+0.5f,mc.y/mc.w*0.5f+0.5f};
                f32 md=depth_sample_fn(muv);
                if(mid.z<md) hi=mid; else lo=mid;
            }
            Vec4f fc=projection*Vec4f{(lo.x+hi.x)*0.5f,(lo.y+hi.y)*0.5f,(lo.z+hi.z)*0.5f,1.f};
            return Vec2f{fc.x/fc.w*0.5f+0.5f,fc.y/fc.w*0.5f+0.5f};
        }
    }
    return std::nullopt;
}

// Screen-edge fade for SSR
[[nodiscard]] inline f32 ssr_screen_fade(Vec2f uv, f32 fade_start=0.8f) {
    Vec2f edge=Vec2f{1.f,1.f}-abs(uv*2.f-Vec2f{1.f,1.f});
    f32 fade=std::min(edge.x,edge.y);
    return std::clamp((fade-(1.f-fade_start))/fade_start,0.f,1.f);
}

// ─── Bloom ────────────────────────────────────────────────────────────────────

struct BloomParams {
    f32 threshold{1.f};
    f32 knee{0.5f};         // soft knee width
    f32 intensity{1.f};
    f32 scatter{0.7f};      // energy scatter between mips
    u32 num_passes{5};
};

// Soft threshold for bloom prefilter
[[nodiscard]] inline Vec3f bloom_prefilter(const Vec3f& color, const BloomParams& p) {
    f32 lum=0.2126f*color.x+0.7152f*color.y+0.0722f*color.z;
    f32 knee_lo=p.threshold-p.knee;
    f32 knee_hi=p.threshold+p.knee;
    f32 rq=std::clamp(lum-knee_lo,0.f,2.f*p.knee);
    f32 weight=(rq*rq)/(4.f*p.knee+0.00001f);
    f32 w=std::max(weight,lum-p.threshold)/(std::max(lum,0.00001f));
    return color*w;
}

// Kawase blur kernel offsets (for one pass)
[[nodiscard]] inline std::array<Vec2f,4> kawase_offsets(u32 pass, Vec2f texel_size) {
    f32 o=f32(pass)+0.5f;
    return {Vec2f{-o,-o}*texel_size, Vec2f{o,-o}*texel_size,
            Vec2f{-o, o}*texel_size, Vec2f{o, o}*texel_size};
}

// Dual Kawase blur sample weight (equal for all 4 taps)
[[nodiscard]] inline f32 kawase_weight() { return 0.25f; }

// Tent filter weights for upsample (3x3)
[[nodiscard]] inline f32 tent_weight(Vec2f uv_offset) {
    // Bilinear weight: distance from center
    f32 fx=std::max(0.f,1.f-std::abs(uv_offset.x));
    f32 fy=std::max(0.f,1.f-std::abs(uv_offset.y));
    return fx*fy;
}

// ─── Lens flare ───────────────────────────────────────────────────────────────

struct LensFlareElement {
    f32  offset{0.f};     // along flare axis (-1 to 1)
    f32  size{0.1f};      // screen-space radius
    Vec3f color{1,0.8f,0.3f};
    f32  intensity{1.f};
    u32  blades{6};       // polygon blade count for aperture
};

struct LensFlare {
    std::vector<LensFlareElement> elements;
    Vec2f sun_screen_pos{};
    f32   base_intensity{1.f};
    f32   occlusion{0.f};  // 0=fully visible, 1=fully occluded

    void build_standard() {
        elements.clear();
        elements.push_back({0.0f,0.04f,{1.0f,0.9f,0.7f},2.0f,6});
        elements.push_back({0.3f,0.02f,{0.8f,0.7f,1.0f},1.0f,5});
        elements.push_back({0.6f,0.06f,{1.0f,0.6f,0.3f},0.8f,6});
        elements.push_back({1.0f,0.08f,{0.6f,0.8f,1.0f},0.6f,7});
        elements.push_back({1.4f,0.03f,{1.0f,1.0f,0.6f},0.4f,5});
    }

    [[nodiscard]] Vec2f element_screen_pos(const LensFlareElement& e) const {
        Vec2f axis=Vec2f{0.5f,0.5f}-sun_screen_pos;
        return sun_screen_pos+axis*e.offset;
    }
};

// ─── Chromatic aberration ────────────────────────────────────────────────────

struct ChromaticAberrationParams {
    f32 strength{0.005f};       // pixel offset per channel
    f32 radial_power{2.f};      // distortion falloff
    bool barrel_distortion{true};
};

[[nodiscard]] inline Vec2f ca_sample_offset(
    Vec2f uv, i32 channel, const ChromaticAberrationParams& p)
{
    Vec2f center=uv-Vec2f{0.5f,0.5f};
    f32 r=length(center);
    f32 dist=std::pow(r,p.radial_power)*p.strength*f32(channel-1);
    return uv+normalize(center)*dist;
}

// ─── Depth of field (bokeh) ──────────────────────────────────────────────────

struct BokehParams {
    f32 focus_dist{5.f};
    f32 aperture{0.1f};
    f32 max_coc{0.02f};   // max circle of confusion (screen-space fraction)
    u32 blade_count{6};   // 0=circle
    f32 blade_rotation{0.f};
};

[[nodiscard]] inline f32 coc_size(f32 depth, const BokehParams& p) {
    return std::min(p.max_coc, p.aperture*std::abs(depth-p.focus_dist)/depth);
}

// Sample point on bokeh disk (polygonal or circular)
[[nodiscard]] inline Vec2f bokeh_sample(f32 u1, f32 u2, const BokehParams& p) {
    if(p.blade_count==0||p.blade_count>16){
        f32 r=std::sqrt(u1);
        return {r*std::cos(k_tau*u2), r*std::sin(k_tau*u2)};
    }
    u32 blade=u32(u1*f32(p.blade_count));
    f32 t=(u1*f32(p.blade_count)-f32(blade));
    f32 a0=k_tau*f32(blade)/f32(p.blade_count)+p.blade_rotation;
    f32 a1=k_tau*f32(blade+1)/f32(p.blade_count)+p.blade_rotation;
    f32 r=std::sqrt(u2);
    Vec2f tri{(1.f-t)*std::cos(a0)+t*std::cos(a1),(1.f-t)*std::sin(a0)+t*std::sin(a1)};
    return tri*r;
}

// ─── Vignette ────────────────────────────────────────────────────────────────

[[nodiscard]] inline f32 vignette(Vec2f uv, f32 inner=0.5f, f32 outer=1.f, f32 power=2.f) {
    Vec2f c=uv-Vec2f{0.5f,0.5f};
    f32 r=length(c)*std::sqrt(2.f);
    return std::pow(std::clamp((outer-r)/(outer-inner),0.f,1.f),power);
}

// ─── Film grain ───────────────────────────────────────────────────────────────

[[nodiscard]] inline f32 film_grain(Vec2f uv, f32 time, f32 strength=0.05f) {
    // Pseudo-random noise at pixel
    f32 n=std::sin(dot(uv,Vec2f{12.9898f,78.233f})+time)*43758.5453f;
    n=n-std::floor(n);
    return (n-0.5f)*strength;
}

// ─── Tonemapping ──────────────────────────────────────────────────────────────

[[nodiscard]] inline Vec3f tonemap_reinhard(const Vec3f& hdr, f32 exposure=1.f) {
    Vec3f mapped=hdr*exposure;
    return {mapped.x/(1.f+mapped.x),mapped.y/(1.f+mapped.y),mapped.z/(1.f+mapped.z)};
}

[[nodiscard]] inline Vec3f tonemap_aces(const Vec3f& hdr, f32 exposure=1.f) {
    Vec3f x=hdr*exposure;
    f32 a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
    auto f=[&](f32 v)->f32{return std::clamp((v*(a*v+b))/(v*(c*v+d)+e),0.f,1.f);};
    return {f(x.x),f(x.y),f(x.z)};
}

[[nodiscard]] inline Vec3f tonemap_filmic(const Vec3f& hdr, f32 exposure=1.f) {
    Vec3f x=Vec3f{hdr.x*exposure,hdr.y*exposure,hdr.z*exposure};
    auto f=[](f32 v)->f32{
        f32 q=std::max(0.f,v-0.004f);
        return (q*(6.2f*q+0.5f))/(q*(6.2f*q+1.7f)+0.06f);};
    return {f(x.x),f(x.y),f(x.z)};
}

[[nodiscard]] inline Vec3f tonemap_lottes(const Vec3f& hdr, f32 a=1.6f, f32 d=0.977f,
    f32 hdr_max=8.f, f32 mid_in=0.18f, f32 mid_out=0.267f)
{
    auto f=[&](f32 x)->f32{
        f32 b=(-std::pow(mid_in,a)+std::pow(hdr_max,a)*mid_out)/
               ((std::pow(hdr_max,a*d)-std::pow(mid_in,a*d))*mid_out);
        f32 c=std::pow(hdr_max,a*d)*std::pow(mid_in,a)/
               ((std::pow(hdr_max,a*d)-std::pow(mid_in,a*d))*mid_out);
        return std::pow(x,a)/(std::pow(x,a*d)*b+c);};
    return {f(hdr.x),f(hdr.y),f(hdr.z)};
}

// sRGB gamma encode/decode
[[nodiscard]] inline f32 linear_to_srgb(f32 v) {
    return v<=0.0031308f?v*12.92f:1.055f*std::pow(v,1.f/2.4f)-0.055f;
}
[[nodiscard]] inline f32 srgb_to_linear(f32 v) {
    return v<=0.04045f?v/12.92f:std::pow((v+0.055f)/1.055f,2.4f);
}
[[nodiscard]] inline Vec3f linear_to_srgb_v(const Vec3f& v){
    return {linear_to_srgb(v.x),linear_to_srgb(v.y),linear_to_srgb(v.z)};}
[[nodiscard]] inline Vec3f srgb_to_linear_v(const Vec3f& v){
    return {srgb_to_linear(v.x),srgb_to_linear(v.y),srgb_to_linear(v.z)};}

} // namespace qs
