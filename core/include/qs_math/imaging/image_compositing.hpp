#pragma once
// image_compositing.hpp — Porter-Duff compositing, all blend modes,
//                          layer math, mask operations, keying
#include "vec4.hpp"
#include "color.hpp"
#include "scalar_math.hpp"

namespace qs::math::compositing {

// ---------------------------------------------------------------------------
// Porter-Duff compositing operators
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec4f porter_duff_over(Vec4f src, Vec4f dst) noexcept {
    f32 a_out=src.w+dst.w*(1.f-src.w);
    if(a_out<EPS_F32) return {0.f,0.f,0.f,0.f};
    Vec3f rgb=(Vec3f{src.x,src.y,src.z}*src.w+Vec3f{dst.x,dst.y,dst.z}*dst.w*(1.f-src.w))*(1.f/a_out);
    return {rgb.x,rgb.y,rgb.z,a_out};
}
[[nodiscard]] constexpr Vec4f porter_duff_under(Vec4f src, Vec4f dst) noexcept { return porter_duff_over(dst,src); }
[[nodiscard]] constexpr Vec4f porter_duff_in(Vec4f src, Vec4f dst) noexcept {
    return {src.x,src.y,src.z,src.w*dst.w};
}
[[nodiscard]] constexpr Vec4f porter_duff_out(Vec4f src, Vec4f dst) noexcept {
    return {src.x,src.y,src.z,src.w*(1.f-dst.w)};
}
[[nodiscard]] constexpr Vec4f porter_duff_atop(Vec4f src, Vec4f dst) noexcept {
    Vec3f rgb=Vec3f{src.x,src.y,src.z}*src.w*(1.f-dst.w)+Vec3f{dst.x,dst.y,dst.z}*dst.w*src.w;
    if(dst.w<EPS_F32) return {0.f,0.f,0.f,0.f};
    rgb=rgb*(1.f/dst.w);
    return {rgb.x,rgb.y,rgb.z,dst.w};
}
[[nodiscard]] constexpr Vec4f porter_duff_xor(Vec4f src, Vec4f dst) noexcept {
    f32 ao=src.w*(1.f-dst.w)+dst.w*(1.f-src.w);
    if(ao<EPS_F32) return {0.f,0.f,0.f,0.f};
    Vec3f rgb=(Vec3f{src.x,src.y,src.z}*src.w*(1.f-dst.w)+Vec3f{dst.x,dst.y,dst.z}*dst.w*(1.f-src.w))*(1.f/ao);
    return {rgb.x,rgb.y,rgb.z,ao};
}

// ---------------------------------------------------------------------------
// Blend modes (Photoshop-compatible)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec3f blend_normal    (Vec3f s,Vec3f d) noexcept { return s; }
[[nodiscard]] constexpr Vec3f blend_multiply  (Vec3f s,Vec3f d) noexcept { return s*d; }
[[nodiscard]] constexpr Vec3f blend_screen    (Vec3f s,Vec3f d) noexcept { return Vec3f{1.f}-(Vec3f{1.f}-s)*(Vec3f{1.f}-d); }
[[nodiscard]] constexpr Vec3f blend_overlay   (Vec3f s,Vec3f d) noexcept {
    auto f=[](f32 a,f32 b)->f32{ return b<0.5f?2.f*a*b:1.f-2.f*(1.f-a)*(1.f-b); };
    return {f(s.x,d.x),f(s.y,d.y),f(s.z,d.z)};
}
[[nodiscard]] constexpr Vec3f blend_hard_light(Vec3f s,Vec3f d) noexcept { return blend_overlay(d,s); }
[[nodiscard]] constexpr Vec3f blend_soft_light(Vec3f s,Vec3f d) noexcept {
    auto f=[](f32 a,f32 b)->f32{
        if(a<0.5f) return b-(1.f-2.f*a)*b*(1.f-b);
        return b+(2.f*a-1.f)*(b<0.25f?(((16.f*b-12.f)*b+4.f)*b):sqrt(b)-b);
    };
    return {f(s.x,d.x),f(s.y,d.y),f(s.z,d.z)};
}
[[nodiscard]] constexpr Vec3f blend_difference(Vec3f s,Vec3f d) noexcept {
    return {abs(s.x-d.x),abs(s.y-d.y),abs(s.z-d.z)};
}
[[nodiscard]] constexpr Vec3f blend_exclusion (Vec3f s,Vec3f d) noexcept {
    return s+d-s*d*2.f;
}
[[nodiscard]] constexpr Vec3f blend_add       (Vec3f s,Vec3f d) noexcept { return (s+d).clamp(Vec3f{0.f},Vec3f{1.f}); }
[[nodiscard]] constexpr Vec3f blend_subtract  (Vec3f s,Vec3f d) noexcept { return (d-s).clamp(Vec3f{0.f},Vec3f{1.f}); }
[[nodiscard]] constexpr Vec3f blend_darken    (Vec3f s,Vec3f d) noexcept { return s.min(d); }
[[nodiscard]] constexpr Vec3f blend_lighten   (Vec3f s,Vec3f d) noexcept { return s.max(d); }
[[nodiscard]] constexpr Vec3f blend_color_dodge(Vec3f s,Vec3f d) noexcept {
    auto f=[](f32 a,f32 b)->f32{ return b>=1.f?1.f:min(1.f,a/(1.f-b)); };
    return {f(d.x,s.x),f(d.y,s.y),f(d.z,s.z)};
}
[[nodiscard]] constexpr Vec3f blend_color_burn (Vec3f s,Vec3f d) noexcept {
    auto f=[](f32 a,f32 b)->f32{ return b<=0.f?0.f:max(0.f,1.f-(1.f-a)/b); };
    return {f(d.x,s.x),f(d.y,s.y),f(d.z,s.z)};
}
[[nodiscard]] constexpr Vec3f blend_linear_light(Vec3f s,Vec3f d) noexcept {
    return (d+s*2.f-Vec3f{1.f}).clamp(Vec3f{0.f},Vec3f{1.f});
}
[[nodiscard]] constexpr Vec3f blend_pin_light   (Vec3f s,Vec3f d) noexcept {
    auto f=[](f32 a,f32 b)->f32{ return b<2.f*a-1.f?2.f*a-1.f:(b>2.f*a?2.f*a:b); };
    return {f(s.x,d.x),f(s.y,d.y),f(s.z,d.z)};
}
[[nodiscard]] constexpr Vec3f blend_vivid_light(Vec3f s,Vec3f d) noexcept {
    auto f=[](f32 a,f32 b)->f32{
        return a<0.5f?(b<=0.f?0.f:max(0.f,1.f-(1.f-b)/(2.f*a)))
                     :(b>=1.f?1.f:min(1.f,b/(2.f*(1.f-a))));
    };
    return {f(s.x,d.x),f(s.y,d.y),f(s.z,d.z)};
}
[[nodiscard]] constexpr Vec3f blend_hue(Vec3f src, Vec3f dst) noexcept {
    ColorHSL ds=ColorHSL::from_rgb({dst.x,dst.y,dst.z});
    ColorHSL ss=ColorHSL::from_rgb({src.x,src.y,src.z});
    ColorRGB r=ColorHSL{ss.h,ds.s,ds.l}.to_rgb();
    return {r.r,r.g,r.b};
}
[[nodiscard]] constexpr Vec3f blend_saturation(Vec3f src, Vec3f dst) noexcept {
    ColorHSL ds=ColorHSL::from_rgb({dst.x,dst.y,dst.z});
    ColorHSL ss=ColorHSL::from_rgb({src.x,src.y,src.z});
    ColorRGB r=ColorHSL{ds.h,ss.s,ds.l}.to_rgb();
    return {r.r,r.g,r.b};
}
[[nodiscard]] constexpr Vec3f blend_color(Vec3f src, Vec3f dst) noexcept {
    ColorHSL ds=ColorHSL::from_rgb({dst.x,dst.y,dst.z});
    ColorHSL ss=ColorHSL::from_rgb({src.x,src.y,src.z});
    ColorRGB r=ColorHSL{ss.h,ss.s,ds.l}.to_rgb();
    return {r.r,r.g,r.b};
}
[[nodiscard]] constexpr Vec3f blend_luminosity(Vec3f src, Vec3f dst) noexcept {
    ColorHSL ds=ColorHSL::from_rgb({dst.x,dst.y,dst.z});
    ColorHSL ss=ColorHSL::from_rgb({src.x,src.y,src.z});
    ColorRGB r=ColorHSL{ds.h,ds.s,ss.l}.to_rgb();
    return {r.r,r.g,r.b};
}

// Apply blend mode with alpha
[[nodiscard]] constexpr Vec4f apply_blend_mode(
    Vec4f src, Vec4f dst,
    Vec3f(*blend_fn)(Vec3f,Vec3f),
    f32 opacity=1.f) noexcept
{
    Vec3f blended=blend_fn({src.x,src.y,src.z},{dst.x,dst.y,dst.z});
    f32 a=src.w*opacity;
    return {
        lerp(dst.x,blended.x,a),
        lerp(dst.y,blended.y,a),
        lerp(dst.z,blended.z,a),
        dst.w+a*(1.f-dst.w)
    };
}

// ---------------------------------------------------------------------------
// Keying (chroma / luma)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 chroma_key(Vec3f pixel, Vec3f key_color, f32 threshold=0.3f) noexcept {
    ColorHSV pk=ColorHSV::from_rgb({pixel.x,pixel.y,pixel.z});
    ColorHSV kk=ColorHSV::from_rgb({key_color.x,key_color.y,key_color.z});
    f32 hue_diff=abs(pk.h-kk.h)/360.f;
    hue_diff=min(hue_diff,1.f-hue_diff);
    return saturate(hue_diff/threshold);
}

[[nodiscard]] constexpr f32 luma_key(Vec3f pixel, f32 threshold, f32 softness=0.1f) noexcept {
    f32 lum=pixel.x*0.2126f+pixel.y*0.7152f+pixel.z*0.0722f;
    return smoothstep(threshold-softness,threshold+softness,lum);
}

// ---------------------------------------------------------------------------
// Mask operations
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 mask_invert(f32 m) noexcept { return 1.f-m; }
[[nodiscard]] constexpr f32 mask_multiply(f32 a, f32 b) noexcept { return a*b; }
[[nodiscard]] constexpr f32 mask_add(f32 a, f32 b) noexcept { return saturate(a+b); }
[[nodiscard]] constexpr f32 mask_subtract(f32 a, f32 b) noexcept { return saturate(a-b); }
[[nodiscard]] constexpr f32 mask_intersect(f32 a, f32 b) noexcept { return min(a,b); }
[[nodiscard]] constexpr f32 mask_union(f32 a, f32 b) noexcept { return max(a,b); }
[[nodiscard]] constexpr f32 mask_difference(f32 a, f32 b) noexcept { return abs(a-b); }

} // namespace qs::math::compositing
