#pragma once
// qs_math/color_grading.hpp — Lift/Gamma/Gain, ASC CDL, color wheels, film emulation (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "color.hpp"
#include <cmath>
#include <array>

namespace qs {

// ─── Lift / Gamma / Gain ──────────────────────────────────────────────────────
// Classic 3-way color correction model used in DaVinci Resolve, etc.
struct LGG {
    Vec3f lift{0,0,0};   // shadow control (additive offset)
    Vec3f gamma{1,1,1};  // midtone control (power curve)
    Vec3f gain{1,1,1};   // highlight control (multiplicative scale)
    f32 master_lift{0};
    f32 master_gamma{1};
    f32 master_gain{1};

    [[nodiscard]] Vec3f apply(Vec3f c) const {
        // Resolve-style LGG: ((c * gain) + lift) ^ (1/gamma)
        c.x=std::pow(std::max(0.f,(c.x*gain.x*master_gain)+(lift.x+master_lift)), 1.f/std::max(1e-4f,gamma.x*master_gamma));
        c.y=std::pow(std::max(0.f,(c.y*gain.y*master_gain)+(lift.y+master_lift)), 1.f/std::max(1e-4f,gamma.y*master_gamma));
        c.z=std::pow(std::max(0.f,(c.z*gain.z*master_gain)+(lift.z+master_lift)), 1.f/std::max(1e-4f,gamma.z*master_gamma));
        return c;
    }
};

// ─── ASC CDL (Color Decision List) ───────────────────────────────────────────
// out = (in * slope + offset) ^ (1/power)
struct ASCCDL {
    Vec3f slope{1,1,1};
    Vec3f offset{0,0,0};
    Vec3f power{1,1,1};
    f32   saturation{1.f};

    [[nodiscard]] Vec3f apply(Vec3f c) const {
        // SOP
        c.x=std::pow(std::max(0.f,c.x*slope.x+offset.x),1.f/std::max(1e-5f,power.x));
        c.y=std::pow(std::max(0.f,c.y*slope.y+offset.y),1.f/std::max(1e-5f,power.y));
        c.z=std::pow(std::max(0.f,c.z*slope.z+offset.z),1.f/std::max(1e-5f,power.z));
        // Saturation: luma = 0.2126R + 0.7152G + 0.0722B
        if(std::abs(saturation-1.f)>1e-4f) {
            f32 luma=0.2126f*c.x+0.7152f*c.y+0.0722f*c.z;
            c=Vec3f{luma,luma,luma}+(c-Vec3f{luma,luma,luma})*saturation;
        }
        return c;
    }
};

// ─── S-Curve (contrast) ───────────────────────────────────────────────────────
// Smooth contrast S-curve: c' = c^(1+a) / (c^(1+a) + (1-c)^(1+a))  — Alexa
[[nodiscard]] inline Vec3f scurve_contrast(Vec3f c, f32 contrast) {
    auto sc=[&](f32 x)->f32{
        f32 a=contrast+1;
        return std::pow(x,a)/(std::pow(x,a)+std::pow(1-x,a));
    };
    return {sc(c.x),sc(c.y),sc(c.z)};
}
// Classic cubic S-curve
[[nodiscard]] inline f32 cubic_scurve(f32 x) { return x*x*(3-2*x); }
// Smooth step (Hermite)
[[nodiscard]] inline Vec3f smoothstep_curve(Vec3f c) {
    return {cubic_scurve(c.x),cubic_scurve(c.y),cubic_scurve(c.z)};
}

// ─── Color wheels (hue rotation + saturation tint per tonal range) ───────────
// Shadow/midtone/highlight hue rotation and saturation controls
struct ColorWheels {
    // Each wheel: (hue_angle_deg, saturation_boost, lightness_offset)
    Vec3f shadow_hsv{0,0,0};    // h,s,v offset for shadows
    Vec3f midtone_hsv{0,0,0};
    Vec3f highlight_hsv{0,0,0};

    // Soft masks for each tonal range
    [[nodiscard]] f32 shadow_mask(f32 luma)    const { return 1-std::smoothstep(0.1f,0.4f,luma); }
    [[nodiscard]] f32 midtone_mask(f32 luma)   const { f32 s=std::smoothstep(0.1f,0.4f,luma); return s*(1-std::smoothstep(0.6f,0.9f,luma)); }
    [[nodiscard]] f32 highlight_mask(f32 luma) const { return std::smoothstep(0.6f,0.9f,luma); }

    [[nodiscard]] Vec3f apply(Vec3f rgb) const {
        f32 luma=0.2126f*rgb.x+0.7152f*rgb.y+0.0722f*rgb.z;
        Vec3f result=rgb;
        // Apply hue rotation per zone (simplified: rotate in OKLab hue)
        auto apply_zone=[&](Vec3f col, Vec3f wheel_hsv, f32 mask)->Vec3f{
            if(mask<1e-5f) return col;
            // Convert to HSV, adjust, convert back
            auto hsv=rgb_to_hsv(col);
            hsv.x+=wheel_hsv.x/360.f;
            hsv.y=std::clamp(hsv.y+wheel_hsv.y,0.f,1.f);
            hsv.z=std::clamp(hsv.z+wheel_hsv.z,0.f,1.f);
            Vec3f adjusted=hsv_to_rgb(hsv);
            return col+(adjusted-col)*mask;
        };
        result=apply_zone(result,shadow_hsv,shadow_mask(luma));
        result=apply_zone(result,midtone_hsv,midtone_mask(luma));
        result=apply_zone(result,highlight_hsv,highlight_mask(luma));
        return result;
    }
};

// ─── Film emulation curves ────────────────────────────────────────────────────
// Kodak 5207 / 5219 style S-curve approximation (log domain)
[[nodiscard]] inline Vec3f film_kodak_emulation(Vec3f linear) {
    // Convert to log, apply film curve, convert back
    auto log_encode=[](f32 x)->f32{ return x<=0?0:std::log10(std::max(1e-5f,x))*0.125f+0.5f; };
    auto film_curve=[](f32 t)->f32{
        // Cubic spline through measured points (approximate)
        if(t<0.1f) return t*0.9f;
        if(t<0.5f) return 0.09f+(t-0.1f)*0.95f;
        if(t<0.9f) return 0.47f+(t-0.5f)*0.8f;
        return 0.79f+(t-0.9f)*0.3f;
    };
    Vec3f log_c{log_encode(linear.x),log_encode(linear.y),log_encode(linear.z)};
    return {film_curve(log_c.x),film_curve(log_c.y),film_curve(log_c.z)};
}

// ─── Look Up Table (3D LUT) ───────────────────────────────────────────────────
struct LUT3D {
    u32 size{33}; // 33^3 common
    std::vector<Vec3f> data; // [r*size*size + g*size + b]

    LUT3D() = default;
    explicit LUT3D(u32 s) : size(s), data(s*s*s,{}) {}

    [[nodiscard]] Vec3f sample(Vec3f c) const {
        if(data.empty()) return c;
        c={std::clamp(c.x,0.f,1.f),std::clamp(c.y,0.f,1.f),std::clamp(c.z,0.f,1.f)};
        f32 rs=c.x*(size-1), gs=c.y*(size-1), bs=c.z*(size-1);
        u32 r0=u32(rs),g0=u32(gs),b0=u32(bs);
        u32 r1=std::min(r0+1,size-1),g1=std::min(g0+1,size-1),b1=std::min(b0+1,size-1);
        f32 fr=rs-r0, fg=gs-g0, fb=bs-b0;
        auto get=[&](u32 r,u32 g,u32 b)->Vec3f{ return data[r*size*size+g*size+b]; };
        // Trilinear interpolation
        auto lerp3=[](Vec3f a,Vec3f b,f32 t)->Vec3f{ return a+(b-a)*t; };
        Vec3f c000=get(r0,g0,b0),c001=get(r0,g0,b1),c010=get(r0,g1,b0),c011=get(r0,g1,b1);
        Vec3f c100=get(r1,g0,b0),c101=get(r1,g0,b1),c110=get(r1,g1,b0),c111=get(r1,g1,b1);
        return lerp3(lerp3(lerp3(c000,c001,fb),lerp3(c010,c011,fb),fg),
                     lerp3(lerp3(c100,c101,fb),lerp3(c110,c111,fb),fg),fr);
    }

    // Build LUT from a color transform function
    void build(const std::function<Vec3f(Vec3f)>& transform) {
        data.resize(size*size*size);
        for(u32 r=0;r<size;++r) for(u32 g=0;g<size;++g) for(u32 b=0;b<size;++b)
            data[r*size*size+g*size+b]=transform({r/f32(size-1),g/f32(size-1),b/f32(size-1)});
    }
};

// ─── Grading pipeline ─────────────────────────────────────────────────────────
struct GradingPipeline {
    ASCCDL  primary;
    LGG     lgg;
    ColorWheels wheels;
    LUT3D   creative_lut;
    f32     exposure{0};    // in stops
    f32     temperature_offset{0}; // in Kelvin
    f32     tint{0};        // green-magenta
    f32     vibrance{0};    // selective saturation

    [[nodiscard]] Vec3f apply(Vec3f linear) const {
        // Exposure
        linear=linear*std::pow(2.f,exposure);
        // White balance (simplified: colour temp shift in RGB)
        if(std::abs(temperature_offset)>1) {
            f32 t=temperature_offset/6500.f;
            linear.x+=t*0.1f; linear.z-=t*0.1f;
        }
        // Primary CDL
        linear=primary.apply(linear);
        // LGG
        linear=lgg.apply(linear);
        // Colour wheels
        linear=wheels.apply(linear);
        // Vibrance (selective saturation: less saturated pixels get more boost)
        if(std::abs(vibrance)>1e-4f) {
            f32 luma=0.2126f*linear.x+0.7152f*linear.y+0.0722f*linear.z;
            f32 sat=std::max({linear.x,linear.y,linear.z})-std::min({linear.x,linear.y,linear.z});
            f32 vib_scale=1+vibrance*(1-sat);
            linear=Vec3f{luma,luma,luma}+(linear-Vec3f{luma,luma,luma})*vib_scale;
        }
        // Creative LUT
        if(!creative_lut.data.empty()) {
            // Tonemap to display range first, apply LUT, then HDR reconstruct (simplified)
            Vec3f ldr={std::clamp(linear.x,0.f,1.f),std::clamp(linear.y,0.f,1.f),std::clamp(linear.z,0.f,1.f)};
            ldr=creative_lut.sample(ldr);
            linear=ldr;
        }
        return linear;
    }
};

// ─── Halation (film) and bloom ───────────────────────────────────────────────
// Gaussian bloom weight at distance r with given sigma
[[nodiscard]] inline f32 bloom_gaussian(f32 r, f32 sigma) {
    return std::exp(-r*r/(2*sigma*sigma))/(2*f32(std::numbers::pi)*sigma*sigma);
}
// Dual exponential (closer to film halation)
[[nodiscard]] inline f32 halation_dual_exp(f32 r, f32 sigma1=0.01f, f32 sigma2=0.1f, f32 mix=0.7f) {
    return mix*bloom_gaussian(r,sigma1)+(1-mix)*bloom_gaussian(r,sigma2);
}

} // namespace qs
