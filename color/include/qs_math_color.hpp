#pragma once
// qs_math/color/include/qs_math_color.hpp (C++20)
// Extended color science: color difference, gamut mapping, tone curves
#include <qs_math/arithmetic/scalar_types.hpp>
#include <cmath>
#include <array>
namespace qs::math::color_ext {
using namespace qs::math;
struct Lab { f32 L,a,b; };
struct XYZ { f32 X,Y,Z; };
struct LCh { f32 L,C,h; };
struct Oklab { f32 L,a,b; };
// ─── CIE Delta-E 2000 ─────────────────────────────────────────────────────────
[[nodiscard]] f32 delta_e2000(Lab lab1, Lab lab2, f32 kL=1,f32 kC=1,f32 kH=1) noexcept;
// ─── sRGB ↔ Linear ────────────────────────────────────────────────────────────
[[nodiscard]] inline f32 srgb_to_linear(f32 v) noexcept {
    return v<=0.04045f?v/12.92f:std::pow((v+0.055f)/1.055f,2.4f);
}
[[nodiscard]] inline f32 linear_to_srgb(f32 v) noexcept {
    return v<=0.0031308f?v*12.92f:1.055f*std::pow(v,1.0f/2.4f)-0.055f;
}
// ─── XYZ ↔ CIE Lab (D65) ─────────────────────────────────────────────────────
[[nodiscard]] Lab xyz_to_lab(XYZ xyz) noexcept;
[[nodiscard]] XYZ lab_to_xyz(Lab lab) noexcept;
// ─── Oklab ────────────────────────────────────────────────────────────────────
[[nodiscard]] Oklab linear_srgb_to_oklab(f32 r,f32 g,f32 b) noexcept;
[[nodiscard]] std::array<f32,3> oklab_to_linear_srgb(Oklab c) noexcept;
// ─── Lab ↔ LCh ────────────────────────────────────────────────────────────────
[[nodiscard]] inline LCh lab_to_lch(Lab l) noexcept {
    return {l.L, std::sqrt(l.a*l.a+l.b*l.b), std::atan2(l.b,l.a)};
}
[[nodiscard]] inline Lab lch_to_lab(LCh c) noexcept {
    return {c.L, c.C*std::cos(c.h), c.C*std::sin(c.h)};
}
// ─── Tone curves ──────────────────────────────────────────────────────────────
[[nodiscard]] inline f32 aces_filmic(f32 x) noexcept {
    const f32 a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
    return std::max(0.0f,std::min(1.0f,(x*(a*x+b))/(x*(c*x+d)+e)));
}
[[nodiscard]] inline f32 reinhard(f32 x) noexcept { return x/(1+x); }
[[nodiscard]] inline f32 reinhard_extended(f32 x,f32 max_white) noexcept {
    return x*(1+x/(max_white*max_white))/(1+x);
}
} // namespace qs::math::color_ext
