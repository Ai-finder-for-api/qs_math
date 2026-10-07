#pragma once
// qs_math/color_naming.hpp — Nearest named color, CSS color math, palette generation (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "color.hpp"
#include <array>
#include <string_view>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

namespace qs {

// ─── Named CSS colors (subset of 148 standard colors) ────────────────────────
struct NamedColor { std::string_view name; f32 r,g,b; };

static constexpr NamedColor CSS_COLORS[] = {
    {"aliceblue",        0.941f,0.973f,1.000f},{"antiquewhite",    0.980f,0.922f,0.843f},
    {"aqua",             0.000f,1.000f,1.000f},{"aquamarine",      0.498f,1.000f,0.831f},
    {"azure",            0.941f,1.000f,1.000f},{"beige",           0.961f,0.961f,0.863f},
    {"bisque",           1.000f,0.894f,0.769f},{"black",           0.000f,0.000f,0.000f},
    {"blanchedalmond",   1.000f,0.922f,0.804f},{"blue",            0.000f,0.000f,1.000f},
    {"blueviolet",       0.541f,0.169f,0.886f},{"brown",           0.647f,0.165f,0.165f},
    {"burlywood",        0.871f,0.722f,0.529f},{"cadetblue",       0.373f,0.620f,0.627f},
    {"chartreuse",       0.498f,1.000f,0.000f},{"chocolate",       0.824f,0.412f,0.118f},
    {"coral",            1.000f,0.498f,0.314f},{"cornflowerblue",  0.392f,0.584f,0.929f},
    {"cornsilk",         1.000f,0.973f,0.863f},{"crimson",         0.863f,0.078f,0.235f},
    {"cyan",             0.000f,1.000f,1.000f},{"darkblue",        0.000f,0.000f,0.545f},
    {"darkcyan",         0.000f,0.545f,0.545f},{"darkgoldenrod",   0.722f,0.525f,0.043f},
    {"darkgray",         0.663f,0.663f,0.663f},{"darkgreen",       0.000f,0.392f,0.000f},
    {"darkkhaki",        0.741f,0.718f,0.420f},{"darkmagenta",     0.545f,0.000f,0.545f},
    {"darkolivegreen",   0.333f,0.420f,0.184f},{"darkorange",      1.000f,0.549f,0.000f},
    {"darkorchid",       0.600f,0.196f,0.800f},{"darkred",         0.545f,0.000f,0.000f},
    {"darksalmon",       0.914f,0.588f,0.478f},{"darkseagreen",    0.561f,0.737f,0.561f},
    {"darkslateblue",    0.282f,0.239f,0.545f},{"darkslategray",   0.184f,0.310f,0.310f},
    {"darkturquoise",    0.000f,0.808f,0.820f},{"darkviolet",      0.580f,0.000f,0.827f},
    {"deeppink",         1.000f,0.078f,0.576f},{"deepskyblue",     0.000f,0.749f,1.000f},
    {"dimgray",          0.412f,0.412f,0.412f},{"dodgerblue",      0.118f,0.565f,1.000f},
    {"firebrick",        0.698f,0.133f,0.133f},{"floralwhite",     1.000f,0.980f,0.941f},
    {"forestgreen",      0.133f,0.545f,0.133f},{"fuchsia",         1.000f,0.000f,1.000f},
    {"gainsboro",        0.863f,0.863f,0.863f},{"ghostwhite",      0.973f,0.973f,1.000f},
    {"gold",             1.000f,0.843f,0.000f},{"goldenrod",       0.855f,0.647f,0.125f},
    {"gray",             0.502f,0.502f,0.502f},{"green",           0.000f,0.502f,0.000f},
    {"greenyellow",      0.678f,1.000f,0.184f},{"honeydew",        0.941f,1.000f,0.941f},
    {"hotpink",          1.000f,0.412f,0.706f},{"indianred",       0.804f,0.361f,0.361f},
    {"indigo",           0.294f,0.000f,0.510f},{"ivory",           1.000f,1.000f,0.941f},
    {"khaki",            0.941f,0.902f,0.549f},{"lavender",        0.902f,0.902f,0.980f},
    {"lavenderblush",    1.000f,0.941f,0.961f},{"lawngreen",       0.486f,0.988f,0.000f},
    {"lemonchiffon",     1.000f,0.980f,0.804f},{"lightblue",       0.678f,0.847f,0.902f},
    {"lightcoral",       0.941f,0.502f,0.502f},{"lightcyan",       0.878f,1.000f,1.000f},
    {"lightgoldenrodyellow",0.980f,0.980f,0.824f},{"lightgray",    0.827f,0.827f,0.827f},
    {"lightgreen",       0.565f,0.933f,0.565f},{"lightpink",       1.000f,0.714f,0.757f},
    {"lightsalmon",      1.000f,0.627f,0.478f},{"lightseagreen",   0.125f,0.698f,0.667f},
    {"lightskyblue",     0.529f,0.808f,0.980f},{"lightslategray",  0.467f,0.533f,0.600f},
    {"lightsteelblue",   0.690f,0.769f,0.871f},{"lightyellow",     1.000f,1.000f,0.878f},
    {"lime",             0.000f,1.000f,0.000f},{"limegreen",       0.196f,0.804f,0.196f},
    {"linen",            0.980f,0.941f,0.902f},{"magenta",         1.000f,0.000f,1.000f},
    {"maroon",           0.502f,0.000f,0.000f},{"mediumaquamarine",0.400f,0.804f,0.667f},
    {"mediumblue",       0.000f,0.000f,0.804f},{"mediumorchid",    0.729f,0.333f,0.827f},
    {"mediumpurple",     0.576f,0.439f,0.847f},{"mediumseagreen",  0.235f,0.702f,0.443f},
    {"mediumslateblue",  0.482f,0.408f,0.933f},{"mediumspringgreen",0.000f,0.980f,0.604f},
    {"mediumturquoise",  0.282f,0.820f,0.800f},{"mediumvioletred", 0.780f,0.082f,0.522f},
    {"midnightblue",     0.098f,0.098f,0.439f},{"mintcream",       0.961f,1.000f,0.980f},
    {"mistyrose",        1.000f,0.894f,0.882f},{"moccasin",        1.000f,0.894f,0.710f},
    {"navy",             0.000f,0.000f,0.502f},{"oldlace",         0.992f,0.961f,0.902f},
    {"olive",            0.502f,0.502f,0.000f},{"olivedrab",       0.420f,0.557f,0.137f},
    {"orange",           1.000f,0.647f,0.000f},{"orangered",       1.000f,0.271f,0.000f},
    {"orchid",           0.855f,0.439f,0.839f},{"palegoldenrod",   0.933f,0.910f,0.667f},
    {"palegreen",        0.596f,0.984f,0.596f},{"paleturquoise",   0.686f,0.933f,0.933f},
    {"palevioletred",    0.847f,0.439f,0.576f},{"papayawhip",      1.000f,0.937f,0.835f},
    {"peachpuff",        1.000f,0.855f,0.725f},{"peru",            0.804f,0.522f,0.247f},
    {"pink",             1.000f,0.753f,0.796f},{"plum",            0.867f,0.627f,0.867f},
    {"powderblue",       0.690f,0.878f,0.902f},{"purple",          0.502f,0.000f,0.502f},
    {"red",              1.000f,0.000f,0.000f},{"rosybrown",       0.737f,0.561f,0.561f},
    {"royalblue",        0.255f,0.412f,0.882f},{"saddlebrown",     0.545f,0.271f,0.075f},
    {"salmon",           0.980f,0.502f,0.447f},{"sandybrown",      0.957f,0.643f,0.376f},
    {"seagreen",         0.180f,0.545f,0.341f},{"seashell",        1.000f,0.961f,0.933f},
    {"sienna",           0.627f,0.322f,0.176f},{"silver",          0.753f,0.753f,0.753f},
    {"skyblue",          0.529f,0.808f,0.922f},{"slateblue",       0.416f,0.353f,0.804f},
    {"slategray",        0.439f,0.502f,0.565f},{"snow",            1.000f,0.980f,0.980f},
    {"springgreen",      0.000f,1.000f,0.498f},{"steelblue",       0.275f,0.510f,0.706f},
    {"tan",              0.824f,0.706f,0.549f},{"teal",            0.000f,0.502f,0.502f},
    {"thistle",          0.847f,0.749f,0.847f},{"tomato",          1.000f,0.388f,0.278f},
    {"turquoise",        0.251f,0.878f,0.816f},{"violet",          0.933f,0.510f,0.933f},
    {"wheat",            0.961f,0.871f,0.702f},{"white",           1.000f,1.000f,1.000f},
    {"whitesmoke",       0.961f,0.961f,0.961f},{"yellow",          1.000f,1.000f,0.000f},
    {"yellowgreen",      0.604f,0.804f,0.196f}
};
static constexpr u32 NUM_CSS_COLORS = sizeof(CSS_COLORS)/sizeof(CSS_COLORS[0]);

// ─── Nearest named color ───────────────────────────────────────────────────────
[[nodiscard]] inline const NamedColor& nearest_css_color(Vec3f rgb) {
    u32 best=0; f32 best_d=1e30f;
    for(u32 i=0;i<NUM_CSS_COLORS;++i) {
        f32 dr=CSS_COLORS[i].r-rgb.x,dg=CSS_COLORS[i].g-rgb.y,db=CSS_COLORS[i].b-rgb.z;
        f32 d=dr*dr+dg*dg+db*db;
        if(d<best_d){best_d=d;best=i;}
    }
    return CSS_COLORS[best];
}

// Nearest named color in perceptual (LAB) space
[[nodiscard]] inline const NamedColor& nearest_css_color_perceptual(Vec3f rgb) {
    // Convert query to LAB
    auto lab_q=rgb_to_lab(rgb);
    u32 best=0; f32 best_d=1e30f;
    for(u32 i=0;i<NUM_CSS_COLORS;++i) {
        auto lab_i=rgb_to_lab({CSS_COLORS[i].r,CSS_COLORS[i].g,CSS_COLORS[i].b});
        f32 dL=lab_q.x-lab_i.x,da=lab_q.y-lab_i.y,db=lab_q.z-lab_i.z;
        f32 d=dL*dL+da*da+db*db;
        if(d<best_d){best_d=d;best=i;}
    }
    return CSS_COLORS[best];
}

// ─── CSS color parsing ────────────────────────────────────────────────────────
[[nodiscard]] inline std::optional<Vec3f> parse_css_color(std::string_view name) {
    for(u32 i=0;i<NUM_CSS_COLORS;++i)
        if(CSS_COLORS[i].name==name) return Vec3f{CSS_COLORS[i].r,CSS_COLORS[i].g,CSS_COLORS[i].b};
    return std::nullopt;
}

// Parse #RRGGBB or #RGB
[[nodiscard]] inline std::optional<Vec3f> parse_hex_color(std::string_view hex) {
    if(hex.empty()||hex[0]!='#') return std::nullopt;
    hex=hex.substr(1);
    auto h2f=[](char hi,char lo)->f32{
        auto hv=[](char c)->u32{ return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:0; };
        return (hv(hi)*16+hv(lo))/255.f;
    };
    if(hex.size()==6) return Vec3f{h2f(hex[0],hex[1]),h2f(hex[2],hex[3]),h2f(hex[4],hex[5])};
    if(hex.size()==3) return Vec3f{h2f(hex[0],hex[0]),h2f(hex[1],hex[1]),h2f(hex[2],hex[2])};
    return std::nullopt;
}

// Format as #RRGGBB
[[nodiscard]] inline std::string to_hex_color(Vec3f rgb) {
    char buf[8];
    u32 r=u32(std::clamp(rgb.x,0.f,1.f)*255),g=u32(std::clamp(rgb.y,0.f,1.f)*255),b=u32(std::clamp(rgb.z,0.f,1.f)*255);
    std::snprintf(buf,sizeof(buf),"#%02X%02X%02X",r,g,b);
    return buf;
}

// ─── Palette generation ────────────────────────────────────────────────────────

// Complementary color (opposite on color wheel)
[[nodiscard]] inline Vec3f complementary(Vec3f rgb) {
    auto hsv=rgb_to_hsv(rgb); hsv.x=std::fmod(hsv.x+0.5f,1.f); return hsv_to_rgb(hsv);
}

// Analogous colors (adjacent on color wheel)
[[nodiscard]] inline std::array<Vec3f,3> analogous(Vec3f rgb, f32 step=1.f/12) {
    auto hsv=rgb_to_hsv(rgb);
    return { hsv_to_rgb({std::fmod(hsv.x-step+1,1.f),hsv.y,hsv.z}),
             rgb,
             hsv_to_rgb({std::fmod(hsv.x+step,1.f),hsv.y,hsv.z}) };
}

// Triadic palette (120° apart)
[[nodiscard]] inline std::array<Vec3f,3> triadic(Vec3f rgb) {
    auto hsv=rgb_to_hsv(rgb);
    return { rgb,
             hsv_to_rgb({std::fmod(hsv.x+1.f/3,1.f),hsv.y,hsv.z}),
             hsv_to_rgb({std::fmod(hsv.x+2.f/3,1.f),hsv.y,hsv.z}) };
}

// Split-complementary
[[nodiscard]] inline std::array<Vec3f,3> split_complementary(Vec3f rgb, f32 split=1.f/12) {
    auto hsv=rgb_to_hsv(rgb);
    f32 comp=hsv.x+0.5f;
    return { rgb,
             hsv_to_rgb({std::fmod(comp-split+1,1.f),hsv.y,hsv.z}),
             hsv_to_rgb({std::fmod(comp+split,1.f),hsv.y,hsv.z}) };
}

// Tetradic (square) palette (90° apart)
[[nodiscard]] inline std::array<Vec3f,4> tetradic(Vec3f rgb) {
    auto hsv=rgb_to_hsv(rgb);
    return { rgb,
             hsv_to_rgb({std::fmod(hsv.x+0.25f,1.f),hsv.y,hsv.z}),
             hsv_to_rgb({std::fmod(hsv.x+0.50f,1.f),hsv.y,hsv.z}),
             hsv_to_rgb({std::fmod(hsv.x+0.75f,1.f),hsv.y,hsv.z}) };
}

// Monochromatic palette (vary value and saturation)
[[nodiscard]] inline std::vector<Vec3f> monochromatic(Vec3f rgb, u32 n=5) {
    auto hsv=rgb_to_hsv(rgb);
    std::vector<Vec3f> palette;
    for(u32 i=0;i<n;++i) {
        f32 t=f32(i)/(n-1);
        Vec3f h={hsv.x, hsv.y*(0.5f+t*0.5f), 0.2f+t*0.8f};
        palette.push_back(hsv_to_rgb(h));
    }
    return palette;
}

// Generate perceptually uniform gradient (interpolate in LAB space)
[[nodiscard]] inline std::vector<Vec3f> lab_gradient(Vec3f rgb0, Vec3f rgb1, u32 n=8) {
    auto lab0=rgb_to_lab(rgb0), lab1=rgb_to_lab(rgb1);
    std::vector<Vec3f> grad;
    for(u32 i=0;i<n;++i) {
        f32 t=f32(i)/(n-1);
        Vec3f lab=lab0+(lab1-lab0)*t;
        grad.push_back(lab_to_rgb(lab));
    }
    return grad;
}

// Dominant color extraction (K-means on pixel colors, single step)
[[nodiscard]] inline std::vector<Vec3f> dominant_colors(
    const std::vector<Vec3f>& pixels, u32 k=5, u32 iters=20)
{
    if(pixels.size()<k) return pixels;
    // Random init (use evenly spaced samples)
    std::vector<Vec3f> centers(k);
    for(u32 i=0;i<k;++i) centers[i]=pixels[i*u32(pixels.size())/k];
    for(u32 it=0;it<iters;++it) {
        std::vector<Vec3f> sums(k,{}); std::vector<u32> cnts(k,0);
        for(const auto& p:pixels) {
            u32 best=0; f32 bd=1e30f;
            for(u32 j=0;j<k;++j){ f32 d=dist_sq(p,centers[j]); if(d<bd){bd=d;best=j;} }
            sums[best]=sums[best]+p; ++cnts[best];
        }
        for(u32 j=0;j<k;++j) if(cnts[j]>0) centers[j]=sums[j]*(1.f/cnts[j]);
    }
    return centers;
}

// Color temperature to RGB (approximate, K)
[[nodiscard]] inline Vec3f kelvin_to_rgb(f32 K) {
    K=std::clamp(K,1000.f,40000.f)/100.f;
    f32 r,g,b;
    if(K<=66){ r=1; g=std::clamp((99.4708025861f*std::log(K)-161.1195681661f)/255,0.f,1.f); }
    else { r=std::clamp((329.698727446f*std::pow(K-60,-0.1332047592f))/255,0.f,1.f); g=std::clamp((288.1221695283f*std::pow(K-60,-0.0755148492f))/255,0.f,1.f); }
    if(K>=66) b=1;
    else if(K<=19) b=0;
    else b=std::clamp((138.5177312231f*std::log(K-10)-305.0447927307f)/255,0.f,1.f);
    return {r,g,b};
}

} // namespace qs
