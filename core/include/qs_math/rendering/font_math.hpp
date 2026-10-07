#pragma once
// qs_math/font_math.hpp — Bezier font rendering, SDF glyph, text layout (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "curves.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <array>
#include <string>

namespace qs {

// ─── Glyph contour ────────────────────────────────────────────────────────────

struct GlyphPoint {
    Vec2f pos{};
    bool  on_curve{true};  // true = on-curve, false = off-curve (quadratic control)
};

struct GlyphContour {
    std::vector<GlyphPoint> points;
    bool is_clockwise{true};

    // Signed area (positive = CW in screen coords)
    [[nodiscard]] f32 signed_area() const {
        f32 a=0.f; u32 N=u32(points.size());
        for(u32 i=0;i<N;++i){
            const Vec2f& p=points[i].pos;
            const Vec2f& q=points[(i+1)%N].pos;
            a+=p.x*q.y-q.x*p.y;
        }
        return a*0.5f;
    }
};

struct GlyphMetrics {
    f32 advance_width{};
    f32 left_bearing{};
    f32 ascent{}, descent{};
    f32 x_min{},x_max{},y_min{},y_max{};
};

struct Glyph {
    std::vector<GlyphContour> contours;
    GlyphMetrics metrics;
    u32 codepoint{};
};

// ─── SDF glyph generation ─────────────────────────────────────────────────────

// Distance from point P to quadratic bezier segment (P0,P1,P2)
[[nodiscard]] inline f32 dist_to_quad_bezier(
    const Vec2f& P,
    const Vec2f& P0, const Vec2f& P1, const Vec2f& P2)
{
    Vec2f A=P1-P0, B=P2-P1-A;
    Vec2f C=P0-P;
    f32 a=dot(B,B), b=3.f*dot(A,B), c=2.f*dot(A,A)+dot(C,B), d=dot(C,A);
    // Cubic solve for t
    // Simplified: sample multiple t values
    f32 min_d=std::numeric_limits<f32>::max();
    for (u32 i=0;i<=16;++i) {
        f32 t=f32(i)/16.f;
        f32 mt=1.f-t;
        Vec2f bp=P0*(mt*mt)+P1*(2.f*mt*t)+P2*(t*t);
        min_d=std::min(min_d,length_sq(P-bp));
    }
    return std::sqrt(min_d);
}

// Signed distance from point to glyph contour
[[nodiscard]] inline f32 glyph_signed_distance(
    const Vec2f& p, const Glyph& glyph, f32 scale=1.f)
{
    f32 min_dist=std::numeric_limits<f32>::max();
    // Point-in-polygon check (winding number)
    int winding=0;

    for (const auto& contour : glyph.contours) {
        u32 N=u32(contour.points.size());
        for (u32 i=0;i<N;++i) {
            Vec2f a=contour.points[i].pos*scale;
            Vec2f b=contour.points[(i+1)%N].pos*scale;
            bool b_on=contour.points[i].on_curve;
            bool c_on=contour.points[(i+1)%N].on_curve;

            if (b_on && c_on) {
                // Line segment
                Vec2f ab=b-a, ap=p-a;
                f32 t=std::clamp(dot(ap,ab)/dot(ab,ab+1e-10f),0.f,1.f);
                min_dist=std::min(min_dist,length(p-(a+ab*t)));
                // Winding
                if(a.y<=p.y){if(b.y>p.y&&(b.x-a.x)*(p.y-a.y)>(p.x-a.x)*(b.y-a.y))++winding;}
                else{if(b.y<=p.y&&(b.x-a.x)*(p.y-a.y)<(p.x-a.x)*(b.y-a.y))--winding;}
            } else {
                // Quadratic bezier
                Vec2f ctrl=b;
                Vec2f end=contour.points[(i+2)%N].pos*scale;
                min_dist=std::min(min_dist,dist_to_quad_bezier(p,a,ctrl,end));
                ++i; // skip control point
            }
        }
    }
    return (winding!=0?-1.f:1.f)*min_dist;
}

// Generate SDF texture for a glyph
inline void generate_glyph_sdf(
    const Glyph& glyph, std::vector<f32>& sdf,
    u32 tex_w, u32 tex_h, f32 scale=1.f, f32 spread=4.f)
{
    sdf.resize(tex_w*tex_h);
    GlyphMetrics m=glyph.metrics;
    f32 cell_w=scale*(m.x_max-m.x_min+2.f*spread)/f32(tex_w);
    f32 cell_h=scale*(m.y_max-m.y_min+2.f*spread)/f32(tex_h);
    for (u32 y=0;y<tex_h;++y)
    for (u32 x=0;x<tex_w;++x) {
        Vec2f p{m.x_min-spread+f32(x)*cell_w, m.y_min-spread+f32(y)*cell_h};
        f32 d=glyph_signed_distance(p,glyph,scale);
        sdf[y*tex_w+x]=std::clamp(d/spread*0.5f+0.5f,0.f,1.f);
    }
}

// ─── Text layout ─────────────────────────────────────────────────────────────

struct TextLayoutParams {
    f32 font_size{16.f};
    f32 line_height{1.2f};   // relative to font_size
    f32 letter_spacing{0.f};
    f32 word_spacing{0.f};
    f32 max_width{0.f};      // 0 = no wrap
    bool right_to_left{false};
};

struct LayoutGlyph {
    u32   codepoint{};
    Vec2f pos{};             // baseline position
    f32   scale{};
};

[[nodiscard]] inline std::vector<LayoutGlyph> layout_text(
    const std::string& text,
    const std::vector<GlyphMetrics>& metrics_table, // indexed by codepoint
    const TextLayoutParams& params)
{
    std::vector<LayoutGlyph> result;
    f32 x=0.f, y=0.f;
    f32 scale=params.font_size;
    for (u8 c : text) {
        if (c=='\n') {
            x=0.f; y-=params.font_size*params.line_height;
            continue;
        }
        if (c==' ') {
            x+=params.font_size*0.5f+params.word_spacing;
            continue;
        }
        u32 cp=u32(c);
        if (cp < metrics_table.size()) {
            const auto& m=metrics_table[cp];
            if (params.max_width>0.f && x+m.advance_width*scale>params.max_width) {
                x=0.f; y-=params.font_size*params.line_height;
            }
            result.push_back({cp,{x+m.left_bearing*scale,y},scale});
            x+=m.advance_width*scale+params.letter_spacing;
        }
    }
    return result;
}

// ─── Kerning ──────────────────────────────────────────────────────────────────

struct KerningPair {
    u32 left{}, right{};
    f32 adjustment{};
};

[[nodiscard]] inline f32 get_kerning(
    u32 left_cp, u32 right_cp,
    std::span<const KerningPair> table)
{
    for (const auto& k : table)
        if(k.left==left_cp&&k.right==right_cp) return k.adjustment;
    return 0.f;
}

// ─── Glyph outline sampling ───────────────────────────────────────────────────

// Flatten a glyph to line segments for rendering
[[nodiscard]] inline std::vector<std::pair<Vec2f,Vec2f>> flatten_glyph(
    const Glyph& g, f32 scale=1.f, u32 bezier_steps=8)
{
    std::vector<std::pair<Vec2f,Vec2f>> lines;
    for (const auto& contour : g.contours) {
        u32 N=u32(contour.points.size());
        for (u32 i=0;i<N;++i) {
            const GlyphPoint& curr=contour.points[i];
            const GlyphPoint& next=contour.points[(i+1)%N];
            if(curr.on_curve&&next.on_curve) {
                lines.push_back({curr.pos*scale,next.pos*scale});
            } else if(curr.on_curve&&!next.on_curve) {
                const GlyphPoint& after=contour.points[(i+2)%N];
                Vec2f p0=curr.pos*scale, p1=next.pos*scale, p2=after.pos*scale;
                Vec2f prev=p0;
                for (u32 s=1;s<=bezier_steps;++s) {
                    f32 t=f32(s)/f32(bezier_steps);
                    f32 mt=1.f-t;
                    Vec2f pt=p0*(mt*mt)+p1*(2*mt*t)+p2*(t*t);
                    lines.push_back({prev,pt});
                    prev=pt;
                }
                ++i;
            }
        }
    }
    return lines;
}

} // namespace qs
