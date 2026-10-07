#pragma once
// qs_math/displacement_math.hpp — PN triangles, hardware tessellation math, micro-displacement (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <cmath>
#include <array>
#include <algorithm>

namespace qs {

// ─── PN Triangles (Curved PN Patches — Vlachos 2001) ─────────────────────────

struct PNTriangle {
    // Control points (10 cubic Bezier points for position)
    Vec3f b300, b030, b003;  // corner points
    Vec3f b210, b120;        // edge 0-1
    Vec3f b021, b012;        // edge 1-2
    Vec3f b102, b201;        // edge 0-2
    Vec3f b111;              // center

    // Normal control points (6 for quadratic normals)
    Vec3f n200, n020, n002;
    Vec3f n110, n011, n101;

    static PNTriangle from_triangle(
        const Vec3f& p0, const Vec3f& p1, const Vec3f& p2,
        const Vec3f& n0, const Vec3f& n1, const Vec3f& n2)
    {
        PNTriangle pn;
        pn.b300 = p0; pn.b030 = p1; pn.b003 = p2;
        pn.n200 = n0; pn.n020 = n1; pn.n002 = n2;

        auto edge_cp = [](const Vec3f& pi, const Vec3f& pj, const Vec3f& ni) {
            f32 w = dot(pj - pi, ni);
            return (pi*2.f + pj - ni*w) * (1.f/3.f);
        };
        pn.b210 = edge_cp(p0,p1,n0); pn.b120 = edge_cp(p1,p0,n1);
        pn.b021 = edge_cp(p1,p2,n1); pn.b012 = edge_cp(p2,p1,n2);
        pn.b102 = edge_cp(p2,p0,n2); pn.b201 = edge_cp(p0,p2,n0);

        Vec3f E = (pn.b210+pn.b120+pn.b021+pn.b012+pn.b102+pn.b201)*(1.f/6.f);
        Vec3f V = (p0+p1+p2)*(1.f/3.f);
        pn.b111 = E + (E-V)*0.5f;

        auto edge_n = [](const Vec3f& ni, const Vec3f& nj, const Vec3f& pi, const Vec3f& pj) {
            f32 v = dot(pj-pi, ni+nj);
            f32 d = dot(pj-pi, pj-pi);
            return normalize(ni+nj - (v/(d+1e-10f))*(pj-pi));
        };
        pn.n110 = edge_n(n0,n1,p0,p1);
        pn.n011 = edge_n(n1,n2,p1,p2);
        pn.n101 = edge_n(n0,n2,p0,p2);
        return pn;
    }

    // Evaluate tessellated position at barycentric coords (u,v,w=1-u-v)
    [[nodiscard]] Vec3f eval_pos(f32 u, f32 v) const {
        f32 w = 1.f-u-v;
        f32 u2=u*u,v2=v*v,w2=w*w;
        f32 u3=u2*u,v3=v2*v,w3=w2*w;
        return b300*w3 + b030*u3 + b003*v3
             + b210*(3*w2*u) + b120*(3*w*u2) + b201*(3*w2*v)
             + b021*(3*u2*v) + b102*(3*w*v2) + b012*(3*u*v2)
             + b111*(6*w*u*v);
    }

    // Evaluate interpolated normal at barycentric coords
    [[nodiscard]] Vec3f eval_normal(f32 u, f32 v) const {
        f32 w = 1.f-u-v;
        Vec3f n = n200*(w*w) + n020*(u*u) + n002*(v*v)
                + n110*(w*u) + n011*(u*v) + n101*(w*v);
        return normalize(n);
    }
};

// ─── Phong Tessellation ───────────────────────────────────────────────────────

// Project point onto tangent plane of a vertex
[[nodiscard]] inline Vec3f phong_project(
    const Vec3f& p, const Vec3f& vertex, const Vec3f& normal)
{
    return p - dot(p - vertex, normal) * normal;
}

// Phong tessellated position at barycentric coords
[[nodiscard]] inline Vec3f phong_tess_pos(
    f32 u, f32 v,
    const Vec3f& p0, const Vec3f& p1, const Vec3f& p2,
    const Vec3f& n0, const Vec3f& n1, const Vec3f& n2,
    f32 alpha=0.75f)
{
    f32 w = 1.f-u-v;
    Vec3f linear = p0*w + p1*u + p2*v;
    Vec3f phong  = phong_project(linear,p0,n0)*w
                 + phong_project(linear,p1,n1)*u
                 + phong_project(linear,p2,n2)*v;
    return linear*(1.f-alpha) + phong*alpha;
}

// ─── Hardware tessellation LOD ────────────────────────────────────────────────

struct TessLODParams {
    f32 min_tess{1.0f};
    f32 max_tess{64.0f};
    f32 lod_near{1.0f};     // distance for max tess
    f32 lod_far{50.0f};     // distance for min tess
    f32 screen_height{1080.f};
    f32 fov_y_rad{1.0472f}; // 60 deg
};

[[nodiscard]] inline f32 tess_lod_distance(f32 dist, const TessLODParams& p) {
    f32 t = std::clamp((dist - p.lod_near)/(p.lod_far - p.lod_near), 0.f, 1.f);
    return p.max_tess*(1.f-t) + p.min_tess*t;
}

// Screen-space edge length based tessellation factor
[[nodiscard]] inline f32 tess_edge_screen(
    const Vec3f& a, const Vec3f& b,
    const Vec3f& cam_pos, f32 target_pixels,
    const TessLODParams& p)
{
    f32 edge_len = length(b - a);
    f32 dist = length((a+b)*0.5f - cam_pos);
    f32 proj_len = edge_len / (dist + 1e-5f)
                 * p.screen_height / (2.f*std::tan(p.fov_y_rad*0.5f));
    f32 tess = proj_len / target_pixels;
    return std::clamp(tess, p.min_tess, p.max_tess);
}

// Adaptive inner/outer tessellation levels from edge factors
struct TessLevels {
    f32 outer[4]{1,1,1,1};
    f32 inner[2]{1,1};
};

[[nodiscard]] inline TessLevels compute_tess_levels_tri(
    const Vec3f& p0, const Vec3f& p1, const Vec3f& p2,
    const Vec3f& cam, f32 target_px, const TessLODParams& lod)
{
    TessLevels t;
    t.outer[0] = tess_edge_screen(p1,p2,cam,target_px,lod);
    t.outer[1] = tess_edge_screen(p0,p2,cam,target_px,lod);
    t.outer[2] = tess_edge_screen(p0,p1,cam,target_px,lod);
    t.inner[0] = (t.outer[0]+t.outer[1]+t.outer[2])*(1.f/3.f);
    return t;
}

// ─── Micro-displacement (sub-triangle level) ──────────────────────────────────

struct MicroDispParams {
    f32 amplitude{0.05f};
    f32 base_freq{4.0f};
    u32 octaves{4};
    f32 persistence{0.5f};
    f32 lacunarity{2.0f};
};

// Simple fractal displacement value at UV
[[nodiscard]] inline f32 micro_disp_value(Vec2f uv, const MicroDispParams& p) {
    f32 val=0.f, amp=1.f, freq=p.base_freq, total_amp=0.f;
    for (u32 i=0;i<p.octaves;++i) {
        // Pseudo-noise via sin
        f32 s = std::sin(uv.x*freq*6.2831f+42.f) * std::sin(uv.y*freq*6.2831f+13.7f);
        val += s*amp; total_amp+=amp;
        amp *= p.persistence; freq *= p.lacunarity;
    }
    return (val/total_amp) * p.amplitude;
}

[[nodiscard]] inline Vec3f apply_micro_displacement(
    const Vec3f& pos, const Vec3f& normal,
    Vec2f uv, const MicroDispParams& p)
{
    f32 d = micro_disp_value(uv, p);
    return pos + normal * d;
}

// ─── Displacement map Jacobian for normals ────────────────────────────────────

// Given displacement partial derivatives, compute perturbed normal
[[nodiscard]] inline Vec3f displace_normal(
    const Vec3f& n, const Vec3f& dPdu, const Vec3f& dPdv,
    f32 dDdu, f32 dDdv, f32 scale=1.f)
{
    // Perturb tangent/bitangent by displacement gradient
    Vec3f t_new = dPdu + n * dDdu * scale;
    Vec3f b_new = dPdv + n * dDdv * scale;
    return normalize(cross(t_new, b_new));
}

// Finite-difference displacement gradient
[[nodiscard]] inline std::pair<f32,f32> disp_gradient_fd(
    Vec2f uv, const MicroDispParams& p, f32 eps=0.001f)
{
    f32 d0  = micro_disp_value(uv, p);
    f32 ddu = (micro_disp_value({uv.x+eps, uv.y}, p) - d0) / eps;
    f32 ddv = (micro_disp_value({uv.x, uv.y+eps}, p) - d0) / eps;
    return {ddu, ddv};
}

// ─── Shell / Height-field tessellation ────────────────────────────────────────

// Shell mapping: offset along normal by height map value
[[nodiscard]] inline Vec3f shell_map_pos(
    const Vec3f& base_pos, const Vec3f& normal,
    f32 height, f32 shell_thickness)
{
    return base_pos + normal * (height * shell_thickness);
}

// Cone step mapping iteration (parallax cone mapping)
[[nodiscard]] inline Vec2f cone_step_parallax(
    Vec2f uv, const Vec3f& view_ts, // view dir in tangent space
    auto height_fn,                  // height_fn(Vec2f uv) -> f32 in [0,1]
    u32 max_steps=16, f32 min_layers=4.f, f32 max_layers=16.f)
{
    f32 num_layers = std::lerp(max_layers, min_layers, std::abs(view_ts.y));
    f32 layer_depth = 1.f / num_layers;
    f32 cur_depth   = 0.f;
    Vec2f step{view_ts.x / view_ts.y * layer_depth, view_ts.z / view_ts.y * layer_depth};
    step.x *= -0.1f; step.y *= -0.1f;
    Vec2f cur_uv = uv;
    f32 cur_map = height_fn(cur_uv);
    for (u32 i=0; i<max_steps && cur_depth < cur_map; ++i) {
        cur_uv.x += step.x; cur_uv.y += step.y;
        cur_map = height_fn(cur_uv);
        cur_depth += layer_depth;
    }
    // Interpolate between last two steps
    Vec2f prev_uv{cur_uv.x-step.x, cur_uv.y-step.y};
    f32 after  = cur_map - cur_depth;
    f32 before = height_fn(prev_uv) - cur_depth + layer_depth;
    f32 w2 = after/(after-before);
    return {prev_uv.x*w2 + cur_uv.x*(1.f-w2), prev_uv.y*w2 + cur_uv.y*(1.f-w2)};
}

} // namespace qs
