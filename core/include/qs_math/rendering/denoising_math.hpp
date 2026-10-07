#pragma once
// qs_math/denoising_math.hpp — Bilateral, NL-means, SVGF, A-SVGF spatio-temporal kernels (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <span>
#include <functional>

namespace qs {

// ─── Bilateral Filter ─────────────────────────────────────────────────────────

struct BilateralParams {
    f32 sigma_spatial{3.0f};
    f32 sigma_range{0.1f};
    u32 radius{3};
};

[[nodiscard]] inline f32 bilateral_weight(
    f32 dist_sq, f32 range_diff_sq,
    f32 sigma_s, f32 sigma_r)
{
    return std::exp(-dist_sq/(2.f*sigma_s*sigma_s)
                   -range_diff_sq/(2.f*sigma_r*sigma_r));
}

// Bilateral filter on a Vec3f image buffer
inline void bilateral_filter(
    const Vec3f* src, Vec3f* dst,
    u32 w, u32 h, const BilateralParams& p)
{
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        Vec3f center = src[y*w+x];
        Vec3f acc{0,0,0}; f32 wsum = 0.f;
        i32 r = i32(p.radius);
        for (i32 dy = -r; dy <= r; ++dy)
        for (i32 dx = -r; dx <= r; ++dx) {
            i32 nx = std::clamp(i32(x)+dx, 0, i32(w-1));
            i32 ny = std::clamp(i32(y)+dy, 0, i32(h-1));
            Vec3f c = src[u32(ny)*w+u32(nx)];
            Vec3f diff = c - center;
            f32 rd = dot(diff,diff);
            f32 sd = f32(dx*dx+dy*dy);
            f32 wt = bilateral_weight(sd, rd, p.sigma_spatial, p.sigma_range);
            acc.x += c.x*wt; acc.y += c.y*wt; acc.z += c.z*wt;
            wsum += wt;
        }
        dst[y*w+x] = wsum > 0.f ? Vec3f{acc.x/wsum,acc.y/wsum,acc.z/wsum} : center;
    }
}

// ─── Joint Bilateral (guide image) ───────────────────────────────────────────

inline void joint_bilateral_filter(
    const Vec3f* src, Vec3f* dst,
    const Vec3f* guide,
    u32 w, u32 h, const BilateralParams& p)
{
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        Vec3f gcenter = guide[y*w+x];
        Vec3f acc{0,0,0}; f32 wsum = 0.f;
        i32 r = i32(p.radius);
        for (i32 dy = -r; dy <= r; ++dy)
        for (i32 dx = -r; dx <= r; ++dx) {
            i32 nx2 = std::clamp(i32(x)+dx, 0, i32(w-1));
            i32 ny2 = std::clamp(i32(y)+dy, 0, i32(h-1));
            Vec3f c = src[u32(ny2)*w+u32(nx2)];
            Vec3f gn = guide[u32(ny2)*w+u32(nx2)];
            Vec3f diff = gn - gcenter;
            f32 rd = dot(diff,diff);
            f32 sd = f32(dx*dx+dy*dy);
            f32 wt = bilateral_weight(sd, rd, p.sigma_spatial, p.sigma_range);
            acc.x+=c.x*wt; acc.y+=c.y*wt; acc.z+=c.z*wt;
            wsum += wt;
        }
        dst[y*w+x] = wsum > 0.f ? Vec3f{acc.x/wsum,acc.y/wsum,acc.z/wsum} : src[y*w+x];
    }
}

// ─── Non-Local Means ──────────────────────────────────────────────────────────

struct NLMeansParams {
    f32 h{0.1f};        // filtering param
    u32 search_r{10};   // search window radius
    u32 patch_r{3};     // patch radius
};

[[nodiscard]] inline f32 nlm_patch_distance(
    const Vec3f* img, u32 w, u32 h,
    i32 px, i32 py, i32 qx, i32 qy, u32 patch_r)
{
    f32 dist = 0.f; u32 n = 0;
    i32 r = i32(patch_r);
    for (i32 dy = -r; dy <= r; ++dy)
    for (i32 dx = -r; dx <= r; ++dx) {
        i32 px2 = std::clamp(px+dx,0,i32(w-1));
        i32 py2 = std::clamp(py+dy,0,i32(h-1));
        i32 qx2 = std::clamp(qx+dx,0,i32(w-1));
        i32 qy2 = std::clamp(qy+dy,0,i32(h-1));
        Vec3f dp = img[u32(py2)*w+u32(px2)] - img[u32(qy2)*w+u32(qx2)];
        dist += dot(dp,dp); ++n;
    }
    return n > 0 ? dist/f32(n) : 0.f;
}

inline void nlmeans_filter(
    const Vec3f* src, Vec3f* dst, u32 w, u32 h, const NLMeansParams& p)
{
    f32 h2 = p.h * p.h;
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        Vec3f acc{0,0,0}; f32 wsum = 0.f;
        i32 sr = i32(p.search_r);
        for (i32 dy = -sr; dy <= sr; ++dy)
        for (i32 dx = -sr; dx <= sr; ++dx) {
            i32 qx = std::clamp(i32(x)+dx,0,i32(w-1));
            i32 qy = std::clamp(i32(y)+dy,0,i32(h-1));
            f32 d2 = nlm_patch_distance(src,w,h,i32(x),i32(y),qx,qy,p.patch_r);
            f32 wt = std::exp(-std::max(0.f,d2-0.f)/h2);
            Vec3f c = src[u32(qy)*w+u32(qx)];
            acc.x+=c.x*wt; acc.y+=c.y*wt; acc.z+=c.z*wt;
            wsum+=wt;
        }
        dst[y*w+x] = wsum>0.f ? Vec3f{acc.x/wsum,acc.y/wsum,acc.z/wsum} : src[y*w+x];
    }
}

// ─── SVGF — Spatio-Temporal Variance Guided Filter ───────────────────────────

struct SVGFGBuffer {
    Vec3f color{};
    Vec3f albedo{};
    Vec3f normal{};
    f32   depth{};
    Vec2f motion{};       // screen-space motion vector in [0,1]
    f32   luminance_var{};
};

[[nodiscard]] inline f32 svgf_luminance(const Vec3f& c) {
    return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
}

struct SVGFWeights {
    f32 w_depth{1.0f};
    f32 w_normal{128.f};
    f32 w_lum{4.0f};
};

[[nodiscard]] inline f32 svgf_edge_weight(
    const SVGFGBuffer& p, const SVGFGBuffer& q,
    const SVGFWeights& w)
{
    f32 depth_diff = std::abs(p.depth - q.depth) / (std::abs(p.depth) + 1e-4f);
    f32 wz = std::exp(-w.w_depth * depth_diff);
    f32 wn = std::pow(std::max(0.f, dot(p.normal, q.normal)), w.w_normal);
    f32 lp = svgf_luminance(p.color);
    f32 lq = svgf_luminance(q.color);
    f32 wl = std::exp(-w.w_lum * std::abs(lp - lq) / (std::sqrt(p.luminance_var) + 1e-4f));
    return wz * wn * wl;
}

// SVGF à-trous wavelet pass (one iteration at step size `step`)
inline void svgf_atrous_pass(
    const SVGFGBuffer* gbuf_src,
    const Vec3f* color_in, Vec3f* color_out,
    const f32* var_in, f32* var_out,
    u32 w, u32 h, u32 step,
    const SVGFWeights& wts)
{
    static constexpr f32 kernel3[3] = {1.f/4.f, 1.f/2.f, 1.f/4.f};
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        u32 idx = y*w+x;
        const SVGFGBuffer& cp = gbuf_src[idx];
        Vec3f cacc{0,0,0}; f32 vacc=0.f, wsum=0.f;
        for (i32 dy = -1; dy <= 1; ++dy)
        for (i32 dx = -1; dx <= 1; ++dx) {
            i32 nx2 = std::clamp(i32(x)+dx*i32(step),0,i32(w-1));
            i32 ny2 = std::clamp(i32(y)+dy*i32(step),0,i32(h-1));
            u32 nidx = u32(ny2)*w+u32(nx2);
            f32 ew = svgf_edge_weight(cp, gbuf_src[nidx], wts);
            f32 kw = kernel3[dx+1]*kernel3[dy+1];
            f32 total = ew*kw;
            cacc.x += color_in[nidx].x*total;
            cacc.y += color_in[nidx].y*total;
            cacc.z += color_in[nidx].z*total;
            vacc   += var_in[nidx]*total*total;
            wsum   += total;
        }
        color_out[idx] = wsum>0.f ? Vec3f{cacc.x/wsum,cacc.y/wsum,cacc.z/wsum} : color_in[idx];
        var_out[idx]   = wsum>0.f ? vacc/(wsum*wsum) : var_in[idx];
    }
}

// Temporal accumulation with alpha blending
inline void svgf_temporal_accumulate(
    const Vec3f* cur, Vec3f* history, f32* hist_var, f32* cur_var,
    const SVGFGBuffer* gbuf, u32 w, u32 h, f32 alpha=0.1f)
{
    for (u32 i = 0; i < w*h; ++i) {
        history[i].x = history[i].x*(1.f-alpha) + cur[i].x*alpha;
        history[i].y = history[i].y*(1.f-alpha) + cur[i].y*alpha;
        history[i].z = history[i].z*(1.f-alpha) + cur[i].z*alpha;
        f32 lum = svgf_luminance(cur[i]);
        f32 lum_h = svgf_luminance(history[i]);
        cur_var[i] = (1.f-alpha)*hist_var[i] + alpha*(lum-lum_h)*(lum-lum_h);
        hist_var[i] = cur_var[i];
    }
}

// ─── A-SVGF (adaptive SVGF) gradient-based firefly suppression ───────────────

[[nodiscard]] inline f32 asvgf_gradient(
    f32 lum_cur, f32 lum_prev, f32 variance)
{
    f32 d = std::abs(lum_cur - lum_prev);
    return d / (std::sqrt(variance) + 1e-4f);
}

[[nodiscard]] inline f32 asvgf_temporal_alpha(
    f32 gradient, f32 alpha_min=0.05f, f32 alpha_max=0.3f)
{
    f32 t = std::clamp(1.f - gradient, 0.f, 1.f);
    return alpha_min + t*(alpha_max - alpha_min);
}

// ─── Variance estimation (3x3 spatial) ───────────────────────────────────────

inline void estimate_variance(
    const Vec3f* color, f32* var_out, u32 w, u32 h)
{
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        f32 sum=0.f, sum2=0.f; u32 cnt=0;
        for (i32 dy=-1;dy<=1;++dy)
        for (i32 dx=-1;dx<=1;++dx) {
            i32 nx2=std::clamp(i32(x)+dx,0,i32(w-1));
            i32 ny2=std::clamp(i32(y)+dy,0,i32(h-1));
            f32 l = svgf_luminance(color[u32(ny2)*w+u32(nx2)]);
            sum+=l; sum2+=l*l; ++cnt;
        }
        f32 mean=sum/f32(cnt);
        var_out[y*w+x]=std::max(0.f, sum2/f32(cnt)-mean*mean);
    }
}

// ─── Firefly suppression ──────────────────────────────────────────────────────

inline void suppress_fireflies(Vec3f* color, u32 w, u32 h, f32 threshold=10.f) {
    for (u32 y=0;y<h;++y)
    for (u32 x=0;x<w;++x) {
        f32 local_max=0.f;
        for(i32 dy=-1;dy<=1;++dy)
        for(i32 dx=-1;dx<=1;++dx) {
            if(dx==0&&dy==0) continue;
            i32 nx2=std::clamp(i32(x)+dx,0,i32(w-1));
            i32 ny2=std::clamp(i32(y)+dy,0,i32(h-1));
            local_max=std::max(local_max, svgf_luminance(color[u32(ny2)*w+u32(nx2)]));
        }
        f32 lum = svgf_luminance(color[y*w+x]);
        if (lum > local_max * threshold && lum > 1e-4f) {
            f32 scale = local_max*threshold/lum;
            color[y*w+x].x*=scale; color[y*w+x].y*=scale; color[y*w+x].z*=scale;
        }
    }
}

} // namespace qs
