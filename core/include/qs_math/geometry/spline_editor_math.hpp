#pragma once
// qs_math/spline_editor_math.hpp — Handle dragging, tangent constraints, auto-smooth, break/align (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "curves.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <numbers>
#include <functional>

namespace qs {

// ─── Spline control point with tangent handles ────────────────────────────────
enum class TangentMode : u8 { Free, Aligned, Mirrored, Auto, Vector };

struct SplineKnot {
    Vec3f pos{};
    Vec3f tan_in{};   // incoming tangent (offset from pos)
    Vec3f tan_out{};  // outgoing tangent (offset from pos)
    TangentMode mode{TangentMode::Mirrored};
    f32   weight{1.f}; // for NURBS / weighted splines
};

// ─── Catmull-Rom auto tangent computation ─────────────────────────────────────
[[nodiscard]] inline Vec3f catmull_rom_tangent(Vec3f p_prev, Vec3f p_cur, Vec3f p_next, f32 tension=0.5f) {
    return (p_next-p_prev)*tension;
}

// Auto-smooth tangents for a spline (Catmull-Rom style)
inline void auto_smooth_tangents(std::vector<SplineKnot>& knots, f32 tension=0.5f) {
    u32 n=u32(knots.size());
    for(u32 i=0;i<n;++i) {
        if(knots[i].mode!=TangentMode::Auto) continue;
        Vec3f p_prev=i>0   ?knots[i-1].pos:knots[i].pos-(knots[i+1<n?i+1:i].pos-knots[i].pos);
        Vec3f p_next=i+1<n ?knots[i+1].pos:knots[i].pos+(knots[i].pos-knots[i>0?i-1:i].pos);
        Vec3f t=catmull_rom_tangent(p_prev,knots[i].pos,p_next,tension);
        knots[i].tan_in=-t; knots[i].tan_out=t;
    }
}

// ─── Tangent constraint enforcement ──────────────────────────────────────────

// After dragging tan_out: enforce mode on tan_in
inline void enforce_tangent_constraint(SplineKnot& k, bool dragged_out) {
    switch(k.mode) {
        case TangentMode::Mirrored:
            if(dragged_out) k.tan_in=-k.tan_out;
            else            k.tan_out=-k.tan_in;
            break;
        case TangentMode::Aligned: {
            // Keep same direction, preserve other's magnitude
            if(dragged_out) {
                f32 in_len=length(k.tan_in);
                if(in_len>1e-6f) k.tan_in=-normalize(k.tan_out)*in_len;
            } else {
                f32 out_len=length(k.tan_out);
                if(out_len>1e-6f) k.tan_out=-normalize(k.tan_in)*out_len;
            }
            break;
        }
        case TangentMode::Free:    break;
        case TangentMode::Auto:    break;
        case TangentMode::Vector:  break;
    }
}

// ─── Bezier spline editor helpers ─────────────────────────────────────────────

// Evaluate a cubic Bezier segment with editor control points
[[nodiscard]] inline Vec3f bezier_seg_eval(const SplineKnot& a, const SplineKnot& b, f32 t) {
    return cubic_bezier(a.pos, a.pos+a.tan_out, b.pos+b.tan_in, b.pos, t);
}

// Evaluate derivative
[[nodiscard]] inline Vec3f bezier_seg_tangent(const SplineKnot& a, const SplineKnot& b, f32 t) {
    return cubic_bezier_deriv(a.pos, a.pos+a.tan_out, b.pos+b.tan_in, b.pos, t);
}

// Arc length of a Bezier segment (numerical integration, n subdivisions)
[[nodiscard]] inline f32 bezier_seg_length(const SplineKnot& a, const SplineKnot& b, u32 n=64) {
    f32 len=0; Vec3f prev=a.pos;
    for(u32 i=1;i<=n;++i) {
        Vec3f cur=bezier_seg_eval(a,b,f32(i)/n);
        len+=dist(prev,cur); prev=cur;
    }
    return len;
}

// Reparametrize by arc length (build lookup table)
struct ArcLengthTable {
    std::vector<f32> t_vals;   // parameter values
    std::vector<f32> arc_vals; // arc length at each t

    // Build table for a bezier segment
    void build(const SplineKnot& a, const SplineKnot& b, u32 n=200) {
        t_vals.resize(n+1); arc_vals.resize(n+1);
        arc_vals[0]=0; t_vals[0]=0;
        Vec3f prev=a.pos;
        for(u32 i=1;i<=n;++i) {
            t_vals[i]=f32(i)/n;
            Vec3f cur=bezier_seg_eval(a,b,t_vals[i]);
            arc_vals[i]=arc_vals[i-1]+dist(prev,cur);
            prev=cur;
        }
    }

    // Get parameter t at arc length s (binary search)
    [[nodiscard]] f32 t_at_length(f32 s) const {
        if(s<=0) return 0;
        if(s>=arc_vals.back()) return t_vals.back();
        u32 lo=0, hi=u32(arc_vals.size())-1;
        while(hi-lo>1){ u32 mid=(lo+hi)/2; if(arc_vals[mid]<s)lo=mid;else hi=mid; }
        f32 frac=(s-arc_vals[lo])/(arc_vals[hi]-arc_vals[lo]+1e-10f);
        return t_vals[lo]+frac*(t_vals[hi]-t_vals[lo]);
    }

    [[nodiscard]] f32 total_length() const { return arc_vals.empty()?0:arc_vals.back(); }
};

// ─── Closest point on bezier to a world point ─────────────────────────────────
[[nodiscard]] inline f32 bezier_closest_t(const SplineKnot& a, const SplineKnot& b,
    Vec3f point, u32 n=64)
{
    f32 best_t=0; f32 best_d=dist_sq(a.pos,point);
    for(u32 i=1;i<=n;++i) {
        f32 t=f32(i)/n;
        f32 d=dist_sq(bezier_seg_eval(a,b,t),point);
        if(d<best_d){ best_d=d; best_t=t; }
    }
    // Refine with Newton-Raphson
    for(u32 iter=0;iter<5;++iter) {
        Vec3f p=bezier_seg_eval(a,b,best_t);
        Vec3f dp=bezier_seg_tangent(a,b,best_t);
        f32 f=dot(p-point,dp);
        f32 df=dot(dp,dp); // approx 2nd derivative
        if(std::abs(df)<1e-10f) break;
        best_t=std::clamp(best_t-f/df,0.f,1.f);
    }
    return best_t;
}

// ─── Spline hit testing (for editor selection) ────────────────────────────────
struct SplineHit { u32 segment; f32 t; f32 distance; };

[[nodiscard]] inline std::optional<SplineHit> hit_spline(
    const std::vector<SplineKnot>& knots,
    Vec3f ray_origin, Vec3f ray_dir,
    f32 select_radius=0.05f, u32 samples_per_seg=32)
{
    if(knots.size()<2) return std::nullopt;
    std::optional<SplineHit> best;
    f32 best_d=select_radius;
    for(u32 s=0;s+1<u32(knots.size());++s) {
        for(u32 i=0;i<=samples_per_seg;++i) {
            f32 t=f32(i)/samples_per_seg;
            Vec3f p=bezier_seg_eval(knots[s],knots[s+1],t);
            // Distance from point to ray
            Vec3f v=p-ray_origin;
            f32 proj=dot(v,ray_dir);
            Vec3f closest=ray_origin+ray_dir*proj;
            f32 d=dist(p,closest);
            if(d<best_d){ best_d=d; best={s,t,d}; }
        }
    }
    return best;
}

// ─── Insert knot at parameter t on segment s ──────────────────────────────────
[[nodiscard]] inline SplineKnot insert_knot(const SplineKnot& a, const SplineKnot& b, f32 t) {
    // De Casteljau split
    Vec3f p0=a.pos, p1=a.pos+a.tan_out, p2=b.pos+b.tan_in, p3=b.pos;
    Vec3f q0=lerp(p0,p1,t), q1=lerp(p1,p2,t), q2=lerp(p2,p3,t);
    Vec3f r0=lerp(q0,q1,t), r1=lerp(q1,q2,t);
    Vec3f new_pos=lerp(r0,r1,t);
    SplineKnot k;
    k.pos=new_pos;
    k.tan_in =r0-new_pos;
    k.tan_out=r1-new_pos;
    k.mode=TangentMode::Aligned;
    return k;
}

// ─── Smooth a poly-curve (Laplacian smoothing) ────────────────────────────────
inline void smooth_poly_curve(std::vector<Vec3f>& pts, u32 iters=3, f32 factor=0.5f) {
    u32 n=u32(pts.size());
    for(u32 it=0;it<iters;++it) {
        std::vector<Vec3f> new_pts=pts;
        for(u32 i=1;i+1<n;++i)
            new_pts[i]=pts[i]+(pts[i-1]+pts[i+1]-pts[i]*2.f)*(factor*0.5f);
        pts=new_pts;
    }
}

// ─── Convert poly-line to bezier spline (corner detection) ────────────────────
[[nodiscard]] inline std::vector<SplineKnot> polyline_to_spline(
    const std::vector<Vec3f>& pts, f32 corner_threshold_deg=30.f, f32 tension=0.5f)
{
    u32 n=u32(pts.size());
    std::vector<SplineKnot> knots(n);
    for(u32 i=0;i<n;++i) {
        knots[i].pos=pts[i];
        Vec3f p_prev=i>0?pts[i-1]:pts[i];
        Vec3f p_next=i+1<n?pts[i+1]:pts[i];
        Vec3f t=catmull_rom_tangent(p_prev,pts[i],p_next,tension);
        knots[i].tan_in=-t; knots[i].tan_out=t;
        // Detect corner
        if(i>0&&i+1<n) {
            Vec3f d0=normalize(pts[i]-pts[i-1]);
            Vec3f d1=normalize(pts[i+1]-pts[i]);
            f32 cos_a=std::clamp(dot(d0,d1),-1.f,1.f);
            f32 angle=std::acos(cos_a)*180.f/f32(std::numbers::pi);
            if(angle>corner_threshold_deg) knots[i].mode=TangentMode::Free;
        }
    }
    return knots;
}

} // namespace qs
