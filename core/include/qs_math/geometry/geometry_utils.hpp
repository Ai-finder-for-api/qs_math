#pragma once
// geometry_utils.hpp — Misc geometry utilities:
//   point generation (Poisson disk, Fibonacci sphere, Halton, Blue noise),
//   Voronoi helpers, Delaunay helpers, convex decomposition,
//   curve parameterization, arc length tables
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include "noise.hpp"
#include "hashing.hpp"
#include <array>

namespace qs::math {

// ---------------------------------------------------------------------------
// Fibonacci sphere — uniformly distributed points on unit sphere
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f fibonacci_sphere_point(i32 index, i32 total) noexcept {
    f32 phi=TWO_PI*f32(index)*INV_PHI;
    f32 y=1.f-2.f*f32(index)/f32(total-1);
    f32 r=sqrt(max(0.f,1.f-y*y));
    return {r*cos(phi), y, r*sin(phi)};
}
inline void fibonacci_sphere(Vec3f* out, i32 n) noexcept {
    for(i32 i=0;i<n;++i) out[i]=fibonacci_sphere_point(i,n);
}

// ---------------------------------------------------------------------------
// Fibonacci disk — 2D analog
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec2f fibonacci_disk_point(i32 index, i32 total) noexcept {
    f32 r=sqrt(f32(index)/f32(total));
    f32 phi=TWO_PI*f32(index)*INV_PHI;
    return {r*cos(phi), r*sin(phi)};
}

// ---------------------------------------------------------------------------
// Halton sequence 2D (multi-sample patterns)
// ---------------------------------------------------------------------------
inline void halton_sequence_2d(Vec2f* out, i32 n, i32 base1=2, i32 base2=3) noexcept {
    for(i32 i=0;i<n;++i){
        f32 x=0.f,y=0.f,f=1.f;
        i32 idx=i+1;
        i32 tmp=idx; f=1.f;
        while(tmp>0){ f/=base1; x+=f*(tmp%base1); tmp/=base1; }
        tmp=idx; f=1.f;
        while(tmp>0){ f/=base2; y+=f*(tmp%base2); tmp/=base2; }
        out[i]={x,y};
    }
}

// ---------------------------------------------------------------------------
// Poisson disk sampling 2D (Mitchell's best-candidate approximation)
// ---------------------------------------------------------------------------
inline i32 poisson_disk_2d(Vec2f* out, i32 max_points,
                            f32 domain_w, f32 domain_h, f32 min_dist,
                            PCG32& rng) noexcept
{
    i32 n=0;
    i32 candidates=30;
    while(n<max_points){
        Vec2f best; f32 best_dist=0.f;
        for(i32 k=0;k<candidates;++k){
            Vec2f p{rng.range_f32(0.f,domain_w), rng.range_f32(0.f,domain_h)};
            f32 d=F32_INF;
            for(i32 j=0;j<n;++j) d=min(d,(p-out[j]).length());
            if(d>best_dist){best_dist=d;best=p;}
        }
        if(best_dist<min_dist&&n>0) break;
        out[n++]=best;
    }
    return n;
}

// ---------------------------------------------------------------------------
// UV sphere generation (lat/lon grid)
// ---------------------------------------------------------------------------
inline void generate_uv_sphere(i32 lat_segs, i32 lon_segs,
                                 Vec3f* pos, Vec2f* uv, u32* indices,
                                 i32& vert_count, i32& idx_count) noexcept
{
    vert_count=0; idx_count=0;
    for(i32 lat=0;lat<=lat_segs;++lat){
        f32 phi=PI*lat/lat_segs;
        for(i32 lon=0;lon<=lon_segs;++lon){
            f32 theta=TWO_PI*lon/lon_segs;
            f32 x=sin(phi)*cos(theta), y=cos(phi), z=sin(phi)*sin(theta);
            pos[vert_count]={x,y,z};
            uv[vert_count]={f32(lon)/lon_segs, f32(lat)/lat_segs};
            ++vert_count;
        }
    }
    for(i32 lat=0;lat<lat_segs;++lat){
        for(i32 lon=0;lon<lon_segs;++lon){
            u32 a=lat*(lon_segs+1)+lon, b=a+lon_segs+1;
            indices[idx_count++]=a; indices[idx_count++]=b;   indices[idx_count++]=a+1;
            indices[idx_count++]=b; indices[idx_count++]=b+1; indices[idx_count++]=a+1;
        }
    }
}

// ---------------------------------------------------------------------------
// Icosphere generation
// ---------------------------------------------------------------------------
inline void generate_icosphere_base(Vec3f* verts, u32* indices,
                                     i32& vert_count, i32& idx_count) noexcept
{
    constexpr f32 t=(1.f+SQRT5)*0.5f; // golden ratio
    constexpr f32 s=1.f/sqrt(1.f+t*t);
    vert_count=12; idx_count=20*3;
    static constexpr Vec3f v[12]={
        {-1, t, 0},{1, t, 0},{-1,-t, 0},{1,-t, 0},
        {0,-1, t},{0, 1, t},{0,-1,-t},{0, 1,-t},
        { t, 0,-1},{t, 0, 1},{-t, 0,-1},{-t, 0, 1}
    };
    for(i32 i=0;i<12;++i) verts[i]=(v[i]*s).normalized();
    static constexpr u32 idx[60]={
        0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,
        1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,
        3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,
        4,9,5,2,4,11,6,2,10,8,6,7,9,8,1
    };
    for(i32 i=0;i<60;++i) indices[i]=idx[i];
}

// ---------------------------------------------------------------------------
// Arc length parameterization (table-based)
// ---------------------------------------------------------------------------
template<typename CurveFn, FloatScalar T=f32>
struct ArcLengthTable {
    static constexpr i32 SAMPLES=256;
    T params[SAMPLES];   // original t values
    T lengths[SAMPLES];  // cumulative arc lengths
    T total_length{T(0)};

    void build(CurveFn fn) noexcept {
        lengths[0]=T(0); params[0]=T(0);
        T dt=T(1)/(SAMPLES-1);
        for(i32 i=1;i<SAMPLES;++i){
            T t0=T(i-1)*dt, t1=T(i)*dt;
            auto p0=fn(t0), p1=fn(t1);
            lengths[i]=lengths[i-1]+(p1-p0).length();
            params[i]=t1;
        }
        total_length=lengths[SAMPLES-1];
    }

    // Convert arc-length s to parameter t
    [[nodiscard]] T s_to_t(T s) const noexcept {
        if(total_length<T(EPS_LENGTH)) return T(0);
        s=clamp(s,T(0),total_length);
        // Binary search
        i32 lo=0, hi=SAMPLES-1;
        while(lo<hi-1){
            i32 mid=(lo+hi)/2;
            if(lengths[mid]<s) lo=mid; else hi=mid;
        }
        T frac=(s-lengths[lo])/(lengths[hi]-lengths[lo]+T(EPS_LENGTH));
        return lerp(params[lo],params[hi],frac);
    }
    [[nodiscard]] T length() const noexcept { return total_length; }
};

// ---------------------------------------------------------------------------
// Angle → 2D direction with spread (cone sampling)
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f cone_sample(Vec3f axis, f32 half_angle_rad, f32 u1, f32 u2) noexcept {
    f32 cos_theta=lerp(cos(half_angle_rad),1.f,u1);
    f32 sin_theta=sqrt(max(0.f,1.f-cos_theta*cos_theta));
    f32 phi=TWO_PI*u2;
    Vec3f local{sin_theta*cos(phi), sin_theta*sin(phi), cos_theta};
    Vec3f t,b; orthonormal_basis(axis,t,b);
    return (t*local.x+b*local.y+axis*local.z).normalized();
}

// ---------------------------------------------------------------------------
// Signed distance field helpers
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f32 sdf_sphere(Vec3f p, f32 r) noexcept { return p.length()-r; }
[[nodiscard]] constexpr f32 sdf_box(Vec3f p, Vec3f b) noexcept {
    Vec3f q=p.abs()-b;
    return q.max(Vec3f{0.f}).length()+min(max(q.x,max(q.y,q.z)),0.f);
}
[[nodiscard]] constexpr f32 sdf_torus(Vec3f p, f32 R, f32 r) noexcept {
    Vec2f q{Vec2f{p.x,p.z}.length()-R, p.y};
    return q.length()-r;
}
[[nodiscard]] constexpr f32 sdf_cylinder(Vec3f p, f32 r, f32 h) noexcept {
    Vec2f d{Vec2f{p.x,p.z}.length()-r, abs(p.y)-h*0.5f};
    return min(max(d.x,d.y),0.f)+d.max(Vec2f{0.f}).length();
}
[[nodiscard]] constexpr f32 sdf_capsule(Vec3f p, Vec3f a, Vec3f b, f32 r) noexcept {
    Vec3f ab=b-a, ap=p-a;
    f32 t=saturate(dot(ap,ab)/max(ab.length_sq(),EPS_LENGTH));
    return (ap-ab*t).length()-r;
}
// SDF smooth union/subtraction/intersection
[[nodiscard]] constexpr f32 sdf_smooth_union(f32 d1, f32 d2, f32 k) noexcept {
    f32 h=max(k-abs(d1-d2),0.f)/k;
    return min(d1,d2)-h*h*k*0.25f;
}
[[nodiscard]] constexpr f32 sdf_smooth_subtract(f32 d1, f32 d2, f32 k) noexcept {
    return -sdf_smooth_union(-d1,d2,k);
}
[[nodiscard]] constexpr f32 sdf_smooth_intersect(f32 d1, f32 d2, f32 k) noexcept {
    return -sdf_smooth_union(-d1,-d2,k);
}
// Gradient of SDF (finite differences)
[[nodiscard]] inline Vec3f sdf_normal(auto&& sdf, Vec3f p, f32 eps=0.001f) noexcept {
    return Vec3f{
        sdf({p.x+eps,p.y,p.z})-sdf({p.x-eps,p.y,p.z}),
        sdf({p.x,p.y+eps,p.z})-sdf({p.x,p.y-eps,p.z}),
        sdf({p.x,p.y,p.z+eps})-sdf({p.x,p.y,p.z-eps})
    }.normalized();
}

} // namespace qs::math
