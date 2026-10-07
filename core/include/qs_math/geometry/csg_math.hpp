#pragma once
// qs_math/geometry/csg_math.hpp — Constructive solid geometry: SDF ops, boolean, blending
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

struct V3 { f32 x,y,z;
    [[nodiscard]] constexpr V3 operator+(V3 b) const noexcept {return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr V3 operator-(V3 b) const noexcept {return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr V3 operator*(f32 s) const noexcept {return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f32 dot(V3 b)  const noexcept {return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr f32 len2()     const noexcept {return dot(*this);}
    [[nodiscard]]           f32 len()      const noexcept {return std::sqrt(len2());}
    [[nodiscard]] constexpr V3  abs()      const noexcept {return{x<0?-x:x,y<0?-y:y,z<0?-z:z};}
    [[nodiscard]] constexpr f32 maxcomp()  const noexcept {return x>y?(x>z?x:z):(y>z?y:z);}
    [[nodiscard]] constexpr V3  max(V3 b)  const noexcept {return{x>b.x?x:b.x,y>b.y?y:b.y,z>b.z?z:b.z};}
    [[nodiscard]] constexpr V3  min(V3 b)  const noexcept {return{x<b.x?x:b.x,y<b.y?y:b.y,z<b.z?z:b.z};}
};
[[nodiscard]] constexpr f32 sdf_clamp(f32 v,f32 lo,f32 hi) noexcept {return v<lo?lo:(v>hi?hi:v);}
[[nodiscard]] constexpr f32 sdf_mix(f32 a,f32 b,f32 t) noexcept {return a*(1-t)+b*t;}
[[nodiscard]] constexpr f32 sdf_sign(f32 v) noexcept {return v<0?-1.f:1.f;}

// ─── Primitive SDFs ───────────────────────────────────────────────────────────
[[nodiscard]] inline f32 sdf_sphere(V3 p, f32 r) noexcept { return p.len()-r; }
[[nodiscard]] constexpr f32 sdf_box(V3 p, V3 b) noexcept {
    V3 q=p.abs()-b; return q.max({0,0,0}).len()+std::min(q.maxcomp(),0.f);
}
[[nodiscard]] inline f32 sdf_cylinder(V3 p, f32 r, f32 h) noexcept {
    V3 d={std::sqrt(p.x*p.x+p.z*p.z)-r,std::abs(p.y)-h,0};
    return std::min(std::max(d.x,d.y),0.f)+V3{std::max(d.x,0.f),std::max(d.y,0.f),0}.len();
}
[[nodiscard]] inline f32 sdf_cone(V3 p, f32 angle, f32 h) noexcept {
    f32 q=std::sqrt(p.x*p.x+p.z*p.z);
    V3 w={q,p.y,0};
    V3 a={std::cos(angle),std::sin(angle),0};
    f32 d=w.dot(a)-h*sdf_clamp(w.dot(a)/h,0,1);
    return sdf_sign(std::max(-p.y,w.dot({-std::sin(angle),std::cos(angle),0})-h))*std::sqrt(d);
}
[[nodiscard]] inline f32 sdf_torus(V3 p, f32 R, f32 r) noexcept {
    f32 q=std::sqrt(p.x*p.x+p.z*p.z)-R;
    return std::sqrt(q*q+p.y*p.y)-r;
}
[[nodiscard]] inline f32 sdf_capsule(V3 p, V3 a, V3 b, f32 r) noexcept {
    V3 pa=p-a,ba=b-a;
    f32 t=sdf_clamp(pa.dot(ba)/ba.dot(ba),0,1);
    return (pa-ba*t).len()-r;
}
[[nodiscard]] inline f32 sdf_plane(V3 p, V3 n, f32 d) noexcept { return p.dot(n)-d; }
[[nodiscard]] inline f32 sdf_ellipsoid(V3 p, V3 r) noexcept {
    f32 k0=V3{p.x/r.x,p.y/r.y,p.z/r.z}.len();
    f32 k1=V3{p.x/(r.x*r.x),p.y/(r.y*r.y),p.z/(r.z*r.z)}.len();
    return k0*(k0-1)/k1;
}
[[nodiscard]] inline f32 sdf_rounded_box(V3 p, V3 b, f32 rad) noexcept {
    return sdf_box(p,b)-rad;
}
[[nodiscard]] inline f32 sdf_link(V3 p, f32 le, f32 r1, f32 r2) noexcept {
    V3 q={p.x, std::max(std::abs(p.y)-le,0.f), p.z};
    return std::sqrt((std::sqrt(q.x*q.x+q.y*q.y)-r1)*(std::sqrt(q.x*q.x+q.y*q.y)-r1)+q.z*q.z)-r2;
}
[[nodiscard]] inline f32 sdf_octahedron(V3 p, f32 s) noexcept {
    p=p.abs(); f32 m=p.x+p.y+p.z-s;
    V3 q; if(3*p.x<m) q=p; else if(3*p.y<m) q={p.y,p.x,p.z}; else q={p.z,p.x,p.y};
    f32 k=sdf_clamp((q.y-q.z+s)*0.5f,0,s*0.5f); // corrected
    return V3{q.x,q.y-s+k,q.z-k}.len();
}

// ─── Boolean operations ───────────────────────────────────────────────────────
[[nodiscard]] constexpr f32 sdf_union      (f32 a, f32 b) noexcept { return a<b?a:b; }
[[nodiscard]] constexpr f32 sdf_subtract   (f32 a, f32 b) noexcept { return a>-b?a:-b; }
[[nodiscard]] constexpr f32 sdf_intersect  (f32 a, f32 b) noexcept { return a>b?a:b; }
[[nodiscard]] constexpr f32 sdf_xor        (f32 a, f32 b) noexcept {
    return sdf_intersect(sdf_union(a,b),sdf_subtract(sdf_union(a,b),sdf_intersect(a,b)));
}

// ─── Smooth boolean operations ────────────────────────────────────────────────
[[nodiscard]] constexpr f32 sdf_smooth_union(f32 a, f32 b, f32 k) noexcept {
    f32 h=sdf_clamp(0.5f+0.5f*(b-a)/k,0,1);
    return sdf_mix(b,a,h)-k*h*(1-h);
}
[[nodiscard]] constexpr f32 sdf_smooth_subtract(f32 a, f32 b, f32 k) noexcept {
    f32 h=sdf_clamp(0.5f-0.5f*(b+a)/k,0,1);
    return sdf_mix(b,-a,h)+k*h*(1-h);
}
[[nodiscard]] constexpr f32 sdf_smooth_intersect(f32 a, f32 b, f32 k) noexcept {
    f32 h=sdf_clamp(0.5f-0.5f*(b-a)/k,0,1);
    return sdf_mix(b,a,h)+k*h*(1-h);
}
[[nodiscard]] constexpr f32 sdf_chamfer_union(f32 a, f32 b, f32 r) noexcept {
    return std::min(std::min(a,b),(a-r+b)*0.70710678f);
}
[[nodiscard]] constexpr f32 sdf_engrave(f32 a, f32 b, f32 r) noexcept {
    return std::max(a,-sdf_smooth_union(a,b,r));
}

// ─── Domain operations ────────────────────────────────────────────────────────
[[nodiscard]] constexpr V3 sdf_repeat(V3 p, V3 c) noexcept {
    auto fmod=[](f32 a,f32 b){f32 r=a-int(a/b)*b;return r<0?r+b:r;};
    return{fmod(p.x,c.x)-c.x*0.5f,fmod(p.y,c.y)-c.y*0.5f,fmod(p.z,c.z)-c.z*0.5f};
}
[[nodiscard]] constexpr V3 sdf_twist(V3 p, f32 k) noexcept {
    f32 c=std::cos(k*p.y),s=std::sin(k*p.y);
    return{c*p.x-s*p.z,p.y,s*p.x+c*p.z};
}
[[nodiscard]] constexpr V3 sdf_bend(V3 p, f32 k) noexcept {
    f32 c=std::cos(k*p.x),s=std::sin(k*p.x);
    return{c*p.x-s*p.y,s*p.x+c*p.y,p.z};
}
[[nodiscard]] constexpr V3 sdf_mirror_x(V3 p) noexcept {return{p.x<0?-p.x:p.x,p.y,p.z};}
[[nodiscard]] constexpr V3 sdf_mirror_y(V3 p) noexcept {return{p.x,p.y<0?-p.y:p.y,p.z};}
[[nodiscard]] constexpr V3 sdf_mirror_z(V3 p) noexcept {return{p.x,p.y,p.z<0?-p.z:p.z};}

// ─── SDF normal estimation (central difference) ────────────────────────────────
template<typename F>
[[nodiscard]] inline V3 sdf_normal(F&& f, V3 p, f32 eps=1e-4f) noexcept {
    return {
        f({p.x+eps,p.y,p.z})-f({p.x-eps,p.y,p.z}),
        f({p.x,p.y+eps,p.z})-f({p.x,p.y-eps,p.z}),
        f({p.x,p.y,p.z+eps})-f({p.x,p.y,p.z-eps})
    };
}

// ─── Ray marching ────────────────────────────────────────────────────────────
template<typename F>
[[nodiscard]] inline f32 ray_march(F&& sdf, V3 ro, V3 rd,
                                    f32 tmin=0.01f,f32 tmax=100.f,int max_steps=128) noexcept {
    f32 t=tmin;
    for(int i=0;i<max_steps;++i) {
        f32 d=sdf(ro+rd*t);
        if(d<1e-4f*t) return t;
        t+=d;
        if(t>tmax) break;
    }
    return tmax;
}

// ─── Ambient occlusion (SDF-based) ───────────────────────────────────────────
template<typename F>
[[nodiscard]] inline f32 sdf_ao(F&& sdf, V3 p, V3 n, int steps=5, f32 step=0.1f) noexcept {
    f32 occ=0,fac=1;
    for(int i=1;i<=steps;++i) {
        f32 d=sdf(p+n*(f32(i)*step));
        occ+=(f32(i)*step-d)*fac;
        fac*=0.75f;
    }
    return 1.f-sdf_clamp(2.f*occ,0,1);
}

// ─── Signed distance to 2D shapes (for layered CSG) ──────────────────────────
[[nodiscard]] constexpr f32 sdf2d_circle(f32 x,f32 y,f32 r) noexcept {
    return std::sqrt(x*x+y*y)-r;
}
[[nodiscard]] constexpr f32 sdf2d_box(f32 px,f32 py,f32 bx,f32 by) noexcept {
    f32 dx=std::abs(px)-bx,dy=std::abs(py)-by;
    return std::sqrt(std::max(dx,0.f)*std::max(dx,0.f)+std::max(dy,0.f)*std::max(dy,0.f))
          +std::min(std::max(dx,dy),0.f);
}
[[nodiscard]] constexpr f32 sdf2d_segment(f32 px,f32 py,f32 ax,f32 ay,f32 bx,f32 by) noexcept {
    f32 papx=px-ax,papy=py-ay,babx=bx-ax,baby=by-ay;
    f32 t=sdf_clamp((papx*babx+papy*baby)/(babx*babx+baby*baby),0,1);
    f32 dx=papx-babx*t,dy=papy-baby*t;
    return std::sqrt(dx*dx+dy*dy);
}

// ─── Fractal SDFs ────────────────────────────────────────────────────────────
[[nodiscard]] constexpr f32 sdf_mandelbulb(V3 p, int iters=8) noexcept {
    V3 w=p; f32 m=w.dot(w); f32 dz=1;
    for(int i=0;i<iters;++i) {
        dz=8*std::pow(std::sqrt(m),7.f)*dz+1;
        f32 r=w.len();
        if(r>4) break;
        f32 b=8*std::acos(sdf_clamp(w.y/r,-1,1));
        f32 a=8*std::atan2(w.x,w.z);
        f32 r8=std::pow(r,8.f);
        w=p+V3{std::sin(b)*std::sin(a),std::cos(b),std::sin(b)*std::cos(a)}*r8;
        m=w.dot(w); if(m>4096) break;
    }
    return 0.25f*std::log(m)*std::sqrt(m)/dz;
}
[[nodiscard]] constexpr f32 sdf_menger_sponge(V3 p, int iter=3) noexcept {
    f32 d=sdf_box(p,{1,1,1});
    f32 s=1;
    for(int i=0;i<iter;++i) {
        V3 a={std::fmod(std::abs(p.x)*s,2.f)-1,
              std::fmod(std::abs(p.y)*s,2.f)-1,
              std::fmod(std::abs(p.z)*s,2.f)-1};
        s*=3; f32 r=1.f/s;
        f32 c=std::max(sdf2d_box(a.x,a.y,.3f,.3f),
              std::max(sdf2d_box(a.y,a.z,.3f,.3f),
                       sdf2d_box(a.x,a.z,.3f,.3f)));
        d=std::max(d,-c*r);
    }
    return d;
}
} // namespace qs::math
