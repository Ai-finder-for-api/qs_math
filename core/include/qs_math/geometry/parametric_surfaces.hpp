#pragma once
// qs_math/geometry/parametric_surfaces.hpp — Quadrics, minimal surfaces, surfaces of revolution
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

struct P3 { f64 x,y,z;
    [[nodiscard]] constexpr P3 operator+(P3 b) const noexcept {return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr P3 operator*(f64 s) const noexcept {return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64 dot(P3 b) const noexcept {return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr P3 cross(P3 b) const noexcept {return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    [[nodiscard]] f64 len() const noexcept {return std::sqrt(dot(*this));}
};

// ─── Quadric surfaces ─────────────────────────────────────────────────────────
// General quadric: Ax²+By²+Cz²+Dxy+Eyz+Fxz+Gx+Hy+Iz+J=0
struct Quadric { f64 A,B,C,D,E,F,G,H,I,J;
    [[nodiscard]] constexpr f64 eval(f64 x,f64 y,f64 z) const noexcept {
        return A*x*x+B*y*y+C*z*z+D*x*y+E*y*z+F*x*z+G*x+H*y+I*z+J;
    }
    // Gradient (normal direction)
    [[nodiscard]] constexpr P3 grad(f64 x,f64 y,f64 z) const noexcept {
        return{2*A*x+D*y+F*z+G,2*B*y+D*x+E*z+H,2*C*z+E*y+F*x+I};
    }
};
// Named quadrics
[[nodiscard]] constexpr Quadric sphere(f64 cx,f64 cy,f64 cz,f64 r) noexcept {
    return{1,1,1,0,0,0,-2*cx,-2*cy,-2*cz,cx*cx+cy*cy+cz*cz-r*r};
}
[[nodiscard]] constexpr Quadric ellipsoid(f64 a,f64 b,f64 c) noexcept {
    return{1/(a*a),1/(b*b),1/(c*c),0,0,0,0,0,0,-1};
}
[[nodiscard]] constexpr Quadric hyperboloid_one_sheet(f64 a,f64 b,f64 c) noexcept {
    return{1/(a*a),1/(b*b),-1/(c*c),0,0,0,0,0,0,-1};
}
[[nodiscard]] constexpr Quadric paraboloid(f64 a,f64 b) noexcept {
    return{1/(a*a),1/(b*b),0,0,0,0,0,0,-1,0};
}
[[nodiscard]] constexpr Quadric hyperbolic_paraboloid(f64 a,f64 b) noexcept {
    return{1/(a*a),-1/(b*b),0,0,0,0,0,0,-1,0};
}
[[nodiscard]] constexpr Quadric cone(f64 a,f64 b,f64 c) noexcept {
    return{1/(a*a),1/(b*b),-1/(c*c),0,0,0,0,0,0,0};
}

// ─── Parametric surfaces ──────────────────────────────────────────────────────
// Sphere
[[nodiscard]] constexpr P3 sphere_point(f64 u, f64 v, f64 r=1) noexcept {
    return{r*std::cos(u)*std::sin(v),r*std::sin(u)*std::sin(v),r*std::cos(v)};
}
// Torus (R=major, r=minor radius)
[[nodiscard]] constexpr P3 torus_point(f64 u, f64 v, f64 R=2, f64 r=1) noexcept {
    return{(R+r*std::cos(v))*std::cos(u),(R+r*std::cos(v))*std::sin(u),r*std::sin(v)};
}
// Klein bottle (immersed in R³)
[[nodiscard]] constexpr P3 klein_bottle(f64 u, f64 v) noexcept {
    if(u<3.14159265358979323846)
        return{(2+std::cos(u/2)*std::sin(v)-std::sin(u/2)*std::sin(2*v))*std::cos(u),
               (2+std::cos(u/2)*std::sin(v)-std::sin(u/2)*std::sin(2*v))*std::sin(u),
               std::sin(u/2)*std::sin(v)+std::cos(u/2)*std::sin(2*v)};
    return{(2+std::cos(u/2)*std::sin(v)-std::sin(u/2)*std::sin(2*v))*std::cos(u),
           (2+std::cos(u/2)*std::sin(v)-std::sin(u/2)*std::sin(2*v))*std::sin(u),
           std::sin(u/2)*std::sin(v)+std::cos(u/2)*std::sin(2*v)};
}
// Boy surface
[[nodiscard]] constexpr P3 boy_surface(f64 u, f64 v) noexcept {
    f64 su=std::sin(u),cu=std::cos(u),sv=std::sin(v),cv=std::cos(v);
    f64 s2v=std::sin(2*v),c2v=std::cos(2*v),s2u=std::sin(2*u);
    return{0.5*su*s2v-cu*cv,0.5*su*s2v+cu*cv,s2u*0.5+0.5*(1-su*su*s2v*s2v*0.5)};
}
// Enneper minimal surface
[[nodiscard]] constexpr P3 enneper(f64 u, f64 v) noexcept {
    return{u-u*u*u/3+u*v*v,v-v*v*v/3+v*u*u,u*u-v*v};
}
// Scherk's surface: e^z*cos(y)=cos(x)
[[nodiscard]] constexpr f64 scherk_z(f64 x, f64 y) noexcept {
    f64 cy=std::cos(y),cx=std::cos(x);
    if(std::abs(cy)<1e-10||cx/cy<=0) return 0;
    return std::log(std::abs(cx/cy));
}
// Helicoid (minimal): x=u*cos(v), y=u*sin(v), z=v
[[nodiscard]] constexpr P3 helicoid(f64 u, f64 v) noexcept {
    return{u*std::cos(v),u*std::sin(v),v};
}
// Catenoid (minimal): x=cosh(v)*cos(u), y=cosh(v)*sin(u), z=v
[[nodiscard]] constexpr P3 catenoid(f64 u, f64 v) noexcept {
    return{std::cosh(v)*std::cos(u),std::cosh(v)*std::sin(u),v};
}

// ─── Surface of revolution: rotate profile f(t) around z-axis ────────────────
template<typename ProfileFn>
[[nodiscard]] constexpr P3 surface_of_revolution(ProfileFn&& f, f64 t, f64 phi) noexcept {
    auto [r,z]=f(t);
    return{r*std::cos(phi),r*std::sin(phi),z};
}

// ─── First and second fundamental forms ───────────────────────────────────────
// First fundamental form: E=r_u·r_u, F=r_u·r_v, G=r_v·r_v
struct FundamentalForm { f64 E,F,G; };
template<typename SurfFn>
[[nodiscard]] constexpr FundamentalForm first_form(SurfFn&& S, f64 u, f64 v, f64 h=1e-5) noexcept {
    auto pu=S(u+h,v),pm_u=S(u-h,v),pv=S(u,v+h),pm_v=S(u,v-h);
    P3 ru={(pu.x-pm_u.x)/(2*h),(pu.y-pm_u.y)/(2*h),(pu.z-pm_u.z)/(2*h)};
    P3 rv={(pv.x-pm_v.x)/(2*h),(pv.y-pm_v.y)/(2*h),(pv.z-pm_v.z)/(2*h)};
    return{ru.dot(ru),ru.dot(rv),rv.dot(rv)};
}

// Second fundamental form: L,M,N
struct SecFundForm { f64 L,M,N; };
template<typename SurfFn>
[[nodiscard]] constexpr SecFundForm second_form(SurfFn&& S, f64 u, f64 v, f64 h=1e-5) noexcept {
    auto p=S(u,v);
    auto pu=S(u+h,v),pm_u=S(u-h,v),pv=S(u,v+h),pm_v=S(u,v-h);
    auto puu_raw=S(u+h,v),pmm_raw=S(u-h,v);
    P3 ru={(pu.x-pm_u.x)/(2*h),(pu.y-pm_u.y)/(2*h),(pu.z-pm_u.z)/(2*h)};
    P3 rv={(pv.x-pm_v.x)/(2*h),(pv.y-pm_v.y)/(2*h),(pv.z-pm_v.z)/(2*h)};
    P3 n=ru.cross(rv); f64 nl=n.len(); if(nl<1e-12) return{};
    n=n*(1/nl);
    P3 ruu={(puu_raw.x-2*p.x+pmm_raw.x)/(h*h),(puu_raw.y-2*p.y+pmm_raw.y)/(h*h),(puu_raw.z-2*p.z+pmm_raw.z)/(h*h)};
    auto pvv_raw=S(u,v+h),pmv_raw=S(u,v-h);
    P3 rvv={(pvv_raw.x-2*p.x+pmv_raw.x)/(h*h),(pvv_raw.y-2*p.y+pmv_raw.y)/(h*h),(pvv_raw.z-2*p.z+pmv_raw.z)/(h*h)};
    auto puv_pp=S(u+h,v+h),puv_pm=S(u+h,v-h),puv_mp=S(u-h,v+h),puv_mm=S(u-h,v-h);
    P3 ruv={(puv_pp.x-puv_pm.x-puv_mp.x+puv_mm.x)/(4*h*h),
            (puv_pp.y-puv_pm.y-puv_mp.y+puv_mm.y)/(4*h*h),
            (puv_pp.z-puv_pm.z-puv_mp.z+puv_mm.z)/(4*h*h)};
    return{n.dot(ruu),n.dot(ruv),n.dot(rvv)};
}

// Gaussian curvature K=(LN-M²)/(EG-F²)
[[nodiscard]] constexpr f64 gaussian_curvature(const FundamentalForm& I, const SecFundForm& II) noexcept {
    f64 d=I.E*I.G-I.F*I.F;
    return d>1e-15?(II.L*II.N-II.M*II.M)/d:0;
}
// Mean curvature H=(EN-2FM+GL)/(2(EG-F²))
[[nodiscard]] constexpr f64 mean_curvature(const FundamentalForm& I, const SecFundForm& II) noexcept {
    f64 d=I.E*I.G-I.F*I.F;
    return d>1e-15?(I.E*II.N-2*I.F*II.M+I.G*II.L)/(2*d):0;
}
// Principal curvatures: κ = H ± sqrt(H²-K)
[[nodiscard]] constexpr std::pair<f64,f64> principal_curvatures(f64 H, f64 K) noexcept {
    f64 disc=H*H-K; if(disc<0) disc=0;
    return{H-std::sqrt(disc),H+std::sqrt(disc)};
}
} // namespace qs::math
