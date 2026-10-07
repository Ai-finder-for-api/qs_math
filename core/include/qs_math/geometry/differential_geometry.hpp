#pragma once
// qs_math/geometry/differential_geometry.hpp — Curves, torsion, Frenet frame, Gauss-Bonnet
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

struct DGV3 { f64 x,y,z;
    [[nodiscard]] constexpr DGV3 operator+(DGV3 b) const noexcept{return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr DGV3 operator-(DGV3 b) const noexcept{return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr DGV3 operator*(f64 s)  const noexcept{return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64  dot(DGV3 b) const noexcept{return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr DGV3 cross(DGV3 b) const noexcept{return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    [[nodiscard]] constexpr f64  len2() const noexcept{return dot(*this);}
    [[nodiscard]]           f64  len()  const noexcept{return std::sqrt(len2());}
    [[nodiscard]]           DGV3 norm() const noexcept{f64 l=len();return l>1e-12?DGV3{x/l,y/l,z/l}:*this;}
};

// ─── Space curve (parametric) ─────────────────────────────────────────────────
template<typename Curve>
struct FrenetFrame {
    DGV3 T{},N{},B{}; // tangent, normal, binormal
    f64 kappa{0},tau{0}; // curvature, torsion

    // Compute Frenet frame at parameter t
    constexpr void compute(Curve&& gamma, f64 t, f64 h=1e-5) noexcept {
        auto gp=gamma(t+h),gm=gamma(t-h);
        auto gpp=gamma(t+h+h),gmm=gamma(t-h-h);
        // First derivative
        DGV3 r1={(gp.x-gm.x)/(2*h),(gp.y-gm.y)/(2*h),(gp.z-gm.z)/(2*h)};
        // Second derivative
        auto g0=gamma(t);
        DGV3 r2={(gpp.x-2*g0.x+gmm.x)/(4*h*h),(gpp.y-2*g0.y+gmm.y)/(4*h*h),(gpp.z-2*g0.z+gmm.z)/(4*h*h)};
        // Third derivative
        auto gp3h=gamma(t+3*h/2),gm3h=gamma(t-3*h/2);
        DGV3 r3={(gp3h.x-3*gp.x+3*gm.x-gm3h.x)/(h*h*h/4),
                 (gp3h.y-3*gp.y+3*gm.y-gm3h.y)/(h*h*h/4),
                 (gp3h.z-3*gp.z+3*gm.z-gm3h.z)/(h*h*h/4)};
        f64 speed=r1.len();
        if(speed<1e-12) return;
        T=r1.norm();
        DGV3 r1xr2=r1.cross(r2);
        f64 r1xr2_len=r1xr2.len();
        kappa=r1xr2_len/(speed*speed*speed);
        if(r1xr2_len>1e-12) {
            B=r1xr2.norm();
            N=B.cross(T);
            // Torsion: τ = (r' × r'')·r''' / |r' × r''|²
            tau=r1xr2.dot(r3)/std::max(r1xr2_len*r1xr2_len,1e-30);
        }
    }
};

// ─── Curvature of a 2D curve y=f(x) ──────────────────────────────────────────
template<typename F>
[[nodiscard]] constexpr f64 curvature_2d(F&& f, f64 x, f64 h=1e-5) noexcept {
    f64 fp=(f(x+h)-f(x-h))/(2*h);
    f64 fpp=(f(x+h)-2*f(x)+f(x-h))/(h*h);
    return fpp/std::pow(1+fp*fp,1.5);
}

// ─── Evolute of a curve (center of curvature locus) ──────────────────────────
template<typename Curve>
[[nodiscard]] constexpr std::pair<f64,f64> evolute_2d(Curve&& gamma, f64 t, f64 h=1e-5) noexcept {
    auto p=gamma(t),pp=gamma(t+h),pm=gamma(t-h);
    f64 dx=(pp.x-pm.x)/(2*h),dy=(pp.y-pm.y)/(2*h);
    f64 ddx=(pp.x-2*p.x+pm.x)/(h*h),ddy=(pp.y-2*p.y+pm.y)/(h*h);
    f64 kappa=(dx*ddy-dy*ddx)/std::pow(dx*dx+dy*dy,1.5);
    f64 R=std::abs(kappa)>1e-15?1/kappa:1e15;
    f64 len=std::sqrt(dx*dx+dy*dy);
    f64 nx=-dy/len, ny=dx/len; // unit normal
    return{p.x+R*nx,p.y+R*ny};
}

// ─── Connection 1-form and parallel transport ──────────────────────────────────
// Parallel transport of vector V along curve γ: DV/dt = 0
// ∂V^k/∂t + Γ^k_{ij} V^i (∂γ^j/∂t) = 0
template<int N>
[[nodiscard]] constexpr std::array<f64,N> parallel_transport_step(
    const std::array<f64,N>& V,
    const std::array<f64,N>& gamma_dot, // dγ/dt
    const std::array<std::array<std::array<f64,N>,N>,N>& Gamma, // Christoffel
    f64 dt) noexcept
{
    std::array<f64,N> dV{};
    for(int k=0;k<N;++k)
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            dV[k]-=Gamma[k][i][j]*V[i]*gamma_dot[j];
    std::array<f64,N> Vnew=V;
    for(int k=0;k<N;++k) Vnew[k]+=dV[k]*dt;
    return Vnew;
}

// ─── Gauss-Bonnet theorem ─────────────────────────────────────────────────────
// ∫∫_M K dA + ∫_∂M κ_g ds = 2π χ(M)
// For compact surface without boundary: ∫∫ K dA = 2π χ(M)
[[nodiscard]] constexpr f64 gauss_bonnet_euler_char(f64 total_gaussian_curvature) noexcept {
    return total_gaussian_curvature/(2*3.14159265358979323846);
}
// Discrete Gauss-Bonnet (angle defect)
[[nodiscard]] constexpr f64 angle_defect(const f64* face_angles, int n_faces) noexcept {
    f64 sum=0; for(int i=0;i<n_faces;++i) sum+=face_angles[i];
    return 2*3.14159265358979323846-sum;
}

// ─── Exponential map at a point on a surface ──────────────────────────────────
// exp_p(v): geodesic starting at p with tangent vector v, evaluated at t=1
template<int N, typename MetricFn>
[[nodiscard]] constexpr std::array<f64,N> exp_map(
    MetricFn&& g_fn,  // metric at a point
    const std::array<f64,N>& p,
    const std::array<f64,N>& v,
    int steps=100) noexcept
{
    f64 dt=1.0/steps;
    std::array<f64,N> pos=p, vel=v;
    for(int step=0;step<steps;++step) {
        auto g=g_fn(pos);
        // Christoffel symbols via metric (numerical)
        std::array<f64,N> acc{};
        for(int k=0;k<N;++k) for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
            // Approximate Γ^k_{ij} ≈ 0 for flat metric (first order)
            // Full impl: use metric derivatives
            acc[k]-=0*vel[i]*vel[j]; // placeholder
        }
        for(int i=0;i<N;++i){pos[i]+=vel[i]*dt;vel[i]+=acc[i]*dt;}
    }
    return pos;
}

// ─── Logarithmic map ──────────────────────────────────────────────────────────
// log_p(q): tangent vector at p pointing toward q with |v|=geodesic distance
template<int N>
[[nodiscard]] constexpr std::array<f64,N> log_map_euclidean(
    const std::array<f64,N>& p,
    const std::array<f64,N>& q) noexcept
{
    std::array<f64,N> v{};
    for(int i=0;i<N;++i) v[i]=q[i]-p[i];
    return v;
}

// ─── Covariant derivative ─────────────────────────────────────────────────────
template<int N, typename VField>
[[nodiscard]] constexpr std::array<f64,N> covariant_derivative(
    VField&& V_field,
    const std::array<f64,N>& point,
    const std::array<f64,N>& direction,
    const std::array<std::array<std::array<f64,N>,N>,N>& Gamma,
    f64 h=1e-5) noexcept
{
    // ∇_X V^k = X^i ∂_i V^k + Γ^k_{ij} X^i V^j
    auto V0=V_field(point);
    std::array<f64,N> result{};
    for(int k=0;k<N;++k) {
        // Directional derivative ∂_X V^k
        std::array<f64,N> pp=point,pm=point;
        for(int i=0;i<N;++i){pp[i]+=h*direction[i];pm[i]-=h*direction[i];}
        auto Vp=V_field(pp),Vm=V_field(pm);
        result[k]=(Vp[k]-Vm[k])/(2*h);
        // Christoffel correction
        for(int i=0;i<N;++i) for(int j=0;j<N;++j)
            result[k]+=Gamma[k][i][j]*direction[i]*V0[j];
    }
    return result;
}
} // namespace qs::math
