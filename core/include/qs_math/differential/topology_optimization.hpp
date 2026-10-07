#pragma once
// qs_math/topology_optimization.hpp — SIMP, sensitivity filter, stress-based removal (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <numeric>

namespace qs {

// ─── 2D FE mesh for SIMP ─────────────────────────────────────────────────────

struct TOProblem {
    u32 nelx{}, nely{};  // number of elements
    f32 volfrac{0.5f};   // target volume fraction
    f32 penal{3.f};      // penalization power (SIMP)
    f32 rmin{1.5f};      // filter radius (elements)
    f32 E0{1.f};         // solid Young's modulus
    f32 Emin{1e-9f};     // void modulus
    f32 nu{0.3f};        // Poisson ratio
};

// Element stiffness matrix contribution (plane stress, bilinear quad)
inline void assemble_ke(f32 ke[8][8], f32 E, f32 nu) {
    f32 k=E/(1.f-nu*nu);
    f32 a=k*(1.f-nu)/2.f, b=k*nu/2.f, c=k*(1.f+nu)/8.f;
    // Simplified 8x8 element stiffness (plane stress Q4)
    static constexpr f32 ke_base[8][8]={
        { 12, 3,-6,-3,-6,-3, 0, 3},
        {  3,12, 3, 0,-3,-6,-3,-6},
        { -6, 3,12,-3, 0,-3, 6,-3},
        { -3, 0,-3,12, 3, 6,-3,-6},
        { -6,-3, 0, 3,12,-3, 6, 3},
        { -3,-6,-3, 6,-3,12, 3, 0},
        {  0,-3, 6,-3, 6, 3,12,-3},
        {  3,-6,-3,-6, 3, 0,-3,12}};
    f32 scale=E/(1.f-nu*nu)/24.f;
    for(u32 i=0;i<8;++i) for(u32 j=0;j<8;++j)
        ke[i][j]=ke_base[i][j]*scale;
}

// SIMP density -> stiffness interpolation
[[nodiscard]] inline f32 simp_modulus(f32 rho, f32 E0, f32 Emin, f32 penal) {
    return Emin+(E0-Emin)*std::pow(rho,penal);
}

// Compliance (objective: minimize)
[[nodiscard]] inline f32 compliance(
    std::span<const f32> U,   // displacement vector
    std::span<const f32> F)   // force vector
{
    f32 c=0.f;
    for(u32 i=0;i<u32(std::min(U.size(),F.size()));++i) c+=F[i]*U[i];
    return c;
}

// ─── Sensitivity filter ───────────────────────────────────────────────────────

inline void sensitivity_filter(
    const std::vector<f32>& dc,   // raw sensitivities
    std::vector<f32>& dc_filt,    // filtered sensitivities
    const std::vector<f32>& rho,  // density
    u32 nelx, u32 nely, f32 rmin)
{
    u32 N=nelx*nely;
    dc_filt.assign(N,0.f);
    for(u32 iy=0;iy<nely;++iy)
    for(u32 ix=0;ix<nelx;++ix){
        u32 i=iy*nelx+ix;
        f32 sum=0.f;
        i32 R=i32(std::ceil(rmin-0.5f));
        for(i32 dy=-R;dy<=R;++dy)
        for(i32 dx=-R;dx<=R;++dx){
            u32 nx2=u32(i32(ix)+dx), ny2=u32(i32(iy)+dy);
            if(nx2>=nelx||ny2>=nely) continue;
            f32 fac=std::max(0.f,rmin-std::sqrt(f32(dx*dx+dy*dy)));
            u32 j=ny2*nelx+nx2;
            sum+=fac; dc_filt[i]+=fac*rho[j]*dc[j];
        }
        dc_filt[i]/=std::max(1e-8f,rho[i]*sum);
    }
}

// ─── OC (Optimality Criteria) update ─────────────────────────────────────────

inline void oc_update(
    std::vector<f32>& rho, const std::vector<f32>& dc_filt,
    f32 volfrac, f32 move=0.2f, f32 eta=0.5f)
{
    u32 N=u32(rho.size());
    // Bisection on Lagrange multiplier
    f32 lo=0.f,hi=1e9f;
    while((hi-lo)/(hi+lo)>1e-4f){
        f32 lam=(lo+hi)*0.5f;
        f32 vol=0.f;
        for(u32 i=0;i<N;++i){
            f32 rho_new=rho[i]*std::pow(std::max(1e-10f,-dc_filt[i]/lam),eta);
            rho_new=std::clamp(rho_new,std::max(1e-3f,rho[i]-move),std::min(1.f,rho[i]+move));
            vol+=rho_new;
        }
        if(vol/f32(N)>volfrac) lo=lam; else hi=lam;
    }
    f32 lam=(lo+hi)*0.5f;
    for(u32 i=0;i<N;++i){
        f32 rho_new=rho[i]*std::pow(std::max(1e-10f,-dc_filt[i]/lam),eta);
        rho[i]=std::clamp(rho_new,std::max(1e-3f,rho[i]-move),std::min(1.f,rho[i]+move));
    }
}

// ─── Stress-based element removal ────────────────────────────────────────────

struct StressElement {
    Vec2f centroid{};
    f32 von_mises{};
    f32 density{1.f};
    u32 id{};
};

[[nodiscard]] inline f32 von_mises_2d(f32 sx, f32 sy, f32 txy) {
    return std::sqrt(sx*sx-sx*sy+sy*sy+3.f*txy*txy);
}

inline void stress_removal_step(
    std::vector<StressElement>& elements,
    f32 target_volfrac, f32 removal_rate=0.02f)
{
    u32 N=u32(elements.size());
    f32 current_vol=0.f;
    for(const auto& e:elements) current_vol+=e.density;
    f32 target=target_volfrac*f32(N);
    if(current_vol<=target) return;

    f32 stress_threshold=0.f;
    std::vector<f32> stresses(N);
    for(u32 i=0;i<N;++i) stresses[i]=elements[i].von_mises;
    std::sort(stresses.begin(),stresses.end());
    f32 remove_vol=(current_vol-target)*removal_rate;
    u32 remove_idx=u32(remove_vol);
    if(remove_idx<N) stress_threshold=stresses[remove_idx];

    for(auto& e:elements)
        if(e.von_mises<stress_threshold) e.density=std::max(0.01f,e.density*0.9f);
}

// ─── Lattice infill generation ────────────────────────────────────────────────

enum class LatticeType { BCC, FCC, Octet, Gyroid, Schwartz };

[[nodiscard]] inline f32 lattice_sdf(const Vec3f& p, LatticeType type, f32 cell_size, f32 thickness) {
    Vec3f q{std::fmod(p.x,cell_size)/cell_size*2.f*k_pi,
             std::fmod(p.y,cell_size)/cell_size*2.f*k_pi,
             std::fmod(p.z,cell_size)/cell_size*2.f*k_pi};
    f32 val;
    switch(type){
    case LatticeType::Gyroid:
        val=std::sin(q.x)*std::cos(q.y)+std::sin(q.y)*std::cos(q.z)+std::sin(q.z)*std::cos(q.x);
        break;
    case LatticeType::Schwartz:
        val=std::cos(q.x)+std::cos(q.y)+std::cos(q.z);
        break;
    case LatticeType::BCC:
        val=std::cos(q.x)+std::cos(q.y)+std::cos(q.z)+std::cos(q.x)*std::cos(q.y)*std::cos(q.z);
        break;
    default:
        val=std::cos(q.x)*std::cos(q.y)+std::cos(q.y)*std::cos(q.z)+std::cos(q.x)*std::cos(q.z);
        break;
    }
    return std::abs(val)-thickness;
}

// Relative density of lattice (approximate)
[[nodiscard]] inline f32 lattice_relative_density(LatticeType type, f32 strut_diameter, f32 cell_size) {
    f32 t=strut_diameter/cell_size;
    switch(type){
    case LatticeType::BCC:   return 3.14f*t*t*std::sqrt(3.f)/2.f;
    case LatticeType::FCC:   return 3.14f*t*t*std::sqrt(2.f)/2.f;
    case LatticeType::Octet: return 6.f*3.14f*t*t/std::sqrt(2.f);
    default: return t*t*k_pi;
    }
}

// ─── Homogenization (effective properties) ────────────────────────────────────

// Voigt upper bound: E_eff = sum(Ei*Vi)
[[nodiscard]] inline f32 voigt_modulus(f32 E1, f32 V1, f32 E2, f32 V2) {
    return E1*V1+E2*V2;
}

// Reuss lower bound: 1/E_eff = sum(Vi/Ei)
[[nodiscard]] inline f32 reuss_modulus(f32 E1, f32 V1, f32 E2, f32 V2) {
    return 1.f/(V1/(E1+1e-15f)+V2/(E2+1e-15f));
}

// Hashin-Shtrikman bounds
[[nodiscard]] inline std::pair<f32,f32> hashin_shtrikman(
    f32 E1, f32 E2, f32 nu1, f32 nu2, f32 V1)
{
    f32 V2=1.f-V1;
    f32 K1=E1/(3.f*(1.f-2.f*nu1)), K2=E2/(3.f*(1.f-2.f*nu2));
    f32 G1=E1/(2.f*(1.f+nu1)),     G2=E2/(2.f*(1.f+nu2));
    // Upper bound (stiffer phase matrix)
    auto Khs=[&](f32 Km,f32 Ki,f32 Gm,f32 Vi)->f32{
        return Km+Vi/(1.f/(Ki-Km)+3.f*(1.f-Vi)/(3.f*Km+4.f*Gm));};
    auto Ghs=[&](f32 Gm,f32 Gi,f32 Km,f32 Vi)->f32{
        return Gm+Vi/(1.f/(Gi-Gm)+6.f*(Km+2.f*Gm)*(1.f-Vi)/(5.f*Gm*(3.f*Km+4.f*Gm)));};
    f32 K_hi=Khs(K2,K1,G2,V1), G_hi=Ghs(G2,G1,K2,V1);
    f32 K_lo=Khs(K1,K2,G1,V2), G_lo=Ghs(G1,G2,K1,V2);
    f32 E_hi=9.f*K_hi*G_hi/(3.f*K_hi+G_hi);
    f32 E_lo=9.f*K_lo*G_lo/(3.f*K_lo+G_lo);
    return {E_lo,E_hi};
}

} // namespace qs
