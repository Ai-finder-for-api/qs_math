#pragma once
// qs_math/multi_physics.hpp — Coupled FEM/SPH, electrostatics, magnetics, heat transfer (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <array>
#include <functional>

namespace qs {

// ─── Heat diffusion ───────────────────────────────────────────────────────────

struct HeatParams {
    f32 thermal_conductivity{50.f};  // W/(m·K) — steel
    f32 density{7800.f};             // kg/m³
    f32 specific_heat{500.f};        // J/(kg·K)
    f32 emissivity{0.8f};            // for radiation
    static constexpr f32 sigma_sb=5.670374419e-8f; // Stefan-Boltzmann

    [[nodiscard]] f32 thermal_diffusivity() const {
        return thermal_conductivity/(density*specific_heat);
    }
};

// 3D finite-difference heat equation step (explicit)
inline void heat_diffuse_step(
    std::vector<f32>& T, u32 nx, u32 ny, u32 nz,
    f32 cell_size, f32 dt, const HeatParams& p,
    f32 ambient_T=293.15f)
{
    f32 alpha=p.thermal_diffusivity();
    f32 r=alpha*dt/(cell_size*cell_size);
    if(r>1.f/6.f){ dt=cell_size*cell_size/(6.f*alpha); r=1.f/6.f; }
    std::vector<f32> Tn=T;
    for(u32 z=1;z<nz-1;++z)
    for(u32 y=1;y<ny-1;++y)
    for(u32 x=1;x<nx-1;++x){
        u32 i=x+nx*(y+ny*z);
        Tn[i]=T[i]+r*(T[i+1]+T[i-1]+T[i+nx]+T[i-nx]+T[i+nx*ny]+T[i-nx*ny]-6.f*T[i]);
    }
    T=Tn;
}

// Newton cooling boundary condition
inline void apply_convection_bc(
    std::vector<f32>& T, u32 nx, u32 ny, u32 nz,
    f32 h_conv, f32 T_inf, f32 dt, const HeatParams& p, f32 cell_size)
{
    // Apply to all boundary faces
    f32 biot=h_conv*cell_size/p.thermal_conductivity;
    f32 factor=1.f-biot;
    auto apply=[&](u32 x,u32 y,u32 z){
        u32 i=x+nx*(y+ny*z);
        T[i]=factor*T[i]+biot*T_inf;
    };
    for(u32 y=0;y<ny;++y) for(u32 x=0;x<nx;++x){apply(x,y,0);apply(x,y,nz-1);}
    for(u32 z=0;z<nz;++z) for(u32 x=0;x<nx;++x){apply(x,0,z);apply(x,ny-1,z);}
    for(u32 z=0;z<nz;++z) for(u32 y=0;y<ny;++y){apply(0,y,z);apply(nx-1,y,z);}
}

// Radiation heat flux: q_rad = epsilon * sigma * (T^4 - T_amb^4)
[[nodiscard]] inline f32 radiation_flux(f32 T_surf, f32 T_amb, const HeatParams& p) {
    return p.emissivity*HeatParams::sigma_sb*(std::pow(T_surf,4.f)-std::pow(T_amb,4.f));
}

// ─── Electrostatics ───────────────────────────────────────────────────────────

static constexpr f32 k_eps0 = 8.8541878e-12f; // F/m
static constexpr f32 k_ke   = 8.9875517e9f;   // N·m²/C²  (1/(4pi eps0))

// Coulomb force between two point charges
[[nodiscard]] inline Vec3f coulomb_force(
    const Vec3f& pos_a, f32 q_a,
    const Vec3f& pos_b, f32 q_b)
{
    Vec3f r=pos_a-pos_b;
    f32 dist2=dot(r,r);
    if(dist2<1e-20f) return {};
    return r*(k_ke*q_a*q_b/(dist2*std::sqrt(dist2)));
}

// Electric field from point charge
[[nodiscard]] inline Vec3f electric_field_point(
    const Vec3f& field_pos, const Vec3f& charge_pos, f32 q)
{
    Vec3f r=field_pos-charge_pos;
    f32 dist2=dot(r,r);
    if(dist2<1e-20f) return {};
    return r*(k_ke*q/(dist2*std::sqrt(dist2)));
}

// Electric potential from collection of charges
[[nodiscard]] inline f32 electric_potential(
    const Vec3f& pos,
    std::span<const Vec3f> charge_positions,
    std::span<const f32> charges)
{
    f32 V=0.f;
    for(u32 i=0;i<u32(std::min(charge_positions.size(),charges.size()));++i){
        f32 r=length(pos-charge_positions[i]);
        if(r<1e-10f) continue;
        V+=k_ke*charges[i]/r;
    }
    return V;
}

// Laplace solver for potential (Jacobi iteration)
inline void laplace_jacobi(
    std::vector<f32>& phi, u32 nx, u32 ny,
    std::span<const bool> dirichlet,
    u32 iters=100)
{
    for(u32 iter=0;iter<iters;++iter){
        std::vector<f32> pn=phi;
        for(u32 y=1;y<ny-1;++y)
        for(u32 x=1;x<nx-1;++x){
            u32 i=y*nx+x;
            if(dirichlet[i]) continue;
            pn[i]=(phi[i+1]+phi[i-1]+phi[i+nx]+phi[i-nx])*0.25f;
        }
        phi=pn;
    }
}

// ─── Magnetostatics ───────────────────────────────────────────────────────────

static constexpr f32 k_mu0 = 1.25663706e-6f; // H/m

// Biot-Savart: dB from current element I*dl at field point r
[[nodiscard]] inline Vec3f biot_savart_element(
    const Vec3f& field_pos, const Vec3f& source_pos,
    const Vec3f& current_dl, f32 I)
{
    Vec3f r=field_pos-source_pos;
    f32 r3=std::pow(dot(r,r),1.5f);
    if(r3<1e-20f) return {};
    return cross(current_dl,r)*(k_mu0*I/(4.f*k_pi*r3));
}

// Magnetic field from a straight wire segment
[[nodiscard]] inline Vec3f wire_B_field(
    const Vec3f& p, const Vec3f& a, const Vec3f& b, f32 I)
{
    Vec3f ab=b-a, ap=p-a;
    Vec3f h=ab*dot(ap,ab)/dot(ab,ab);
    Vec3f perp=ap-h;
    f32 d=length(perp);
    if(d<1e-10f) return {};
    Vec3f dir=normalize(cross(ab,perp));
    f32 L=length(ab);
    f32 cos1=dot(a-p,normalize(ab))/length(a-p);
    f32 cos2=dot(b-p,normalize(ab))/length(b-p);
    return dir*(k_mu0*I*(cos1-cos2)/(4.f*k_pi*d));
}

// Solenoid field (inside, uniform along axis)
[[nodiscard]] inline Vec3f solenoid_B_field(
    f32 n_turns_per_meter, f32 current)
{
    return {0.f, k_mu0*n_turns_per_meter*current, 0.f};
}

// ─── Fluid-structure interaction (simplified) ─────────────────────────────────

struct FSIState {
    Vec3f fluid_velocity{};
    f32   fluid_pressure{};
    Vec3f structure_displacement{};
    Vec3f structure_velocity{};
};

// One FSI coupling step (partitioned staggered)
inline void fsi_coupling_step(
    FSIState& state,
    f32 rho_fluid, f32 area, f32 dt,
    f32 stiffness, f32 damping, f32 mass)
{
    // Fluid force on structure (pressure + drag)
    Vec3f fluid_force = Vec3f{0,0,0};
    fluid_force.x = state.fluid_pressure * area
                  + 0.5f * rho_fluid * state.fluid_velocity.x * std::abs(state.fluid_velocity.x) * area * 0.47f;
    // Structure dynamics (spring-mass-damper)
    Vec3f spring_force = state.structure_displacement * (-stiffness);
    Vec3f damp_force   = state.structure_velocity * (-damping);
    Vec3f total_force  = fluid_force + spring_force + damp_force;
    state.structure_velocity     = state.structure_velocity + total_force * (dt / mass);
    state.structure_displacement = state.structure_displacement + state.structure_velocity * dt;
    // Feedback: structure velocity modifies fluid boundary
    state.fluid_velocity.x -= state.structure_velocity.x * 0.5f; // simplified coupling
}

// ─── Plasma / MHD ─────────────────────────────────────────────────────────────

// Alfvén speed: v_A = B/sqrt(mu0*rho)
[[nodiscard]] inline f32 alfven_speed(f32 B_magnitude, f32 plasma_density) {
    return B_magnitude / std::sqrt(k_mu0 * plasma_density + 1e-30f);
}

// Larmor radius: r_L = m*v_perp / (|q|*B)
[[nodiscard]] inline f32 larmor_radius(
    f32 mass, f32 v_perpendicular, f32 charge, f32 B)
{
    return mass * v_perpendicular / (std::abs(charge) * B + 1e-30f);
}

// Cyclotron frequency: omega_c = |q|*B/m
[[nodiscard]] inline f32 cyclotron_frequency(f32 charge, f32 B, f32 mass) {
    return std::abs(charge) * B / (mass + 1e-30f);
}

// Debye length: lambda_D = sqrt(eps0*kT/(n*e^2))
[[nodiscard]] inline f32 debye_length(f32 temperature_eV, f32 number_density) {
    static constexpr f32 e_charge = 1.60218e-19f;
    f32 kT = temperature_eV * e_charge;
    return std::sqrt(k_eps0 * kT / (number_density * e_charge * e_charge + 1e-30f));
}

// ─── Chemical reaction kinetics ───────────────────────────────────────────────

struct ReactionParams {
    std::vector<f32> rate_constants;  // k_i for each reaction
    std::vector<std::vector<i32>> stoichiometry; // stoich matrix [reaction][species]
};

// ODE right-hand side for mass-action kinetics
[[nodiscard]] inline std::vector<f32> reaction_rhs(
    std::span<const f32> concentrations,
    const ReactionParams& params)
{
    u32 n_species=u32(concentrations.size());
    u32 n_reactions=u32(params.rate_constants.size());
    std::vector<f32> dcdt(n_species,0.f);

    for(u32 r=0;r<n_reactions;++r){
        if(r>=params.stoichiometry.size()) continue;
        // Reaction rate: k * product of reactant concentrations
        f32 rate=params.rate_constants[r];
        for(u32 s=0;s<n_species&&s<params.stoichiometry[r].size();++s)
            if(params.stoichiometry[r][s]<0)
                rate*=std::pow(concentrations[s],-f32(params.stoichiometry[r][s]));
        for(u32 s=0;s<n_species&&s<params.stoichiometry[r].size();++s)
            dcdt[s]+=f32(params.stoichiometry[r][s])*rate;
    }
    return dcdt;
}

// Arrhenius rate constant
[[nodiscard]] inline f32 arrhenius(f32 A, f32 Ea, f32 T) {
    static constexpr f32 R_gas = 8.314f;
    return A * std::exp(-Ea / (R_gas * T));
}

} // namespace qs
