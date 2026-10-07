#pragma once
// qs_math/orbital_mechanics.hpp — Kepler equation, orbital elements, perturbations (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <cmath>
#include <numbers>
#include <cassert>

namespace qs {

static constexpr f64 G_GRAV = 6.674e-11;   // m³/(kg·s²)
static constexpr f64 M_SUN  = 1.989e30;    // kg
static constexpr f64 AU     = 1.496e11;    // m

// ─── Keplerian orbital elements ───────────────────────────────────────────────
struct OrbitalElements {
    f64 a{1};       // semi-major axis [m]
    f64 e{0};       // eccentricity
    f64 i{0};       // inclination [rad]
    f64 Omega{0};   // longitude of ascending node [rad]
    f64 omega{0};   // argument of periapsis [rad]
    f64 M0{0};      // mean anomaly at epoch [rad]
    f64 t0{0};      // epoch [s]
    f64 mu{G_GRAV*M_SUN}; // GM of central body
};

// ─── Anomaly conversions ──────────────────────────────────────────────────────

// Solve Kepler's equation M = E - e sin(E) for eccentric anomaly E
[[nodiscard]] inline f64 eccentric_anomaly(f64 M, f64 e, u32 max_iter=50, f64 tol=1e-12) {
    M=std::fmod(M,2*std::numbers::pi_v<f64>);
    f64 E=M; // initial guess
    for(u32 i=0;i<max_iter;++i) {
        f64 dE=(M-E+e*std::sin(E))/(1-e*std::cos(E));
        E+=dE;
        if(std::abs(dE)<tol) break;
    }
    return E;
}

// True anomaly from eccentric anomaly
[[nodiscard]] inline f64 true_anomaly_from_E(f64 E, f64 e) {
    return 2*std::atan2(std::sqrt(1+e)*std::sin(E*0.5),std::sqrt(1-e)*std::cos(E*0.5));
}

// Mean anomaly from true anomaly
[[nodiscard]] inline f64 mean_anomaly_from_true(f64 nu, f64 e) {
    f64 E=2*std::atan2(std::sqrt(1-e)*std::sin(nu*0.5),std::sqrt(1+e)*std::cos(nu*0.5));
    return E-e*std::sin(E);
}

// For hyperbolic orbits: M = e sinh(H) - H
[[nodiscard]] inline f64 hyperbolic_anomaly(f64 M, f64 e, u32 max_iter=50, f64 tol=1e-12) {
    f64 H=std::log(2*std::abs(M)/e+1.8); if(M<0) H=-H;
    for(u32 i=0;i<max_iter;++i) {
        f64 dH=(M-e*std::sinh(H)+H)/(e*std::cosh(H)-1);
        H+=dH;
        if(std::abs(dH)<tol) break;
    }
    return H;
}

// ─── Position and velocity from orbital elements ──────────────────────────────
struct StateVector { Vec3<f64> r, v; };

[[nodiscard]] inline StateVector orbital_elements_to_state(const OrbitalElements& oe, f64 t) {
    f64 dt=t-oe.t0;
    f64 n=std::sqrt(oe.mu/(oe.a*oe.a*oe.a)); // mean motion
    f64 M=oe.M0+n*dt;
    f64 E=eccentric_anomaly(M,oe.e);
    f64 nu=true_anomaly_from_E(E,oe.e);
    // Distance
    f64 r_mag=oe.a*(1-oe.e*std::cos(E));
    // Position in orbital plane
    f64 p=oe.a*(1-oe.e*oe.e); // semi-latus rectum
    Vec3<f64> r_orb{r_mag*std::cos(nu), r_mag*std::sin(nu), 0};
    // Velocity in orbital plane
    f64 h=std::sqrt(oe.mu*p);
    Vec3<f64> v_orb{-std::sin(nu)*oe.mu/h, (oe.e+std::cos(nu))*oe.mu/h, 0};
    // Rotation matrices (Euler 3-1-3: Omega, i, omega)
    auto Rz=[](f64 a)->std::array<std::array<f64,3>,3>{
        f64 c=std::cos(a),s=std::sin(a);
        return {{{c,-s,0},{s,c,0},{0,0,1}}};
    };
    auto Rx=[](f64 a)->std::array<std::array<f64,3>,3>{
        f64 c=std::cos(a),s=std::sin(a);
        return {{{1,0,0},{0,c,-s},{0,s,c}}};
    };
    auto mul33=[](const auto& A,const auto& B){
        std::array<std::array<f64,3>,3> C{};
        for(u32 i=0;i<3;++i) for(u32 j=0;j<3;++j) for(u32 k=0;k<3;++k) C[i][j]+=A[i][k]*B[k][j];
        return C;
    };
    auto rot=mul33(Rz(oe.Omega),mul33(Rx(oe.i),Rz(oe.omega)));
    auto apply=[&](Vec3<f64> v_)->Vec3<f64>{
        return {rot[0][0]*v_.x+rot[0][1]*v_.y+rot[0][2]*v_.z,
                rot[1][0]*v_.x+rot[1][1]*v_.y+rot[1][2]*v_.z,
                rot[2][0]*v_.x+rot[2][1]*v_.y+rot[2][2]*v_.z};
    };
    return {apply(r_orb), apply(v_orb)};
}

// ─── State vector to orbital elements (inverse) ────────────────────────────────
[[nodiscard]] inline OrbitalElements state_to_orbital_elements(
    Vec3<f64> r, Vec3<f64> v, f64 mu=G_GRAV*M_SUN)
{
    OrbitalElements oe; oe.mu=mu;
    f64 r_mag=length(r), v_mag=length(v);
    // Specific angular momentum h = r × v
    Vec3<f64> h=cross(r,v);
    f64 h_mag=length(h);
    // Eccentricity vector
    Vec3<f64> e_vec=(v*cross(h,r)*(1/mu))-r*(1/r_mag);
    f64 e_mag=length(e_vec); // Simplified: e=(v^2/mu - 1/r)*r - (r·v)/mu * v
    // Use vis-viva
    f64 E_spec=v_mag*v_mag*0.5-mu/r_mag; // specific mechanical energy
    if(std::abs(E_spec)<1e-15) oe.a=1e15;
    else oe.a=-mu/(2*E_spec);
    oe.e=std::sqrt(std::max(0.0,1+2*E_spec*h_mag*h_mag/(mu*mu)));
    // Inclination
    Vec3<f64> z{0,0,1};
    oe.i=std::acos(std::clamp(dot(h,z)/h_mag,-1.0,1.0));
    // Node vector N = z × h
    Vec3<f64> N=cross(z,h); f64 N_mag=length(N);
    if(N_mag>1e-15){
        oe.Omega=std::acos(std::clamp(N.x/N_mag,-1.0,1.0));
        if(N.y<0) oe.Omega=2*std::numbers::pi_v<f64>-oe.Omega;
    }
    // Argument of periapsis
    if(N_mag>1e-15&&e_mag>1e-10){
        oe.omega=std::acos(std::clamp(dot(N,e_vec)/(N_mag*e_mag),-1.0,1.0));
        if(e_vec.z<0) oe.omega=2*std::numbers::pi_v<f64>-oe.omega;
    }
    // True anomaly
    if(e_mag>1e-10){
        f64 nu=std::acos(std::clamp(dot(e_vec,r)/(e_mag*r_mag),-1.0,1.0));
        if(dot(r,v)<0) nu=2*std::numbers::pi_v<f64>-nu;
        oe.M0=mean_anomaly_from_true(nu,oe.e);
    }
    return oe;
}

// ─── Orbital transfers ────────────────────────────────────────────────────────

// Hohmann transfer: ΔV for two-burn transfer between circular orbits
struct HohmannTransfer {
    f64 delta_v1{0}; // burn 1 [m/s]
    f64 delta_v2{0}; // burn 2 [m/s]
    f64 transfer_time{0}; // seconds
    f64 total_delta_v() const { return std::abs(delta_v1)+std::abs(delta_v2); }
};

[[nodiscard]] inline HohmannTransfer hohmann(f64 r1, f64 r2, f64 mu=G_GRAV*M_SUN) {
    f64 v1=std::sqrt(mu/r1), v2=std::sqrt(mu/r2);
    f64 a_t=0.5*(r1+r2);
    f64 v_t1=std::sqrt(mu*(2/r1-1/a_t));
    f64 v_t2=std::sqrt(mu*(2/r2-1/a_t));
    f64 T=2*std::numbers::pi_v<f64>*std::sqrt(a_t*a_t*a_t/mu);
    return {v_t1-v1, v2-v_t2, T*0.5};
}

// Vis-viva equation: v² = μ(2/r - 1/a)
[[nodiscard]] inline f64 vis_viva(f64 r, f64 a, f64 mu=G_GRAV*M_SUN) {
    return std::sqrt(mu*(2/r-1/a));
}

// ─── Perturbations ────────────────────────────────────────────────────────────

// J2 precession rate (oblate Earth effect) on Ω and ω
// dΩ/dt = -3/2 n J2 (Re/p)² cos(i)
[[nodiscard]] inline f64 j2_nodal_precession(f64 a, f64 e, f64 i,
    f64 mu=3.986004418e14, f64 J2=1.082636e-3, f64 Re=6.3781e6)
{
    f64 p=a*(1-e*e), n=std::sqrt(mu/(a*a*a));
    return -1.5*n*J2*(Re*Re/(p*p))*std::cos(i);
}

[[nodiscard]] inline f64 j2_apsidal_precession(f64 a, f64 e, f64 i,
    f64 mu=3.986004418e14, f64 J2=1.082636e-3, f64 Re=6.3781e6)
{
    f64 p=a*(1-e*e), n=std::sqrt(mu/(a*a*a));
    return 0.75*n*J2*(Re*Re/(p*p))*(5*std::cos(i)*std::cos(i)-1);
}

// Atmospheric drag acceleration: a_drag = -0.5 Cd (A/m) ρ v² v̂
[[nodiscard]] inline Vec3<f64> atmospheric_drag_accel(Vec3<f64> v, f64 Cd, f64 A_over_m, f64 rho) {
    f64 v_mag=length(v);
    if(v_mag<1e-15) return {};
    return v*(-0.5*Cd*A_over_m*rho*v_mag);
}

// Exponential atmosphere density model
[[nodiscard]] inline f64 atmosphere_density(f64 altitude_km) {
    if(altitude_km>1000) return 0;
    // Simple exponential model
    static constexpr f64 H0[]  ={7.249,6.349,6.682,7.554,8.382,7.714,6.549,5.799,5.382,5.877,7.263,9.473,10.587};
    static constexpr f64 rho0[]={1.225,1.112e-1,1.291e-2,1.418e-3,1.643e-4,1.670e-5,8.510e-7,2.790e-8,1.233e-9,5.819e-11,2.418e-12,9.158e-14,3.725e-16};
    static constexpr f64 h0[]  ={0,25,30,40,50,60,70,80,90,100,200,300,500};
    u32 layer=0;
    for(u32 i=0;i<12;++i) if(altitude_km>=h0[i]) layer=i;
    return rho0[layer]*std::exp(-(altitude_km-h0[layer])/H0[layer]);
}

// Tisserand parameter (comet/asteroid orbit classification near Jupiter)
[[nodiscard]] inline f64 tisserand(f64 a, f64 e, f64 i, f64 a_J=5.2) {
    return a_J/a + 2*std::cos(i)*std::sqrt(a/a_J*(1-e*e));
}

} // namespace qs
