#pragma once
// qs_math/continuum_mechanics.hpp — Stress/strain tensors, elasticity, deformation gradients (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "tensor_math.hpp"
#include <cmath>
#include <array>

namespace qs {

// ─── Strain tensors ───────────────────────────────────────────────────────────

// Small strain tensor: ε = 0.5*(∇u + (∇u)^T)
[[nodiscard]] inline Mat3f small_strain(const Mat3f& grad_u) {
    // grad_u[i][j] = du_i/dx_j
    return (grad_u + grad_u.transposed()) * 0.5f;
}

// Deformation gradient F = I + ∇u (for large deformation)
[[nodiscard]] inline Mat3f deformation_gradient(const Mat3f& grad_u) {
    return Mat3f::identity() + grad_u;
}

// Right Cauchy-Green strain tensor C = F^T F
[[nodiscard]] inline Mat3f right_cauchy_green(const Mat3f& F) {
    return F.transposed() * F;
}

// Left Cauchy-Green (Finger) strain tensor B = F F^T
[[nodiscard]] inline Mat3f left_cauchy_green(const Mat3f& F) {
    return F * F.transposed();
}

// Green-Lagrange strain E = 0.5*(C - I) = 0.5*(F^T F - I)
[[nodiscard]] inline Mat3f green_lagrange_strain(const Mat3f& F) {
    return (right_cauchy_green(F) - Mat3f::identity()) * 0.5f;
}

// Eulerian Almansi strain e = 0.5*(I - B^{-1})
[[nodiscard]] inline Mat3f almansi_strain(const Mat3f& F) {
    auto B=left_cauchy_green(F);
    auto Binv=B.inversed();
    return (Mat3f::identity() - Binv) * 0.5f;
}

// Volumetric strain J = det(F)
[[nodiscard]] inline f32 volumetric_strain(const Mat3f& F) { return F.determinant(); }

// ─── Stress tensors ───────────────────────────────────────────────────────────

// Voigt notation: symmetric 3×3 tensor as 6-component vector [σ11,σ22,σ33,σ12,σ13,σ23]
using Voigt6 = std::array<f32,6>;

[[nodiscard]] inline Voigt6 mat3_to_voigt(const Mat3f& S) {
    return {S.m[0][0],S.m[1][1],S.m[2][2],S.m[0][1],S.m[0][2],S.m[1][2]};
}
[[nodiscard]] inline Mat3f voigt_to_mat3(const Voigt6& v) {
    return {v[0],v[3],v[4], v[3],v[1],v[5], v[4],v[5],v[2]};
}

// Hydrostatic pressure p = -tr(σ)/3
[[nodiscard]] inline f32 hydrostatic_pressure(const Mat3f& sigma) {
    return -(sigma.m[0][0]+sigma.m[1][1]+sigma.m[2][2])/3.f;
}

// Deviatoric stress s = σ - p*I (traceless)
[[nodiscard]] inline Mat3f deviatoric_stress(const Mat3f& sigma) {
    f32 p=hydrostatic_pressure(sigma);
    Mat3f s=sigma;
    s.m[0][0]+=p; s.m[1][1]+=p; s.m[2][2]+=p;
    return s;
}

// Von Mises equivalent stress σ_VM = sqrt(3/2 * s:s)
[[nodiscard]] inline f32 von_mises_stress(const Mat3f& sigma) {
    Mat3f s=deviatoric_stress(sigma);
    f32 ss=0;
    for(u32 i=0;i<3;++i) for(u32 j=0;j<3;++j) ss+=s.m[i][j]*s.m[i][j];
    return std::sqrt(1.5f*ss);
}

// Tresca yield criterion: max shear stress = (σ_max - σ_min)/2
[[nodiscard]] inline f32 tresca_stress(f32 s1,f32 s2,f32 s3) {
    f32 mn=std::min({s1,s2,s3}), mx=std::max({s1,s2,s3});
    return (mx-mn)*0.5f;
}

// ─── Linear elasticity: constitutive law σ = C : ε ───────────────────────────

// Isotropic elastic stiffness tensor in Voigt form C (6×6)
// Parameterised by Young's modulus E and Poisson's ratio ν
struct ElasticStiffness {
    f32 data[6][6]{};

    static ElasticStiffness isotropic(f32 E, f32 nu) {
        ElasticStiffness C;
        f32 lam=E*nu/((1+nu)*(1-2*nu)); // Lamé first parameter
        f32 mu =E/(2*(1+nu));            // shear modulus
        f32 C11=lam+2*mu, C12=lam, C44=mu;
        // Normal-normal block
        C.data[0][0]=C.data[1][1]=C.data[2][2]=C11;
        C.data[0][1]=C.data[0][2]=C.data[1][0]=C.data[1][2]=C.data[2][0]=C.data[2][1]=C12;
        // Shear block (Voigt: factor 2 in engineering shear)
        C.data[3][3]=C.data[4][4]=C.data[5][5]=C44;
        return C;
    }

    // Apply C : ε (Voigt)
    [[nodiscard]] Voigt6 apply(const Voigt6& eps) const {
        Voigt6 sig{}; for(u32 i=0;i<6;++i) for(u32 j=0;j<6;++j) sig[i]+=data[i][j]*eps[j]; return sig;
    }

    // Bulk modulus K = E/(3*(1-2ν))
    static f32 bulk_modulus(f32 E,f32 nu){ return E/(3*(1-2*nu)); }
    // Shear modulus G = E/(2*(1+ν))
    static f32 shear_modulus(f32 E,f32 nu){ return E/(2*(1+nu)); }
    // P-wave modulus M = K + 4G/3
    static f32 p_wave_modulus(f32 E,f32 nu){ return bulk_modulus(E,nu)+4*shear_modulus(E,nu)/3; }
    // Oedometric (constrained) modulus
    static f32 oedometric_modulus(f32 E,f32 nu){ return E*(1-nu)/((1+nu)*(1-2*nu)); }
};

// Compute stress from strain (linear isotropic)
[[nodiscard]] inline Mat3f hookes_law(const Mat3f& strain, f32 E, f32 nu) {
    auto C=ElasticStiffness::isotropic(E,nu);
    auto eps=mat3_to_voigt(strain);
    auto sig=C.apply(eps);
    return voigt_to_mat3(sig);
}

// ─── Hyperelastic materials ───────────────────────────────────────────────────

// Neo-Hookean: W = mu/2*(I1-3) - mu*log(J) + lambda/2*(log J)^2
struct NeoHookean {
    f32 mu{1.f}, lambda{1.f};

    // First Piola-Kirchhoff stress P = dW/dF
    [[nodiscard]] Mat3f stress_PK1(const Mat3f& F) const {
        f32 J=F.determinant();
        Mat3f Finvt=F.inversed().transposed();
        return F*mu + Finvt*(lambda*std::log(J)-mu);
    }

    // Cauchy stress σ = (1/J) P F^T
    [[nodiscard]] Mat3f cauchy_stress(const Mat3f& F) const {
        f32 J=F.determinant();
        auto P=stress_PK1(F);
        return P*F.transposed()*(1.f/J);
    }

    [[nodiscard]] f32 strain_energy(const Mat3f& F) const {
        f32 J=F.determinant();
        auto C=right_cauchy_green(F);
        f32 I1=C.m[0][0]+C.m[1][1]+C.m[2][2];
        f32 lnJ=std::log(J);
        return mu*0.5f*(I1-3)-mu*lnJ+lambda*0.5f*lnJ*lnJ;
    }
};

// Mooney-Rivlin: W = C1*(I1-3) + C2*(I2-3) + incompressibility
struct MooneyRivlin {
    f32 C1{0.5f}, C2{0.1f};

    [[nodiscard]] f32 strain_energy(const Mat3f& F) const {
        auto C_=right_cauchy_green(F);
        f32 I1=C_.m[0][0]+C_.m[1][1]+C_.m[2][2];
        // I2 = (I1^2 - C:C)/2
        f32 CC=0; for(u32 i=0;i<3;++i) for(u32 j=0;j<3;++j) CC+=C_.m[i][j]*C_.m[i][j];
        f32 I2=0.5f*(I1*I1-CC);
        return C1*(I1-3)+C2*(I2-3);
    }
};

// ─── Failure criteria ──────────────────────────────────────────────────────────
struct YieldCriteria {
    // Von Mises: yield if σ_VM >= σ_y
    static bool von_mises(const Mat3f& sigma, f32 yield_stress) {
        return von_mises_stress(sigma)>=yield_stress;
    }
    // Drucker-Prager: yield if q - p*phi >= d
    static bool drucker_prager(const Mat3f& sigma, f32 phi, f32 d_cohesion) {
        f32 p=-hydrostatic_pressure(sigma);
        f32 q=std::sqrt(1.5f)*von_mises_stress(sigma)/std::sqrt(3.f);
        return q-p*phi>=d_cohesion;
    }
    // Mohr-Coulomb: τ = c + σ_n * tan(φ)
    static bool mohr_coulomb(f32 sigma_n, f32 tau, f32 cohesion, f32 friction_angle) {
        return tau>=cohesion+sigma_n*std::tan(friction_angle);
    }
};

// ─── Viscoelasticity (Maxwell model) ──────────────────────────────────────────
struct MaxwellModel {
    f32 E_spring{1.f}; // elastic modulus
    f32 eta{1.f};      // viscosity

    // Relaxation modulus E(t) = E * exp(-t/τ), τ = η/E
    [[nodiscard]] f32 relaxation_modulus(f32 t) const {
        return E_spring*std::exp(-t*E_spring/eta);
    }
    // Creep compliance J(t) = (1/E + t/η)
    [[nodiscard]] f32 creep_compliance(f32 t) const { return 1/E_spring+t/eta; }
};

// Standard Linear Solid (Zener model): spring in series with Maxwell element
struct ZenerModel {
    f32 E_inf{0.5f};   // long-term modulus
    f32 E1{0.5f};      // Maxwell spring
    f32 eta{1.f};      // Maxwell dashpot
    f32 tau() const { return eta/E1; }
    [[nodiscard]] f32 relaxation_modulus(f32 t) const {
        return E_inf+E1*std::exp(-t/tau());
    }
};

} // namespace qs
