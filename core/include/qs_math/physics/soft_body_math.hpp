#pragma once
// qs_math/soft_body_math.hpp — FEM tetrahedra, co-rotational elasticity, neo-Hookean (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "mat4.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>

namespace qs {

// ─── Tetrahedron ─────────────────────────────────────────────────────────────

struct Tetrahedron {
    u32 v[4]{};  // vertex indices
};

// Rest-state shape matrix Dm = [p1-p0, p2-p0, p3-p0]
[[nodiscard]] inline Mat3f tet_shape_matrix(
    const Vec3f& p0, const Vec3f& p1, const Vec3f& p2, const Vec3f& p3)
{
    Vec3f c0 = p1-p0, c1 = p2-p0, c2 = p3-p0;
    return Mat3f{c0.x,c1.x,c2.x, c0.y,c1.y,c2.y, c0.z,c1.z,c2.z};
}

[[nodiscard]] inline f32 tet_volume(
    const Vec3f& p0, const Vec3f& p1, const Vec3f& p2, const Vec3f& p3)
{
    return dot(p1-p0, cross(p2-p0, p3-p0)) / 6.f;
}

// Deformation gradient F = Ds * Dm_inv
[[nodiscard]] inline Mat3f deformation_gradient(
    const Vec3f& x0, const Vec3f& x1, const Vec3f& x2, const Vec3f& x3,
    const Mat3f& Dm_inv)
{
    Mat3f Ds = tet_shape_matrix(x0,x1,x2,x3);
    return Ds * Dm_inv;
}

// ─── Green strain tensor ──────────────────────────────────────────────────────

[[nodiscard]] inline Mat3f green_strain(const Mat3f& F) {
    Mat3f Ft = transpose(F);
    Mat3f FtF = Ft * F;
    // E = 0.5*(Ft*F - I)
    return (FtF - Mat3f::identity()) * 0.5f;
}

// ─── Neo-Hookean energy/forces ────────────────────────────────────────────────

struct NeoHookeanMaterial {
    f32 youngs_modulus{1000.f};
    f32 poisson_ratio{0.45f};

    [[nodiscard]] f32 mu() const {
        return youngs_modulus / (2.f*(1.f+poisson_ratio));
    }
    [[nodiscard]] f32 lambda() const {
        return youngs_modulus*poisson_ratio / ((1.f+poisson_ratio)*(1.f-2.f*poisson_ratio));
    }
};

// Neo-Hookean strain energy density W = mu/2*(I1-3) - mu*ln(J) + lambda/2*(ln J)^2
[[nodiscard]] inline f32 neohookean_energy(const Mat3f& F, const NeoHookeanMaterial& mat) {
    f32 J = determinant(F);
    if (J <= 0.f) return 1e10f; // inverted
    Mat3f Ft = transpose(F);
    Mat3f FtF = Ft*F;
    f32 I1 = FtF.m[0][0]+FtF.m[1][1]+FtF.m[2][2]; // trace(Ft*F)
    f32 lnJ = std::log(J);
    f32 mu = mat.mu(), lam = mat.lambda();
    return mu*0.5f*(I1-3.f) - mu*lnJ + lam*0.5f*lnJ*lnJ;
}

// First Piola-Kirchhoff stress P = dW/dF
[[nodiscard]] inline Mat3f neohookean_PK1(const Mat3f& F, const NeoHookeanMaterial& mat) {
    f32 J = determinant(F);
    if (J <= 0.f) return Mat3f{};
    Mat3f Finv_T = transpose(inverse(F));
    f32 lnJ = std::log(J);
    f32 mu = mat.mu(), lam = mat.lambda();
    // P = mu*(F - F^{-T}) + lam*lnJ*F^{-T}
    return F*mu + Finv_T*(lam*lnJ - mu);
}

// ─── Co-rotational elasticity ─────────────────────────────────────────────────

// Polar decomposition F = R*S via Gram-Schmidt + SVD-free approximation
[[nodiscard]] inline std::pair<Mat3f,Mat3f> polar_decompose_approx(const Mat3f& F) {
    // One-shot Gram-Schmidt approximation for near-rotation matrices
    Vec3f c0{F.m[0][0],F.m[1][0],F.m[2][0]};
    Vec3f c1{F.m[0][1],F.m[1][1],F.m[2][1]};
    Vec3f c2{F.m[0][2],F.m[1][2],F.m[2][2]};
    Vec3f r0 = normalize(c0);
    Vec3f r1 = normalize(c1 - r0*dot(r0,c1));
    Vec3f r2 = cross(r0,r1);
    Mat3f R{r0.x,r1.x,r2.x, r0.y,r1.y,r2.y, r0.z,r1.z,r2.z};
    Mat3f S = transpose(R) * F;
    return {R, S};
}

// Corotational elastic forces for a tetrahedron
struct CorotElastic {
    f32 youngs{1000.f};
    f32 poisson{0.35f};

    [[nodiscard]] f32 mu()     const { return youngs/(2.f*(1.f+poisson)); }
    [[nodiscard]] f32 lambda() const { return youngs*poisson/((1.f+poisson)*(1.f-2.f*poisson)); }

    // Elastic force on each vertex (returns f0..f3 as force vectors)
    void compute_forces(
        const Vec3f& x0, const Vec3f& x1, const Vec3f& x2, const Vec3f& x3,
        const Mat3f& Dm_inv, f32 rest_vol,
        Vec3f forces[4]) const
    {
        Mat3f F = deformation_gradient(x0,x1,x2,x3,Dm_inv);
        auto [R,S] = polar_decompose_approx(F);
        Mat3f E = (S + transpose(S))*0.5f - Mat3f::identity();
        // Cauchy stress sigma = 2*mu*E + lambda*tr(E)*I
        f32 trE = E.m[0][0]+E.m[1][1]+E.m[2][2];
        Mat3f sigma = E*(2.f*mu()) + Mat3f::identity()*(lambda()*trE);
        Mat3f P = R * sigma;  // first PK stress (corotational)
        // Forces: f_i = -V * P * (Dm_inv columns)
        Mat3f H = P * transpose(Dm_inv) * (-rest_vol);
        forces[1] = {H.m[0][0],H.m[1][0],H.m[2][0]};
        forces[2] = {H.m[0][1],H.m[1][1],H.m[2][1]};
        forces[3] = {H.m[0][2],H.m[1][2],H.m[2][2]};
        forces[0] = -(forces[1]+forces[2]+forces[3]);
    }
};

// ─── Soft body mesh ───────────────────────────────────────────────────────────

struct SoftBodyParticle {
    Vec3f pos{}, prev_pos{}, vel{};
    f32   inv_mass{1.f};
};

struct SoftBodyMesh {
    std::vector<SoftBodyParticle> particles;
    std::vector<Tetrahedron> tets;

    // Precomputed rest data
    std::vector<Mat3f> Dm_inv;
    std::vector<f32>   rest_vol;

    void precompute_rest() {
        Dm_inv.resize(tets.size());
        rest_vol.resize(tets.size());
        for (u32 t = 0; t < u32(tets.size()); ++t) {
            const auto& tet = tets[t];
            const Vec3f& p0 = particles[tet.v[0]].pos;
            const Vec3f& p1 = particles[tet.v[1]].pos;
            const Vec3f& p2 = particles[tet.v[2]].pos;
            const Vec3f& p3 = particles[tet.v[3]].pos;
            Mat3f Dm = tet_shape_matrix(p0,p1,p2,p3);
            Dm_inv[t] = inverse(Dm);
            rest_vol[t] = std::abs(tet_volume(p0,p1,p2,p3));
        }
    }
};

// ─── Volume preservation constraint (XPBD) ───────────────────────────────────

inline void solve_volume_constraint(
    SoftBodyParticle& a, SoftBodyParticle& b,
    SoftBodyParticle& c, SoftBodyParticle& d,
    f32 rest_vol_val, f32 compliance, f32& lambda, f32 dt)
{
    f32 cur_vol = tet_volume(a.pos,b.pos,c.pos,d.pos);
    f32 C = cur_vol - rest_vol_val;
    Vec3f grad_a = cross(b.pos-c.pos, d.pos-c.pos)*(1.f/6.f);
    Vec3f grad_b = cross(c.pos-a.pos, d.pos-a.pos)*(1.f/6.f);
    Vec3f grad_c = cross(a.pos-b.pos, d.pos-b.pos)*(1.f/6.f);
    Vec3f grad_d = cross(b.pos-a.pos, c.pos-a.pos)*(1.f/6.f);
    f32 alpha = compliance/(dt*dt);
    f32 denom = a.inv_mass*dot(grad_a,grad_a)+b.inv_mass*dot(grad_b,grad_b)
              + c.inv_mass*dot(grad_c,grad_c)+d.inv_mass*dot(grad_d,grad_d)+alpha;
    if (denom < 1e-10f) return;
    f32 dl = (-C - alpha*lambda)/denom;
    lambda += dl;
    a.pos = a.pos + grad_a*(a.inv_mass*dl);
    b.pos = b.pos + grad_b*(b.inv_mass*dl);
    c.pos = c.pos + grad_c*(c.inv_mass*dl);
    d.pos = d.pos + grad_d*(d.inv_mass*dl);
}

// ─── Shape matching (PBD soft body shortcut) ─────────────────────────────────

[[nodiscard]] inline Vec3f shape_match_center(
    std::span<const SoftBodyParticle> pts)
{
    Vec3f c{0,0,0}; f32 total_m = 0.f;
    for (const auto& p : pts) {
        f32 m = p.inv_mass > 0.f ? 1.f/p.inv_mass : 1e10f;
        c = c + p.pos*m; total_m += m;
    }
    return c*(1.f/total_m);
}

} // namespace qs
