#pragma once
// qs_math/topology/manifold.hpp — Differential geometry on discrete surfaces:
//   cotangent Laplacian, geodesic distance, parallel transport, Gauss curvature (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <cmath>
#include <span>

namespace qs::math::topology {

// ─── Half-edge surface descriptor ────────────────────────────────────────
// Lightweight mirror of the half-edge structure needed for curvature math.
struct HalfEdgeSurface {
    std::span<const Vec3f>   positions;   // vertex positions
    std::span<const u32>     faces;       // triangle indices (3 per face)
    std::span<const u32>     next;        // next half-edge index
    std::span<const u32>     twin;        // twin half-edge (or U32_MAX if boundary)
    std::span<const u32>     he_origin;   // origin vertex of half-edge
    std::span<const u32>     face_he;     // one half-edge per face
    std::span<const u32>     vert_he;     // one outgoing half-edge per vertex

    [[nodiscard]] u32 vertex_count() const noexcept { return u32(positions.size()); }
    [[nodiscard]] u32 face_count()   const noexcept { return u32(faces.size()/3); }
};

// ─── Triangle geometry helpers ────────────────────────────────────────────
struct TriGeom {
    Vec3f a, b, c;
    [[nodiscard]] Vec3f ab() const noexcept { return b-a; }
    [[nodiscard]] Vec3f ac() const noexcept { return c-a; }
    [[nodiscard]] Vec3f bc() const noexcept { return c-b; }
    [[nodiscard]] Vec3f normal() const noexcept { return cross(ab(), ac()); }
    [[nodiscard]] f32   area() const noexcept { return 0.5f*normal().length(); }

    // Cotangent of angle at vertex A = cot(alpha_A)
    [[nodiscard]] f32 cot_A() const noexcept {
        Vec3f u=ab(), v=ac();
        f32 cos_a = dot(u,v);
        f32 sin_a = cross(u,v).length();
        return (std::abs(sin_a) > EPS_F32) ? cos_a/sin_a : 0.f;
    }
    [[nodiscard]] f32 cot_B() const noexcept {
        Vec3f u=a-b, v=c-b;
        f32 cos_b=dot(u,v), sin_b=cross(u,v).length();
        return (std::abs(sin_b) > EPS_F32) ? cos_b/sin_b : 0.f;
    }
    [[nodiscard]] f32 cot_C() const noexcept {
        Vec3f u=a-c, v=b-c;
        f32 cos_c=dot(u,v), sin_c=cross(u,v).length();
        return (std::abs(sin_c) > EPS_F32) ? cos_c/sin_c : 0.f;
    }
    // Voronoi area contribution of vertex A to face
    [[nodiscard]] f32 voronoi_area_A() const noexcept {
        return (1.f/8.f)*(dot(ab(),ab())*cot_C() + dot(ac(),ac())*cot_B());
    }
};

// ─── Discrete Gauss curvature (angle defect) ──────────────────────────────
// K(v) = 2*pi - sum of face angles at v
// Returns per-vertex Gauss curvature (K * A_voronoi) or plain angle defect
[[nodiscard]] inline std::vector<f32> gauss_curvature(
    const std::span<const Vec3f> pos,
    const std::span<const u32>   faces) noexcept
{
    int nv = int(pos.size()), nf = int(faces.size()/3);
    std::vector<f32> angle_sum(nv, 0.f), voronoi_area(nv, 0.f);
    for (int fi = 0; fi < nf; ++fi) {
        u32 ia=faces[fi*3], ib=faces[fi*3+1], ic=faces[fi*3+2];
        TriGeom t{pos[ia], pos[ib], pos[ic]};
        Vec3f ab=t.ab(), ac=t.ac(), ba=-ab, bc=t.bc(), ca=-ac, cb=-bc;
        f32 ang_a = std::acos(clamp(dot(ab.normalized(),ac.normalized()), -1.f, 1.f));
        f32 ang_b = std::acos(clamp(dot(ba.normalized(),bc.normalized()), -1.f, 1.f));
        f32 ang_c = std::acos(clamp(dot(ca.normalized(),cb.normalized()), -1.f, 1.f));
        angle_sum[ia] += ang_a;
        angle_sum[ib] += ang_b;
        angle_sum[ic] += ang_c;
        f32 va = t.voronoi_area_A();
        // Rotate triangle to get voronoi for b,c
        TriGeom tb{pos[ib], pos[ic], pos[ia]};
        TriGeom tc{pos[ic], pos[ia], pos[ib]};
        voronoi_area[ia] += va;
        voronoi_area[ib] += tb.voronoi_area_A();
        voronoi_area[ic] += tc.voronoi_area_A();
    }
    std::vector<f32> K(nv);
    for (int i = 0; i < nv; ++i)
        K[i] = (voronoi_area[i] > EPS_F32) ? (f32(TWO_PI) - angle_sum[i]) / voronoi_area[i] : 0.f;
    return K;
}

// ─── Mean curvature via cotangent Laplacian ───────────────────────────────
// H(v) = (1/2A) * |sum_j (cot_alpha_j + cot_beta_j)(p_j - p_i)|
[[nodiscard]] inline std::vector<f32> mean_curvature(
    const std::span<const Vec3f> pos,
    const std::span<const u32>   faces) noexcept
{
    int nv = int(pos.size()), nf = int(faces.size()/3);
    std::vector<Vec3f> Lp(nv, {0,0,0});
    std::vector<f32>   area(nv, 0.f);

    for (int fi = 0; fi < nf; ++fi) {
        u32 ia=faces[fi*3], ib=faces[fi*3+1], ic=faces[fi*3+2];
        TriGeom t{pos[ia], pos[ib], pos[ic]};
        f32 cA=t.cot_A(), cB=t.cot_B(), cC=t.cot_C();
        f32 A = t.area();
        // cot weight for edge (b,c): cot at a
        Lp[ib] = Lp[ib] + (pos[ic]-pos[ib])*cA;
        Lp[ic] = Lp[ic] + (pos[ib]-pos[ic])*cA;
        Lp[ia] = Lp[ia] + (pos[ic]-pos[ia])*cB;
        Lp[ic] = Lp[ic] + (pos[ia]-pos[ic])*cB;
        Lp[ia] = Lp[ia] + (pos[ib]-pos[ia])*cC;
        Lp[ib] = Lp[ib] + (pos[ia]-pos[ib])*cC;
        area[ia] += A/3.f; area[ib] += A/3.f; area[ic] += A/3.f;
    }
    std::vector<f32> H(nv);
    for (int i = 0; i < nv; ++i) {
        f32 a = area[i];
        H[i] = (a > EPS_F32) ? 0.5f * Lp[i].length() / a : 0.f;
    }
    return H;
}

// ─── Shape operator / principal curvatures ────────────────────────────────
struct PrincipalCurvatures { f32 k1, k2; Vec3f d1, d2; };

[[nodiscard]] inline PrincipalCurvatures principal_curvatures(f32 H, f32 K) noexcept {
    f32 disc = max(0.f, H*H - K);
    f32 sq = std::sqrt(disc);
    return {H + sq, H - sq, {1,0,0}, {0,1,0}}; // directions require full shape op
}

// ─── Geodesic distance (heat method — Crane et al. 2013) ─────────────────
// Step 1: Solve (I - t*L)*u = delta_v0 (heat flow, t = h^2)
// Step 2: Normalize gradient, solve Poisson for distance
// This is the core math kernel; full solve requires a linear solver wrapper.

// Compute divergence of unit gradient field X at vertex v
[[nodiscard]] inline f32 divergence_at_vertex(
    u32 v,
    const std::span<const Vec3f> pos,
    const std::span<const u32>   faces,
    const std::span<const Vec3f> X_field) noexcept
{
    int nf = int(faces.size()/3);
    f32 div = 0.f;
    for (int fi = 0; fi < nf; ++fi) {
        u32 ia=faces[fi*3], ib=faces[fi*3+1], ic=faces[fi*3+2];
        int local = -1;
        if(ia==v) local=0; else if(ib==v) local=1; else if(ic==v) local=2;
        if(local<0) continue;
        u32 vj = faces[fi*3+(local+1)%3], vk = faces[fi*3+(local+2)%3];
        TriGeom t{pos[v], pos[vj], pos[vk]};
        Vec3f ej = pos[vj]-pos[v], ek = pos[vk]-pos[v];
        Vec3f Xf = X_field[fi];
        div += t.cot_C() * dot(ej, Xf) + t.cot_B() * dot(ek, Xf);
    }
    return div * 0.5f;
}

// ─── Parallel transport along edge ────────────────────────────────────────
// Transport vector w from tangent plane at p0 (normal n0) to plane at p1 (normal n1)
[[nodiscard]] inline Vec3f parallel_transport(Vec3f w, Vec3f n0, Vec3f n1) noexcept {
    Vec3f axis = cross(n0, n1);
    f32 sin_a = axis.length();
    f32 cos_a = dot(n0, n1);
    if (sin_a < EPS_F32) return w; // normals parallel
    axis = axis * (1.f/sin_a);
    // Rodrigues rotation of w around axis by angle asin(sin_a)
    return w*cos_a + cross(axis,w)*sin_a + axis*dot(axis,w)*(1.f-cos_a);
}

// ─── Discrete exterior derivative ────────────────────────────────────────
// d0 (gradient): (d0 f)_e = f(v1) - f(v0) for oriented edge e=(v0,v1)
// d1 (curl):     (d1 omega)_f = sum of omega on boundary of face
[[nodiscard]] inline f32 d0_edge(const std::vector<f32>& f, u32 v0, u32 v1) noexcept {
    return f[v1] - f[v0];
}

// ─── Euler characteristic ─────────────────────────────────────────────────
[[nodiscard]] constexpr int euler_characteristic(int V, int E, int F) noexcept {
    return V - E + F; // = 2 for closed genus-0 surface
}
[[nodiscard]] constexpr int genus_from_euler(int chi) noexcept {
    return (2 - chi) / 2;
}

// ─── Conformal parameterization (LSCM gradient) ───────────────────────────
// Per-triangle conformal energy: E_LSCM = area * |dz/dw - conj(dz/dw')|^2
// Returns LSCM energy for a single triangle given UV coords
[[nodiscard]] inline f32 lscm_triangle_energy(
    Vec3f p0, Vec3f p1, Vec3f p2,
    Vec2f uv0, Vec2f uv1, Vec2f uv2) noexcept
{
    // Local 2D basis from 3D triangle
    Vec3f e1 = p1-p0, e2 = p2-p0;
    f32 a = e1.length();
    Vec3f t1 = (a > EPS_F32) ? e1*(1.f/a) : Vec3f{1,0,0};
    Vec3f n  = cross(e1,e2); f32 nl = n.length();
    Vec3f t2 = (nl > EPS_F32) ? cross(n*(1.f/nl), t1) : Vec3f{0,1,0};
    Vec2f lp1 = {dot(e1,t1), dot(e1,t2)};
    Vec2f lp2 = {dot(e2,t1), dot(e2,t2)};
    f32 area2 = lp1.x*lp2.y - lp1.y*lp2.x;
    if (std::abs(area2) < EPS_F32) return 0.f;
    f32 inv_a2 = 1.f / area2;
    // Jacobian columns
    Vec2f duv1 = uv1-uv0, duv2 = uv2-uv0;
    f32 J00=(duv1.x*lp2.y - duv2.x*lp1.y)*inv_a2;
    f32 J01=(duv2.x*lp1.x - duv1.x*lp2.x)*inv_a2;
    f32 J10=(duv1.y*lp2.y - duv2.y*lp1.y)*inv_a2;
    f32 J11=(duv2.y*lp1.x - duv1.y*lp2.x)*inv_a2;
    // LSCM: (J00-J11)^2 + (J01+J10)^2
    f32 d0 = J00-J11, d1 = J01+J10;
    return 0.5f*std::abs(area2)*(d0*d0+d1*d1);
}

} // namespace qs::math::topology
