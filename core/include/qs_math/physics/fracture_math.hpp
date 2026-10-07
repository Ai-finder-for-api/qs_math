#pragma once
// qs_math/fracture_math.hpp — Voronoi fracture, crack propagation, brittle/ductile (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>

namespace qs {

// ─── Voronoi fracture sites ───────────────────────────────────────────────────

struct FractureSite {
    Vec3f pos{};
    f32   energy{1.f};   // relative energy / mass of this shard
    u32   id{};
};

inline std::vector<FractureSite> generate_fracture_sites(
    const Vec3f& bbox_min, const Vec3f& bbox_max,
    u32 count, u64 seed = 42)
{
    std::vector<FractureSite> sites;
    sites.reserve(count);
    Pcg32 rng{seed};
    Vec3f range = bbox_max - bbox_min;
    for (u32 i = 0; i < count; ++i) {
        Vec3f p{bbox_min.x + rng.next_f32()*range.x,
                bbox_min.y + rng.next_f32()*range.y,
                bbox_min.z + rng.next_f32()*range.z};
        sites.push_back({p, 1.f, i});
    }
    return sites;
}

// Assign a point to its nearest Voronoi site
[[nodiscard]] inline u32 nearest_fracture_site(
    const Vec3f& p, const std::vector<FractureSite>& sites)
{
    u32 best = 0;
    f32 best_d = length_sq(p - sites[0].pos);
    for (u32 i = 1; i < u32(sites.size()); ++i) {
        f32 d = length_sq(p - sites[i].pos);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

// Voronoi cell membership for a triangle (vote by centroid)
[[nodiscard]] inline u32 triangle_fracture_cell(
    const Vec3f& a, const Vec3f& b, const Vec3f& c,
    const std::vector<FractureSite>& sites)
{
    Vec3f centroid = (a+b+c)*(1.f/3.f);
    return nearest_fracture_site(centroid, sites);
}

// ─── Fracture plane intersection ──────────────────────────────────────────────

// Plane defined by two Voronoi sites: perpendicular bisector
struct FracturePlane {
    Vec3f normal{};
    f32   d{};    // dot(normal, point_on_plane)

    [[nodiscard]] static FracturePlane from_sites(const FractureSite& a, const FractureSite& b) {
        Vec3f n = normalize(b.pos - a.pos);
        Vec3f mid = (a.pos + b.pos) * 0.5f;
        return {n, dot(n, mid)};
    }

    [[nodiscard]] f32 signed_dist(const Vec3f& p) const {
        return dot(normal, p) - d;
    }

    [[nodiscard]] bool on_positive_side(const Vec3f& p) const {
        return signed_dist(p) >= 0.f;
    }
};

// Clip edge against fracture plane, returns intersection point
[[nodiscard]] inline Vec3f clip_edge_plane(
    const Vec3f& a, const Vec3f& b, const FracturePlane& plane)
{
    f32 da = plane.signed_dist(a);
    f32 db = plane.signed_dist(b);
    f32 t  = da / (da - db);
    return a + (b-a)*t;
}

// ─── Crack propagation ────────────────────────────────────────────────────────

struct CrackParams {
    Vec3f impact_pos{};
    Vec3f impact_dir{};
    f32   impact_energy{1.0f};
    f32   material_toughness{0.5f};  // KIc (MPa sqrt(m))
    f32   crack_speed{0.4f};         // fraction of Rayleigh wave speed
    u32   max_branches{3};
    f32   branch_angle{0.35f};       // radians (~20 deg)
    u64   seed{0};
};

struct CrackTip {
    Vec3f pos{};
    Vec3f dir{};
    f32   energy{};
    u32   depth{0};
};

// Propagate crack tip one step, optionally branching
inline void propagate_crack(
    CrackTip& tip, std::vector<CrackTip>& new_tips,
    const CrackParams& p, f32 step_size, Pcg32& rng)
{
    // Reduce energy by material toughness
    f32 K = p.material_toughness;
    tip.energy -= K * step_size;
    if (tip.energy <= 0.f) return;

    // Advance tip
    tip.pos = tip.pos + tip.dir * step_size;

    // Random perturbation
    f32 perturb = (rng.next_f32()-0.5f)*0.2f;
    Vec3f up = (std::abs(tip.dir.y)<0.9f) ? Vec3f{0,1,0} : Vec3f{1,0,0};
    Vec3f side = normalize(cross(tip.dir, up));
    tip.dir = normalize(tip.dir + side*perturb);

    // Branching probability
    if (tip.depth < p.max_branches && tip.energy > K*2.f) {
        f32 branch_prob = 0.05f * (tip.energy / p.impact_energy);
        if (rng.next_f32() < branch_prob) {
            // Create branch tip
            f32 a = p.branch_angle * (rng.next_f32()>0.5f?1.f:-1.f);
            Vec3f bdir{
                tip.dir.x*std::cos(a) - tip.dir.z*std::sin(a),
                tip.dir.y,
                tip.dir.x*std::sin(a) + tip.dir.z*std::cos(a)
            };
            new_tips.push_back({tip.pos, normalize(bdir), tip.energy*0.3f, tip.depth+1});
            tip.energy *= 0.7f;
        }
    }
}

// ─── Stress tensor analysis ───────────────────────────────────────────────────

struct StressTensor {
    f32 s[3][3]{};  // symmetric 3x3

    [[nodiscard]] f32 von_mises() const {
        f32 sx=s[0][0],sy=s[1][1],sz=s[2][2];
        f32 txy=s[0][1],tyz=s[1][2],txz=s[0][2];
        return std::sqrt(0.5f*((sx-sy)*(sx-sy)+(sy-sz)*(sy-sz)+(sz-sx)*(sz-sx)
                              +6.f*(txy*txy+tyz*tyz+txz*txz)));
    }

    [[nodiscard]] f32 max_principal() const {
        // Largest eigenvalue approx via Gershgorin circle
        f32 r0 = std::abs(s[0][1])+std::abs(s[0][2]);
        f32 r1 = std::abs(s[1][0])+std::abs(s[1][2]);
        f32 r2 = std::abs(s[2][0])+std::abs(s[2][1]);
        return std::max({s[0][0]+r0, s[1][1]+r1, s[2][2]+r2});
    }

    [[nodiscard]] Vec3f max_principal_dir() const {
        // Power iteration (one step approximation)
        Vec3f v{s[0][0],s[1][0],s[2][0]};
        f32 l = length(v);
        return l > 1e-8f ? v/l : Vec3f{1,0,0};
    }
};

// ─── Brittle fracture criterion ───────────────────────────────────────────────

struct BrittleMaterial {
    f32 tensile_strength{50.f};   // MPa
    f32 shear_strength{30.f};     // MPa
    f32 youngs_modulus{70e3f};    // MPa (aluminium-like)
    f32 poisson_ratio{0.33f};
    f32 fracture_toughness{23.f}; // MPa*sqrt(mm)
};

[[nodiscard]] inline bool brittle_fracture_criterion(
    const StressTensor& stress, const BrittleMaterial& mat)
{
    return stress.von_mises() >= mat.tensile_strength;
}

// Griffith crack criterion: critical stress for crack of half-length a
[[nodiscard]] inline f32 griffith_critical_stress(f32 crack_half_len,
    const BrittleMaterial& mat, f32 surface_energy=1.0f)
{
    return std::sqrt(2.f * mat.youngs_modulus * surface_energy
                   / (k_pi * crack_half_len));
}

// Stress intensity factor for edge crack
[[nodiscard]] inline f32 stress_intensity_K1(f32 stress, f32 crack_len) {
    return stress * std::sqrt(k_pi * crack_len);
}

// ─── Ductile fracture (plastic deformation before fracture) ──────────────────

struct DuctileMaterial {
    f32 yield_stress{250.f};     // MPa
    f32 ultimate_stress{400.f};  // MPa
    f32 elongation{0.25f};       // fracture strain
    f32 hardening_coeff{0.3f};   // work hardening exponent
};

// Plastic strain after yield (power law hardening)
[[nodiscard]] inline f32 plastic_strain(f32 total_strain,
    const DuctileMaterial& mat, f32 elastic_modulus=200e3f)
{
    f32 elastic_strain = mat.yield_stress / elastic_modulus;
    return std::max(0.f, total_strain - elastic_strain);
}

[[nodiscard]] inline f32 flow_stress(f32 eps_plastic, const DuctileMaterial& mat) {
    return mat.yield_stress * std::pow(1.f + eps_plastic, mat.hardening_coeff);
}

[[nodiscard]] inline bool ductile_fracture(f32 eps_plastic, const DuctileMaterial& mat) {
    return eps_plastic >= mat.elongation;
}

// ─── Fragment dynamics (post-fracture) ────────────────────────────────────────

struct Fragment {
    Vec3f center_mass{};
    Vec3f velocity{};
    Vec3f angular_velocity{};
    f32   mass{1.f};
    u32   cell_id{};
};

inline void apply_impulse_to_fragment(
    Fragment& frag, const Vec3f& impulse, const Vec3f& contact_point)
{
    frag.velocity = frag.velocity + impulse * (1.f / frag.mass);
    Vec3f r = contact_point - frag.center_mass;
    // Approximate I = 2/5 m r^2 sphere
    f32 approx_inertia = 0.4f * frag.mass * 0.01f; // rough
    frag.angular_velocity = frag.angular_velocity + cross(r, impulse) / approx_inertia;
}

[[nodiscard]] inline Vec3f fragment_velocity_at(
    const Fragment& f, const Vec3f& point)
{
    return f.velocity + cross(f.angular_velocity, point - f.center_mass);
}

} // namespace qs
