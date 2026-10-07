#pragma once
// qs_math/combat_math.hpp — Ballistics, hit probability, damage falloff, suppression (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "random.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <span>

namespace qs {

// ─── Ballistics ───────────────────────────────────────────────────────────────

struct Projectile {
    Vec3f pos{};
    Vec3f vel{};
    f32   mass{0.01f};      // kg
    f32   drag_coeff{0.47f};// Cd (sphere)
    f32   cross_section{};  // m^2
    f32   lifetime{10.f};   // s
    f32   age{0.f};
};

struct AtmosphericConditions {
    f32 air_density{1.225f}; // kg/m^3 at sea level
    Vec3f wind{};
};

// 4th-order Runge-Kutta ballistic integration
inline void ballistic_step(Projectile& p, const AtmosphericConditions& atm, f32 dt) {
    auto accel=[&](const Vec3f& v)->Vec3f{
        Vec3f rel=v-atm.wind;
        f32 spd=length(rel);
        Vec3f drag_dir=spd>1e-6f?-rel/spd:Vec3f{};
        f32 drag_mag=0.5f*atm.air_density*spd*spd*p.drag_coeff*p.cross_section/p.mass;
        return Vec3f{0,-9.81f,0}+drag_dir*drag_mag;
    };
    Vec3f k1v=accel(p.vel);
    Vec3f k2v=accel(p.vel+k1v*dt*0.5f);
    Vec3f k3v=accel(p.vel+k2v*dt*0.5f);
    Vec3f k4v=accel(p.vel+k3v*dt);
    p.vel=p.vel+(k1v+k2v*2.f+k3v*2.f+k4v)*(dt/6.f);
    p.pos=p.pos+p.vel*dt;
    p.age+=dt;
}

// Predict time of flight to a distance (iterative)
[[nodiscard]] inline f32 tof_to_range(
    const Vec3f& muzzle_vel, f32 range, f32 drag=0.f, f32 dt=0.01f)
{
    Vec3f v=muzzle_vel, p{};
    for(f32 t=0.f;t<30.f;t+=dt){
        Vec3f spd_v=v; f32 spd=length(spd_v);
        v.y-=9.81f*dt;
        if(drag>0.f) v=v-spd_v*(drag*spd*dt);
        p=p+v*dt;
        if(length(Vec2f{p.x,p.z})>=range) return t;
    }
    return 30.f;
}

// Lead angle for moving target
[[nodiscard]] inline Vec3f compute_lead(
    const Vec3f& shooter, const Vec3f& target,
    const Vec3f& target_vel, f32 projectile_speed)
{
    Vec3f d=target-shooter;
    f32 dist=length(d);
    f32 tof=dist/projectile_speed;
    Vec3f predicted=target+target_vel*tof;
    return normalize(predicted-shooter);
}

// ─── Hit probability ─────────────────────────────────────────────────────────

struct AccuracyParams {
    f32 base_spread_rad{0.001f};  // inherent weapon accuracy (radians)
    f32 movement_spread{0.005f};  // additional spread when moving
    f32 crouch_modifier{0.5f};    // multiplier when crouched
    f32 prone_modifier{0.25f};
    f32 recoil_spread{0.003f};    // per shot
    f32 recoil_recovery{5.f};     // rad/s
    f32 suppression_spread{0.01f};// under fire
};

struct ShooterState {
    f32 movement_speed{0.f};
    bool crouched{false}, prone{false};
    f32 current_recoil{0.f};
    f32 suppression{0.f};

    [[nodiscard]] f32 effective_spread(const AccuracyParams& p) const {
        f32 s=p.base_spread_rad;
        if(movement_speed>0.1f) s+=p.movement_spread*movement_speed;
        if(prone) s*=p.prone_modifier;
        else if(crouched) s*=p.crouch_modifier;
        s+=current_recoil;
        s+=suppression*p.suppression_spread;
        return s;
    }

    void fire(const AccuracyParams& p){current_recoil+=p.recoil_spread;}
    void update(f32 dt,const AccuracyParams& p){
        current_recoil=std::max(0.f,current_recoil-p.recoil_recovery*dt);
        suppression=std::max(0.f,suppression-dt);
    }
};

// Circular error probable (CEP): radius containing 50% of shots
[[nodiscard]] inline f32 cep_from_spread(f32 spread_rad, f32 range) {
    f32 sigma_linear=std::tan(spread_rad)*range;
    return sigma_linear*1.1774f; // 50th percentile of Rayleigh distribution
}

// P(hit) for circular target of radius r at range d with spread s
[[nodiscard]] inline f32 p_hit_circular(f32 target_r, f32 range, f32 spread_rad) {
    f32 sigma=std::tan(spread_rad)*range;
    f32 x=target_r/sigma;
    return 1.f-std::exp(-x*x*0.5f); // Rayleigh CDF approximation
}

// Sample shot direction (Gaussian spread)
[[nodiscard]] inline Vec3f sample_shot_dir(
    const Vec3f& aim_dir, f32 spread_rad, Pcg32& rng)
{
    // Box-Muller for 2D Gaussian offset
    f32 u1=std::max(1e-10f,rng.next_f32()), u2=rng.next_f32();
    f32 r=std::sqrt(-2.f*std::log(u1))*spread_rad;
    f32 theta=k_tau*u2;
    Vec3f up=(std::abs(aim_dir.y)<0.9f)?Vec3f{0,1,0}:Vec3f{1,0,0};
    Vec3f right=normalize(cross(aim_dir,up));
    Vec3f upv=cross(right,aim_dir);
    return normalize(aim_dir+right*(r*std::cos(theta))+upv*(r*std::sin(theta)));
}

// ─── Damage model ─────────────────────────────────────────────────────────────

struct DamageParams {
    f32 base_damage{50.f};
    f32 min_damage{10.f};
    f32 max_range{200.f};    // full damage
    f32 min_range{5.f};      // damage falloff start
    f32 falloff_power{2.f};
    f32 armour_penetration{30.f}; // mm RHA equivalent
};

[[nodiscard]] inline f32 damage_at_range(f32 dist, const DamageParams& p) {
    if(dist<=p.min_range) return p.base_damage;
    if(dist>=p.max_range) return p.min_damage;
    f32 t=(dist-p.min_range)/(p.max_range-p.min_range);
    return p.base_damage+(p.min_damage-p.base_damage)*std::pow(t,p.falloff_power);
}

[[nodiscard]] inline f32 armour_mitigation(f32 armour_mm, f32 penetration_mm, f32 impact_angle_rad=0.f) {
    // Effective armour increases with oblique angle (slope factor)
    f32 effective_armour=armour_mm/std::max(0.1f,std::cos(impact_angle_rad));
    f32 pen_ratio=penetration_mm/effective_armour;
    if(pen_ratio>=1.f) return 0.f;  // penetrated
    if(pen_ratio<=0.f) return 1.f;  // blocked
    return 1.f-pen_ratio*pen_ratio; // partial damage
}

// Critical hit zones (hit location multipliers)
struct HitLocation {
    const char* name{};
    f32 damage_mult{1.f};
    f32 area_fraction{};  // fraction of total hitbox area
};

static constexpr std::array<HitLocation,5> k_humanoid_locations = {{
    {"head",   3.0f, 0.06f},
    {"torso",  1.5f, 0.35f},
    {"arms",   0.7f, 0.20f},
    {"legs",   0.8f, 0.30f},
    {"hands",  0.5f, 0.09f}
}};

[[nodiscard]] inline u32 random_hit_location(Pcg32& rng) {
    f32 r=rng.next_f32(), acc=0.f;
    for(u32 i=0;i<k_humanoid_locations.size();++i){
        acc+=k_humanoid_locations[i].area_fraction;
        if(r<acc) return i;
    }
    return u32(k_humanoid_locations.size()-1);
}

// ─── Suppression ─────────────────────────────────────────────────────────────

struct SuppressionSystem {
    f32 value{0.f};           // 0=none, 1=fully suppressed
    f32 decay{0.5f};          // per second
    f32 near_miss_add{0.3f};  // per near miss
    f32 hit_add{0.5f};        // per hit nearby
    f32 sustained_add{0.1f};  // per bullet in vicinity per second

    void near_miss(){ value=std::min(1.f,value+near_miss_add); }
    void hit_nearby(){ value=std::min(1.f,value+hit_add); }
    void sustained_fire(f32 rounds_per_sec,f32 dt){ value=std::min(1.f,value+sustained_add*rounds_per_sec*dt); }
    void update(f32 dt){ value=std::max(0.f,value-decay*dt); }

    [[nodiscard]] f32 movement_penalty() const { return value*0.5f; }
    [[nodiscard]] f32 accuracy_penalty() const { return value; }
    [[nodiscard]] bool is_suppressed() const { return value>0.5f; }
};

// ─── Cover rating ─────────────────────────────────────────────────────────────

[[nodiscard]] inline f32 cover_rating(
    const Vec3f& shooter, const Vec3f& target,
    std::span<const std::pair<Vec3f,f32>> cover_objects) // (center, radius)
{
    f32 coverage=0.f;
    Vec3f dir=normalize(target-shooter);
    for(const auto& [c,r]:cover_objects){
        Vec3f to_cover=c-shooter;
        f32 proj=dot(to_cover,dir);
        if(proj<0.f) continue;
        Vec3f closest=shooter+dir*proj;
        f32 d=length(c-closest);
        if(d<r) coverage=std::min(1.f,coverage+r/(d+r));
    }
    return coverage;
}

} // namespace qs
