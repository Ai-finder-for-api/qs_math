#pragma once
// qs_math/game_math.hpp — Health/damage formulas, probability tables, loot math, spawn distribution (C++20)
#include "scalar_types.hpp"
#include "random.hpp"
#include "distributions.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>

namespace qs {

// ─── Damage / health formulas ─────────────────────────────────────────────────

// Diablo-style damage reduction: DR = armour / (armour + k)
[[nodiscard]] inline f32 damage_reduction_ratio(f32 armour, f32 k=100.f) {
    return armour/(armour+k);
}
[[nodiscard]] inline f32 effective_hp(f32 hp, f32 armour, f32 k=100.f) {
    return hp/(1-damage_reduction_ratio(armour,k));
}

// WoW-style: mitigation = min(armour/(armour + 400 + 85*attacker_level), 0.75)
[[nodiscard]] inline f32 wow_mitigation(f32 armour, u32 attacker_level) {
    return std::min(armour/(armour+400+85.f*attacker_level), 0.75f);
}

// Dark Souls-style: absorption stacks multiplicatively
[[nodiscard]] inline f32 stacked_absorption(const std::vector<f32>& absorptions) {
    f32 r=1;
    for(f32 a:absorptions) r*=(1-std::clamp(a,0.f,1.f));
    return 1-r;
}

// XP curve: xp_to_level(n) = base * n^exp
[[nodiscard]] inline u64 xp_for_level(u32 level, f64 base=100, f64 exp_=1.5) {
    return u64(base*std::pow(f64(level),exp_));
}
[[nodiscard]] inline u64 total_xp_for_level(u32 level, f64 base=100, f64 exp_=1.5) {
    u64 total=0;
    for(u32 l=1;l<level;++l) total+=xp_for_level(l,base,exp_);
    return total;
}

// Critical hit: base_dmg * crit_multiplier if roll < crit_chance
[[nodiscard]] inline f32 apply_crit(f32 base_dmg, f32 crit_chance, f32 crit_mult, PCG32& rng) {
    return rng.next_float()<crit_chance?base_dmg*crit_mult:base_dmg;
}

// Dodge/block check
[[nodiscard]] inline bool attack_hits(f32 hit_chance, f32 dodge_chance, f32 block_chance, PCG32& rng) {
    if(rng.next_float()>hit_chance) return false;
    if(rng.next_float()<dodge_chance) return false;
    if(rng.next_float()<block_chance) return false;
    return true;
}

// DPS calculation: dps = dmg * hit_rate * (1 + crit_chance*(crit_mult-1)) / attack_speed
[[nodiscard]] inline f32 dps(f32 base_dmg, f32 attack_speed, f32 hit_chance,
    f32 crit_chance=0.05f, f32 crit_mult=2.f)
{
    f32 avg_dmg=base_dmg*(1+crit_chance*(crit_mult-1));
    return avg_dmg*hit_chance/attack_speed;
}

// Stat scaling (Path of Exile style): value * (1 + sum_increased) * product_more
[[nodiscard]] inline f32 poe_scaling(f32 base, f32 sum_increased, const std::vector<f32>& more_mults) {
    f32 total=base*(1+sum_increased);
    for(f32 m:more_mults) total*=m;
    return total;
}

// Cooldown reduction: effective_cd = base_cd * (1 - cdr), max cdr usually capped
[[nodiscard]] inline f32 cooldown_reduction(f32 base_cd, f32 cdr, f32 max_cdr=0.4f) {
    return base_cd*(1-std::min(cdr,max_cdr));
}

// ─── Loot / drop tables ───────────────────────────────────────────────────────

struct LootEntry {
    std::string name;
    f32 weight{1.f};
    u32 quantity_min{1}, quantity_max{1};
    f32 quality_multiplier{1.f};
};

struct LootTable {
    std::vector<LootEntry> entries;
    f32 nothing_weight{0.f}; // chance to drop nothing
    bool guaranteed{false};   // always drop at least one item

    // Build normalised CDF
    [[nodiscard]] std::vector<f32> build_cdf() const {
        f32 total=nothing_weight;
        for(const auto& e:entries) total+=e.weight;
        std::vector<f32> cdf; f32 acc=0;
        if(!guaranteed){ acc=nothing_weight/total; cdf.push_back(acc); }
        for(const auto& e:entries){ acc+=e.weight/total; cdf.push_back(acc); }
        return cdf;
    }

    // Roll once; returns index into entries (-1 = nothing)
    [[nodiscard]] i32 roll(PCG32& rng) const {
        auto cdf=build_cdf();
        f32 u=rng.next_float();
        for(u32 i=0;i<u32(cdf.size());++i) if(u<cdf[i]) return i32(i)-(guaranteed?0:1);
        return -1;
    }

    // Roll n drops
    [[nodiscard]] std::vector<i32> roll_n(u32 n, PCG32& rng) const {
        std::vector<i32> results(n);
        for(auto& r:results) r=roll(rng);
        return results;
    }
};

// Alias table for O(1) weighted sampling (Vose's algorithm)
struct AliasTable {
    std::vector<f32> prob;
    std::vector<u32> alias;

    void build(const std::vector<f32>& weights) {
        u32 n=u32(weights.size());
        prob.resize(n); alias.resize(n,0);
        f32 total=0; for(auto w:weights) total+=w;
        std::vector<f32> p(n); for(u32 i=0;i<n;++i) p[i]=weights[i]*n/total;
        std::vector<u32> small,large;
        for(u32 i=0;i<n;++i) (p[i]<1.f?small:large).push_back(i);
        while(!small.empty()&&!large.empty()) {
            u32 s=small.back(); small.pop_back();
            u32 l=large.back(); large.pop_back();
            prob[s]=p[s]; alias[s]=l;
            p[l]=(p[l]+p[s])-1.f;
            (p[l]<1.f?small:large).push_back(l);
        }
        for(u32 i:large) prob[i]=1.f;
        for(u32 i:small) prob[i]=1.f;
    }

    [[nodiscard]] u32 sample(PCG32& rng) const {
        u32 n=u32(prob.size());
        u32 i=rng.next_u32()%n;
        return rng.next_float()<prob[i]?i:alias[i];
    }
};

// Loot quality tier (magic/rare/legendary) with escalating weights
[[nodiscard]] inline u32 roll_item_tier(f32 magic_find, PCG32& rng) {
    // tier 0=normal, 1=magic, 2=rare, 3=legendary
    f32 leg =0.001f*(1+magic_find);
    f32 rare=0.05f *(1+magic_find*0.5f);
    f32 mag =0.25f *(1+magic_find*0.2f);
    f32 u=rng.next_float();
    if(u<leg) return 3;
    if(u<leg+rare) return 2;
    if(u<leg+rare+mag) return 1;
    return 0;
}

// ─── Spawn / wave distributions ───────────────────────────────────────────────

// Poisson process spawn: expected spawns in dt given rate λ
[[nodiscard]] inline u32 poisson_spawn_count(f32 lambda_dt, PCG32& rng) {
    // Knuth algorithm
    f32 L=std::exp(-lambda_dt); u32 k=0; f32 p=1;
    do { ++k; p*=rng.next_float(); } while(p>L);
    return k-1;
}

// Spawn on circle perimeter
[[nodiscard]] inline Vec3f spawn_on_circle(Vec3f center, f32 radius, f32 y_offset, PCG32& rng) {
    f32 angle=rng.next_float()*2*f32(std::numbers::pi);
    return {center.x+radius*std::cos(angle), center.y+y_offset, center.z+radius*std::sin(angle)};
}

// Spawn in annulus (donut ring)
[[nodiscard]] inline Vec3f spawn_in_annulus(Vec3f center, f32 r_min, f32 r_max, f32 y, PCG32& rng) {
    f32 angle=rng.next_float()*2*f32(std::numbers::pi);
    f32 r=std::sqrt(r_min*r_min+rng.next_float()*(r_max*r_max-r_min*r_min));
    return {center.x+r*std::cos(angle), y, center.z+r*std::sin(angle)};
}

// Wave composition: weighted mix of enemy types per wave
struct WaveEntry { u32 type_id; u32 count; f32 difficulty; };
[[nodiscard]] inline std::vector<WaveEntry> generate_wave(u32 wave_num, f32 budget) {
    // Budget grows: budget(n) = base * n^1.15
    f32 wave_budget=budget*std::pow(f32(wave_num),1.15f);
    std::vector<WaveEntry> entries;
    // Simple: fill with cheapest units first, then upgrade
    f32 unit_cost=std::pow(1.2f,f32(wave_num/5));
    u32 count=u32(wave_budget/unit_cost);
    entries.push_back({wave_num/5, count, unit_cost});
    if(wave_num%10==0) entries.push_back({10+wave_num/10, 1, wave_budget*0.5f}); // boss
    return entries;
}

// ─── Game economy / balance math ─────────────────────────────────────────────

// Sigmoid power curve for stat scaling (avoids runaway growth)
[[nodiscard]] inline f32 sigmoid_stat(f32 base, f32 growth, u32 level, f32 soft_cap=100.f) {
    f32 linear=base+growth*level;
    return soft_cap*linear/(linear+soft_cap);
}

// Economy inflation model: price(t) = base * (1+rate)^t
[[nodiscard]] inline f64 economy_price(f64 base, f64 rate, u32 time_units) {
    return base*std::pow(1+rate,f64(time_units));
}

// Supply-demand equilibrium price (simple linear model)
[[nodiscard]] inline f32 equilibrium_price(f32 supply, f32 demand, f32 k=1.f) {
    return k*demand/(supply+1e-5f);
}

// Random enchantment rolls: pick m from n affixes with weights
[[nodiscard]] inline std::vector<u32> roll_enchants(u32 n_affixes, u32 n_pick,
    const std::vector<f32>& weights, PCG32& rng)
{
    assert(weights.size()==n_affixes);
    AliasTable at; at.build(weights);
    std::vector<u32> picked;
    std::vector<bool> used(n_affixes,false);
    u32 attempts=0;
    while(picked.size()<n_pick&&attempts<n_pick*10) {
        u32 idx=at.sample(rng);
        if(!used[idx]){ used[idx]=true; picked.push_back(idx); }
        ++attempts;
    }
    return picked;
}

// Breakpoint table (common in ARPGs): stat thresholds that grant integer bonuses
[[nodiscard]] inline u32 breakpoint_value(f32 stat, const std::vector<f32>& thresholds) {
    u32 count=0;
    for(f32 t:thresholds) if(stat>=t) ++count;
    return count;
}

// ─── Procedural character stats ───────────────────────────────────────────────
struct StatBlock {
    f32 str{10},dex{10},int_{10},con{10},wis{10},cha{10};

    // D&D 5e modifier: (stat-10)/2 floored
    [[nodiscard]] i32 modifier(f32 stat) const { return i32(std::floor((stat-10)*0.5f)); }
    [[nodiscard]] i32 str_mod() const { return modifier(str); }
    [[nodiscard]] i32 dex_mod() const { return modifier(dex); }
    [[nodiscard]] i32 int_mod() const { return modifier(int_); }
    [[nodiscard]] i32 con_mod() const { return modifier(con); }

    // 4d6 drop lowest (D&D stat generation)
    static StatBlock roll_4d6_drop_lowest(PCG32& rng) {
        auto roll_stat=[&]()->f32{
            std::array<u32,4> d; for(auto& v:d) v=1+rng.next_u32()%6;
            std::sort(d.begin(),d.end());
            return f32(d[1]+d[2]+d[3]);
        };
        return {roll_stat(),roll_stat(),roll_stat(),roll_stat(),roll_stat(),roll_stat()};
    }

    // Point buy (D&D 5e): 27 points, stats 8-15
    static StatBlock point_buy(u32 pts[6]) {
        static constexpr u32 costs[]={0,0,0,0,0,0,0,0,0,1,2,3,4,5,7,9};
        StatBlock s;
        f32* stats[]={&s.str,&s.dex,&s.int_,&s.con,&s.wis,&s.cha};
        for(u32 i=0;i<6;++i) *stats[i]=f32(std::clamp(pts[i],8u,15u));
        return s;
    }
};

// ─── Fog of war math ──────────────────────────────────────────────────────────
// Visibility in circle radius r, blocked by obstacles
// Raycasting: shoot N rays, mark visible cells
[[nodiscard]] inline std::vector<Vec2i> fow_visible_cells(
    Vec2i origin, f32 radius, u32 rays,
    const std::function<bool(Vec2i)>& blocks_sight,
    f32 cell_size=1.f)
{
    std::vector<Vec2i> visible;
    i32 r=i32(std::ceil(radius/cell_size));
    std::vector<std::vector<bool>> seen(u32(2*r+1),std::vector<bool>(u32(2*r+1),false));
    for(u32 ri=0;ri<rays;++ri) {
        f32 angle=ri*2*f32(std::numbers::pi)/rays;
        f32 dx=std::cos(angle)*cell_size, dy=std::sin(angle)*cell_size;
        f32 x=f32(origin.x), y=f32(origin.y);
        for(f32 t=0;t<radius;t+=cell_size) {
            Vec2i cell{i32(std::round(x)),i32(std::round(y))};
            i32 ox=cell.x-origin.x+r, oy=cell.y-origin.y+r;
            if(ox>=0&&ox<2*r+1&&oy>=0&&oy<2*r+1&&!seen[u32(ox)][u32(oy)]){
                seen[u32(ox)][u32(oy)]=true; visible.push_back(cell);
            }
            if(blocks_sight(cell)) break;
            x+=dx; y+=dy;
        }
    }
    return visible;
}

} // namespace qs
