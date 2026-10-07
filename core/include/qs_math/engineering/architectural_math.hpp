#pragma once
// qs_math/architectural_math.hpp — Floor plan gen, structural load, room acoustics (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <string>
#include <span>

namespace qs {

// ─── Room ─────────────────────────────────────────────────────────────────────

struct Room {
    Vec2f origin{};
    Vec2f size{};
    std::string label{};
    u32 id{};

    [[nodiscard]] f32 area() const { return size.x*size.y; }
    [[nodiscard]] Vec2f center() const { return origin+size*0.5f; }
    [[nodiscard]] bool overlaps(const Room& o) const {
        return !(origin.x+size.x<=o.origin.x||o.origin.x+o.origin.x<=origin.x||
                 origin.y+size.y<=o.origin.y||o.origin.y+o.size.y<=origin.y);
    }
    [[nodiscard]] bool contains(const Vec2f& p) const {
        return p.x>=origin.x&&p.x<=origin.x+size.x&&p.y>=origin.y&&p.y<=origin.y+size.y;
    }
};

// ─── BSP floor plan ───────────────────────────────────────────────────────────

struct BSPNode {
    Vec2f origin{}, size{};
    std::unique_ptr<BSPNode> left{}, right{};
    Room room;
    bool has_room{false};

    [[nodiscard]] bool is_leaf() const { return !left&&!right; }
};

inline std::unique_ptr<BSPNode> bsp_split(
    Vec2f origin, Vec2f size, u32 depth, u32 max_depth,
    f32 min_room_ratio=0.45f, Pcg32& rng={})
{
    auto node=std::make_unique<BSPNode>();
    node->origin=origin; node->size=size;

    if(depth>=max_depth||size.x<8.f||size.y<8.f){
        // Create leaf room with inset walls
        f32 inset_x=size.x*0.1f+rng.next_f32()*size.x*0.1f;
        f32 inset_y=size.y*0.1f+rng.next_f32()*size.y*0.1f;
        node->room={
            {origin.x+inset_x,origin.y+inset_y},
            {size.x-inset_x*2.f,size.y-inset_y*2.f},
            "",u32(depth)};
        node->has_room=true;
        return node;
    }

    bool split_h=size.x<=size.y;
    if(size.x>size.y*1.25f) split_h=false;
    else if(size.y>size.x*1.25f) split_h=true;
    else split_h=rng.next_f32()>0.5f;

    f32 split_min=min_room_ratio;
    f32 split_max=1.f-min_room_ratio;
    f32 split_t=split_min+(split_max-split_min)*rng.next_f32();

    if(split_h){
        f32 split_y=size.y*split_t;
        node->left=bsp_split(origin,{size.x,split_y},depth+1,max_depth,min_room_ratio,rng);
        node->right=bsp_split({origin.x,origin.y+split_y},{size.x,size.y-split_y},depth+1,max_depth,min_room_ratio,rng);
    } else {
        f32 split_x=size.x*split_t;
        node->left=bsp_split(origin,{split_x,size.y},depth+1,max_depth,min_room_ratio,rng);
        node->right=bsp_split({origin.x+split_x,origin.y},{size.x-split_x,size.y},depth+1,max_depth,min_room_ratio,rng);
    }
    return node;
}

inline void bsp_collect_rooms(const BSPNode* node, std::vector<Room>& rooms){
    if(!node) return;
    if(node->has_room){rooms.push_back(node->room);return;}
    bsp_collect_rooms(node->left.get(),rooms);
    bsp_collect_rooms(node->right.get(),rooms);
}

[[nodiscard]] inline std::vector<Room> generate_floor_plan(
    f32 width, f32 height, u32 max_depth=5, u64 seed=42)
{
    Pcg32 rng{seed};
    auto root=bsp_split({0,0},{width,height},0,max_depth,0.4f,rng);
    std::vector<Room> rooms;
    bsp_collect_rooms(root.get(),rooms);
    for(u32 i=0;i<u32(rooms.size());++i) rooms[i].id=i;
    return rooms;
}

// ─── Structural load (simple beam theory) ────────────────────────────────────

struct BeamSection {
    f32 width{0.3f};   // m
    f32 height{0.5f};  // m
    f32 E{30e9f};      // Young's modulus (Pa), concrete default

    [[nodiscard]] f32 moment_of_inertia() const { return width*height*height*height/12.f; }
    [[nodiscard]] f32 section_modulus()   const { return moment_of_inertia()/(height*0.5f); }
};

// Simply supported beam, UDL: max deflection and moment
[[nodiscard]] inline f32 simply_supported_udl_deflection(
    f32 w,    // N/m distributed load
    f32 L,    // span (m)
    const BeamSection& beam)
{
    return 5.f*w*L*L*L*L/(384.f*beam.E*beam.moment_of_inertia());
}

[[nodiscard]] inline f32 simply_supported_udl_max_moment(f32 w, f32 L) {
    return w*L*L/8.f;
}

[[nodiscard]] inline f32 simply_supported_udl_max_stress(f32 w, f32 L, const BeamSection& beam) {
    return simply_supported_udl_max_moment(w,L)/beam.section_modulus();
}

// Cantilever beam, point load at tip
[[nodiscard]] inline f32 cantilever_point_deflection(f32 P, f32 L, const BeamSection& beam) {
    return P*L*L*L/(3.f*beam.E*beam.moment_of_inertia());
}

// Column buckling (Euler)
[[nodiscard]] inline f32 euler_buckling_load(
    const BeamSection& beam, f32 length, f32 k=1.f) // k=1 pinned-pinned
{
    return k_pi*k_pi*beam.E*beam.moment_of_inertia()/(k*length*k*length);
}

// ─── Room acoustics (Sabine / Eyring) ────────────────────────────────────────

struct AcousticMaterial {
    const char* name{};
    f32 alpha_125{},alpha_250{},alpha_500{},alpha_1k{},alpha_2k{},alpha_4k{};
};

static constexpr AcousticMaterial k_concrete_bare{"bare concrete",0.01f,0.01f,0.02f,0.02f,0.03f,0.03f};
static constexpr AcousticMaterial k_carpet_thick {"thick carpet", 0.02f,0.06f,0.14f,0.37f,0.60f,0.65f};
static constexpr AcousticMaterial k_acoustic_panel{"acoustic panel",0.20f,0.45f,0.90f,0.95f,0.90f,0.80f};
static constexpr AcousticMaterial k_glass_window {"glass window", 0.35f,0.25f,0.18f,0.12f,0.07f,0.04f};
static constexpr AcousticMaterial k_wood_floor   {"wood floor",   0.15f,0.11f,0.10f,0.07f,0.06f,0.07f};

// Sabine reverberation time T60
[[nodiscard]] inline f32 sabine_rt60(f32 volume_m3, f32 total_absorption) {
    return 0.161f*volume_m3/(total_absorption+1e-10f);
}

// Eyring formula (more accurate for higher absorption)
[[nodiscard]] inline f32 eyring_rt60(f32 volume_m3, f32 total_absorption, f32 total_area_m2) {
    f32 alpha_bar=total_absorption/(total_area_m2+1e-10f);
    return 0.161f*volume_m3/(-total_area_m2*std::log(1.f-alpha_bar+1e-10f));
}

// Critical distance (room transition radius)
[[nodiscard]] inline f32 critical_distance(f32 room_constant_m2, f32 directivity_Q=1.f) {
    return 0.057f*std::sqrt(directivity_Q*room_constant_m2);
}

// Room constant R = S*alpha/(1-alpha)
[[nodiscard]] inline f32 room_constant(f32 total_surface_m2, f32 avg_absorption) {
    return total_surface_m2*avg_absorption/(1.f-avg_absorption+1e-10f);
}

// ─── Daylighting factor ───────────────────────────────────────────────────────

[[nodiscard]] inline f32 daylight_factor(
    f32 window_area_m2, f32 floor_area_m2,
    f32 glass_transmittance=0.7f,
    f32 sky_component=0.4f) // fraction of sky visible
{
    return window_area_m2*glass_transmittance*sky_component/floor_area_m2*100.f;
}

// Window-wall ratio
[[nodiscard]] inline f32 window_wall_ratio(f32 window_area, f32 wall_area) {
    return wall_area>0.f?window_area/wall_area:0.f;
}

// ─── Thermal comfort ──────────────────────────────────────────────────────────

// PMV (Predicted Mean Vote) — simplified Fanger
[[nodiscard]] inline f32 pmv_simplified(
    f32 air_temp_C, f32 mean_radiant_temp_C,
    f32 air_velocity_ms, f32 relative_humidity,
    f32 metabolic_rate=1.2f, // met (1=seated)
    f32 clothing_insulation=1.0f) // clo
{
    f32 operative_temp=(air_temp_C+mean_radiant_temp_C)*0.5f;
    f32 comfort_temp=21.8f+0.31f*operative_temp; // simplified Humphreys
    f32 pmv=(operative_temp-comfort_temp)*0.33f;
    pmv-=air_velocity_ms*0.5f; // cooling effect
    pmv+=(relative_humidity-50.f)*0.01f;
    return std::clamp(pmv,-3.f,3.f);
}

// PPD (Predicted Percentage Dissatisfied) from PMV
[[nodiscard]] inline f32 ppd_from_pmv(f32 pmv) {
    return 100.f-95.f*std::exp(-0.03353f*pmv*pmv*pmv*pmv-0.2179f*pmv*pmv);
}

// ─── Sight-line analysis ──────────────────────────────────────────────────────

// Check if seats in a theatre have unobstructed sightlines (C-value)
[[nodiscard]] inline f32 c_value(f32 eye_height, f32 floor_rise, f32 row_depth, f32 dist_to_screen) {
    // C = (d+row_depth)*(eye_height-k)/(d) - (eye_height-k)  (simplified)
    // k = height of object seen over head in front
    f32 k=eye_height-0.12f; // 12cm above eye level
    return (dist_to_screen+row_depth)*(eye_height-k)/(dist_to_screen)-k;
}

} // namespace qs
