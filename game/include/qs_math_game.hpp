#pragma once
// qs_math/game/include/qs_math_game.hpp — Game math utilities (C++20)
// Screen-space, tile-map, game-object transforms, grid helpers
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/mat4.hpp>
#include <cmath>
#include <array>
namespace qs::math::game {
using namespace qs::math;
// ─── Tile / grid helpers ──────────────────────────────────────────────────────
[[nodiscard]] inline Vec2<i32> world_to_tile(Vec2<f32> world, f32 tile_size) noexcept {
    return { (i32)std::floor(world.x/tile_size), (i32)std::floor(world.y/tile_size) };
}
[[nodiscard]] inline Vec2<f32> tile_to_world(Vec2<i32> tile, f32 tile_size) noexcept {
    return { tile.x*tile_size+tile_size*0.5f, tile.y*tile_size+tile_size*0.5f };
}
[[nodiscard]] inline i32 tile_index(Vec2<i32> t, i32 grid_w) noexcept { return t.y*grid_w+t.x; }
[[nodiscard]] inline Vec2<i32> index_to_tile(i32 idx, i32 grid_w) noexcept { return {idx%grid_w,idx/grid_w}; }
// ─── Isometric projection ────────────────────────────────────────────────────
// Tile (tx,ty) → screen (sx,sy) for classic 2:1 isometric
[[nodiscard]] inline Vec2<f32> iso_tile_to_screen(Vec2<i32> t, f32 tw, f32 th) noexcept {
    return { (t.x-t.y)*(tw*0.5f), (t.x+t.y)*(th*0.5f) };
}
[[nodiscard]] inline Vec2<i32> iso_screen_to_tile(Vec2<f32> s, f32 tw, f32 th) noexcept {
    return { (i32)std::floor(s.x/tw+s.y/th), (i32)std::floor(s.y/th-s.x/tw) };
}
// ─── Camera 2D ───────────────────────────────────────────────────────────────
struct Camera2D {
    Vec2<f32> position{0,0};
    f32 zoom{1};
    f32 rotation{0};
    [[nodiscard]] Vec2<f32> world_to_screen(Vec2<f32> w, Vec2<f32> viewport_size) const noexcept;
    [[nodiscard]] Vec2<f32> screen_to_world(Vec2<f32> s, Vec2<f32> viewport_size) const noexcept;
    [[nodiscard]] Mat4<f32> view_matrix() const noexcept;
};
// ─── Pathfinding heuristics ───────────────────────────────────────────────────
[[nodiscard]] inline f32 heuristic_manhattan(Vec2<i32> a, Vec2<i32> b) noexcept {
    return (f32)(std::abs(a.x-b.x)+std::abs(a.y-b.y));
}
[[nodiscard]] inline f32 heuristic_chebyshev(Vec2<i32> a, Vec2<i32> b) noexcept {
    return (f32)std::max(std::abs(a.x-b.x),std::abs(a.y-b.y));
}
[[nodiscard]] inline f32 heuristic_euclidean(Vec2<i32> a, Vec2<i32> b) noexcept {
    return std::sqrt((f32)((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)));
}
// ─── Bresenham line rasterization ────────────────────────────────────────────
void bresenham_line(Vec2<i32> a, Vec2<i32> b,
    std::vector<Vec2<i32>>& out) noexcept;
// ─── Flood-fill (4-connected) ────────────────────────────────────────────────
void flood_fill(bool* grid, i32 w, i32 h,
    Vec2<i32> start, bool fill_val,
    std::vector<Vec2<i32>>& filled) noexcept;
// ─── Overlap: AABB2D ─────────────────────────────────────────────────────────
struct Rect2D { Vec2<f32> min, max;
    [[nodiscard]] bool contains(Vec2<f32> p) const noexcept { return p.x>=min.x&&p.x<=max.x&&p.y>=min.y&&p.y<=max.y; }
    [[nodiscard]] bool overlaps(const Rect2D& o) const noexcept { return min.x<o.max.x&&max.x>o.min.x&&min.y<o.max.y&&max.y>o.min.y; }
    [[nodiscard]] Rect2D expand(f32 margin) const noexcept { return {min-Vec2<f32>{margin,margin},max+Vec2<f32>{margin,margin}}; }
    [[nodiscard]] Vec2<f32> size() const noexcept { return max-min; }
    [[nodiscard]] Vec2<f32> center() const noexcept { return (min+max)*0.5f; }
};
// ─── Smooth camera follow ────────────────────────────────────────────────────
[[nodiscard]] inline Vec2<f32> smooth_follow(Vec2<f32> cam, Vec2<f32> target,
    f32 speed, f32 dt) noexcept
{
    return cam+(target-cam)*std::min(1.0f,speed*dt);
}
// ─── Screenshake ──────────────────────────────────────────────────────────────
struct ScreenShake {
    f32 trauma{0};     // 0..1; decreases over time
    f32 decay{1.5f};
    f32 max_offset{20};
    f32 max_rotation{0.05f};
    void add(f32 t) noexcept { trauma=std::min(1.0f,trauma+t); }
    void update(f32 dt) noexcept { trauma=std::max(0.0f,trauma-decay*dt); }
    Vec2<f32> offset(f32 seed) const noexcept;
};
// ─── Stat / attribute system helpers ──────────────────────────────────────────
struct StatValue {
    f32 base{0},additive{0},multiplicative{1};
    [[nodiscard]] f32 final() const noexcept { return (base+additive)*multiplicative; }
    void reset_mods() noexcept { additive=0; multiplicative=1; }
};
} // namespace qs::math::game
#include <vector>
