
#pragma once
// qs_math/procedural/include/qs_math_procedural.hpp (C++20)
// Procedural generation: terrain, L-systems, WFC, dungeon, maze, grammar.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
namespace qs::math::proc {
using namespace qs::math;
// ─── Heightmap generation ─────────────────────────────────────────────────────
struct Heightmap {
    i32 w{0},h{0};
    std::vector<f32> data;
    Heightmap(i32 w,i32 h):w(w),h(h),data(w*h,0){}
    [[nodiscard]] f32& at(i32 x,i32 y)noexcept{return data[y*w+x];}
    [[nodiscard]] f32  at(i32 x,i32 y)const noexcept{return data[y*w+x];}
    void normalise() noexcept;
    void erode_thermal(f32 talus, i32 iters) noexcept;
    void erode_hydraulic(i32 iters) noexcept;
    [[nodiscard]] Vec3<f32> normal_at(i32 x, i32 y, f32 cell_size=1) const noexcept;
};
void gen_heightmap_fbm(Heightmap& hm, i32 octaves=8,
    f32 lacunarity=2.0f, f32 gain=0.5f, u32 seed=0) noexcept;
void gen_heightmap_diamond_square(Heightmap& hm, f32 roughness=0.5f, u32 seed=0) noexcept;
// ─── L-system ─────────────────────────────────────────────────────────────────
struct LSystem {
    std::string axiom;
    std::unordered_map<char,std::string> rules;
    [[nodiscard]] std::string iterate(i32 n) const noexcept;
};
// Turtle graphics from L-system string -> 3D polyline
std::vector<Vec3<f32>> lsystem_to_lines(const std::string& s,
    f32 step=1, f32 angle_deg=25, Vec3<f32> start={}) noexcept;
// ─── Maze generation (recursive backtracker) ──────────────────────────────────
struct Maze {
    i32 w{0},h{0};
    std::vector<u8> cells; // bit flags: N=1 S=2 E=4 W=8 = open passage
    Maze(i32 w,i32 h):w(w),h(h),cells(w*h,0){}
    void generate(u32 seed=0) noexcept;
    [[nodiscard]] bool can_go(i32 x,i32 y,i32 dir) const noexcept;
    [[nodiscard]] std::vector<Vec2<i32>> solve(Vec2<i32> start, Vec2<i32> end) const noexcept;
};
// ─── Poisson disk sampling ────────────────────────────────────────────────────
std::vector<Vec2<f32>> poisson_disk(f32 w, f32 h, f32 min_dist,
    i32 max_tries=30, u32 seed=0) noexcept;
// ─── Wave Function Collapse (simple tiled) ────────────────────────────────────
struct WFCTile { i32 id; std::array<i32,4> allowed_neighbours[4]; f32 weight{1}; };
struct WFCGrid {
    i32 w,h,n_tiles;
    std::vector<std::vector<bool>> possible; // [y*w+x][tile_id]
    void collapse(u32 seed=0) noexcept;
    [[nodiscard]] i32 get(i32 x,i32 y) const noexcept;
};
} // namespace qs::math::proc
