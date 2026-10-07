
// qs_math/game/game_ext_impl.cpp (C++20)
// Extended game math: camera controllers, animation blending, state machines,
// damage/hitbox math, inventory geometry, loot tables, procedural AI paths.
#include "include/qs_math_game.hpp"
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/mat4.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
namespace qs::math::game {

// ─── Orbital / third-person camera ───────────────────────────────────────────
struct OrbitCamera {
    Vec3<f32> target{0,0,0};
    f32 yaw{0}, pitch{0.5f}, distance{5};
    f32 min_pitch{-1.4f}, max_pitch{1.4f};
    f32 min_dist{1}, max_dist{50};
    f32 pan_speed{0.01f}, zoom_speed{0.5f}, rotate_speed{0.005f};

    void rotate(f32 dx, f32 dy) noexcept {
        yaw   += dx * rotate_speed;
        pitch  = std::max(min_pitch, std::min(max_pitch, pitch + dy * rotate_speed));
    }
    void zoom(f32 delta) noexcept {
        distance = std::max(min_dist, std::min(max_dist, distance - delta * zoom_speed));
    }
    void pan(f32 dx, f32 dy) noexcept {
        const Vec3<f32> right{std::cos(yaw), 0, -std::sin(yaw)};
        const Vec3<f32> up{0, 1, 0};
        target = target + right * (dx * pan_speed * distance)
                        + up    * (dy * pan_speed * distance);
    }
    [[nodiscard]] Vec3<f32> position() const noexcept {
        const f32 cp = std::cos(pitch), sp = std::sin(pitch);
        const f32 cy = std::cos(yaw),   sy = std::sin(yaw);
        return target + Vec3<f32>{cp*sy, sp, cp*cy} * distance;
    }
    [[nodiscard]] Mat4<f32> view_matrix() const noexcept {
        const Vec3<f32> pos = position();
        const Vec3<f32> fwd = (target - pos).normalized();
        const Vec3<f32> right = fwd.cross(Vec3<f32>{0,1,0}).normalized();
        const Vec3<f32> up = right.cross(fwd);
        Mat4<f32> m = Mat4<f32>::identity();
        m[0][0]=right.x; m[1][0]=right.y; m[2][0]=right.z; m[3][0]=-right.dot(pos);
        m[0][1]=up.x;    m[1][1]=up.y;    m[2][1]=up.z;    m[3][1]=-up.dot(pos);
        m[0][2]=-fwd.x;  m[1][2]=-fwd.y;  m[2][2]=-fwd.z;  m[3][2]=fwd.dot(pos);
        return m;
    }
    void smooth_to(const OrbitCamera& target_cam, f32 alpha) noexcept {
        yaw      += (target_cam.yaw      - yaw)      * alpha;
        pitch    += (target_cam.pitch    - pitch)    * alpha;
        distance += (target_cam.distance - distance) * alpha;
        target = target + (target_cam.target - target) * alpha;
    }
};

// ─── First-person camera ──────────────────────────────────────────────────────
struct FPSCamera {
    Vec3<f32> position{0,1.7f,0};
    f32 yaw{0}, pitch{0};
    f32 move_speed{5}, sprint_mult{2}, sensitivity{0.002f};
    f32 max_pitch{1.5f};

    void rotate(f32 dx, f32 dy) noexcept {
        yaw   -= dx * sensitivity;
        pitch  = std::max(-max_pitch, std::min(max_pitch, pitch - dy * sensitivity));
    }
    [[nodiscard]] Vec3<f32> forward() const noexcept {
        return {std::sin(yaw)*std::cos(pitch), -std::sin(pitch), std::cos(yaw)*std::cos(pitch)};
    }
    [[nodiscard]] Vec3<f32> right() const noexcept {
        return {std::cos(yaw), 0, -std::sin(yaw)};
    }
    void move(f32 fwd, f32 strafe, f32 dt, bool sprint=false) noexcept {
        const f32 speed = move_speed * (sprint ? sprint_mult : 1) * dt;
        position = position + forward() * (fwd * speed) + right() * (strafe * speed);
    }
    [[nodiscard]] Mat4<f32> view_matrix() const noexcept {
        const Vec3<f32> fwd = forward(), r = right();
        const Vec3<f32> up = r.cross(fwd);
        Mat4<f32> m = Mat4<f32>::identity();
        m[0][0]=r.x;    m[1][0]=r.y;    m[2][0]=r.z;    m[3][0]=-r.dot(position);
        m[0][1]=up.x;   m[1][1]=up.y;   m[2][1]=up.z;   m[3][1]=-up.dot(position);
        m[0][2]=-fwd.x; m[1][2]=-fwd.y; m[2][2]=-fwd.z; m[3][2]=fwd.dot(position);
        return m;
    }
};

// ─── Animation blending ───────────────────────────────────────────────────────
struct AnimPose {
    std::vector<Vec3<f32>> positions;
    std::vector<Vec4<f32>> rotations; // quaternion per bone
    std::vector<Vec3<f32>> scales;
    i32 n_bones{0};

    void blend(const AnimPose& a, const AnimPose& b, f32 t) noexcept {
        n_bones = a.n_bones;
        positions.resize(n_bones);
        rotations.resize(n_bones);
        scales.resize(n_bones);
        for (i32 i = 0; i < n_bones; ++i) {
            positions[i] = a.positions[i] + (b.positions[i] - a.positions[i]) * t;
            scales[i]    = a.scales[i]    + (b.scales[i]    - a.scales[i])    * t;
            // NLERP for rotation
            const Vec4<f32>& qa = a.rotations[i];
            Vec4<f32> qb = b.rotations[i];
            const f32 dot = qa.x*qb.x+qa.y*qb.y+qa.z*qb.z+qa.w*qb.w;
            if (dot < 0) { qb.x=-qb.x; qb.y=-qb.y; qb.z=-qb.z; qb.w=-qb.w; }
            Vec4<f32> qr{qa.x+(qb.x-qa.x)*t, qa.y+(qb.y-qa.y)*t,
                         qa.z+(qb.z-qa.z)*t, qa.w+(qb.w-qa.w)*t};
            const f32 len=std::sqrt(qr.x*qr.x+qr.y*qr.y+qr.z*qr.z+qr.w*qr.w);
            if (len > 1e-7f) { qr.x/=len; qr.y/=len; qr.z/=len; qr.w/=len; }
            rotations[i] = qr;
        }
    }

    void additive_blend(const AnimPose& base, const AnimPose& additive, f32 weight) noexcept {
        n_bones = base.n_bones;
        positions.resize(n_bones);
        rotations.resize(n_bones);
        scales.resize(n_bones);
        for (i32 i = 0; i < n_bones; ++i) {
            positions[i] = base.positions[i] + additive.positions[i] * weight;
            scales[i]    = base.scales[i]    + additive.scales[i] * weight;
            rotations[i] = base.rotations[i]; // simplified
        }
    }
};

// ─── Inverse kinematics (FABRIK, 2D chain) ────────────────────────────────────
std::vector<Vec2<f32>> fabrik_2d(const std::vector<f32>& bone_lengths,
    Vec2<f32> root, Vec2<f32> target, i32 max_iter=20, f32 tol=0.01f) noexcept
{
    i32 n = (i32)bone_lengths.size() + 1;
    std::vector<Vec2<f32>> joints(n);
    joints[0] = root;
    // Init along x axis
    for (i32 i = 1; i < n; ++i)
        joints[i] = {joints[i-1].x + bone_lengths[i-1], joints[i-1].y};

    const f32 total_len = [&]{ f32 s=0; for(auto l:bone_lengths)s+=l; return s; }();
    if ((target - root).length() >= total_len) {
        // Fully stretched toward target
        const Vec2<f32> dir = (target - root).normalized();
        for (i32 i = 1; i < n; ++i)
            joints[i] = joints[i-1] + dir * bone_lengths[i-1];
        return joints;
    }

    for (i32 iter = 0; iter < max_iter; ++iter) {
        // Forward pass
        joints[n-1] = target;
        for (i32 i = n-2; i >= 0; --i) {
            const Vec2<f32> dir = (joints[i] - joints[i+1]).normalized();
            joints[i] = joints[i+1] + dir * bone_lengths[i];
        }
        // Backward pass
        joints[0] = root;
        for (i32 i = 1; i < n; ++i) {
            const Vec2<f32> dir = (joints[i] - joints[i-1]).normalized();
            joints[i] = joints[i-1] + dir * bone_lengths[i-1];
        }
        if ((joints[n-1] - target).length() < tol) break;
    }
    return joints;
}

// ─── Weighted loot table ──────────────────────────────────────────────────────
struct LootTable {
    struct Entry { i32 id; f32 weight; };
    std::vector<Entry> entries;
    f32 total_weight{0};

    void add(i32 id, f32 weight) noexcept { entries.push_back({id, weight}); total_weight += weight; }
    [[nodiscard]] i32 roll(f32 rand_0_1) const noexcept {
        f32 r = rand_0_1 * total_weight;
        for (auto& e : entries) { r -= e.weight; if (r <= 0) return e.id; }
        return entries.empty() ? -1 : entries.back().id;
    }
    [[nodiscard]] f32 probability(i32 id) const noexcept {
        for (auto& e : entries) if (e.id == id) return e.weight / total_weight;
        return 0;
    }
    [[nodiscard]] f32 expected_rolls_until(i32 id) const noexcept {
        const f32 p = probability(id);
        return p > 1e-7f ? 1.0f / p : 1e7f;
    }
};

// ─── Grid-based line-of-sight (DDA) ──────────────────────────────────────────
bool line_of_sight_grid(const bool* blocked, i32 grid_w, i32 grid_h,
    Vec2<i32> from, Vec2<i32> to) noexcept
{
    i32 x = from.x, y = from.y;
    const i32 dx = std::abs(to.x - from.x), dy = std::abs(to.y - from.y);
    const i32 sx = from.x < to.x ? 1 : -1, sy = from.y < to.y ? 1 : -1;
    i32 err = dx - dy;
    for (;;) {
        if (x < 0 || x >= grid_w || y < 0 || y >= grid_h) return false;
        if (blocked[y * grid_w + x]) return false;
        if (x == to.x && y == to.y) return true;
        const i32 e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }
    }
}

// ─── Projectile trajectory prediction ────────────────────────────────────────
// Returns launch angle (radians) to hit target, or -1 if unreachable
[[nodiscard]] f32 ballistic_angle(Vec2<f32> from, Vec2<f32> to,
    f32 speed, f32 gravity=9.81f) noexcept
{
    const f32 dx = to.x - from.x, dy = to.y - from.y;
    const f32 v2 = speed * speed;
    const f32 disc = v2*v2 - gravity*(gravity*dx*dx + 2*dy*v2);
    if (disc < 0) return -1;
    // Two solutions — return the lower angle (direct shot)
    return std::atan2(v2 - std::sqrt(disc), gravity * dx);
}

// ─── Damage falloff ───────────────────────────────────────────────────────────
[[nodiscard]] f32 damage_falloff(f32 base_damage, f32 dist,
    f32 min_range, f32 max_range, f32 min_mult=0.1f) noexcept
{
    if (dist <= min_range) return base_damage;
    if (dist >= max_range) return base_damage * min_mult;
    const f32 t = (dist - min_range) / (max_range - min_range);
    return base_damage * (1 - t*(1 - min_mult));
}

// ─── Hitbox swept test (OBB vs ray) ──────────────────────────────────────────
[[nodiscard]] f32 obb_ray_hit(Vec3<f32> ray_origin, Vec3<f32> ray_dir,
    Vec3<f32> box_center, Vec3<f32> box_axes[3], Vec3<f32> box_half) noexcept
{
    const Vec3<f32> d = box_center - ray_origin;
    f32 tmin = -1e30f, tmax = 1e30f;
    for (i32 i = 0; i < 3; ++i) {
        const f32 e = box_axes[i].dot(d);
        const f32 f = box_axes[i].dot(ray_dir);
        if (std::abs(f) > 1e-7f) {
            const f32 half = (&box_half.x)[i];
            f32 t1 = (e - half) / f, t2 = (e + half) / f;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return -1;
        } else {
            const f32 half = (&box_half.x)[i];
            if (std::abs(e) > half) return -1;
        }
    }
    return tmin >= 0 ? tmin : tmax >= 0 ? tmax : -1;
}

// ─── Experience curve helpers ─────────────────────────────────────────────────
[[nodiscard]] i32 level_from_xp(f64 xp, f64 base=100, f64 exponent=2.0) noexcept {
    // level n requires base * n^exponent XP total
    i32 lvl = 1;
    while (base * std::pow(lvl+1, exponent) <= xp) ++lvl;
    return lvl;
}
[[nodiscard]] f64 xp_for_level(i32 level, f64 base=100, f64 exponent=2.0) noexcept {
    return base * std::pow(level, exponent);
}
[[nodiscard]] f64 xp_progress_in_level(f64 xp, i32 level,
    f64 base=100, f64 exponent=2.0) noexcept
{
    const f64 lo = xp_for_level(level, base, exponent);
    const f64 hi = xp_for_level(level+1, base, exponent);
    return (xp - lo) / (hi - lo);
}

// ─── Cooldown manager ─────────────────────────────────────────────────────────
struct Cooldown {
    f32 duration{1}, remaining{0};
    [[nodiscard]] bool ready() const noexcept { return remaining <= 0; }
    void trigger() noexcept { remaining = duration; }
    void update(f32 dt) noexcept { remaining = std::max(0.0f, remaining - dt); }
    [[nodiscard]] f32 progress() const noexcept { return 1 - remaining / duration; }
};

// ─── Minimap projection ───────────────────────────────────────────────────────
[[nodiscard]] Vec2<f32> world_to_minimap(Vec3<f32> world_pos,
    Vec3<f32> world_min, Vec3<f32> world_max,
    Vec2<f32> minimap_size) noexcept
{
    const f32 tx = (world_pos.x - world_min.x) / (world_max.x - world_min.x);
    const f32 tz = (world_pos.z - world_min.z) / (world_max.z - world_min.z);
    return {tx * minimap_size.x, (1 - tz) * minimap_size.y};
}

// ─── State machine helpers ────────────────────────────────────────────────────
struct FSMTransition {
    i32 from_state, to_state;
    std::function<bool()> condition;
};

struct FSM {
    i32 current{0};
    std::vector<FSMTransition> transitions;
    std::function<void(i32)> on_enter, on_exit;

    void update() noexcept {
        for (auto& t : transitions) {
            if (t.from_state == current && t.condition && t.condition()) {
                if (on_exit)  on_exit(current);
                current = t.to_state;
                if (on_enter) on_enter(current);
                break;
            }
        }
    }
    void add(i32 from, i32 to, std::function<bool()> cond) noexcept {
        transitions.push_back({from, to, std::move(cond)});
    }
};

// ─── Tile autotiling (4-bit Wang) ─────────────────────────────────────────────
// Returns tile variant index (0-15) based on N/E/S/W neighbor bitmask
[[nodiscard]] i32 wang_tile_index(bool n, bool e, bool s, bool w) noexcept {
    return (n?1:0)|(e?2:0)|(s?4:0)|(w?8:0);
}
// 8-directional autotiling (Wang blob, 47 tiles)
[[nodiscard]] i32 blob_tile_index(bool n,bool ne,bool e,bool se,
    bool s,bool sw,bool w,bool nw) noexcept
{
    const i32 base = (n?1:0)|(e?2:0)|(s?4:0)|(w?8:0);
    i32 diag = 0;
    if (n&&e) diag |= (ne?1:0)<<4;
    if (s&&e) diag |= (se?1:0)<<5;
    if (s&&w) diag |= (sw?1:0)<<6;
    if (n&&w) diag |= (nw?1:0)<<7;
    return base | diag;
}

} // namespace qs::math::game
