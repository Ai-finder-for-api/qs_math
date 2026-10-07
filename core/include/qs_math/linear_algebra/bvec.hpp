#pragma once
// qs_math/bvec.hpp — Boolean vectors (BVec2, BVec3, BVec4) for mask operations
#include "scalar_types.hpp"

namespace qs::math {

struct BVec2 {
    bool x{false}, y{false};
    constexpr BVec2() noexcept = default;
    constexpr BVec2(bool x, bool y) noexcept : x(x), y(y) {}
    constexpr explicit BVec2(bool s) noexcept : x(s), y(s) {}
    [[nodiscard]] constexpr bool all()  const noexcept { return x && y; }
    [[nodiscard]] constexpr bool any()  const noexcept { return x || y; }
    [[nodiscard]] constexpr bool none() const noexcept { return !x && !y; }
    constexpr BVec2 operator&&(BVec2 o) const noexcept { return {x&&o.x, y&&o.y}; }
    constexpr BVec2 operator||(BVec2 o) const noexcept { return {x||o.x, y||o.y}; }
    constexpr BVec2 operator!()         const noexcept { return {!x, !y}; }
    constexpr bool  operator==(BVec2 o) const noexcept { return x==o.x && y==o.y; }
};

struct BVec3 {
    bool x{false}, y{false}, z{false};
    constexpr BVec3() noexcept = default;
    constexpr BVec3(bool x,bool y,bool z) noexcept : x(x),y(y),z(z) {}
    constexpr explicit BVec3(bool s) noexcept : x(s),y(s),z(s) {}
    [[nodiscard]] constexpr bool all()  const noexcept { return x&&y&&z; }
    [[nodiscard]] constexpr bool any()  const noexcept { return x||y||z; }
    [[nodiscard]] constexpr bool none() const noexcept { return !x&&!y&&!z; }
    constexpr BVec3 operator&&(BVec3 o) const noexcept { return {x&&o.x,y&&o.y,z&&o.z}; }
    constexpr BVec3 operator||(BVec3 o) const noexcept { return {x||o.x,y||o.y,z||o.z}; }
    constexpr BVec3 operator!()         const noexcept { return {!x,!y,!z}; }
    constexpr bool  operator==(BVec3 o) const noexcept { return x==o.x&&y==o.y&&z==o.z; }
};

struct BVec4 {
    bool x{false}, y{false}, z{false}, w{false};
    constexpr BVec4() noexcept = default;
    constexpr BVec4(bool x,bool y,bool z,bool w) noexcept : x(x),y(y),z(z),w(w) {}
    constexpr explicit BVec4(bool s) noexcept : x(s),y(s),z(s),w(s) {}
    [[nodiscard]] constexpr bool all()  const noexcept { return x&&y&&z&&w; }
    [[nodiscard]] constexpr bool any()  const noexcept { return x||y||z||w; }
    [[nodiscard]] constexpr bool none() const noexcept { return !x&&!y&&!z&&!w; }
    constexpr BVec4 operator&&(BVec4 o) const noexcept { return {x&&o.x,y&&o.y,z&&o.z,w&&o.w}; }
    constexpr BVec4 operator||(BVec4 o) const noexcept { return {x||o.x,y||o.y,z||o.z,w||o.w}; }
    constexpr BVec4 operator!()         const noexcept { return {!x,!y,!z,!w}; }
    constexpr bool  operator==(BVec4 o) const noexcept { return x==o.x&&y==o.y&&z==o.z&&w==o.w; }
};

// Component-wise comparison helpers returning BVec
template<typename T> constexpr BVec2 equal(T ax,T ay,T bx,T by) noexcept { return {ax==bx,ay==by}; }
template<typename T> constexpr BVec3 equal(T ax,T ay,T az,T bx,T by,T bz) noexcept { return {ax==bx,ay==by,az==bz}; }

// Select (mix based on mask)
template<typename T>
constexpr T select(bool cond, T a, T b) noexcept { return cond ? a : b; }

} // namespace qs::math

// ─── Integer vector masks (IVec2/3/4 mask ops) ────────────────────────────
// Useful for SIMD-style lane masking in scalar fallback code

[[nodiscard]] constexpr u32 bvec2_mask(BVec2 v) noexcept { return (u32(v.x))|(u32(v.y)<<1); }
[[nodiscard]] constexpr u32 bvec3_mask(BVec3 v) noexcept { return (u32(v.x))|(u32(v.y)<<1)|(u32(v.z)<<2); }
[[nodiscard]] constexpr u32 bvec4_mask(BVec4 v) noexcept { return (u32(v.x))|(u32(v.y)<<1)|(u32(v.z)<<2)|(u32(v.w)<<3); }

[[nodiscard]] constexpr BVec2 mask_to_bvec2(u32 m) noexcept { return {bool(m&1),bool(m&2)}; }
[[nodiscard]] constexpr BVec3 mask_to_bvec3(u32 m) noexcept { return {bool(m&1),bool(m&2),bool(m&4)}; }
[[nodiscard]] constexpr BVec4 mask_to_bvec4(u32 m) noexcept { return {bool(m&1),bool(m&2),bool(m&4),bool(m&8)}; }

// Count active lanes
[[nodiscard]] constexpr int popcount(BVec2 v) noexcept { return int(v.x)+int(v.y); }
[[nodiscard]] constexpr int popcount(BVec3 v) noexcept { return int(v.x)+int(v.y)+int(v.z); }
[[nodiscard]] constexpr int popcount(BVec4 v) noexcept { return int(v.x)+int(v.y)+int(v.z)+int(v.w); }

// First active lane index (-1 if none)
[[nodiscard]] constexpr int first_true(BVec2 v) noexcept { return v.x?0:v.y?1:-1; }
[[nodiscard]] constexpr int first_true(BVec3 v) noexcept { return v.x?0:v.y?1:v.z?2:-1; }
[[nodiscard]] constexpr int first_true(BVec4 v) noexcept { return v.x?0:v.y?1:v.z?2:v.w?3:-1; }

// Blend: result[i] = mask[i] ? a[i] : b[i] — scalar fallback
template<typename T>
[[nodiscard]] constexpr T blend(bool mask, T a, T b) noexcept { return mask?a:b; }

// XOR mask
[[nodiscard]] constexpr BVec2 bvec_xor(BVec2 a, BVec2 b) noexcept { return {a.x^b.x,a.y^b.y}; }
[[nodiscard]] constexpr BVec3 bvec_xor(BVec3 a, BVec3 b) noexcept { return {a.x^b.x,a.y^b.y,a.z^b.z}; }
[[nodiscard]] constexpr BVec4 bvec_xor(BVec4 a, BVec4 b) noexcept { return {a.x^b.x,a.y^b.y,a.z^b.z,a.w^b.w}; }

// Exact one-hot check (exactly one lane true)
[[nodiscard]] constexpr bool exactly_one(BVec2 v) noexcept { return popcount(v)==1; }
[[nodiscard]] constexpr bool exactly_one(BVec3 v) noexcept { return popcount(v)==1; }
[[nodiscard]] constexpr bool exactly_one(BVec4 v) noexcept { return popcount(v)==1; }
