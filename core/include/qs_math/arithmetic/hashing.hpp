#pragma once
// qs_math/hashing.hpp — Vector hashes, Morton codes, spatial hashes
#include "vec2.hpp"
#include "vec3.hpp"
#include "scalar_types.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// General hash functions
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 fnv1a_64(const u8* data, u64 len) noexcept {
    u64 h=14695981039346656037ULL;
    for(u64 i=0;i<len;++i){ h^=data[i]; h*=1099511628211ULL; }
    return h;
}

[[nodiscard]] constexpr u32 murmur_mix(u32 h) noexcept {
    h^=h>>16; h*=0x85ebca6b; h^=h>>13; h*=0xc2b2ae35; h^=h>>16; return h;
}

[[nodiscard]] constexpr u64 splitmix64(u64 x) noexcept {
    x+=0x9e3779b97f4a7c15; x=(x^(x>>30))*0xbf58476d1ce4e5b9;
    x=(x^(x>>27))*0x94d049bb133111eb; return x^(x>>31);
}

// ---------------------------------------------------------------------------
// Vector hashes
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 hash(Vec2i v) noexcept {
    u64 a=splitmix64(u64(v.x)), b=splitmix64(u64(v.y));
    return a^(b*0x9e3779b97f4a7c15+((a<<6)|(a>>58)));
}
[[nodiscard]] constexpr u64 hash(Vec3i v) noexcept {
    u64 a=splitmix64(u64(v.x)), b=splitmix64(u64(v.y)), c=splitmix64(u64(v.z));
    return a^(b*0x9e3779b97f4a7c15+((a<<6)|(a>>58)))^(c*0x6c62272e07bb0142+((a<<7)|(a>>57)));
}
[[nodiscard]] inline u64 hash(Vec2f v) noexcept {
    u32 xi,yi; __builtin_memcpy(&xi,&v.x,4); __builtin_memcpy(&yi,&v.y,4);
    return hash(Vec2i{i32(xi),i32(yi)});
}
[[nodiscard]] inline u64 hash(Vec3f v) noexcept {
    u32 xi,yi,zi; __builtin_memcpy(&xi,&v.x,4); __builtin_memcpy(&yi,&v.y,4); __builtin_memcpy(&zi,&v.z,4);
    return hash(Vec3i{i32(xi),i32(yi),i32(zi)});
}

// ---------------------------------------------------------------------------
// Morton codes (Z-order curve) — interleave 2D/3D integer coords
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 morton_part1by1(u32 x) noexcept {
    u64 v=x&0x0000ffff;
    v=(v|(v<<8))&0x00ff00ff; v=(v|(v<<4))&0x0f0f0f0f;
    v=(v|(v<<2))&0x33333333; v=(v|(v<<1))&0x55555555; return v;
}
[[nodiscard]] constexpr u32 morton_part1by2_3d(u32 x) noexcept {
    u32 v=x&0x000003ff;
    v=(v|(v<<16))&0xff0000ff; v=(v|(v<<8))&0x0300f00f;
    v=(v|(v<<4))&0x030c30c3; v=(v|(v<<2))&0x09249249; return v;
}

[[nodiscard]] constexpr u64 morton2(u32 x, u32 y) noexcept {
    return morton_part1by1(x)|(morton_part1by1(y)<<1);
}
[[nodiscard]] constexpr u64 morton2(Vec2i v) noexcept { return morton2(u32(v.x),u32(v.y)); }

[[nodiscard]] constexpr u64 morton3(u32 x, u32 y, u32 z) noexcept {
    return morton_part1by2_3d(x)|(morton_part1by2_3d(y)<<1)|(morton_part1by2_3d(z)<<2);
}
[[nodiscard]] constexpr u64 morton3(Vec3i v) noexcept { return morton3(u32(v.x),u32(v.y),u32(v.z)); }

// ---------------------------------------------------------------------------
// Spatial hash — hash a float position to a grid cell key
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 spatial_hash2(Vec2f pos, f32 cell_size) noexcept {
    i32 cx=(i32)floor(pos.x/cell_size), cy=(i32)floor(pos.y/cell_size);
    return hash(Vec2i{cx,cy});
}
[[nodiscard]] constexpr u64 spatial_hash3(Vec3f pos, f32 cell_size) noexcept {
    i32 cx=(i32)floor(pos.x/cell_size), cy=(i32)floor(pos.y/cell_size), cz=(i32)floor(pos.z/cell_size);
    return hash(Vec3i{cx,cy,cz});
}
[[nodiscard]] constexpr Vec2i spatial_cell2(Vec2f pos, f32 cell_size) noexcept {
    return {(i32)floor(pos.x/cell_size),(i32)floor(pos.y/cell_size)};
}
[[nodiscard]] constexpr Vec3i spatial_cell3(Vec3f pos, f32 cell_size) noexcept {
    return {(i32)floor(pos.x/cell_size),(i32)floor(pos.y/cell_size),(i32)floor(pos.z/cell_size)};
}

} // namespace qs::math

// ---------------------------------------------------------------------------
// xxHash-inspired 32-bit hash (fast, good distribution)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u32 xxhash32(u32 x) noexcept {
    x += 0x9e3779b9u;
    x  = ((x<<13)|(x>>19)) * 0xbf58476du;
    x ^= x >> 16;
    return x;
}
[[nodiscard]] constexpr u64 xxhash64(u64 x) noexcept {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// ---------------------------------------------------------------------------
// SipHash-1-2 (cryptographic quality, fast)
// ---------------------------------------------------------------------------
[[nodiscard]] inline u64 siphash(const u8* data, u64 len, u64 k0=0x736f6d6570736575ULL, u64 k1=0x646f72616e646f6dULL) noexcept {
    u64 v0=k0^0x736f6d6570736575ULL, v1=k1^0x646f72616e646f6dULL;
    u64 v2=k0^0x6c7967656e657261ULL, v3=k1^0x7465646279746573ULL;
    auto sipround=[&](){
        v0+=v1;v1=(v1<<13)|(v1>>51);v1^=v0;v0=(v0<<32)|(v0>>32);
        v2+=v3;v3=(v3<<16)|(v3>>48);v3^=v2;
        v0+=v3;v3=(v3<<21)|(v3>>43);v3^=v0;
        v2+=v1;v1=(v1<<17)|(v1>>47);v1^=v2;v2=(v2<<32)|(v2>>32);
    };
    u64 i=0;
    for(;i+8<=len;i+=8){ u64 m=0; __builtin_memcpy(&m,data+i,8); v3^=m; sipround(); sipround(); v0^=m; }
    u64 last=(len&0xff)<<56;
    u64 remain=len-i;
    for(u64 j=0;j<remain;++j) last|=(u64(data[i+j])<<(8*j));
    v3^=last; sipround(); sipround(); v0^=last;
    v2^=0xff; sipround(); sipround(); sipround(); sipround();
    return v0^v1^v2^v3;
}

// ---------------------------------------------------------------------------
// String hash — FNV-1a 32-bit (common for identifiers)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u32 fnv1a_32(const char* s, u32 h=2166136261u) noexcept {
    while(*s) { h^=u8(*s++); h*=16777619u; } return h;
}
// consteval string hash for switch on string literals
[[nodiscard]] consteval u32 operator""_hash(const char* s, std::size_t) noexcept {
    return fnv1a_32(s);
}

// ---------------------------------------------------------------------------
// Combine two hashes (boost::hash_combine style)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 hash_combine(u64 seed, u64 v) noexcept {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed<<6) + (seed>>2));
}
[[nodiscard]] constexpr u32 hash_combine32(u32 seed, u32 v) noexcept {
    return seed ^ (v + 0x9e3779b9u + (seed<<6) + (seed>>2));
}

// ---------------------------------------------------------------------------
// Decode Morton codes back to (x,y,z)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u32 morton2_decode_x(u64 m) noexcept {
    u64 v=m&0x5555555555555555ULL;
    v=(v|(v>>1))&0x3333333333333333ULL;
    v=(v|(v>>2))&0x0f0f0f0f0f0f0f0fULL;
    v=(v|(v>>4))&0x00ff00ff00ff00ffULL;
    v=(v|(v>>8))&0x0000ffff0000ffffULL;
    v=(v|(v>>16))&0x00000000ffffffffULL;
    return u32(v);
}
[[nodiscard]] constexpr u32 morton2_decode_y(u64 m) noexcept { return morton2_decode_x(m>>1); }
