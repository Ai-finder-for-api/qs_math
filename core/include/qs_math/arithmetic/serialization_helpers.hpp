#pragma once
// qs_math/serialization_helpers.hpp — Binary math serialization, JSON conversion helpers
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include "transform.hpp"
#include <cstring>

namespace qs::math::serial {

// ---------------------------------------------------------------------------
// Binary write helpers (little-endian)
// ---------------------------------------------------------------------------
inline void write_f32(u8* buf, f32 v)    noexcept { __builtin_memcpy(buf,&v,4); }
inline void write_f64(u8* buf, f64 v)    noexcept { __builtin_memcpy(buf,&v,8); }
inline void write_u32(u8* buf, u32 v)    noexcept { __builtin_memcpy(buf,&v,4); }
inline void write_u64(u8* buf, u64 v)    noexcept { __builtin_memcpy(buf,&v,8); }
inline f32  read_f32(const u8* buf)      noexcept { f32 v; __builtin_memcpy(&v,buf,4); return v; }
inline f64  read_f64(const u8* buf)      noexcept { f64 v; __builtin_memcpy(&v,buf,8); return v; }
inline u32  read_u32(const u8* buf)      noexcept { u32 v; __builtin_memcpy(&v,buf,4); return v; }
inline u64  read_u64(const u8* buf)      noexcept { u64 v; __builtin_memcpy(&v,buf,8); return v; }

// Binary sizes
inline constexpr u64 SIZE_VEC2F  = 8;
inline constexpr u64 SIZE_VEC3F  = 12;
inline constexpr u64 SIZE_VEC4F  = 16;
inline constexpr u64 SIZE_MAT4F  = 64;
inline constexpr u64 SIZE_QUATF  = 16;
inline constexpr u64 SIZE_AABBF  = 24;

// Vec2
inline void write(u8* buf, Vec2f v)  noexcept { write_f32(buf,v.x); write_f32(buf+4,v.y); }
inline Vec2f read_vec2f(const u8* b) noexcept { return {read_f32(b),read_f32(b+4)}; }

// Vec3
inline void write(u8* buf, Vec3f v)  noexcept { write_f32(buf,v.x); write_f32(buf+4,v.y); write_f32(buf+8,v.z); }
inline Vec3f read_vec3f(const u8* b) noexcept { return {read_f32(b),read_f32(b+4),read_f32(b+8)}; }

// Vec4
inline void write(u8* buf, Vec4f v)  noexcept {
    write_f32(buf,v.x); write_f32(buf+4,v.y); write_f32(buf+8,v.z); write_f32(buf+12,v.w);
}
inline Vec4f read_vec4f(const u8* b) noexcept { return {read_f32(b),read_f32(b+4),read_f32(b+8),read_f32(b+12)}; }

// Quat
inline void write(u8* buf, Quatf q)  noexcept {
    write_f32(buf,q.x); write_f32(buf+4,q.y); write_f32(buf+8,q.z); write_f32(buf+12,q.w);
}
inline Quatf read_quatf(const u8* b) noexcept { return {read_f32(b),read_f32(b+4),read_f32(b+8),read_f32(b+12)}; }

// Mat4 (column-major, 16 floats)
inline void write(u8* buf, Mat4f m)  noexcept {
    for(i32 c=0;c<4;++c) for(i32 r=0;r<4;++r) write_f32(buf+(c*4+r)*4,m.col[c][r]);
}
inline Mat4f read_mat4f(const u8* b) noexcept {
    Mat4f m;
    for(i32 c=0;c<4;++c) for(i32 r=0;r<4;++r) m.col[c][r]=read_f32(b+(c*4+r)*4);
    return m;
}

// Transform
inline void write(u8* buf, Transformf t) noexcept {
    write(buf,t.translation); write(buf+12,t.rotation); write(buf+28,Vec4f{t.scale.x,t.scale.y,t.scale.z,0.f});
}
inline Transformf read_transformf(const u8* b) noexcept {
    return {read_vec3f(b), read_quatf(b+12), read_vec3f(b+28)};
}

// AABB
inline void write(u8* buf, AABBf a)  noexcept { write(buf,a.min_pt); write(buf+12,a.max_pt); }
inline AABBf read_aabbf(const u8* b) noexcept { return {read_vec3f(b),read_vec3f(b+12)}; }

// Version-safe header
struct MathChunkHeader {
    u32 magic{0x514D4154};  // 'QMAT'
    u32 version{1};
    u32 count{0};
    u32 elem_type{0};       // 0=vec3f, 1=mat4f, etc.
};
inline void write(u8* buf, MathChunkHeader h) noexcept {
    write_u32(buf,h.magic); write_u32(buf+4,h.version);
    write_u32(buf+8,h.count); write_u32(buf+12,h.elem_type);
}
inline MathChunkHeader read_header(const u8* b) noexcept {
    return {read_u32(b),read_u32(b+4),read_u32(b+8),read_u32(b+12)};
}
[[nodiscard]] constexpr bool header_valid(MathChunkHeader h) noexcept { return h.magic==0x514D4154&&h.version<=10; }

} // namespace qs::math::serial

// ─── Endian-aware read/write helpers ──────────────────────────────────────
#include <bit>

[[nodiscard]] constexpr u16 byteswap16(u16 v) noexcept { return u16((v>>8)|(v<<8)); }
[[nodiscard]] constexpr u32 byteswap32(u32 v) noexcept {
    return ((v&0xFF000000u)>>24)|((v&0x00FF0000u)>>8)|((v&0x0000FF00u)<<8)|((v&0x000000FFu)<<24);
}
[[nodiscard]] constexpr u64 byteswap64(u64 v) noexcept {
    v=((v>>8)&0x00FF00FF00FF00FFULL)|((v&0x00FF00FF00FF00FFULL)<<8);
    v=((v>>16)&0x0000FFFF0000FFFFULL)|((v&0x0000FFFF0000FFFFULL)<<16);
    return (v>>32)|(v<<32);
}
[[nodiscard]] constexpr bool is_little_endian() noexcept {
    return std::endian::native == std::endian::little;
}
[[nodiscard]] constexpr u32 to_little_endian(u32 v) noexcept {
    return is_little_endian() ? v : byteswap32(v);
}
[[nodiscard]] constexpr u32 from_little_endian(u32 v) noexcept { return to_little_endian(v); }
[[nodiscard]] constexpr u32 to_big_endian   (u32 v) noexcept {
    return is_little_endian() ? byteswap32(v) : v;
}
[[nodiscard]] constexpr u32 from_big_endian (u32 v) noexcept { return to_big_endian(v); }

// Pack 4 bytes into u32 (little-endian)
[[nodiscard]] constexpr u32 pack_bytes(u8 b0,u8 b1,u8 b2,u8 b3) noexcept {
    return u32(b0)|(u32(b1)<<8)|(u32(b2)<<16)|(u32(b3)<<24);
}
// Unpack u32 to 4 bytes
inline void unpack_bytes(u32 v, u8& b0,u8& b1,u8& b2,u8& b3) noexcept {
    b0=u8(v);b1=u8(v>>8);b2=u8(v>>16);b3=u8(v>>24);
}

// ─── Compact angle encoding ───────────────────────────────────────────────
// Encode angle in [-pi, pi] to u16
[[nodiscard]] constexpr u16 encode_angle_u16(f32 radians) noexcept {
    return u16(u32((radians + f32(PI)) * (65535.0f / f32(2.0*PI))) & 0xFFFF);
}
[[nodiscard]] constexpr f32 decode_angle_u16(u16 v) noexcept {
    return f32(v) * f32(2.0*PI / 65535.0) - f32(PI);
}

// ─── Bit reader / writer helpers ───────────────────────────────────────────
struct BitWriter {
    u8* buf; u32 byte_pos{0}; u8 bit_pos{0};
    BitWriter(u8* b) noexcept : buf(b) {}
    void write(u32 val, u8 bits) noexcept {
        for(u8 i=0;i<bits;++i){
            if(val&(1u<<i)) buf[byte_pos]|=(1u<<bit_pos);
            else            buf[byte_pos]&=~(1u<<bit_pos);
            if(++bit_pos==8){bit_pos=0;++byte_pos;}
        }
    }
    [[nodiscard]] u32 bytes_used() const noexcept { return byte_pos+(bit_pos?1:0); }
};
struct BitReader {
    const u8* buf; u32 byte_pos{0}; u8 bit_pos{0};
    BitReader(const u8* b) noexcept : buf(b) {}
    [[nodiscard]] u32 read(u8 bits) noexcept {
        u32 r=0;
        for(u8 i=0;i<bits;++i){
            if(buf[byte_pos]&(1u<<bit_pos)) r|=(1u<<i);
            if(++bit_pos==8){bit_pos=0;++byte_pos;}
        }
        return r;
    }
};
