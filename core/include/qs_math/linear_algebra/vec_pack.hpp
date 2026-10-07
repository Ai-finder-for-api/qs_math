#pragma once
// vec_pack.hpp — SNORM/UNORM packing, 10_10_10_2, R11G11B10, RGB9E5, BC-related helpers
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "scalar_math.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// UNORM / SNORM
// ---------------------------------------------------------------------------
// Float [0,1] → u8 [0,255]
[[nodiscard]] constexpr u8  float_to_unorm8 (f32 v) noexcept { return u8(saturate(v)*255.f+0.5f); }
[[nodiscard]] constexpr f32 unorm8_to_float (u8  v) noexcept { return v/255.f; }
[[nodiscard]] constexpr u16 float_to_unorm16(f32 v) noexcept { return u16(saturate(v)*65535.f+0.5f); }
[[nodiscard]] constexpr f32 unorm16_to_float(u16 v) noexcept { return v/65535.f; }

// Float [-1,1] → i8 [-127,127]
[[nodiscard]] constexpr i8  float_to_snorm8 (f32 v) noexcept { return i8(clamp(v,-1.f,1.f)*127.f+0.5f); }
[[nodiscard]] constexpr f32 snorm8_to_float (i8  v) noexcept { return max(-1.f,v/127.f); }
[[nodiscard]] constexpr i16 float_to_snorm16(f32 v) noexcept { return i16(clamp(v,-1.f,1.f)*32767.f+0.5f); }
[[nodiscard]] constexpr f32 snorm16_to_float(i16 v) noexcept { return max(-1.f,v/32767.f); }

// Pack Vec2 UNORM → u32
[[nodiscard]] constexpr u32 pack_unorm8_vec2(Vec2f v) noexcept {
    return u32(float_to_unorm8(v.x)) | (u32(float_to_unorm8(v.y))<<8);
}
[[nodiscard]] constexpr Vec2f unpack_unorm8_vec2(u32 p) noexcept {
    return {unorm8_to_float(u8(p&0xFF)), unorm8_to_float(u8((p>>8)&0xFF))};
}

// Pack Vec4 UNORM u8x4 → u32
[[nodiscard]] constexpr u32 pack_unorm8_vec4(Vec4f v) noexcept {
    return u32(float_to_unorm8(v.x))      | (u32(float_to_unorm8(v.y))<<8)
         | (u32(float_to_unorm8(v.z))<<16)| (u32(float_to_unorm8(v.w))<<24);
}
[[nodiscard]] constexpr Vec4f unpack_unorm8_vec4(u32 p) noexcept {
    return {unorm8_to_float(u8(p&0xFF)), unorm8_to_float(u8((p>>8)&0xFF)),
            unorm8_to_float(u8((p>>16)&0xFF)), unorm8_to_float(u8((p>>24)&0xFF))};
}

// Pack Vec4 SNORM i8x4 → i32
[[nodiscard]] constexpr i32 pack_snorm8_vec4(Vec4f v) noexcept {
    return i32(u8(float_to_snorm8(v.x)))       | (i32(u8(float_to_snorm8(v.y)))<<8)
         | (i32(u8(float_to_snorm8(v.z)))<<16) | (i32(u8(float_to_snorm8(v.w)))<<24);
}

// 10_10_10_2 packing (D3D / Vulkan UNORM format)
[[nodiscard]] constexpr u32 pack_10_10_10_2_unorm(Vec4f v) noexcept {
    u32 x=u32(saturate(v.x)*1023.f+0.5f)&0x3FF;
    u32 y=u32(saturate(v.y)*1023.f+0.5f)&0x3FF;
    u32 z=u32(saturate(v.z)*1023.f+0.5f)&0x3FF;
    u32 w=u32(saturate(v.w)*3.f+0.5f)&0x3;
    return x|(y<<10)|(z<<20)|(w<<30);
}
[[nodiscard]] constexpr Vec4f unpack_10_10_10_2_unorm(u32 p) noexcept {
    return {f32(p&0x3FF)/1023.f, f32((p>>10)&0x3FF)/1023.f,
            f32((p>>20)&0x3FF)/1023.f, f32((p>>30)&0x3)/3.f};
}

// Pack normal into 10_10_10_2 SNORM
[[nodiscard]] constexpr u32 pack_normal_10_10_10_2(Vec3f n) noexcept {
    auto s10=[](f32 v)->u32{ return u32(i32(clamp(v,-1.f,1.f)*511.f))&0x3FF; };
    return s10(n.x)|(s10(n.y)<<10)|(s10(n.z)<<20);
}

// R11G11B10 float pack (DXGI_FORMAT_R11G11B10_FLOAT style)
[[nodiscard]] inline u32 pack_r11g11b10_float(Vec3f v) noexcept {
    // Each component: R/G 11-bit (5e6m), B 10-bit (5e5m)
    auto to11=[](f32 f)->u32{
        if(f<=0.f) return 0;
        u32 x; __builtin_memcpy(&x,&f,4);
        u32 e=((x>>23)&0xFF)-112, m=(x>>12)&0x3F;
        if(e>=31) return 0x7BF;
        if(e<0) return 0;
        return (e<<6)|m;
    };
    auto to10=[](f32 f)->u32{
        if(f<=0.f) return 0;
        u32 x; __builtin_memcpy(&x,&f,4);
        u32 e=((x>>23)&0xFF)-112, m=(x>>13)&0x1F;
        if(e>=31) return 0x3DF;
        if(e<0) return 0;
        return (e<<5)|m;
    };
    return to11(v.x)|(to11(v.y)<<11)|(to10(v.z)<<22);
}

// RGB9E5 shared-exponent packing
[[nodiscard]] inline u32 pack_rgb9e5(Vec3f v) noexcept {
    f32 mx=max(max(v.x,v.y),v.z);
    if(mx<=0.f) return 0;
    i32 exp_shared=(i32)ceil(log2(mx+1e-10f));
    f32 scale=pow(2.f,f32(exp_shared-9));
    u32 r=u32(v.x/scale+0.5f)&0x1FF;
    u32 g=u32(v.y/scale+0.5f)&0x1FF;
    u32 b=u32(v.z/scale+0.5f)&0x1FF;
    return r|(g<<9)|(b<<18)|(u32(exp_shared+15)<<27);
}

} // namespace qs::math
