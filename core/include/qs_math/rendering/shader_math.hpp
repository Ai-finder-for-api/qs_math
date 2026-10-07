#pragma once
// shader_math.hpp — GPU shader helper math: packed normals, tangent space,
//                   parallax mapping, triplanar, detail blending,
//                   screen-space effects, dithering, procedural textures
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat3.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"
#include "noise.hpp"

namespace qs::math::shader {

// ---------------------------------------------------------------------------
// Tangent space
// ---------------------------------------------------------------------------
[[nodiscard]] inline Mat3f build_tbn(Vec3f N, Vec3f T, f32 handedness) noexcept {
    Vec3f B=cross(N,T)*handedness;
    return {T,B,N}; // columns: T, B, N
}

[[nodiscard]] inline Vec3f tangent_to_world(Vec3f tangent_normal, Mat3f TBN) noexcept {
    return (TBN*tangent_normal).normalized();
}

[[nodiscard]] inline Vec3f world_to_tangent(Vec3f world_normal, Mat3f TBN) noexcept {
    return (TBN.transposed()*world_normal).normalized();
}

// Unpack normal map (DX convention: [0,1] → [-1,1])
[[nodiscard]] constexpr Vec3f unpack_normal_dx(Vec3f packed) noexcept {
    return Vec3f{packed.x*2.f-1.f, packed.y*2.f-1.f, packed.z*2.f-1.f}.normalized();
}
// OpenGL convention (green channel flipped)
[[nodiscard]] constexpr Vec3f unpack_normal_gl(Vec3f packed) noexcept {
    return Vec3f{packed.x*2.f-1.f, -(packed.y*2.f-1.f), packed.z*2.f-1.f}.normalized();
}

// Reconstruct Z from XY (for BC5 / two-channel normal maps)
[[nodiscard]] constexpr Vec3f reconstruct_z(Vec2f xy_packed) noexcept {
    Vec2f xy=xy_packed*2.f-Vec2f{1.f};
    f32 z=sqrt(max(0.f,1.f-xy.x*xy.x-xy.y*xy.y));
    return {xy.x,xy.y,z};
}

// Blend normals (partial derivatives method)
[[nodiscard]] constexpr Vec3f blend_normals_pd(Vec3f n1, Vec3f n2) noexcept {
    Vec3f t=n1+Vec3f{0.f,0.f,1.f};
    Vec3f u=n2*Vec3f{-1.f,-1.f, 1.f};
    return (t*dot(t,u)-u*t.z).normalized();
}

// Blend normals (reoriented normal mapping)
[[nodiscard]] constexpr Vec3f blend_normals_rnm(Vec3f base, Vec3f detail) noexcept {
    Vec3f t=base+Vec3f{0.f,0.f,1.f};
    Vec3f u=detail*Vec3f{-1.f,-1.f,1.f};
    return (t*dot(t,u)-u).normalized();
}

// ---------------------------------------------------------------------------
// Parallax mapping
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec2f parallax_offset(
    Vec2f uv, Vec3f view_tangent, f32 height, f32 scale=0.05f) noexcept
{
    return uv+view_tangent.xy()*(height*scale/max(view_tangent.z,EPS_F32));
}

[[nodiscard]] inline Vec2f steep_parallax(
    Vec2f uv, Vec3f view_tangent,
    auto&& sample_height,
    i32 num_layers=16, f32 scale=0.1f) noexcept
{
    f32 layer_depth=1.f/num_layers;
    f32 curr_depth=0.f;
    Vec2f delta=view_tangent.xy()*scale/max(view_tangent.z,EPS_F32)/num_layers;
    Vec2f curr_uv=uv;
    f32 h=sample_height(curr_uv);
    while(curr_depth<h){
        curr_uv-=delta; curr_depth+=layer_depth; h=sample_height(curr_uv);
    }
    return curr_uv;
}

// ---------------------------------------------------------------------------
// Triplanar mapping
// ---------------------------------------------------------------------------
struct TriplanarWeights {
    f32 x{0.f},y{0.f},z{0.f};
};

[[nodiscard]] constexpr TriplanarWeights triplanar_weights(Vec3f N, f32 sharpness=4.f) noexcept {
    Vec3f a=N.abs();
    a={pow(a.x,sharpness),pow(a.y,sharpness),pow(a.z,sharpness)};
    f32 inv=1.f/(a.x+a.y+a.z+EPS_F32);
    return {a.x*inv,a.y*inv,a.z*inv};
}

[[nodiscard]] inline Vec4f triplanar_sample(
    Vec3f P, Vec3f N, f32 scale,
    auto&& sample_fn) noexcept
{
    auto [wx,wy,wz]=triplanar_weights(N);
    return sample_fn(Vec2f{P.y*scale,P.z*scale})*wx
         + sample_fn(Vec2f{P.x*scale,P.z*scale})*wy
         + sample_fn(Vec2f{P.x*scale,P.y*scale})*wz;
}

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------
[[nodiscard]] inline Vec3f procedural_brick(Vec2f uv, f32 ratio=0.5f) noexcept {
    f32 row=floor(uv.y);
    f32 offset=(frac(row*0.5f)>0.25f)?0.5f:0.f;
    Vec2f brick_uv{frac(uv.x+offset),frac(uv.y)};
    f32 mortar=0.02f;
    f32 fx=smoothstep(0.f,mortar,brick_uv.x)*smoothstep(1.f,1.f-mortar,brick_uv.x);
    f32 fy=smoothstep(0.f,mortar,brick_uv.y)*smoothstep(ratio,ratio-mortar,brick_uv.y);
    return Vec3f{fx*fy};
}

[[nodiscard]] inline f32 procedural_checker(Vec2f uv, f32 scale=1.f) noexcept {
    Vec2f s=floor(uv*scale);
    return frac((s.x+s.y)*0.5f)<0.5f?0.f:1.f;
}

[[nodiscard]] inline f32 procedural_hexagon(Vec2f uv) noexcept {
    static constexpr Vec2f s{1.f,1.732f};
    Vec2f p=uv;
    Vec2f a=p-round(p/s)*s;
    Vec2f b=p-round((p-Vec2f{0.5f,0.866f})/s)*s;
    return min(max(abs(a.x),abs(a.y)*0.577f+a.x*0.5f),
               max(abs(b.x),abs(b.y)*0.577f+b.x*0.5f));
}

[[nodiscard]] inline Vec3f procedural_wood(Vec3f P, f32 ring_count=8.f) noexcept {
    f32 d=Vec2f{P.x,P.z}.length();
    f32 noise_val=perlin3(P.x*0.5f,P.y*0.2f,P.z*0.5f)*0.5f;
    f32 rings=frac((d+noise_val)*ring_count);
    f32 wood=smoothstep(0.f,0.1f,rings)*smoothstep(0.9f,0.8f,rings);
    return Vec3f{0.6f+wood*0.4f, 0.3f+wood*0.2f, 0.1f+wood*0.05f};
}

// ---------------------------------------------------------------------------
// Screen-space helpers
// ---------------------------------------------------------------------------
// Reconstruct view-space position from depth
[[nodiscard]] inline Vec3f reconstruct_view_pos(
    Vec2f uv, f32 depth, Mat4f inv_proj) noexcept
{
    Vec4f clip{uv.x*2.f-1.f, uv.y*2.f-1.f, depth*2.f-1.f, 1.f};
    Vec4f view=inv_proj*clip;
    return Vec3f{view.x,view.y,view.z}*(1.f/view.w);
}

// SSAO kernel sample
[[nodiscard]] inline Vec3f ssao_kernel_sample(i32 index, i32 kernel_size, PCG32& rng) noexcept {
    Vec3f sample=rng.unit_sphere();
    sample=sample*rng.next_f32(); // bias toward origin
    f32 scale=f32(index)/kernel_size;
    scale=lerp(0.1f,1.f,scale*scale);
    return sample*scale;
}

// Screen-space reflection march step
[[nodiscard]] constexpr Vec3f ssr_step(
    Vec3f pos, Vec3f dir, f32 step_size, i32 step_index) noexcept
{
    return pos+dir*step_size*(1.f+step_index*0.1f);
}

// Dithered transparency
[[nodiscard]] constexpr bool dithered_discard(Vec2f screen_pos, f32 alpha) noexcept {
    // 4x4 Bayer matrix
    static constexpr f32 bayer[16]={
        0.0625f,0.5625f,0.1875f,0.6875f,
        0.8125f,0.3125f,0.9375f,0.4375f,
        0.25f,  0.75f,  0.125f, 0.625f,
        1.0f,   0.5f,   0.875f, 0.375f
    };
    i32 bx=i32(screen_pos.x)%4, by=i32(screen_pos.y)%4;
    return alpha<bayer[by*4+bx];
}

// Fog
[[nodiscard]] constexpr Vec3f apply_fog_linear(
    Vec3f color, Vec3f fog_color, f32 depth, f32 fog_near, f32 fog_far) noexcept
{
    f32 t=saturate((depth-fog_near)/(fog_far-fog_near));
    return lerp(color,fog_color,t);
}

[[nodiscard]] constexpr Vec3f apply_fog_exp2(
    Vec3f color, Vec3f fog_color, f32 depth, f32 density) noexcept
{
    f32 t=1.f-exp(-density*density*depth*depth);
    return lerp(color,fog_color,saturate(t));
}

// Height-based fog
[[nodiscard]] constexpr f32 height_fog_density(
    f32 world_y, f32 fog_base_y, f32 fog_height, f32 fog_density) noexcept
{
    return fog_density*exp(-max(0.f,world_y-fog_base_y)/fog_height);
}

} // namespace qs::math::shader
