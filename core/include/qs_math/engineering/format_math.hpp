#pragma once
// format_math.hpp — Mathematical helpers for Quantum Suite custom binary formats:
//   .q3d, .qmesh, .qrig, .qanim, .qscene, .qgame, .qlevel, .qtex, .qimage, .qvideo
//   These are math-level helpers only — no I/O, no serialization.
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include "transform.hpp"
#include "scalar_math.hpp"
#include "vec_pack.hpp"
#include "animation_math.hpp"

namespace qs::math::fmt {

// ---------------------------------------------------------------------------
// .qmesh — Vertex layout math
// ---------------------------------------------------------------------------
// Standard vertex formats
struct VertexP3N3T2 {
    Vec3f position, normal;
    Vec2f texcoord;
};
struct VertexP3N3T4T2 {  // with tangent
    Vec3f position, normal;
    Vec4f tangent;         // xyz=tangent, w=handedness
    Vec2f texcoord;
};
struct VertexP3N3T4T2UV2 { // with second UV
    Vec3f position, normal;
    Vec4f tangent;
    Vec2f texcoord0, texcoord1;
};
struct VertexSkinned {
    Vec3f     position, normal;
    Vec4f     tangent;
    Vec2f     texcoord;
    Vec4f     bone_weights;
    Vec4<i32> bone_indices;
};

// Vertex attribute packing
[[nodiscard]] inline u32 pack_normal_to_oct32(Vec3f n) noexcept {
    // Pack normal via oct encoding into two i16
    Vec2f oct=oct_encode_normal(n);
    i16 x=i16(clamp(oct.x,-1.f,1.f)*32767.f);
    i16 y=i16(clamp(oct.y,-1.f,1.f)*32767.f);
    return u32(u16(x))|(u32(u16(y))<<16);
}
[[nodiscard]] inline Vec3f unpack_normal_from_oct32(u32 packed) noexcept {
    i16 x=i16(packed&0xFFFF), y=i16(packed>>16);
    Vec2f oct{x/32767.f,y/32767.f};
    return oct_decode_normal(oct);
}

// Tangent frame from packed normal+tangent
[[nodiscard]] inline void unpack_tangent_frame(
    u32 packed_normal, Vec4f packed_tangent,
    Vec3f& N, Vec3f& T, Vec3f& B) noexcept
{
    N=unpack_normal_from_oct32(packed_normal);
    T=Vec3f{packed_tangent.x,packed_tangent.y,packed_tangent.z};
    B=cross(N,T)*packed_tangent.w;
}

// Mesh bounding sphere from vertex positions (for .qmesh header)
[[nodiscard]] inline Spheref compute_mesh_bsphere(const Vec3f* positions, i32 n) noexcept {
    return bounding_sphere_ritter(positions,n);
}

// Mesh AABB (for .qmesh header)
[[nodiscard]] inline AABBf compute_mesh_aabb(const Vec3f* positions, i32 n) noexcept {
    return aabb_from_points(positions,n);
}

// ---------------------------------------------------------------------------
// .q3d / .qscene — Scene graph math
// ---------------------------------------------------------------------------
struct SceneNodeTransform {
    Transformf local;
    Mat4f      world_matrix;     // cached world transform
    u32        parent_id{~0u};   // ~0 = root
    u32        node_id{0};

    void compute_world(const Mat4f* parent_world) noexcept {
        world_matrix = parent_world?(*parent_world)*local.to_mat4():local.to_mat4();
    }
};

// Compute all world matrices for a flat scene node array (BFS order)
inline void compute_scene_world_matrices(
    SceneNodeTransform* nodes, i32 count) noexcept
{
    for(i32 i=0;i<count;++i){
        if(nodes[i].parent_id==~0u){
            nodes[i].compute_world(nullptr);
        } else {
            nodes[i].compute_world(&nodes[nodes[i].parent_id].world_matrix);
        }
    }
}

// Frustum cull scene nodes (returns bitfield of visible nodes)
inline void frustum_cull_scene(
    const SceneNodeTransform* nodes, const AABBf* local_aabbs, i32 count,
    const Frustumf& frustum, bool* visible) noexcept
{
    for(i32 i=0;i<count;++i){
        // Transform AABB corners into world space
        AABBf world_aabb{};
        auto corners=local_aabbs[i].corners();
        for(auto& c:corners){
            Vec3f wp=nodes[i].world_matrix.transform_point(c);
            world_aabb=world_aabb.is_valid()?world_aabb.merge(wp):AABBf{wp,wp};
        }
        visible[i]=frustum.intersects_aabb(world_aabb);
    }
}

// ---------------------------------------------------------------------------
// .qanim — Animation curve math
// ---------------------------------------------------------------------------
// Compress animation track (removes redundant keys within tolerance)
inline i32 compress_anim_track(
    const anim::AnimKey* keys, i32 n, anim::AnimKey* out, f32 tol=0.001f) noexcept
{
    if(n<=2){ for(i32 i=0;i<n;++i) out[i]=keys[i]; return n; }
    i32 count=0;
    out[count++]=keys[0];
    for(i32 i=1;i<n-1;++i){
        // Check if this key can be removed (linear interpolation within tolerance)
        f32 t=(keys[i].time-keys[i-1].time)/(keys[i+1].time-keys[i-1].time);
        f32 predicted=lerp(keys[i-1].value,keys[i+1].value,t);
        if(abs(keys[i].value-predicted)>tol) out[count++]=keys[i];
    }
    out[count++]=keys[n-1];
    return count;
}

// Resample animation at fixed time step
inline i32 resample_anim_track(
    const anim::AnimKey* keys, i32 n, f32 dt, anim::AnimKey* out, i32 max_out) noexcept
{
    if(n==0) return 0;
    f32 t_start=keys[0].time, t_end=keys[n-1].time;
    i32 count=0;
    for(f32 t=t_start;t<=t_end&&count<max_out;t+=dt){
        f32 v=evaluate_anim_curve(keys,n,t);
        out[count++]={t,v};
    }
    return count;
}

// ---------------------------------------------------------------------------
// .qrig — Skeleton math
// ---------------------------------------------------------------------------
// Build bind pose inverse matrices from skeleton
inline void build_inverse_bind_matrices(
    const anim::Skeleton& skel, Mat4f* inv_bind, i32 n) noexcept
{
    Mat4f world[anim::MAX_BONES];
    // Compute world transforms of bind pose
    for(i32 i=0;i<min(n,skel.bone_count);++i){
        Mat4f local=skel.bones[i].bind_pose.to_mat4();
        i32 p=skel.bones[i].parent;
        world[i]=(p<0)?local:world[p]*local;
    }
    for(i32 i=0;i<min(n,skel.bone_count);++i) inv_bind[i]=world[i].inversed();
}

// ---------------------------------------------------------------------------
// .qtex — Texture math
// ---------------------------------------------------------------------------
// BC4 (ATI1) single-channel block compression metrics
[[nodiscard]] inline f32 bc4_interpolate(f32 c0, f32 c1, i32 code) noexcept {
    if(code==0) return c0; if(code==1) return c1;
    if(c0>c1){
        switch(code){
            case 2: return (6.f*c0+1.f*c1)/7.f;
            case 3: return (5.f*c0+2.f*c1)/7.f;
            case 4: return (4.f*c0+3.f*c1)/7.f;
            case 5: return (3.f*c0+4.f*c1)/7.f;
            case 6: return (2.f*c0+5.f*c1)/7.f;
            case 7: return (1.f*c0+6.f*c1)/7.f;
        }
    } else {
        switch(code){
            case 2: return (4.f*c0+1.f*c1)/5.f;
            case 3: return (3.f*c0+2.f*c1)/5.f;
            case 4: return (2.f*c0+3.f*c1)/5.f;
            case 5: return (1.f*c0+4.f*c1)/5.f;
            case 6: return 0.f;
            case 7: return 1.f;
        }
    }
    return 0.f;
}

// BC5 (ATI2) RGTC normal map block error metric
[[nodiscard]] inline f32 bc5_block_error(
    const Vec2f* pixels, Vec2f c0, Vec2f c1) noexcept
{
    f32 err=0.f;
    for(i32 i=0;i<16;++i){
        f32 best=F32_INF;
        for(i32 r=0;r<8;++r) for(i32 g=0;g<8;++g){
            Vec2f interp{bc4_interpolate(c0.x,c1.x,r),bc4_interpolate(c0.y,c1.y,g)};
            Vec2f d=interp-pixels[i];
            f32 e=d.length_sq(); best=min(best,e);
        }
        err+=best;
    }
    return err;
}

// Mip generation (box filter)
inline void generate_mip(const Vec4f* src, i32 sw, i32 sh, Vec4f* dst) noexcept {
    i32 dw=max(1,sw/2), dh=max(1,sh/2);
    for(i32 y=0;y<dh;++y) for(i32 x=0;x<dw;++x){
        Vec4f s={0.f};
        s=s+src[(y*2)*sw+(x*2)];
        s=s+src[(y*2)*sw+min(x*2+1,sw-1)];
        s=s+src[min(y*2+1,sh-1)*sw+(x*2)];
        s=s+src[min(y*2+1,sh-1)*sw+min(x*2+1,sw-1)];
        dst[y*dw+x]=s*0.25f;
    }
}

// ---------------------------------------------------------------------------
// .qgame / .qlevel — Game-specific math
// ---------------------------------------------------------------------------
// NavMesh helper: is a triangle walkable (slope check)
[[nodiscard]] constexpr bool navmesh_walkable(Vec3f n, f32 max_slope_deg=45.f) noexcept {
    return n.y>=cos(max_slope_deg*DEG_TO_RAD);
}

// Spawn point validity (above floor, not inside geometry)
[[nodiscard]] constexpr bool spawn_valid(Vec3f pos, f32 floor_y, f32 clearance=1.8f) noexcept {
    return pos.y>floor_y&&pos.y<floor_y+clearance;
}

// Trigger zone overlap check
[[nodiscard]] constexpr bool trigger_overlap(
    Vec3f player_pos, f32 player_r, Vec3f trigger_pos, Vec3f trigger_half_extents) noexcept
{
    AABBf box{trigger_pos-trigger_half_extents, trigger_pos+trigger_half_extents};
    Spheref sphere{player_pos,player_r};
    return box.intersects(sphere);
}

// Loot table weighted selection
[[nodiscard]] inline i32 loot_select(const f32* weights, i32 n, f32 roll) noexcept {
    f32 sum=0.f; for(i32 i=0;i<n;++i) sum+=weights[i];
    roll*=sum;
    for(i32 i=0;i<n;++i){ roll-=weights[i]; if(roll<=0.f) return i; }
    return n-1;
}

} // namespace qs::math::fmt
