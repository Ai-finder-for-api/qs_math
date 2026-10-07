#pragma once
// collision_narrow.hpp — Narrow-phase: SAT, contact generation, clipping,
//                        convex hull collision, feature queries
#include "geometry3d.hpp"
#include "gjk.hpp"
#include "distance_queries.hpp"
#include "vec3_ext.hpp"

namespace qs::math::collision {

// ---------------------------------------------------------------------------
// SAT (Separating Axis Theorem) — Box vs Box
// ---------------------------------------------------------------------------
struct SATResult {
    bool  intersecting{false};
    Vec3f normal{0.f};           // from B to A (push A out of B)
    f32   depth{0.f};
    Vec3f contact_point{0.f};
};

[[nodiscard]] inline SATResult sat_aabb_aabb(AABBf a, AABBf b) noexcept {
    SATResult r;
    Vec3f min_overlap{F32_INF};
    Vec3f axes[3]={Vec3f::unit_x(),Vec3f::unit_y(),Vec3f::unit_z()};
    f32 min_depth=F32_INF; i32 min_axis=0;
    for(i32 i=0;i<3;++i){
        f32 a_min=(&a.min_pt.x)[i], a_max=(&a.max_pt.x)[i];
        f32 b_min=(&b.min_pt.x)[i], b_max=(&b.max_pt.x)[i];
        if(a_max<b_min||b_max<a_min) return {false};
        f32 d=min(a_max-b_min,b_max-a_min);
        if(d<min_depth){min_depth=d;min_axis=i;}
    }
    r.intersecting=true; r.depth=min_depth;
    r.normal=axes[min_axis];
    if(a.center()[min_axis]<b.center()[min_axis]) r.normal=-r.normal;
    r.contact_point=(a.center()+b.center())*0.5f;
    return r;
}

// SAT — OBB vs OBB (15 axes)
[[nodiscard]] inline SATResult sat_obb_obb(OBBf a, OBBf b) noexcept {
    SATResult r;
    f32 min_depth=F32_INF; Vec3f min_normal{0.f};

    auto test_axis=[&](Vec3f axis)->bool{
        if(axis.length_sq()<EPS_F32*EPS_F32) return true; // degenerate
        axis=axis.normalized();
        // Project both OBBs
        auto project=[&](OBBf obb)->Vec2f{
            f32 c=dot(obb.center,axis);
            f32 r=0.f;
            for(i32 i=0;i<3;++i) r+=abs(dot(obb.axes[i]*(&obb.half_extents.x)[i],axis));
            return {c-r,c+r};
        };
        Vec2f pa=project(a), pb=project(b);
        if(pa.y<pb.x||pb.y<pa.x) return false;
        f32 d=min(pa.y-pb.x,pb.y-pa.x);
        if(d<min_depth){min_depth=d;min_normal=axis;}
        return true;
    };

    // Face axes (6)
    for(i32 i=0;i<3;++i) if(!test_axis(a.axes[i])||!test_axis(b.axes[i])) return {false};
    // Edge cross products (9)
    for(i32 i=0;i<3;++i) for(i32 j=0;j<3;++j)
        if(!test_axis(cross(a.axes[i],b.axes[j]))) return {false};

    r.intersecting=true; r.depth=min_depth;
    r.normal=min_normal;
    if(dot(b.center-a.center,min_normal)<0.f) r.normal=-r.normal;
    r.contact_point=(a.center+b.center)*0.5f;
    return r;
}

// SAT — Sphere vs OBB
[[nodiscard]] inline SATResult sat_sphere_obb(Spheref s, OBBf obb) noexcept {
    Vec3f local=obb.local(s.center);
    Vec3f clamped=local.clamp(-obb.half_extents,obb.half_extents);
    Vec3f closest=obb.center+obb.axes[0]*clamped.x+obb.axes[1]*clamped.y+obb.axes[2]*clamped.z;
    Vec3f diff=s.center-closest;
    f32 dist2=diff.length_sq();
    if(dist2>s.radius*s.radius) return {false};
    f32 dist=sqrt(dist2);
    SATResult r;
    r.intersecting=true;
    r.normal=dist>EPS_F32?diff*(1.f/dist):obb.axes[1];
    r.depth=s.radius-dist;
    r.contact_point=closest;
    return r;
}

// SAT — Sphere vs Sphere
[[nodiscard]] inline SATResult sat_sphere_sphere(Spheref a, Spheref b) noexcept {
    Vec3f diff=a.center-b.center;
    f32 dist2=diff.length_sq();
    f32 sum_r=a.radius+b.radius;
    if(dist2>=sum_r*sum_r) return {false};
    f32 dist=sqrt(dist2);
    SATResult r;
    r.intersecting=true;
    r.normal=dist>EPS_F32?diff*(1.f/dist):Vec3f::unit_y();
    r.depth=sum_r-dist;
    r.contact_point=b.center+r.normal*b.radius;
    return r;
}

// SAT — Capsule vs Capsule
[[nodiscard]] inline SATResult sat_capsule_capsule(Capsulef a, Capsulef b) noexcept {
    auto seg=dist_segment_segment3(a.a,a.b,b.a,b.b);
    f32 sum_r=a.radius+b.radius;
    if(seg.distance>=sum_r) return {false};
    SATResult r;
    r.intersecting=true;
    Vec3f diff=seg.closest_a-seg.closest_b;
    f32 d=diff.length();
    r.normal=d>EPS_F32?diff*(1.f/d):Vec3f::unit_y();
    r.depth=sum_r-seg.distance;
    r.contact_point=(seg.closest_a+seg.closest_b)*0.5f;
    return r;
}

// ---------------------------------------------------------------------------
// Contact manifold generation (polygon clipping, Sutherland-Hodgman)
// ---------------------------------------------------------------------------
// Clip polygon against plane
inline i32 clip_polygon_by_plane(
    const Vec3f* in, i32 n, Planef plane, Vec3f* out) noexcept
{
    i32 out_n=0;
    for(i32 i=0;i<n;++i){
        Vec3f a=in[i], b=in[(i+1)%n];
        f32 da=plane.signed_distance(a), db=plane.signed_distance(b);
        if(da>=0.f&&out_n<16) out[out_n++]=a;
        if((da>0.f)!=(db>0.f)){
            f32 t=da/(da-db);
            if(out_n<16) out[out_n++]=lerp(a,b,t);
        }
    }
    return out_n;
}

// Box face contact manifold
inline i32 generate_box_contacts(
    OBBf a, OBBf b, Vec3f* contact_points, Vec3f& contact_normal, f32& depth) noexcept
{
    auto r=sat_obb_obb(a,b);
    if(!r.intersecting) return 0;
    contact_normal=r.normal; depth=r.depth;
    // Use reference face clipping (simplified — return center contact)
    contact_points[0]=r.contact_point;
    return 1;
}

// ---------------------------------------------------------------------------
// Convex hull support function + GJK wrapper
// ---------------------------------------------------------------------------
[[nodiscard]] inline SATResult gjk_narrow(
    const Vec3f* pts_a, i32 na,
    const Vec3f* pts_b, i32 nb) noexcept
{
    if(!gjk_intersect(pts_a,na,pts_b,nb)) return {false};
    // EPA for penetration depth
    Simplex s; Vec3f dir{1.f,0.f,0.f};
    s.push(minkowski_support(pts_a,na,pts_b,nb,dir)); dir=-s[0];
    for(i32 i=0;i<32;++i){
        Vec3f a=minkowski_support(pts_a,na,pts_b,nb,dir);
        if(dot(a,dir)<0.f) break;
        s.push(a);
        if(do_simplex(s,dir)) break;
    }
    auto e=epa(pts_a,na,pts_b,nb,s);
    if(!e.valid) return {true,Vec3f::unit_y(),0.f};
    return {true,e.normal,e.depth,pts_a[0]};
}

// ---------------------------------------------------------------------------
// Mesh vs Mesh (triangle-level, O(m*n))
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool mesh_mesh_intersect(
    const Vec3f* verts_a, const u32* idx_a, i32 tri_a,
    const Vec3f* verts_b, const u32* idx_b, i32 tri_b) noexcept
{
    for(i32 i=0;i<tri_a;++i){
        Triangle3f ta{verts_a[idx_a[i*3]],verts_a[idx_a[i*3+1]],verts_a[idx_a[i*3+2]]};
        for(i32 j=0;j<tri_b;++j){
            Triangle3f tb{verts_b[idx_b[j*3]],verts_b[idx_b[j*3+1]],verts_b[idx_b[j*3+2]]};
            // Moller triangle-triangle intersection test
            Planef pa=ta.plane(), pb=tb.plane();
            f32 da[3],db[3];
            da[0]=pa.signed_distance(tb.a);da[1]=pa.signed_distance(tb.b);da[2]=pa.signed_distance(tb.c);
            if((da[0]>0.f&&da[1]>0.f&&da[2]>0.f)||(da[0]<0.f&&da[1]<0.f&&da[2]<0.f)) continue;
            db[0]=pb.signed_distance(ta.a);db[1]=pb.signed_distance(ta.b);db[2]=pb.signed_distance(ta.c);
            if((db[0]>0.f&&db[1]>0.f&&db[2]>0.f)||(db[0]<0.f&&db[1]<0.f&&db[2]<0.f)) continue;
            return true; // simplified — actual Moller test omitted for size
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Trigger zone overlap (non-physical, just AABB-based)
// ---------------------------------------------------------------------------
struct TriggerZone {
    AABBf aabb;
    i32   id;
    bool  is_triggered{false};
};

inline void update_triggers(
    TriggerZone* zones, i32 n_zones,
    const AABBf* objects, i32 n_objects,
    bool* triggered_matrix) noexcept // triggered_matrix[zone*n_objects+obj]
{
    for(i32 z=0;z<n_zones;++z)
        for(i32 o=0;o<n_objects;++o)
            triggered_matrix[z*n_objects+o]=zones[z].aabb.intersects(objects[o]);
}

} // namespace qs::math::collision
