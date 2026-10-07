#pragma once
// collision_shapes.hpp — Compound shapes, convex hull generation,
//                        heightmap collision, triangle mesh BVH,
//                        shape queries, support functions for GJK
#include "geometry3d.hpp"
#include "gjk.hpp"
#include "collision_broad.hpp"

namespace qs::math::collision {

// ---------------------------------------------------------------------------
// Convex hull (Quickhull, simplified 3D)
// ---------------------------------------------------------------------------
struct ConvexHull {
    static constexpr i32 MAX_VERTS=256;
    Vec3f verts[MAX_VERTS];
    i32   vert_count{0};

    // Support function for GJK
    [[nodiscard]] Vec3f support(Vec3f dir) const noexcept {
        return support_point_cloud(verts,vert_count,dir);
    }

    [[nodiscard]] AABBf aabb() const noexcept {
        return aabb_from_points(verts,vert_count);
    }

    // Compute from point cloud (gift wrapping approximation)
    void build(const Vec3f* pts, i32 n) noexcept {
        if(n<=0) return;
        // Use bounding box extreme points as initial hull
        vert_count=0;
        AABBf box=aabb_from_points(pts,n);
        auto add=[&](Vec3f p){
            for(i32 i=0;i<vert_count;++i)
                if((verts[i]-p).length_sq()<EPS_F32) return;
            if(vert_count<MAX_VERTS) verts[vert_count++]=p;
        };
        // Start with extremes
        for(i32 axis=0;axis<3;++axis){
            Vec3f mn=pts[0],mx=pts[0];
            for(i32 i=1;i<n;++i){
                if((&pts[i].x)[axis]<(&mn.x)[axis]) mn=pts[i];
                if((&pts[i].x)[axis]>(&mx.x)[axis]) mx=pts[i];
            }
            add(mn); add(mx);
        }
        // Add points on convex hull (extreme in 26 directions)
        static constexpr Vec3f dirs[8]={
            {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},
            {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}
        };
        for(auto& d:dirs) add(support_point_cloud(pts,n,d.normalized()));
        // Fill remaining from input
        for(i32 i=0;i<n&&vert_count<MAX_VERTS;++i) add(pts[i]);
    }
};

// ---------------------------------------------------------------------------
// Compound shape — multiple primitive shapes
// ---------------------------------------------------------------------------
enum class ShapeType { AABB, OBB, Sphere, Capsule, Cylinder, Cone, ConvexHull };

struct CollisionShape {
    ShapeType type{ShapeType::Sphere};
    union {
        AABBf     aabb_data;
        OBBf      obb_data;
        Spheref   sphere_data;
        Capsulef  capsule_data;
    };
    Vec3f local_offset{0.f};
    Quatf local_rotation{Quatf::identity()};

    [[nodiscard]] AABBf world_aabb(Vec3f world_pos, Quatf world_rot) const noexcept {
        Vec3f wp=world_rot.rotate(local_offset)+world_pos;
        switch(type){
            case ShapeType::Sphere: return aabb_from_sphere(Spheref{wp,sphere_data.radius});
            case ShapeType::AABB:   return aabb_data;
            case ShapeType::Capsule:{
                Vec3f a=wp+world_rot.rotate(capsule_data.a-local_offset);
                Vec3f b=wp+world_rot.rotate(capsule_data.b-local_offset);
                Vec3f r{capsule_data.radius};
                return {a.min(b)-r, a.max(b)+r};
            }
            default: return AABBf{wp-Vec3f{1.f},wp+Vec3f{1.f}};
        }
    }
};

struct CompoundShape {
    static constexpr i32 MAX_SHAPES=32;
    CollisionShape shapes[MAX_SHAPES];
    i32 count{0};

    void add(CollisionShape s) noexcept { if(count<MAX_SHAPES) shapes[count++]=s; }

    [[nodiscard]] AABBf world_aabb(Vec3f pos, Quatf rot) const noexcept {
        if(count==0) return {};
        AABBf result=shapes[0].world_aabb(pos,rot);
        for(i32 i=1;i<count;++i) result=result.merge(shapes[i].world_aabb(pos,rot));
        return result;
    }
};

// ---------------------------------------------------------------------------
// Heightmap collision shape
// ---------------------------------------------------------------------------
struct HeightmapCollider {
    const f32* heights{nullptr};
    i32   width{0}, height{0};
    f32   cell_size{1.f};
    f32   height_scale{1.f};
    Vec3f origin{0.f};

    [[nodiscard]] f32 height_at(f32 x, f32 z) const noexcept {
        f32 u=(x-origin.x)/((width-1)*cell_size);
        f32 v=(z-origin.z)/((height-1)*cell_size);
        u=clamp(u,0.f,1.f); v=clamp(v,0.f,1.f);
        f32 fx=u*(width-1), fz=v*(height-1);
        i32 ix=(i32)fx, iz=(i32)fz;
        ix=min(ix,width-2); iz=min(iz,height-2);
        f32 bfx=fx-ix, bfz=fz-iz;
        f32 h00=heights[iz*width+ix],     h10=heights[iz*width+ix+1];
        f32 h01=heights[(iz+1)*width+ix], h11=heights[(iz+1)*width+ix+1];
        return bilinear(h00,h10,h01,h11,bfx,bfz)*height_scale+origin.y;
    }

    [[nodiscard]] Vec3f normal_at(f32 x, f32 z) const noexcept {
        f32 eps=cell_size;
        f32 dx=(height_at(x+eps,z)-height_at(x-eps,z))/(2.f*eps);
        f32 dz=(height_at(x,z+eps)-height_at(x,z-eps))/(2.f*eps);
        return Vec3f{-dx,1.f,-dz}.normalized();
    }

    // Sphere vs heightmap
    [[nodiscard]] bool collide_sphere(Spheref s, Vec3f& contact, Vec3f& normal, f32& depth) const noexcept {
        f32 terrain_y=height_at(s.center.x,s.center.z);
        f32 penetration=terrain_y+s.radius-s.center.y;
        if(penetration<=0.f) return false;
        normal=normal_at(s.center.x,s.center.z);
        contact={s.center.x,terrain_y,s.center.z};
        depth=penetration;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Triangle mesh BVH (static, for mesh colliders)
// ---------------------------------------------------------------------------
struct MeshBVH {
    static constexpr i32 MAX_NODES=4096;
    static constexpr i32 MAX_TRIS=8192;

    struct Node {
        AABBf aabb;
        i32   left{-1},right{-1};
        i32   tri_start{0},tri_count{0};
        bool  is_leaf() const noexcept { return tri_count>0; }
    };

    Node  nodes[MAX_NODES];
    i32   node_count{0};
    u32   tri_indices[MAX_TRIS];
    const Vec3f* verts{nullptr};
    const u32*   indices{nullptr};
    i32   tri_count{0};

    void build(const Vec3f* v, const u32* idx, i32 n_tris) noexcept {
        verts=v; indices=idx; tri_count=min(n_tris,(i32)MAX_TRIS);
        for(i32 i=0;i<tri_count;++i) tri_indices[i]=i;
        node_count=0;
        build_recursive(0,tri_count,0);
    }

    template<typename Fn>
    void query_ray(Ray3f ray, f32 max_t, Fn&& callback) const noexcept {
        if(node_count==0) return;
        i32 stack[128]; i32 sp=0; stack[sp++]=0;
        while(sp>0){
            i32 n=stack[--sp];
            auto hit=ray_aabb(ray,nodes[n].aabb);
            if(!hit.hit||hit.t>max_t) continue;
            if(nodes[n].is_leaf()){
                for(i32 i=0;i<nodes[n].tri_count;++i){
                    u32 ti=tri_indices[nodes[n].tri_start+i];
                    Triangle3f tri{verts[indices[ti*3]],verts[indices[ti*3+1]],verts[indices[ti*3+2]]};
                    auto r=ray_triangle(ray,tri);
                    if(r.hit&&r.t<=max_t) callback(ti,r);
                }
            } else {
                if(nodes[n].left >=0) stack[sp++]=nodes[n].left;
                if(nodes[n].right>=0) stack[sp++]=nodes[n].right;
            }
        }
    }

private:
    i32 build_recursive(i32 start, i32 end, i32 depth) noexcept {
        if(node_count>=MAX_NODES) return -1;
        i32 nidx=node_count++;
        // Compute AABB
        nodes[nidx].aabb=AABBf{};
        for(i32 i=start;i<end;++i){
            u32 ti=tri_indices[i];
            for(i32 v=0;v<3;++v)
                nodes[nidx].aabb=nodes[nidx].aabb.merge(verts[indices[ti*3+v]]);
        }
        if(end-start<=4||depth>12){
            nodes[nidx].tri_start=start; nodes[nidx].tri_count=end-start;
            return nidx;
        }
        // Split on longest axis at median
        i32 axis=nodes[nidx].aabb.longest_axis();
        i32 mid=(start+end)/2;
        // Partition
        for(i32 i=start;i<end-1;++i) for(i32 j=i+1;j<end;++j){
            Vec3f ca=Triangle3f{verts[indices[tri_indices[i]*3]],verts[indices[tri_indices[i]*3+1]],verts[indices[tri_indices[i]*3+2]]}.centroid();
            Vec3f cb=Triangle3f{verts[indices[tri_indices[j]*3]],verts[indices[tri_indices[j]*3+1]],verts[indices[tri_indices[j]*3+2]]}.centroid();
            if((&ca.x)[axis]>(&cb.x)[axis]){u32 t=tri_indices[i];tri_indices[i]=tri_indices[j];tri_indices[j]=t;}
        }
        nodes[nidx].left =build_recursive(start,mid,depth+1);
        nodes[nidx].right=build_recursive(mid,end,depth+1);
        return nidx;
    }
};

} // namespace qs::math::collision
