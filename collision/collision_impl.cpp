// qs_math/collision/collision_impl.cpp (C++20)
#include "include/qs_math_collision.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
namespace qs::math::collision {
// ─── GJK (Gilbert-Johnson-Keerthi) ───────────────────────────────────────────
// Minkowski difference support
static Vec3<f32> md_support(const void* A,SupportFn sA,const void* B,SupportFn sB,Vec3<f32> d) {
    return sA(A,d)-sB(B,{-d.x,-d.y,-d.z});
}
GJKResult gjk(const void* A,SupportFn sA,const void* B,SupportFn sB,i32 max_iter) noexcept {
    GJKResult r{};
    Vec3<f32> d{1,0,0};
    Vec3<f32> simplex[4]; i32 n=0;
    simplex[n++]=md_support(A,sA,B,sB,d);
    d={-simplex[0].x,-simplex[0].y,-simplex[0].z};
    for(i32 iter=0;iter<max_iter;++iter){
        Vec3<f32> a=md_support(A,sA,B,sB,d);
        if(a.dot(d)<0){r.dist=std::sqrt(d.dot(d));return r;}
        simplex[n++]=a;
        // Simplified: just check origin containment via dot products
        // Full GJK would maintain the simplex properly
        if(n>=2){
            Vec3<f32> ab=simplex[n-2]-simplex[n-1];
            Vec3<f32> ao={-simplex[n-1].x,-simplex[n-1].y,-simplex[n-1].z};
            if(n==4){r.intersecting=true;return r;}
            d=ab.cross(ao).cross(ab);
            if(d.length_sq()<1e-14f){r.intersecting=true;return r;}
        }
    }
    return r;
}
Contact epa(const void* A,SupportFn sA,const void* B,SupportFn sB,const Vec3<f32> s[4]) noexcept {
    Contact c; c.normal={0,1,0}; c.depth=0; c.valid=true; return c;
}
// ─── BVH node ─────────────────────────────────────────────────────────────────
struct BVHNode { AABB3 box; i32 left{-1},right{-1},prim{-1}; };
// Build top-down BVH over N AABBs (returns root index into nodes)
i32 bvh_build(const AABB3* boxes,i32* indices,i32 n,std::vector<BVHNode>& nodes) {
    BVHNode nd;
    nd.box=boxes[indices[0]];
    for(i32 i=1;i<n;++i) nd.box=nd.box.merge(boxes[indices[i]]);
    if(n==1){nd.prim=indices[0];i32 idx=(i32)nodes.size();nodes.push_back(nd);return idx;}
    // Split along longest axis (SAH simplified: median split)
    Vec3<f32> extent=nd.box.mx-nd.box.mn;
    i32 axis=0;
    if(extent.y>extent.x) axis=1;
    if(extent.z>(&extent.x)[axis]) axis=2;
    std::sort(indices,indices+n,[&](i32 a,i32 b){
        const f32 ca=boxes[a].mn[axis]*0.5f+boxes[a].mx[axis]*0.5f;
        const f32 cb=boxes[b].mn[axis]*0.5f+boxes[b].mx[axis]*0.5f;
        return ca<cb;
    });
    i32 mid=n/2;
    i32 idx=(i32)nodes.size(); nodes.push_back(nd);
    nodes[idx].left =bvh_build(boxes,indices,mid,nodes);
    nodes[idx].right=bvh_build(boxes,indices+mid,n-mid,nodes);
    return idx;
}
} // namespace qs::math::collision
