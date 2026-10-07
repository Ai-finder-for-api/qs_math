#pragma once
// qs_math/geometry/spatial_indexing.hpp — BVH, KD-tree, octree, R-tree, spatial hashing
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

struct SV3 { f32 x,y,z;
    [[nodiscard]] constexpr SV3 operator+(SV3 b) const noexcept{return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr SV3 operator-(SV3 b) const noexcept{return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr SV3 operator*(f32 s) const noexcept{return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f32 dot(SV3 b) const noexcept{return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr f32 len2() const noexcept{return dot(*this);}
    [[nodiscard]] f32 len() const noexcept{return std::sqrt(len2());}
    [[nodiscard]] constexpr SV3 min_v(SV3 b) const noexcept{return{x<b.x?x:b.x,y<b.y?y:b.y,z<b.z?z:b.z};}
    [[nodiscard]] constexpr SV3 max_v(SV3 b) const noexcept{return{x>b.x?x:b.x,y>b.y?y:b.y,z>b.z?z:b.z};}
};

// ─── AABB ─────────────────────────────────────────────────────────────────────
struct SAABB { SV3 lo{1e9f,1e9f,1e9f}, hi{-1e9f,-1e9f,-1e9f};
    constexpr void expand(SV3 p) noexcept{lo=lo.min_v(p);hi=hi.max_v(p);}
    constexpr void expand(SAABB b) noexcept{lo=lo.min_v(b.lo);hi=hi.max_v(b.hi);}
    [[nodiscard]] constexpr SV3 center() const noexcept{return (lo+hi)*0.5f;}
    [[nodiscard]] constexpr SV3 extent() const noexcept{return hi-lo;}
    [[nodiscard]] constexpr f32 surface_area() const noexcept{
        SV3 e=extent(); return 2*(e.x*e.y+e.y*e.z+e.z*e.x);}
    [[nodiscard]] constexpr bool contains(SV3 p) const noexcept{
        return p.x>=lo.x&&p.x<=hi.x&&p.y>=lo.y&&p.y<=hi.y&&p.z>=lo.z&&p.z<=hi.z;}
    [[nodiscard]] constexpr bool overlaps(SAABB b) const noexcept{
        return lo.x<=b.hi.x&&hi.x>=b.lo.x&&lo.y<=b.hi.y&&hi.y>=b.lo.y&&lo.z<=b.hi.z&&hi.z>=b.lo.z;}
    // Ray-AABB intersection (slab method)
    [[nodiscard]] constexpr bool ray_hit(SV3 ro, SV3 inv_rd, f32 tmin=0, f32 tmax=1e9f) const noexcept{
        f32 tx0=(lo.x-ro.x)*inv_rd.x, tx1=(hi.x-ro.x)*inv_rd.x;
        f32 ty0=(lo.y-ro.y)*inv_rd.y, ty1=(hi.y-ro.y)*inv_rd.y;
        f32 tz0=(lo.z-ro.z)*inv_rd.z, tz1=(hi.z-ro.z)*inv_rd.z;
        f32 tN=std::max({std::min(tx0,tx1),std::min(ty0,ty1),std::min(tz0,tz1),tmin});
        f32 tF=std::min({std::max(tx0,tx1),std::max(ty0,ty1),std::max(tz0,tz1),tmax});
        return tN<=tF;}
    // Longest axis (for SAH split)
    [[nodiscard]] constexpr int longest_axis() const noexcept{
        SV3 e=extent(); return e.x>e.y?(e.x>e.z?0:2):(e.y>e.z?1:2);}
};

// ─── BVH node ─────────────────────────────────────────────────────────────────
struct BVHNode {
    SAABB box{};
    int left{-1},right{-1};    // child indices (-1 = leaf)
    int prim_start{0},prim_count{0}; // leaf data
    [[nodiscard]] constexpr bool is_leaf() const noexcept{return left<0;}
};

// SAH cost for split at axis/position
[[nodiscard]] constexpr f32 sah_cost(f32 pL,f32 pR,int nL,int nR,f32 traversal=1,f32 intersect=1) noexcept {
    return traversal+intersect*(pL*nL+pR*nR);
}

// Build flat BVH over N primitives (Morton code based)
[[nodiscard]] constexpr u32 morton_encode(f32 x,f32 y,f32 z) noexcept {
    // Map [0,1]^3 to 10-bit Morton code
    auto expand=[](u32 v)->u32{
        v&=0x3ff;
        v=(v|(v<<16))&0x30000ff;
        v=(v|(v<<8))&0x300f00f;
        v=(v|(v<<4))&0x30c30c3;
        v=(v|(v<<2))&0x9249249;
        return v;
    };
    u32 xi=u32(x*1023),yi=u32(y*1023),zi=u32(z*1023);
    return expand(xi)|(expand(yi)<<1)|(expand(zi)<<2);
}

// ─── KD-tree node ─────────────────────────────────────────────────────────────
struct KDNode {
    f32 split_pos{0};
    int split_axis{0}; // 0=x,1=y,2=z
    int left{-1},right{-1};
    int point_idx{-1}; // leaf: index of point (-1=internal)
    [[nodiscard]] constexpr bool is_leaf() const noexcept{return point_idx>=0;}
};

// KD-tree nearest neighbor search state
struct KDQuery {
    SV3 query{};
    f32 best_dist2{1e18f};
    int best_idx{-1};
};

// KD-tree traversal (iterative, max depth 32)
template<int MAX_NODES>
[[nodiscard]] constexpr int kd_nearest(
    const std::array<KDNode,MAX_NODES>& nodes,
    const std::array<SV3,MAX_NODES>& pts,
    SV3 query) noexcept
{
    KDQuery q{query};
    struct Frame{int node;f32 tmin,tmax;};
    std::array<Frame,64> stack{};int si=0;
    stack[si++]={0,0,1e18f};
    while(si>0){
        auto[node,tmin,tmax]=stack[--si];
        if(node<0)continue;
        const KDNode& n=nodes[node];
        if(n.is_leaf()){
            SV3 d=pts[n.point_idx]-query;
            f32 d2=d.dot(d);
            if(d2<q.best_dist2){q.best_dist2=d2;q.best_idx=n.point_idx;}
            continue;
        }
        int axis=n.split_axis;
        f32 qv=(axis==0?query.x:axis==1?query.y:query.z);
        f32 dv=qv-n.split_pos,dv2=dv*dv;
        int near=dv<0?n.left:n.right, far=dv<0?n.right:n.left;
        if(si+2<64){
            if(dv2<q.best_dist2)stack[si++]={far,0,dv2};
            stack[si++]={near,0,0};
        }
    }
    return q.best_idx;
}

// ─── Octree node ──────────────────────────────────────────────────────────────
struct OctreeNode {
    SAABB box{};
    int children[8]{-1,-1,-1,-1,-1,-1,-1,-1};
    int point_start{0},point_count{0};
    int depth{0};
    [[nodiscard]] constexpr bool is_leaf() const noexcept{return children[0]<0;}
    [[nodiscard]] constexpr int child_idx(SV3 p) const noexcept{
        SV3 c=box.center();
        return (p.x>c.x?1:0)|(p.y>c.y?2:0)|(p.z>c.z?4:0);}
};

// ─── Spatial hash ─────────────────────────────────────────────────────────────
[[nodiscard]] constexpr u64 spatial_hash(i32 x,i32 y,i32 z) noexcept {
    // Large prime hash
    return (u64(x)*73856093)^(u64(y)*19349663)^(u64(z)*83492791);
}
[[nodiscard]] constexpr u64 point_to_cell_hash(SV3 p,f32 cell_size) noexcept {
    i32 x=i32(std::floor(p.x/cell_size));
    i32 y=i32(std::floor(p.y/cell_size));
    i32 z=i32(std::floor(p.z/cell_size));
    return spatial_hash(x,y,z);
}

// ─── Frustum culling ──────────────────────────────────────────────────────────
struct Frustum {
    // 6 planes: (normal, d) where plane: n·x + d = 0
    std::array<std::array<f32,4>,6> planes{};

    // Extract from combined projection-view matrix (row-major)
    static constexpr Frustum from_matrix(const f32 m[16]) noexcept {
        Frustum f;
        // Left, Right, Bottom, Top, Near, Far
        for(int i=0;i<4;++i){
            f.planes[0][i]=m[3*4+i]+m[0*4+i];  // left
            f.planes[1][i]=m[3*4+i]-m[0*4+i];  // right
            f.planes[2][i]=m[3*4+i]+m[1*4+i];  // bottom
            f.planes[3][i]=m[3*4+i]-m[1*4+i];  // top
            f.planes[4][i]=m[3*4+i]+m[2*4+i];  // near
            f.planes[5][i]=m[3*4+i]-m[2*4+i];  // far
        }
        // Normalize
        for(auto& p:f.planes){
            f32 len=std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
            if(len>1e-6f)for(auto& v:p)v/=len;
        }
        return f;
    }

    [[nodiscard]] constexpr bool cull_aabb(const SAABB& box) const noexcept {
        for(auto& p:planes){
            // Positive vertex (most in direction of normal)
            SV3 pv{p[0]>=0?box.hi.x:box.lo.x,
                   p[1]>=0?box.hi.y:box.lo.y,
                   p[2]>=0?box.hi.z:box.lo.z};
            if(pv.x*p[0]+pv.y*p[1]+pv.z*p[2]+p[3]<0) return true; // outside
        }
        return false;
    }
    [[nodiscard]] constexpr bool cull_sphere(SV3 center,f32 radius) const noexcept {
        for(auto& p:planes)
            if(center.x*p[0]+center.y*p[1]+center.z*p[2]+p[3]<-radius) return true;
        return false;
    }
};
} // namespace qs::math
