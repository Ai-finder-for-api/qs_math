#pragma once
// qs_math/geometry/spatial_partitioning.hpp — KD-tree, BVH, octree, grid,
//   BSP tree, R-tree helpers, nearest neighbor, range queries (C++20)
#include "../linear_algebra/vec3.hpp"
#include "../linear_algebra/vec2.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <algorithm>
#include <array>
#include <span>
#include <functional>
#include <queue>
#include <cmath>
#include <memory>

namespace qs::math::geometry {

// ─── AABB ─────────────────────────────────────────────────────────────────
struct AABB3 {
    Vec3f lo, hi;
    [[nodiscard]] Vec3f center()  const noexcept { return (lo+hi)*0.5f; }
    [[nodiscard]] Vec3f extents() const noexcept { return (hi-lo)*0.5f; }
    [[nodiscard]] f32   volume()  const noexcept { Vec3f e=hi-lo; return e.x*e.y*e.z; }
    [[nodiscard]] f32   surface_area() const noexcept {
        Vec3f e=hi-lo; return 2.f*(e.x*e.y+e.y*e.z+e.z*e.x);
    }
    [[nodiscard]] bool  contains(Vec3f p) const noexcept {
        return p.x>=lo.x&&p.x<=hi.x&&p.y>=lo.y&&p.y<=hi.y&&p.z>=lo.z&&p.z<=hi.z;
    }
    [[nodiscard]] bool  intersects(const AABB3& o) const noexcept {
        return lo.x<=o.hi.x&&hi.x>=o.lo.x&&lo.y<=o.hi.y&&hi.y>=o.lo.y&&lo.z<=o.hi.z&&hi.z>=o.lo.z;
    }
    [[nodiscard]] AABB3 expand(Vec3f p) const noexcept {
        return {{min(lo.x,p.x),min(lo.y,p.y),min(lo.z,p.z)},
                {max(hi.x,p.x),max(hi.y,p.y),max(hi.z,p.z)}};
    }
    [[nodiscard]] AABB3 merge(const AABB3& o) const noexcept {
        return {{min(lo.x,o.lo.x),min(lo.y,o.lo.y),min(lo.z,o.lo.z)},
                {max(hi.x,o.hi.x),max(hi.y,o.hi.y),max(hi.z,o.hi.z)}};
    }
    // Ray-AABB intersection (slab method), returns t_min (negative = miss)
    [[nodiscard]] f32 ray_intersect(Vec3f org, Vec3f inv_dir) const noexcept {
        f32 tx1=(lo.x-org.x)*inv_dir.x, tx2=(hi.x-org.x)*inv_dir.x;
        f32 ty1=(lo.y-org.y)*inv_dir.y, ty2=(hi.y-org.y)*inv_dir.y;
        f32 tz1=(lo.z-org.z)*inv_dir.z, tz2=(hi.z-org.z)*inv_dir.z;
        f32 tmin=max({min(tx1,tx2),min(ty1,ty2),min(tz1,tz2)});
        f32 tmax=min({max(tx1,tx2),max(ty1,ty2),max(tz1,tz2)});
        return tmax>=max(0.f,tmin)?tmin:-1.f;
    }
    static AABB3 empty() noexcept { return {{1e30f,1e30f,1e30f},{-1e30f,-1e30f,-1e30f}}; }
};

// ─── 3D KD-Tree ───────────────────────────────────────────────────────────
struct KDNode {
    Vec3f point;
    int   left=-1, right=-1, idx=-1;
    int   axis=0;
    f32   split=0;
};

struct KDTree3 {
    std::vector<KDNode> nodes;
    std::vector<Vec3f>  pts;

    void build(std::vector<Vec3f> points) {
        pts=std::move(points);
        int n=int(pts.size());
        std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
        nodes.reserve(2*n);
        build_rec(idx,0,n,0);
    }

    int build_rec(std::vector<int>& idx, int lo, int hi, int depth) {
        if(lo>=hi) return -1;
        int axis=depth%3;
        int mid=(lo+hi)/2;
        std::nth_element(idx.begin()+lo,idx.begin()+mid,idx.begin()+hi,
            [&](int a,int b){ return (&pts[a].x)[axis]<(&pts[b].x)[axis]; });
        int node=int(nodes.size());
        nodes.push_back({pts[idx[mid]],{},{},idx[mid],axis,(&pts[idx[mid]].x)[axis]});
        nodes[node].left  = build_rec(idx,lo,mid,depth+1);
        nodes[node].right = build_rec(idx,mid+1,hi,depth+1);
        return node;
    }

    // Nearest neighbor
    struct NNResult { int idx; f32 dist_sq; };
    void nn_search(int node, Vec3f q, NNResult& best) const noexcept {
        if(node<0) return;
        const KDNode& n=nodes[node];
        Vec3f d=q-n.point;
        f32 dist_sq=d.x*d.x+d.y*d.y+d.z*d.z;
        if(dist_sq<best.dist_sq){best.dist_sq=dist_sq;best.idx=n.idx;}
        f32 dx=(&q.x)[n.axis]-n.split;
        int first=dx<0?n.left:n.right, second=dx<0?n.right:n.left;
        nn_search(first,q,best);
        if(dx*dx<best.dist_sq) nn_search(second,q,best);
    }
    [[nodiscard]] NNResult nearest(Vec3f q) const noexcept {
        NNResult best{-1,1e30f}; nn_search(0,q,best); return best;
    }

    // K-nearest neighbors
    [[nodiscard]] std::vector<NNResult> knn(Vec3f q, int k) const noexcept {
        using P=std::pair<f32,int>;
        std::priority_queue<P> pq;
        std::function<void(int)> rec=[&](int node){
            if(node<0) return;
            const KDNode& n=nodes[node];
            Vec3f d=q-n.point;
            f32 dist_sq=d.x*d.x+d.y*d.y+d.z*d.z;
            if(int(pq.size())<k||(dist_sq<pq.top().first)){
                pq.push({dist_sq,n.idx});
                if(int(pq.size())>k) pq.pop();
            }
            f32 dx=(&q.x)[n.axis]-n.split;
            int first=dx<0?n.left:n.right,second=dx<0?n.right:n.left;
            rec(first);
            f32 worst=pq.empty()?1e30f:pq.top().first;
            if(dx*dx<worst) rec(second);
        };
        rec(0);
        std::vector<NNResult> result;
        while(!pq.empty()){result.push_back({pq.top().second,pq.top().first});pq.pop();}
        return result;
    }

    // Range query: all points within radius r
    [[nodiscard]] std::vector<int> range_query(Vec3f q, f32 r) const noexcept {
        f32 r2=r*r;
        std::vector<int> result;
        std::function<void(int)> rec=[&](int node){
            if(node<0) return;
            const KDNode& n=nodes[node];
            Vec3f d=q-n.point;
            if(d.x*d.x+d.y*d.y+d.z*d.z<=r2) result.push_back(n.idx);
            f32 dx=(&q.x)[n.axis]-n.split;
            int first=dx<0?n.left:n.right,second=dx<0?n.right:n.left;
            rec(first);
            if(dx*dx<=r2) rec(second);
        };
        rec(0); return result;
    }
};

// ─── BVH (Bounding Volume Hierarchy) ─────────────────────────────────────
struct BVHNode {
    AABB3 bounds;
    int left=-1,right=-1; // child node indices (-1 = leaf)
    int prim_begin=0,prim_count=0; // leaf primitive range
};

struct BVHBuildPrim {
    AABB3 bounds;
    Vec3f centroid;
    int   idx;
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<int>     prim_ids; // reordered primitive indices

    void build(std::span<const AABB3> prim_bounds) {
        int n=int(prim_bounds.size());
        std::vector<BVHBuildPrim> prims(n);
        for(int i=0;i<n;++i){
            prims[i]={prim_bounds[i],prim_bounds[i].center(),i};
        }
        prim_ids.resize(n);
        nodes.reserve(2*n);
        build_rec(prims,0,n);
    }

    int build_rec(std::vector<BVHBuildPrim>& prims, int lo, int hi) {
        int node=int(nodes.size()); nodes.push_back({});
        // Compute bounds of all centroids and primitives
        AABB3 bbox=AABB3::empty(), cbox=AABB3::empty();
        for(int i=lo;i<hi;++i){
            bbox=bbox.merge(prims[i].bounds);
            cbox=cbox.expand(prims[i].centroid);
        }
        nodes[node].bounds=bbox;
        int n=hi-lo;
        if(n<=4){
            // Leaf
            nodes[node].prim_begin=int(prim_ids.size());
            nodes[node].prim_count=n;
            for(int i=lo;i<hi;++i) prim_ids.push_back(prims[i].idx);
            return node;
        }
        // SAH split on longest axis
        Vec3f ext=cbox.hi-cbox.lo;
        int axis=0;
        if(ext.y>ext.x) axis=1;
        if((&ext.x)[2]>(&ext.x)[axis]) axis=2;
        // Binned SAH (8 bins)
        const int BINS=8;
        struct Bin{AABB3 b=AABB3::empty();int cnt=0;};
        std::array<Bin,BINS> bins;
        f32 lo_c=(&cbox.lo.x)[axis], scale=BINS/((&cbox.hi.x)[axis]-lo_c+1e-10f);
        for(int i=lo;i<hi;++i){
            int b=int(((&prims[i].centroid.x)[axis]-lo_c)*scale);
            b=clamp(b,0,BINS-1);
            bins[b].b=bins[b].b.merge(prims[i].bounds);
            ++bins[b].cnt;
        }
        f32 best_cost=1e30f; int best_split=1;
        for(int s=1;s<BINS;++s){
            AABB3 lbox=AABB3::empty(),rbox=AABB3::empty();int lc=0,rc=0;
            for(int b=0;b<s;++b){lbox=lbox.merge(bins[b].b);lc+=bins[b].cnt;}
            for(int b=s;b<BINS;++b){rbox=rbox.merge(bins[b].b);rc+=bins[b].cnt;}
            f32 cost=lbox.surface_area()*lc+rbox.surface_area()*rc;
            if(cost<best_cost){best_cost=cost;best_split=s;}
        }
        f32 split_val=lo_c+(best_split/scale);
        auto mid_it=std::partition(prims.begin()+lo,prims.begin()+hi,
            [&](const BVHBuildPrim& p){return (&p.centroid.x)[axis]<split_val;});
        int mid=int(mid_it-prims.begin());
        if(mid==lo||mid==hi) mid=(lo+hi)/2;
        nodes[node].left  = build_rec(prims,lo,mid);
        nodes[node].right = build_rec(prims,mid,hi);
        return node;
    }

    // Traverse: callback returns true to continue
    void traverse(Vec3f ray_org, Vec3f inv_dir, std::function<bool(int prim_id)> cb) const noexcept {
        std::vector<int> stack; stack.reserve(64); stack.push_back(0);
        while(!stack.empty()){
            int n=stack.back(); stack.pop_back();
            const BVHNode& nd=nodes[n];
            if(nd.bounds.ray_intersect(ray_org,inv_dir)<0) continue;
            if(nd.left<0){
                for(int i=nd.prim_begin;i<nd.prim_begin+nd.prim_count;++i)
                    if(!cb(prim_ids[i])) return;
            } else {
                stack.push_back(nd.left);
                stack.push_back(nd.right);
            }
        }
    }
};

// ─── Uniform Grid ─────────────────────────────────────────────────────────
struct UniformGrid3 {
    AABB3 bounds;
    Vec3<i32> dims;
    f32 cell_size;
    std::vector<std::vector<int>> cells;

    void build(AABB3 b, f32 cs, std::span<const Vec3f> pts) {
        bounds=b; cell_size=cs;
        Vec3f e=b.hi-b.lo;
        dims={max(1,i32(e.x/cs)+1),max(1,i32(e.y/cs)+1),max(1,i32(e.z/cs)+1)};
        cells.resize(dims.x*dims.y*dims.z);
        for(int i=0;i<int(pts.size());++i){
            auto ci=cell_of(pts[i]);
            if(ci.x>=0&&ci.x<dims.x&&ci.y>=0&&ci.y<dims.y&&ci.z>=0&&ci.z<dims.z)
                cells[ci.x*dims.y*dims.z+ci.y*dims.z+ci.z].push_back(i);
        }
    }

    [[nodiscard]] Vec3<i32> cell_of(Vec3f p) const noexcept {
        Vec3f r=p-bounds.lo;
        return {i32(r.x/cell_size),i32(r.y/cell_size),i32(r.z/cell_size)};
    }

    void query_sphere(Vec3f center, f32 radius, std::function<void(int)> cb) const noexcept {
        i32 r=i32(std::ceil(radius/cell_size));
        auto ci=cell_of(center);
        for(i32 dx=-r;dx<=r;++dx) for(i32 dy=-r;dy<=r;++dy) for(i32 dz=-r;dz<=r;++dz){
            i32 ix=ci.x+dx,iy=ci.y+dy,iz=ci.z+dz;
            if(ix<0||ix>=dims.x||iy<0||iy>=dims.y||iz<0||iz>=dims.z) continue;
            for(int id:cells[ix*dims.y*dims.z+iy*dims.z+iz]) cb(id);
        }
    }
};

// ─── Octree ───────────────────────────────────────────────────────────────
struct OctreeNode {
    AABB3 bounds;
    std::array<int,8> children{-1,-1,-1,-1,-1,-1,-1,-1};
    std::vector<int>  items;
    bool is_leaf=true;
    int  depth=0;
};

struct Octree {
    std::vector<OctreeNode> nodes;
    std::vector<Vec3f>      positions;
    int max_items_per_node=8, max_depth=8;

    void build(AABB3 root_bounds, std::span<const Vec3f> pts) {
        positions.assign(pts.begin(),pts.end());
        nodes.clear(); nodes.push_back({root_bounds});
        for(int i=0;i<int(pts.size());++i) insert(0,i);
    }

    void insert(int node, int idx) {
        auto& n=nodes[node];
        if(n.is_leaf){
            n.items.push_back(idx);
            if(int(n.items.size())>max_items_per_node&&n.depth<max_depth){
                subdivide(node);
            }
        } else {
            int child=child_of(n.bounds,positions[idx]);
            if(n.children[child]>=0) insert(n.children[child],idx);
        }
    }

    int child_of(const AABB3& b, Vec3f p) const noexcept {
        Vec3f c=b.center();
        return (p.x>=c.x?1:0)|(p.y>=c.y?2:0)|(p.z>=c.z?4:0);
    }

    void subdivide(int node) {
        auto& n=nodes[node]; n.is_leaf=false;
        Vec3f c=n.bounds.center(), lo=n.bounds.lo, hi=n.bounds.hi;
        Vec3f corners[2]={lo,hi};
        for(int i=0;i<8;++i){
            Vec3f clo={i&1?c.x:lo.x,i&2?c.y:lo.y,i&4?c.z:lo.z};
            Vec3f chi={i&1?hi.x:c.x,i&2?hi.y:c.y,i&4?hi.z:c.z};
            int cn=int(nodes.size());
            nodes.push_back({{clo,chi},{},{},{},{},{},{},{},{},{},true,n.depth+1});
            nodes[node].children[i]=cn;
        }
        auto items=nodes[node].items; nodes[node].items.clear();
        for(int idx:items) insert(node,idx);
    }
};

} // namespace qs::math::geometry
