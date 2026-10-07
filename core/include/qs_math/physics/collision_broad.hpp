#pragma once
// collision_broad.hpp — Broad-phase collision: AABB BVH, sweep-and-prune,
//                       uniform grid, dynamic AABB tree, pair management
#include "geometry3d.hpp"
#include "bounding_volume.hpp"
#include "hashing.hpp"
#include "scalar_math.hpp"

namespace qs::math::collision {

// ---------------------------------------------------------------------------
// Bounding volume hierarchy (dynamic AABB tree, Erin Catto style)
// ---------------------------------------------------------------------------
struct BVHNode {
    AABBf   aabb;
    i32     parent{-1};
    i32     child[2]{-1,-1};
    i32     height{0};       // leaf=0
    i32     user_data{-1};   // index into user's object array
    bool    is_leaf() const noexcept { return child[0]==-1; }
};

struct DynamicAABBTree {
    static constexpr i32 MAX_NODES = 8192;
    BVHNode nodes[MAX_NODES];
    i32     root{-1};
    i32     free_list{0};
    i32     node_count{0};

    DynamicAABBTree() noexcept {
        for(i32 i=0;i<MAX_NODES-1;++i) nodes[i].parent=i+1;
        nodes[MAX_NODES-1].parent=-1;
    }

    i32 alloc_node() noexcept {
        if(free_list==-1) return -1;
        i32 idx=free_list;
        free_list=nodes[idx].parent;
        nodes[idx]={};
        nodes[idx].height=-1;
        ++node_count;
        return idx;
    }

    void free_node(i32 idx) noexcept {
        nodes[idx].parent=free_list;
        nodes[idx].height=-1;
        free_list=idx;
        --node_count;
    }

    // Insert a leaf with given AABB, returns node index
    i32 insert(AABBf aabb, i32 user_data) noexcept {
        i32 leaf=alloc_node();
        if(leaf<0) return -1;
        // Fat AABB (expand by 0.1 for movement prediction)
        nodes[leaf].aabb=aabb.expanded(0.1f);
        nodes[leaf].user_data=user_data;
        nodes[leaf].height=0;
        insert_leaf(leaf);
        return leaf;
    }

    void remove(i32 leaf) noexcept {
        remove_leaf(leaf);
        free_node(leaf);
    }

    // Re-insert a leaf when its AABB changes
    bool update(i32 leaf, AABBf new_aabb) noexcept {
        if(nodes[leaf].aabb.contains(new_aabb)) return false;
        remove_leaf(leaf);
        nodes[leaf].aabb=new_aabb.expanded(0.1f);
        insert_leaf(leaf);
        return true;
    }

    // Query: call callback(user_data_a, user_data_b) for all overlapping pairs
    template<typename Fn>
    void query_aabb(AABBf aabb, Fn&& callback) const noexcept {
        if(root<0) return;
        i32 stack[256]; i32 sp=0;
        stack[sp++]=root;
        while(sp>0){
            i32 idx=stack[--sp];
            if(!nodes[idx].aabb.intersects(aabb)) continue;
            if(nodes[idx].is_leaf()){
                callback(nodes[idx].user_data);
            } else {
                if(nodes[idx].child[0]>=0) stack[sp++]=nodes[idx].child[0];
                if(nodes[idx].child[1]>=0) stack[sp++]=nodes[idx].child[1];
            }
        }
    }

    // Ray query
    template<typename Fn>
    void query_ray(Ray3f ray, f32 max_t, Fn&& callback) const noexcept {
        if(root<0) return;
        i32 stack[256]; i32 sp=0;
        stack[sp++]=root;
        while(sp>0){
            i32 idx=stack[--sp];
            auto hit=ray_aabb(ray,nodes[idx].aabb);
            if(!hit.hit||hit.t>max_t) continue;
            if(nodes[idx].is_leaf()){
                callback(nodes[idx].user_data, hit.t);
            } else {
                if(nodes[idx].child[0]>=0) stack[sp++]=nodes[idx].child[0];
                if(nodes[idx].child[1]>=0) stack[sp++]=nodes[idx].child[1];
            }
        }
    }

    // Collect all overlapping pairs (for self-collision detection)
    template<typename Fn>
    void collect_pairs(Fn&& callback) const noexcept {
        if(root<0) return;
        struct Pair { i32 a,b; };
        Pair stack[1024]; i32 sp=0;
        // Traverse and collect leaf-leaf overlaps
        query_subtree_pairs(root, callback);
    }

private:
    void insert_leaf(i32 leaf) noexcept {
        if(root==-1){ root=leaf; nodes[root].parent=-1; return; }
        // Find best sibling (surface area heuristic)
        AABBf leaf_aabb=nodes[leaf].aabb;
        i32 best=root;
        while(!nodes[best].is_leaf()){
            f32 area=nodes[best].aabb.surface_area();
            AABBf combined=nodes[best].aabb.merge(leaf_aabb);
            f32 combined_area=combined.surface_area();
            f32 cost=2.f*combined_area;
            f32 inherit_cost=2.f*(combined_area-area);
            f32 cost0=nodes[nodes[best].child[0]].is_leaf()?
                leaf_aabb.merge(nodes[nodes[best].child[0]].aabb).surface_area()+inherit_cost:
                leaf_aabb.merge(nodes[nodes[best].child[0]].aabb).surface_area()-nodes[nodes[best].child[0]].aabb.surface_area()+inherit_cost;
            f32 cost1=nodes[nodes[best].child[1]].is_leaf()?
                leaf_aabb.merge(nodes[nodes[best].child[1]].aabb).surface_area()+inherit_cost:
                leaf_aabb.merge(nodes[nodes[best].child[1]].aabb).surface_area()-nodes[nodes[best].child[1]].aabb.surface_area()+inherit_cost;
            if(cost<cost0&&cost<cost1) break;
            best=(cost0<cost1)?nodes[best].child[0]:nodes[best].child[1];
        }
        // Create new parent
        i32 old_parent=nodes[best].parent;
        i32 new_parent=alloc_node();
        if(new_parent<0) return;
        nodes[new_parent].parent=old_parent;
        nodes[new_parent].aabb=nodes[best].aabb.merge(leaf_aabb);
        nodes[new_parent].height=nodes[best].height+1;
        if(old_parent>=0){
            if(nodes[old_parent].child[0]==best) nodes[old_parent].child[0]=new_parent;
            else nodes[old_parent].child[1]=new_parent;
        } else root=new_parent;
        nodes[new_parent].child[0]=best;
        nodes[new_parent].child[1]=leaf;
        nodes[best].parent=new_parent;
        nodes[leaf].parent=new_parent;
        // Walk back and refit
        i32 idx=nodes[leaf].parent;
        while(idx>=0){
            i32 c0=nodes[idx].child[0], c1=nodes[idx].child[1];
            nodes[idx].height=1+max(nodes[c0].height,nodes[c1].height);
            nodes[idx].aabb=nodes[c0].aabb.merge(nodes[c1].aabb);
            idx=nodes[idx].parent;
        }
    }

    void remove_leaf(i32 leaf) noexcept {
        if(leaf==root){ root=-1; return; }
        i32 parent=nodes[leaf].parent;
        i32 sibling=(nodes[parent].child[0]==leaf)?nodes[parent].child[1]:nodes[parent].child[0];
        i32 grand=nodes[parent].parent;
        if(grand>=0){
            if(nodes[grand].child[0]==parent) nodes[grand].child[0]=sibling;
            else nodes[grand].child[1]=sibling;
            nodes[sibling].parent=grand;
            free_node(parent);
            i32 idx=grand;
            while(idx>=0){
                i32 c0=nodes[idx].child[0], c1=nodes[idx].child[1];
                nodes[idx].aabb=nodes[c0].aabb.merge(nodes[c1].aabb);
                nodes[idx].height=1+max(nodes[c0].height,nodes[c1].height);
                idx=nodes[idx].parent;
            }
        } else {
            root=sibling;
            nodes[sibling].parent=-1;
            free_node(parent);
        }
    }

    template<typename Fn>
    void query_subtree_pairs(i32 node, Fn&& cb) const noexcept {
        if(node<0||nodes[node].is_leaf()) return;
        i32 c0=nodes[node].child[0], c1=nodes[node].child[1];
        query_subtree_pairs(c0,cb);
        query_subtree_pairs(c1,cb);
        query_subtree_overlap(c0,c1,cb);
    }

    template<typename Fn>
    void query_subtree_overlap(i32 a, i32 b, Fn&& cb) const noexcept {
        if(a<0||b<0) return;
        if(!nodes[a].aabb.intersects(nodes[b].aabb)) return;
        if(nodes[a].is_leaf()&&nodes[b].is_leaf()){
            cb(nodes[a].user_data,nodes[b].user_data); return;
        }
        if(nodes[b].is_leaf()||((!nodes[a].is_leaf())&&nodes[a].aabb.volume()>=nodes[b].aabb.volume())){
            query_subtree_overlap(nodes[a].child[0],b,cb);
            query_subtree_overlap(nodes[a].child[1],b,cb);
        } else {
            query_subtree_overlap(a,nodes[b].child[0],cb);
            query_subtree_overlap(a,nodes[b].child[1],cb);
        }
    }
};

// ---------------------------------------------------------------------------
// Sweep and Prune (SAP) — 1-axis, n objects
// ---------------------------------------------------------------------------
struct SAPEndpoint {
    f32  value;
    i32  obj_id;
    bool is_max;
};

struct SAPAxis {
    static constexpr i32 MAX_OBJ=1024;
    SAPEndpoint endpoints[MAX_OBJ*2];
    i32 count{0};

    void insert(i32 obj_id, f32 min_val, f32 max_val) noexcept {
        if(count+2>MAX_OBJ*2) return;
        endpoints[count++]={min_val,obj_id,false};
        endpoints[count++]={max_val,obj_id,true};
    }

    void sort() noexcept {
        // Insertion sort (good for nearly-sorted each frame)
        for(i32 i=1;i<count;++i){
            SAPEndpoint key=endpoints[i]; i32 j=i-1;
            while(j>=0&&endpoints[j].value>key.value){ endpoints[j+1]=endpoints[j]; --j; }
            endpoints[j+1]=key;
        }
    }

    template<typename Fn>
    void find_pairs(Fn&& callback) noexcept {
        sort();
        i32 active[MAX_OBJ]; i32 active_count=0;
        for(i32 i=0;i<count;++i){
            if(!endpoints[i].is_max){
                // Start — check against all active
                for(i32 j=0;j<active_count;++j)
                    callback(endpoints[i].obj_id, active[j]);
                if(active_count<MAX_OBJ) active[active_count++]=endpoints[i].obj_id;
            } else {
                // End — remove from active
                for(i32 j=0;j<active_count;++j)
                    if(active[j]==endpoints[i].obj_id){ active[j]=active[--active_count]; break; }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Uniform Grid broad-phase
// ---------------------------------------------------------------------------
struct UniformGrid {
    static constexpr i32 MAX_CELLS=4096;
    static constexpr i32 MAX_PER_CELL=32;

    struct Cell {
        i32 objects[MAX_PER_CELL];
        i32 count{0};
        void add(i32 id) noexcept { if(count<MAX_PER_CELL) objects[count++]=id; }
        void clear() noexcept { count=0; }
    };

    Cell  cells[MAX_CELLS];
    Vec3f origin{0.f};
    f32   cell_size{2.f};
    i32   dims[3]{8,8,8};

    void clear() noexcept {
        for(i32 i=0;i<dims[0]*dims[1]*dims[2];++i) cells[i].clear();
    }

    i32 cell_index(i32 x,i32 y,i32 z) const noexcept {
        x=clamp(x,0,dims[0]-1); y=clamp(y,0,dims[1]-1); z=clamp(z,0,dims[2]-1);
        return z*dims[0]*dims[1]+y*dims[0]+x;
    }

    void insert_aabb(i32 obj_id, AABBf aabb) noexcept {
        i32 x0=(i32)((aabb.min_pt.x-origin.x)/cell_size);
        i32 y0=(i32)((aabb.min_pt.y-origin.y)/cell_size);
        i32 z0=(i32)((aabb.min_pt.z-origin.z)/cell_size);
        i32 x1=(i32)((aabb.max_pt.x-origin.x)/cell_size);
        i32 y1=(i32)((aabb.max_pt.y-origin.y)/cell_size);
        i32 z1=(i32)((aabb.max_pt.z-origin.z)/cell_size);
        for(i32 z=z0;z<=z1;++z) for(i32 y=y0;y<=y1;++y) for(i32 x=x0;x<=x1;++x)
            cells[cell_index(x,y,z)].add(obj_id);
    }

    template<typename Fn>
    void collect_pairs(Fn&& callback) const noexcept {
        // Collect unique pairs from multi-cell objects
        for(i32 c=0;c<dims[0]*dims[1]*dims[2];++c){
            const Cell& cell=cells[c];
            for(i32 i=0;i<cell.count;++i)
                for(i32 j=i+1;j<cell.count;++j)
                    callback(cell.objects[i],cell.objects[j]);
        }
    }
};

// ---------------------------------------------------------------------------
// Pair cache (dedup broad-phase pairs)
// ---------------------------------------------------------------------------
struct CollisionPair { i32 a,b; };

inline i32 deduplicate_pairs(CollisionPair* pairs, i32 count) noexcept {
    // Sort and remove duplicates
    for(i32 i=0;i<count;++i) if(pairs[i].a>pairs[i].b){i32 t=pairs[i].a;pairs[i].a=pairs[i].b;pairs[i].b=t;}
    // Sort by a then b
    for(i32 i=1;i<count;++i){
        CollisionPair k=pairs[i]; i32 j=i-1;
        while(j>=0&&(pairs[j].a>k.a||(pairs[j].a==k.a&&pairs[j].b>k.b))){pairs[j+1]=pairs[j];--j;}
        pairs[j+1]=k;
    }
    i32 n=0;
    for(i32 i=0;i<count;++i){
        if(n==0||pairs[i].a!=pairs[n-1].a||pairs[i].b!=pairs[n-1].b)
            pairs[n++]=pairs[i];
    }
    return n;
}

} // namespace qs::math::collision
