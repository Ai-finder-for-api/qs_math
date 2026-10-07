// qs_math/tests/test_hashing_spatial.cpp
#include "qs_math/hashing.hpp"
#include "qs_math/spatial.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::spatial;

static void test_morton2() {
    assert(morton2(1u,0u)==1u);
    assert(morton2(0u,1u)==2u);
    assert(morton2(1u,1u)==3u);
    assert(morton2(2u,0u)==4u);
}
static void test_spatial_hash() {
    Vec3f p1{0.5f,0.5f,0.5f}, p2{0.5f,0.5f,0.5f}, p3{1.5f,0.5f,0.5f};
    f32 cs=1.f;
    assert(spatial_hash3(p1,cs)==spatial_hash3(p2,cs));
    assert(spatial_hash3(p1,cs)!=spatial_hash3(p3,cs));
}
static void test_grid_index() {
    assert(grid_index_2d(0,0,10)==0);
    assert(grid_index_2d(1,0,10)==1);
    assert(grid_index_2d(0,1,10)==10);
    Vec2i c=grid_coord_2d(11,10);
    assert(c.x==1 && c.y==1);
}
static void test_octree() {
    u64 root=0;
    u64 child=octree_child(root,3);
    assert(octree_parent(child)==root);
}
static void test_vec_hash_consistency() {
    Vec3f a{1.f,2.f,3.f};
    assert(hash(a)==hash(a));
    Vec3f b{1.f,2.f,4.f};
    assert(hash(a)!=hash(b));
}
int main() {
    test_morton2(); printf("morton2 OK\n");
    test_spatial_hash(); printf("spatial_hash OK\n");
    test_grid_index(); printf("grid_index OK\n");
    test_octree(); printf("octree OK\n");
    test_vec_hash_consistency(); printf("vec_hash OK\n");
    printf("=== test_hashing_spatial PASSED ===\n");
    return 0;
}
