// qs_math/tests/test_geometry.cpp
#include "qs_math/geometry3d.hpp"
#include "qs_math/intersection.hpp"
#include "qs_math/bounding_volume.hpp"
#include "qs_math/geometry2d.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::test;

static void test_ray_aabb() {
    AABBf box{Vec3f{-1,-1,-1}, Vec3f{1,1,1}};
    Ray3f ray{Vec3f{0,0,-5}, Vec3f{0,0,1}};
    auto hit = ray_aabb(ray, box);
    assert(hit.hit);
    assert(approx_eq(hit.t, 4.f, 1e-4f));
    // Miss
    Ray3f miss{Vec3f{5,0,-5}, Vec3f{0,0,1}};
    assert(!ray_aabb(miss,box).hit);
}
static void test_ray_sphere() {
    Spheref s{{0,0,0}, 1.f};
    Ray3f ray{{0,0,-5},{0,0,1}};
    auto hit = ray_sphere(ray,s);
    assert(hit.hit);
    assert(approx_eq(hit.t, 4.f, 1e-4f));
    Ray3f miss{{5,0,-5},{0,0,1}};
    assert(!ray_sphere(miss,s).hit);
}
static void test_ray_triangle() {
    Triangle3f tri{{-1,0,0},{1,0,0},{0,1,0}};
    Ray3f ray{{0,0.3f,-5},{0,0,1}};
    auto hit = ray_triangle(ray,tri);
    assert(hit.hit);
    Ray3f miss{{0,2.f,-5},{0,0,1}};
    assert(!ray_triangle(miss,tri).hit);
}
static void test_aabb() {
    AABBf a{{0,0,0},{1,1,1}}, b{{0.5f,0.5f,0.5f},{2,2,2}};
    assert(aabb_aabb(a,b));
    AABBf c{{5,5,5},{6,6,6}};
    assert(!aabb_aabb(a,c));
    assert(a.contains({0.5f,0.5f,0.5f}));
    assert(!a.contains({2,2,2}));
    assert(approx_eq(a.volume(),1.f));
    auto corners=a.corners(); assert(corners.size()==8);
}
static void test_sphere() {
    Spheref s{{0,0,0},2.f};
    assert(s.contains({1,0,0}));
    assert(!s.contains({3,0,0}));
    Spheref s2{{1,0,0},2.f};
    assert(s.intersects(s2));
    Spheref merged=s.merge(Spheref{{5,0,0},1.f});
    assert(merged.radius>2.f);
}
static void test_plane() {
    Planef p{Vec3f{0,1,0},0.f}; // XZ plane at y=0
    assert(approx_eq(p.signed_distance({0,1,0}),1.f));
    assert(approx_eq(p.signed_distance({0,-1,0}),-1.f));
    Vec3f proj=p.project({1,2,3});
    assert(approx_eq(proj.y,0.f,1e-5f));
}
static void test_bounding_sphere() {
    Vec3f pts[]={{-1,0,0},{1,0,0},{0,1,0},{0,-1,0}};
    Spheref bs=bounding_sphere_ritter(pts,4);
    for(auto& p:pts) assert(bs.contains(p));
}
static void test_rect2() {
    Rect2f r{Vec2f{0,0},Vec2f{2,2}};
    assert(r.contains({1,1}));
    assert(!r.contains({3,3}));
    assert(approx_eq(r.area(),4.f));
    assert(approx_eq(r.center(),Vec2f{1,1}));
}
int main() {
    test_ray_aabb(); printf("ray_aabb OK\n");
    test_ray_sphere(); printf("ray_sphere OK\n");
    test_ray_triangle(); printf("ray_triangle OK\n");
    test_aabb(); printf("aabb OK\n");
    test_sphere(); printf("sphere OK\n");
    test_plane(); printf("plane OK\n");
    test_bounding_sphere(); printf("bounding_sphere OK\n");
    test_rect2(); printf("rect2 OK\n");
    printf("=== test_geometry PASSED ===\n");
    return 0;
}
