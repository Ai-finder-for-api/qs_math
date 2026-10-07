// qs_math/tests/test_transform.cpp
#include "qs_math/transform.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::test;

static void test_identity() {
    Transformf t = Transformf::identity();
    Vec3f v{1,2,3};
    assert(approx_eq(t.transform_point(v), v));
    assert(approx_eq(t.transform_dir(v), v));
}
static void test_translation() {
    Transformf t{{1,2,3},Quatf::identity(),{1,1,1}};
    assert(approx_eq(t.transform_point({0,0,0}),Vec3f{1,2,3}));
    assert(approx_eq(t.transform_dir({1,0,0}),Vec3f{1,0,0}));
}
static void test_rotation() {
    Quatf q = Quatf::from_axis_angle(Vec3f::unit_y(), HALF_PI);
    Transformf t{{0,0,0},q,{1,1,1}};
    Vec3f v = t.transform_point(Vec3f::unit_x());
    assert(approx_eq(v, Vec3f{0,0,-1}, 1e-5f));
}
static void test_scale() {
    Transformf t{{0,0,0},Quatf::identity(),{2,3,4}};
    Vec3f v = t.transform_point({1,1,1});
    assert(approx_eq(v, Vec3f{2,3,4}));
}
static void test_inverse() {
    Transformf t{{1,2,3},Quatf::from_axis_angle({1,0,0},0.5f),{2,2,2}};
    Vec3f p{5,6,7};
    Vec3f tp = t.transform_point(p);
    Vec3f back = t.inverse_transform_point(tp);
    assert(approx_eq(back, p, 1e-4f));
}
static void test_compose() {
    Transformf a{{1,0,0},Quatf::identity(),{1,1,1}};
    Transformf b{{0,1,0},Quatf::identity(),{1,1,1}};
    Transformf c = a * b;
    Vec3f p = c.transform_point({0,0,0});
    assert(approx_eq(p, Vec3f{1,1,0}));
}
static void test_mat4_roundtrip() {
    Transformf t{{1,2,3},Quatf::from_axis_angle({0,1,0},0.7f),{2,1,3}};
    Mat4f m = t.to_mat4();
    Transformf t2 = Transformf::from_mat4(m);
    Vec3f p{4,5,6};
    assert(approx_eq(t.transform_point(p), t2.transform_point(p), 1e-3f));
}
static void test_lerp_transform() {
    Transformf a = Transformf::identity();
    Transformf b{{2,0,0},Quatf::identity(),{2,2,2}};
    Transformf mid = Transformf::lerp(a,b,0.5f);
    assert(approx_eq(mid.translation, Vec3f{1,0,0}));
    assert(approx_eq(mid.scale, Vec3f{1.5f,1.5f,1.5f}));
}
int main() {
    test_identity();  printf("identity OK\n");
    test_translation();printf("translation OK\n");
    test_rotation();  printf("rotation OK\n");
    test_scale();     printf("scale OK\n");
    test_inverse();   printf("inverse OK\n");
    test_compose();   printf("compose OK\n");
    test_mat4_roundtrip(); printf("mat4_roundtrip OK\n");
    test_lerp_transform(); printf("lerp_transform OK\n");
    printf("=== test_transform PASSED ===\n");
    return 0;
}
