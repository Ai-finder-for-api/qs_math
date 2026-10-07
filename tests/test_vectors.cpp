// qs_math/tests/test_vectors.cpp
#include "qs_math/vec2.hpp"
#include "qs_math/vec3.hpp"
#include "qs_math/vec4.hpp"
#include "qs_math/vec_ops.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::test;

static void test_vec2() {
    Vec2f a{1,0}, b{0,1};
    assert(approx_eq(dot(a,b),0.f));
    assert(approx_eq(length(a),1.f));
    Vec2f c=a+b; assert(approx_eq(c,Vec2f{1,1}));
    Vec2f n=normalize(Vec2f{3,4}); assert(approx_eq(length(n),1.f));
    assert(approx_eq(a.cross(b),1.f));
    assert(approx_eq(distance(Vec2f{0,0},Vec2f{3,4}),5.f));
    Vec2f r=Vec2f{1,0}.reflect(Vec2f{0,1}); assert(approx_eq(r,Vec2f{1,0}));
    Vec2f lerped=lerp(Vec2f{0,0},Vec2f{2,4},0.5f); assert(approx_eq(lerped,Vec2f{1,2}));
}
static void test_vec3() {
    Vec3f x=Vec3f::unit_x(), y=Vec3f::unit_y();
    Vec3f z=cross(x,y); assert(approx_eq(z,Vec3f::unit_z()));
    assert(approx_eq(dot(x,y),0.f));
    assert(approx_eq(length(Vec3f{1,2,2}),3.f));
    Vec3f n=normalize(Vec3f{1,2,2}); assert(approx_eq(length(n),1.f));
    Vec3f proj=Vec3f{1,1,0}.project_onto(Vec3f::unit_x()); assert(approx_eq(proj,Vec3f{1,0,0}));
    Vec3f refl=Vec3f{1,-1,0}.reflect(Vec3f::unit_y()); assert(approx_eq(refl,Vec3f{1,1,0}));
    assert(Vec3f::zero().is_zero());
}
static void test_vec4() {
    Vec4f a{1,2,3,4};
    assert(approx_eq(dot(a,a),30.f));
    Vec4f n=normalize(a); assert(approx_eq(length(n),1.f));
    assert(approx_eq(a.xyz(),Vec3f{1,2,3}));
    Vec4f lerped=lerp(Vec4f{0},Vec4f{4,4,4,4},0.25f); assert(approx_eq(lerped,Vec4f{1,1,1,1}));
}
static void test_vec_ops() {
    Vec3f a{1,0,0}, b{0,1,0};
    f32 ang=angle_between(a,b);
    assert(approx_eq(ang,HALF_PI,1e-5f));
    Vec3f c=move_towards(Vec3f{0},Vec3f{10,0,0},3.f);
    assert(approx_eq(c,Vec3f{3,0,0}));
    Vec3f bary=barycentric(Vec3f{0,0,0},Vec3f{1,0,0},Vec3f{0,1,0},Vec3f{0.5f,0.25f,0});
    assert(bary.x+bary.y+bary.z > 0.99f);
}
static void test_swizzles() {
    Vec3f v{1,2,3};
    assert(approx_eq(v.xy(),Vec2f{1,2}));
    assert(approx_eq(v.zyx(),Vec3f{3,2,1}));
    Vec4f w{1,2,3,4};
    assert(approx_eq(w.xyz(),Vec3f{1,2,3}));
}
int main() {
    test_vec2(); printf("vec2 OK\n");
    test_vec3(); printf("vec3 OK\n");
    test_vec4(); printf("vec4 OK\n");
    test_vec_ops(); printf("vec_ops OK\n");
    test_swizzles(); printf("swizzles OK\n");
    printf("=== test_vectors PASSED ===\n");
    return 0;
}
