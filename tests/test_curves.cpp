// qs_math/tests/test_curves.cpp
#include "qs_math/curves.hpp"
#include "qs_math/interpolation.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::test;

static void test_bezier_endpoints() {
    Vec3f p0{0,0,0},p1{1,2,0},p2{2,2,0},p3{3,0,0};
    assert(approx_eq(bezier_cubic(p0,p1,p2,p3,0.f),p0));
    assert(approx_eq(bezier_cubic(p0,p1,p2,p3,1.f),p3));
}
static void test_hermite() {
    Vec3f p0{0,0,0},m0{1,0,0},p1{1,0,0},m1{1,0,0};
    Vec3f at0=hermite(p0,m0,p1,m1,0.f);
    Vec3f at1=hermite(p0,m0,p1,m1,1.f);
    assert(approx_eq(at0,p0)); assert(approx_eq(at1,p1));
}
static void test_catmull_rom() {
    Vec3f p0{-1,0,0},p1{0,0,0},p2{1,1,0},p3{2,0,0};
    Vec3f at0=catmull_rom(p0,p1,p2,p3,0.f);
    Vec3f at1=catmull_rom(p0,p1,p2,p3,1.f);
    assert(approx_eq(at0,p1)); assert(approx_eq(at1,p2));
}
static void test_easing() {
    using namespace easing;
    assert(approx_eq(linear(0.f),0.f)); assert(approx_eq(linear(1.f),1.f));
    assert(approx_eq(quad_in(0.f),0.f)); assert(approx_eq(quad_in(1.f),1.f));
    assert(approx_eq(quad_out(0.f),0.f)); assert(approx_eq(quad_out(1.f),1.f));
    assert(approx_eq(bounce_out(0.f),0.f,1e-4f)); assert(approx_eq(bounce_out(1.f),1.f,1e-4f));
    // Monotonicity check for quad
    f32 prev=0.f;
    for(i32 i=1;i<=10;++i){ f32 v=quad_in(i*0.1f); assert(v>=prev); prev=v; }
}
static void test_bilinear() {
    f32 v=bilinear(0.f,1.f,0.f,1.f,0.5f,0.5f);
    assert(approx_eq(v,0.5f));
    v=bilinear(0.f,1.f,0.f,1.f,1.f,0.f);
    assert(approx_eq(v,1.f));
}
int main() {
    test_bezier_endpoints(); printf("bezier_endpoints OK\n");
    test_hermite(); printf("hermite OK\n");
    test_catmull_rom(); printf("catmull_rom OK\n");
    test_easing(); printf("easing OK\n");
    test_bilinear(); printf("bilinear OK\n");
    printf("=== test_curves PASSED ===\n");
    return 0;
}
