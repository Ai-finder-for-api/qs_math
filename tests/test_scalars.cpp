// qs_math/tests/test_scalars.cpp
#include "qs_math/scalar_math.hpp"
#include "qs_math/constants.hpp"
#include "qs_math/trig.hpp"
#include "qs_math/angle.hpp"
#include "qs_math/precision.hpp"
#include <cassert>
#include <cstdio>

using namespace qs::math;
using namespace qs::math::literals;

static void test_clamp() {
    assert(clamp(5.f,0.f,1.f)==1.f);
    assert(clamp(-1.f,0.f,1.f)==0.f);
    assert(clamp(0.5f,0.f,1.f)==0.5f);
    assert(clamp(10,0,5)==5);
}
static void test_lerp() {
    assert(nearly_equal(lerp(0.f,10.f,0.5f),5.f));
    assert(nearly_equal(lerp(0.f,10.f,0.f),0.f));
    assert(nearly_equal(lerp(0.f,10.f,1.f),10.f));
}
static void test_smoothstep() {
    assert(nearly_equal(smoothstep(0.f,1.f,0.f),0.f));
    assert(nearly_equal(smoothstep(0.f,1.f,1.f),1.f));
    f32 mid=smoothstep(0.f,1.f,0.5f);
    assert(mid>0.f&&mid<1.f);
}
static void test_trig() {
    assert(nearly_equal(sin(0.f),0.f));
    assert(nearly_equal(cos(0.f),1.f));
    assert(nearly_equal(sin(Radians<f32>{HALF_PI}),1.f,1e-5f));
    f32 s,c; sincos(PI/4.f,s,c);
    assert(nearly_equal(s,c,1e-5f));
}
static void test_angle_types() {
    Degreesf d{180.f};
    Radiansf r = d.to_radians();
    assert(nearly_equal(r.value,PI,1e-5f));
    auto d2 = r.to_degrees();
    assert(nearly_equal(d2.value,180.f,1e-4f));
    Degreesf n{-190.f};
    assert(n.normalized().value > -180.f);
}
static void test_power_of_two() {
    assert(is_power_of_two(1));  assert(is_power_of_two(64));
    assert(!is_power_of_two(3)); assert(!is_power_of_two(0));
    assert(next_power_of_two(5)==8);
    assert(next_power_of_two(8)==8);
}
static void test_wrap() {
    assert(nearly_equal(wrap(1.5f,0.f,1.f),0.5f));
    assert(nearly_equal(wrap(-0.1f,0.f,1.f),0.9f,1e-5f));
}
static void test_precision() {
    assert(equal_ulp(1.f,1.f));
    assert(!equal_ulp(1.f,2.f));
    assert(equal_abs(0.f,1e-7f,1e-6f));
    assert(!is_subnormal(1.f));
}
static void test_fast_rsqrt() {
    f32 v=fast_rsqrt(4.f);
    assert(nearly_equal(v,0.5f,1e-3f));
}

int main() {
    test_clamp(); printf("clamp OK\n");
    test_lerp();  printf("lerp OK\n");
    test_smoothstep(); printf("smoothstep OK\n");
    test_trig();  printf("trig OK\n");
    test_angle_types(); printf("angle_types OK\n");
    test_power_of_two(); printf("power_of_two OK\n");
    test_wrap();  printf("wrap OK\n");
    test_precision(); printf("precision OK\n");
    test_fast_rsqrt(); printf("fast_rsqrt OK\n");
    printf("=== test_scalars PASSED ===\n");
    return 0;
}
