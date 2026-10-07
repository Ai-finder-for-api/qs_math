// qs_math/tests/test_serialization.cpp
#include "qs_math/serialization_helpers.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::serial;
using namespace qs::math::test;

static void test_vec3_roundtrip() {
    u8 buf[12];
    Vec3f v{1.5f,2.5f,3.5f};
    write(buf,v);
    Vec3f back=read_vec3f(buf);
    assert(approx_eq(v,back));
}
static void test_mat4_roundtrip() {
    u8 buf[64];
    Mat4f m=Mat4f::from_translation({1,2,3});
    write(buf,m);
    Mat4f back=read_mat4f(buf);
    assert(approx_eq(m,back));
}
static void test_quat_roundtrip() {
    u8 buf[16];
    Quatf q=Quatf::from_axis_angle({0,1,0},1.f);
    write(buf,q);
    Quatf back=read_quatf(buf);
    assert(approx_eq(q,back));
}
static void test_header() {
    u8 buf[16];
    MathChunkHeader h{0x514D4154,1,42,0};
    write(buf,h);
    MathChunkHeader back=read_header(buf);
    assert(header_valid(back));
    assert(back.count==42);
}
int main() {
    test_vec3_roundtrip(); printf("vec3_roundtrip OK\n");
    test_mat4_roundtrip(); printf("mat4_roundtrip OK\n");
    test_quat_roundtrip(); printf("quat_roundtrip OK\n");
    test_header(); printf("header OK\n");
    printf("=== test_serialization PASSED ===\n");
    return 0;
}
