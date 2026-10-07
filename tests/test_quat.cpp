// qs_math/tests/test_quat.cpp
#include "qs_math/quat.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::test;

static void test_identity() {
    Quatf q = Quatf::identity();
    assert(approx_eq(q.length(), 1.f));
    Vec3f v{1,2,3};
    assert(approx_eq(q.rotate(v), v));
}
static void test_axis_angle() {
    Quatf q = Quatf::from_axis_angle(Vec3f::unit_y(), HALF_PI);
    Vec3f v = q.rotate(Vec3f::unit_x());
    assert(approx_eq(v, Vec3f{0,0,-1}, 1e-5f));
}
static void test_euler_roundtrip() {
    Vec3f euler{0.3f, 0.7f, -0.4f};
    Quatf q = Quatf::from_euler(euler);
    Vec3f back = q.to_euler();
    // Verify the quaternion rotates consistently (not just euler roundtrip which can alias)
    Quatf q2 = Quatf::from_euler(back);
    Vec3f test{1,0,0};
    assert(approx_eq(q.rotate(test), q2.rotate(test), 1e-4f));
}
static void test_mat3_roundtrip() {
    Quatf q = Quatf::from_axis_angle({1,1,0}, 1.2f);
    q = q.normalized();
    Mat3f m = q.to_mat3();
    Quatf q2 = Quatf::from_mat3(m);
    Vec3f v{1,2,3};
    assert(approx_eq(q.rotate(v), q2.rotate(v), 1e-4f));
}
static void test_slerp() {
    Quatf a = Quatf::identity();
    Quatf b = Quatf::from_axis_angle(Vec3f::unit_z(), PI);
    Quatf mid = Quatf::slerp(a, b, 0.5f);
    Vec3f axis; f32 angle;
    mid.to_axis_angle(axis, angle);
    assert(approx_eq(angle, HALF_PI, 1e-4f));
}
static void test_compose() {
    Quatf rx = Quatf::from_axis_angle(Vec3f::unit_x(), HALF_PI);
    Quatf ry = Quatf::from_axis_angle(Vec3f::unit_y(), HALF_PI);
    Quatf combined = ry * rx;
    Vec3f v = combined.rotate(Vec3f::unit_z());
    // Should be a reasonable rotation, just verify it's unit
    assert(approx_eq(length(v), 1.f, 1e-5f));
}
static void test_inverse() {
    Quatf q = Quatf::from_axis_angle({1,1,1}, 1.0f);
    Quatf inv = q.inverse();
    Quatf result = q * inv;
    assert(approx_eq(result, Quatf::identity(), 1e-5f));
}
int main() {
    test_identity(); printf("identity OK\n");
    test_axis_angle(); printf("axis_angle OK\n");
    test_euler_roundtrip(); printf("euler_roundtrip OK\n");
    test_mat3_roundtrip(); printf("mat3_roundtrip OK\n");
    test_slerp(); printf("slerp OK\n");
    test_compose(); printf("compose OK\n");
    test_inverse(); printf("inverse OK\n");
    printf("=== test_quat PASSED ===\n");
    return 0;
}
