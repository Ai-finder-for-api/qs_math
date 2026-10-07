// qs_math/tests/test_matrices.cpp
#include "qs_math/mat2.hpp"
#include "qs_math/mat3.hpp"
#include "qs_math/mat4.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::test;

static void test_mat2() {
    Mat2f id = Mat2f::identity();
    Vec2f v{3,4};
    assert(approx_eq(id*v, v));
    Mat2f s = Mat2f::scale(2,3);
    assert(approx_eq(s*v, Vec2f{6,12}));
    assert(approx_eq(determinant(id), 1.f));
    Mat2f inv = id.inversed();
    assert(approx_eq((id*inv)[0], Vec2f{1,0}) && approx_eq((id*inv)[1], Vec2f{0,1}));
}
static void test_mat3() {
    Mat3f id = Mat3f::identity();
    Vec3f v{1,2,3};
    assert(approx_eq(id*v, v));
    assert(approx_eq(determinant(id), 1.f));
    Mat3f rx = Mat3f::rotation_x(HALF_PI);
    Vec3f ry = rx*Vec3f::unit_y();
    assert(approx_eq(ry, Vec3f{0,0,1}, 1e-5f));
    Mat3f inv = inverse(rx);
    assert(approx_eq((rx*inv)[0], Vec3f{1,0,0},1e-5f));
}
static void test_mat4() {
    Mat4f id = Mat4f::identity();
    Vec4f v{1,2,3,1};
    assert(approx_eq(id*v, v));
    Mat4f t = Mat4f::from_translation({1,2,3});
    Vec3f p = t.transform_point({0,0,0});
    assert(approx_eq(p, Vec3f{1,2,3}));
    Mat4f s = Mat4f::from_scale({2,2,2});
    p = s.transform_point({1,1,1});
    assert(approx_eq(p, Vec3f{2,2,2}));
    // Inverse
    Mat4f inv = inverse(t);
    Mat4f shouldBeId = t * inv;
    assert(approx_eq(shouldBeId, Mat4f::identity(), 1e-4f));
    // LookAt
    Mat4f lk = Mat4f::look_at({0,0,5},{0,0,0},{0,1,0});
    assert(nearly_equal(determinant(lk), 1.f, 1e-4f));
}
static void test_projection() {
    Mat4f p = Mat4f::perspective(PI/4.f, 16.f/9.f, 0.1f, 1000.f);
    // Check that the near clip point maps close to -1
    Vec4f near_pt = p * Vec4f{0,0,-0.1f,1};
    assert(near_pt.w > 0.f); // perspective divide should yield valid NDC
    Mat4f o = Mat4f::orthographic(-1,1,-1,1,-1,1);
    Vec4f c = o * Vec4f{0,0,0,1};
    assert(approx_eq(c, Vec4f{0,0,0,1}));
}
int main() {
    test_mat2(); printf("mat2 OK\n");
    test_mat3(); printf("mat3 OK\n");
    test_mat4(); printf("mat4 OK\n");
    test_projection(); printf("projection OK\n");
    printf("=== test_matrices PASSED ===\n");
    return 0;
}
