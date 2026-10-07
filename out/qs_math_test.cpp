// qs_math_test.cpp
// Build: cl /nologo /std:c++20 /EHsc /Icore/include /I..\qs_core\include
//           /Fout/qs_math_test.exe out/qs_math_core.lib qs_math_test.cpp
// Run:   out\qs_math_test.exe

#include "qs_math/arithmetic/scalar_types.hpp"
#include "qs_math/arithmetic/scalar_math.hpp"
#include "qs_math/arithmetic/constants.hpp"
#include "qs_math/linear_algebra/vec2.hpp"
#include "qs_math/linear_algebra/vec3.hpp"
#include "qs_math/linear_algebra/vec4.hpp"
#include "qs_math/linear_algebra/mat3.hpp"
#include "qs_math/linear_algebra/mat4.hpp"
#include "qs_math/linear_algebra/quat.hpp"

#include <cstdio>
#include <cmath>
#include <cassert>

// ─── Minimal test harness ─────────────────────────────────────────────────
static int s_pass = 0, s_fail = 0;

#define CHECK(expr) do { \
    if (expr) { ++s_pass; } \
    else { ++s_fail; std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
} while(0)

#define CHECKF(a, b) CHECK(std::abs((double)(a) - (double)(b)) < 1e-5)

#define SECTION(name) std::printf("\n[%s]\n", name)

using namespace qs::math;

// ─── scalar_types & scalar_math ───────────────────────────────────────────
void test_scalars() {
    SECTION("scalar_types / scalar_math");

    f32 a = 3.0f, b = 4.0f;
    CHECKF(clamp(a, 0.f, 2.f), 2.0f);
    CHECKF(clamp(a, 0.f, 10.f), 3.0f);
    CHECKF(lerp(0.f, 10.f, 0.5f), 5.0f);
    CHECKF(clamp01(1.5f), 1.0f);
    CHECKF(clamp01(-0.5f), 0.0f);

    CHECK(min(a, b) == 3.0f);
    CHECK(max(a, b) == 4.0f);

    f32 sq = a * a + b * b;
    CHECKF(std::sqrt(sq), 5.0f);
}

// ─── Vec2 ─────────────────────────────────────────────────────────────────
void test_vec2() {
    SECTION("Vec2f");

    Vec2f a{3.f, 4.f};
    Vec2f b{1.f, 2.f};

    CHECKF(a.length(), 5.0f);
    CHECKF(dot(a, b), 11.0f);

    Vec2f sum = a + b;
    CHECKF(sum.x, 4.f);
    CHECKF(sum.y, 6.f);

    Vec2f n = a.normalized();
    CHECKF(n.length(), 1.0f);

    Vec2f scaled = a * 2.f;
    CHECKF(scaled.x, 6.f);
    CHECKF(scaled.y, 8.f);
}

// ─── Vec3 ─────────────────────────────────────────────────────────────────
void test_vec3() {
    SECTION("Vec3f");

    Vec3f x{1.f, 0.f, 0.f};
    Vec3f y{0.f, 1.f, 0.f};
    Vec3f z{0.f, 0.f, 1.f};

    Vec3f c = cross(x, y);
    CHECKF(c.x, 0.f);
    CHECKF(c.y, 0.f);
    CHECKF(c.z, 1.f);

    CHECKF(dot(x, y), 0.f);
    CHECKF(dot(x, x), 1.f);

    Vec3f v{3.f, 4.f, 0.f};
    CHECKF(v.length(), 5.f);

    Vec3f n = v.normalized();
    CHECKF(n.length(), 1.0f);
    CHECKF(n.x, 0.6f);
    CHECKF(n.y, 0.8f);

    Vec3f lerped = lerp(x, z, 0.5f);
    CHECKF(lerped.x, 0.5f);
    CHECKF(lerped.z, 0.5f);
}

// ─── Vec4 ─────────────────────────────────────────────────────────────────
void test_vec4() {
    SECTION("Vec4f");

    Vec4f a{1.f, 2.f, 3.f, 4.f};
    Vec4f b{4.f, 3.f, 2.f, 1.f};

    Vec4f sum = a + b;
    CHECKF(sum.x, 5.f);
    CHECKF(sum.y, 5.f);
    CHECKF(sum.z, 5.f);
    CHECKF(sum.w, 5.f);

    CHECKF(dot(a, b), 20.f);  // 4+6+6+4

    Vec4f scaled = a * 2.f;
    CHECKF(scaled.x, 2.f);
    CHECKF(scaled.w, 8.f);
}

// ─── Mat3 ─────────────────────────────────────────────────────────────────
void test_mat3() {
    SECTION("Mat3f");

    Mat3f I = Mat3f::identity();
    Vec3f v{1.f, 2.f, 3.f};
    Vec3f result = I * v;
    CHECKF(result.x, 1.f);
    CHECKF(result.y, 2.f);
    CHECKF(result.z, 3.f);

    // Rotation 90° around Z
    Mat3f Rz = Mat3f::rotation_z(f32(PI) * 0.5f);
    Vec3f rx = Rz * Vec3f{1.f, 0.f, 0.f};
    CHECKF(rx.x,  0.f);
    CHECKF(rx.y,  1.f);
    CHECKF(rx.z,  0.f);

    // Transpose of identity = identity
    Mat3f It = I.transposed();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            CHECKF(It(i,j), I(i,j));
}

// ─── Mat4 ─────────────────────────────────────────────────────────────────
void test_mat4() {
    SECTION("Mat4f");

    Mat4f I = Mat4f::identity();
    Vec4f v{1.f, 2.f, 3.f, 1.f};
    Vec4f r = I * v;
    CHECKF(r.x, 1.f);
    CHECKF(r.w, 1.f);

    // Translation
    Mat4f T = Mat4f::translation({5.f, 0.f, 0.f});
    Vec4f p{0.f, 0.f, 0.f, 1.f};
    Vec4f tp = T * p;
    CHECKF(tp.x, 5.f);
    CHECKF(tp.y, 0.f);
    CHECKF(tp.z, 0.f);

    // Scale
    Mat4f S = Mat4f::scale({2.f, 3.f, 4.f});
    Vec4f sp = S * v;
    CHECKF(sp.x, 2.f);
    CHECKF(sp.y, 6.f);
    CHECKF(sp.z, 12.f);
}

// ─── Quaternion ───────────────────────────────────────────────────────────
void test_quat() {
    SECTION("Quatf");

    Quatf I = Quatf::identity();
    CHECK(std::abs(I.w - 1.f) < 1e-5f);

    // 90° rotation around Y
    Quatf ry = Quatf::from_axis_angle({0.f, 1.f, 0.f}, f32(PI) * 0.5f);
    Vec3f x{1.f, 0.f, 0.f};
    Vec3f rx = ry * x;
    CHECKF(rx.x,  0.f);
    CHECKF(rx.y,  0.f);
    CHECKF(rx.z, -1.f);

    // Normalize
    Quatf q{1.f, 1.f, 1.f, 1.f};
    Quatf qn = q.normalized();
    f32 len = std::sqrt(qn.x*qn.x + qn.y*qn.y + qn.z*qn.z + qn.w*qn.w);
    CHECKF(len, 1.f);

    // Slerp identity -> identity = identity
    Quatf s = Quatf::slerp(I, I, 0.5f);
    CHECKF(s.w, 1.f);
}

// ─── Constants ────────────────────────────────────────────────────────────
void test_constants() {
    SECTION("constants");

    CHECKF(f64(PI),      3.14159265358979);
    CHECKF(f64(TWO_PI),  6.28318530717959);
    CHECKF(f64(INV_PI),  0.31830988618379);
    CHECKF(f64(SQRT2),   1.41421356237310);
    CHECKF(f64(E),       2.71828182845905);
    CHECKF(f64(PHI),     1.61803398874989);
}

// ─── Entry point ──────────────────────────────────────────────────────────
int main() {
    std::printf("qs_math_core — build verification test\n");
    std::printf("========================================\n");

    test_scalars();
    test_vec2();
    test_vec3();
    test_vec4();
    test_mat3();
    test_mat4();
    test_quat();
    test_constants();

    std::printf("\n========================================\n");
    std::printf("  passed: %d\n", s_pass);
    std::printf("  failed: %d\n", s_fail);
    std::printf("========================================\n");

    return s_fail > 0 ? 1 : 0;
}