// qs_math/tests/test_color.cpp
#include "qs_math/color.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::test;

static void test_hsv_roundtrip() {
    ColorRGB r{0.8f,0.2f,0.5f};
    ColorHSV h=ColorHSV::from_rgb(r);
    ColorRGB back=h.to_rgb();
    assert(approx_eq(r.r,back.r,1e-4f));
    assert(approx_eq(r.g,back.g,1e-4f));
    assert(approx_eq(r.b,back.b,1e-4f));
}
static void test_hsl_roundtrip() {
    ColorRGB r{0.3f,0.7f,0.1f};
    ColorHSL h=ColorHSL::from_rgb(r);
    ColorRGB back=h.to_rgb();
    assert(approx_eq(r.r,back.r,1e-4f));
}
static void test_oklab_roundtrip() {
    ColorRGB r{0.5f,0.3f,0.8f};
    ColorRGB lin=to_linear(r);
    ColorOKLAB lab=ColorOKLAB::from_linear_rgb(lin);
    ColorRGB back=to_srgb(lab.to_linear_rgb());
    assert(approx_eq(r.r,back.r,1e-4f));
}
static void test_xyz_roundtrip() {
    ColorRGB r{0.4f,0.6f,0.2f};
    ColorRGB lin=to_linear(r);
    ColorXYZ xyz=ColorXYZ::from_linear_rgb(lin);
    ColorRGB back=xyz.to_linear_rgb();
    assert(approx_eq(lin.r,back.r,1e-4f));
}
static void test_cmyk_roundtrip() {
    ColorRGB r{0.6f,0.2f,0.8f};
    ColorCMYK c=ColorCMYK::from_rgb(r);
    ColorRGB back=c.to_rgb();
    assert(approx_eq(r.r,back.r,1e-4f));
}
static void test_pack_unpack() {
    ColorRGBA c{1.f,0.5f,0.25f,0.75f};
    u32 packed=pack_rgba(c);
    ColorRGBA back=unpack_rgba(packed);
    assert(approx_eq(c.r,back.r,1.f/255.f+1e-4f));
}
static void test_lerp_color() {
    ColorRGB a=ColorRGB::black(), b=ColorRGB::white();
    ColorRGB mid=lerp(a,b,0.5f);
    assert(approx_eq(mid.r,0.5f,1e-5f));
}
int main() {
    test_hsv_roundtrip();  printf("hsv roundtrip OK\n");
    test_hsl_roundtrip();  printf("hsl roundtrip OK\n");
    test_oklab_roundtrip();printf("oklab roundtrip OK\n");
    test_xyz_roundtrip();  printf("xyz roundtrip OK\n");
    test_cmyk_roundtrip(); printf("cmyk roundtrip OK\n");
    test_pack_unpack();    printf("pack_unpack OK\n");
    test_lerp_color();     printf("lerp_color OK\n");
    printf("=== test_color PASSED ===\n");
    return 0;
}
