// qs_math/tests/test_noise.cpp
#include "qs_math/noise.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>
using namespace qs::math;
using namespace qs::math::test;

static void test_perlin2_range() {
    f32 mn=1e9f,mx=-1e9f;
    for(i32 x=0;x<100;++x) for(i32 y=0;y<100;++y) {
        f32 v=perlin2(x*0.1f,y*0.1f);
        mn=min(mn,v); mx=max(mx,v);
    }
    assert(mn>=-1.1f && mx<=1.1f);
}
static void test_perlin3_range() {
    f32 mn=1e9f,mx=-1e9f;
    for(i32 i=0;i<1000;++i){
        f32 v=perlin3(i*0.1f,i*0.07f,i*0.13f);
        mn=min(mn,v); mx=max(mx,v);
    }
    assert(mn>=-1.1f && mx<=1.1f);
}
static void test_simplex2_range() {
    f32 mn=1e9f,mx=-1e9f;
    for(i32 i=0;i<1000;++i){
        f32 v=simplex2(i*0.1f,i*0.077f);
        mn=min(mn,v); mx=max(mx,v);
    }
    assert(mn>=-1.1f && mx<=1.1f);
}
static void test_value_noise() {
    f32 v=value2(1.5f,2.7f);
    assert(v>=0.f && v<=1.f);
    v=value3(1.1f,2.2f,3.3f);
    assert(v>=0.f && v<=1.f);
}
static void test_fbm() {
    f32 v=fbm2(0.5f,0.5f);
    assert(v>-2.f && v<2.f);
}
static void test_cellular() {
    auto r=cellular2(1.5f,2.3f);
    assert(r.f1>=0.f && r.f2>=r.f1);
}
int main() {
    test_perlin2_range();  printf("perlin2 range OK\n");
    test_perlin3_range();  printf("perlin3 range OK\n");
    test_simplex2_range(); printf("simplex2 range OK\n");
    test_value_noise();    printf("value noise OK\n");
    test_fbm();            printf("fbm OK\n");
    test_cellular();       printf("cellular OK\n");
    printf("=== test_noise PASSED ===\n");
    return 0;
}
