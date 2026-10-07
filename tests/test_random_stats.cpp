// qs_math/tests/test_random_stats.cpp
#include "qs_math/random.hpp"
#include "qs_math/statistics.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
using namespace qs::math;
using namespace qs::math::stats;
using namespace qs::math::test;

static void test_pcg32() {
    PCG32 rng{42};
    f32 mn=1e9f,mx=-1e9f;
    for(i32 i=0;i<10000;++i){
        f32 v=rng.next_f32();
        assert(v>=0.f && v<1.f);
        mn=min(mn,v); mx=max(mx,v);
    }
    assert(mn<0.01f && mx>0.99f); // good coverage
}
static void test_pcg32_gaussian() {
    PCG32 rng{123};
    f32 samples[1000];
    for(auto& s:samples) s=rng.gaussian(0.f,1.f);
    RunningStats<f32> rs;
    for(auto s:samples) rs.push(s);
    assert(approx_eq(rs.mean(),0.f,0.15f));    // mean ~0
    assert(approx_eq(rs.stddev(),1.f,0.15f));  // stddev ~1
}
static void test_weighted() {
    PCG32 rng{42};
    f32 w[]={0.f,1.f,0.f}; // always pick index 1
    for(i32 i=0;i<100;++i) assert(rng.weighted(w,3)==1);
}
static void test_running_stats() {
    RunningStats<f32> rs;
    f32 data[]={1.f,2.f,3.f,4.f,5.f};
    for(f32 v:data) rs.push(v);
    assert(approx_eq(rs.mean(),3.f));
    assert(rs.minimum()==1.f && rs.maximum()==5.f);
    assert(approx_eq(rs.variance(),2.5f,1e-4f));
}
static void test_histogram() {
    Histogram<f32,10> h{0.f,1.f};
    PCG32 rng{7};
    for(i32 i=0;i<10000;++i) h.push(rng.next_f32());
    // Each bucket should have roughly 1000 samples
    for(i32 i=0;i<10;++i) assert(h.buckets[i]>500 && h.buckets[i]<1500);
}
static void test_covariance() {
    f32 xs[]={1,2,3,4,5}, ys[]={2,4,6,8,10};
    f32 cov=covariance(xs,ys,5);
    assert(approx_eq(cov,5.f,1e-3f)); // perfect correlation
    f32 r=pearson_r(xs,ys,5);
    assert(approx_eq(r,1.f,1e-3f));
}
int main() {
    test_pcg32(); printf("pcg32 OK\n");
    test_pcg32_gaussian(); printf("gaussian OK\n");
    test_weighted(); printf("weighted OK\n");
    test_running_stats(); printf("running_stats OK\n");
    test_histogram(); printf("histogram OK\n");
    test_covariance(); printf("covariance OK\n");
    printf("=== test_random_stats PASSED ===\n");
    return 0;
}
