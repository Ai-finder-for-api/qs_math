// qs_math/tests/test_numerical.cpp
#include "qs_math/numerical.hpp"
#include "qs_math/testing_utils.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>
using namespace qs::math;
using namespace qs::math::numerical;
using namespace qs::math::test;

static void test_bisection() {
    // f(x) = x^2 - 2, root = sqrt(2)
    auto f=[](f64 x){ return x*x-2.0; };
    auto r=bisection(f, 1.0, 2.0);
    assert(r.converged);
    assert(approx_eq(r.root,1.41421356,1e-8));
}
static void test_newton() {
    auto f =[](f64 x){ return x*x-2.0; };
    auto df=[](f64 x){ return 2.0*x; };
    auto r=newton(f,df,1.5);
    assert(r.converged);
    assert(approx_eq(r.root,1.41421356,1e-9));
}
static void test_derivative() {
    auto f=[](f64 x){ return x*x*x; }; // d/dx = 3x^2
    f64 d=derivative(f,2.0);
    assert(approx_eq(d,12.0,1e-6));
}
static void test_integrate_simpson() {
    // integral of x^2 from 0 to 1 = 1/3
    auto f=[](f64 x){ return x*x; };
    f64 v=integrate_simpson(f,0.0,1.0,100);
    assert(approx_eq(v,1.0/3.0,1e-8));
}
static void test_integrate_gauss() {
    auto f=[](f64 x){ return x*x; };
    f64 v=integrate_gauss5(f,0.0,1.0);
    assert(approx_eq(v,1.0/3.0,1e-10));
}
int main() {
    test_bisection(); printf("bisection OK\n");
    test_newton(); printf("newton OK\n");
    test_derivative(); printf("derivative OK\n");
    test_integrate_simpson(); printf("integrate_simpson OK\n");
    test_integrate_gauss(); printf("integrate_gauss OK\n");
    printf("=== test_numerical PASSED ===\n");
    return 0;
}
