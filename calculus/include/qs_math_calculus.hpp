#pragma once
// qs_math/calculus/include/qs_math_calculus.hpp (C++20)
#include <qs_math/arithmetic/scalar_types.hpp>
#include <functional>
#include <cmath>
#include <vector>
#include <array>
namespace qs::math::calculus {
using namespace qs::math;
[[nodiscard]] inline f64 deriv(std::function<f64(f64)> f, f64 x, f64 h=1e-5) noexcept {
    return (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h);
}
[[nodiscard]] inline f64 deriv2(std::function<f64(f64)> f, f64 x, f64 h=1e-4) noexcept {
    return (-f(x+2*h)+16*f(x+h)-30*f(x)+16*f(x-h)-f(x-2*h))/(12*h*h);
}
[[nodiscard]] f64 integrate(std::function<f64(f64)> f, f64 a, f64 b, f64 tol=1e-9) noexcept;
[[nodiscard]] inline f64 partial(std::function<f64(const f64*)> f,
    const f64* x, i32 n, i32 i, f64 h=1e-5) noexcept {
    std::vector<f64> xp(x,x+n),xm(x,x+n);
    xp[i]+=h; xm[i]-=h;
    return (f(xp.data())-f(xm.data()))/(2*h);
}
[[nodiscard]] inline f64 laplacian(std::function<f64(const f64*)> f,
    const f64* x, i32 n, f64 h=1e-4) noexcept {
    f64 sum=0, f0=f(x);
    std::vector<f64> xp(x,x+n),xm(x,x+n);
    for(i32 i=0;i<n;++i){
        xp[i]+=h; xm[i]-=h;
        sum+=(f(xp.data())-2*f0+f(xm.data()))/(h*h);
        xp[i]=xm[i]=x[i];
    }
    return sum;
}
void fourier_coeffs(std::function<f64(f64)> f, i32 N, f64* an, f64* bn) noexcept;
} // namespace qs::math::calculus
