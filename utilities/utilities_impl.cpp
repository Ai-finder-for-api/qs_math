
// qs_math/utilities/utilities_impl.cpp (C++20)
#include "include/qs_math_utilities.hpp"
#include <cmath>
#include <algorithm>
namespace qs::math::util {

void prefix_sum(const f32* in, f32* out, i32 n) noexcept {
    if(n<=0)return; out[0]=in[0];
    for(i32 i=1;i<n;++i)out[i]=out[i-1]+in[i];
}
void exclusive_scan(const f32* in, f32* out, i32 n, f32 init) noexcept {
    if(n<=0)return; out[0]=init;
    for(i32 i=1;i<n;++i)out[i]=out[i-1]+in[i-1];
}
void RunningStats::push(f64 x) noexcept {
    ++count; const f64 delta=x-mean; mean+=delta/count;
    M2+=delta*(x-mean);
    min_v=std::min(min_v,x); max_v=std::max(max_v,x);
}
f64 PiecewiseLinear::eval(f64 x) const noexcept {
    if(xs.empty())return 0;
    if(x<=xs.front())return ys.front();
    if(x>=xs.back())return ys.back();
    auto it=std::lower_bound(xs.begin(),xs.end(),x);
    i32 i=(i32)(it-xs.begin());
    const f64 t=(x-xs[i-1])/(xs[i]-xs[i-1]);
    return ys[i-1]+t*(ys[i]-ys[i-1]);
}
f64 PiecewiseLinear::inv_eval(f64 y) const noexcept {
    if(ys.empty())return 0;
    if(y<=ys.front())return xs.front();
    if(y>=ys.back())return xs.back();
    auto it=std::lower_bound(ys.begin(),ys.end(),y);
    i32 i=(i32)(it-ys.begin());
    const f64 t=(y-ys[i-1])/(ys[i]-ys[i-1]);
    return xs[i-1]+t*(xs[i]-xs[i-1]);
}
// Explicit template instantiations
template struct SinTable<256>;
template struct SinTable<512>;
template struct SinTable<1024>;
template struct SinTable<4096>;
template struct Factorial<0>;
template struct Factorial<1>;
template struct Factorial<5>;
template struct Factorial<10>;
template struct Factorial<12>;
template struct Binomial<4,2>;
template struct Binomial<6,3>;
template struct Binomial<10,5>;
// Additional utility implementations
[[nodiscard]] std::string format_float(f64 v, i32 precision=6) noexcept {
    // Simple float formatter
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
    return {buf};
}
[[nodiscard]] f64 horner_eval(const f64* coeffs, i32 n, f64 x) noexcept {
    if(n<=0)return 0;
    f64 r=coeffs[n-1];
    for(i32 i=n-2;i>=0;--i)r=r*x+coeffs[i];
    return r;
}
[[nodiscard]] f64 compensated_sum(const f64* data, i32 n) noexcept {
    KahanSum s;
    for(i32 i=0;i<n;++i)s.add(data[i]);
    return s.get();
}
[[nodiscard]] f64 log_sum_exp(const f64* data, i32 n) noexcept {
    if(n<=0)return 0;
    f64 mx=*std::max_element(data,data+n);
    f64 s=0;for(i32 i=0;i<n;++i)s+=std::exp(data[i]-mx);
    return mx+std::log(s);
}
} // namespace qs::math::util
