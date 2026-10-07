#pragma once
// qs_math/statistics.hpp — Mean, median, variance, stddev, covariance, histogram, running stats
#include "scalar_math.hpp"
#include "scalar_types.hpp"

namespace qs::math::stats {

// ---------------------------------------------------------------------------
// Basic descriptive statistics (pointer+count, no allocation)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr T mean(const T* data, i32 n) noexcept {
    T sum=T(0); for(i32 i=0;i<n;++i) sum+=data[i];
    return n>0?sum/T(n):T(0);
}

template<FloatScalar T>
[[nodiscard]] constexpr T variance(const T* data, i32 n, bool population=false) noexcept {
    if(n<=1) return T(0);
    T m=mean(data,n), sum=T(0);
    for(i32 i=0;i<n;++i) { T d=data[i]-m; sum+=d*d; }
    return sum/T(population?n:n-1);
}

template<FloatScalar T>
[[nodiscard]] inline T stddev(const T* data, i32 n, bool population=false) noexcept {
    return qs::math::sqrt(variance(data,n,population));
}

template<FloatScalar T>
[[nodiscard]] constexpr T covariance(const T* xs, const T* ys, i32 n) noexcept {
    if(n<=1) return T(0);
    T mx=mean(xs,n), my=mean(ys,n), sum=T(0);
    for(i32 i=0;i<n;++i) sum+=(xs[i]-mx)*(ys[i]-my);
    return sum/T(n-1);
}

template<FloatScalar T>
[[nodiscard]] constexpr T pearson_r(const T* xs, const T* ys, i32 n) noexcept {
    T cov=covariance(xs,ys,n);
    T sx=stddev(xs,n), sy=stddev(ys,n);
    if(sx<T(EPS_F32)||sy<T(EPS_F32)) return T(0);
    return cov/(sx*sy);
}

// min/max/range
template<FloatScalar T>
constexpr void min_max(const T* data, i32 n, T& mn, T& mx) noexcept {
    mn=mx=data[0];
    for(i32 i=1;i<n;++i){ mn=qs::math::min(mn,data[i]); mx=qs::math::max(mx,data[i]); }
}

template<FloatScalar T>
[[nodiscard]] constexpr T median(T* data, i32 n) noexcept {
    // Quickselect-based (modifies data order)
    // Simple insertion sort for small n
    for(i32 i=1;i<n;++i){
        T key=data[i]; i32 j=i-1;
        while(j>=0&&data[j]>key){data[j+1]=data[j];--j;} data[j+1]=key;
    }
    return (n%2==0)?(data[n/2-1]+data[n/2])*T(0.5):data[n/2];
}

template<FloatScalar T>
[[nodiscard]] constexpr T sum(const T* data, i32 n) noexcept {
    T s=T(0); for(i32 i=0;i<n;++i) s+=data[i]; return s;
}

// ---------------------------------------------------------------------------
// Running (online) statistics — Welford's algorithm
// ---------------------------------------------------------------------------
template<FloatScalar T=f32>
struct RunningStats {
    i64 count{0};
    T   m{T(0)};   // running mean
    T   M2{T(0)};  // running sum of squared deviations
    T   mn{T(0)},mx{T(0)};

    constexpr void reset() noexcept { count=0; m=M2=mn=mx=T(0); }

    constexpr void push(T x) noexcept {
        ++count;
        if(count==1){ mn=mx=x; }
        else { mn=qs::math::min(mn,x); mx=qs::math::max(mx,x); }
        T delta=x-m; m+=delta/T(count);
        T delta2=x-m; M2+=delta*delta2;
    }

    [[nodiscard]] constexpr T mean()     const noexcept { return m; }
    [[nodiscard]] constexpr T variance() const noexcept { return count<2?T(0):M2/T(count-1); }
    [[nodiscard]] inline     T stddev()  const noexcept { return qs::math::sqrt(variance()); }
    [[nodiscard]] constexpr T minimum()  const noexcept { return mn; }
    [[nodiscard]] constexpr T maximum()  const noexcept { return mx; }
    [[nodiscard]] constexpr T range()    const noexcept { return mx-mn; }
    [[nodiscard]] constexpr i64 n()      const noexcept { return count; }
};

// ---------------------------------------------------------------------------
// Histogram — fixed-size bucket array
// ---------------------------------------------------------------------------
template<FloatScalar T, i32 N>
struct Histogram {
    static_assert(N>0, "N must be positive");
    T   lo{T(0)}, hi{T(1)};
    u64 buckets[N]{};
    u64 total{0};
    u64 underflow{0}, overflow{0};

    constexpr Histogram() noexcept = default;
    constexpr Histogram(T lo, T hi) noexcept : lo(lo), hi(hi) {}

    constexpr void reset() noexcept {
        for(auto& b:buckets) b=0; total=underflow=overflow=0;
    }

    constexpr void push(T x) noexcept {
        ++total;
        if(x<lo){++underflow;return;} if(x>=hi){++overflow;return;}
        i32 idx=(i32)(f64(x-lo)/f64(hi-lo)*N);
        idx=clamp(idx,0,N-1);
        ++buckets[idx];
    }

    [[nodiscard]] constexpr f64 frequency(i32 i) const noexcept {
        return total>0?f64(buckets[i])/f64(total):0.0;
    }
    [[nodiscard]] constexpr T bucket_lo(i32 i) const noexcept { return lo+(hi-lo)*T(i)/T(N); }
    [[nodiscard]] constexpr T bucket_hi(i32 i) const noexcept { return lo+(hi-lo)*T(i+1)/T(N); }
    [[nodiscard]] constexpr T bucket_mid(i32 i) const noexcept { return (bucket_lo(i)+bucket_hi(i))*T(0.5); }
    [[nodiscard]] constexpr u64 mode_bucket()   const noexcept {
        u64 mx=0; i32 idx=0;
        for(i32 i=0;i<N;++i) if(buckets[i]>mx){mx=buckets[i];idx=i;}
        return idx;
    }
};

// ---------------------------------------------------------------------------
// Percentile (approximate, from sorted data)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr T percentile(const T* sorted_data, i32 n, f64 p) noexcept {
    if(n<=0) return T(0);
    f64 idx=(p/100.0)*(n-1);
    i32 lo=(i32)idx; i32 hi=lo+1;
    if(hi>=n) return sorted_data[n-1];
    f64 frac=idx-lo;
    return T(lerp(f64(sorted_data[lo]),f64(sorted_data[hi]),frac));
}

} // namespace qs::math::stats
