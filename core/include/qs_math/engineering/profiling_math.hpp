#pragma once
// qs_math/profiling_math.hpp — Stats collection, histograms, moving averages, percentiles (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <array>
#include <chrono>

namespace qs {

// ─── Running statistics (Welford) ────────────────────────────────────────────

struct RunningStats {
    u64   n{0};
    f64   mean{0}, M2{0};
    f64   min_val{ 1e300}, max_val{-1e300};

    void push(f64 x) {
        ++n;
        f64 delta = x - mean;
        mean += delta / f64(n);
        f64 delta2 = x - mean;
        M2 += delta * delta2;
        min_val = std::min(min_val, x);
        max_val = std::max(max_val, x);
    }

    [[nodiscard]] f64 variance()   const { return n>1 ? M2/f64(n-1) : 0.; }
    [[nodiscard]] f64 stddev()     const { return std::sqrt(variance()); }
    [[nodiscard]] f64 get_mean()   const { return mean; }
    [[nodiscard]] f64 get_min()    const { return min_val; }
    [[nodiscard]] f64 get_max()    const { return max_val; }
    [[nodiscard]] f64 range()      const { return max_val-min_val; }
    [[nodiscard]] u64 count()      const { return n; }
    [[nodiscard]] f64 cv()         const { return mean>1e-12 ? stddev()/mean : 0.; }

    void reset() { *this = {}; }
};

// ─── Exponential moving average ───────────────────────────────────────────────

struct EMA {
    f32 value{0.f};
    f32 alpha{0.1f};  // smoothing factor (0=no update, 1=no history)
    bool initialized{false};

    void push(f32 x) {
        if (!initialized) { value=x; initialized=true; return; }
        value = value*(1.f-alpha) + x*alpha;
    }

    [[nodiscard]] f32 get() const { return value; }

    [[nodiscard]] static f32 alpha_from_halflife(f32 halflife_samples) {
        return 1.f - std::exp(-0.6931472f / halflife_samples);
    }
};

// ─── Ring buffer for timing ───────────────────────────────────────────────────

template<u32 N=128>
struct RingBuffer {
    std::array<f32,N> data{};
    u32 pos{0};
    u32 count{0};

    void push(f32 v) {
        data[pos] = v;
        pos = (pos+1)%N;
        if (count<N) ++count;
    }

    [[nodiscard]] f32 avg() const {
        if (!count) return 0.f;
        f32 s=0.f; for (u32 i=0;i<count;++i) s+=data[i];
        return s/f32(count);
    }

    [[nodiscard]] f32 latest() const {
        return data[(pos+N-1)%N];
    }

    [[nodiscard]] f32 percentile(f32 p) const {
        if (!count) return 0.f;
        std::array<f32,N> sorted;
        for (u32 i=0;i<count;++i) sorted[i]=data[i];
        std::sort(sorted.begin(), sorted.begin()+count);
        u32 idx = u32(p/100.f*f32(count-1));
        return sorted[idx];
    }
};

// ─── Histogram ───────────────────────────────────────────────────────────────

struct Histogram {
    std::vector<u64> bins;
    f64 lo{}, hi{};
    u64 total{0}, underflow{0}, overflow{0};

    void init(f64 lo_, f64 hi_, u32 num_bins) {
        lo=lo_; hi=hi_; bins.assign(num_bins,0);
    }

    void push(f64 x) {
        ++total;
        if (x < lo) { ++underflow; return; }
        if (x >= hi) { ++overflow; return; }
        u32 b = u32((x-lo)/(hi-lo)*f64(bins.size()));
        b = std::min(b, u32(bins.size()-1));
        ++bins[b];
    }

    [[nodiscard]] f64 bin_center(u32 i) const {
        return lo + (f64(i)+0.5)*(hi-lo)/f64(bins.size());
    }

    [[nodiscard]] f64 mode() const {
        u32 best=0;
        for (u32 i=1;i<bins.size();++i) if (bins[i]>bins[best]) best=i;
        return bin_center(best);
    }

    [[nodiscard]] f64 percentile(f64 p) const {
        f64 target = f64(total)*p/100.;
        f64 cum = 0.;
        for (u32 i=0;i<bins.size();++i) {
            cum += f64(bins[i]);
            if (cum >= target) return bin_center(i);
        }
        return hi;
    }

    void reset() { std::fill(bins.begin(),bins.end(),0); total=underflow=overflow=0; }
};

// ─── Frame timer ─────────────────────────────────────────────────────────────

struct FrameTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TP    = std::chrono::time_point<Clock>;

    TP    frame_start{};
    f32   last_dt{0.f};
    EMA   smooth_dt{};
    RingBuffer<128> history;

    void begin_frame() { frame_start = Clock::now(); smooth_dt.alpha=0.05f; }

    f32 end_frame() {
        auto now = Clock::now();
        last_dt = std::chrono::duration<f32>(now - frame_start).count();
        smooth_dt.push(last_dt);
        history.push(last_dt);
        return last_dt;
    }

    [[nodiscard]] f32 fps() const { return smooth_dt.value > 0.f ? 1.f/smooth_dt.value : 0.f; }
    [[nodiscard]] f32 ms()  const { return smooth_dt.value * 1000.f; }
    [[nodiscard]] f32 p95_ms() const { return history.percentile(95.f)*1000.f; }
};

// ─── GPU timing helpers ───────────────────────────────────────────────────────

struct GPUTimingStats {
    RunningStats shadow_pass, geometry_pass, lighting_pass,
                 post_process, ui_pass, total;

    void reset_all() {
        shadow_pass.reset(); geometry_pass.reset(); lighting_pass.reset();
        post_process.reset(); ui_pass.reset(); total.reset();
    }
};

// ─── Sampling profiler math ───────────────────────────────────────────────────

// Confidence interval for mean (t-distribution, n>30 ~ normal approx)
[[nodiscard]] inline std::pair<f64,f64> confidence_interval_95(const RunningStats& s) {
    if (s.n < 2) return {s.mean, s.mean};
    f64 se = s.stddev() / std::sqrt(f64(s.n));
    f64 z = 1.96; // 95% CI
    return {s.mean - z*se, s.mean + z*se};
}

// Regression slope via least squares (perf trend over time)
[[nodiscard]] inline f64 perf_trend_slope(std::span<const f32> samples) {
    u32 N = u32(samples.size());
    if (N < 2) return 0.;
    f64 sx=0,sy=0,sxy=0,sxx=0;
    for (u32 i=0;i<N;++i) {
        sx+=f64(i); sy+=f64(samples[i]);
        sxy+=f64(i)*f64(samples[i]); sxx+=f64(i)*f64(i);
    }
    f64 d = f64(N)*sxx-sx*sx;
    return d>0. ? (f64(N)*sxy-sx*sy)/d : 0.;
}

} // namespace qs
