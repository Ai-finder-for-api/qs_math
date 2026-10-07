#pragma once
// qs_math/benchmarks_math.hpp — Timing utilities, throughput math, FLOP counting (C++20)
#include "scalar_types.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string_view>
#include <array>
#include <cassert>

namespace qs {

// ─── High-resolution timer ────────────────────────────────────────────────────
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    using Point = std::chrono::time_point<Clock>;

    Point start_time;

    Timer() : start_time(Clock::now()) {}

    void reset() { start_time=Clock::now(); }

    [[nodiscard]] f64 elapsed_ns()  const { return f64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start_time).count()); }
    [[nodiscard]] f64 elapsed_us()  const { return elapsed_ns()*1e-3; }
    [[nodiscard]] f64 elapsed_ms()  const { return elapsed_ns()*1e-6; }
    [[nodiscard]] f64 elapsed_s()   const { return elapsed_ns()*1e-9; }
};

// ─── Benchmark result ─────────────────────────────────────────────────────────
struct BenchResult {
    std::string name;
    f64 mean_ns{0};
    f64 min_ns{0};
    f64 max_ns{0};
    f64 stddev_ns{0};
    f64 median_ns{0};
    u64 iterations{0};
    u64 flops{0};       // per iteration
    f64 gflops{0};      // computed throughput

    [[nodiscard]] std::string summary() const {
        char buf[256];
        std::snprintf(buf,sizeof(buf),
            "%-30s  mean=%6.1fns  min=%6.1fns  max=%6.1fns  sigma=%5.1fns  %s",
            name.c_str(), mean_ns, min_ns, max_ns, stddev_ns,
            gflops>0?((std::snprintf(buf+200,55,"%.2f GFLOP/s",gflops)),buf+200):"");
        return buf;
    }
};

// ─── Benchmark runner ─────────────────────────────────────────────────────────
[[nodiscard]] inline BenchResult benchmark(std::string_view name,
    const std::function<void()>& fn,
    u32 warmup=5, u32 runs=50, u64 flops_per_iter=0)
{
    // Warmup
    for(u32 i=0;i<warmup;++i) fn();
    // Timed runs
    std::vector<f64> times(runs);
    Timer t;
    for(u32 i=0;i<runs;++i) {
        t.reset(); fn(); times[i]=t.elapsed_ns();
    }
    BenchResult r;
    r.name=name;
    r.iterations=runs;
    r.flops=flops_per_iter;
    r.min_ns=*std::min_element(times.begin(),times.end());
    r.max_ns=*std::max_element(times.begin(),times.end());
    r.mean_ns=std::accumulate(times.begin(),times.end(),0.0)/runs;
    f64 var=0; for(auto v:times) var+=(v-r.mean_ns)*(v-r.mean_ns); r.stddev_ns=std::sqrt(var/runs);
    std::sort(times.begin(),times.end()); r.median_ns=times[runs/2];
    if(flops_per_iter>0) r.gflops=f64(flops_per_iter)/(r.mean_ns); // GFLOP/s = FLOP / ns
    return r;
}

// Benchmark a loop of n iterations (whole loop timed, divides by n)
[[nodiscard]] inline BenchResult benchmark_n(std::string_view name,
    const std::function<void(u32)>& fn, u32 n,
    u32 warmup=3, u32 runs=20, u64 flops_per_element=0)
{
    auto wrapped=[&]{ fn(n); };
    auto r=benchmark(name,wrapped,warmup,runs,flops_per_element*n);
    // Per-element timing
    BenchResult re=r;
    re.mean_ns/=n; re.min_ns/=n; re.max_ns/=n; re.median_ns/=n; re.stddev_ns/=n;
    re.flops=flops_per_element;
    if(flops_per_element>0) re.gflops=f64(flops_per_element)/re.mean_ns;
    return re;
}

// ─── FLOP counting helpers ────────────────────────────────────────────────────

// Basic arithmetic FLOP counts
[[nodiscard]] constexpr u64 flops_matvec(u32 rows, u32 cols) { return u64(rows)*cols*2; } // n mults + n adds
[[nodiscard]] constexpr u64 flops_matmul(u32 m, u32 k, u32 n) { return u64(m)*k*n*2; }
[[nodiscard]] constexpr u64 flops_dot(u32 n) { return u64(n)*2-1; }
[[nodiscard]] constexpr u64 flops_axpy(u32 n) { return u64(n)*2; }    // y = a*x + y
[[nodiscard]] constexpr u64 flops_norm(u32 n) { return u64(n)*2+1; }  // n multiplies + n-1 adds + 1 sqrt

// Roofline model: computes expected GFLOP/s given arithmetic intensity
// AI = FLOPs / bytes; bandwidth in GB/s; peak_flops in GFLOP/s
[[nodiscard]] inline f64 roofline_performance(f64 arithmetic_intensity,
    f64 memory_bandwidth_GBs, f64 peak_flops_Gs)
{
    // Roofline: min(peak_flops, bandwidth * AI)
    return std::min(peak_flops_Gs, memory_bandwidth_GBs*arithmetic_intensity);
}

// Estimate if computation is memory-bound or compute-bound
[[nodiscard]] inline bool is_memory_bound(f64 arithmetic_intensity,
    f64 memory_bandwidth_GBs, f64 peak_flops_Gs)
{
    return memory_bandwidth_GBs*arithmetic_intensity < peak_flops_Gs;
}

// ─── Throughput / bandwidth math ─────────────────────────────────────────────

// Bytes per second from operation count and time
[[nodiscard]] inline f64 throughput_GBs(u64 bytes, f64 time_ns) {
    return f64(bytes)/(time_ns); // = GB/s  (bytes/ns = GB/s)
}

// Elements per second
[[nodiscard]] inline f64 elements_per_second(u64 n, f64 time_ns) {
    return f64(n)/(time_ns*1e-9);
}

// Effective memory bandwidth from measured time
[[nodiscard]] inline f64 effective_bandwidth(u64 read_bytes, u64 write_bytes, f64 time_ns) {
    return throughput_GBs(read_bytes+write_bytes, time_ns);
}

// ─── Statistical analysis of timing data ─────────────────────────────────────
struct TimingSeries {
    std::vector<f64> samples_ns;

    void record(f64 t) { samples_ns.push_back(t); }

    [[nodiscard]] f64 mean()   const { if(samples_ns.empty())return 0; f64 s=0;for(auto v:samples_ns)s+=v;return s/samples_ns.size(); }
    [[nodiscard]] f64 stddev() const { f64 m=mean(),s=0; for(auto v:samples_ns)s+=(v-m)*(v-m); return std::sqrt(s/std::max(1ull,u64(samples_ns.size()))); }
    [[nodiscard]] f64 cv()     const { f64 m=mean(); return m>0?stddev()/m:0; } // coefficient of variation
    [[nodiscard]] f64 percentile(f32 p) const {
        if(samples_ns.empty()) return 0;
        auto sorted=samples_ns; std::sort(sorted.begin(),sorted.end());
        u32 idx=u32(p*(sorted.size()-1)/100);
        return sorted[idx];
    }
    [[nodiscard]] f64 p50() const { return percentile(50); }
    [[nodiscard]] f64 p95() const { return percentile(95); }
    [[nodiscard]] f64 p99() const { return percentile(99); }

    // Remove outliers beyond k*sigma
    [[nodiscard]] TimingSeries trimmed(f32 k=3.f) const {
        f64 m=mean(),s=stddev();
        TimingSeries r;
        for(auto v:samples_ns) if(std::abs(v-m)<k*s) r.samples_ns.push_back(v);
        return r;
    }
};

// ─── Comparative benchmarking ─────────────────────────────────────────────────
struct BenchSuite {
    std::vector<BenchResult> results;

    void add(BenchResult r) { results.push_back(std::move(r)); }

    // Normalise all results relative to the first (baseline)
    [[nodiscard]] std::vector<f64> relative_speedup() const {
        if(results.empty()) return {};
        f64 base=results[0].mean_ns;
        std::vector<f64> sp(results.size());
        for(u32 i=0;i<u32(results.size());++i) sp[i]=base/results[i].mean_ns;
        return sp;
    }

    void print_table() const {
        std::printf("%-30s %10s %10s %10s %8s\n","Name","Mean(ns)","Min(ns)","Speedup","GFLOP/s");
        std::printf("%s\n",std::string(70,'-').c_str());
        f64 base=results.empty()?1:results[0].mean_ns;
        for(auto& r:results)
            std::printf("%-30s %10.1f %10.1f %8.2fx %8.2f\n",
                r.name.c_str(),r.mean_ns,r.min_ns,base/r.mean_ns,r.gflops);
    }
};

// ─── RDTSC cycle counter (x86) ────────────────────────────────────────────────
#if defined(__x86_64__)||defined(_M_X64)||defined(__i386__)
#include <cstdint>
[[nodiscard]] inline u64 rdtsc() {
    u32 lo,hi;
    __asm__ volatile("rdtsc":"=a"(lo),"=d"(hi));
    return (u64(hi)<<32)|lo;
}
[[nodiscard]] inline u64 rdtscp(u32& aux) {
    u32 lo,hi;
    __asm__ volatile("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux));
    return (u64(hi)<<32)|lo;
}
#else
[[nodiscard]] inline u64 rdtsc() {
    return u64(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}
#endif

// ─── Cache miss estimation ────────────────────────────────────────────────────
// Estimate cache lines accessed in a strided access pattern
[[nodiscard]] constexpr u64 cache_lines_accessed(u64 n, u32 stride_bytes, u32 cache_line_bytes=64) {
    if(stride_bytes>=cache_line_bytes) return n; // each access is a new cache line
    return (n*stride_bytes+cache_line_bytes-1)/cache_line_bytes;
}

} // namespace qs
