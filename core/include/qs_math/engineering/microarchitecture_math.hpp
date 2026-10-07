#pragma once
// qs_math/microarchitecture_math.hpp — SIMD lane math, cache-line packing, prefetch patterns (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <span>
#include <numeric>
#include <bit>

namespace qs {

// ─── Cache line helpers ───────────────────────────────────────────────────────

static constexpr u32 k_cache_line_bytes = 64;
static constexpr u32 k_page_size_bytes  = 4096;

[[nodiscard]] inline u32 cache_lines_needed(u32 bytes) {
    return (bytes + k_cache_line_bytes - 1) / k_cache_line_bytes;
}

[[nodiscard]] inline u32 align_to_cache_line(u32 bytes) {
    return cache_lines_needed(bytes) * k_cache_line_bytes;
}

// Estimate cache miss rate for stride access pattern
[[nodiscard]] inline f32 stride_cache_efficiency(u32 stride_bytes, u32 cache_size_bytes=32768) {
    // If stride > cache_line, every access is a miss
    if(stride_bytes >= k_cache_line_bytes) return 0.f;
    // Elements per cache line
    f32 elements_per_line = f32(k_cache_line_bytes) / f32(stride_bytes);
    return std::min(1.f, elements_per_line / f32(k_cache_line_bytes));
}

// ─── AoS <-> SoA layout math ─────────────────────────────────────────────────

// Estimate bandwidth for AoS vs SoA access pattern
struct LayoutMetrics {
    u32 struct_size_bytes{};     // size of one element struct
    u32 accessed_field_bytes{};  // bytes we actually need per element
    u32 n_elements{};
    u32 cache_line_bytes{k_cache_line_bytes};

    [[nodiscard]] f32 aos_utilization() const {
        // AoS: load whole cache line, use only accessed_field_bytes
        f32 elems_per_line = f32(cache_line_bytes) / f32(struct_size_bytes);
        f32 useful = elems_per_line * f32(accessed_field_bytes);
        return useful / f32(cache_line_bytes);
    }

    [[nodiscard]] f32 soa_utilization() const {
        // SoA: field is contiguous, full utilization
        return std::min(1.f, f32(accessed_field_bytes) / f32(cache_line_bytes));
    }

    [[nodiscard]] f32 speedup_soa_over_aos() const {
        f32 a=aos_utilization(), s=soa_utilization();
        return a>1e-6f ? s/a : 1.f;
    }
};

// ─── Morton (Z-order) SoA reordering ─────────────────────────────────────────

// Reorder array indices by Morton code for cache-coherent 2D access
[[nodiscard]] inline std::vector<u32> morton_reorder_2d(u32 nx, u32 ny) {
    std::vector<std::pair<u64,u32>> order;
    order.reserve(nx*ny);
    for(u32 y=0;y<ny;++y)
    for(u32 x=0;x<nx;++x){
        u64 code=0;
        for(u32 bit=0;bit<16;++bit)
            code|=(u64((x>>bit)&1)<<(2*bit))|(u64((y>>bit)&1)<<(2*bit+1));
        order.push_back({code,y*nx+x});
    }
    std::sort(order.begin(),order.end());
    std::vector<u32> indices(nx*ny);
    for(u32 i=0;i<u32(order.size());++i) indices[i]=order[i].second;
    return indices;
}

// ─── SIMD-width batch helpers ─────────────────────────────────────────────────

static constexpr u32 k_simd_f32_width = 8;  // AVX: 8 floats
static constexpr u32 k_simd_f64_width = 4;  // AVX: 4 doubles

[[nodiscard]] inline u32 simd_padded_count(u32 n, u32 width=k_simd_f32_width) {
    return (n+width-1)/width*width;
}

[[nodiscard]] inline u32 simd_loop_iterations(u32 n, u32 width=k_simd_f32_width) {
    return n/width;
}

[[nodiscard]] inline u32 simd_remainder(u32 n, u32 width=k_simd_f32_width) {
    return n%width;
}

// Estimate theoretical GFLOPS for a SIMD-vectorized loop
struct SIMDPerfEstimate {
    f32 operations_per_element{};
    f32 elements_per_second{};
    f32 gflops{};
};

[[nodiscard]] inline SIMDPerfEstimate estimate_simd_perf(
    u32 n_elements,
    f32 ops_per_element,
    f32 clock_ghz=3.5f,
    u32 simd_width=k_simd_f32_width,
    f32 fma_throughput=0.5f) // cycles per FMA instruction
{
    f32 scalar_cycles = ops_per_element * fma_throughput;
    f32 simd_cycles   = scalar_cycles / f32(simd_width);
    f32 elements_per_cycle = 1.f / simd_cycles;
    f32 elements_per_second = elements_per_cycle * clock_ghz * 1e9f;
    f32 gflops = elements_per_second * ops_per_element / 1e9f;
    return {ops_per_element, elements_per_second, gflops};
}

// ─── Memory bandwidth estimation ─────────────────────────────────────────────

struct BandwidthAnalysis {
    u64 bytes_read{};
    u64 bytes_written{};
    f32 arithmetic_intensity{}; // FLOPS/byte (roofline model)

    [[nodiscard]] f32 roofline_performance(
        f32 peak_gflops, f32 peak_bandwidth_GBps) const
    {
        f32 bandwidth_limit = arithmetic_intensity * peak_bandwidth_GBps;
        return std::min(peak_gflops, bandwidth_limit);
    }

    [[nodiscard]] bool is_memory_bound(f32 peak_gflops, f32 peak_bandwidth_GBps) const {
        f32 ridge_point = peak_gflops / peak_bandwidth_GBps; // FLOP/byte at ridge
        return arithmetic_intensity < ridge_point;
    }
};

// ─── Prefetch distance calculation ───────────────────────────────────────────

[[nodiscard]] inline u32 optimal_prefetch_distance(
    f32 memory_latency_cycles, f32 cycles_per_element)
{
    // Prefetch distance = ceil(latency / cycles_per_element)
    return u32(std::ceil(memory_latency_cycles / (cycles_per_element + 1e-6f)));
}

// ─── Scatter / gather pattern cost ───────────────────────────────────────────

[[nodiscard]] inline f32 gather_efficiency(
    std::span<const u32> indices,
    u32 array_stride_bytes = 4)
{
    if(indices.empty()) return 0.f;
    u32 n=u32(indices.size());
    u32 unique_lines=0;
    std::vector<u32> seen_lines;
    for(u32 idx:indices){
        u32 line=(idx*array_stride_bytes)/k_cache_line_bytes;
        if(std::find(seen_lines.begin(),seen_lines.end(),line)==seen_lines.end()){
            seen_lines.push_back(line);
            ++unique_lines;
        }
    }
    // Efficiency = elements per cache line accessed
    return f32(n)/f32(unique_lines*k_cache_line_bytes/array_stride_bytes);
}

// ─── False sharing detection ──────────────────────────────────────────────────

struct ThreadData {
    u32 thread_id{};
    u32 element_idx{};
};

[[nodiscard]] inline bool has_false_sharing(
    std::span<const ThreadData> accesses,
    u32 element_size_bytes=4)
{
    for(u32 i=0;i<u32(accesses.size());++i)
    for(u32 j=i+1;j<u32(accesses.size());++j){
        if(accesses[i].thread_id==accesses[j].thread_id) continue;
        u32 li=accesses[i].element_idx*element_size_bytes/k_cache_line_bytes;
        u32 lj=accesses[j].element_idx*element_size_bytes/k_cache_line_bytes;
        if(li==lj) return true;
    }
    return false;
}

// ─── Tile size optimization ───────────────────────────────────────────────────

// Optimal tile size for matrix multiply (L1 cache tiling)
[[nodiscard]] inline u32 optimal_tile_size(
    u32 l1_cache_bytes=32768, u32 element_bytes=4, u32 n_matrices=3)
{
    // T^2 * n_matrices * element_bytes <= L1
    return u32(std::sqrt(f32(l1_cache_bytes) / f32(n_matrices * element_bytes)));
}

// Work estimate for tiled matrix multiply
[[nodiscard]] inline u64 tiled_matmul_cache_lines(u32 N, u32 tile) {
    u64 tiles_per_dim=(N+tile-1)/tile;
    u64 total_tiles=tiles_per_dim*tiles_per_dim*tiles_per_dim;
    u64 lines_per_tile_load=cache_lines_needed(tile*tile*4); // f32
    return total_tiles*lines_per_tile_load*3; // A,B,C
}

// ─── Compiler hint abstractions ───────────────────────────────────────────────

// Arithmetic for likely / unlikely branch prediction
[[nodiscard]] inline f32 branch_prediction_penalty(
    f32 misprediction_rate, f32 pipeline_depth=15.f)
{
    return misprediction_rate * pipeline_depth; // cycles penalty per branch
}

[[nodiscard]] inline f32 branch_friendly_cpi(
    f32 ipc_no_branch, f32 branch_freq, f32 mispred_rate, f32 pipeline=15.f)
{
    f32 penalty_per_insn = branch_freq * mispred_rate * pipeline;
    return 1.f/ipc_no_branch + penalty_per_insn;
}

} // namespace qs
