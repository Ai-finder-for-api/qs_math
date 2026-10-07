#pragma once
// qs_math/convolution_reverb.hpp — FFT convolution, partitioned convolution, Schroeder reverb (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <array>
#include <numbers>

namespace qs {

// ─── Utilities ────────────────────────────────────────────────────────────────

using Complex = std::complex<f32>;

[[nodiscard]] inline u32 next_pow2(u32 n) {
    u32 p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ─── Cooley-Tukey FFT (in-place, power-of-2) ──────────────────────────────────

inline void fft_inplace(std::vector<Complex>& a, bool inverse) {
    u32 n = u32(a.size());
    for (u32 i=1,j=0; i<n; ++i) {
        u32 bit = n>>1;
        for (; j&bit; bit>>=1) j ^= bit;
        j ^= bit;
        if (i<j) std::swap(a[i],a[j]);
    }
    for (u32 len=2; len<=n; len<<=1) {
        f32 ang = 2.f*k_pi/f32(len)*(inverse?-1.f:1.f);
        Complex wlen{std::cos(ang), std::sin(ang)};
        for (u32 i=0; i<n; i+=len) {
            Complex w{1,0};
            for (u32 j=0; j<len/2; ++j,w*=wlen) {
                Complex u2=a[i+j], v2=w*a[i+j+len/2];
                a[i+j]=u2+v2; a[i+j+len/2]=u2-v2;
            }
        }
    }
    if (inverse) { f32 inv=1.f/f32(n); for (auto& x:a) x*=inv; }
}

// ─── Direct FFT convolution ───────────────────────────────────────────────────

inline std::vector<f32> fft_convolve(
    std::span<const f32> x, std::span<const f32> h)
{
    u32 out_len = u32(x.size()+h.size()-1);
    u32 N = next_pow2(out_len);
    std::vector<Complex> cx(N,{0,0}), ch(N,{0,0});
    for (u32 i=0;i<u32(x.size());++i) cx[i]={x[i],0};
    for (u32 i=0;i<u32(h.size());++i) ch[i]={h[i],0};
    fft_inplace(cx,false); fft_inplace(ch,false);
    for (u32 i=0;i<N;++i) cx[i]*=ch[i];
    fft_inplace(cx,true);
    std::vector<f32> out(out_len);
    for (u32 i=0;i<out_len;++i) out[i]=cx[i].real();
    return out;
}

// ─── Overlap-Add (OLA) streaming convolution ─────────────────────────────────

struct OLAConvolver {
    std::vector<f32>    ir;       // impulse response
    std::vector<Complex> H;       // FFT of zero-padded IR
    std::vector<f32>    overlap;  // tail from previous block
    u32 block_size{};
    u32 fft_size{};

    void init(std::span<const f32> impulse_response, u32 blk) {
        block_size = blk;
        ir.assign(impulse_response.begin(), impulse_response.end());
        fft_size = next_pow2(blk + u32(ir.size()) - 1);
        H.resize(fft_size);
        for (u32 i=0;i<fft_size;++i) H[i]={i<ir.size()?ir[i]:0.f,0};
        fft_inplace(H, false);
        overlap.assign(fft_size, 0.f);
    }

    // Process one block, output same size as input block
    std::vector<f32> process(std::span<const f32> input) {
        u32 n = std::min(u32(input.size()), block_size);
        std::vector<Complex> X(fft_size,{0,0});
        for (u32 i=0;i<n;++i) X[i]={input[i],0};
        fft_inplace(X,false);
        for (u32 i=0;i<fft_size;++i) X[i]*=H[i];
        fft_inplace(X,true);
        std::vector<f32> out(n);
        for (u32 i=0;i<n;++i) out[i]=X[i].real()+overlap[i];
        for (u32 i=n;i<fft_size;++i) overlap[i-n]=X[i].real()+(i<fft_size?overlap[i]:0.f);
        overlap.resize(fft_size, 0.f);
        return out;
    }
};

// ─── Partitioned convolution (low-latency) ────────────────────────────────────

struct PartitionedConvolver {
    u32 partition_size{};
    u32 num_partitions{};
    std::vector<std::vector<Complex>> H_parts;
    std::vector<std::vector<f32>> input_queue;
    std::vector<f32> accumulator;
    u32 write_pos{};

    void init(std::span<const f32> ir, u32 part_size) {
        partition_size = part_size;
        u32 fft_sz = part_size*2;
        num_partitions = (u32(ir.size())+part_size-1)/part_size;
        H_parts.resize(num_partitions, std::vector<Complex>(fft_sz));
        input_queue.resize(num_partitions, std::vector<f32>(part_size,0.f));
        accumulator.assign(fft_sz*2, 0.f);
        for (u32 p=0;p<num_partitions;++p) {
            for (u32 i=0;i<fft_sz;++i) {
                u32 idx=p*part_size+i;
                H_parts[p][i]={idx<ir.size()?ir[idx]:0.f,0};
            }
            fft_inplace(H_parts[p],false);
        }
    }

    f32 process_sample(f32 sample) {
        // Simplified single-sample partitioned processing
        u32 fft_sz = partition_size*2;
        input_queue[0][write_pos%partition_size] = sample;
        write_pos = (write_pos+1)%partition_size;
        if (write_pos == 0) {
            // New block: convolve each partition
            std::fill(accumulator.begin(),accumulator.end(),0.f);
            for (u32 p=0;p<num_partitions;++p) {
                std::vector<Complex> X(fft_sz,{0,0});
                for (u32 i=0;i<partition_size;++i)
                    X[i]={input_queue[p][i],0};
                fft_inplace(X,false);
                for (u32 i=0;i<fft_sz;++i) X[i]*=H_parts[p][i];
                fft_inplace(X,true);
                for (u32 i=0;i<fft_sz;++i) accumulator[i]+=X[i].real();
            }
        }
        return accumulator[write_pos];
    }
};

// ─── Schroeder Reverb ─────────────────────────────────────────────────────────

// Comb filter (feedback delay line with lowpass)
struct CombFilter {
    std::vector<f32> buf;
    u32 pos{};
    f32 feedback{0.84f};
    f32 damp{0.2f};
    f32 filterstore{};

    void init(u32 delay_samples) { buf.assign(delay_samples,0.f); }

    f32 process(f32 input) {
        f32 output = buf[pos];
        filterstore = output*(1.f-damp) + filterstore*damp;
        buf[pos] = input + filterstore*feedback;
        pos = (pos+1)%u32(buf.size());
        return output;
    }
};

// All-pass filter
struct AllPassFilter {
    std::vector<f32> buf;
    u32 pos{};
    f32 feedback{0.5f};

    void init(u32 delay_samples) { buf.assign(delay_samples,0.f); }

    f32 process(f32 input) {
        f32 bufout = buf[pos];
        f32 output = -input + bufout;
        buf[pos] = input + bufout*feedback;
        pos = (pos+1)%u32(buf.size());
        return output;
    }
};

struct SchroederReverb {
    static constexpr u32 NUM_COMBS=8, NUM_ALLPASS=4;
    CombFilter  combs[NUM_COMBS];
    AllPassFilter allpass[NUM_ALLPASS];
    f32 wet{0.3f}, dry{0.7f}, room_size{0.5f};

    // Standard Freeverb delay lengths at 44100 Hz
    static constexpr u32 COMB_DELAYS[NUM_COMBS]   = {1116,1188,1277,1356,1422,1491,1557,1617};
    static constexpr u32 AP_DELAYS[NUM_ALLPASS]    = {556, 441, 341, 225};

    void init(u32 sample_rate=44100) {
        f32 scale = f32(sample_rate)/44100.f;
        for (u32 i=0;i<NUM_COMBS;++i) {
            combs[i].init(u32(COMB_DELAYS[i]*scale));
            combs[i].feedback = 0.84f + room_size*0.15f;
        }
        for (u32 i=0;i<NUM_ALLPASS;++i)
            allpass[i].init(u32(AP_DELAYS[i]*scale));
    }

    f32 process(f32 input) {
        f32 out = 0.f;
        f32 in_scaled = input * 0.015f;
        for (auto& c : combs) out += c.process(in_scaled);
        for (auto& ap : allpass) out = ap.process(out);
        return input*dry + out*wet;
    }
};

// ─── Room impulse response synthesis ─────────────────────────────────────────

struct RIRParams {
    f32 room_x{5.f}, room_y{3.f}, room_z{4.f}; // meters
    f32 absorption{0.4f};       // 0=mirror 1=absorb
    Vec3f source{2.5f,1.5f,2.f};
    Vec3f listener{3.f,1.5f,2.f};
    u32   max_reflections{100};
    u32   sample_rate{44100};
};

// Image-source method: generate early reflection impulse response
inline std::vector<f32> synthesize_rir(const RIRParams& p) {
    f32 c = 343.f; // speed of sound m/s
    f32 max_time = f32(p.max_reflections) / (c * 2.f); // rough estimate
    u32 len = u32(max_time * f32(p.sample_rate)) + 1;
    std::vector<f32> ir(std::min(len, u32(p.sample_rate*2)), 0.f);

    // First-order image sources (6 walls)
    auto add_image = [&](Vec3f img, i32 order) {
        f32 dist = std::sqrt((img.x-p.listener.x)*(img.x-p.listener.x)+
                             (img.y-p.listener.y)*(img.y-p.listener.y)+
                             (img.z-p.listener.z)*(img.z-p.listener.z));
        if (dist < 1e-4f) return;
        f32 delay = dist/c;
        u32 sample = u32(delay * f32(p.sample_rate));
        if (sample < ir.size()) {
            f32 amp = std::pow(1.f-p.absorption, f32(std::abs(order))) / dist;
            ir[sample] += amp;
        }
    };

    // Direct
    add_image(p.source, 0);
    // First-order reflections
    Vec3f s = p.source;
    add_image({-s.x, s.y, s.z}, 1);
    add_image({2*p.room_x-s.x, s.y, s.z}, 1);
    add_image({s.x, -s.y, s.z}, 1);
    add_image({s.x, 2*p.room_y-s.y, s.z}, 1);
    add_image({s.x, s.y, -s.z}, 1);
    add_image({s.x, s.y, 2*p.room_z-s.z}, 1);
    return ir;
}

} // namespace qs
