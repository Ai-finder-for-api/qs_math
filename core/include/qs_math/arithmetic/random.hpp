#pragma once
// qs_math/random.hpp — PCG, XorShift, seed utils, distributions
#include "scalar_types.hpp"
#include "constants.hpp"
#include "scalar_math.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "trig.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// PCG32 — high quality, fast, small state
// ---------------------------------------------------------------------------
struct PCG32 {
    u64 state{0x853c49e6748fea9bULL};
    u64 inc{0xda3e39cb94b95bdbULL};

    constexpr explicit PCG32(u64 seed=0x853c49e6748fea9bULL, u64 seq=1) noexcept {
        state=0; inc=(seq<<1)|1;
        next_u32(); state+=seed; next_u32();
    }

    [[nodiscard]] constexpr u32 next_u32() noexcept {
        u64 old=state;
        state=old*6364136223846793005ULL+inc;
        u32 xsh=u32(((old>>18)^old)>>27);
        u32 rot=u32(old>>59);
        return (xsh>>rot)|(xsh<<((-rot)&31));
    }

    [[nodiscard]] constexpr u64 next_u64() noexcept {
        return (u64(next_u32())<<32)|next_u32();
    }

    // [0,1)
    [[nodiscard]] constexpr f32 next_f32() noexcept {
        return f32(next_u32()>>8)*0x1.0p-24f;
    }
    [[nodiscard]] constexpr f64 next_f64() noexcept {
        return f64(next_u64()>>11)*0x1.0p-53;
    }

    // [lo, hi)
    [[nodiscard]] constexpr f32 range_f32(f32 lo, f32 hi) noexcept { return lo+(hi-lo)*next_f32(); }
    [[nodiscard]] constexpr i32 range_i32(i32 lo, i32 hi)  noexcept {
        u32 r=u32(hi-lo); return lo+i32(next_u32()%r);
    }

    // [0, n)  — unbiased
    [[nodiscard]] constexpr u32 bounded_u32(u32 n) noexcept {
        u32 threshold=(-n)%n;
        while(true){ u32 r=next_u32(); if(r>=threshold) return r%n; }
    }

    // Bool with probability p
    [[nodiscard]] constexpr bool chance(f32 p) noexcept { return next_f32()<p; }

    // Gaussian (Box-Muller)
    [[nodiscard]] f32 gaussian(f32 mean=0.f, f32 stddev=1.f) noexcept {
        f32 u1=next_f32(), u2=next_f32();
        f32 z=sqrt(-2.f*log(u1+EPS_F32))*cos(TWO_PI*u2);
        return mean+stddev*z;
    }

    // Poisson
    [[nodiscard]] i32 poisson(f32 lambda) noexcept {
        f32 L=exp(-lambda); i32 k=0; f32 p=1.f;
        do { ++k; p*=next_f32(); } while(p>L);
        return k-1;
    }

    // Weighted random — returns index
    [[nodiscard]] i32 weighted(const f32* weights, i32 count) noexcept {
        f32 total=0.f;
        for(i32 i=0;i<count;++i) total+=weights[i];
        f32 r=next_f32()*total;
        for(i32 i=0;i<count;++i){ r-=weights[i]; if(r<=0.f) return i; }
        return count-1;
    }

    // Shuffle
    template<typename T>
    void shuffle(T* arr, i32 n) noexcept {
        for(i32 i=n-1;i>0;--i){
            i32 j=i32(bounded_u32(u32(i+1)));
            T tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp;
        }
    }

    // Unit vector on sphere
    [[nodiscard]] Vec3f unit_sphere() noexcept {
        f32 theta=next_f32()*TWO_PI, z=next_f32()*2.f-1.f;
        f32 r=sqrt(max(0.f,1.f-z*z));
        return {r*cos(theta), r*sin(theta), z};
    }
    // Random point in unit disk
    [[nodiscard]] Vec2f unit_disk() noexcept {
        while(true){
            Vec2f p{range_f32(-1.f,1.f),range_f32(-1.f,1.f)};
            if(p.length_sq()<=1.f) return p;
        }
    }
    // Random point in hemisphere oriented along +Y
    [[nodiscard]] Vec3f hemisphere() noexcept {
        Vec3f p=unit_sphere();
        return p.y<0.f?Vec3f{p.x,-p.y,p.z}:p;
    }
};

// ---------------------------------------------------------------------------
// XorShift64 — ultra-fast, decent quality
// ---------------------------------------------------------------------------
struct XorShift64 {
    u64 state{0xdeadbeefcafe1234ULL};
    constexpr explicit XorShift64(u64 seed) noexcept : state(seed?seed:0xdeadbeefcafe1234ULL) {}
    [[nodiscard]] constexpr u64 next() noexcept {
        state^=state<<13; state^=state>>7; state^=state<<17; return state;
    }
    [[nodiscard]] constexpr f32 next_f32() noexcept { return f32(next()>>40)*0x1.0p-24f; }
    [[nodiscard]] constexpr f32 range_f32(f32 lo, f32 hi) noexcept { return lo+(hi-lo)*next_f32(); }
};

// ---------------------------------------------------------------------------
// XorShift128+ — 128-bit state, very fast
// ---------------------------------------------------------------------------
struct XorShift128 {
    u64 s0{0x180ec6d33cfd0abaULL}, s1{0xd5a61266f0c9392cULL};
    constexpr XorShift128(u64 a, u64 b) noexcept : s0(a), s1(b) {}
    [[nodiscard]] constexpr u64 next() noexcept {
        u64 t=s0, s=s1; s0=s;
        t^=t<<23; t^=t>>18; t^=s^(s>>5); s1=t;
        return t+s;
    }
    [[nodiscard]] constexpr f32 next_f32() noexcept { return f32(next()>>40)*0x1.0p-24f; }
};

// ---------------------------------------------------------------------------
// Seed utilities
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 hash_seed(u64 seed) noexcept {
    seed^=seed>>33; seed*=0xff51afd7ed558ccdULL;
    seed^=seed>>33; seed*=0xc4ceb9fe1a85ec53ULL;
    seed^=seed>>33; return seed;
}
[[nodiscard]] constexpr u64 combine_seeds(u64 a, u64 b) noexcept {
    return a^(b*0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2));
}

// Global thread-local PCG (seeded from time/address in impl)
// Note: actual seeding done in .cpp; here we declare interface
extern thread_local PCG32 g_rng;

// Convenience wrappers
[[nodiscard]] inline f32 random_f32()                    noexcept { return g_rng.next_f32(); }
[[nodiscard]] inline f32 random_f32(f32 lo, f32 hi)     noexcept { return g_rng.range_f32(lo,hi); }
[[nodiscard]] inline i32 random_i32(i32 lo, i32 hi)     noexcept { return g_rng.range_i32(lo,hi); }
[[nodiscard]] inline bool random_bool(f32 p=0.5f)        noexcept { return g_rng.chance(p); }
[[nodiscard]] inline f32 random_gaussian(f32 m, f32 s)  noexcept { return g_rng.gaussian(m,s); }
[[nodiscard]] inline Vec3f random_unit_sphere()          noexcept { return g_rng.unit_sphere(); }
[[nodiscard]] inline Vec2f random_unit_disk()            noexcept { return g_rng.unit_disk(); }

} // namespace qs::math
