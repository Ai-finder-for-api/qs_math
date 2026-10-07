#pragma once
// random_ext.hpp — Extended random: distributions, stratified sampling,
//                  importance sampling, Monte Carlo helpers, PCG64, LCG,
//                  quasi-Monte Carlo, sample warping, rejection sampling
#include "random.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "trig.hpp"
#include "constants.hpp"
#include "scalar_math.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// PCG64 — 64-bit output, 128-bit state
// ---------------------------------------------------------------------------
struct PCG64 {
    u64 state[2]{0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};
    u64 inc[2]{1, 1};

    explicit PCG64(u64 seed=42, u64 seq=1) noexcept {
        state[0]=state[1]=0;
        inc[0]=(seq<<1)|1; inc[1]=((seq+1)<<1)|1;
        next_u64(); state[0]+=seed; next_u64();
        next_u64(); state[1]+=seed; next_u64();
    }

    [[nodiscard]] u64 next_u64() noexcept {
        // Xorshift128+
        u64 s1=state[0], s0=state[1];
        state[0]=s0;
        s1^=s1<<23; s1^=s1>>18; s1^=s0; s1^=s0>>5;
        state[1]=s1;
        return s1+s0;
    }
    [[nodiscard]] f64 next_f64() noexcept { return f64(next_u64()>>11)*0x1.0p-53; }
    [[nodiscard]] f32 next_f32() noexcept { return f32(next_u64()>>40)*0x1.0p-24f; }
    [[nodiscard]] f64 range_f64(f64 lo,f64 hi) noexcept { return lo+(hi-lo)*next_f64(); }
};

// ---------------------------------------------------------------------------
// LCG — simple, fast, deterministic
// ---------------------------------------------------------------------------
struct LCG32 {
    u32 state{1};
    constexpr explicit LCG32(u32 seed) noexcept : state(seed) {}
    [[nodiscard]] constexpr u32 next() noexcept {
        state=state*1664525+1013904223; return state;
    }
    [[nodiscard]] constexpr f32 next_f32() noexcept { return f32(next()>>8)*0x1.0p-24f; }
};

// ---------------------------------------------------------------------------
// Stratified sampling
// ---------------------------------------------------------------------------
// Generate N stratified 1D samples in [0,1)
inline void stratified_1d(f32* out, i32 n, PCG32& rng) noexcept {
    f32 inv=1.f/n;
    for(i32 i=0;i<n;++i) out[i]=(i+rng.next_f32())*inv;
}

// Generate N×M stratified 2D samples
inline void stratified_2d(Vec2f* out, i32 nx, i32 ny, PCG32& rng) noexcept {
    f32 dx=1.f/nx, dy=1.f/ny;
    i32 idx=0;
    for(i32 j=0;j<ny;++j) for(i32 i=0;i<nx;++i)
        out[idx++]={(i+rng.next_f32())*dx,(j+rng.next_f32())*dy};
    // Shuffle
    for(i32 i=nx*ny-1;i>0;--i){
        i32 j=i32(rng.bounded_u32(u32(i+1)));
        Vec2f tmp=out[i]; out[i]=out[j]; out[j]=tmp;
    }
}

// Latin Hypercube Sampling 2D
inline void latin_hypercube_2d(Vec2f* out, i32 n, PCG32& rng) noexcept {
    f32 inv=1.f/n;
    for(i32 i=0;i<n;++i) out[i]={(i+rng.next_f32())*inv,(i+rng.next_f32())*inv};
    // Shuffle y coordinates
    for(i32 i=n-1;i>0;--i){
        i32 j=i32(rng.bounded_u32(u32(i+1)));
        f32 tmp=out[i].y; out[i].y=out[j].y; out[j].y=tmp;
    }
}

// ---------------------------------------------------------------------------
// Sample warping (for physically-based rendering)
// ---------------------------------------------------------------------------
// Uniform disk warp
[[nodiscard]] inline Vec2f sample_disk_concentric(Vec2f u) noexcept {
    // Shirley-Chiu mapping
    Vec2f offset=2.f*u-Vec2f{1.f};
    if(offset.x==0.f&&offset.y==0.f) return {0.f};
    f32 r,theta;
    if(abs(offset.x)>abs(offset.y)){
        r=offset.x; theta=QUARTER_PI*(offset.y/offset.x);
    } else {
        r=offset.y; theta=HALF_PI-QUARTER_PI*(offset.x/offset.y);
    }
    return Vec2f{cos(theta),sin(theta)}*r;
}

// Cosine hemisphere sample
[[nodiscard]] inline Vec3f sample_cosine_hemisphere(Vec2f u) noexcept {
    Vec2f d=sample_disk_concentric(u);
    f32 z=sqrt(max(0.f,1.f-d.x*d.x-d.y*d.y));
    return {d.x,d.y,z};
}

// Uniform hemisphere
[[nodiscard]] inline Vec3f sample_uniform_hemisphere(Vec2f u) noexcept {
    f32 cos_theta=u.x, sin_theta=sqrt(max(0.f,1.f-u.x*u.x));
    f32 phi=TWO_PI*u.y;
    return {sin_theta*cos(phi),sin_theta*sin(phi),cos_theta};
}

// Uniform sphere
[[nodiscard]] inline Vec3f sample_uniform_sphere(Vec2f u) noexcept {
    f32 cos_theta=1.f-2.f*u.x, sin_theta=sqrt(max(0.f,1.f-cos_theta*cos_theta));
    f32 phi=TWO_PI*u.y;
    return {sin_theta*cos(phi),sin_theta*sin(phi),cos_theta};
}

// GGX VNDF sample
[[nodiscard]] inline Vec3f sample_ggx_vndf(Vec3f Ve, f32 roughness, Vec2f u) noexcept {
    Vec3f Vh={roughness*Ve.x,roughness*Ve.y,Ve.z};
    Vh=Vh.normalized();
    f32 lensq=Vh.x*Vh.x+Vh.y*Vh.y;
    Vec3f T1=lensq>EPS_LENGTH?Vec3f{-Vh.y,Vh.x,0.f}*(1.f/sqrt(lensq)):Vec3f{1.f,0.f,0.f};
    Vec3f T2=cross(Vh,T1);
    f32 r=sqrt(u.x); f32 phi=TWO_PI*u.y;
    f32 t1=r*cos(phi), t2=r*sin(phi);
    f32 s=0.5f*(1.f+Vh.z); t2=(1.f-s)*sqrt(1.f-t1*t1)+s*t2;
    Vec3f Nh=T1*t1+T2*t2+Vh*sqrt(max(0.f,1.f-t1*t1-t2*t2));
    return Vec3f{roughness*Nh.x,roughness*Nh.y,max(0.f,Nh.z)}.normalized();
}

// Beckmann distribution sample
[[nodiscard]] inline Vec3f sample_beckmann(f32 roughness, Vec2f u) noexcept {
    f32 phi=TWO_PI*u.y;
    f32 tan2=-(roughness*roughness)*log(max(1e-7f,u.x));
    f32 cos_theta=1.f/sqrt(1.f+tan2);
    f32 sin_theta=sqrt(max(0.f,1.f-cos_theta*cos_theta));
    return {sin_theta*cos(phi),sin_theta*sin(phi),cos_theta};
}

// Triangle uniform sampling
[[nodiscard]] inline Vec3f sample_triangle(Vec3f a,Vec3f b,Vec3f c,Vec2f u) noexcept {
    f32 su0=sqrt(u.x);
    f32 ba=1.f-su0, bb=u.y*su0, bc=1.f-ba-bb;
    return a*ba+b*bb+c*bc;
}

// ---------------------------------------------------------------------------
// Probability distributions
// ---------------------------------------------------------------------------
struct NormalDist {
    f32 mean{0.f}, stddev{1.f};
    [[nodiscard]] f32 pdf(f32 x) const noexcept {
        f32 d=(x-mean)/stddev;
        return exp(-0.5f*d*d)/(stddev*SQRT_PI*INV_SQRT2*2.f);
    }
    [[nodiscard]] f32 cdf(f32 x) const noexcept {
        return 0.5f*(1.f+erf((x-mean)/(stddev*SQRT2)));
    }
    [[nodiscard]] f32 sample(PCG32& rng) const noexcept {
        return mean+stddev*rng.gaussian();
    }
};

struct ExponentialDist {
    f32 lambda{1.f};
    [[nodiscard]] constexpr f32 pdf(f32 x) const noexcept { return x>=0.f?lambda*exp(-lambda*x):0.f; }
    [[nodiscard]] constexpr f32 cdf(f32 x) const noexcept { return x>=0.f?1.f-exp(-lambda*x):0.f; }
    [[nodiscard]] f32 sample(PCG32& rng) const noexcept { return -log(max(1e-7f,rng.next_f32()))/lambda; }
};

struct UniformDist {
    f32 lo{0.f}, hi{1.f};
    [[nodiscard]] constexpr f32 pdf(f32 x) const noexcept {
        return (x>=lo&&x<=hi)?1.f/(hi-lo):0.f;
    }
    [[nodiscard]] f32 sample(PCG32& rng) const noexcept { return rng.range_f32(lo,hi); }
};

struct BetaDist {
    f32 alpha{1.f}, beta{1.f};
    [[nodiscard]] f32 sample(PCG32& rng) const noexcept {
        // Johnk's method
        while(true){
            f32 u=pow(rng.next_f32(),1.f/alpha);
            f32 v=pow(rng.next_f32(),1.f/beta);
            if(u+v<=1.f) return u/(u+v);
        }
    }
};

// ---------------------------------------------------------------------------
// Importance sampling helpers
// ---------------------------------------------------------------------------
struct ImportanceSample {
    Vec3f direction;
    f32   pdf{1.f};
    f32   weight{1.f}; // sample weight = f/pdf
};

[[nodiscard]] inline ImportanceSample cosine_hemisphere_sample_is(Vec2f u) noexcept {
    Vec3f d=sample_cosine_hemisphere(u);
    f32 pdf=max(EPS_F32,d.z)*INV_PI;
    return {d,pdf,1.f};
}

// Alias method for discrete importance sampling
struct AliasTable {
    static constexpr i32 MAX_N=256;
    f32 prob[MAX_N]; i32 alias_[MAX_N]; i32 n{0};

    void build(const f32* weights, i32 count) noexcept {
        n=count<MAX_N?count:MAX_N;
        f32 sum=0.f; for(i32 i=0;i<n;++i) sum+=weights[i];
        f32 inv=f32(n)/sum;
        f32 scaled[MAX_N]; for(i32 i=0;i<n;++i) scaled[i]=weights[i]*inv;
        i32 small[MAX_N],large[MAX_N],ns=0,nl=0;
        for(i32 i=0;i<n;++i){
            if(scaled[i]<1.f) small[ns++]=i; else large[nl++]=i;
        }
        for(i32 i=0;i<n;++i){prob[i]=1.f;alias_[i]=i;}
        while(ns>0&&nl>0){
            i32 s=small[--ns], l=large[--nl];
            prob[s]=scaled[s]; alias_[s]=l;
            scaled[l]=(scaled[l]+scaled[s])-1.f;
            if(scaled[l]<1.f) small[ns++]=l; else large[nl++]=l;
        }
    }

    [[nodiscard]] i32 sample(PCG32& rng) const noexcept {
        i32 i=i32(rng.bounded_u32(u32(n)));
        return rng.next_f32()<prob[i]?i:alias_[i];
    }
};

// ---------------------------------------------------------------------------
// Quasi-Monte Carlo (Sobol sequence, first few dimensions)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 sobol_d0(i32 index) noexcept {
    // Dimension 0: Van der Corput base 2
    u32 n=u32(index); u32 r=0;
    while(n){ r=(r<<1)|(n&1); n>>=1; }
    return f32(r)*0x1.0p-32f;
}

[[nodiscard]] inline f32 sobol_d1(i32 index) noexcept {
    // Dimension 1: Sobol direction numbers (simplified)
    static const u32 V[32]={
        0x80000000,0xC0000000,0xA0000000,0xF0000000,0x88000000,0xCC000000,
        0xAA000000,0xFF000000,0x80800000,0xC0C00000,0xA0A00000,0xF0F00000,
        0x88880000,0xCCCC0000,0xAAAA0000,0xFFFF0000,0x80008000,0xC000C000,
        0xA000A000,0xF000F000,0x88008800,0xCC00CC00,0xAA00AA00,0xFF00FF00,
        0x80800800,0xC0C00C00,0xA0A00A00,0xF0F00F00,0x80000000,0xC0000000,
        0xA0000000,0xF0000000
    };
    u32 n=u32(index), r=0;
    for(i32 b=0;n;++b,n>>=1) if(n&1) r^=V[b];
    return f32(r)*0x1.0p-32f;
}

[[nodiscard]] inline Vec2f sobol_2d(i32 index) noexcept {
    return {sobol_d0(index), sobol_d1(index)};
}

// ---------------------------------------------------------------------------
// Rejection sampling generic
// ---------------------------------------------------------------------------
template<typename TargetFn, typename ProposalFn, typename SampleFn>
[[nodiscard]] f32 rejection_sample(TargetFn target, ProposalFn proposal,
                                    SampleFn draw, f32 M, PCG32& rng,
                                    i32 max_tries=1000) noexcept {
    for(i32 i=0;i<max_tries;++i){
        f32 x=draw(rng);
        f32 u=rng.next_f32();
        if(u<=target(x)/(M*proposal(x))) return x;
    }
    return draw(rng); // fallback
}

// ---------------------------------------------------------------------------
// Markov Chain Monte Carlo (Metropolis-Hastings, 1D)
// ---------------------------------------------------------------------------
template<typename TargetFn>
inline void metropolis_1d(TargetFn target, f32 x0, f32* samples, i32 n,
                           f32 step=0.1f, PCG32& rng=g_rng) noexcept {
    f32 x=x0, fx=target(x);
    for(i32 i=0;i<n;++i){
        f32 xp=x+rng.range_f32(-step,step);
        f32 fxp=target(xp);
        if(rng.next_f32()<fxp/(fx+EPS_F32)){ x=xp; fx=fxp; }
        samples[i]=x;
    }
}

} // namespace qs::math
