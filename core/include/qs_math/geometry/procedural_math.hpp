#pragma once
// procedural_math.hpp — Procedural generation mathematics:
//   L-systems, space filling curves, Wang tiles, reaction-diffusion,
//   cellular automata, fractal dimension, Lindenmayer systems,
//   grammar-based generation, WFC math helpers
#include "vec2.hpp"
#include "vec3.hpp"
#include "scalar_math.hpp"
#include "noise.hpp"
#include "random.hpp"
#include "trig.hpp"

namespace qs::math::procedural {

// ---------------------------------------------------------------------------
// L-System turtle state
// ---------------------------------------------------------------------------
struct TurtleState {
    Vec3f pos{0.f};
    Vec3f dir{0.f,1.f,0.f};  // heading
    Vec3f up{0.f,0.f,1.f};
    f32   step{1.f};
    f32   angle{25.f};        // branching angle (degrees)
    i32   depth{0};
};

struct TurtleStack {
    static constexpr i32 MAX=64;
    TurtleState stack[MAX]; i32 top{0};
    void push(TurtleState s) noexcept { if(top<MAX) stack[top++]=s; }
    TurtleState pop() noexcept { return top>0?stack[--top]:TurtleState{}; }
};

// Apply L-system move
inline void turtle_forward(TurtleState& t, Vec3f* out_from, Vec3f* out_to) noexcept {
    *out_from=t.pos;
    t.pos=t.pos+t.dir*t.step;
    *out_to=t.pos;
}
inline void turtle_rotate_left(TurtleState& t) noexcept {
    Quatf r=Quatf::from_axis_angle(t.up,t.angle*DEG_TO_RAD);
    t.dir=r.rotate(t.dir).normalized();
}
inline void turtle_rotate_right(TurtleState& t) noexcept {
    Quatf r=Quatf::from_axis_angle(t.up,-t.angle*DEG_TO_RAD);
    t.dir=r.rotate(t.dir).normalized();
}
inline void turtle_pitch_up(TurtleState& t) noexcept {
    Vec3f right=cross(t.dir,t.up).normalized();
    Quatf r=Quatf::from_axis_angle(right,t.angle*DEG_TO_RAD);
    t.dir=r.rotate(t.dir).normalized();
    t.up=r.rotate(t.up).normalized();
}
inline void turtle_roll(TurtleState& t) noexcept {
    Quatf r=Quatf::from_axis_angle(t.dir,t.angle*DEG_TO_RAD);
    t.up=r.rotate(t.up).normalized();
}

// ---------------------------------------------------------------------------
// Space-filling curves (index ↔ 2D/3D position)
// ---------------------------------------------------------------------------
// Hilbert curve (2D, n levels)
[[nodiscard]] constexpr Vec2i hilbert_d2xy(i32 n, i32 d) noexcept {
    i32 x=0,y=0;
    for(i32 s=1;s<n;s<<=1){
        i32 rx=(d/2)&1, ry=(d^rx)&1;
        if(!ry){ if(rx){ x=s-1-x; y=s-1-y; } i32 t=x; x=y; y=t; }
        x+=s*rx; y+=s*ry; d>>=2;
    }
    return {x,y};
}
[[nodiscard]] constexpr i32 hilbert_xy2d(i32 n, Vec2i xy) noexcept {
    i32 d=0;
    for(i32 s=n/2;s>0;s>>=1){
        i32 rx=(xy.x&s)?1:0, ry=(xy.y&s)?1:0;
        d+=s*s*((3*rx)^ry);
        // Rotate
        if(!ry){ if(rx){ xy.x=s-1-xy.x; xy.y=s-1-xy.y; } i32 t=xy.x; xy.x=xy.y; xy.y=t; }
    }
    return d;
}

// Morton Z-curve (re-export from hashing.hpp convenience)
[[nodiscard]] constexpr u64 zorder_2d(u32 x, u32 y) noexcept { return morton2(x,y); }
[[nodiscard]] constexpr u64 zorder_3d(u32 x, u32 y, u32 z) noexcept { return morton3(x,y,z); }

// ---------------------------------------------------------------------------
// Reaction-diffusion (Gray-Scott model)
// ---------------------------------------------------------------------------
struct GrayScottParams {
    f32 feed{0.055f};     // feed rate
    f32 kill{0.062f};     // kill rate
    f32 DA{1.0f};         // diffusion rate A
    f32 DB{0.5f};         // diffusion rate B
    f32 dt{1.0f};
};

inline void gray_scott_step(
    f32* A, f32* B, f32* nA, f32* nB,
    i32 w, i32 h, GrayScottParams p) noexcept
{
    for(i32 y=0;y<h;++y) for(i32 x=0;x<w;++x){
        i32 i=y*w+x;
        // Laplacian (5-point stencil)
        auto wrap=[&](i32 v,i32 max)->i32{ return (v+max)%max; };
        f32 lapA=A[wrap(y-1,h)*w+x]+A[wrap(y+1,h)*w+x]+A[y*w+wrap(x-1,w)]+A[y*w+wrap(x+1,w)]-4.f*A[i];
        f32 lapB=B[wrap(y-1,h)*w+x]+B[wrap(y+1,h)*w+x]+B[y*w+wrap(x-1,w)]+B[y*w+wrap(x+1,w)]-4.f*B[i];
        f32 a=A[i], b=B[i], ab2=a*b*b;
        nA[i]=a+p.dt*(p.DA*lapA-ab2+p.feed*(1.f-a));
        nB[i]=b+p.dt*(p.DB*lapB+ab2-(p.kill+p.feed)*b);
        nA[i]=clamp(nA[i],0.f,1.f); nB[i]=clamp(nB[i],0.f,1.f);
    }
}

// ---------------------------------------------------------------------------
// Cellular automata
// ---------------------------------------------------------------------------
// Conway's Game of Life step
inline void life_step(const bool* in, bool* out, i32 w, i32 h) noexcept {
    for(i32 y=0;y<h;++y) for(i32 x=0;x<w;++x){
        i32 n=0;
        for(i32 dy=-1;dy<=1;++dy) for(i32 dx=-1;dx<=1;++dx){
            if(dx==0&&dy==0) continue;
            i32 nx=(x+dx+w)%w, ny=(y+dy+h)%h;
            if(in[ny*w+nx]) ++n;
        }
        bool c=in[y*w+x];
        out[y*w+x]=(c&&(n==2||n==3))||(!c&&n==3);
    }
}

// Wolfram 1D cellular automaton
[[nodiscard]] constexpr bool wolfram_rule(u8 rule, u8 left, u8 center, u8 right) noexcept {
    u8 pattern=(left<<2)|(center<<1)|right;
    return (rule>>pattern)&1;
}
inline void wolfram_step(const u8* row, u8* next_row, i32 w, u8 rule) noexcept {
    for(i32 x=0;x<w;++x){
        u8 l=row[(x-1+w)%w], c=row[x], r=row[(x+1)%w];
        next_row[x]=wolfram_rule(rule,l,c,r)?1:0;
    }
}

// ---------------------------------------------------------------------------
// Poisson disk 3D
// ---------------------------------------------------------------------------
inline i32 poisson_disk_3d(
    Vec3f* out, i32 max_pts, f32 min_dist,
    Vec3f domain_min, Vec3f domain_max, PCG32& rng) noexcept
{
    i32 n=0; i32 candidates=30;
    Vec3f dom=domain_max-domain_min;
    while(n<max_pts){
        Vec3f best; f32 best_dist=0.f;
        for(i32 k=0;k<candidates;++k){
            Vec3f p=domain_min+Vec3f{rng.range_f32(0.f,dom.x),rng.range_f32(0.f,dom.y),rng.range_f32(0.f,dom.z)};
            f32 d=F32_INF;
            for(i32 j=0;j<n;++j) d=min(d,(p-out[j]).length());
            if(d>best_dist){best_dist=d;best=p;}
        }
        if(best_dist<min_dist&&n>0) break;
        out[n++]=best;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Fractal dimension estimation (box counting)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 box_counting_dim(
    const Vec2f* pts, i32 n, i32 min_scale=1, i32 max_scale=128) noexcept
{
    // For a set of 2D points, count occupied boxes at each scale
    f32 log_r1=log(f32(min_scale)), log_n1=0.f;
    f32 log_r2=log(f32(max_scale)), log_n2=0.f;
    // At scale min_scale
    {
        i32 cnt=0;
        for(i32 i=0;i<n;++i) for(i32 j=i+1;j<n;++j)
            if((pts[i]-pts[j]).length()<min_scale) { ++cnt; break; }
        log_n1=log(f32(cnt+1));
    }
    {
        i32 cnt=0;
        for(i32 i=0;i<n;++i) for(i32 j=i+1;j<n;++j)
            if((pts[i]-pts[j]).length()<max_scale) { ++cnt; break; }
        log_n2=log(f32(cnt+1));
    }
    f32 dlog_r=log_r2-log_r1;
    return abs(dlog_r)>EPS_F32?(log_n2-log_n1)/dlog_r:1.f;
}

// ---------------------------------------------------------------------------
// Wave Function Collapse helpers (constraint propagation math)
// ---------------------------------------------------------------------------
// Entropy of a superposition state (Shannon)
[[nodiscard]] constexpr f32 wfc_entropy(const f32* weights, i32 n) noexcept {
    f32 sum=0.f, log_sum=0.f;
    for(i32 i=0;i<n;++i) if(weights[i]>0.f){ sum+=weights[i]; log_sum+=weights[i]*log(weights[i]); }
    return sum>EPS_F32?log(sum)-log_sum/sum:0.f;
}

// Find lowest-entropy uncollapsed cell (for observation)
[[nodiscard]] inline i32 wfc_min_entropy_cell(
    const f32** cell_weights, const i32* cell_counts,
    const bool* collapsed, i32 n_cells, PCG32& rng) noexcept
{
    f32 min_e=F32_INF; i32 best=-1;
    for(i32 i=0;i<n_cells;++i){
        if(collapsed[i]) continue;
        f32 e=wfc_entropy(cell_weights[i],cell_counts[i]);
        if(e<min_e){ min_e=e; best=i; }
    }
    return best;
}

} // namespace qs::math::procedural
