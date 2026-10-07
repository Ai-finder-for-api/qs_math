#pragma once
// qs_math/terrain_erosion.hpp — Hydraulic erosion, thermal erosion, wind erosion (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <span>

namespace qs {

// ─── Heightmap helpers ────────────────────────────────────────────────────────

struct Heightmap {
    std::vector<f32> data;
    u32 width{}, height{};

    void resize(u32 w, u32 h, f32 init=0.f) {
        width=w; height=h; data.assign(w*h, init);
    }

    [[nodiscard]] f32& at(u32 x, u32 y)       { return data[y*width+x]; }
    [[nodiscard]] f32  at(u32 x, u32 y) const  { return data[y*width+x]; }

    [[nodiscard]] f32 sample_bilinear(f32 x, f32 y) const {
        x = std::clamp(x, 0.f, f32(width-1));
        y = std::clamp(y, 0.f, f32(height-1));
        u32 ix=u32(x), iy=u32(y);
        f32 fx=x-f32(ix), fy=y-f32(iy);
        u32 ix1=std::min(ix+1,width-1), iy1=std::min(iy+1,height-1);
        return at(ix,iy)*(1-fx)*(1-fy)+at(ix1,iy)*(fx)*(1-fy)
              +at(ix,iy1)*(1-fx)*(fy)+at(ix1,iy1)*(fx)*(fy);
    }

    [[nodiscard]] Vec3f normal_at(u32 x, u32 y, f32 cell_size=1.f) const {
        u32 xl=x>0?x-1:x, xr=x<width-1?x+1:x;
        u32 yd=y>0?y-1:y, yu=y<height-1?y+1:y;
        f32 dzdx = (at(xr,y)-at(xl,y)) / (f32(xr-xl)*cell_size);
        f32 dzdy = (at(x,yu)-at(x,yd)) / (f32(yu-yd)*cell_size);
        return normalize(Vec3f{-dzdx, 1.f, -dzdy});
    }

    [[nodiscard]] f32 slope_at(u32 x, u32 y) const {
        Vec3f n = normal_at(x,y);
        return std::acos(std::clamp(n.y,-1.f,1.f));
    }
};

// ─── Thermal erosion ─────────────────────────────────────────────────────────

struct ThermalErosionParams {
    f32 talus_angle{0.6f};  // ~34 degrees in radians
    f32 erosion_rate{0.5f};
    u32 iterations{50};
};

// One iteration of thermal erosion (talus angle)
inline void thermal_erosion_step(Heightmap& hm, const ThermalErosionParams& p) {
    static constexpr std::array<std::pair<i32,i32>,8> neighbors = {
        std::make_pair(-1,-1), std::make_pair(0,-1), std::make_pair(1,-1),
        std::make_pair(-1, 0),                       std::make_pair(1, 0),
        std::make_pair(-1, 1), std::make_pair(0, 1), std::make_pair(1, 1)};

    for (u32 y=0;y<hm.height;++y)
    for (u32 x=0;x<hm.width;++x) {
        f32 h = hm.at(x,y);
        f32 total_diff = 0.f;
        f32 max_diff = 0.f;
        for (auto [dx,dy] : neighbors) {
            u32 nx2=u32(std::clamp(i32(x)+dx,0,i32(hm.width-1)));
            u32 ny2=u32(std::clamp(i32(y)+dy,0,i32(hm.height-1)));
            f32 diff = h - hm.at(nx2,ny2);
            if (diff > p.talus_angle) {
                total_diff += diff; max_diff = std::max(max_diff, diff);
            }
        }
        if (total_diff <= 0.f) continue;
        for (auto [dx,dy] : neighbors) {
            u32 nx2=u32(std::clamp(i32(x)+dx,0,i32(hm.width-1)));
            u32 ny2=u32(std::clamp(i32(y)+dy,0,i32(hm.height-1)));
            f32 diff = h - hm.at(nx2,ny2);
            if (diff > p.talus_angle) {
                f32 move = p.erosion_rate * (max_diff - p.talus_angle) * diff / total_diff;
                hm.at(x,y)     -= move;
                hm.at(nx2,ny2) += move;
            }
        }
    }
}

inline void thermal_erode(Heightmap& hm, const ThermalErosionParams& p) {
    for (u32 i=0;i<p.iterations;++i) thermal_erosion_step(hm,p);
}

// ─── Hydraulic erosion (particle-based, Musgrave/Olsen) ──────────────────────

struct HydraulicParams {
    u32 num_droplets{50000};
    f32 inertia{0.05f};
    f32 capacity{4.0f};
    f32 deposition{0.3f};
    f32 erosion_rate{0.3f};
    f32 evaporation{0.01f};
    f32 gravity{4.0f};
    u32 max_steps{30};
    f32 min_slope{0.01f};
    u64 seed{0};
};

struct WaterDroplet {
    f32 x{}, y{};        // position
    f32 dx{}, dy{};      // direction (normalized)
    f32 speed{1.f};
    f32 water{1.f};
    f32 sediment{0.f};
};

inline void hydraulic_erode(Heightmap& hm, const HydraulicParams& p) {
    Pcg32 rng{p.seed};
    std::vector<f32> erosion_brush_weights(9, 1.f/9.f);

    for (u32 drop=0; drop<p.num_droplets; ++drop) {
        WaterDroplet d;
        d.x = rng.next_f32() * f32(hm.width-1);
        d.y = rng.next_f32() * f32(hm.height-1);
        d.dx = 0.f; d.dy = 0.f;

        for (u32 step=0; step<p.max_steps; ++step) {
            u32 cx=u32(d.x), cy=u32(d.y);
            f32 fx=d.x-f32(cx), fy=d.y-f32(cy);
            if (cx>=hm.width-1||cy>=hm.height-1) break;

            // Gradient via bilinear
            f32 h00=hm.at(cx,cy), h10=hm.at(cx+1,cy);
            f32 h01=hm.at(cx,cy+1), h11=hm.at(cx+1,cy+1);
            f32 gx=(h10-h00)*(1-fy)+(h11-h01)*fy;
            f32 gy=(h01-h00)*(1-fx)+(h11-h10)*fx;

            // Update direction
            d.dx = d.dx*p.inertia - gx*(1.f-p.inertia);
            d.dy = d.dy*p.inertia - gy*(1.f-p.inertia);
            f32 dl = std::sqrt(d.dx*d.dx+d.dy*d.dy);
            if (dl < 1e-6f) break;
            d.dx/=dl; d.dy/=dl;

            f32 new_x = d.x+d.dx, new_y = d.y+d.dy;
            if (new_x<0||new_x>f32(hm.width-1)||new_y<0||new_y>f32(hm.height-1)) break;

            f32 old_h = hm.sample_bilinear(d.x,d.y);
            f32 new_h = hm.sample_bilinear(new_x,new_y);
            f32 delta_h = new_h - old_h;

            // Sediment capacity
            f32 cap = std::max(-delta_h, p.min_slope) * d.speed * d.water * p.capacity;

            if (d.sediment > cap || delta_h > 0.f) {
                // Deposit
                f32 deposit = (delta_h > 0.f) ? std::min(delta_h, d.sediment)
                                              : (d.sediment - cap) * p.deposition;
                d.sediment -= deposit;
                hm.at(cx,cy)   += deposit*(1-fx)*(1-fy);
                hm.at(cx+1,cy) += deposit*fx*(1-fy);
                hm.at(cx,cy+1) += deposit*(1-fx)*fy;
                hm.at(std::min(cx+1,hm.width-1),std::min(cy+1,hm.height-1)) += deposit*fx*fy;
            } else {
                // Erode
                f32 erode = std::min((cap - d.sediment)*p.erosion_rate, -delta_h);
                hm.at(cx,cy)   -= erode*(1-fx)*(1-fy);
                hm.at(cx+1,cy) -= erode*fx*(1-fy);
                hm.at(cx,cy+1) -= erode*(1-fx)*fy;
                hm.at(std::min(cx+1,hm.width-1),std::min(cy+1,hm.height-1)) -= erode*fx*fy;
                d.sediment += erode;
            }

            d.speed = std::sqrt(std::max(0.f, d.speed*d.speed + delta_h*p.gravity));
            d.water *= (1.f - p.evaporation);
            if (d.water < 0.01f) break;
            d.x = new_x; d.y = new_y;
        }
    }
}

// ─── Wind erosion (saltation / aeolian) ──────────────────────────────────────

struct WindErosionParams {
    Vec2f wind_dir{1.f,0.f};   // normalized wind direction (x,z)
    f32   wind_strength{0.3f};
    f32   saltation_dist{3.f};
    f32   suspension_thresh{0.1f};
    f32   deposition_rate{0.5f};
    u32   iterations{100};
};

inline void wind_erosion_step(Heightmap& hm, const WindErosionParams& p) {
    u32 W=hm.width, H=hm.height;
    std::vector<f32> sediment(W*H, 0.f);

    // Pickup
    for (u32 y=0;y<H;++y)
    for (u32 x=0;x<W;++x) {
        f32 slope = hm.slope_at(x,y);
        f32 exposed = std::max(0.f, 1.f - slope / k_pi_over_two);
        f32 pickup = exposed * p.wind_strength;
        hm.at(x,y) -= pickup;
        sediment[y*W+x] += pickup;
    }

    // Transport + deposit
    std::vector<f32> new_sediment(W*H, 0.f);
    for (u32 y=0;y<H;++y)
    for (u32 x=0;x<W;++x) {
        f32 s = sediment[y*W+x];
        if (s < 1e-6f) continue;
        f32 tx = f32(x) + p.wind_dir.x*p.saltation_dist;
        f32 ty = f32(y) + p.wind_dir.y*p.saltation_dist;
        tx = std::clamp(tx,0.f,f32(W-1));
        ty = std::clamp(ty,0.f,f32(H-1));
        u32 nx2=u32(tx), ny2=u32(ty);
        f32 fx=tx-f32(nx2), fy=ty-f32(ny2);
        u32 nx1=std::min(nx2+1,W-1), ny1=std::min(ny2+1,H-1);
        new_sediment[ny2*W+nx2]  += s*(1-fx)*(1-fy);
        new_sediment[ny2*W+nx1]  += s*fx*(1-fy);
        new_sediment[ny1*W+nx2]  += s*(1-fx)*fy;
        new_sediment[ny1*W+nx1]  += s*fx*fy;
    }

    // Deposit
    for (u32 i=0;i<W*H;++i)
        hm.data[i] += new_sediment[i]*p.deposition_rate;
}

inline void wind_erode(Heightmap& hm, const WindErosionParams& p) {
    for (u32 i=0;i<p.iterations;++i) wind_erosion_step(hm,p);
}

} // namespace qs
