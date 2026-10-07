#pragma once
// qs_math/spatial_audio_math.hpp — Reverb zones, occlusion, HRTF helpers, ambisonics (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "geometry3d.hpp"
#include "intersection.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <functional>

namespace qs {

// ─── 3D spatialization math ────────────────────────────────────────────────────

// Distance attenuation models
enum class AttenuationModel : u8 { Linear, InverseDistance, InverseDistanceClamped, Exponential, None };

[[nodiscard]] inline f32 distance_attenuation(f32 dist, f32 ref_dist, f32 max_dist,
    f32 rolloff, AttenuationModel model)
{
    dist=std::max(dist,ref_dist);
    switch(model) {
        case AttenuationModel::Linear:
            return std::clamp(1-(rolloff*(dist-ref_dist)/(max_dist-ref_dist)),0.f,1.f);
        case AttenuationModel::InverseDistance:
            return ref_dist/(ref_dist+rolloff*(dist-ref_dist));
        case AttenuationModel::InverseDistanceClamped:
            dist=std::clamp(dist,ref_dist,max_dist);
            return ref_dist/(ref_dist+rolloff*(dist-ref_dist));
        case AttenuationModel::Exponential:
            return std::pow(dist/ref_dist,-rolloff);
        case AttenuationModel::None: default:
            return 1.f;
    }
}

// Doppler shift factor: f' = f * (c + v_listener) / (c + v_source)
[[nodiscard]] inline f32 doppler_factor(Vec3f listener_pos, Vec3f listener_vel,
    Vec3f source_pos, Vec3f source_vel, f32 speed_of_sound=343.f)
{
    Vec3f dir=normalize(source_pos-listener_pos);
    f32 vl=dot(listener_vel,dir);
    f32 vs=dot(source_vel,dir);
    return std::max(0.001f,(speed_of_sound+vl)/(speed_of_sound+vs));
}

// Panning: stereo pan from azimuth angle (radians)
[[nodiscard]] inline Vec2f stereo_pan(f32 azimuth) {
    // VBAP-style constant-power panning
    f32 a=azimuth+f32(std::numbers::pi)*0.5f;
    return {std::cos(a),std::sin(a)};
}

// ─── Azimuth/elevation from listener perspective ─────────────────────────────
struct AzimuthElevation {
    f32 azimuth{0};   // radians, 0=front, π/2=right
    f32 elevation{0}; // radians, 0=horizon, π/2=above
    f32 distance{0};
};

[[nodiscard]] inline AzimuthElevation world_to_listener_space(
    Vec3f source_pos, Vec3f listener_pos,
    Vec3f listener_forward, Vec3f listener_up)
{
    Vec3f dir=source_pos-listener_pos;
    f32 d=length(dir);
    if(d<1e-7f) return {};
    dir=dir/d;
    Vec3f right=cross(listener_forward,listener_up);
    f32 az=std::atan2(dot(dir,right),dot(dir,listener_forward));
    f32 el=std::asin(std::clamp(dot(dir,listener_up),-1.f,1.f));
    return {az,el,d};
}

// ─── HRTF (Head-Related Transfer Function) helpers ────────────────────────────
// Simplified analytical HRTF model (Woodworth spherical head model)
struct HRTFModel {
    f32 head_radius{0.0875f}; // m (average human head radius)
    f32 ear_offset{f32(std::numbers::pi)*0.1f}; // ear offset from front in radians

    // ITD — Interaural Time Difference in seconds
    [[nodiscard]] f32 itd(f32 azimuth, f32 speed_of_sound=343.f) const {
        // Woodworth: ITD = (r/c)(azimuth + sin(azimuth))
        return (head_radius/speed_of_sound)*(azimuth+std::sin(azimuth));
    }

    // ILD — Interaural Level Difference in dB
    [[nodiscard]] f32 ild_dB(f32 azimuth, f32 freq=1000.f) const {
        // Simplified frequency-dependent ILD
        f32 ka=2*f32(std::numbers::pi)*freq*head_radius/343.f;
        f32 shadow=20*std::log10(std::max(0.01f,std::abs(std::cos(azimuth*0.5f))));
        return shadow*(1+ka*0.5f);
    }

    // Pinna notch frequency (simplified)
    [[nodiscard]] f32 pinna_notch_hz(f32 elevation) const {
        return 8000.f+4000.f*std::sin(elevation);
    }

    // Generate stereo gains from azimuth/elevation
    [[nodiscard]] std::pair<f32,f32> stereo_gains(f32 azimuth) const {
        // Left/right gain based on azimuth
        f32 pan=std::sin(azimuth)*0.5f+0.5f; // [0,1]
        return {std::sqrt(1-pan),std::sqrt(pan)};
    }
};

// HRTF lookup table interpolation (nearest-neighbour + bilinear in az/el)
struct HRTFTable {
    u32 n_az{12},n_el{5};       // azimuth/elevation samples
    u32 ir_len{128};             // impulse response length
    std::vector<f32> left_ir, right_ir; // [n_az * n_el * ir_len]

    [[nodiscard]] u32 idx(u32 az_i,u32 el_i) const { return (el_i*n_az+az_i)*ir_len; }

    // Nearest-neighbour lookup
    [[nodiscard]] std::pair<const f32*,const f32*> lookup(f32 azimuth, f32 elevation) const {
        u32 az_i=u32(std::round((azimuth+f32(std::numbers::pi))/(2*f32(std::numbers::pi)/n_az)))%n_az;
        u32 el_i=u32(std::clamp(i32(std::round((elevation+f32(std::numbers::pi)*0.5f)/(f32(std::numbers::pi)/n_el))),0,i32(n_el)-1));
        u32 i=idx(az_i,el_i);
        if(i+ir_len>u32(left_ir.size())) return {nullptr,nullptr};
        return {left_ir.data()+i,right_ir.data()+i};
    }
};

// ─── Ambisonics (FOA/HOA) ─────────────────────────────────────────────────────
// First Order Ambisonics: W,X,Y,Z channels (B-format)
struct AmbisonicsFOA {
    f32 W{0},X{0},Y{0},Z{0};

    // Encode a point source at given direction
    static AmbisonicsFOA encode(Vec3f direction, f32 gain=1.f) {
        direction=normalize(direction);
        return {
            gain*f32(1.0/std::numbers::sqrt2),  // W: omnidirectional
            gain*direction.x,                    // X: front-back
            gain*direction.y,                    // Y: left-right
            gain*direction.z                     // Z: up-down
        };
    }

    // Decode to stereo (left/right)
    [[nodiscard]] Vec2f decode_stereo(f32 width=1.f) const {
        return {(W+Y*width)*f32(1.0/std::numbers::sqrt2),
                (W-Y*width)*f32(1.0/std::numbers::sqrt2)};
    }

    // Decode to 5.0 (L,R,C,LS,RS)
    [[nodiscard]] std::array<f32,5> decode_5() const {
        static constexpr f32 sq2=f32(std::numbers::sqrt2_v<f32>);
        return {
            (W+X+Y)*0.5f,           // L
            (W+X-Y)*0.5f,           // R
            (W+X)*sq2*0.5f,         // C (simplified)
            (W-X+Y)*0.5f,           // LS
            (W-X-Y)*0.5f            // RS
        };
    }

    // Rotate B-format by Euler angles
    [[nodiscard]] AmbisonicsFOA rotate(f32 yaw, f32 pitch, f32 roll) const {
        f32 cy=std::cos(yaw),sy=std::sin(yaw);
        f32 cp=std::cos(pitch),sp=std::sin(pitch);
        f32 cr=std::cos(roll),sr=std::sin(roll);
        // Yaw rotation matrix
        f32 Xr=X*cy-Y*sy, Yr=X*sy+Y*cy;
        // Pitch
        f32 Xrp=Xr*cp+Z*sp, Zrp=-Xr*sp+Z*cp;
        // Roll
        f32 Yrr=Yr*cr-Zrp*sr, Zrr=Yr*sr+Zrp*cr;
        return {W,Xrp,Yrr,Zrr};
    }

    AmbisonicsFOA operator+(const AmbisonicsFOA& o) const { return {W+o.W,X+o.X,Y+o.Y,Z+o.Z}; }
    AmbisonicsFOA operator*(f32 s) const { return {W*s,X*s,Y*s,Z*s}; }
};

// Higher Order Ambisonics: up to 3rd order (16 channels)
struct AmbisonicsHOA {
    static constexpr u32 MAX_ORDER=3;
    static constexpr u32 NUM_CHANNELS=16; // (3+1)² = 16
    std::array<f32,NUM_CHANNELS> ch{};

    // Encode point source using real spherical harmonics up to order N
    static AmbisonicsHOA encode(Vec3f dir, f32 gain=1.f, u32 order=3) {
        dir=normalize(dir);
        f32 x=dir.x,y=dir.y,z=dir.z;
        AmbisonicsHOA a;
        // Order 0
        a.ch[0]=gain*1.f;
        if(order<1) return a;
        // Order 1 (W,Y,Z,X in ACN)
        a.ch[1]=gain*y; a.ch[2]=gain*z; a.ch[3]=gain*x;
        if(order<2) return a;
        // Order 2 (5 channels)
        a.ch[4]=gain*(2*z*z-x*x-y*y)*0.5f;         // z²-..
        a.ch[5]=gain*(x*z);                          // xz
        a.ch[6]=gain*(x*x-y*y)*0.5f;                // x²-y²
        a.ch[7]=gain*(x*y);                          // xy
        a.ch[8]=gain*(y*z);                          // yz
        if(order<3) return a;
        // Order 3 (7 channels)
        a.ch[9] =gain*y*(5*y*y-3);    // simplified
        a.ch[10]=gain*x*z*(5*z*z-1);
        a.ch[11]=gain*y*(4*z*z-1);
        a.ch[12]=gain*z*(2*z*z-3*x*x-3*y*y)*0.5f;
        a.ch[13]=gain*x*(4*z*z-1);
        a.ch[14]=gain*z*(x*x-y*y);
        a.ch[15]=gain*x*(x*x-3*y*y);
        return a;
    }

    AmbisonicsHOA operator+(const AmbisonicsHOA& o) const {
        AmbisonicsHOA r; for(u32 i=0;i<NUM_CHANNELS;++i) r.ch[i]=ch[i]+o.ch[i]; return r;
    }
    AmbisonicsHOA operator*(f32 s) const {
        AmbisonicsHOA r; for(u32 i=0;i<NUM_CHANNELS;++i) r.ch[i]=ch[i]*s; return r;
    }
};

// ─── Reverb zone ───────────────────────────────────────────────────────────────
struct ReverbZone {
    Sphere<f32>  bounds;
    f32 min_distance{0};  // inner blend radius
    f32 decay_time{1.5f}; // RT60 in seconds
    f32 pre_delay{0.02f}; // pre-delay in seconds
    f32 dry_mix{0.5f};
    f32 wet_mix{0.3f};
    f32 diffusion{0.7f};
    f32 density{0.7f};
    f32 hf_rolloff{5000.f}; // high-frequency rolloff in Hz
    f32 lf_boost{200.f};    // low-frequency boost in Hz

    [[nodiscard]] f32 blend_weight(Vec3f listener_pos) const {
        f32 d=dist(listener_pos,bounds.center);
        if(d>bounds.radius) return 0;
        f32 outer=bounds.radius, inner=bounds.radius*min_distance;
        return std::clamp((outer-d)/(outer-inner+1e-5f),0.f,1.f);
    }
};

// Blend reverb parameters from multiple overlapping zones
[[nodiscard]] inline ReverbZone blend_reverb_zones(
    const std::vector<ReverbZone>& zones, Vec3f listener)
{
    ReverbZone result;
    f32 total_w=0;
    for(const auto& z:zones) {
        f32 w=z.blend_weight(listener);
        if(w<1e-4f) continue;
        result.decay_time+=z.decay_time*w;
        result.wet_mix+=z.wet_mix*w;
        result.diffusion+=z.diffusion*w;
        result.density+=z.density*w;
        total_w+=w;
    }
    if(total_w>1e-5f){
        result.decay_time/=total_w; result.wet_mix/=total_w;
        result.diffusion/=total_w; result.density/=total_w;
    }
    return result;
}

// ─── Acoustic occlusion (ray-based) ──────────────────────────────────────────
struct OcclusionResult {
    f32 gain{1};       // [0,1] — 1=no occlusion
    f32 lp_factor{1};  // low-pass amount — occluded sound is muffled
};

// Cast N rays from source to listener; count how many are unblocked
[[nodiscard]] inline OcclusionResult acoustic_occlusion(
    Vec3f source, Vec3f listener,
    const std::function<bool(Vec3f,Vec3f)>& ray_blocked,
    u32 n_rays=8)
{
    u32 unblocked=0;
    Vec3f dir=normalize(listener-source);
    f32 d=dist(source,listener);
    // Random perturbation rays around direct path
    static const Vec2f offsets[8]={{0,0},{0.05f,0},{-0.05f,0},{0,0.05f},{0,0-0.05f},{0.035f,0.035f},{-0.035f,0.035f},{0.035f,-0.035f}};
    for(u32 i=0;i<n_rays;++i) {
        Vec3f perp1=std::abs(dir.x)<0.9f?Vec3f{1,0,0}:Vec3f{0,1,0};
        perp1=normalize(cross(dir,perp1));
        Vec3f perp2=cross(dir,perp1);
        Vec3f jitter=perp1*offsets[i%8].x+perp2*offsets[i%8].y;
        Vec3f dst=listener+jitter*d*0.1f;
        if(!ray_blocked(source,dst)) ++unblocked;
    }
    f32 ratio=f32(unblocked)/n_rays;
    return {ratio, 0.3f+ratio*0.7f}; // partial LP even when mostly unblocked
}

} // namespace qs
