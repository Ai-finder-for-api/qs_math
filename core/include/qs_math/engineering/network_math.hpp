#pragma once
// qs_math/network_math.hpp — Latency compensation, lag prediction, delta compression (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "quat.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <deque>
#include <array>
#include <optional>
#include <span>

namespace qs {

// ─── Entity state snapshot ───────────────────────────────────────────────────

struct EntitySnapshot {
    f64   timestamp{};      // server time in seconds
    Vec3f position{};
    Vec3f velocity{};
    Quatf rotation{0,0,0,1};
    Vec3f angular_vel{};
    u32   sequence{};
};

// ─── Entity interpolation buffer ─────────────────────────────────────────────

struct InterpolationBuffer {
    std::deque<EntitySnapshot> snapshots;
    f64  render_delay{0.1};     // seconds behind server time (buffer)
    u32  max_snapshots{32};

    void add(const EntitySnapshot& s){
        snapshots.push_back(s);
        if(snapshots.size()>max_snapshots) snapshots.pop_front();
    }

    // Interpolate at render time (server_time - render_delay)
    [[nodiscard]] std::optional<EntitySnapshot> sample(f64 server_time) const {
        f64 t=server_time-render_delay;
        if(snapshots.size()<2) return snapshots.empty()?std::nullopt:std::optional{snapshots.front()};
        for(u32 i=0;i+1<snapshots.size();++i){
            const auto& a=snapshots[i], &b=snapshots[i+1];
            if(t>=a.timestamp&&t<=b.timestamp){
                f32 alpha=f32((t-a.timestamp)/(b.timestamp-a.timestamp));
                EntitySnapshot out;
                out.timestamp=t;
                out.position=a.position+(b.position-a.position)*alpha;
                out.velocity=a.velocity+(b.velocity-a.velocity)*alpha;
                out.rotation=slerp(a.rotation,b.rotation,alpha);
                out.sequence=a.sequence;
                return out;
            }
        }
        // Extrapolate if ahead
        if(!snapshots.empty()&&t>snapshots.back().timestamp){
            const auto& last=snapshots.back();
            f32 dt=f32(t-last.timestamp);
            EntitySnapshot out=last;
            out.position=last.position+last.velocity*dt;
            out.timestamp=t;
            return out;
        }
        return std::nullopt;
    }
};

// ─── Lag compensation (server-side rewind) ───────────────────────────────────

struct LagCompHistory {
    std::deque<EntitySnapshot> history;
    f64 max_history_duration{1.0}; // seconds

    void record(const EntitySnapshot& s, f64 current_time){
        history.push_back(s);
        while(!history.empty()&&current_time-history.front().timestamp>max_history_duration)
            history.pop_front();
    }

    [[nodiscard]] std::optional<EntitySnapshot> rewind(f64 target_time) const {
        if(history.empty()) return std::nullopt;
        for(i32 i=i32(history.size())-1;i>=1;--i){
            const auto& a=history[i-1], &b=history[i];
            if(target_time>=a.timestamp&&target_time<=b.timestamp){
                f32 alpha=f32((target_time-a.timestamp)/(b.timestamp-a.timestamp));
                EntitySnapshot out;
                out.position=a.position+(b.position-a.position)*alpha;
                out.velocity=a.velocity;
                out.rotation=slerp(a.rotation,b.rotation,alpha);
                out.timestamp=target_time;
                return out;
            }
        }
        return history.front();
    }
};

// ─── Dead reckoning ──────────────────────────────────────────────────────────

[[nodiscard]] inline EntitySnapshot dead_reckon(
    const EntitySnapshot& last_known, f64 current_time)
{
    f32 dt=f32(current_time-last_known.timestamp);
    EntitySnapshot out=last_known;
    out.position=last_known.position+last_known.velocity*dt;
    out.timestamp=current_time;
    return out;
}

// Snap + blend correction
struct DeadReckonCorrection {
    Vec3f visual_pos{};
    Vec3f correction_vel{};
    f32   blend_time{0.1f};
    f32   blend_t{0.f};

    void apply_correction(const Vec3f& authoritative_pos, const Vec3f& current_visual) {
        correction_vel=(authoritative_pos-current_visual)/blend_time;
        visual_pos=current_visual;
        blend_t=0.f;
    }

    Vec3f update(f32 dt) {
        if(blend_t<blend_time){
            visual_pos=visual_pos+correction_vel*dt;
            blend_t+=dt;
        }
        return visual_pos;
    }
};

// ─── RTT / latency estimation ─────────────────────────────────────────────────

struct RTTEstimator {
    f32 rtt{0.1f};          // estimated round-trip time (seconds)
    f32 rtt_var{0.02f};     // variance
    f32 alpha{0.125f};      // EWMA gain for RTT
    f32 beta{0.25f};        // EWMA gain for variance

    void update_sample(f32 new_rtt) {
        f32 err=new_rtt-rtt;
        rtt+=alpha*err;
        rtt_var=rtt_var*(1.f-beta)+beta*std::abs(err);
    }

    [[nodiscard]] f32 srtt() const { return rtt; }
    [[nodiscard]] f32 rto()  const { return rtt+4.f*rtt_var; } // RFC 6298
    [[nodiscard]] f32 one_way() const { return rtt*0.5f; }
};

// ─── Delta compression ────────────────────────────────────────────────────────

struct QuantizedVec3 {
    i16 x{},y{},z{};  // fixed-point, scale 1/128 m

    [[nodiscard]] static QuantizedVec3 encode(const Vec3f& v) {
        return {i16(v.x*128.f), i16(v.y*128.f), i16(v.z*128.f)};
    }

    [[nodiscard]] Vec3f decode() const {
        return {f32(x)/128.f, f32(y)/128.f, f32(z)/128.f};
    }
};

struct QuantizedQuat {
    i16 x{},y{},z{};  // smallest-three compression, w inferred
    u8  largest_idx{};

    [[nodiscard]] static QuantizedQuat encode(const Quatf& q) {
        // Find largest component
        f32 vals[4]={std::abs(q.x),std::abs(q.y),std::abs(q.z),std::abs(q.w)};
        u8 largest=0;
        for(u8 i=1;i<4;++i) if(vals[i]>vals[largest]) largest=i;
        f32 sign=((&q.x)[largest]<0.f)?-1.f:1.f;
        Quatf nq={q.x*sign,q.y*sign,q.z*sign,q.w*sign};
        f32 comp[4]={nq.x,nq.y,nq.z,nq.w};
        QuantizedQuat qq; qq.largest_idx=largest;
        u32 out=0;
        for(u32 i=0,j=0;i<4;++i){
            if(i==largest) continue;
            i16& slot=(j==0?qq.x:j==1?qq.y:qq.z);
            slot=i16(comp[i]*23169.f); // 2^14 * sqrt(2) for max precision
            ++j;
        }
        return qq;
    }

    [[nodiscard]] Quatf decode() const {
        f32 comp[3]={f32(x)/23169.f,f32(y)/23169.f,f32(z)/23169.f};
        f32 sum=comp[0]*comp[0]+comp[1]*comp[1]+comp[2]*comp[2];
        f32 largest=std::sqrt(std::max(0.f,1.f-sum));
        Quatf q;
        u32 j=0;
        for(u32 i=0;i<4;++i){
            if(i==largest_idx){(&q.x)[i]=largest;}
            else{(&q.x)[i]=comp[j++];}
        }
        return normalize(q);
    }
};

// Simple XOR delta for integer data
[[nodiscard]] inline std::vector<u8> xor_delta(
    std::span<const u8> base, std::span<const u8> current)
{
    u32 n=u32(std::min(base.size(),current.size()));
    std::vector<u8> delta(n);
    for(u32 i=0;i<n;++i) delta[i]=base[i]^current[i];
    return delta;
}

// ─── Jitter buffer ────────────────────────────────────────────────────────────

struct JitterBuffer {
    std::deque<std::pair<f64,u32>> packets; // (timestamp, sequence)
    f64 target_delay{0.05};   // seconds
    f64 min_delay{0.02};
    f64 max_delay{0.3};
    f32 alpha{0.05f};         // adaptation speed

    void receive(f64 arrival_time, u32 seq, f64 send_time) {
        f64 transit=arrival_time-send_time;
        target_delay=target_delay*(1.f-alpha)+transit*alpha;
        target_delay=std::clamp(target_delay,min_delay,max_delay);
        packets.push_back({send_time+target_delay,seq});
    }

    [[nodiscard]] std::optional<u32> pop(f64 current_time) {
        if(packets.empty()) return std::nullopt;
        if(packets.front().first<=current_time){
            u32 seq=packets.front().second;
            packets.pop_front();
            return seq;
        }
        return std::nullopt;
    }
};

} // namespace qs
