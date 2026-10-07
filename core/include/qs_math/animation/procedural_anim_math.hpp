#pragma once
// qs_math/procedural_anim_math.hpp — Secondary motion, jiggle, breath, secondary dynamics (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "quat.hpp"
#include "mat4.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

namespace qs {

// ─── Damped spring ────────────────────────────────────────────────────────────

struct DampedSpring3D {
    Vec3f pos{}, vel{}, target{};
    f32 frequency{5.f};   // natural freq (Hz)
    f32 damping{0.7f};    // damping ratio (1=critical)

    Vec3f update(f32 dt, const Vec3f& new_target) {
        target=new_target;
        f32 w=k_tau*frequency;
        f32 d=2.f*damping*w;
        Vec3f acc=(target-pos)*(w*w)-(vel*d);
        vel=vel+acc*dt;
        pos=pos+vel*dt;
        return pos;
    }

    void teleport(const Vec3f& p){ pos=p; vel={}; }
};

// ─── Jiggle bone ─────────────────────────────────────────────────────────────

struct JiggleBone {
    Vec3f world_pos{};
    Vec3f velocity{};
    Vec3f rest_offset{0,0.1f,0}; // offset from parent in local space
    f32 stiffness{50.f};
    f32 damping{5.f};
    f32 mass{0.1f};
    f32 gravity_scale{0.5f};
    f32 max_deviation{0.3f};

    Vec3f update(f32 dt, const Vec3f& parent_world, const Mat4f& parent_to_world) {
        // Desired position in world space
        Vec4f rest_w=parent_to_world*Vec4f{rest_offset.x,rest_offset.y,rest_offset.z,1.f};
        Vec3f target{rest_w.x,rest_w.y,rest_w.z};
        // Spring force
        Vec3f force=(target-world_pos)*stiffness-velocity*damping;
        force.y-=9.81f*mass*gravity_scale;
        velocity=velocity+force*(1.f/mass)*dt;
        world_pos=world_pos+velocity*dt;
        // Clamp deviation
        Vec3f dev=world_pos-target;
        f32 dev_len=length(dev);
        if(dev_len>max_deviation) world_pos=target+dev/dev_len*max_deviation;
        return world_pos;
    }
};

// ─── Breathing animation ──────────────────────────────────────────────────────

struct BreathCycle {
    f32 phase{0.f};
    f32 rate{0.25f};          // breaths/second (rest)
    f32 inhale_ratio{0.4f};   // fraction of cycle for inhale
    f32 chest_amplitude{0.02f};
    f32 belly_amplitude{0.015f};
    f32 head_nod_amplitude{0.003f};

    struct Result {
        f32 chest_expand{};     // up/down
        f32 belly_expand{};
        f32 head_nod{};
        f32 shoulder_raise{};
    };

    Result update(f32 dt, f32 exertion=0.f) {
        f32 effective_rate=rate*(1.f+exertion*3.f);
        phase=std::fmod(phase+effective_rate*dt,1.f);
        // Non-sinusoidal: quick inhale, slow exhale
        f32 breath_shape;
        if(phase<inhale_ratio)
            breath_shape=std::sin(k_pi*phase/inhale_ratio);
        else
            breath_shape=std::sin(k_pi*(1.f-phase)/(1.f-inhale_ratio));

        f32 amp=1.f+exertion*2.f;
        return {
            chest_expand*breath_shape*amp,
            belly_amplitude*std::sin(k_pi*phase)*amp,
            head_nod_amplitude*std::sin(k_tau*phase),
            chest_expand*0.3f*breath_shape*amp
        };
    }
};

// ─── Procedural walk cycle ────────────────────────────────────────────────────

struct WalkCycle {
    f32 phase{0.f};             // 0..1 cycle
    f32 step_length{0.6f};      // meters
    f32 step_height{0.12f};     // meters
    f32 body_bob_amp{0.025f};
    f32 body_sway_amp{0.02f};
    f32 arm_swing_amp{0.3f};    // radians

    struct StepResult {
        Vec3f left_foot_offset{};
        Vec3f right_foot_offset{};
        Vec3f body_offset{};
        f32 left_arm_angle{};
        f32 right_arm_angle{};
    };

    StepResult evaluate(f32 speed_mps) {
        f32 freq=speed_mps/step_length;
        phase=std::fmod(phase+freq*0.016f,1.f); // assume ~60fps tick
        // Left foot: phase, Right foot: phase + 0.5
        auto foot=[&](f32 p)->Vec3f{
            f32 fwd=std::sin(k_tau*p)*step_length*0.5f;
            f32 up=std::max(0.f,std::sin(k_tau*p*2.f))*step_height*(speed_mps>0.1f?1.f:0.f);
            f32 side=std::cos(k_tau*p)*0.05f;
            return {side,up,fwd};
        };
        Vec3f lf=foot(phase);
        Vec3f rf=foot(std::fmod(phase+0.5f,1.f));
        // Body bob (up twice per stride)
        f32 bob=std::abs(std::sin(k_tau*phase))*body_bob_amp;
        f32 sway=std::sin(k_tau*phase)*body_sway_amp;
        return {lf,rf,{sway,bob,0.f},
            std::sin(k_tau*phase)*arm_swing_amp,
            -std::sin(k_tau*phase)*arm_swing_amp};
    }
};

// ─── Look-at (procedural head/eye) ───────────────────────────────────────────

struct LookAtController {
    Vec3f current_dir{0,0,1};
    Vec3f target_dir{0,0,1};
    f32   turn_speed{3.f};   // rad/s
    f32   max_horizontal{k_pi*0.5f};
    f32   max_vertical{k_pi*0.3f};

    Quatf update(f32 dt, const Vec3f& world_look_target, const Quatf& head_rot) {
        Vec3f desired=normalize(world_look_target);
        target_dir=desired;
        // Slerp current toward target
        f32 angle=std::acos(std::clamp(dot(current_dir,target_dir),-1.f,1.f));
        f32 max_turn=turn_speed*dt;
        if(angle>max_turn) {
            current_dir=normalize(current_dir+normalize(target_dir-current_dir*dot(current_dir,target_dir))*max_turn);
        } else {
            current_dir=target_dir;
        }
        // Clamp to limits
        // Build rotation from forward to current_dir
        Vec3f forward{0,0,1};
        Vec3f axis=cross(forward,current_dir);
        f32 ax_len=length(axis);
        if(ax_len>1e-6f) {
            Quatf q=Quatf::from_axis_angle(axis/ax_len, std::acos(std::clamp(dot(forward,current_dir),-1.f,1.f)));
            return head_rot*q;
        }
        return head_rot;
    }
};

// ─── Secondary physics for cloth/hair strand ─────────────────────────────────

struct HairStrand {
    std::vector<Vec3f> points;
    std::vector<Vec3f> velocities;
    f32 segment_length{0.05f};
    f32 stiffness{30.f};
    f32 damping{3.f};
    f32 gravity{9.81f};
    u32 segments{8};

    void init(const Vec3f& root, const Vec3f& dir) {
        points.resize(segments+1);
        velocities.assign(segments+1,{});
        for(u32 i=0;i<=segments;++i)
            points[i]=root+dir*(f32(i)*segment_length);
    }

    void update(f32 dt, const Vec3f& root_pos, const Vec3f& wind={}) {
        points[0]=root_pos;
        velocities[0]={};
        for(u32 i=1;i<=segments;++i){
            Vec3f prev=points[i-1];
            Vec3f f=Vec3f{0,-gravity,0}+wind;
            // Constraint pull toward previous segment rest length
            Vec3f d=points[i]-prev;
            f32 l=length(d);
            if(l>1e-6f) f=f+(d/l)*(-(l-segment_length)*stiffness)-(velocities[i]*damping);
            velocities[i]=velocities[i]+f*dt;
            points[i]=points[i]+velocities[i]*dt;
            // Enforce segment length constraint
            Vec3f dir2=points[i]-prev;
            f32 dl=length(dir2);
            if(dl>1e-6f) points[i]=prev+dir2/dl*segment_length;
        }
    }
};

// ─── Camera follow math ───────────────────────────────────────────────────────

struct CameraFollowState {
    Vec3f smooth_pos{};
    Vec3f smooth_vel{};
    f32   follow_stiffness{5.f};
    f32   follow_damping{2.f};
    f32   lag_distance{1.5f};    // max lag before catching up
    bool  lazy_follow{true};

    Vec3f update(f32 dt, const Vec3f& target_pos) {
        Vec3f desired=target_pos;
        if(lazy_follow){
            Vec3f err=desired-smooth_pos;
            f32 err_len=length(err);
            if(err_len<lag_distance) return smooth_pos;
            desired=target_pos-(err/err_len)*lag_distance;
        }
        Vec3f force=(desired-smooth_pos)*follow_stiffness-smooth_vel*follow_damping;
        smooth_vel=smooth_vel+force*dt;
        smooth_pos=smooth_pos+smooth_vel*dt;
        return smooth_pos;
    }
};

} // namespace qs
