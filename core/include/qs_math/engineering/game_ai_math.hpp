#pragma once
// qs_math/game_ai_math.hpp — Behaviour trees, utility AI, GOAP, steering (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec2.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <algorithm>
#include <array>
#include <span>
#include <memory>

namespace qs {

// ─── Behaviour tree status ────────────────────────────────────────────────────

enum class BTStatus { Running, Success, Failure };

struct BTNode {
    virtual ~BTNode() = default;
    virtual BTStatus tick(f32 dt) = 0;
    std::string name{};
};

struct BTSequence : BTNode {
    std::vector<std::shared_ptr<BTNode>> children;
    u32 current{0};
    BTStatus tick(f32 dt) override {
        while(current<children.size()){
            auto s=children[current]->tick(dt);
            if(s==BTStatus::Failure){current=0;return BTStatus::Failure;}
            if(s==BTStatus::Running) return BTStatus::Running;
            ++current;
        }
        current=0; return BTStatus::Success;
    }
};

struct BTSelector : BTNode {
    std::vector<std::shared_ptr<BTNode>> children;
    u32 current{0};
    BTStatus tick(f32 dt) override {
        while(current<children.size()){
            auto s=children[current]->tick(dt);
            if(s==BTStatus::Success){current=0;return BTStatus::Success;}
            if(s==BTStatus::Running) return BTStatus::Running;
            ++current;
        }
        current=0; return BTStatus::Failure;
    }
};

struct BTLeaf : BTNode {
    std::function<BTStatus(f32)> fn;
    BTStatus tick(f32 dt) override { return fn?fn(dt):BTStatus::Failure; }
};

struct BTDecorator : BTNode {
    std::shared_ptr<BTNode> child;
    enum class Type { Invert, AlwaysSucceed, AlwaysFail, Repeat } type{Type::Invert};
    u32 repeat_count{3}, repeats_done{0};
    BTStatus tick(f32 dt) override {
        if(!child) return BTStatus::Failure;
        auto s=child->tick(dt);
        switch(type){
        case Type::Invert:       return s==BTStatus::Success?BTStatus::Failure:BTStatus::Success;
        case Type::AlwaysSucceed:return BTStatus::Success;
        case Type::AlwaysFail:   return BTStatus::Failure;
        case Type::Repeat:
            if(s==BTStatus::Running) return BTStatus::Running;
            ++repeats_done;
            if(repeats_done>=repeat_count){repeats_done=0;return BTStatus::Success;}
            return BTStatus::Running;
        }
        return s;
    }
};

// ─── Utility AI ───────────────────────────────────────────────────────────────

struct UtilityAction {
    std::string name{};
    std::function<f32()> score_fn;    // returns 0..1
    std::function<void(f32)> execute; // called with dt
    f32 score{0.f};
    f32 weight{1.f};
    f32 inertia{0.1f}; // bonus for currently selected action
};

struct UtilityAI {
    std::vector<UtilityAction> actions;
    u32 current_action{0};
    f32 cooldown{0.f};
    f32 decision_interval{0.1f};

    void update(f32 dt) {
        cooldown-=dt;
        if(cooldown<=0.f){
            cooldown=decision_interval;
            for(u32 i=0;i<actions.size();++i){
                actions[i].score=actions[i].score_fn()*actions[i].weight;
                if(i==current_action) actions[i].score+=actions[i].inertia;
            }
            current_action=u32(std::max_element(actions.begin(),actions.end(),
                [](const UtilityAction& a,const UtilityAction& b){return a.score<b.score;})-actions.begin());
        }
        if(current_action<actions.size()&&actions[current_action].execute)
            actions[current_action].execute(dt);
    }
};

// ─── GOAP (Goal-Oriented Action Planning) ────────────────────────────────────

struct WorldState {
    std::vector<std::pair<u32,bool>> facts; // id -> value

    [[nodiscard]] bool get(u32 id) const {
        for(auto[k,v]:facts) if(k==id) return v;
        return false;
    }

    void set(u32 id, bool v){
        for(auto&[k,val]:facts) if(k==id){val=v;return;}
        facts.push_back({id,v});
    }

    [[nodiscard]] bool satisfies(const WorldState& goal) const {
        for(auto[k,v]:goal.facts) if(get(k)!=v) return false;
        return true;
    }

    [[nodiscard]] u32 distance(const WorldState& goal) const {
        u32 d=0;
        for(auto[k,v]:goal.facts) if(get(k)!=v) ++d;
        return d;
    }
};

struct GOAPAction {
    std::string name{};
    WorldState preconditions;
    WorldState effects;
    f32 cost{1.f};
    std::function<bool()> is_achievable;
    std::function<void(f32)> execute;
};

// A* GOAP planner
[[nodiscard]] inline std::vector<u32> goap_plan(
    const WorldState& start, const WorldState& goal,
    std::span<const GOAPAction> actions)
{
    struct Node {
        WorldState state;
        f32 g{0.f},h{0.f};
        std::vector<u32> path;
        f32 f() const { return g+h; }
    };

    std::vector<Node> open;
    open.push_back({start,0.f,f32(start.distance(goal)),{}});

    for(u32 iter=0;iter<200&&!open.empty();++iter){
        // Find lowest f
        u32 best=0;
        for(u32 i=1;i<open.size();++i) if(open[i].f()<open[best].f()) best=i;
        Node cur=open[best];
        open.erase(open.begin()+best);

        if(cur.state.satisfies(goal)) return cur.path;

        for(u32 ai=0;ai<u32(actions.size());++ai){
            const auto& a=actions[ai];
            if(!cur.state.satisfies(a.preconditions)) continue;
            if(a.is_achievable&&!a.is_achievable()) continue;

            WorldState ns=cur.state;
            for(auto[k,v]:a.effects.facts) ns.set(k,v);
            std::vector<u32> np=cur.path;
            np.push_back(ai);
            open.push_back({ns,cur.g+a.cost,f32(ns.distance(goal)),np});
        }
    }
    return {};
}

// ─── Steering behaviours ──────────────────────────────────────────────────────

struct SteeringAgent {
    Vec3f pos{},vel{},desired_vel{};
    f32 max_speed{5.f},max_force{10.f},mass{1.f};

    void apply_force(const Vec3f& f,f32 dt){
        Vec3f acc=f*(1.f/mass);
        vel=vel+acc*dt;
        f32 spd=length(vel);
        if(spd>max_speed) vel=vel/spd*max_speed;
        pos=pos+vel*dt;
    }
};

[[nodiscard]] inline Vec3f seek(const SteeringAgent& a, const Vec3f& target) {
    Vec3f desired=normalize(target-a.pos)*a.max_speed;
    return desired-a.vel;
}

[[nodiscard]] inline Vec3f flee(const SteeringAgent& a, const Vec3f& threat) {
    Vec3f desired=normalize(a.pos-threat)*a.max_speed;
    return desired-a.vel;
}

[[nodiscard]] inline Vec3f arrive(const SteeringAgent& a, const Vec3f& target, f32 slow_r=3.f) {
    Vec3f d=target-a.pos;
    f32 dist=length(d);
    f32 spd=dist<slow_r?a.max_speed*(dist/slow_r):a.max_speed;
    Vec3f desired=dist>1e-6f?d/dist*spd:Vec3f{};
    return desired-a.vel;
}

[[nodiscard]] inline Vec3f wander(SteeringAgent& a, f32& wander_angle, f32 dt,
    f32 wander_r=1.f, f32 wander_dist=3.f, f32 wander_jitter=1.f, Pcg32& rng={})
{
    wander_angle+=(rng.next_f32()*2.f-1.f)*wander_jitter*dt;
    Vec3f vel_dir=length(a.vel)>1e-6f?normalize(a.vel):Vec3f{0,0,1};
    Vec3f up=(std::abs(vel_dir.y)<0.9f)?Vec3f{0,1,0}:Vec3f{1,0,0};
    Vec3f right=normalize(cross(vel_dir,up));
    Vec3f circle_center=a.pos+vel_dir*wander_dist;
    Vec3f wander_pt=circle_center+right*(std::cos(wander_angle)*wander_r)
                                 +up*(std::sin(wander_angle)*wander_r);
    return seek(a,wander_pt);
}

[[nodiscard]] inline Vec3f separation(const SteeringAgent& a,
    std::span<const SteeringAgent> neighbours, f32 desired_sep=2.f)
{
    Vec3f steer{};u32 count=0;
    for(const auto& nb:neighbours){
        f32 d=length(a.pos-nb.pos);
        if(d>0.f&&d<desired_sep){steer=steer+normalize(a.pos-nb.pos)/d;++count;}
    }
    if(count>0){steer=steer*(1.f/f32(count));if(length(steer)>0.f)steer=normalize(steer)*a.max_speed-a.vel;}
    return steer;
}

[[nodiscard]] inline Vec3f alignment(const SteeringAgent& a,
    std::span<const SteeringAgent> neighbours)
{
    Vec3f avg{};u32 count=0;
    for(const auto& nb:neighbours){avg=avg+nb.vel;++count;}
    if(!count) return {};
    avg=avg*(1.f/f32(count));
    return avg-a.vel;
}

[[nodiscard]] inline Vec3f cohesion(const SteeringAgent& a,
    std::span<const SteeringAgent> neighbours)
{
    Vec3f center{};u32 count=0;
    for(const auto& nb:neighbours){center=center+nb.pos;++count;}
    if(!count) return {};
    return seek(a,center*(1.f/f32(count)));
}

// ─── Finite state machine ─────────────────────────────────────────────────────

template<typename StateEnum>
struct FSM {
    StateEnum current{};
    struct Transition {
        StateEnum from,to;
        std::function<bool()> condition;
        std::function<void()> on_exit, on_enter;
    };
    std::vector<Transition> transitions;
    std::unordered_map<u32,std::function<void(f32)>> update_fns;

    void add_transition(StateEnum from,StateEnum to,
        std::function<bool()> cond,
        std::function<void()> exit={},
        std::function<void()> enter={})
    { transitions.push_back({from,to,cond,exit,enter}); }

    void update(f32 dt){
        for(auto&t:transitions){
            if(t.from==current&&t.condition&&t.condition()){
                if(t.on_exit) t.on_exit();
                current=t.to;
                if(t.on_enter) t.on_enter();
                break;
            }
        }
        auto it=update_fns.find(u32(current));
        if(it!=update_fns.end()) it->second(dt);
    }
};

// ─── Perception system ────────────────────────────────────────────────────────

struct PerceptionStimulus {
    Vec3f position{};
    f32   strength{1.f};
    u32   type{0};   // 0=visual, 1=audio, 2=smell
    f32   age{0.f};  // seconds since generated
};

struct PerceptionSensor {
    Vec3f pos{};
    Vec3f facing{0,0,1};
    f32 view_range{15.f};
    f32 view_angle{k_pi*0.5f};  // half-angle
    f32 hearing_range{10.f};

    [[nodiscard]] bool can_see(const Vec3f& target) const {
        Vec3f d=target-pos;
        f32 dist=length(d);
        if(dist>view_range) return false;
        f32 angle=std::acos(std::clamp(dot(d/dist,facing),-1.f,1.f));
        return angle<view_angle;
    }

    [[nodiscard]] bool can_hear(const PerceptionStimulus& s) const {
        return s.type==1&&length(s.position-pos)<hearing_range*s.strength;
    }
};

} // namespace qs
