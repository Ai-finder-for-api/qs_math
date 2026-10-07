#pragma once
// qs_math/l_systems.hpp — L-system grammar, turtle geometry, fractal plants (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <stack>
#include <array>
#include <functional>

namespace qs {

// ─── L-system grammar ────────────────────────────────────────────────────────

struct LSystemRule {
    char predecessor{};
    std::string successor{};
    f32 probability{1.f}; // for stochastic L-systems
};

struct LSystem {
    std::string axiom{};
    std::vector<LSystemRule> rules;
    u32 iterations{4};

    // Add a deterministic rule
    void add_rule(char pred, const std::string& succ) {
        rules.push_back({pred, succ, 1.f});
    }

    // Add stochastic rule (multiple successors with probabilities, must sum to 1)
    void add_stochastic_rule(char pred, const std::string& succ, f32 prob) {
        rules.push_back({pred, succ, prob});
    }

    // Generate string for n iterations
    [[nodiscard]] std::string generate(u32 n=0) const {
        u32 iters = (n==0) ? iterations : n;
        std::string current = axiom;
        for (u32 i=0;i<iters;++i) {
            std::string next;
            next.reserve(current.size()*4);
            for (char c : current) {
                bool replaced = false;
                for (const auto& rule : rules) {
                    if (rule.predecessor == c) {
                        next += rule.successor;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) next += c;
            }
            current = std::move(next);
        }
        return current;
    }
};

// ─── Turtle state ─────────────────────────────────────────────────────────────

struct TurtleState {
    Vec3f pos{0,0,0};
    Vec3f heading{0,1,0};     // H
    Vec3f left{-1,0,0};       // L
    Vec3f up{0,0,1};          // U
    f32   step{1.f};
    f32   angle{25.7f};       // degrees, typical for plants
    f32   width{0.1f};
    u32   depth{0};
};

struct TurtleSegment {
    Vec3f start{}, end{};
    f32 width_start{}, width_end{};
    u32 depth{};
    bool is_leaf{};
};

// ─── 3D Turtle interpreter ────────────────────────────────────────────────────

struct Turtle3D {
    TurtleState state;
    std::stack<TurtleState> stack;
    std::vector<TurtleSegment> segments;
    std::vector<Vec3f> leaf_positions;

    void reset(f32 step=1.f, f32 angle_deg=25.7f) {
        while(!stack.empty()) stack.pop();
        segments.clear(); leaf_positions.clear();
        state = {};
        state.step = step; state.angle = angle_deg;
    }

    // Rotate around Up axis (yaw)
    void yaw(f32 deg) {
        f32 r = deg * k_deg_to_rad;
        f32 c=std::cos(r), s=std::sin(r);
        Vec3f h = state.heading, l = state.left;
        state.heading = h*c + l*s;
        state.left    = h*(-s) + l*c;
        state.heading = normalize(state.heading);
        state.left    = normalize(state.left);
    }

    // Rotate around Left axis (pitch)
    void pitch(f32 deg) {
        f32 r = deg * k_deg_to_rad;
        f32 c=std::cos(r), s=std::sin(r);
        Vec3f h=state.heading, u=state.up;
        state.heading = h*c + u*s;
        state.up      = h*(-s) + u*c;
        state.heading = normalize(state.heading);
        state.up      = normalize(state.up);
    }

    // Rotate around Heading axis (roll)
    void roll(f32 deg) {
        f32 r = deg * k_deg_to_rad;
        f32 c=std::cos(r), s=std::sin(r);
        Vec3f l=state.left, u=state.up;
        state.left = l*c + u*s;
        state.up   = l*(-s) + u*c;
        state.left = normalize(state.left);
        state.up   = normalize(state.up);
    }

    void forward(bool draw=true) {
        Vec3f next = state.pos + state.heading * state.step;
        if (draw) {
            segments.push_back({state.pos, next, state.width,
                                  state.width*0.9f, state.depth, false});
        }
        state.pos = next;
        ++state.depth;
    }

    void interpret(const std::string& lstr) {
        for (char c : lstr) {
            switch(c) {
            case 'F': forward(true);  break;
            case 'f': forward(false); break;
            case 'G': forward(true);  break;
            case '+': yaw( state.angle);   break;
            case '-': yaw(-state.angle);   break;
            case '^': pitch( state.angle); break;
            case '&': pitch(-state.angle); break;
            case '<': roll( state.angle);  break;
            case '>': roll(-state.angle);  break;
            case '|': yaw(180.f);          break;
            case '[': stack.push(state);   break;
            case ']':
                if (!stack.empty()) {
                    state = stack.top(); stack.pop();
                }
                break;
            case 'L': leaf_positions.push_back(state.pos); break;
            case '!': state.width *= 0.7f;  break;
            case '"': state.step  *= 0.9f;  break;
            default: break;
            }
        }
    }
};

// ─── Preset L-systems ─────────────────────────────────────────────────────────

[[nodiscard]] inline LSystem make_dragon_curve() {
    LSystem ls; ls.axiom="FX"; ls.iterations=10;
    ls.add_rule('X',"X+YF+");
    ls.add_rule('Y',"-FX-Y");
    return ls;
}

[[nodiscard]] inline LSystem make_hilbert_3d() {
    LSystem ls; ls.axiom="A"; ls.iterations=3;
    ls.add_rule('A',"B-F+CFC+F-D&F^D-F+&&CFC+F+B//");
    ls.add_rule('B',"A&F^CFB^F^D^^-F-D^|F^B|FC^F^A//");
    ls.add_rule('C',"|D^|F^B-F+C^F^A&&FA&F^C+F+B^F^D//");
    ls.add_rule('D',"|CFB-F+B|FA&F^A&&FB-F+B|FC//");
    return ls;
}

[[nodiscard]] inline LSystem make_plant_a() {
    // Typical plant (Prusinkiewicz book, p.25)
    LSystem ls; ls.axiom="X"; ls.iterations=5;
    ls.add_rule('X',"F+[[X]-X]-F[-FX]+X");
    ls.add_rule('F',"FF");
    return ls;
}

[[nodiscard]] inline LSystem make_plant_b() {
    LSystem ls; ls.axiom="F"; ls.iterations=5;
    ls.add_rule('F',"FF-[-F+F+F]+[+F-F-F]");
    return ls;
}

[[nodiscard]] inline LSystem make_bush() {
    LSystem ls; ls.axiom="Y"; ls.iterations=4;
    ls.add_rule('Y',"YFX[+Y][-Y]");
    ls.add_rule('X',"X[-FFF][+FFF]FX");
    return ls;
}

[[nodiscard]] inline LSystem make_sierpinski_triangle() {
    LSystem ls; ls.axiom="F-G-G"; ls.iterations=6;
    ls.add_rule('F',"F-G+F+G-F");
    ls.add_rule('G',"GG");
    return ls;
}

[[nodiscard]] inline LSystem make_barnsley_fern() {
    LSystem ls; ls.axiom="X"; ls.iterations=7;
    ls.add_rule('X',"F+[[X]-X]-F[-FX]+X");
    ls.add_rule('F',"FF");
    return ls;
}

// ─── Fractal dimension (box counting) ────────────────────────────────────────

[[nodiscard]] inline f32 box_counting_dimension(
    const std::vector<Vec3f>& points,
    u32 num_scales = 8)
{
    if (points.empty()) return 0.f;
    // Find bounding box
    Vec3f mn=points[0], mx=points[0];
    for (const auto& p : points) {
        mn.x=std::min(mn.x,p.x); mn.y=std::min(mn.y,p.y); mn.z=std::min(mn.z,p.z);
        mx.x=std::max(mx.x,p.x); mx.y=std::max(mx.y,p.y); mx.z=std::max(mx.z,p.z);
    }
    f32 max_range = std::max({mx.x-mn.x, mx.y-mn.y, mx.z-mn.z});
    if (max_range < 1e-8f) return 0.f;

    std::vector<f32> log_N, log_inv_eps;
    for (u32 s=0;s<num_scales;++s) {
        f32 eps = max_range / f32(1<<(s+1));
        std::unordered_map<u64,bool> boxes;
        for (const auto& p : points) {
            i32 bx=i32((p.x-mn.x)/eps), by=i32((p.y-mn.y)/eps), bz=i32((p.z-mn.z)/eps);
            u64 key = u64(bx)*1000003ULL + u64(by)*1000033ULL + u64(bz);
            boxes[key] = true;
        }
        log_N.push_back(std::log(f32(boxes.size())));
        log_inv_eps.push_back(std::log(1.f/eps));
    }
    // Linear regression slope = fractal dimension
    f32 n=f32(num_scales), sx=0,sy=0,sxy=0,sxx=0;
    for (u32 i=0;i<num_scales;++i){sx+=log_inv_eps[i];sy+=log_N[i];sxy+=log_inv_eps[i]*log_N[i];sxx+=log_inv_eps[i]*log_inv_eps[i];}
    return (n*sxy-sx*sy)/(n*sxx-sx*sx);
}

} // namespace qs
