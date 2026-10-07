#pragma once
// qs_math/particle_system_math.hpp — Emitters, forces, attractors, life curves, colliders (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <functional>

namespace qs {

// ─── Particle ─────────────────────────────────────────────────────────────────

struct Particle {
    Vec3f pos{};
    Vec3f vel{};
    Vec4f color{1,1,1,1};   // RGBA
    f32   life{1.f};        // normalized [0,1] remaining
    f32   age{0.f};
    f32   size{1.f};
    f32   rotation{0.f};
    f32   angular_vel{0.f};
    u32   flags{0};
};

// ─── Life curve (normalized age -> value) ─────────────────────────────────────

struct LifeCurve {
    std::vector<std::pair<f32,f32>> keyframes; // (t, value), t in [0,1]

    void add_key(f32 t, f32 v) { keyframes.push_back({t,v}); }

    [[nodiscard]] f32 eval(f32 t) const {
        if (keyframes.empty()) return 1.f;
        if (keyframes.size()==1) return keyframes[0].second;
        for (u32 i=0;i<u32(keyframes.size())-1;++i) {
            auto [t0,v0]=keyframes[i];
            auto [t1,v1]=keyframes[i+1];
            if (t<=t1) {
                f32 frac=(t-t0)/(t1-t0+1e-8f);
                return v0*(1-frac)+v1*frac;
            }
        }
        return keyframes.back().second;
    }

    static LifeCurve linear_fade() {
        LifeCurve c; c.add_key(0.f,1.f); c.add_key(1.f,0.f); return c;
    }
    static LifeCurve constant() {
        LifeCurve c; c.add_key(0.f,1.f); c.add_key(1.f,1.f); return c;
    }
};

// ─── Emitter shapes ───────────────────────────────────────────────────────────

[[nodiscard]] inline Vec3f emit_point(const Vec3f& pos, Pcg32&) { return pos; }

[[nodiscard]] inline Vec3f emit_sphere(const Vec3f& center, f32 r, Pcg32& rng) {
    Vec3f d{rng.next_f32()*2-1,rng.next_f32()*2-1,rng.next_f32()*2-1};
    return center + normalize(d)*r*rng.next_f32();
}

[[nodiscard]] inline Vec3f emit_sphere_surface(const Vec3f& center, f32 r, Pcg32& rng) {
    Vec3f d{rng.next_f32()*2-1,rng.next_f32()*2-1,rng.next_f32()*2-1};
    return center + normalize(d)*r;
}

[[nodiscard]] inline Vec3f emit_box(const Vec3f& mn, const Vec3f& mx, Pcg32& rng) {
    return {mn.x+(mx.x-mn.x)*rng.next_f32(),
            mn.y+(mx.y-mn.y)*rng.next_f32(),
            mn.z+(mx.z-mn.z)*rng.next_f32()};
}

[[nodiscard]] inline Vec3f emit_disk(const Vec3f& center, const Vec3f& normal, f32 r, Pcg32& rng) {
    Vec3f up=(std::abs(normal.y)<0.9f)?Vec3f{0,1,0}:Vec3f{1,0,0};
    Vec3f right=normalize(cross(normal,up));
    Vec3f fwd=cross(right,normal);
    f32 theta=rng.next_f32()*k_tau;
    f32 rad=std::sqrt(rng.next_f32())*r;
    return center+right*(std::cos(theta)*rad)+fwd*(std::sin(theta)*rad);
}

[[nodiscard]] inline Vec3f emit_cone(const Vec3f& apex, const Vec3f& dir, f32 half_angle, f32 len, Pcg32& rng) {
    f32 theta=rng.next_f32()*k_tau;
    f32 phi=rng.next_f32()*half_angle;
    Vec3f up=(std::abs(dir.y)<0.9f)?Vec3f{0,1,0}:Vec3f{1,0,0};
    Vec3f right=normalize(cross(dir,up));
    Vec3f fwd=cross(right,dir);
    Vec3f cone_dir=dir*std::cos(phi)+right*std::cos(theta)*std::sin(phi)+fwd*std::sin(theta)*std::sin(phi);
    return apex+cone_dir*(rng.next_f32()*len);
}

// ─── Initial velocity distributions ──────────────────────────────────────────

[[nodiscard]] inline Vec3f vel_directional(const Vec3f& dir, f32 speed, f32 spread, Pcg32& rng) {
    Vec3f perturb{rng.next_f32()*2-1,rng.next_f32()*2-1,rng.next_f32()*2-1};
    return normalize(dir+perturb*spread)*(speed*(0.8f+rng.next_f32()*0.4f));
}

[[nodiscard]] inline Vec3f vel_radial(const Vec3f& center, const Vec3f& pos, f32 speed, Pcg32& rng) {
    Vec3f d=pos-center; f32 l=length(d);
    Vec3f dir=l>1e-6f?d/l:Vec3f{0,1,0};
    return dir*(speed*(0.8f+rng.next_f32()*0.4f));
}

// ─── Forces ───────────────────────────────────────────────────────────────────

struct GravityForce  { Vec3f acceleration{0,-9.81f,0}; };
struct DragForce     { f32 coefficient{0.1f}; };
struct WindForce     { Vec3f velocity{1,0,0}; f32 turbulence{0.1f}; };
struct VortexForce   { Vec3f axis{0,1,0}; Vec3f center{}; f32 strength{5.f}; f32 inward{0.5f}; };
struct AttractorForce{ Vec3f center{}; f32 strength{10.f}; f32 falloff{2.f}; f32 kill_radius{0.1f}; };
struct NoiseForce    { f32 strength{1.f}; f32 frequency{0.5f}; f32 scroll_speed{0.3f}; };

inline void apply_gravity(Particle& p, const GravityForce& f, f32 dt) {
    p.vel=p.vel+f.acceleration*dt;
}

inline void apply_drag(Particle& p, const DragForce& f, f32 dt) {
    p.vel=p.vel*(1.f-std::min(1.f,f.coefficient*dt));
}

inline void apply_wind(Particle& p, const WindForce& f, f32 dt, Pcg32& rng) {
    Vec3f turb{rng.next_f32()*2-1,rng.next_f32()*2-1,rng.next_f32()*2-1};
    p.vel=p.vel+(f.velocity+turb*f.turbulence)*dt;
}

inline void apply_vortex(Particle& p, const VortexForce& f, f32 dt) {
    Vec3f r=p.pos-f.center;
    Vec3f tangent=cross(f.axis,r); f32 tl=length(tangent);
    if(tl<1e-6f) return;
    tangent=tangent/tl;
    Vec3f inward=-normalize(r-f.axis*dot(r,f.axis));
    Vec3f force=tangent*f.strength+inward*f.strength*f.inward;
    p.vel=p.vel+force*dt;
}

inline void apply_attractor(Particle& p, const AttractorForce& f, f32 dt) {
    Vec3f d=f.center-p.pos;
    f32 dist=length(d);
    if(dist<1e-6f) return;
    f32 falloff=std::pow(dist,f.falloff);
    p.vel=p.vel+(d/dist)*(f.strength/falloff)*dt;
}

// ─── Collision with plane ─────────────────────────────────────────────────────

struct PlaneCollider { Vec3f normal{0,1,0}; f32 d{0}; f32 restitution{0.5f}; f32 friction{0.3f}; };

inline bool collide_plane(Particle& p, const PlaneCollider& col) {
    f32 dist=dot(p.pos,col.normal)-col.d;
    if(dist<0.f) {
        p.pos=p.pos-col.normal*dist;
        f32 vn=dot(p.vel,col.normal);
        if(vn<0.f) {
            p.vel=p.vel-col.normal*vn*(1.f+col.restitution);
            p.vel=p.vel*(1.f-col.friction);
        }
        return true;
    }
    return false;
}

// ─── Color over life ──────────────────────────────────────────────────────────

struct ColorGradient {
    std::vector<std::pair<f32,Vec4f>> stops;
    void add(f32 t, Vec4f c){stops.push_back({t,c});}
    [[nodiscard]] Vec4f eval(f32 t) const {
        if(stops.empty()) return {1,1,1,1};
        if(stops.size()==1) return stops[0].second;
        for(u32 i=0;i<u32(stops.size())-1;++i) {
            auto[t0,c0]=stops[i]; auto[t1,c1]=stops[i+1];
            if(t<=t1){f32 f=(t-t0)/(t1-t0+1e-8f);
                return {c0.x*(1-f)+c1.x*f,c0.y*(1-f)+c1.y*f,c0.z*(1-f)+c1.z*f,c0.w*(1-f)+c1.w*f};}
        }
        return stops.back().second;
    }
};

// ─── Particle system update ───────────────────────────────────────────────────

struct ParticleSystemParams {
    f32 lifetime_min{1.f}, lifetime_max{3.f};
    f32 size_start{1.f},   size_end{0.f};
    f32 speed_min{1.f},    speed_max{5.f};
    u32 max_particles{1000};
    f32 emit_rate{50.f};   // particles per second
    LifeCurve   size_curve   = LifeCurve::linear_fade();
    ColorGradient color_gradient;
};

struct ParticleSystem {
    std::vector<Particle> particles;
    std::vector<f32>      total_lifetimes;
    f32 emit_accumulator{0.f};
    Pcg32 rng{42};

    void update(f32 dt, const ParticleSystemParams& p, auto emit_particle_fn) {
        // Emit
        emit_accumulator+=p.emit_rate*dt;
        while(emit_accumulator>=1.f && particles.size()<p.max_particles) {
            emit_particle_fn(particles.emplace_back(), rng);
            f32 lt=p.lifetime_min+(p.lifetime_max-p.lifetime_min)*rng.next_f32();
            total_lifetimes.push_back(lt);
            particles.back().life=1.f;
            --emit_accumulator;
        }
        // Update
        for(u32 i=0;i<u32(particles.size());) {
            auto& pt=particles[i];
            pt.age+=dt;
            f32 t=1.f-pt.life;
            pt.life=std::max(0.f,pt.life-dt/total_lifetimes[i]);
            pt.pos=pt.pos+pt.vel*dt;
            pt.rotation+=pt.angular_vel*dt;
            pt.size=p.size_start*(1.f-t)+p.size_end*t;
            pt.color=p.color_gradient.eval(1.f-pt.life);
            if(pt.life<=0.f){
                particles[i]=particles.back(); particles.pop_back();
                total_lifetimes[i]=total_lifetimes.back(); total_lifetimes.pop_back();
            } else ++i;
        }
    }
};

// ─── Trail renderer math ──────────────────────────────────────────────────────

struct TrailPoint { Vec3f pos{}; f32 width{}; Vec4f color{}; f32 timestamp{}; };

struct Trail {
    std::vector<TrailPoint> points;
    f32 max_length{2.f};  // seconds
    f32 min_vertex_dist{0.05f};

    void add(const Vec3f& pos, f32 width, Vec4f color, f32 t) {
        if(!points.empty()&&length(pos-points.back().pos)<min_vertex_dist) return;
        points.push_back({pos,width,color,t});
        while(!points.empty()&&t-points.front().timestamp>max_length)
            points.erase(points.begin());
    }

    [[nodiscard]] f32 age_normalized(u32 idx, f32 current_t) const {
        if(points.empty()) return 0.f;
        return 1.f-(current_t-points[idx].timestamp)/max_length;
    }
};

} // namespace qs
