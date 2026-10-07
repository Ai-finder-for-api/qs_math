#pragma once
// forge_visuals_math.hpp — Mathematics for Forge:Visuals node graph system:
//   block layout, graph routing, bezier edge curves, port math,
//   snap-to-grid, connection validation math, viewport transforms,
//   minimap math, selection rect, node clustering
#include "vec2.hpp"
#include "vec4.hpp"
#include "scalar_math.hpp"
#include "curves.hpp"
#include "geometry2d.hpp"

namespace qs::math::forge_visuals {

// ---------------------------------------------------------------------------
// Node / Block layout
// ---------------------------------------------------------------------------
struct NodeRect {
    Vec2f pos;    // top-left
    Vec2f size;

    [[nodiscard]] constexpr Vec2f center() const noexcept { return pos+size*0.5f; }
    [[nodiscard]] constexpr Rect2f to_rect() const noexcept { return {pos,pos+size}; }
    [[nodiscard]] constexpr bool contains(Vec2f p) const noexcept { return to_rect().contains(p); }
    [[nodiscard]] constexpr bool overlaps(NodeRect o) const noexcept { return to_rect().intersects(o.to_rect()); }

    // Port positions (left side = inputs, right side = outputs)
    [[nodiscard]] constexpr Vec2f input_port(i32 idx, i32 count, f32 header_h=24.f, f32 port_h=20.f) const noexcept {
        f32 y=pos.y+header_h+port_h*idx+port_h*0.5f;
        return {pos.x, y};
    }
    [[nodiscard]] constexpr Vec2f output_port(i32 idx, i32 count, f32 header_h=24.f, f32 port_h=20.f) const noexcept {
        f32 y=pos.y+header_h+port_h*idx+port_h*0.5f;
        return {pos.x+size.x, y};
    }
};

// ---------------------------------------------------------------------------
// Connection / Edge curves
// ---------------------------------------------------------------------------
// Cubic Bezier between two ports with automatic tangents
[[nodiscard]] constexpr Vec2f edge_ctrl_from(Vec2f from, f32 tension=100.f) noexcept {
    return from+Vec2f{tension,0.f};
}
[[nodiscard]] constexpr Vec2f edge_ctrl_to(Vec2f to, f32 tension=100.f) noexcept {
    return to-Vec2f{tension,0.f};
}
[[nodiscard]] inline Vec2f sample_edge(Vec2f from, Vec2f to, f32 t, f32 tension=100.f) noexcept {
    Vec2f c1=edge_ctrl_from(from,tension);
    Vec2f c2=edge_ctrl_to(to,tension);
    return bezier_cubic(from,c1,c2,to,t);
}

// Adaptive tension based on node distance
[[nodiscard]] constexpr f32 adaptive_tension(Vec2f from, Vec2f to) noexcept {
    return clamp((to-from).length()*0.4f,50.f,250.f);
}

// Edge tangent direction at t=0 and t=1
[[nodiscard]] constexpr Vec2f edge_tangent(Vec2f from, Vec2f to, f32 t, f32 tension=100.f) noexcept {
    Vec2f c1=edge_ctrl_from(from,tension), c2=edge_ctrl_to(to,tension);
    return bezier_cubic_deriv(from,c1,c2,to,t).normalized();
}

// Closest point on edge to a test point (for hit testing)
[[nodiscard]] inline f32 edge_closest_t(Vec2f from, Vec2f to, Vec2f p, i32 samples=20) noexcept {
    f32 best_t=0.f, best_d=F32_INF;
    f32 tension=adaptive_tension(from,to);
    for(i32 i=0;i<=samples;++i){
        f32 t=f32(i)/samples;
        Vec2f ep=sample_edge(from,to,t,tension);
        f32 d=(ep-p).length_sq();
        if(d<best_d){ best_d=d; best_t=t; }
    }
    return best_t;
}

[[nodiscard]] inline f32 edge_point_distance(Vec2f from, Vec2f to, Vec2f p, i32 samples=20) noexcept {
    f32 t=edge_closest_t(from,to,p,samples);
    f32 tension=adaptive_tension(from,to);
    return (sample_edge(from,to,t,tension)-p).length();
}

// ---------------------------------------------------------------------------
// Grid snapping
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Vec2f snap_to_grid(Vec2f pos, f32 grid_size) noexcept {
    return {round(pos.x/grid_size)*grid_size, round(pos.y/grid_size)*grid_size};
}
[[nodiscard]] constexpr Vec2f snap_to_grid(Vec2f pos, Vec2f grid_size) noexcept {
    return {round(pos.x/grid_size.x)*grid_size.x, round(pos.y/grid_size.y)*grid_size.y};
}
[[nodiscard]] constexpr bool is_on_grid(Vec2f pos, f32 grid_size, f32 eps=0.5f) noexcept {
    Vec2f snapped=snap_to_grid(pos,grid_size);
    return (pos-snapped).length()<eps;
}

// ---------------------------------------------------------------------------
// Viewport transform
// ---------------------------------------------------------------------------
struct GraphViewport {
    Vec2f pan{0.f};     // world origin in screen space
    f32   zoom{1.f};    // world units per pixel (1=100%)
    Vec2f screen_size{1280.f,720.f};

    [[nodiscard]] constexpr Vec2f world_to_screen(Vec2f world) const noexcept {
        return (world-pan)*zoom+screen_size*0.5f;
    }
    [[nodiscard]] constexpr Vec2f screen_to_world(Vec2f screen) const noexcept {
        return (screen-screen_size*0.5f)/zoom+pan;
    }
    [[nodiscard]] constexpr f32 world_to_screen_scale(f32 ws) const noexcept { return ws*zoom; }
    [[nodiscard]] constexpr f32 screen_to_world_scale(f32 ss) const noexcept { return ss/zoom; }
    [[nodiscard]] constexpr Rect2f visible_world_rect() const noexcept {
        Vec2f tl=screen_to_world({0.f,0.f});
        Vec2f br=screen_to_world(screen_size);
        return {tl,br};
    }
    void zoom_at(Vec2f screen_focus, f32 factor) noexcept {
        Vec2f world_focus=screen_to_world(screen_focus);
        zoom*=factor;
        zoom=clamp(zoom,0.05f,20.f);
        pan=world_focus-(screen_focus-screen_size*0.5f)/zoom;
    }
};

// ---------------------------------------------------------------------------
// Minimap math
// ---------------------------------------------------------------------------
struct MinimapTransform {
    Vec2f world_min, world_max;
    Vec2f minimap_pos, minimap_size;

    [[nodiscard]] Vec2f world_to_minimap(Vec2f world) const noexcept {
        Vec2f world_range=world_max-world_min;
        if(world_range.x<EPS_F32||world_range.y<EPS_F32) return minimap_pos;
        Vec2f normalized=(world-world_min)/world_range;
        return minimap_pos+normalized*minimap_size;
    }
    [[nodiscard]] Vec2f minimap_to_world(Vec2f mm) const noexcept {
        Vec2f normalized=(mm-minimap_pos)/minimap_size;
        return world_min+normalized*(world_max-world_min);
    }
    void fit_to_nodes(const NodeRect* nodes, i32 n) noexcept {
        if(n==0) return;
        world_min=world_max=nodes[0].pos;
        for(i32 i=0;i<n;++i){
            world_min=world_min.min(nodes[i].pos);
            world_max=world_max.max(nodes[i].pos+nodes[i].size);
        }
        // Add padding
        Vec2f pad=(world_max-world_min)*0.1f+Vec2f{50.f};
        world_min=world_min-pad; world_max=world_max+pad;
    }
};

// ---------------------------------------------------------------------------
// Selection rectangle math
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr Rect2f selection_rect(Vec2f a, Vec2f b) noexcept {
    return {a.min(b), a.max(b)};
}

inline i32 nodes_in_selection(
    const NodeRect* nodes, i32 n, Rect2f sel, i32* selected, i32 max_sel) noexcept
{
    i32 count=0;
    for(i32 i=0;i<n&&count<max_sel;++i)
        if(sel.intersects(nodes[i].to_rect())) selected[count++]=i;
    return count;
}

// ---------------------------------------------------------------------------
// Node auto-layout (force-directed)
// ---------------------------------------------------------------------------
struct ForceLayout {
    static constexpr f32 K_REPEL=5000.f;
    static constexpr f32 K_ATTRACT=0.1f;
    static constexpr f32 DAMPING=0.8f;
    static constexpr f32 MIN_DIST=10.f;

    Vec2f vel[512]{};

    void step(Vec2f* pos, const i32* edges, i32 n_edges, i32 n_nodes, f32 dt=0.016f) noexcept {
        Vec2f forces[512]={};
        // Repulsive forces between all pairs
        for(i32 i=0;i<n_nodes;++i) for(i32 j=i+1;j<n_nodes;++j){
            Vec2f d=pos[i]-pos[j];
            f32 dist=max(d.length(),MIN_DIST);
            Vec2f f=d.normalized()*(K_REPEL/(dist*dist));
            forces[i]=forces[i]+f;
            forces[j]=forces[j]-f;
        }
        // Attractive forces along edges
        for(i32 e=0;e<n_edges;++e){
            i32 a=edges[e*2], b=edges[e*2+1];
            Vec2f d=pos[b]-pos[a];
            f32 dist=max(d.length(),MIN_DIST);
            Vec2f f=d*(K_ATTRACT*dist);
            forces[a]=forces[a]+f; forces[b]=forces[b]-f;
        }
        // Integrate
        for(i32 i=0;i<n_nodes;++i){
            vel[i]=(vel[i]+forces[i]*dt)*DAMPING;
            pos[i]=pos[i]+vel[i]*dt;
        }
    }
};

// ---------------------------------------------------------------------------
// Type compatibility math (for port validation)
// ---------------------------------------------------------------------------
// Numeric type IDs for graph port types
enum class PortTypeClass { Numeric, Vector, Matrix, Quaternion, Color, Boolean, String, Asset, Custom };

[[nodiscard]] constexpr bool types_compatible(PortTypeClass from, PortTypeClass to) noexcept {
    if(from==to) return true;
    // Numeric can connect to Vector (scalar promotion)
    if(from==PortTypeClass::Numeric&&to==PortTypeClass::Vector) return true;
    // Color connects to Vector (RGB = Vec3)
    if(from==PortTypeClass::Color&&to==PortTypeClass::Vector) return true;
    if(from==PortTypeClass::Vector&&to==PortTypeClass::Color) return true;
    return false;
}

// Implicit cast cost (lower = preferred)
[[nodiscard]] constexpr f32 cast_cost(PortTypeClass from, PortTypeClass to) noexcept {
    if(from==to) return 0.f;
    if(types_compatible(from,to)) return 1.f;
    return F32_INF;
}

} // namespace qs::math::forge_visuals
