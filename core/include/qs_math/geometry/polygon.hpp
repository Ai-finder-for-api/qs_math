#pragma once
// polygon.hpp — 2D/3D polygon operations: area, centroid, winding,
//               triangulation, point-in-polygon, offsetting, clipping (Sutherland-Hodgman)
#include "vec2.hpp"
#include "vec3.hpp"
#include "geometry2d.hpp"
#include "scalar_math.hpp"
#include <array>

namespace qs::math {

// Signed area of 2D polygon (positive = CCW)
template<FloatScalar T>
[[nodiscard]] constexpr T polygon_signed_area(const Vec2<T>* verts, i32 n) noexcept {
    T area=T(0);
    for(i32 i=0,j=n-1;i<n;j=i++) area+=(verts[j].x+verts[i].x)*(verts[j].y-verts[i].y);
    return area*T(0.5);
}
template<FloatScalar T>
[[nodiscard]] constexpr T polygon_area(const Vec2<T>* v, i32 n) noexcept { return abs(polygon_signed_area(v,n)); }
template<FloatScalar T>
[[nodiscard]] constexpr bool polygon_is_ccw(const Vec2<T>* v, i32 n) noexcept { return polygon_signed_area(v,n)>T(0); }

// Centroid of polygon
template<FloatScalar T>
[[nodiscard]] constexpr Vec2<T> polygon_centroid(const Vec2<T>* v, i32 n) noexcept {
    T area=T(0); Vec2<T> c{T(0)};
    for(i32 i=0,j=n-1;i<n;j=i++){
        T cross_=v[j].x*v[i].y-v[i].x*v[j].y;
        area+=cross_;
        c.x+=(v[j].x+v[i].x)*cross_;
        c.y+=(v[j].y+v[i].y)*cross_;
    }
    area*=T(0.5); T inv=T(1)/(T(6)*area);
    return {c.x*inv, c.y*inv};
}

// Perimeter of polygon
template<FloatScalar T>
[[nodiscard]] inline T polygon_perimeter(const Vec2<T>* v, i32 n) noexcept {
    T p=T(0);
    for(i32 i=0,j=n-1;i<n;j=i++) p+=(v[i]-v[j]).length();
    return p;
}

// Point-in-polygon (ray casting algorithm)
template<FloatScalar T>
[[nodiscard]] constexpr bool point_in_polygon(Vec2<T> p, const Vec2<T>* v, i32 n) noexcept {
    bool inside=false;
    for(i32 i=0,j=n-1;i<n;j=i++){
        if(((v[i].y>p.y)!=(v[j].y>p.y)) &&
           (p.x<(v[j].x-v[i].x)*(p.y-v[i].y)/(v[j].y-v[i].y)+v[i].x))
            inside=!inside;
    }
    return inside;
}

// Winding number (more robust than ray casting for degenerate cases)
template<FloatScalar T>
[[nodiscard]] constexpr i32 winding_number(Vec2<T> p, const Vec2<T>* v, i32 n) noexcept {
    i32 wn=0;
    for(i32 i=0,j=n-1;i<n;j=i++){
        if(v[j].y<=p.y){
            if(v[i].y>p.y&&cross(v[i]-v[j],p-v[j])>T(0)) ++wn;
        } else {
            if(v[i].y<=p.y&&cross(v[i]-v[j],p-v[j])<T(0)) --wn;
        }
    }
    return wn;
}

// Simple fan triangulation (concave polygons not supported)
// Returns triangle count; tris[] must have 3*(n-2) elements
template<FloatScalar T>
[[nodiscard]] constexpr i32 triangulate_fan(const Vec2<T>* v, i32 n, Vec2<T>* tris) noexcept {
    if(n<3) return 0;
    i32 count=0;
    for(i32 i=1;i<n-1;++i){
        tris[count*3+0]=v[0];
        tris[count*3+1]=v[i];
        tris[count*3+2]=v[i+1];
        ++count;
    }
    return count;
}

// Ear-clipping triangulation (handles simple concave polygons)
// Returns number of triangles; out must hold 3*(n-2) Vec2 elements
template<FloatScalar T>
i32 triangulate_ear_clip(Vec2<T>* v_in, i32 n, Vec2<T>* out) noexcept {
    if(n<3) return 0;
    // Copy to local array (n <= 64 assumed)
    constexpr i32 MAX_POLY=256;
    Vec2<T> poly[MAX_POLY]; i32 idx[MAX_POLY];
    i32 sz=n<MAX_POLY?n:MAX_POLY;
    for(i32 i=0;i<sz;++i){poly[i]=v_in[i]; idx[i]=i;}
    i32 tri_count=0;
    while(sz>3){
        bool found=false;
        for(i32 i=0;i<sz;++i){
            i32 a=(i+sz-1)%sz, b=i, c=(i+1)%sz;
            // Is ear?
            if(cross(poly[b]-poly[a],poly[c]-poly[a])<=T(0)) continue;
            bool is_ear=true;
            for(i32 j=0;j<sz&&is_ear;++j){
                if(j==a||j==b||j==c) continue;
                Vec3<T> bary{T(0)};
                // Simple inside test
                T d1=cross(poly[b]-poly[a],poly[j]-poly[a]);
                T d2=cross(poly[c]-poly[b],poly[j]-poly[b]);
                T d3=cross(poly[a]-poly[c],poly[j]-poly[c]);
                bool neg=(d1<T(0))||(d2<T(0))||(d3<T(0));
                bool pos=(d1>T(0))||(d2>T(0))||(d3>T(0));
                if(!(neg&&pos)) is_ear=false;
            }
            if(is_ear){
                out[tri_count*3+0]=poly[a];
                out[tri_count*3+1]=poly[b];
                out[tri_count*3+2]=poly[c];
                ++tri_count;
                for(i32 j=b;j<sz-1;++j) poly[j]=poly[j+1];
                --sz; found=true; break;
            }
        }
        if(!found) break; // degenerate
    }
    if(sz==3){
        out[tri_count*3+0]=poly[0]; out[tri_count*3+1]=poly[1]; out[tri_count*3+2]=poly[2];
        ++tri_count;
    }
    return tri_count;
}

// Polygon offsetting (Minkowski sum with disk, approximate)
template<FloatScalar T>
inline void polygon_offset(const Vec2<T>* v, i32 n, T offset, Vec2<T>* out) noexcept {
    for(i32 i=0;i<n;++i){
        i32 prev=(i+n-1)%n, next=(i+1)%n;
        Vec2<T> a=v[i]-v[prev], b=v[next]-v[i];
        Vec2<T> na{-a.y,a.x}; na=na*(T(1)/na.length());
        Vec2<T> nb{-b.y,b.x}; nb=nb*(T(1)/nb.length());
        Vec2<T> n2=(na+nb).normalized();
        T scale=T(1)/max(dot(n2,na),T(EPS_F32));
        out[i]=v[i]+n2*(offset*scale);
    }
}

// Sutherland-Hodgman polygon clipping against axis-aligned half-plane
// ax+by <= c
template<FloatScalar T, i32 MAX_OUT=64>
i32 clip_polygon_halfplane(const Vec2<T>* in, i32 n, T a, T b, T c, Vec2<T>* out) noexcept {
    i32 out_n=0;
    for(i32 i=0;i<n;++i){
        Vec2<T> curr=in[i], prev=in[(i+n-1)%n];
        T d_curr=a*curr.x+b*curr.y-c;
        T d_prev=a*prev.x+b*prev.y-c;
        if(d_prev<=T(0)) { if(out_n<MAX_OUT) out[out_n++]=prev; }
        if((d_curr<T(0))!=(d_prev<T(0))){
            T t=d_prev/(d_prev-d_curr);
            if(out_n<MAX_OUT) out[out_n++]=lerp(prev,curr,t);
        }
    }
    return out_n;
}

// Sutherland-Hodgman against AABB
template<FloatScalar T, i32 MAX_OUT=64>
i32 clip_polygon_aabb(const Vec2<T>* in, i32 n, Rect2<T> box, Vec2<T>* out, Vec2<T>* tmp) noexcept {
    i32 sz=n;
    sz=clip_polygon_halfplane(in,sz,T(1),T(0),box.max_pt.x,out);
    sz=clip_polygon_halfplane(out,sz,T(-1),T(0),-box.min_pt.x,tmp);
    sz=clip_polygon_halfplane(tmp,sz,T(0),T(1),box.max_pt.y,out);
    sz=clip_polygon_halfplane(out,sz,T(0),T(-1),-box.min_pt.y,tmp);
    for(i32 i=0;i<sz;++i) out[i]=tmp[i];
    return sz;
}

// 3D polygon normal & area
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> polygon3d_normal(const Vec3<T>* v, i32 n) noexcept {
    Vec3<T> normal{T(0)};
    for(i32 i=1;i<n-1;++i) normal=normal+cross(v[i]-v[0],v[i+1]-v[0]);
    return normal.normalized();
}

template<FloatScalar T>
[[nodiscard]] inline T polygon3d_area(const Vec3<T>* v, i32 n) noexcept {
    Vec3<T> sum{T(0)};
    for(i32 i=1;i<n-1;++i) sum=sum+cross(v[i]-v[0],v[i+1]-v[0]);
    return sum.length()*T(0.5);
}

} // namespace qs::math
