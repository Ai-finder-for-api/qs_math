#pragma once
// gjk.hpp — GJK (Gilbert-Johnson-Keerthi) collision detection +
//            EPA (Expanding Polytope Algorithm) for penetration depth
#include "geometry3d.hpp"
#include "distance_queries.hpp"
#include <array>

namespace qs::math {

// ---------------------------------------------------------------------------
// GJK Result
// ---------------------------------------------------------------------------
struct GJKResult {
    bool     intersecting{false};
    f32      distance{0.f};     // 0 if intersecting
    Vec3f    closest_a;         // closest point on shape A
    Vec3f    closest_b;         // closest point on shape B
    i32      iterations{0};
};

// Minkowski difference support
inline Vec3f minkowski_support(
    const Vec3f* pts_a, i32 na,
    const Vec3f* pts_b, i32 nb,
    Vec3f dir) noexcept
{
    return support_point_cloud(pts_a,na,dir)
         - support_point_cloud(pts_b,nb,-dir);
}

// Triple product: (a × b) × c = b(a·c) - a(b·c)
inline Vec3f triple(Vec3f a, Vec3f b, Vec3f c) noexcept {
    return b*dot(a,c)-a*dot(b,c);
}

// Simplex for GJK (max 4 points in 3D)
struct Simplex {
    Vec3f pts[4];
    i32 n{0};
    void push(Vec3f p) {
        for(i32 i=n;i>0;--i) pts[i]=pts[i-1];
        pts[0]=p; if(n<4) ++n;
    }
    Vec3f& operator[](i32 i) { return pts[i]; }
    const Vec3f& operator[](i32 i) const { return pts[i]; }
};

// Check if simplex contains origin and find next direction
inline bool do_simplex(Simplex& s, Vec3f& dir) noexcept {
    switch(s.n){
    case 2: {
        Vec3f ab=s[1]-s[0]; Vec3f ao=-s[0];
        if(dot(ab,ao)>0.f) dir=triple(ab,ao,ab);
        else { s.n=1; dir=ao; }
        return false;
    }
    case 3: {
        Vec3f ab=s[1]-s[0],ac=s[2]-s[0],ao=-s[0];
        Vec3f abc=cross(ab,ac);
        if(dot(cross(abc,ac),ao)>0.f){
            if(dot(ac,ao)>0.f){ s.pts[1]=s.pts[2]; s.n=2; dir=triple(ac,ao,ac); }
            else { s.n=2; if(dot(ab,ao)>0.f) dir=triple(ab,ao,ab); else dir=ao; }
        } else if(dot(cross(ab,abc),ao)>0.f){
            s.n=2; if(dot(ab,ao)>0.f) dir=triple(ab,ao,ab); else dir=ao;
        } else if(dot(abc,ao)>0.f) dir=abc;
        else { s.pts[3]=s.pts[0]; s.pts[0]=s.pts[1]; s.pts[1]=s.pts[2]; s.pts[2]=s.pts[3]; s.n=3; dir=-abc; }
        return false;
    }
    case 4: {
        // Tetrahedron case — check if origin is inside
        Vec3f ab=s[1]-s[0],ac=s[2]-s[0],ad=s[3]-s[0],ao=-s[0];
        Vec3f abc=cross(ab,ac),acd=cross(ac,ad),adb=cross(ad,ab);
        if(dot(abc,ao)>0.f){ s.n=3; return do_simplex(s,dir); }
        if(dot(acd,ao)>0.f){ s.pts[1]=s.pts[2]; s.pts[2]=s.pts[3]; s.n=3; return do_simplex(s,dir); }
        if(dot(adb,ao)>0.f){ s.pts[2]=s.pts[1]; s.pts[1]=s.pts[3]; s.n=3; return do_simplex(s,dir); }
        return true; // origin inside tetrahedron
    }
    default: return false;
    }
}

// GJK intersection test (convex point clouds)
[[nodiscard]] inline bool gjk_intersect(
    const Vec3f* pts_a, i32 na,
    const Vec3f* pts_b, i32 nb,
    i32 max_iter=32) noexcept
{
    Simplex s;
    Vec3f dir{1.f,0.f,0.f};
    s.push(minkowski_support(pts_a,na,pts_b,nb,dir));
    dir=-s[0];
    for(i32 i=0;i<max_iter;++i){
        Vec3f a=minkowski_support(pts_a,na,pts_b,nb,dir);
        if(dot(a,dir)<0.f) return false;
        s.push(a);
        if(do_simplex(s,dir)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// EPA (Expanding Polytope Algorithm) for penetration depth
// ---------------------------------------------------------------------------
struct EPAResult {
    Vec3f normal;    // collision normal (A push direction)
    f32   depth{0.f};// penetration depth
    bool  valid{false};
};

// Simplified EPA for convex point clouds
[[nodiscard]] inline EPAResult epa(
    const Vec3f* pts_a, i32 na,
    const Vec3f* pts_b, i32 nb,
    Simplex s,
    i32 max_iter=32) noexcept
{
    // Expand simplex into polytope — using face list (max 64 faces)
    constexpr i32 MAX_F=64;
    struct Face { Vec3f pts[3]; Vec3f normal; f32 dist; };
    Face faces[MAX_F]; i32 nf=0;

    auto add_face=[&](Vec3f a, Vec3f b, Vec3f c){
        if(nf>=MAX_F) return;
        Vec3f n=cross(b-a,c-a); f32 len=n.length();
        if(len<EPS_INTERSECT) return;
        n=n*(1.f/len);
        faces[nf++]={a,b,c,n,dot(n,a)};
    };

    // Build initial faces from tetrahedron
    if(s.n==4){
        add_face(s[0],s[1],s[2]); add_face(s[0],s[1],s[3]);
        add_face(s[0],s[2],s[3]); add_face(s[1],s[2],s[3]);
    } else { return {}; }

    for(i32 iter=0;iter<max_iter&&nf>0;++iter){
        // Find closest face
        i32 cf=0;
        for(i32 i=1;i<nf;++i) if(faces[i].dist<faces[cf].dist) cf=i;
        Face& face=faces[cf];
        Vec3f support=minkowski_support(pts_a,na,pts_b,nb,face.normal);
        f32 d=dot(face.normal,support);
        if(d-face.dist<EPS_INTERSECT){
            return {face.normal,face.dist,true};
        }
        // Expand — remove visible faces and add new ones
        Vec3f p=support; Face new_faces[MAX_F]; i32 nnf=0;
        struct Edge { Vec3f a,b; };
        Edge edges[MAX_F*3]; i32 ne=0;
        for(i32 i=0;i<nf;++i){
            if(dot(faces[i].normal,p-faces[i].pts[0])>0.f){
                // Visible — extract edges
                for(i32 e=0;e<3;++e){
                    Vec3f ea=faces[i].pts[e], eb=faces[i].pts[(e+1)%3];
                    bool shared=false;
                    for(i32 j=0;j<ne;++j){
                        if(nearly_equal(edges[j].a,eb)&&nearly_equal(edges[j].b,ea)){
                            edges[j]=edges[--ne]; shared=true; break;
                        }
                    }
                    if(!shared&&ne<MAX_F*3) edges[ne++]={ea,eb};
                }
            } else {
                if(nnf<MAX_F) new_faces[nnf++]=faces[i];
            }
        }
        for(i32 i=0;i<ne&&nnf<MAX_F;++i){
            Vec3f n=cross(edges[i].b-edges[i].a,p-edges[i].a);
            f32 len=n.length();
            if(len<EPS_INTERSECT) continue;
            n=n*(1.f/len);
            new_faces[nnf++]={edges[i].a,edges[i].b,p,n,dot(n,edges[i].a)};
        }
        for(i32 i=0;i<nnf;++i) faces[i]=new_faces[i];
        nf=nnf;
    }
    return {};
}

} // namespace qs::math
