#pragma once
// qs_math/radiosity_math.hpp — Form factors, hemicube, progressive radiosity, irradiance (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>
#include <array>

namespace qs {

// ─── Radiosity patch ─────────────────────────────────────────────────────────

struct RadiosityPatch {
    Vec3f center{};
    Vec3f normal{};
    f32   area{};
    Vec3f reflectivity{1,1,1};   // diffuse reflectance per channel
    Vec3f emission{0,0,0};       // self-emission (light sources)
    Vec3f radiosity{0,0,0};      // current radiosity
    Vec3f irradiance{0,0,0};     // received irradiance
    u32   id{};
};

// ─── Form factor (analytical for planar patches) ──────────────────────────────

// Point-to-point form factor differential element
[[nodiscard]] inline f32 form_factor_differential(
    const Vec3f& p_i, const Vec3f& n_i,
    const Vec3f& p_j, const Vec3f& n_j,
    f32 visibility=1.f)
{
    Vec3f r=p_j-p_i;
    f32 dist2=dot(r,r);
    if(dist2<1e-10f) return 0.f;
    f32 dist=std::sqrt(dist2);
    Vec3f r_hat=r/dist;
    f32 cos_i=std::max(0.f,dot(n_i,r_hat));
    f32 cos_j=std::max(0.f,-dot(n_j,r_hat));
    return cos_i*cos_j*visibility/(k_pi*dist2);
}

// Nusselt analog form factor from patch i to patch j
[[nodiscard]] inline f32 form_factor_patches(
    const RadiosityPatch& Pi, const RadiosityPatch& Pj,
    f32 visibility=1.f)
{
    f32 dF=form_factor_differential(Pi.center,Pi.normal,Pj.center,Pj.normal,visibility);
    return dF*Pj.area;
}

// ─── Form factor matrix ───────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<f32> compute_form_factor_matrix(
    std::span<const RadiosityPatch> patches,
    auto visibility_fn) // (u32 i, u32 j) -> f32 in [0,1]
{
    u32 N=u32(patches.size());
    std::vector<f32> F(N*N,0.f);
    for(u32 i=0;i<N;++i)
    for(u32 j=0;j<N;++j) {
        if(i==j) continue;
        f32 vis=visibility_fn(i,j);
        F[i*N+j]=form_factor_patches(patches[i],patches[j],vis);
    }
    // Reciprocity normalization: F_ij * Ai = F_ji * Aj
    for(u32 i=0;i<N;++i) {
        f32 row_sum=0.f;
        for(u32 j=0;j<N;++j) row_sum+=F[i*N+j];
        if(row_sum>1.f) for(u32 j=0;j<N;++j) F[i*N+j]/=row_sum;
    }
    return F;
}

// ─── Progressive radiosity solver ────────────────────────────────────────────

// One iteration of progressive radiosity (shooting from most energetic patch)
inline u32 progressive_radiosity_step(
    std::vector<RadiosityPatch>& patches,
    const std::vector<f32>& F_matrix,
    std::vector<Vec3f>& residual)
{
    u32 N=u32(patches.size());
    // Find patch with maximum residual energy
    u32 shooter=0;
    f32 max_energy=0.f;
    for(u32 i=0;i<N;++i){
        f32 e=length(residual[i])*patches[i].area;
        if(e>max_energy){max_energy=e;shooter=i;}
    }

    // Shoot radiosity from shooter to all others
    const Vec3f& shoot=residual[shooter];
    for(u32 j=0;j<N;++j) {
        if(j==shooter) continue;
        f32 fij=F_matrix[shooter*N+j];
        Vec3f contrib={shoot.x*patches[j].reflectivity.x*fij,
                       shoot.y*patches[j].reflectivity.y*fij,
                       shoot.z*patches[j].reflectivity.z*fij};
        patches[j].radiosity=patches[j].radiosity+contrib;
        residual[j]=residual[j]+contrib;
    }
    residual[shooter]={0,0,0};
    return shooter;
}

inline void run_progressive_radiosity(
    std::vector<RadiosityPatch>& patches,
    const std::vector<f32>& F,
    u32 iterations=100)
{
    std::vector<Vec3f> residual(patches.size());
    // Seed with emission
    for(u32 i=0;i<u32(patches.size());++i){
        patches[i].radiosity=patches[i].emission;
        residual[i]=patches[i].emission;
    }
    for(u32 iter=0;iter<iterations;++iter)
        progressive_radiosity_step(patches,F,residual);
}

// ─── Hemicube form factors ────────────────────────────────────────────────────

struct HemicubeWeights {
    u32 resolution{32};
    std::vector<f32> top_weights;    // res*res
    std::vector<f32> side_weights;   // res*(res/2) * 4

    void compute() {
        u32 R=resolution;
        top_weights.resize(R*R);
        side_weights.resize(R*(R/2)*4);
        f32 dA=4.f/f32(R*R); // solid angle element
        for(u32 y=0;y<R;++y)
        for(u32 x=0;x<R;++x) {
            // Map pixel to direction on hemisphere top
            f32 u=(f32(x)+0.5f)/f32(R)*2.f-1.f;
            f32 v=(f32(y)+0.5f)/f32(R)*2.f-1.f;
            f32 len2=u*u+v*v+1.f;
            top_weights[y*R+x]=dA/(k_pi*len2*len2);
        }
        // Side face weights (4 sides, half-height)
        u32 H=R/2;
        for(u32 s=0;s<4;++s)
        for(u32 y=0;y<H;++y)
        for(u32 x=0;x<R;++x) {
            f32 u=(f32(x)+0.5f)/f32(R)*2.f-1.f;
            f32 v=(f32(y)+0.5f)/f32(H);
            f32 len2=u*u+v*v+1.f;
            side_weights[s*H*R+y*R+x]=v*dA*0.5f/(k_pi*len2*len2);
        }
    }
};

// ─── Irradiance probe (spherical harmonics) ───────────────────────────────────

// SH basis functions (order 2, L0..L2) — 9 coefficients
[[nodiscard]] inline std::array<f32,9> sh_basis(const Vec3f& n) {
    return {
        0.282095f,                               // L0
        0.488603f*n.y,                           // L1,-1
        0.488603f*n.z,                           // L1,0
        0.488603f*n.x,                           // L1,1
        1.092548f*n.x*n.y,                       // L2,-2
        1.092548f*n.y*n.z,                       // L2,-1
        0.315392f*(3.f*n.z*n.z-1.f),             // L2,0
        1.092548f*n.x*n.z,                       // L2,1
        0.546274f*(n.x*n.x-n.y*n.y)             // L2,2
    };
}

// Project irradiance onto SH (Monte Carlo integration)
[[nodiscard]] inline std::array<Vec3f,9> project_irradiance_sh(
    std::span<const Vec3f> directions,
    std::span<const Vec3f> radiances)
{
    std::array<Vec3f,9> coeffs{};
    f32 weight=4.f*k_pi/f32(directions.size());
    for(u32 k=0;k<u32(directions.size());++k){
        auto basis=sh_basis(directions[k]);
        for(u32 l=0;l<9;++l)
            coeffs[l]=coeffs[l]+radiances[k]*(basis[l]*weight);
    }
    return coeffs;
}

// Reconstruct irradiance from SH at normal n
[[nodiscard]] inline Vec3f reconstruct_sh_irradiance(
    const std::array<Vec3f,9>& coeffs, const Vec3f& n)
{
    auto basis=sh_basis(n);
    Vec3f irr{0,0,0};
    for(u32 l=0;l<9;++l) irr=irr+coeffs[l]*basis[l];
    return irr * k_pi; // Lambert integral factor
}

// ─── Lightmap texel irradiance ────────────────────────────────────────────────

struct LightmapTexel {
    Vec3f world_pos{};
    Vec3f world_normal{};
    Vec2f uv{};
    Vec3f irradiance{};
    bool  valid{false};
};

[[nodiscard]] inline Vec3f compute_texel_irradiance(
    const LightmapTexel& texel,
    std::span<const RadiosityPatch> patches,
    auto visibility_fn)
{
    Vec3f irr{0,0,0};
    for(const auto& p:patches){
        if(length(p.radiosity)<1e-6f) continue;
        f32 dF=form_factor_differential(texel.world_pos,texel.world_normal,
                                        p.center,-p.normal,visibility_fn(texel.world_pos,p.center));
        irr=irr+p.radiosity*(dF*p.area);
    }
    return irr;
}

} // namespace qs
