#pragma once
// qs_math/rendering/shading_math.hpp — Irradiance, SH lighting, IBL, LTC, hair/fur BSDF
#include "../arithmetic/scalar_types.hpp"
#include "pbr_math.hpp"
#include <cmath>
#include <array>

namespace qs::math {

// ─── Spherical harmonics (real, ZH then full) ─────────────────────────────────
// SH basis functions Y_l^m(θ,φ) — band 0,1,2 (9 coefficients)
[[nodiscard]] constexpr f32 sh_Y00()                         noexcept {return  0.28209479177387814f;}
[[nodiscard]] constexpr f32 sh_Y1m1(f32 y)                  noexcept {return -0.48860251190291987f*y;}
[[nodiscard]] constexpr f32 sh_Y10 (f32 z)                  noexcept {return  0.48860251190291987f*z;}
[[nodiscard]] constexpr f32 sh_Y1p1(f32 x)                  noexcept {return -0.48860251190291987f*x;}
[[nodiscard]] constexpr f32 sh_Y2m2(f32 x,f32 y)            noexcept {return  1.0925484305920792f*x*y;}
[[nodiscard]] constexpr f32 sh_Y2m1(f32 y,f32 z)            noexcept {return -1.0925484305920792f*y*z;}
[[nodiscard]] constexpr f32 sh_Y20 (f32 x,f32 y,f32 z)      noexcept {return  0.31539156525252005f*(2*z*z-x*x-y*y);}
[[nodiscard]] constexpr f32 sh_Y2p1(f32 x,f32 z)            noexcept {return -1.0925484305920792f*x*z;}
[[nodiscard]] constexpr f32 sh_Y2p2(f32 x,f32 y)            noexcept {return  0.54627421529603947f*(x*x-y*y);}

// Project direction into 9 SH coefficients
[[nodiscard]] constexpr std::array<f32,9> sh_project(R3 d) noexcept {
    return{sh_Y00(),sh_Y1m1(d.y),sh_Y10(d.z),sh_Y1p1(d.x),
           sh_Y2m2(d.x,d.y),sh_Y2m1(d.y,d.z),sh_Y20(d.x,d.y,d.z),sh_Y2p1(d.x,d.z),sh_Y2p2(d.x,d.y)};
}

// Evaluate SH expansion at direction d
[[nodiscard]] constexpr f32 sh_eval(const std::array<f32,9>& c, R3 d) noexcept {
    auto b=sh_project(d);
    f32 r=0; for(int i=0;i<9;++i) r+=c[i]*b[i];
    return r;
}

// SH convolution with Lambertian kernel (band-limited: A0=π, A1=2π/3, A2=π/4)
[[nodiscard]] constexpr std::array<f32,9> sh_lambertian(const std::array<f32,9>& c) noexcept {
    constexpr f32 A0=3.14159265358979323846f,A1=2*3.14159265358979323846f/3,A2=3.14159265358979323846f/4;
    return{c[0]*A0, c[1]*A1,c[2]*A1,c[3]*A1, c[4]*A2,c[5]*A2,c[6]*A2,c[7]*A2,c[8]*A2};
}

// Irradiance at normal n from SH9 lighting
[[nodiscard]] constexpr f32 sh_irradiance(const std::array<f32,9>& sh, R3 n) noexcept {
    auto convolved=sh_lambertian(sh);
    return std::max(0.0f,sh_eval(convolved,n));
}

// ─── Linearly Transformed Cosines (LTC) for area lights ──────────────────────
struct LTCMatrix { f32 m[3][3]{}; };

[[nodiscard]] constexpr R3 ltc_transform(const LTCMatrix& M, R3 w) noexcept {
    return{M.m[0][0]*w.x+M.m[0][1]*w.y+M.m[0][2]*w.z,
           M.m[1][0]*w.x+M.m[1][1]*w.y+M.m[1][2]*w.z,
           M.m[2][0]*w.x+M.m[2][1]*w.y+M.m[2][2]*w.z};
}

// Evaluate LTC BRDF at incoming direction wi
[[nodiscard]] constexpr f32 ltc_eval(const LTCMatrix& Minv, R3 wi) noexcept {
    R3 wo=ltc_transform(Minv,wi);
    f32 len=wo.len();
    if(len<1e-6f) return 0;
    wo=wo.norm();
    // Jacobian: |det(M^{-1})|/|wo|³
    f32 det=Minv.m[0][0]*(Minv.m[1][1]*Minv.m[2][2]-Minv.m[1][2]*Minv.m[2][1])
           -Minv.m[0][1]*(Minv.m[1][0]*Minv.m[2][2]-Minv.m[1][2]*Minv.m[2][0])
           +Minv.m[0][2]*(Minv.m[1][0]*Minv.m[2][1]-Minv.m[1][1]*Minv.m[2][0]);
    return std::max(0.0f,wo.y)/(k_pi_f*len*len*len)*std::abs(det);
}

// LTC polygon integral (area light radiance, Heitz et al.)
// Sum of edge contributions via edge integral
[[nodiscard]] constexpr f32 ltc_edge_integral(R3 v1, R3 v2) noexcept {
    f32 dot=std::max(-1.0f,std::min(1.0f,v1.dot(v2)));
    f32 theta=std::acos(dot);
    if(theta<1e-5f) return 0;
    R3 c=v1.cross(v2);
    return c.y*theta/std::sin(theta); // y-component of cross product / sin(theta)*theta
}

// ─── Hair/Fur BSDF (Marschner model) ─────────────────────────────────────────
// Longitudinal scattering (M term) - Gaussian lobe
[[nodiscard]] constexpr f32 hair_M(f32 theta_i, f32 theta_r, f32 width) noexcept {
    f32 v=width*width;
    f32 sin_ti=std::sin(theta_i),sin_tr=std::sin(theta_r);
    f32 cos_ti=std::cos(theta_i),cos_tr=std::cos(theta_r);
    // M(θ_i,θ_r) = cosh((cos_ti*cos_tr)/v) * cos_ti/(2v*cosh(1/v))
    f32 t=cos_ti*cos_tr/v;
    return std::exp(std::min(t,87.0f))/(2*v*std::cosh(1.0f/v)+1e-7f)*cos_ti;
}

// Azimuthal scattering (N term) - simplified
[[nodiscard]] constexpr f32 hair_N_R(f32 phi, f32 eta) noexcept {
    // R lobe (reflection off cuticle)
    f32 h=std::cos(phi/2);
    f32 fr=fresnel_dielectric(std::acos(h),eta);
    return fr*2*std::cos(phi/2);
}
[[nodiscard]] constexpr f32 hair_N_TT(f32 phi, f32 eta, R3 sigma_a) noexcept {
    // TT lobe (transmission): Gaussian approximation
    f32 dphi=phi-k_pi_f;
    return std::exp(-dphi*dphi*0.5f)/(std::sqrt(2*k_pi_f)+1e-7f);
}

// ─── Subsurface scattering (Jensen's dipole) ──────────────────────────────────
struct DiffusionProfile {
    R3 sigma_s{}, sigma_a{};  // scattering and absorption coefficients
    f32 g{0.0f};              // asymmetry factor (Henyey-Greenstein)
    f32 eta{1.5f};            // IOR

    [[nodiscard]] constexpr R3 sigma_t()       const noexcept { return sigma_s+sigma_a; }
    [[nodiscard]] constexpr R3 sigma_s_prime() const noexcept { return sigma_s*(1-g); }
    [[nodiscard]] constexpr R3 sigma_t_prime() const noexcept { return sigma_s_prime()+sigma_a; }
    [[nodiscard]] constexpr R3 alpha_prime()   const noexcept {
        auto sp=sigma_s_prime(),tp=sigma_t_prime();
        return{sp.x/(tp.x+1e-7f),sp.y/(tp.y+1e-7f),sp.z/(tp.z+1e-7f)};
    }
    [[nodiscard]] constexpr R3 d() const noexcept {
        auto tp=sigma_t_prime();
        return{1/(3*tp.x+1e-7f),1/(3*tp.y+1e-7f),1/(3*tp.z+1e-7f)};
    }
    [[nodiscard]] constexpr R3 effective_transport() const noexcept {
        auto tp=sigma_t_prime();
        auto a=alpha_prime();
        return{std::sqrt(3*tp.x*sigma_a.x+1e-7f),std::sqrt(3*tp.y*sigma_a.y+1e-7f),std::sqrt(3*tp.z*sigma_a.z+1e-7f)};
    }
};

// Dipole Rd(r) - diffuse reflectance at distance r
[[nodiscard]] constexpr R3 dipole_Rd(f32 r, const DiffusionProfile& p) noexcept {
    auto zr=p.d(),zv=p.d()*(1+4.0f/3*(1+0.0636f*p.eta)/p.eta);
    auto sigma_tr=p.effective_transport();
    auto eval_term=[&](R3 z)->R3{
        f32 dr=std::sqrt(r*r+z.x*z.x),dv=std::sqrt(r*r+z.y*z.y);
        R3 st=sigma_tr;
        f32 Cr=std::exp(-st.x*dr)*(1+st.x*dr)/(dr*dr*dr+1e-7f);
        f32 Cv=std::exp(-st.y*dv)*(1+st.y*dv)/(dv*dv*dv+1e-7f);
        return{z.x*Cr-z.y*Cv,z.x*Cr-z.y*Cv,z.x*Cr-z.y*Cv}; // simplified
    };
    auto a=p.alpha_prime();
    auto t=eval_term(R3{zr.x,zv.x,0});
    return{a.x*t.x/(4*k_pi_f),a.y*t.y/(4*k_pi_f),a.z*t.z/(4*k_pi_f)};
}

// ─── Precomputed ambient occlusion bent normal ────────────────────────────────
// Integrate visibility over hemisphere to get bent normal and AO factor
template<int SAMPLES>
[[nodiscard]] constexpr std::pair<R3,f32> compute_bent_normal_ao(
    R3 geometric_normal,
    const std::array<R3,SAMPLES>& samples,      // pre-generated hemisphere samples
    const std::array<bool,SAMPLES>& visibility) noexcept
{
    R3 bent{};f32 ao=0;
    auto [t,b]=build_onb(geometric_normal);
    for(int i=0;i<SAMPLES;++i) {
        R3 ws=local_to_world(samples[i],geometric_normal,t,b);
        if(visibility[i]) {
            f32 NdotS=std::max(0.0f,geometric_normal.dot(ws));
            bent.x+=ws.x*NdotS; bent.y+=ws.y*NdotS; bent.z+=ws.z*NdotS;
            ao+=NdotS;
        }
    }
    f32 inv=2.0f/SAMPLES;
    bent.x*=inv;bent.y*=inv;bent.z*=inv;
    ao*=inv;
    return{bent.norm(),ao};
}

// ─── Voxel cone tracing helpers ────────────────────────────────────────────────
// Aperture angle → mip level for cone tracing
[[nodiscard]] constexpr f32 cone_to_mip(f32 cone_angle, f32 dist, f32 voxel_size) noexcept {
    return std::log2(2*dist*std::tan(cone_angle*0.5f)/voxel_size+1);
}
// Specular cone half-angle from roughness
[[nodiscard]] constexpr f32 roughness_to_cone_angle(f32 roughness) noexcept {
    return std::atan(roughness*roughness);
}
} // namespace qs::math
