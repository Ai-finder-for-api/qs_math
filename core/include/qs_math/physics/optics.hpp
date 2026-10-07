#pragma once
// qs_math/physics/optics.hpp — Ray optics, wave optics, diffraction, polarization
#include "../arithmetic/scalar_types.hpp"
#include <cmath>
#include <array>

namespace qs::math {

// ─── Ray optics ───────────────────────────────────────────────────────────────
struct OV3 { f64 x,y,z;
    [[nodiscard]] constexpr OV3 operator+(OV3 b) const noexcept{return{x+b.x,y+b.y,z+b.z};}
    [[nodiscard]] constexpr OV3 operator-(OV3 b) const noexcept{return{x-b.x,y-b.y,z-b.z};}
    [[nodiscard]] constexpr OV3 operator*(f64 s) const noexcept{return{x*s,y*s,z*s};}
    [[nodiscard]] constexpr f64  dot(OV3 b) const noexcept{return x*b.x+y*b.y+z*b.z;}
    [[nodiscard]] constexpr OV3  cross(OV3 b) const noexcept{return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    [[nodiscard]] f64 len() const noexcept{return std::sqrt(dot(*this));}
    [[nodiscard]] OV3 norm() const noexcept{f64 l=len();return l>1e-12?OV3{x/l,y/l,z/l}:*this;}
};

// Snell's law refraction vector
[[nodiscard]] constexpr OV3 refract(OV3 d, OV3 n, f64 n1, f64 n2) noexcept {
    f64 eta=n1/n2, cos_i=-d.dot(n);
    f64 sin2_t=eta*eta*(1-cos_i*cos_i);
    if(sin2_t>1) return {}; // TIR
    f64 cos_t=std::sqrt(1-sin2_t);
    return d*eta+n*(eta*cos_i-cos_t);
}
[[nodiscard]] constexpr OV3 reflect_ray(OV3 d, OV3 n) noexcept {
    return d-n*(2*d.dot(n));
}

// Thin lens equation: 1/f = 1/do + 1/di
[[nodiscard]] constexpr f64 thin_lens_image(f64 focal_len, f64 object_dist) noexcept {
    f64 inv_di=1/focal_len-1/object_dist;
    return inv_di>1e-15?1/inv_di:-1;
}
[[nodiscard]] constexpr f64 magnification(f64 image_dist, f64 object_dist) noexcept {
    return -image_dist/object_dist;
}
// Lensmaker's equation: 1/f = (n-1)[1/R1 - 1/R2 + (n-1)d/(nR1R2)]
[[nodiscard]] constexpr f64 lensmaker_focal(f64 n, f64 R1, f64 R2, f64 d=0) noexcept {
    return 1.0/((n-1)*(1/R1-1/R2+(n-1)*d/(n*R1*R2+1e-15)));
}

// ─── Jones calculus (polarization) ───────────────────────────────────────────
// Jones vector: [Ex, Ey] complex amplitudes
struct JonesVector { Cx Ex{1,0},Ey{0,0};
    [[nodiscard]] constexpr f64 intensity() const noexcept{return Ex.mag2()+Ey.mag2();}
    [[nodiscard]] constexpr bool is_linear() const noexcept{
        return std::abs(Ex.im)<1e-10&&std::abs(Ey.im)<1e-10;}
    [[nodiscard]] constexpr f64 polarization_angle() const noexcept{
        return std::atan2(Ey.re,Ex.re);}
};
// Jones matrices
[[nodiscard]] constexpr JonesVector jones_linear_polarizer(JonesVector j, f64 theta) noexcept {
    f64 c=std::cos(theta),s=std::sin(theta);
    return{Cx{c*c*j.Ex.re-c*s*j.Ey.re,c*c*j.Ex.im-c*s*j.Ey.im},
           Cx{c*s*j.Ex.re-s*s*j.Ey.re,c*s*j.Ex.im-s*s*j.Ey.im}};
}
[[nodiscard]] constexpr JonesVector jones_quarter_wave(JonesVector j, f64 theta) noexcept {
    // QWP fast axis at angle theta
    f64 c=std::cos(theta),s=std::sin(theta);
    Cx Ex=Cx{c*c,0}*j.Ex+Cx{c*s,0}*j.Ey;
    Cx Ey=Cx{c*s,0}*j.Ex+Cx{0,-1}*(Cx{s*s,0}*j.Ey); // simplified
    return{Ex,Ey};
}

// ─── Stokes parameters (partially polarized light) ────────────────────────────
struct StokesVector { f64 S0,S1,S2,S3;
    [[nodiscard]] constexpr f64 dop() const noexcept{
        return std::sqrt(S1*S1+S2*S2+S3*S3)/(S0+1e-15);}
    [[nodiscard]] constexpr bool is_fully_polarized(f64 tol=1e-6) const noexcept{
        return std::abs(dop()-1)<tol;}
};
// Jones → Stokes
[[nodiscard]] constexpr StokesVector jones_to_stokes(JonesVector j) noexcept {
    return{j.Ex.mag2()+j.Ey.mag2(),
           j.Ex.mag2()-j.Ey.mag2(),
           2*(j.Ex*j.Ey.conj()).re,
           2*(j.Ex*j.Ey.conj()).im};
}

// ─── Mueller matrix (transforms Stokes vectors) ───────────────────────────────
using MuellerMatrix = std::array<std::array<f64,4>,4>;
[[nodiscard]] constexpr StokesVector mueller_apply(const MuellerMatrix& M, StokesVector s) noexcept {
    std::array<f64,4> v={s.S0,s.S1,s.S2,s.S3};
    std::array<f64,4> r{};
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) r[i]+=M[i][j]*v[j];
    return{r[0],r[1],r[2],r[3]};
}
// Linear polarizer Mueller matrix at angle theta
[[nodiscard]] constexpr MuellerMatrix mueller_linear_polarizer(f64 theta) noexcept {
    f64 c=std::cos(2*theta),s=std::sin(2*theta);
    return{{{{0.5,0.5*c,0.5*s,0}},{0.5*c,0.5*c*c,0.5*c*s,0},{0.5*s,0.5*c*s,0.5*s*s,0},{0,0,0,0}}}};
}

// ─── Wave optics ──────────────────────────────────────────────────────────────
// Huygens-Fresnel diffraction integral (1D, scalar)
template<int N>
[[nodiscard]] constexpr Cx huygens_fresnel_1d(
    const std::array<f64,N>& aperture, f64 dx,
    f64 lambda, f64 z, f64 x_obs) noexcept
{
    Cx sum{};
    f64 k=2*3.14159265358979323846/lambda;
    for(int i=0;i<N;++i) {
        f64 xi=(i-N/2)*dx;
        f64 r=std::sqrt(z*z+(x_obs-xi)*(x_obs-xi));
        sum=sum+Cx{aperture[i]*std::cos(k*r)/r,aperture[i]*std::sin(k*r)/r};
    }
    return sum;
}

// Fraunhofer diffraction (far field, FT of aperture)
// Single slit: I(theta) = I0*(sin(beta)/beta)^2, beta=pi*a*sin(theta)/lambda
[[nodiscard]] constexpr f64 single_slit_intensity(f64 theta, f64 a, f64 lambda) noexcept {
    f64 beta=3.14159265358979323846*a*std::sin(theta)/lambda;
    if(std::abs(beta)<1e-10) return 1.0;
    return (std::sin(beta)/beta)*(std::sin(beta)/beta);
}
// Double slit: I = I_single * cos²(delta/2), delta=2π*d*sin(theta)/lambda
[[nodiscard]] constexpr f64 double_slit_intensity(f64 theta,f64 a,f64 d,f64 lambda) noexcept {
    f64 delta=2*3.14159265358979323846*d*std::sin(theta)/lambda;
    f64 cos2=std::cos(delta*0.5); cos2*=cos2;
    return single_slit_intensity(theta,a,lambda)*cos2;
}
// Diffraction grating: maxima at d*sin(theta)=m*lambda
[[nodiscard]] constexpr f64 grating_order_angle(int m, f64 d, f64 lambda) noexcept {
    f64 s=m*lambda/d; return std::abs(s)<=1?std::asin(s):-1;
}
// Resolving power R=mN (m=order, N=number of slits)
[[nodiscard]] constexpr f64 grating_resolving_power(int m, int N) noexcept { return m*N; }

// ─── Interferometry ───────────────────────────────────────────────────────────
// Young's double slit: path difference Δ = d*sin(θ)
[[nodiscard]] constexpr f64 fringe_visibility(f64 I_max, f64 I_min) noexcept {
    return (I_max-I_min)/(I_max+I_min+1e-15);
}
[[nodiscard]] constexpr f64 michelson_fringe_intensity(f64 delta, f64 I0=1) noexcept {
    return I0*(1+std::cos(delta));
}
// Fabry-Perot transmission: T = 1/(1 + F*sin²(delta/2))
[[nodiscard]] constexpr f64 fabry_perot_transmission(f64 delta, f64 R) noexcept {
    f64 F=4*R/((1-R)*(1-R));
    f64 sd=std::sin(delta*0.5); return 1/(1+F*sd*sd);
}
[[nodiscard]] constexpr f64 fabry_perot_finesse(f64 R) noexcept {
    return 3.14159265358979323846*std::sqrt(R)/(1-R);
}

// ─── Gaussian beam optics ─────────────────────────────────────────────────────
struct GaussianBeam {
    f64 lambda, w0; // wavelength, waist radius
    f64 z_R{0};    // Rayleigh range
    constexpr GaussianBeam(f64 lam, f64 waist) noexcept
        : lambda(lam), w0(waist),
          z_R(3.14159265358979323846*waist*waist/lam) {}
    [[nodiscard]] constexpr f64 w(f64 z) const noexcept{
        return w0*std::sqrt(1+(z/z_R)*(z/z_R));}
    [[nodiscard]] constexpr f64 R_curvature(f64 z) const noexcept{
        return z>1e-12?z*(1+(z_R/z)*(z_R/z)):1e30;}
    [[nodiscard]] constexpr f64 gouy_phase(f64 z) const noexcept{
        return std::atan2(z,z_R);}
    [[nodiscard]] constexpr f64 intensity(f64 r, f64 z) const noexcept{
        f64 wz=w(z);
        return (w0/wz)*(w0/wz)*std::exp(-2*r*r/(wz*wz));}
    [[nodiscard]] constexpr f64 divergence_angle() const noexcept{
        return lambda/(3.14159265358979323846*w0);}
};
} // namespace qs::math
