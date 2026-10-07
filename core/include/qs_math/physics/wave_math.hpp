#pragma once
// qs_math/wave_math.hpp — Ocean wave spectra, Gerstner waves, FFT ocean, wave particles (C++20)
#include "scalar_types.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "random.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <span>

namespace qs {

// ─── Gerstner / trochoidal wave ───────────────────────────────────────────────

struct GerstnerWave {
    Vec2f direction{1.f,0.f};    // normalized direction
    f32   amplitude{1.f};
    f32   wavelength{10.f};
    f32   speed{1.f};
    f32   steepness{0.5f};       // Q: 0=sine, 1=trochoidal peak

    [[nodiscard]] f32 frequency() const { return k_tau/wavelength; }
    [[nodiscard]] f32 phase_speed() const { return std::sqrt(9.81f/frequency()); }

    // Displacement at (x,z,t)
    [[nodiscard]] Vec3f displace(f32 x, f32 z, f32 t) const {
        f32 k=frequency();
        f32 dot_xz=direction.x*x+direction.y*z;
        f32 phi=k*dot_xz-speed*t;
        f32 QA=steepness*amplitude;
        return {
            QA*direction.x*std::cos(phi),
            amplitude*std::sin(phi),
            QA*direction.y*std::cos(phi)};
    }

    [[nodiscard]] Vec3f normal(f32 x, f32 z, f32 t) const {
        f32 k=frequency();
        f32 dot_xz=direction.x*x+direction.y*z;
        f32 phi=k*dot_xz-speed*t;
        f32 WA=k*amplitude;
        f32 QWA=steepness*WA;
        return normalize(Vec3f{
            -direction.x*WA*std::cos(phi),
            1.f-steepness*WA*std::sin(phi),
            -direction.y*WA*std::cos(phi)});
    }
};

// Sum multiple Gerstner waves
[[nodiscard]] inline Vec3f gerstner_sum(
    f32 x, f32 z, f32 t,
    std::span<const GerstnerWave> waves)
{
    Vec3f disp{x,0,z};
    Vec3f acc{0,0,0};
    for (const auto& w : waves) acc=acc+w.displace(x,z,t);
    return disp+acc;
}

// ─── Ocean wave spectra ───────────────────────────────────────────────────────

// Phillips spectrum
[[nodiscard]] inline f32 phillips_spectrum(
    Vec2f k_vec, Vec2f wind_dir, f32 wind_speed, f32 A=1e-4f)
{
    f32 k=length(k_vec);
    if(k<1e-6f) return 0.f;
    f32 L=wind_speed*wind_speed/9.81f;
    f32 kL2=k*L;
    f32 k_dot_w=dot(normalize(k_vec),wind_dir);
    return A*std::exp(-1.f/(kL2*kL2))/(k*k*k*k)*k_dot_w*k_dot_w;
}

// JONSWAP spectrum (more realistic)
[[nodiscard]] inline f32 jonswap_spectrum(f32 omega, f32 omega_peak, f32 alpha=0.0081f, f32 gamma=3.3f) {
    f32 sigma=(omega<=omega_peak)?0.07f:0.09f;
    f32 r=std::exp(-std::pow(omega-omega_peak,2.f)/(2.f*sigma*sigma*omega_peak*omega_peak));
    f32 PM=alpha*9.81f*9.81f/std::pow(omega,5.f)*std::exp(-1.25f*std::pow(omega_peak/omega,4.f));
    return PM*std::pow(gamma,r);
}

// ─── FFT Ocean (Tessendorf 2001) ──────────────────────────────────────────────

struct FFTOcean {
    u32 N{128};             // grid resolution (power of 2)
    f32 L{500.f};           // patch size (meters)
    Vec2f wind_dir{1.f,0.f};
    f32 wind_speed{12.f};
    f32 phillips_A{1e-4f};

    std::vector<std::complex<f32>> h0;    // initial spectrum
    std::vector<std::complex<f32>> h0_conj; // conjugate

    void init(u64 seed=0) {
        h0.resize(N*N); h0_conj.resize(N*N);
        Pcg32 rng{seed};
        for (u32 y=0;y<N;++y)
        for (u32 x=0;x<N;++x) {
            Vec2f k_vec{
                (f32(i32(x)-i32(N/2))*k_tau/L),
                (f32(i32(y)-i32(N/2))*k_tau/L)};
            f32 ph=phillips_spectrum(k_vec,wind_dir,wind_speed,phillips_A);
            f32 sqrt_ph=std::sqrt(ph*0.5f);
            // Box-Muller
            f32 u1=rng.next_f32()+1e-6f, u2=rng.next_f32();
            f32 r=std::sqrt(-2.f*std::log(u1));
            f32 ang=k_tau*u2;
            h0[y*N+x]={r*std::cos(ang)*sqrt_ph, r*std::sin(ang)*sqrt_ph};
            // Conjugate for H(-k)
            f32 ph2=phillips_spectrum(-k_vec,wind_dir,wind_speed,phillips_A);
            f32 sp2=std::sqrt(ph2*0.5f);
            h0_conj[y*N+x]={r*std::cos(ang)*sp2,-r*std::sin(ang)*sp2};
        }
    }

    // Evaluate spectrum at time t
    [[nodiscard]] std::vector<std::complex<f32>> evaluate(f32 t) const {
        std::vector<std::complex<f32>> ht(N*N);
        for (u32 y=0;y<N;++y)
        for (u32 x=0;x<N;++x) {
            Vec2f k_vec{(f32(i32(x)-i32(N/2))*k_tau/L),(f32(i32(y)-i32(N/2))*k_tau/L)};
            f32 k=length(k_vec);
            f32 omega=std::sqrt(9.81f*k);  // dispersion relation
            f32 c=std::cos(omega*t), s=std::sin(omega*t);
            ht[y*N+x]=h0[y*N+x]*std::complex<f32>{c,s}+h0_conj[y*N+x]*std::complex<f32>{c,-s};
        }
        return ht;
    }
};

// ─── Wave particles (Yuksel 2007) ─────────────────────────────────────────────

struct WaveParticle {
    Vec2f pos{};
    Vec2f dir{};
    f32   amplitude{};
    f32   radius{};
    f32   speed{1.f};
    f32   lifetime{};
};

inline void update_wave_particles(
    std::vector<WaveParticle>& particles, f32 dt)
{
    for (auto& p : particles) {
        p.pos=p.pos+p.dir*p.speed*dt;
        p.radius+=p.speed*dt;
        p.lifetime-=dt;
    }
    particles.erase(std::remove_if(particles.begin(),particles.end(),
        [](const WaveParticle&p){return p.lifetime<=0.f;}),particles.end());
}

[[nodiscard]] inline f32 wave_particle_height(
    const Vec2f& query, const WaveParticle& p)
{
    f32 d=length(query-p.pos);
    if(d>p.radius) return 0.f;
    f32 t=d/p.radius;
    // Envelope (raised cosine)
    f32 env=std::cos(t*k_pi_over_two);
    return p.amplitude*env*env*std::cos(d*k_tau/p.radius);
}

// ─── Shallow water equations ──────────────────────────────────────────────────

struct ShallowWaterGrid {
    std::vector<f32> h;    // water height
    std::vector<f32> u_v;  // x-velocity
    std::vector<f32> v_v;  // y-velocity
    u32 nx{},ny{};
    f32 dx{1.f}, g{9.81f};

    void resize(u32 x, u32 y, f32 rest_h=1.f) {
        nx=x;ny=y; h.assign(x*y,rest_h); u_v.assign(x*y,0.f); v_v.assign(x*y,0.f);
    }

    void step(f32 dt) {
        std::vector<f32> hn=h, un=u_v, vn=v_v;
        f32 inv_dx=1.f/dx;
        for (u32 y2=1;y2<ny-1;++y2)
        for (u32 x2=1;x2<nx-1;++x2) {
            u32 i=y2*nx+x2;
            f32 dhdx=(h[i+1]-h[i-1])*0.5f*inv_dx;
            f32 dhdy=(h[i+nx]-h[i-nx])*0.5f*inv_dx;
            f32 dudx=(u_v[i+1]-u_v[i-1])*0.5f*inv_dx;
            f32 dvdy=(v_v[i+nx]-v_v[i-nx])*0.5f*inv_dx;
            hn[i]=h[i]-dt*h[i]*(dudx+dvdy);
            un[i]=u_v[i]-dt*(u_v[i]*dudx+g*dhdx);
            vn[i]=v_v[i]-dt*(v_v[i]*dvdy+g*dhdy);
        }
        h=hn; u_v=un; v_v=vn;
    }
};

} // namespace qs
