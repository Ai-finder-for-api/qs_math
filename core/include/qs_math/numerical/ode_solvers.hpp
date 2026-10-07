#pragma once
// qs_math/numerical/ode_solvers.hpp — ODE integration: RK4, RK45 (Dormand-Prince),
//   Adams-Bashforth, implicit Euler, Verlet, leapfrog, symplectic integrators (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../linear_algebra/vec3.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <array>

namespace qs::math::numerical {

// ─── State vector concept ─────────────────────────────────────────────────
// ODE: dy/dt = f(t, y),  y ∈ R^N
// We use std::vector<f64> as the generic state; specialized for Vec3f too.

using StateVec = std::vector<f64>;
using ODE      = std::function<StateVec(f64 t, const StateVec& y)>;

inline StateVec state_add(const StateVec& a, const StateVec& b) {
    StateVec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i]+b[i];
    return r;
}
inline StateVec state_scale(const StateVec& a, f64 s) {
    StateVec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i]*s;
    return r;
}
inline StateVec state_axpy(f64 a, const StateVec& x, const StateVec& y) {
    StateVec r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = a*x[i]+y[i];
    return r;
}
[[nodiscard]] inline f64 state_norm(const StateVec& y) {
    f64 s = 0.0;
    for (auto v : y) s += v*v;
    return std::sqrt(s);
}

// ─── Classic RK4 ──────────────────────────────────────────────────────────
[[nodiscard]] inline StateVec rk4_step(const ODE& f, f64 t, const StateVec& y, f64 h) {
    StateVec k1 = state_scale(f(t,            y),                       h);
    StateVec k2 = state_scale(f(t+0.5*h,      state_axpy(0.5,k1,y)),   h);
    StateVec k3 = state_scale(f(t+0.5*h,      state_axpy(0.5,k2,y)),   h);
    StateVec k4 = state_scale(f(t+h,          state_axpy(1.0,k3,y)),    h);
    StateVec yn = y;
    for (size_t i = 0; i < y.size(); ++i)
        yn[i] += (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
    return yn;
}

struct ODESolution {
    std::vector<f64>       t;
    std::vector<StateVec>  y;
};

[[nodiscard]] inline ODESolution rk4_integrate(
    const ODE& f, f64 t0, f64 tf, StateVec y0, f64 h)
{
    ODESolution sol;
    sol.t.push_back(t0);
    sol.y.push_back(y0);
    f64 t = t0;
    StateVec y = y0;
    while (t < tf - 1e-14) {
        f64 dt = min(h, tf - t);
        y = rk4_step(f, t, y, dt);
        t += dt;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }
    return sol;
}

// ─── Dormand-Prince RK45 (adaptive step) ─────────────────────────────────
// Butcher tableau coefficients
struct RK45Result { StateVec y4, y5; f64 err_norm; };

[[nodiscard]] inline RK45Result rk45_step(const ODE& f, f64 t, const StateVec& y, f64 h) {
    // Dormand-Prince coefficients
    static constexpr f64 c2=1./5.,  c3=3./10., c4=4./5., c5=8./9.;
    static constexpr f64 a21=1./5.;
    static constexpr f64 a31=3./40.,    a32=9./40.;
    static constexpr f64 a41=44./45.,   a42=-56./15.,  a43=32./9.;
    static constexpr f64 a51=19372./6561., a52=-25360./2187., a53=64448./6561., a54=-212./729.;
    static constexpr f64 a61=9017./3168., a62=-355./33., a63=46732./5247., a64=49./176., a65=-5103./18656.;
    // 4th order weights
    static constexpr f64 b1=35./384., b3=500./1113., b4=125./192., b5=-2187./6784., b6=11./84.;
    // Error coefficients (4th - 5th)
    static constexpr f64 e1=71./57600., e3=-71./16695., e4=71./1920., e5=-17253./339200., e6=22./525., e7=-1./40.;

    StateVec k1=state_scale(f(t,                                              y),h);
    StateVec k2=state_scale(f(t+c2*h, state_axpy(a21,k1,y)),                  h);
    StateVec k3=state_scale(f(t+c3*h, state_add(state_axpy(a31,k1,y),state_scale(k2,a32))), h);

    auto tmp4=y; for(size_t i=0;i<y.size();++i) tmp4[i]+=a41*k1[i]+a42*k2[i]+a43*k3[i];
    StateVec k4=state_scale(f(t+c4*h, tmp4), h);

    auto tmp5=y; for(size_t i=0;i<y.size();++i) tmp5[i]+=a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i];
    StateVec k5=state_scale(f(t+c5*h, tmp5), h);

    auto tmp6=y; for(size_t i=0;i<y.size();++i) tmp6[i]+=a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i];
    StateVec k6=state_scale(f(t+h, tmp6), h);

    // 5th order solution
    StateVec y5=y;
    for(size_t i=0;i<y.size();++i) y5[i]+=b1*k1[i]+b3*k3[i]+b4*k4[i]+b5*k5[i]+b6*k6[i];
    StateVec k7=state_scale(f(t+h, y5), h);

    // 4th order (alias for error estimation)
    StateVec err(y.size());
    for(size_t i=0;i<y.size();++i)
        err[i]=e1*k1[i]+e3*k3[i]+e4*k4[i]+e5*k5[i]+e6*k6[i]+e7*k7[i];

    f64 en = state_norm(err);
    // y4 for return (FSAL: use y5 as the advance)
    return {y5, y5, en};
}

[[nodiscard]] inline ODESolution rk45_integrate(
    const ODE& f, f64 t0, f64 tf, StateVec y0,
    f64 rtol=1e-6, f64 atol=1e-9, f64 h0=1e-3)
{
    ODESolution sol;
    sol.t.push_back(t0);
    sol.y.push_back(y0);
    f64 t = t0, h = h0;
    StateVec y = y0;
    while (t < tf - 1e-14) {
        h = min(h, tf - t);
        auto res = rk45_step(f, t, y, h);
        f64 scale = max(atol, rtol * state_norm(y));
        f64 err = res.err_norm / (scale * std::sqrt(f64(y.size())));
        if (err <= 1.0 || h < 1e-14) {
            t += h; y = res.y5;
            sol.t.push_back(t);
            sol.y.push_back(y);
        }
        // PI step size control
        h *= min(5.0, max(0.2, 0.9 * std::pow(1.0/(err+1e-30), 0.2)));
    }
    return sol;
}

// ─── Adams-Bashforth 4-step (multi-step, fixed h) ────────────────────────
class AdamsBashforth4 {
    ODE f_;
    f64 h_;
    std::array<StateVec,4> k_{};
    std::array<f64,4>      t_{};
    int count_ = 0;
public:
    AdamsBashforth4(const ODE& f, f64 h) : f_(f), h_(h) {}

    // Warm up with RK4
    void init(f64 t0, const StateVec& y0) {
        k_[0]=f_(t0,y0); t_[0]=t0;
        StateVec y=y0; f64 t=t0;
        for(int i=1;i<4;++i){
            y=rk4_step(f_,t,y,h_); t+=h_;
            k_[i]=f_(t,y); t_[i]=t;
        }
        count_=4;
    }

    [[nodiscard]] StateVec step(f64 t, const StateVec& y) {
        StateVec yn=y;
        // AB4: y_{n+1} = y_n + h/24*(55k_n - 59k_{n-1} + 37k_{n-2} - 9k_{n-3})
        for(size_t i=0;i<y.size();++i)
            yn[i]+=h_/24.0*(55.0*k_[3][i]-59.0*k_[2][i]+37.0*k_[1][i]-9.0*k_[0][i]);
        // Shift history
        k_[0]=k_[1]; k_[1]=k_[2]; k_[2]=k_[3];
        k_[3]=f_(t+h_,yn);
        return yn;
    }
};

// ─── Implicit Euler (backward Euler, fixed-point iteration) ───────────────
[[nodiscard]] inline StateVec implicit_euler_step(
    const ODE& f, f64 t, const StateVec& y, f64 h, int max_iter=10, f64 tol=1e-8)
{
    StateVec yn = rk4_step(f,t,y,h); // initial guess
    for(int it=0;it<max_iter;++it){
        StateVec fval=f(t+h,yn);
        StateVec ynew=y;
        for(size_t i=0;i<y.size();++i) ynew[i]=y[i]+h*fval[i];
        f64 err=0.0;
        for(size_t i=0;i<y.size();++i) err+=std::abs(ynew[i]-yn[i]);
        yn=ynew;
        if(err<tol) break;
    }
    return yn;
}

// ─── Störmer-Verlet (position Verlet, conservative systems) ───────────────
// q'' = a(q),   state=(q,v), requires acceleration function
using AccelFn = std::function<Vec3f(Vec3f q)>;

struct VerletState { Vec3f q, v; };

[[nodiscard]] inline VerletState verlet_step(
    const AccelFn& accel, VerletState s, f64 h) noexcept
{
    Vec3f a0 = accel(s.q);
    Vec3f q1 = s.q + s.v*f32(h) + a0*(f32(h)*f32(h)*0.5f);
    Vec3f a1 = accel(q1);
    Vec3f v1 = s.v + (a0+a1)*(f32(h)*0.5f);
    return {q1, v1};
}

// ─── Leapfrog / Kick-Drift-Kick ───────────────────────────────────────────
struct LeapfrogState { Vec3f q, v_half; }; // v stored at half-step

[[nodiscard]] inline LeapfrogState leapfrog_init(Vec3f q0, Vec3f v0, const AccelFn& accel, f32 h) noexcept {
    Vec3f a = accel(q0);
    return {q0, v0 + a*(h*0.5f)};
}
[[nodiscard]] inline LeapfrogState leapfrog_step(
    const AccelFn& accel, LeapfrogState s, f32 h) noexcept
{
    Vec3f q1 = s.q + s.v_half*h;
    Vec3f a1 = accel(q1);
    Vec3f v1_half = s.v_half + a1*h;
    return {q1, v1_half};
}
[[nodiscard]] inline Vec3f leapfrog_velocity(const LeapfrogState& s, const AccelFn& accel, f32 h) noexcept {
    return s.v_half - accel(s.q)*(h*0.5f);
}

// ─── Yoshida 4th-order symplectic ─────────────────────────────────────────
// Coefficients from Yoshida 1990
inline constexpr f64 YOSHIDA_W0 = -1.7024143839193153;
inline constexpr f64 YOSHIDA_W1 =  1.3512071919596578;
inline constexpr f64 YOSHIDA_C1 =  YOSHIDA_W1 * 0.5;
inline constexpr f64 YOSHIDA_C2 = (YOSHIDA_W0 + YOSHIDA_W1) * 0.5;
inline constexpr f64 YOSHIDA_D1 =  YOSHIDA_W1;
inline constexpr f64 YOSHIDA_D2 =  YOSHIDA_W0;

[[nodiscard]] inline VerletState yoshida4_step(
    const AccelFn& accel, VerletState s, f32 h) noexcept
{
    f32 c1=f32(YOSHIDA_C1*h), c2=f32(YOSHIDA_C2*h);
    f32 d1=f32(YOSHIDA_D1*h), d2=f32(YOSHIDA_D2*h);
    Vec3f q=s.q, v=s.v;
    q=q+v*c1; v=v+accel(q)*d1;
    q=q+v*c2; v=v+accel(q)*d2;
    q=q+v*c2; v=v+accel(q)*d1;
    q=q+v*c1;
    return {q, v};
}

// ─── Energy conservation check ────────────────────────────────────────────
// For Hamiltonian H = KE + PE
using PotentialFn = std::function<f32(Vec3f)>;
[[nodiscard]] inline f32 hamiltonian(Vec3f v, Vec3f q, f32 mass, const PotentialFn& PE) noexcept {
    return 0.5f*mass*dot(v,v) + PE(q);
}
[[nodiscard]] inline f32 energy_error(f32 H0, f32 Ht) noexcept {
    return std::abs((Ht-H0)/H0);
}

} // namespace qs::math::numerical
