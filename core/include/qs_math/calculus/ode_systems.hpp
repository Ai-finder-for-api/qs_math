#pragma once
// qs_math/ode_systems.hpp — Systems of ODEs: Lotka-Volterra, Van der Pol, N-body, solvers (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <vector>
#include <functional>
#include <cmath>

namespace qs {

using ODEState  = std::vector<f64>;
using ODERhs    = std::function<ODEState(f64 t, const ODEState& y)>;

// ─── Generic ODE solvers ──────────────────────────────────────────────────────

inline ODEState ode_add(const ODEState& a, const ODEState& b, f64 s=1) {
    ODEState r(a.size()); for(u32 i=0;i<u32(a.size());++i) r[i]=a[i]+s*b[i]; return r;
}
inline ODEState ode_scale(const ODEState& a, f64 s) {
    ODEState r(a.size()); for(u32 i=0;i<u32(a.size());++i) r[i]=a[i]*s; return r;
}
inline f64 ode_norm(const ODEState& a) {
    f64 s=0; for(auto v:a) s+=v*v; return std::sqrt(s);
}

// Euler method (1st order)
[[nodiscard]] inline ODEState euler_step(const ODERhs& f, f64 t, const ODEState& y, f64 h) {
    return ode_add(y, f(t,y), h);
}

// RK4 (4th order)
[[nodiscard]] inline ODEState rk4_step(const ODERhs& f, f64 t, const ODEState& y, f64 h) {
    auto k1=f(t,y);
    auto k2=f(t+0.5*h, ode_add(y,k1,0.5*h));
    auto k3=f(t+0.5*h, ode_add(y,k2,0.5*h));
    auto k4=f(t+h,     ode_add(y,k3,h));
    ODEState r(y.size());
    for(u32 i=0;i<u32(y.size());++i) r[i]=y[i]+h/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
    return r;
}

// RKF45 with adaptive step size
struct RKF45Result {
    std::vector<f64> t_vals;
    std::vector<ODEState> y_vals;
    u32 steps{0};
    bool success{true};
};

[[nodiscard]] inline RKF45Result rkf45(const ODERhs& f, f64 t0, f64 tf,
    const ODEState& y0, f64 tol=1e-6, f64 hmin=1e-10, f64 hmax=0.1)
{
    // Fehlberg coefficients
    static constexpr f64 c2=0.25,c3=3./8,c4=12./13,c5=1.,c6=0.5;
    static constexpr f64 a21=0.25;
    static constexpr f64 a31=3./32,a32=9./32;
    static constexpr f64 a41=1932./2197,a42=-7200./2197,a43=7296./2197;
    static constexpr f64 a51=439./216,a52=-8.,a53=3680./513,a54=-845./4104;
    static constexpr f64 a61=-8./27,a62=2.,a63=-3544./2565,a64=1859./4104,a65=-11./40;
    static constexpr f64 b1=16./135,b3=6656./12825,b4=28561./56430,b5=-9./50,b6=2./55;
    static constexpr f64 e1=1./360,e3=-128./4275,e4=-2197./75240,e5=1./50,e6=2./55;

    RKF45Result res;
    f64 t=t0, h=hmax*0.1;
    ODEState y=y0;
    res.t_vals.push_back(t); res.y_vals.push_back(y);
    u32 max_steps=100000;
    for(u32 step=0;step<max_steps&&t<tf;++step) {
        h=std::min(h,tf-t);
        auto k1=f(t,y);
        auto k2=f(t+c2*h, ode_add(y,k1,a21*h));
        auto k3=f(t+c3*h, ode_add(ode_add(y,k1,a31*h),k2,a32*h));
        auto k4=f(t+c4*h, ode_add(ode_add(ode_add(y,k1,a41*h),k2,a42*h),k3,a43*h));
        auto k5=f(t+c5*h, ode_add(ode_add(ode_add(ode_add(y,k1,a51*h),k2,a52*h),k3,a53*h),k4,a54*h));
        auto k6=f(t+c6*h, ode_add(ode_add(ode_add(ode_add(ode_add(y,k1,a61*h),k2,a62*h),k3,a63*h),k4,a64*h),k5,a65*h));
        // 5th order solution
        ODEState y5(y.size());
        for(u32 i=0;i<u32(y.size());++i) y5[i]=y[i]+h*(b1*k1[i]+b3*k3[i]+b4*k4[i]+b5*k5[i]+b6*k6[i]);
        // Error estimate
        f64 err=0; u32 n=u32(y.size());
        for(u32 i=0;i<n;++i) {
            f64 e=h*std::abs(e1*k1[i]+e3*k3[i]+e4*k4[i]+e5*k5[i]+e6*k6[i]);
            f64 sc=tol*(std::abs(y[i])+std::abs(y5[i]))*0.5+1e-10;
            err=std::max(err,e/sc);
        }
        if(err<=1.0) { t+=h; y=y5; res.t_vals.push_back(t); res.y_vals.push_back(y); ++res.steps; }
        f64 factor=std::min(5.0,std::max(0.2,0.9*std::pow(err+1e-30,-0.2)));
        h*=factor;
        if(h<hmin){ res.success=false; break; }
        h=std::min(h,hmax);
    }
    return res;
}

// ─── Specific ODE systems ─────────────────────────────────────────────────────

// Lotka-Volterra predator-prey: dx/dt = αx - βxy, dy/dt = δxy - γy
struct LotkaVolterra {
    f64 alpha{1.0},beta{0.1},delta{0.075},gamma{1.5};
    [[nodiscard]] ODERhs rhs() const {
        return [a=alpha,b=beta,d=delta,g=gamma](f64, const ODEState& y) -> ODEState {
            return {a*y[0]-b*y[0]*y[1], d*y[0]*y[1]-g*y[1]};
        };
    }
    [[nodiscard]] f64 equilibrium_x() const { return gamma/delta; }
    [[nodiscard]] f64 equilibrium_y() const { return alpha/beta; }
    // Conserved quantity: V = δx - γ ln x + βy - α ln y
    [[nodiscard]] f64 conserved(const ODEState& y) const {
        return delta*y[0]-gamma*std::log(y[0])+beta*y[1]-alpha*std::log(y[1]);
    }
};

// Van der Pol oscillator: x'' - μ(1-x²)x' + x = 0
// State: [x, x']
struct VanDerPol {
    f64 mu{1.0};
    [[nodiscard]] ODERhs rhs() const {
        return [m=mu](f64, const ODEState& y) -> ODEState {
            return {y[1], m*(1-y[0]*y[0])*y[1]-y[0]};
        };
    }
    // Stiff for large mu — use implicit solver there
    [[nodiscard]] bool is_stiff() const { return mu > 10.0; }
};

// Lorenz system: chaos
struct Lorenz {
    f64 sigma{10.0},rho{28.0},beta{8.0/3.0};
    [[nodiscard]] ODERhs rhs() const {
        return [s=sigma,r=rho,b=beta](f64, const ODEState& y) -> ODEState {
            return {s*(y[1]-y[0]), y[0]*(r-y[2])-y[1], y[0]*y[1]-b*y[2]};
        };
    }
    // Lyapunov exponent approximation
    [[nodiscard]] f64 largest_lyapunov_approx() const { return 0.9056*std::log(sigma); }
};

// Rossler attractor
struct Rossler {
    f64 a{0.2},b{0.2},c{5.7};
    [[nodiscard]] ODERhs rhs() const {
        return [a=a,b=b,c=c](f64, const ODEState& y) -> ODEState {
            return {-y[1]-y[2], y[0]+a*y[1], b+y[2]*(y[0]-c)};
        };
    }
};

// N-body gravitational system: state = [x1,y1,z1,vx1,vy1,vz1, x2,...]
struct NBody {
    struct Body { f64 mass; };
    std::vector<Body> bodies;
    f64 G{6.674e-11};
    u32 n() const { return u32(bodies.size()); }

    [[nodiscard]] ODERhs rhs() const {
        return [bods=bodies,G=G](f64, const ODEState& y) -> ODEState {
            u32 nb=u32(bods.size());
            ODEState dy(y.size(),0);
            for(u32 i=0;i<nb;++i) {
                // velocity → position derivative
                dy[6*i+0]=y[6*i+3]; dy[6*i+1]=y[6*i+4]; dy[6*i+2]=y[6*i+5];
                // force from all other bodies
                for(u32 j=0;j<nb;++j) {
                    if(i==j) continue;
                    f64 dx=y[6*j]-y[6*i],dy_=y[6*j+1]-y[6*i+1],dz=y[6*j+2]-y[6*i+2];
                    f64 r2=dx*dx+dy_*dy_+dz*dz+1e-20;
                    f64 r3=r2*std::sqrt(r2);
                    f64 f=G*bods[j].mass/r3;
                    dy[6*i+3]+=f*dx; dy[6*i+4]+=f*dy_; dy[6*i+5]+=f*dz;
                }
            }
            return dy;
        };
    }

    // Energy: KE + PE
    [[nodiscard]] f64 total_energy(const ODEState& y) const {
        u32 nb=n(); f64 E=0;
        for(u32 i=0;i<nb;++i) {
            f64 vx=y[6*i+3],vy=y[6*i+4],vz=y[6*i+5];
            E+=0.5*bodies[i].mass*(vx*vx+vy*vy+vz*vz);
            for(u32 j=i+1;j<nb;++j) {
                f64 dx=y[6*j]-y[6*i],dy_=y[6*j+1]-y[6*i+1],dz=y[6*j+2]-y[6*i+2];
                f64 r=std::sqrt(dx*dx+dy_*dy_+dz*dz+1e-20);
                E-=G*bodies[i].mass*bodies[j].mass/r;
            }
        }
        return E;
    }

    // Angular momentum
    [[nodiscard]] Vec3<f64> angular_momentum(const ODEState& y) const {
        Vec3<f64> L{};
        for(u32 i=0;i<n();++i) {
            Vec3<f64> r{y[6*i],y[6*i+1],y[6*i+2]};
            Vec3<f64> v{y[6*i+3],y[6*i+4],y[6*i+5]};
            L = L + cross(r,v)*bodies[i].mass;
        }
        return L;
    }
};

// SIR epidemic model: dS/dt=-βSI/N, dI/dt=βSI/N-γI, dR/dt=γI
struct SIR {
    f64 beta{0.3},gamma{0.1};
    [[nodiscard]] ODERhs rhs() const {
        return [b=beta,g=gamma](f64, const ODEState& y) -> ODEState {
            f64 N=y[0]+y[1]+y[2];
            f64 new_inf=b*y[0]*y[1]/N;
            return {-new_inf, new_inf-g*y[1], g*y[1]};
        };
    }
    [[nodiscard]] f64 R0() const { return beta/gamma; }
    [[nodiscard]] f64 herd_immunity_threshold() const { return 1.0-1.0/R0(); }
};

// Double pendulum (chaotic)
struct DoublePendulum {
    f64 m1{1},m2{1},l1{1},l2{1},g{9.81};
    [[nodiscard]] ODERhs rhs() const {
        return [m1=m1,m2=m2,l1=l1,l2=l2,g=g](f64,const ODEState& y)->ODEState {
            f64 th1=y[0],w1=y[1],th2=y[2],w2=y[3];
            f64 dth=th2-th1, cdt=std::cos(dth), sdt=std::sin(dth);
            f64 den1=(m1+m2)*l1-m2*l1*cdt*cdt;
            f64 den2=l2/l1*den1;
            f64 dw1=(m2*l1*w1*w1*sdt*cdt+m2*g*std::sin(th2)*cdt+m2*l2*w2*w2*sdt-(m1+m2)*g*std::sin(th1))/den1;
            f64 dw2=(-m2*l2*w2*w2*sdt*cdt+(m1+m2)*(g*std::sin(th1)*cdt-l1*w1*w1*sdt-g*std::sin(th2)))/den2;
            return {w1,dw1,w2,dw2};
        };
    }
    [[nodiscard]] f64 energy(const ODEState& y) const {
        f64 th1=y[0],w1=y[1],th2=y[2],w2=y[3];
        f64 x1=l1*std::sin(th1),y1=-l1*std::cos(th1);
        f64 x2=x1+l2*std::sin(th2),y2=y1-l2*std::cos(th2);
        f64 vx1=l1*w1*std::cos(th1),vy1=l1*w1*std::sin(th1);
        f64 vx2=vx1+l2*w2*std::cos(th2),vy2=vy1+l2*w2*std::sin(th2);
        f64 KE=0.5*m1*(vx1*vx1+vy1*vy1)+0.5*m2*(vx2*vx2+vy2*vy2);
        f64 PE=(m1*y1+m2*y2)*g;
        return KE+PE;
    }
};

// Symplectic Euler (energy-conserving for Hamiltonian systems)
[[nodiscard]] inline ODEState symplectic_euler_step(
    const ODERhs& dq_dt, const ODERhs& dp_dt,
    f64 t, const ODEState& q, const ODEState& p, f64 h)
{
    auto p_new = ode_add(p, dp_dt(t,q), h);              // update p first
    auto q_new = ode_add(q, dq_dt(t+h,p_new), h);        // then q with new p
    ODEState state; state.insert(state.end(),q_new.begin(),q_new.end());
    state.insert(state.end(),p_new.begin(),p_new.end());
    return state;
}

// Verlet integrator (position Verlet for 2nd-order ODEs)
struct VerletState { ODEState q, v; };
[[nodiscard]] inline VerletState verlet_step(
    const std::function<ODEState(const ODEState&)>& accel,
    const VerletState& s, f64 h)
{
    auto a0=accel(s.q);
    ODEState q_new=ode_add(ode_add(s.q,s.v,h),a0,0.5*h*h);
    auto a1=accel(q_new);
    ODEState v_new(s.v.size());
    for(u32 i=0;i<u32(s.v.size());++i) v_new[i]=s.v[i]+0.5*h*(a0[i]+a1[i]);
    return {q_new,v_new};
}

} // namespace qs
