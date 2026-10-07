#pragma once
// qs_math/control_systems_math.hpp — PID, state space, Kalman filter, LQR (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <span>

namespace qs {

// ─── PID controller ───────────────────────────────────────────────────────────

struct PIDParams {
    f32 kP{1.f}, kI{0.f}, kD{0.f};
    f32 integral_limit{1e6f};   // anti-windup
    f32 output_min{-1e6f};
    f32 output_max{ 1e6f};
    f32 derivative_filter{0.f}; // low-pass filter coefficient on derivative (0=off)
};

struct PIDState {
    f32 integral{0.f};
    f32 prev_error{0.f};
    f32 filtered_deriv{0.f};
    bool first{true};

    [[nodiscard]] f32 update(f32 error, f32 dt, const PIDParams& p) {
        // Derivative
        f32 deriv=0.f;
        if(!first) {
            f32 raw_deriv=(error-prev_error)/(dt+1e-10f);
            if(p.derivative_filter>0.f)
                filtered_deriv=filtered_deriv*(1.f-p.derivative_filter)+raw_deriv*p.derivative_filter;
            else filtered_deriv=raw_deriv;
            deriv=filtered_deriv;
        }
        first=false;
        // Integral with windup clamp
        integral=std::clamp(integral+error*dt,-p.integral_limit,p.integral_limit);
        prev_error=error;
        f32 out=p.kP*error+p.kI*integral+p.kD*deriv;
        return std::clamp(out,p.output_min,p.output_max);
    }

    void reset() { *this={}; }
};

// ─── Cascaded PID (position -> velocity -> acceleration) ──────────────────────

struct CascadedPID {
    PIDState outer, inner;
    PIDParams outer_p, inner_p;
    f32 inner_output_limit{10.f};

    [[nodiscard]] f32 update(f32 set_point, f32 position, f32 velocity, f32 dt) {
        outer_p.output_min=-inner_output_limit; outer_p.output_max=inner_output_limit;
        f32 vel_setpoint=outer.update(set_point-position, dt, outer_p);
        return inner.update(vel_setpoint-velocity, dt, inner_p);
    }
};

// ─── State space (discrete, fixed size via templates) ─────────────────────────

template<u32 N_STATES, u32 N_INPUTS, u32 N_OUTPUTS>
struct StateSpace {
    using StateVec  = std::array<f32,N_STATES>;
    using InputVec  = std::array<f32,N_INPUTS>;
    using OutputVec = std::array<f32,N_OUTPUTS>;

    f32 A[N_STATES][N_STATES]{};
    f32 B[N_STATES][N_INPUTS]{};
    f32 C[N_OUTPUTS][N_STATES]{};
    f32 D[N_OUTPUTS][N_INPUTS]{};

    [[nodiscard]] StateVec step(const StateVec& x, const InputVec& u) const {
        StateVec xn{};
        for(u32 i=0;i<N_STATES;++i){
            for(u32 j=0;j<N_STATES;++j) xn[i]+=A[i][j]*x[j];
            for(u32 j=0;j<N_INPUTS;++j) xn[i]+=B[i][j]*u[j];
        }
        return xn;
    }

    [[nodiscard]] OutputVec output(const StateVec& x, const InputVec& u) const {
        OutputVec y{};
        for(u32 i=0;i<N_OUTPUTS;++i){
            for(u32 j=0;j<N_STATES;++j) y[i]+=C[i][j]*x[j];
            for(u32 j=0;j<N_INPUTS;++j) y[i]+=D[i][j]*u[j];
        }
        return y;
    }
};

// ─── Kalman filter (1D) ───────────────────────────────────────────────────────

struct KalmanFilter1D {
    f32 x{0.f};     // state estimate
    f32 P{1.f};     // estimate covariance
    f32 Q{0.001f};  // process noise
    f32 R{0.1f};    // measurement noise

    void predict(f32 F=1.f, f32 B=0.f, f32 u=0.f) {
        x=F*x+B*u;
        P=F*F*P+Q;
    }

    [[nodiscard]] f32 update(f32 z) {
        f32 K=P/(P+R);
        x=x+K*(z-x);
        P=(1.f-K)*P;
        return x;
    }
};

// ─── Kalman filter (n-dimensional) ────────────────────────────────────────────

struct KalmanFilterND {
    u32 n{};  // state dim
    std::vector<f32> x;   // state estimate [n]
    std::vector<f32> P;   // covariance [n*n]
    std::vector<f32> Q;   // process noise [n*n]
    std::vector<f32> R;   // measurement noise [m*m]
    std::vector<f32> F;   // state transition [n*n]
    std::vector<f32> H;   // observation [m*n]
    u32 m{};  // measurement dim

    void init(u32 state_dim, u32 meas_dim) {
        n=state_dim; m=meas_dim;
        x.assign(n,0.f); P.assign(n*n,0.f);
        Q.assign(n*n,0.f); R.assign(m*m,0.f);
        F.assign(n*n,0.f); H.assign(m*n,0.f);
        for(u32 i=0;i<n;++i) P[i*n+i]=1.f;
        for(u32 i=0;i<n;++i) F[i*n+i]=1.f;
    }

    // Matrix multiply: C = A (ra x rc) * B (rc x rb)
    static void matmul(const f32*A,const f32*B,f32*C,u32 ra,u32 rc,u32 rb) {
        for(u32 i=0;i<ra;++i) for(u32 j=0;j<rb;++j) {
            f32 s=0.f; for(u32 k=0;k<rc;++k) s+=A[i*rc+k]*B[k*rb+j];
            C[i*rb+j]=s;
        }
    }

    void predict() {
        // x = F*x
        std::vector<f32> xn(n,0.f);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) xn[i]+=F[i*n+j]*x[j];
        x=xn;
        // P = F*P*Ft + Q
        std::vector<f32> FP(n*n), FPFt(n*n);
        matmul(F.data(),P.data(),FP.data(),n,n,n);
        std::vector<f32> Ft(n*n);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) Ft[j*n+i]=F[i*n+j];
        matmul(FP.data(),Ft.data(),FPFt.data(),n,n,n);
        for(u32 i=0;i<n*n;++i) P[i]=FPFt[i]+Q[i];
    }

    void update(std::span<const f32> z) {
        // y = z - H*x
        std::vector<f32> y(m);
        for(u32 i=0;i<m;++i){
            y[i]=z[i];
            for(u32 j=0;j<n;++j) y[i]-=H[i*n+j]*x[j];
        }
        // S = H*P*Ht + R
        std::vector<f32> HP(m*n), Ht(n*m), S(m*m);
        matmul(H.data(),P.data(),HP.data(),m,n,n);
        for(u32 i=0;i<m;++i) for(u32 j=0;j<n;++j) Ht[j*m+i]=H[i*n+j];
        matmul(HP.data(),Ht.data(),S.data(),m,n,m);
        for(u32 i=0;i<m;++i) S[i*m+i]+=R[i*m+i];
        // K = P*Ht*S^-1 (simplified: diagonal S)
        std::vector<f32> PHt(n*m), K(n*m);
        matmul(P.data(),Ht.data(),PHt.data(),n,n,m);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<m;++j)
            K[i*m+j]=PHt[i*m+j]/(S[j*m+j]+1e-10f);
        // x = x + K*y
        for(u32 i=0;i<n;++i) for(u32 j=0;j<m;++j) x[i]+=K[i*m+j]*y[j];
        // P = (I-K*H)*P
        std::vector<f32> KH(n*n),I_KH(n*n);
        matmul(K.data(),H.data(),KH.data(),n,m,n);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
            I_KH[i*n+j]=(i==j?1.f:0.f)-KH[i*n+j];
        std::vector<f32> newP(n*n);
        matmul(I_KH.data(),P.data(),newP.data(),n,n,n);
        P=newP;
    }
};

// ─── LQR gain computation (Riccati iteration) ────────────────────────────────

// Discrete-time LQR via value iteration (solves DARE)
// Returns gain K such that u = -K*x minimizes sum(x'Qx + u'Ru)
[[nodiscard]] inline std::vector<f32> lqr_gain(
    const std::vector<f32>& A, const std::vector<f32>& B,
    const std::vector<f32>& Q, const std::vector<f32>& Rv,
    u32 n, u32 m, u32 iterations=100)
{
    std::vector<f32> P=Q;
    std::vector<f32> K(m*n,0.f);
    auto mm=[&](const f32*A2,const f32*B2,f32*C2,u32 r,u32 c2,u32 rb){
        for(u32 i=0;i<r;++i) for(u32 j=0;j<rb;++j){
            f32 s=0.f; for(u32 k=0;k<c2;++k) s+=A2[i*c2+k]*B2[k*rb+j]; C2[i*rb+j]=s;}};
    for(u32 iter=0;iter<iterations;++iter) {
        // Bt*P*B + R
        std::vector<f32> Bt(m*n),BtP(m*n),BtPB(m*m),BtPBR(m*m);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<m;++j) Bt[j*n+i]=B[i*m+j];
        mm(Bt.data(),P.data(),BtP.data(),m,n,n);
        mm(BtP.data(),B.data(),BtPB.data(),m,n,m);
        for(u32 i=0;i<m*m;++i) BtPBR[i]=BtPB[i]+Rv[i];
        // K = inv(BtPBR)*Bt*P*A  (simplified: diagonal BtPBR)
        std::vector<f32> At(n*n);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) At[j*n+i]=A[i*n+j];
        std::vector<f32> BtPA(m*n); mm(BtP.data(),A.data(),BtPA.data(),m,n,n);
        for(u32 i=0;i<m;++i) for(u32 j=0;j<n;++j)
            K[i*n+j]=BtPA[i*n+j]/(BtPBR[i*m+i]+1e-10f);
        // P = Q + At*P*A - At*P*B*K
        std::vector<f32> AtP(n*n),AtPA(n*n),AtPBK(n*n),newP(n*n);
        mm(At.data(),P.data(),AtP.data(),n,n,n);
        mm(AtP.data(),A.data(),AtPA.data(),n,n,n);
        std::vector<f32> PBK(n*n); std::vector<f32> PB(n*m);
        mm(P.data(),B.data(),PB.data(),n,n,m);
        mm(PB.data(),K.data(),PBK.data(),n,m,n);
        std::vector<f32> AtPBK2(n*n); mm(At.data(),PBK.data(),AtPBK2.data(),n,n,n);
        for(u32 i=0;i<n*n;++i) newP[i]=Q[i]+AtPA[i]-AtPBK2[i];
        P=newP;
    }
    return K;
}

// ─── Spring-damper (second-order system) ─────────────────────────────────────

struct SpringDamper {
    f32 omega_n{10.f};   // natural frequency
    f32 zeta{0.7f};      // damping ratio (1=critical, <1=underdamped)
    f32 x{0.f};          // position
    f32 v{0.f};          // velocity

    [[nodiscard]] f32 update(f32 target, f32 dt) {
        f32 F=-omega_n*omega_n*(x-target)-2.f*zeta*omega_n*v;
        v+=F*dt; x+=v*dt;
        return x;
    }

    void set(f32 pos, f32 vel=0.f){x=pos;v=vel;}
};

} // namespace qs
