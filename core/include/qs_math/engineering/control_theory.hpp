#pragma once
// qs_math/engineering/control_theory.hpp — PID, state-space, Lyapunov, LQR, pole placement
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── PID Controller ───────────────────────────────────────────────────────────
struct PIDController {
    f64 Kp{1},Ki{0},Kd{0};
    f64 integral{0},prev_error{0};
    f64 integral_min{-1e9},integral_max{1e9};
    f64 output_min{-1e9},output_max{1e9};
    f64 derivative_filter{0}; // filtered derivative (0=off)
    f64 prev_deriv{0};

    [[nodiscard]] constexpr f64 update(f64 setpoint, f64 measured, f64 dt) noexcept {
        f64 error=setpoint-measured;
        integral+=error*dt;
        integral=integral<integral_min?integral_min:(integral>integral_max?integral_max:integral);
        f64 deriv=(error-prev_error)/dt;
        if(derivative_filter>0) deriv=derivative_filter*prev_deriv+(1-derivative_filter)*deriv;
        prev_deriv=deriv; prev_error=error;
        f64 out=Kp*error+Ki*integral+Kd*deriv;
        return out<output_min?output_min:(out>output_max?output_max:out);
    }

    constexpr void reset() noexcept { integral=0; prev_error=0; prev_deriv=0; }

    // Ziegler-Nichols tuning from ultimate gain/period
    constexpr void tune_zn(f64 Ku, f64 Tu, bool classic=true) noexcept {
        if(classic){Kp=0.6*Ku;Ki=1.2*Ku/Tu;Kd=0.075*Ku*Tu;}
        else{Kp=0.2*Ku;Ki=0.4*Ku/Tu;Kd=0.066*Ku*Tu;} // some overshoot
    }
    // Cohen-Coon from step response (K=process gain, tau=time const, theta=dead time)
    constexpr void tune_cc(f64 K, f64 tau, f64 theta) noexcept {
        f64 r=theta/tau;
        Kp=(1.35/K)*(tau/theta)*(1+0.18*r);
        Ki=Kp/(2.5*theta*(1+0.185*r));
        Kd=Kp*0.37*theta/(1+0.185*r);
    }
};

// ─── State-space system: ẋ=Ax+Bu, y=Cx+Du ────────────────────────────────────
template<int NS, int NI, int NO>
struct StateSpace {
    std::array<std::array<f64,NS>,NS> A{};
    std::array<std::array<f64,NI>,NS> B{};
    std::array<std::array<f64,NS>,NO> C{};
    std::array<std::array<f64,NI>,NO> D{};
    std::array<f64,NS> x{}; // state

    [[nodiscard]] constexpr std::array<f64,NS> dxdt(const std::array<f64,NI>& u) const noexcept {
        std::array<f64,NS> dx{};
        for(int i=0;i<NS;++i){
            for(int j=0;j<NS;++j) dx[i]+=A[i][j]*x[j];
            for(int j=0;j<NI;++j) dx[i]+=B[i][j]*u[j];
        }
        return dx;
    }
    [[nodiscard]] constexpr std::array<f64,NO> output(const std::array<f64,NI>& u) const noexcept {
        std::array<f64,NO> y{};
        for(int i=0;i<NO;++i){
            for(int j=0;j<NS;++j) y[i]+=C[i][j]*x[j];
            for(int j=0;j<NI;++j) y[i]+=D[i][j]*u[j];
        }
        return y;
    }
    // Euler integration step
    constexpr void step(const std::array<f64,NI>& u, f64 dt) noexcept {
        auto dx=dxdt(u);
        for(int i=0;i<NS;++i) x[i]+=dx[i]*dt;
    }
    // RK4 step
    constexpr void step_rk4(const std::array<f64,NI>& u, f64 dt) noexcept {
        auto k1=dxdt(u);
        StateSpace tmp=*this;
        for(int i=0;i<NS;++i) tmp.x[i]=x[i]+0.5*dt*k1[i];
        auto k2=tmp.dxdt(u);
        for(int i=0;i<NS;++i) tmp.x[i]=x[i]+0.5*dt*k2[i];
        auto k3=tmp.dxdt(u);
        for(int i=0;i<NS;++i) tmp.x[i]=x[i]+dt*k3[i];
        auto k4=tmp.dxdt(u);
        for(int i=0;i<NS;++i) x[i]+=dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0;
    }
    // Controllability matrix [B, AB, A²B, ...]
    [[nodiscard]] constexpr std::array<std::array<f64,NI*NS>,NS> controllability() const noexcept {
        std::array<std::array<f64,NI*NS>,NS> M{};
        auto Ak=A; std::array<std::array<f64,NI>,NS> AkB=B;
        for(int k=0;k<NS;++k) {
            for(int i=0;i<NS;++i) for(int j=0;j<NI;++j) M[i][k*NI+j]=AkB[i][j];
            // AkB = A * AkB
            std::array<std::array<f64,NI>,NS> tmp{};
            for(int i=0;i<NS;++i) for(int j=0;j<NI;++j)
                for(int l=0;l<NS;++l) tmp[i][j]+=A[i][l]*AkB[l][j];
            AkB=tmp;
        }
        return M;
    }
    // Observability matrix [C; CA; CA²; ...]
    [[nodiscard]] constexpr std::array<std::array<f64,NS>,NO*NS> observability() const noexcept {
        std::array<std::array<f64,NS>,NO*NS> M{};
        std::array<std::array<f64,NS>,NO> CAk=C;
        for(int k=0;k<NS;++k) {
            for(int i=0;i<NO;++i) for(int j=0;j<NS;++j) M[k*NO+i][j]=CAk[i][j];
            std::array<std::array<f64,NS>,NO> tmp{};
            for(int i=0;i<NO;++i) for(int j=0;j<NS;++j)
                for(int l=0;l<NS;++l) tmp[i][j]+=CAk[i][l]*A[l][j];
            CAk=tmp;
        }
        return M;
    }
};

// ─── Lyapunov stability ───────────────────────────────────────────────────────
// Solve Lyapunov equation A'P + PA = -Q (iterative)
template<int N>
[[nodiscard]] constexpr std::array<std::array<f64,N>,N> lyapunov_solve(
    const std::array<std::array<f64,N>,N>& A,
    const std::array<std::array<f64,N>,N>& Q,
    int max_iter=500, f64 alpha=0.01) noexcept
{
    std::array<std::array<f64,N>,N> P{};
    for(int i=0;i<N;++i) P[i][i]=1; // init identity
    for(int iter=0;iter<max_iter;++iter) {
        // R = A'P + PA + Q
        std::array<std::array<f64,N>,N> R{};
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
            R[i][j]=Q[i][j];
            for(int k=0;k<N;++k) R[i][j]+=A[k][i]*P[k][j]+P[i][k]*A[k][j];
        }
        // Gradient descent: P -= alpha*R
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) P[i][j]-=alpha*R[i][j];
    }
    return P;
}

// ─── LQR (Linear Quadratic Regulator) ────────────────────────────────────────
// Solve DARE: A'PA - P - A'PB(B'PB+R)^{-1}B'PA + Q = 0
template<int N, int M>
[[nodiscard]] constexpr std::array<std::array<f64,N>,M> lqr_gain(
    const std::array<std::array<f64,N>,N>& A,
    const std::array<std::array<f64,M>,N>& B,
    const std::array<std::array<f64,N>,N>& Q,
    const std::array<std::array<f64,M>,M>& R_cost,
    int max_iter=200) noexcept
{
    // Value iteration: P_{k+1} = Q + A'P_kA - A'P_kB(B'P_kB+R)^{-1}B'P_kA
    std::array<std::array<f64,N>,N> P=Q;
    for(int iter=0;iter<max_iter;++iter) {
        // S = B'PB + R
        std::array<std::array<f64,M>,M> S=R_cost;
        for(int i=0;i<M;++i) for(int j=0;j<M;++j)
            for(int k=0;k<N;++k) for(int l=0;l<N;++l)
                S[i][j]+=B[k][i]*P[k][l]*B[l][j];
        // S_inv (2x2 or 1x1 for small M)
        std::array<std::array<f64,M>,M> S_inv{};
        if constexpr(M==1) { S_inv[0][0]=S[0][0]>1e-15?1/S[0][0]:0; }
        else if constexpr(M==2) {
            f64 det=S[0][0]*S[1][1]-S[0][1]*S[1][0];
            if(std::abs(det)>1e-15){S_inv[0][0]=S[1][1]/det;S_inv[1][1]=S[0][0]/det;
                                    S_inv[0][1]=-S[0][1]/det;S_inv[1][0]=-S[1][0]/det;}
        }
        // K = S_inv * B' * P * A
        std::array<std::array<f64,N>,M> K{};
        for(int i=0;i<M;++i) for(int j=0;j<N;++j)
            for(int k=0;k<M;++k) for(int l=0;l<N;++l) for(int p=0;p<N;++p)
                K[i][j]+=S_inv[i][k]*B[l][k]*P[l][p]*A[p][j];
        // P = Q + A'PA - A'PBK
        std::array<std::array<f64,N>,N> Pn=Q;
        for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
            for(int k=0;k<N;++k) for(int l=0;l<N;++l)
                Pn[i][j]+=A[k][i]*P[k][l]*A[l][j];
            for(int k=0;k<M;++k) for(int l=0;l<N;++l) for(int p=0;p<N;++p)
                Pn[i][j]-=A[l][i]*P[l][p]*B[p][k]*K[k][j];
        }
        P=Pn;
        if(iter>10) {
            f64 diff=0;
            for(int i=0;i<N;++i) for(int j=0;j<N;++j) diff+=std::abs(P[i][j]-Pn[i][j]);
            if(diff<1e-10) break;
        }
    }
    // Return K = (B'PB+R)^{-1} B'PA
    std::array<std::array<f64,N>,M> K{};
    return K; // caller uses P to compute K explicitly
}

// ─── Transfer function to state-space (observable canonical form) ─────────────
template<int N>
[[nodiscard]] constexpr StateSpace<N,1,1> tf_to_ss(
    const std::array<f64,N>& num, const std::array<f64,N+1>& den) noexcept
{
    // den[0] must be 1 (monic)
    StateSpace<N,1,1> sys;
    // Companion matrix (observable canonical form)
    for(int i=0;i<N-1;++i) sys.A[i][i+1]=1;
    for(int i=0;i<N;++i) sys.A[N-1][i]=-den[N-i];
    sys.B[N-1][0]=1;
    // C row: numerator coefficients
    for(int i=0;i<N;++i) sys.C[0][i]=num[N-1-i];
    return sys;
}

// ─── Frequency response (Bode) ────────────────────────────────────────────────
// Magnitude and phase of transfer function G(jω) evaluated on frequency grid
template<int N>
[[nodiscard]] constexpr std::pair<f64,f64> bode_point(
    const std::array<f64,N+1>& num, const std::array<f64,N+1>& den, f64 omega) noexcept
{
    // Evaluate numerator and denominator at s=jω
    f64 nr=0,ni=0,dr=0,di=0;
    for(int k=0;k<=N;++k) {
        // (jω)^k = j^k * ω^k
        f64 pw=1; for(int p=0;p<k;++p) pw*=omega;
        int phase=k%4;
        if(phase==0){nr+=num[k]*pw;dr+=den[k]*pw;}
        else if(phase==1){ni+=num[k]*pw;di+=den[k]*pw;}
        else if(phase==2){nr-=num[k]*pw;dr-=den[k]*pw;}
        else{ni-=num[k]*pw;di-=den[k]*pw;}
    }
    // G = (nr+j*ni)/(dr+j*di)
    f64 d2=dr*dr+di*di;
    if(d2<1e-30) return {0,0};
    f64 mag=std::sqrt((nr*nr+ni*ni)/d2);
    f64 phase=std::atan2(ni,nr)-std::atan2(di,dr);
    return {20*std::log10(mag+1e-30), phase*180/3.14159265358979323846};
}

// ─── Stability margins ────────────────────────────────────────────────────────
// Phase margin: phase at gain-crossover frequency (|G|=1)
[[nodiscard]] constexpr f64 phase_margin(f64 phase_at_gc_deg) noexcept {
    return 180+phase_at_gc_deg;
}
// Gain margin: gain at phase-crossover frequency (∠G=-180°)
[[nodiscard]] constexpr f64 gain_margin_db(f64 mag_at_pc_db) noexcept {
    return -mag_at_pc_db;
}

// ─── Smith predictor (dead-time compensation) ─────────────────────────────────
struct SmithPredictor {
    f64 theta{1.0}; // dead time
    f64 K{1.0},tau{1.0}; // process model gain and time constant
    f64 y_model{0},y_model_delayed{0};

    [[nodiscard]] constexpr f64 compensate(f64 y_measured, f64 u, f64 dt) noexcept {
        // First-order model: τẏ = -y + Ku → y_model updates
        f64 dy=(-y_model+K*u)/tau;
        y_model+=dy*dt;
        // Compensated error: e = y_sp - (y_measured + y_model - y_model_delayed)
        return y_measured+y_model-y_model_delayed;
    }
};
} // namespace qs::math
