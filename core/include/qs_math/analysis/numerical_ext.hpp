#pragma once
// numerical_ext.hpp — Extended numerical methods: ODE solvers, optimization,
//                      sparse matrix helpers, Fourier, SVD power iteration,
//                      constraint solvers, Gauss-Seidel, conjugate gradient
#include "numerical.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include "mat4.hpp"
#include "scalar_math.hpp"

namespace qs::math::numerical {

// ---------------------------------------------------------------------------
// ODE Solvers
// ---------------------------------------------------------------------------
// f(t, y) -> dy/dt
using ODE1D = f64(*)(f64, f64);
using ODE3D = Vec3d(*)(f64, Vec3d);

// Euler method 1D
[[nodiscard]] inline f64 euler_1d(ODE1D f, f64 t0, f64 y0, f64 dt) noexcept {
    return y0+dt*f(t0,y0);
}

// RK4 1D
[[nodiscard]] inline f64 rk4_1d(ODE1D f, f64 t, f64 y, f64 dt) noexcept {
    f64 k1=f(t,y), k2=f(t+dt/2,y+dt*k1/2);
    f64 k3=f(t+dt/2,y+dt*k2/2), k4=f(t+dt,y+dt*k3);
    return y+dt*(k1+2*k2+2*k3+k4)/6;
}

// RK4 3D (vector-valued ODE — for trajectory simulation)
[[nodiscard]] inline Vec3d rk4_3d(ODE3D f, f64 t, Vec3d y, f64 dt) noexcept {
    Vec3d k1=f(t,y), k2=f(t+dt/2,y+k1*(dt/2));
    Vec3d k3=f(t+dt/2,y+k2*(dt/2)), k4=f(t+dt,y+k3*dt);
    return y+(k1+k2*2.0+k3*2.0+k4)*(dt/6);
}

// Runge-Kutta-Fehlberg (adaptive step size)
struct RKF45Result { f64 y; f64 h_new; bool accepted; };
[[nodiscard]] inline RKF45Result rkf45(ODE1D f, f64 t, f64 y, f64 h,
                                        f64 tol=1e-6) noexcept {
    static constexpr f64 c2=0.25,c3=3.0/8,c4=12.0/13,c5=1.0,c6=0.5;
    f64 k1=h*f(t,y), k2=h*f(t+c2*h,y+c2*k1);
    f64 k3=h*f(t+c3*h,y+(3*k1+9*k2)/32);
    f64 k4=h*f(t+c4*h,y+(1932*k1-7200*k2+7296*k3)/2197);
    f64 k5=h*f(t+c5*h,y+(439.0/216)*k1-8*k2+(3680.0/513)*k3-(845.0/4104)*k4);
    f64 k6=h*f(t+c6*h,y-(8.0/27)*k1+2*k2-(3544.0/2565)*k3+(1859.0/4104)*k4-0.275*k5);
    // 4th order
    f64 y4=y+(25.0/216)*k1+(1408.0/2565)*k3+(2197.0/4101)*k4-0.2*k5;
    // 5th order
    f64 y5=y+(16.0/135)*k1+(6656.0/12825)*k3+(28561.0/56430)*k4-0.18*k5+(2.0/55)*k6;
    f64 err=abs(y5-y4);
    f64 h_new=h*0.9*pow(tol/max(err,1e-10),0.2);
    return {y4,h_new,err<=tol};
}

// ---------------------------------------------------------------------------
// Optimization
// ---------------------------------------------------------------------------
// Gradient descent
template<typename Fn, typename GradFn>
[[nodiscard]] inline f64 gradient_descent(Fn&& f, GradFn&& grad,
                                           f64 x0, f64 lr=0.01, i32 max_iter=1000,
                                           f64 tol=1e-8) noexcept {
    f64 x=x0;
    for(i32 i=0;i<max_iter;++i){
        f64 g=grad(x);
        x-=lr*g;
        if(abs(g)<tol) break;
    }
    return x;
}

// Adam optimizer (1D)
struct AdamState1D {
    f64 m{0},v{0},t{0};
    f64 alpha{0.001},beta1{0.9},beta2{0.999},eps{1e-8};
};
[[nodiscard]] inline f64 adam_step(AdamState1D& s, f64 x, f64 grad) noexcept {
    ++s.t;
    s.m=s.beta1*s.m+(1-s.beta1)*grad;
    s.v=s.beta2*s.v+(1-s.beta2)*grad*grad;
    f64 m_hat=s.m/(1-pow(s.beta1,s.t));
    f64 v_hat=s.v/(1-pow(s.beta2,s.t));
    return x-s.alpha*m_hat/(sqrt(v_hat)+s.eps);
}

// Golden section search (1D minimization on [a,b])
template<typename Fn>
[[nodiscard]] inline f64 golden_section(Fn&& f, f64 a, f64 b, f64 tol=1e-8) noexcept {
    const f64 phi=(sqrt(5.0)-1.0)/2.0;
    f64 c=b-phi*(b-a), d=a+phi*(b-a);
    while(abs(b-a)>tol){
        if(f(c)<f(d)){ b=d; d=c; c=b-phi*(b-a); }
        else          { a=c; c=d; d=a+phi*(b-a); }
    }
    return (a+b)/2;
}

// Nelder-Mead simplex (2D minimization)
template<typename Fn>
[[nodiscard]] inline Vec2d nelder_mead_2d(Fn&& f, Vec2d x0,
                                          f64 step=0.1, i32 max_iter=200) noexcept {
    Vec2d s[3]={x0,{x0.x+step,x0.y},{x0.x,x0.y+step}};
    for(i32 iter=0;iter<max_iter;++iter){
        // Sort by f value
        for(i32 i=0;i<2;++i) for(i32 j=i+1;j<3;++j)
            if(f(s[j])<f(s[i])){Vec2d t=s[i];s[i]=s[j];s[j]=t;}
        Vec2d centroid=(s[0]+s[1])*(1.0/2.0);
        Vec2d reflected=centroid*2.0-s[2];
        if(f(reflected)<f(s[1])){ s[2]=reflected; continue; }
        Vec2d contracted=centroid*(1.5)-s[2]*(0.5);
        if(f(contracted)<f(s[2])){ s[2]=contracted; continue; }
        s[1]=(s[0]+s[1])*0.5; s[2]=(s[0]+s[2])*0.5;
        if((s[0]-s[2]).length()<1e-10) break;
    }
    return s[0];
}

// ---------------------------------------------------------------------------
// Linear systems
// ---------------------------------------------------------------------------
// Gaussian elimination with partial pivoting (n<=8 systems)
constexpr i32 MAX_SYS=8;
[[nodiscard]] inline bool gauss_solve(f64 A[MAX_SYS][MAX_SYS], f64 b[MAX_SYS],
                                       f64 x[MAX_SYS], i32 n) noexcept {
    for(i32 p=0;p<n;++p){
        // Pivot
        i32 mx=p;
        for(i32 i=p+1;i<n;++i) if(abs(A[i][p])>abs(A[mx][p])) mx=i;
        for(i32 c=0;c<n;++c){f64 t=A[p][c];A[p][c]=A[mx][c];A[mx][c]=t;}
        f64 tb=b[p];b[p]=b[mx];b[mx]=tb;
        if(abs(A[p][p])<1e-12) return false;
        for(i32 r=p+1;r<n;++r){
            f64 f=A[r][p]/A[p][p];
            for(i32 c=p;c<n;++c) A[r][c]-=f*A[p][c];
            b[r]-=f*b[p];
        }
    }
    for(i32 p=n-1;p>=0;--p){
        x[p]=b[p];
        for(i32 c=p+1;c<n;++c) x[p]-=A[p][c]*x[c];
        x[p]/=A[p][p];
    }
    return true;
}

// Conjugate gradient (for symmetric positive definite systems)
// Ax = b, where A is n×n SPD, n <= 64
inline bool conjugate_gradient(const f64* A, const f64* b, f64* x, i32 n,
                                f64 tol=1e-10, i32 max_iter=100) noexcept {
    constexpr i32 MAX_N=64;
    if(n>MAX_N) return false;
    f64 r[MAX_N],p[MAX_N],Ap[MAX_N];
    for(i32 i=0;i<n;++i){x[i]=0.0;r[i]=b[i];p[i]=b[i];}
    f64 rsq=0.0; for(i32 i=0;i<n;++i) rsq+=r[i]*r[i];
    for(i32 iter=0;iter<max_iter;++iter){
        for(i32 i=0;i<n;++i){ Ap[i]=0.0; for(i32 j=0;j<n;++j) Ap[i]+=A[i*n+j]*p[j]; }
        f64 pAp=0.0; for(i32 i=0;i<n;++i) pAp+=p[i]*Ap[i];
        if(abs(pAp)<1e-14) break;
        f64 alpha=rsq/pAp;
        for(i32 i=0;i<n;++i){ x[i]+=alpha*p[i]; r[i]-=alpha*Ap[i]; }
        f64 rsq_new=0.0; for(i32 i=0;i<n;++i) rsq_new+=r[i]*r[i];
        if(rsq_new<tol*tol) return true;
        f64 beta=rsq_new/rsq; rsq=rsq_new;
        for(i32 i=0;i<n;++i) p[i]=r[i]+beta*p[i];
    }
    return true;
}

// Gauss-Seidel iteration
inline bool gauss_seidel(const f64* A, const f64* b, f64* x, i32 n,
                          f64 tol=1e-8, i32 max_iter=100) noexcept {
    for(i32 iter=0;iter<max_iter;++iter){
        f64 max_diff=0.0;
        for(i32 i=0;i<n;++i){
            f64 sum=b[i];
            for(i32 j=0;j<n;++j) if(j!=i) sum-=A[i*n+j]*x[j];
            f64 xn=abs(A[i*n+i])>1e-14?sum/A[i*n+i]:0.0;
            max_diff=max(max_diff,abs(xn-x[i]));
            x[i]=xn;
        }
        if(max_diff<tol) return true;
    }
    return false;
}

// Power iteration for dominant eigenvalue
[[nodiscard]] inline f64 power_iteration(const f64* A, i32 n, f64* eigvec,
                                          i32 max_iter=100, f64 tol=1e-8) noexcept {
    constexpr i32 MAX_N=32;
    if(n>MAX_N) return 0.0;
    f64 v[MAX_N]; for(i32 i=0;i<n;++i) v[i]=(i==0)?1.0:0.0;
    f64 lam=0.0;
    for(i32 iter=0;iter<max_iter;++iter){
        f64 w[MAX_N]={}; for(i32 i=0;i<n;++i) for(i32 j=0;j<n;++j) w[i]+=A[i*n+j]*v[j];
        f64 norm=0.0; for(i32 i=0;i<n;++i) norm+=w[i]*w[i]; norm=sqrt(norm);
        if(norm<1e-14) break;
        f64 lam_new=0.0;
        for(i32 i=0;i<n;++i){ v[i]=w[i]/norm; lam_new+=v[i]*w[i]; }
        if(abs(lam_new-lam)<tol){ lam=lam_new; break; }
        lam=lam_new;
    }
    for(i32 i=0;i<n;++i) eigvec[i]=v[i];
    return lam;
}

// ---------------------------------------------------------------------------
// Interpolating polynomial (Lagrange)
// ---------------------------------------------------------------------------
[[nodiscard]] inline f64 lagrange_interpolate(const f64* xs, const f64* ys,
                                               i32 n, f64 x) noexcept {
    f64 result=0.0;
    for(i32 i=0;i<n;++i){
        f64 term=ys[i];
        for(i32 j=0;j<n;++j){
            if(j!=i){
                f64 denom=xs[i]-xs[j];
                if(abs(denom)<1e-14) continue;
                term*=(x-xs[j])/denom;
            }
        }
        result+=term;
    }
    return result;
}

// Chebyshev nodes for stable polynomial interpolation
inline void chebyshev_nodes(f64* out, i32 n, f64 a, f64 b) noexcept {
    for(i32 i=0;i<n;++i)
        out[i]=(a+b)/2+(b-a)/2*cos(PI*(2*i+1)/(2*n));
}

} // namespace qs::math::numerical
