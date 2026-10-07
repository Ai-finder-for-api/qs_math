#pragma once
// qs_math/numerical/integration.hpp — Numerical integration: Gaussian quadrature,
//   adaptive Simpson, Romberg, Gauss-Kronrod, Monte Carlo, quasi-MC,
//   multi-dimensional cubature, path integrals (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/random.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <array>
#include <numeric>

namespace qs::math::numerical {

using Fn1D   = std::function<f64(f64)>;
using Fn2D   = std::function<f64(f64,f64)>;
using Fn3D   = std::function<f64(f64,f64,f64)>;
using FnNd   = std::function<f64(const std::vector<f64>&)>;

struct IntegralResult { f64 value, error; int evals; bool converged; };

// ─── Gauss-Legendre (fixed order) ────────────────────────────────────────
// 5-point rule on [a,b]
[[nodiscard]] inline f64 gauss_legendre5(Fn1D f, f64 a, f64 b) noexcept {
    static constexpr f64 x[]={0.0,0.538469310105683,0.90617984593866};
    static constexpr f64 w[]={0.568888888888889,0.478628670499366,0.236926885056189};
    f64 mid=0.5*(a+b), half=0.5*(b-a);
    f64 sum=w[0]*f(mid);
    for(int i=1;i<3;++i)
        sum+=w[i]*(f(mid+half*x[i])+f(mid-half*x[i]));
    return sum*half;
}

// 10-point rule
[[nodiscard]] inline f64 gauss_legendre10(Fn1D f, f64 a, f64 b) noexcept {
    static constexpr f64 X[]={0.1488743389816312,0.4333953941292472,
                               0.6794095682990244,0.8650633666889845,0.9739065285171717};
    static constexpr f64 W[]={0.2955242247147529,0.2692667193099963,
                               0.2190863625159820,0.1494513491505806,0.0666713443086881};
    f64 mid=0.5*(a+b), half=0.5*(b-a), sum=0;
    for(int i=0;i<5;++i)
        sum+=W[i]*(f(mid+half*X[i])+f(mid-half*X[i]));
    return sum*half;
}

// ─── Gauss-Kronrod G7K15 ─────────────────────────────────────────────────
[[nodiscard]] inline IntegralResult gauss_kronrod15(Fn1D f, f64 a, f64 b) noexcept {
    static constexpr f64 xgk[]={0.0,0.2077849550078985,0.4058451513773832,
                                  0.5860872354676911,0.7415311855993945,
                                  0.8648644233597691,0.9491079123427585,
                                  0.9914553711208126};
    static constexpr f64 wgk[]={0.2094821410847278,0.2044329400752989,
                                  0.1903505780647854,0.1690047266392679,
                                  0.1406532597155259,0.1047900103222502,
                                  0.0630920926299786,0.0229353220105292};
    static constexpr f64 wg7[]={0.4179591836734694,0.3818300505051189,
                                  0.2797053914892767,0.1294849661688697};
    f64 mid=0.5*(a+b), half=0.5*(b-a);
    f64 fmid=f(mid);
    f64 result15=wgk[0]*fmid, result7=wg7[0]*fmid;
    for(int i=1;i<8;++i){
        f64 fp=f(mid+half*xgk[i]), fm=f(mid-half*xgk[i]);
        result15+=wgk[i]*(fp+fm);
        if(i%2==1) result7+=wg7[i/2]*(fp+fm);
    }
    result15*=half; result7*=half;
    f64 err=std::abs(result15-result7);
    return {result15, err, 15, true};
}

// ─── Adaptive Simpson ─────────────────────────────────────────────────────
[[nodiscard]] inline f64 simpson(Fn1D f, f64 a, f64 b, f64 fa, f64 fm, f64 fb) noexcept {
    return (b-a)*(fa+4.0*fm+fb)/6.0;
}

struct AdaptResult { f64 value; int evals; };

inline void adaptive_simpson_rec(
    Fn1D& f, f64 a, f64 b, f64 fa, f64 fm, f64 fb,
    f64 whole, f64 tol, int depth, f64& result, int& evals) noexcept
{
    f64 m1=0.5*(a+0.5*(a+b)), m2=0.5*(0.5*(a+b)+b);
    f64 fm1=f(m1), fm2=f(m2); evals+=2;
    f64 fmid=f(0.5*(a+b)); ++evals;
    f64 left=simpson(f,a,0.5*(a+b),fa,fm1,fmid);
    f64 right=simpson(f,0.5*(a+b),b,fmid,fm2,fb);
    f64 delta=(left+right-whole)/15.0;
    if(depth<=0||std::abs(delta)<tol){result+=left+right+delta;return;}
    adaptive_simpson_rec(f,a,0.5*(a+b),fa,fm1,fmid,left,tol/2,depth-1,result,evals);
    adaptive_simpson_rec(f,0.5*(a+b),b,fmid,fm2,fb,right,tol/2,depth-1,result,evals);
}

[[nodiscard]] inline IntegralResult adaptive_simpson(
    Fn1D f, f64 a, f64 b, f64 tol=1e-8, int max_depth=20) noexcept
{
    f64 fa=f(a), fm=f(0.5*(a+b)), fb=f(b);
    f64 whole=simpson(f,a,b,fa,fm,fb);
    f64 result=0; int evals=3;
    adaptive_simpson_rec(f,a,b,fa,fm,fb,whole,tol,max_depth,result,evals);
    return {result,0,evals,true};
}

// ─── Romberg integration ──────────────────────────────────────────────────
[[nodiscard]] inline IntegralResult romberg(
    Fn1D f, f64 a, f64 b, int max_order=12, f64 tol=1e-10) noexcept
{
    int evals=0;
    std::vector<std::vector<f64>> R(max_order, std::vector<f64>(max_order,0));
    R[0][0]=(b-a)*0.5*(f(a)+f(b)); evals+=2;
    for(int n=1;n<max_order;++n){
        int pts=1<<n;
        f64 h=(b-a)/pts, sum=0;
        for(int k=1;k<pts;k+=2){sum+=f(a+k*h);++evals;}
        R[n][0]=0.5*R[n-1][0]+(b-a)/pts*sum;
        for(int m=1;m<=n;++m){
            f64 k=std::pow(4.0,m);
            R[n][m]=(k*R[n][m-1]-R[n-1][m-1])/(k-1);
        }
        if(n>=2&&std::abs(R[n][n]-R[n-1][n-1])<tol)
            return {R[n][n], std::abs(R[n][n]-R[n-1][n-1]), evals, true};
    }
    return {R[max_order-1][max_order-1],1e-6,evals,false};
}

// ─── Adaptive quadrature (Gauss-Kronrod based) ────────────────────────────
[[nodiscard]] inline IntegralResult adaptive_gk(
    Fn1D f, f64 a, f64 b, f64 tol=1e-8, int max_depth=50) noexcept
{
    struct Interval { f64 a,b,val,err; };
    std::vector<Interval> stack;
    auto eval=[&](f64 lo, f64 hi)->Interval{
        auto r=gauss_kronrod15(f,lo,hi);
        return {lo,hi,r.value,r.error};
    };
    stack.push_back(eval(a,b));
    f64 total=stack[0].val, total_err=stack[0].err;
    int evals=15;
    for(int it=0;it<max_depth;++it){
        if(total_err<tol) return {total,total_err,evals,true};
        // Find worst interval
        int worst=0;
        for(int i=1;i<int(stack.size());++i)
            if(stack[i].err>stack[worst].err) worst=i;
        Interval iv=stack[worst];
        stack.erase(stack.begin()+worst);
        total-=iv.val; total_err-=iv.err;
        f64 mid=0.5*(iv.a+iv.b);
        auto L=eval(iv.a,mid), R=eval(mid,iv.b);
        total+=L.val+R.val; total_err+=L.err+R.err;
        evals+=30;
        stack.push_back(L); stack.push_back(R);
    }
    return {total,total_err,evals,false};
}

// ─── Monte Carlo integration ───────────────────────────────────────────────
[[nodiscard]] inline IntegralResult monte_carlo(
    FnNd f, std::vector<f64> lo, std::vector<f64> hi,
    int n_samples=100000, u32 seed=42) noexcept
{
    int d=int(lo.size());
    f64 volume=1; for(int i=0;i<d;++i) volume*=(hi[i]-lo[i]);
    PCG32 rng(seed);
    f64 sum=0, sum2=0;
    std::vector<f64> x(d);
    for(int s=0;s<n_samples;++s){
        for(int i=0;i<d;++i) x[i]=lo[i]+rng.next_f32()*(hi[i]-lo[i]);
        f64 fx=f(x); sum+=fx; sum2+=fx*fx;
    }
    f64 mean=sum/n_samples;
    f64 var=(sum2/n_samples-mean*mean)/n_samples;
    return {mean*volume, std::sqrt(var)*volume, n_samples, true};
}

// ─── Quasi-Monte Carlo (Halton) ───────────────────────────────────────────
[[nodiscard]] inline IntegralResult quasi_monte_carlo(
    FnNd f, std::vector<f64> lo, std::vector<f64> hi, int n=50000) noexcept
{
    int d=int(lo.size());
    static constexpr int primes[]={2,3,5,7,11,13,17,19,23,29,31,37};
    f64 volume=1; for(int i=0;i<d;++i) volume*=(hi[i]-lo[i]);
    f64 sum=0;
    std::vector<f64> x(d);
    for(int s=1;s<=n;++s){
        for(int i=0;i<d&&i<12;++i){
            // Radical inverse in base primes[i]
            f64 r=0,inv=1.0/primes[i]; u64 nn=s;
            while(nn){r+=f64(nn%primes[i])*inv;nn/=primes[i];inv/=primes[i];}
            x[i]=lo[i]+r*(hi[i]-lo[i]);
        }
        sum+=f(x);
    }
    return {sum/n*volume, 0, n, true};
}

// ─── Importance-sampled Monte Carlo ──────────────────────────────────────
// Estimate E[f(x)] where x is sampled from density q, weight = 1/q(x)
[[nodiscard]] inline f64 importance_mc(
    std::function<f64(f64)> f,
    std::function<f64(f64)> sample_q,
    std::function<f64(f64)> pdf_q,
    int n=10000, u32 seed=42) noexcept
{
    PCG32 rng(seed);
    f64 sum=0;
    for(int i=0;i<n;++i){
        f64 x=sample_q(rng.next_f64());
        f64 q=pdf_q(x);
        if(q>1e-300) sum+=f(x)/q;
    }
    return sum/n;
}

// ─── Trapezoidal rule ─────────────────────────────────────────────────────
[[nodiscard]] inline f64 trapezoidal(std::span<const f64> y, f64 dx) noexcept {
    int n=int(y.size()); if(n<2) return 0;
    f64 sum=0.5*(y.front()+y.back());
    for(int i=1;i<n-1;++i) sum+=y[i];
    return sum*dx;
}

[[nodiscard]] inline f64 trapezoidal(Fn1D f, f64 a, f64 b, int n) noexcept {
    f64 h=(b-a)/n, sum=0.5*(f(a)+f(b));
    for(int i=1;i<n;++i) sum+=f(a+i*h);
    return sum*h;
}

// ─── 2D Gaussian quadrature (tensor product) ──────────────────────────────
[[nodiscard]] inline f64 gauss2d(Fn2D f, f64 ax,f64 bx,f64 ay,f64 by,int n=5) noexcept {
    // 5-point GL nodes/weights on [-1,1]
    static constexpr f64 X[]={0.0,0.538469310105683,0.90617984593866};
    static constexpr f64 W[]={0.568888888888889,0.478628670499366,0.236926885056189};
    f64 mx=0.5*(ax+bx),hx=0.5*(bx-ax),my=0.5*(ay+by),hy=0.5*(by-ay),sum=0;
    auto eval=[&](f64 xi,f64 yi)->f64{return f(mx+hx*xi,my+hy*yi);};
    sum+=W[0]*W[0]*eval(0,0);
    for(int i=1;i<3;++i){
        sum+=W[i]*W[0]*(eval(X[i],0)+eval(-X[i],0));
        sum+=W[0]*W[i]*(eval(0,X[i])+eval(0,-X[i]));
        for(int j=1;j<3;++j)
            sum+=W[i]*W[j]*(eval(X[i],X[j])+eval(-X[i],X[j])+eval(X[i],-X[j])+eval(-X[i],-X[j]));
    }
    return sum*hx*hy;
}

// ─── Clenshaw-Curtis ──────────────────────────────────────────────────────
[[nodiscard]] inline f64 clenshaw_curtis(Fn1D f, f64 a, f64 b, int n=32) noexcept {
    // n must be even; uses Chebyshev nodes
    f64 mid=0.5*(a+b), half=0.5*(b-a);
    std::vector<f64> fvals(n+1);
    for(int k=0;k<=n;++k)
        fvals[k]=f(mid+half*std::cos(PI*k/n));
    // Compute Chebyshev moments via DCT
    std::vector<f64> c(n+1);
    for(int j=0;j<=n;++j){
        f64 s=0;
        for(int k=0;k<=n;++k){
            f64 w=(k==0||k==n)?0.5:1.0;
            s+=w*fvals[k]*std::cos(PI*j*k/n);
        }
        c[j]=2.0*s/n;
    }
    // Integrate: integral of cos(j*arccos(x)) from -1 to 1
    f64 sum=0;
    for(int j=0;j<=n;j+=2){
        f64 contrib=(j==0)?2.0:-2.0/(f64(j)*j-1.0);
        if(j==0) contrib=c[0]; else contrib*=c[j];
        sum+=(j==0)?c[0]*2.0:(-2.0/(f64(j*j)-1.0))*c[j];
    }
    return half*sum;
}

// ─── Line integral (path integral) ───────────────────────────────────────
// Integrate F·dr along parametric path r(t), t in [0,1]
[[nodiscard]] inline f64 line_integral(
    std::function<f64(f64,f64,f64)> F_dot_drdt,
    std::function<std::array<f64,3>(f64)> path,
    int n=100) noexcept
{
    f64 dt=1.0/n, sum=0;
    for(int i=0;i<n;++i){
        f64 t=(i+0.5)*dt;
        auto [x,y,z]=path(t);
        sum+=F_dot_drdt(x,y,z)*dt;
    }
    return sum;
}

// Surface integral over parametric surface r(u,v)
[[nodiscard]] inline f64 surface_integral(
    std::function<f64(f64,f64,f64)> f,
    std::function<std::array<f64,3>(f64,f64)> surface,
    f64 u0,f64 u1,f64 v0,f64 v1, int nu=20, int nv=20) noexcept
{
    f64 du=(u1-u0)/nu, dv=(v1-v0)/nv, sum=0, eps=1e-6;
    for(int i=0;i<nu;++i) for(int j=0;j<nv;++j){
        f64 u=u0+(i+0.5)*du, v=v0+(j+0.5)*dv;
        auto [x,y,z]=surface(u,v);
        // Approximate normal magnitude via cross product of partials
        auto [xpu,ypu,zpu]=surface(u+eps,v);
        auto [xpv,ypv,zpv]=surface(u,v+eps);
        f64 dxu=(xpu-x)/eps,dyu=(ypu-y)/eps,dzu=(zpu-z)/eps;
        f64 dxv=(xpv-x)/eps,dyv=(ypv-y)/eps,dzv=(zpv-z)/eps;
        f64 nx=dyu*dzv-dzu*dyv, ny=dzu*dxv-dxu*dzv, nz=dxu*dyv-dyu*dxv;
        f64 dA=std::sqrt(nx*nx+ny*ny+nz*nz);
        sum+=f(x,y,z)*dA*du*dv;
    }
    return sum;
}

// ─── Fourier integral ─────────────────────────────────────────────────────
// Compute integral of f(x)*exp(i*omega*x) from a to b
[[nodiscard]] inline std::pair<f64,f64> fourier_integral(
    Fn1D f, f64 a, f64 b, f64 omega, int n=200) noexcept
{
    f64 h=(b-a)/n, re=0, im=0;
    for(int i=0;i<n;++i){
        f64 x=a+(i+0.5)*h, fx=f(x);
        re+=fx*std::cos(omega*x);
        im+=fx*std::sin(omega*x);
    }
    return {re*h, im*h};
}

// ─── Integral transforms ──────────────────────────────────────────────────
// Laplace transform: L{f}(s) = integral_0^inf f(t) e^{-st} dt
// (Gaussian quadrature on [0,inf] via substitution t=u/(1-u))
[[nodiscard]] inline f64 laplace_transform(Fn1D f, f64 s, int n=100) noexcept {
    f64 sum=0, h=1.0/n;
    for(int i=1;i<n;++i){
        f64 u=i*h, t=u/(1-u), dt=1.0/((1-u)*(1-u));
        sum+=f(t)*std::exp(-s*t)*dt;
    }
    return sum*h;
}

// Mellin transform: M{f}(s) = integral_0^inf f(t) t^{s-1} dt
[[nodiscard]] inline f64 mellin_transform(Fn1D f, f64 s, int n=200) noexcept {
    f64 sum=0, h=1.0/n;
    for(int i=1;i<n;++i){
        f64 u=i*h, t=u/(1-u), dt=1.0/((1-u)*(1-u));
        sum+=f(t)*std::pow(t,s-1)*dt;
    }
    return sum*h;
}

} // namespace qs::math::numerical
