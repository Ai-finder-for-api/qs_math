// qs_math/calculus/calculus_impl.cpp (C++20)
#include "include/qs_math_calculus.hpp"
#include <cmath>
namespace qs::math::calculus {
static f64 simp_adp(std::function<f64(f64)> f,f64 a,f64 b,f64 fa,f64 fb,f64 fc,f64 whole,f64 tol,i32 depth){
    const f64 mid=(a+b)*0.5,h=(b-a);
    const f64 fm1=f((a+mid)*0.5),fm2=f((mid+b)*0.5);
    const f64 left=(h/12)*(fa+4*fm1+fc),right=(h/12)*(fc+4*fm2+fb);
    const f64 delta=left+right-whole;
    if(depth<=0||std::abs(delta)<15*tol) return left+right+delta/15;
    return simp_adp(f,a,mid,fa,fc,fm1,left,tol*0.5,depth-1)
          +simp_adp(f,mid,b,fc,fb,fm2,right,tol*0.5,depth-1);
}
f64 integrate(std::function<f64(f64)> f,f64 a,f64 b,f64 tol) noexcept {
    const f64 fa=f(a),fb=f(b),fc=f((a+b)*0.5),whole=(b-a)/6*(fa+4*fc+fb);
    return simp_adp(f,a,b,fa,fb,fc,whole,tol,50);
}
void fourier_coeffs(std::function<f64(f64)> f,i32 N,f64* an,f64* bn) noexcept {
    constexpr f64 PI=3.14159265358979323846;
    const i32 M=4096; const f64 h=2*PI/M;
    for(i32 k=0;k<=N;++k){
        f64 sa=0,sb=0;
        for(i32 j=0;j<M;++j){const f64 x=-PI+j*h,fx=f(x);sa+=fx*std::cos(k*x);sb+=fx*std::sin(k*x);}
        an[k]=sa*h/PI; bn[k]=sb*h/PI;
    }
    an[0]*=0.5;
}
// Euler method ODE integrator
void euler_ode(std::function<f64(f64,f64)> f,f64 t0,f64 y0,f64 h,i32 n,f64* ts,f64* ys) noexcept {
    ts[0]=t0; ys[0]=y0;
    for(i32 i=1;i<n;++i){ts[i]=ts[i-1]+h;ys[i]=ys[i-1]+h*f(ts[i-1],ys[i-1]);}
}
// Midpoint method (2nd order Runge-Kutta)
void midpoint_ode(std::function<f64(f64,f64)> f,f64 t0,f64 y0,f64 h,i32 n,f64* ts,f64* ys) noexcept {
    ts[0]=t0; ys[0]=y0;
    for(i32 i=1;i<n;++i){
        const f64 k1=f(ts[i-1],ys[i-1]);
        const f64 k2=f(ts[i-1]+h*0.5,ys[i-1]+h*0.5*k1);
        ts[i]=ts[i-1]+h; ys[i]=ys[i-1]+h*k2;
    }
}
} // namespace qs::math::calculus
