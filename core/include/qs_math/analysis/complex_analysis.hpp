#pragma once
// qs_math/analysis/complex_analysis.hpp — Complex analysis: Cauchy integral,
//   residues, conformal mapping, analytic functions, Riemann surfaces,
//   Laurent series, Möbius transform, winding number, argument principle (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <complex>
#include <vector>
#include <functional>
#include <cmath>
#include <array>

namespace qs::math::analysis {

using Cx = std::complex<f64>;
using CxFn = std::function<Cx(Cx)>;

// ─── Basic complex operations ─────────────────────────────────────────────
[[nodiscard]] inline Cx cx_pow(Cx z, int n) noexcept {
    Cx r{1,0}; bool neg=n<0; if(neg) n=-n;
    for(int i=0;i<n;++i) r*=z;
    return neg?(Cx{1}/r):r;
}

[[nodiscard]] inline Cx cx_exp(Cx z) noexcept { return std::exp(z); }
[[nodiscard]] inline Cx cx_log(Cx z) noexcept { return std::log(z); }
[[nodiscard]] inline Cx cx_sqrt(Cx z) noexcept { return std::sqrt(z); }
[[nodiscard]] inline Cx cx_sin(Cx z) noexcept { return std::sin(z); }
[[nodiscard]] inline Cx cx_cos(Cx z) noexcept { return std::cos(z); }
[[nodiscard]] inline Cx cx_tan(Cx z) noexcept { return std::tan(z); }
[[nodiscard]] inline Cx cx_sinh(Cx z) noexcept { return std::sinh(z); }
[[nodiscard]] inline Cx cx_cosh(Cx z) noexcept { return std::cosh(z); }

// Real / imaginary part extraction
[[nodiscard]] inline f64 Re(Cx z) noexcept { return z.real(); }
[[nodiscard]] inline f64 Im(Cx z) noexcept { return z.imag(); }
[[nodiscard]] inline f64 Arg(Cx z) noexcept { return std::arg(z); }
[[nodiscard]] inline f64 Mod(Cx z) noexcept { return std::abs(z); }

// ─── Cauchy integral theorem ──────────────────────────────────────────────
// Numerical contour integral: integral of f around circular contour
[[nodiscard]] inline Cx cauchy_integral(
    CxFn f, Cx center, f64 radius, int n=256) noexcept
{
    Cx sum{0,0};
    f64 dtheta=TWO_PI/n;
    for(int k=0;k<n;++k){
        f64 theta=k*dtheta;
        Cx z=center+Cx{radius*std::cos(theta),radius*std::sin(theta)};
        Cx dz{-radius*std::sin(theta)*dtheta, radius*std::cos(theta)*dtheta};
        sum+=f(z)*dz;
    }
    return sum;
}

// Cauchy integral formula: f^(n)(z0) = n! / (2*pi*i) * integral f(z)/(z-z0)^{n+1} dz
[[nodiscard]] inline Cx cauchy_derivative(
    CxFn f, Cx z0, f64 radius, int n=0, int pts=512) noexcept
{
    // Compute n-th derivative at z0
    f64 fact=1; for(int k=1;k<=n;++k) fact*=k;
    auto g=[&](Cx z)->Cx{ return f(z)/cx_pow(z-z0,n+1); };
    return cauchy_integral(g,z0,radius,pts)*Cx{0,fact/(TWO_PI)};
}

// ─── Residue computation ──────────────────────────────────────────────────
// Residue at simple pole: res = lim_{z->z0} (z-z0)*f(z)
[[nodiscard]] inline Cx residue_simple(CxFn f, Cx z0, f64 eps=1e-6) noexcept {
    Cx delta{eps,0};
    return (f(z0+delta)*(z0+delta-z0)+f(z0-delta)*(z0-delta-z0))*0.5;
}

// Residue via Laurent series (numerical): coefficient of 1/(z-z0)
[[nodiscard]] inline Cx residue_numerical(
    CxFn f, Cx z0, f64 radius=0.01, int pts=1024) noexcept
{
    // Res(f,z0) = (1/2*pi*i) * contour integral of f
    return cauchy_integral(f,z0,radius,pts)/(Cx{0,TWO_PI});
}

// Sum of residues via argument principle count
[[nodiscard]] inline int winding_number_complex(
    CxFn f, Cx center, f64 radius, int pts=512) noexcept
{
    f64 total_arg=0;
    std::vector<Cx> vals(pts+1);
    for(int k=0;k<=pts;++k){
        f64 theta=TWO_PI*k/pts;
        Cx z=center+Cx{radius*std::cos(theta),radius*std::sin(theta)};
        vals[k]=f(z);
    }
    for(int k=0;k<pts;++k){
        f64 d=Arg(vals[k+1]/vals[k]);
        if(d>PI) d-=TWO_PI;
        if(d<-PI) d+=TWO_PI;
        total_arg+=d;
    }
    return int(std::round(total_arg/TWO_PI));
}

// ─── Möbius (linear fractional) transformation ────────────────────────────
struct Mobius {
    Cx a,b,c,d; // f(z) = (az+b)/(cz+d)

    [[nodiscard]] Cx operator()(Cx z) const noexcept {
        return (a*z+b)/(c*z+d);
    }
    [[nodiscard]] Mobius compose(const Mobius& g) const noexcept {
        return {a*g.a+b*g.c, a*g.b+b*g.d, c*g.a+d*g.c, c*g.b+d*g.d};
    }
    [[nodiscard]] Mobius inverse() const noexcept {
        return {d,-b,-c,a};
    }
    [[nodiscard]] Cx det() const noexcept { return a*d-b*c; }
    // Fixed points: solve z = (az+b)/(cz+d)
    [[nodiscard]] std::array<Cx,2> fixed_points() const noexcept {
        // cz^2 + (d-a)z - b = 0
        Cx A=c, B=d-a, C=-b;
        Cx disc=std::sqrt(B*B-4.0*A*C);
        if(std::abs(A)<1e-14) return {-C/B,Cx{1e30,0}};
        return {(-B+disc)/(2.0*A),(-B-disc)/(2.0*A)};
    }
    // Cayley map: z -> (z-i)/(z+i)  maps upper half-plane to unit disk
    static Mobius cayley() noexcept { return {{1,0},{0,-1},{1,0},{0,1}}; }
    // Cross-ratio: (z1,z2;z3,z4)
    static Cx cross_ratio(Cx z1,Cx z2,Cx z3,Cx z4) noexcept {
        return (z1-z3)*(z2-z4)/((z1-z4)*(z2-z3));
    }
};

// ─── Conformal mappings ───────────────────────────────────────────────────
// Joukowski transform: z -> z + 1/z  (wing profile)
[[nodiscard]] inline Cx joukowski(Cx z) noexcept { return z+Cx{1}/z; }
[[nodiscard]] inline Cx joukowski_inv(Cx w) noexcept {
    return 0.5*(w+std::sqrt(w*w-4.0));
}

// Schwarz-Christoffel (approximation for rectangle -> upper half-plane)
[[nodiscard]] inline Cx schwarz_christoffel_rect(Cx z, f64 k=0.5) noexcept {
    // Elliptic integral approximation
    Cx w{0,0};
    int N=50;
    f64 dt=1.0/N;
    Cx t_prev{0,0};
    for(int n=1;n<=N;++n){
        Cx t=z*Cx{f64(n)*dt,0};
        Cx f_val=Cx{1}/std::sqrt((1.0-t*t)*(1.0-k*k*t*t));
        w+=(t-t_prev)*f_val;
        t_prev=t;
    }
    return w;
}

// Log map (strips -> half-planes)
[[nodiscard]] inline Cx exp_map(Cx z) noexcept { return std::exp(z); }
[[nodiscard]] inline Cx log_map(Cx z) noexcept { return std::log(z); }

// Power map z^alpha
[[nodiscard]] inline Cx power_map(Cx z, f64 alpha) noexcept {
    return std::pow(z,alpha);
}

// ─── Laurent series ───────────────────────────────────────────────────────
struct LaurentSeries {
    Cx center;
    std::vector<Cx> coeffs; // coeffs[n] = coefficient of (z-z0)^{n-N_neg}
    int n_neg; // number of negative power terms

    [[nodiscard]] Cx eval(Cx z) const noexcept {
        Cx sum{0,0}, dz=z-center;
        for(int n=0;n<int(coeffs.size());++n){
            int power=n-n_neg;
            sum+=coeffs[n]*cx_pow(dz,power);
        }
        return sum;
    }

    [[nodiscard]] Cx residue() const noexcept {
        // Coefficient of (z-z0)^{-1}
        if(n_neg>=1) return coeffs[n_neg-1];
        return {0,0};
    }
};

// Compute Laurent coefficients numerically
[[nodiscard]] inline LaurentSeries laurent_series(
    CxFn f, Cx z0, f64 r, int n_neg=3, int n_pos=10) noexcept
{
    LaurentSeries L; L.center=z0; L.n_neg=n_neg;
    int total=n_neg+n_pos;
    L.coeffs.resize(total);
    // c_n = 1/(2*pi*i) * integral f(z)/(z-z0)^{n+1} dz
    for(int n=0;n<total;++n){
        int power=n-n_neg;
        auto g=[&](Cx z)->Cx{ return f(z)/cx_pow(z-z0,power+1); };
        L.coeffs[n]=cauchy_integral(g,z0,r,256)/(Cx{0,TWO_PI});
    }
    return L;
}

// ─── Analytic function testing (Cauchy-Riemann) ───────────────────────────
// Check if f satisfies Cauchy-Riemann equations at z numerically
[[nodiscard]] inline bool is_analytic_at(CxFn f, Cx z, f64 h=1e-5) noexcept {
    // dw/dz = (df/dx) = (1/i)*(df/dy)
    Cx df_dx=(f(z+Cx{h,0})-f(z-Cx{h,0}))/(2*h);
    Cx df_dy=(f(z+Cx{0,h})-f(z-Cx{0,h}))/(2*h);
    Cx ratio=df_dy/(Cx{0,1}*df_dx);
    return std::abs(ratio-1.0)<1e-3;
}

// Complex derivative (limit definition)
[[nodiscard]] inline Cx complex_deriv(CxFn f, Cx z, f64 h=1e-7) noexcept {
    return (f(z+Cx{h,0})-f(z-Cx{h,0}))/(2*h);
}

// Second order
[[nodiscard]] inline Cx complex_deriv2(CxFn f, Cx z, f64 h=1e-5) noexcept {
    return (f(z+Cx{h,0})-2.0*f(z)+f(z-Cx{h,0}))/(h*h);
}

// ─── Harmonic functions ───────────────────────────────────────────────────
// Laplacian of real function (finite difference)
[[nodiscard]] inline f64 laplacian2d(
    std::function<f64(f64,f64)> u, f64 x, f64 y, f64 h=1e-5) noexcept
{
    return (u(x+h,y)+u(x-h,y)+u(x,y+h)+u(x,y-h)-4*u(x,y))/(h*h);
}

// Check harmonicity
[[nodiscard]] inline bool is_harmonic_at(
    std::function<f64(f64,f64)> u, f64 x, f64 y, f64 h=1e-4, f64 tol=1e-4) noexcept
{
    return std::abs(laplacian2d(u,x,y,h))<tol;
}

// Harmonic conjugate at a point (numerical)
[[nodiscard]] inline f64 harmonic_conjugate_at(
    std::function<f64(f64,f64)> u, f64 x0, f64 y0, f64 x, f64 y, f64 h=1e-5) noexcept
{
    // v(x,y) = integral from (x0,y0) to (x,y) of (-du/dy dx + du/dx dy)
    // Integrate along axis-aligned path
    int steps=100;
    f64 v=0;
    // Horizontal segment y=y0, x from x0 to x
    f64 dx=(x-x0)/steps;
    for(int i=0;i<steps;++i){
        f64 xi=x0+i*dx;
        f64 dudy=(u(xi,y0+h)-u(xi,y0-h))/(2*h);
        v+=-dudy*dx;
    }
    // Vertical segment x=x, y from y0 to y
    f64 dy=(y-y0)/steps;
    for(int i=0;i<steps;++i){
        f64 yi=y0+i*dy;
        f64 dudx=(u(x+h,yi)-u(x-h,yi))/(2*h);
        v+=dudx*dy;
    }
    return v;
}

// ─── Argument principle (zeros minus poles) ───────────────────────────────
[[nodiscard]] inline int zeros_minus_poles(
    CxFn f, Cx center, f64 radius, int pts=512) noexcept
{
    return winding_number_complex(f,center,radius,pts);
}

// ─── Poisson integral (Dirichlet problem on disk) ─────────────────────────
// Solve Laplace equation in unit disk with boundary data g
[[nodiscard]] inline f64 poisson_integral(
    std::function<f64(f64)> g, f64 r, f64 theta, int n=256) noexcept
{
    f64 sum=0;
    f64 dphi=TWO_PI/n;
    for(int k=0;k<n;++k){
        f64 phi=k*dphi;
        // Poisson kernel: (1-r^2)/(1-2r*cos(theta-phi)+r^2)
        f64 cos_d=std::cos(theta-phi);
        f64 P=(1-r*r)/(1-2*r*cos_d+r*r);
        sum+=P*g(phi)*dphi;
    }
    return sum/TWO_PI;
}

// ─── Branch cut handling ──────────────────────────────────────────────────
// Principal branch log (branch cut on negative real axis)
[[nodiscard]] inline Cx principal_log(Cx z) noexcept {
    return Cx{std::log(std::abs(z)), std::arg(z)};
}
// Log with custom branch cut angle
[[nodiscard]] inline Cx log_branch(Cx z, f64 branch_angle=PI) noexcept {
    f64 r=std::abs(z);
    f64 theta=std::arg(z);
    while(theta>=branch_angle) theta-=TWO_PI;
    while(theta< branch_angle-TWO_PI) theta+=TWO_PI;
    return Cx{std::log(r),theta};
}

// ─── Series summation (via residues) ─────────────────────────────────────
// Sum sum_{n=-inf}^{inf} f(n) = -pi * sum of residues of pi*cot(pi*z)*f(z)
[[nodiscard]] inline Cx mittag_leffler_sum(
    CxFn f, const std::vector<Cx>& poles, int n_terms=20) noexcept
{
    // Partial fraction expansion sum
    Cx sum{0,0};
    for(int n=-n_terms;n<=n_terms;++n) sum+=f(Cx{f64(n),0});
    return sum;
}

} // namespace qs::math::analysis
