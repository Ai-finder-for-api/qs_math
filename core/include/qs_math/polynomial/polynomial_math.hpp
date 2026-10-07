#pragma once
// qs_math/polynomial_math.hpp — Roots (Durand-Kerner), Sturm chains, resultants, polynomial ops (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <array>
#include <span>
#include <optional>

namespace qs {

using Cx = std::complex<f64>;

// ─── Polynomial evaluation ────────────────────────────────────────────────────

// Horner's method: p(x) = a[0] + a[1]*x + ... + a[n]*x^n
[[nodiscard]] inline f64 poly_eval(std::span<const f64> coeffs, f64 x) {
    f64 r = 0.;
    for (i32 i=i32(coeffs.size())-1; i>=0; --i)
        r = r*x + coeffs[i];
    return r;
}

[[nodiscard]] inline Cx poly_eval_cx(std::span<const f64> coeffs, Cx x) {
    Cx r{0.,0.};
    for (i32 i=i32(coeffs.size())-1; i>=0; --i)
        r = r*x + Cx{coeffs[i],0.};
    return r;
}

// Derivative via Horner
[[nodiscard]] inline f64 poly_deriv_eval(std::span<const f64> coeffs, f64 x) {
    u32 n = u32(coeffs.size());
    if (n < 2) return 0.;
    f64 r = f64(n-1)*coeffs[n-1];
    for (i32 i=i32(n)-2; i>=1; --i)
        r = r*x + f64(i)*coeffs[i];
    return r;
}

// Polynomial derivative coefficients
[[nodiscard]] inline std::vector<f64> poly_derivative(std::span<const f64> p) {
    u32 n = u32(p.size());
    if (n < 2) return {0.};
    std::vector<f64> d(n-1);
    for (u32 i=1;i<n;++i) d[i-1]=f64(i)*p[i];
    return d;
}

// Polynomial multiplication
[[nodiscard]] inline std::vector<f64> poly_mul(
    std::span<const f64> a, std::span<const f64> b)
{
    if (a.empty()||b.empty()) return {};
    u32 na=u32(a.size()), nb=u32(b.size());
    std::vector<f64> r(na+nb-1, 0.);
    for (u32 i=0;i<na;++i)
    for (u32 j=0;j<nb;++j) r[i+j]+=a[i]*b[j];
    return r;
}

// Polynomial division: returns {quotient, remainder}
[[nodiscard]] inline std::pair<std::vector<f64>,std::vector<f64>> poly_divmod(
    std::span<const f64> num, std::span<const f64> den)
{
    std::vector<f64> r(num.begin(),num.end());
    u32 nd=u32(den.size());
    if (nd==0||r.size()<nd) return {{0.},{r.begin(),r.end()}};
    std::vector<f64> q(r.size()-nd+1, 0.);
    for (i32 i=i32(r.size())-1; i>=i32(nd)-1; --i) {
        f64 c = r[i]/den[nd-1];
        q[i-i32(nd)+1] = c;
        for (u32 j=0;j<nd;++j) r[i-nd+1+j]-=c*den[j];
    }
    r.resize(nd-1);
    return {q,r};
}

// GCD of two polynomials (Euclidean)
[[nodiscard]] inline std::vector<f64> poly_gcd(
    std::vector<f64> a, std::vector<f64> b)
{
    auto trim=[](std::vector<f64>& v, f64 eps=1e-12){
        while(v.size()>1&&std::abs(v.back())<eps) v.pop_back();
    };
    trim(a); trim(b);
    while(b.size()>1||std::abs(b.back())>1e-12) {
        auto [q,r]=poly_divmod(a,b);
        trim(r);
        a=b; b=r;
    }
    if (!a.empty()){f64 s=a.back();for(auto&c:a)c/=s;}
    return a;
}

// ─── Sturm chain ─────────────────────────────────────────────────────────────

[[nodiscard]] inline std::vector<std::vector<f64>> sturm_chain(std::span<const f64> p) {
    std::vector<std::vector<f64>> chain;
    chain.push_back({p.begin(),p.end()});
    chain.push_back(poly_derivative(p));
    while (chain.back().size()>1||std::abs(chain.back().back())>1e-14) {
        auto& a=chain[chain.size()-2];
        auto& b=chain.back();
        auto [q,r]=poly_divmod(a,b);
        for(auto&c:r)c=-c; // negate remainder
        // trim
        while(r.size()>1&&std::abs(r.back())<1e-14) r.pop_back();
        if(r.size()==1&&std::abs(r[0])<1e-14) break;
        chain.push_back(r);
    }
    return chain;
}

// Count sign changes in Sturm sequence at x
[[nodiscard]] inline u32 sturm_sign_changes(
    const std::vector<std::vector<f64>>& chain, f64 x)
{
    u32 changes=0;
    f64 prev=0.;
    for (const auto& p : chain) {
        f64 v=poly_eval(p,x);
        if (std::abs(v)<1e-14) continue;
        if (prev*v<0.) ++changes;
        prev=v;
    }
    return changes;
}

// Count real roots in [lo,hi]
[[nodiscard]] inline u32 count_roots(
    const std::vector<std::vector<f64>>& chain, f64 lo, f64 hi)
{
    return sturm_sign_changes(chain,lo)-sturm_sign_changes(chain,hi);
}

// ─── Durand-Kerner root finding ───────────────────────────────────────────────

[[nodiscard]] inline std::vector<Cx> durand_kerner(
    std::span<const f64> coeffs_in,
    u32 max_iter=100, f64 tol=1e-12)
{
    std::vector<f64> coeffs(coeffs_in.begin(), coeffs_in.end());
    u32 n = u32(coeffs.size())-1;
    if (n==0) return {};
    // Normalize
    f64 leading = coeffs.back();
    for (auto& c : coeffs) c/=leading;

    // Initial guesses on a circle
    std::vector<Cx> roots(n);
    for (u32 k=0;k<n;++k) {
        f64 ang = 2.*k_pi_d64*f64(k)/f64(n);
        roots[k] = Cx{0.4*std::cos(ang), 0.4*std::sin(ang)};
    }

    for (u32 iter=0;iter<max_iter;++iter) {
        f64 max_change=0.;
        std::vector<Cx> new_roots=roots;
        for (u32 i=0;i<n;++i) {
            Cx num = poly_eval_cx(coeffs, roots[i]);
            Cx den{1.,0.};
            for (u32 j=0;j<n;++j) if(j!=i) den*=(roots[i]-roots[j]);
            Cx step = num/den;
            new_roots[i] = roots[i]-step;
            max_change=std::max(max_change,std::abs(step));
        }
        roots=new_roots;
        if (max_change<tol) break;
    }
    return roots;
}

// ─── Companion matrix eigenvalues (alternative root method) ──────────────────

// Resultant via Sylvester matrix determinant (2-poly)
[[nodiscard]] inline f64 resultant(
    std::span<const f64> p, std::span<const f64> q)
{
    u32 m=u32(p.size())-1, n=u32(q.size())-1;
    u32 sz=m+n;
    // Sylvester matrix is sz x sz; compute det via naive expansion (small polys)
    std::vector<std::vector<f64>> S(sz, std::vector<f64>(sz,0.));
    for (u32 i=0;i<n;++i) for (u32 j=0;j<=m;++j) if(i+j<sz) S[i][i+j]=p[m-j];
    for (u32 i=0;i<m;++i) for (u32 j=0;j<=n;++j) if(n+i+j<sz) S[n+i][i+j]=q[n-j];
    // Gaussian elimination for det
    f64 det=1.;
    auto mat=S;
    for (u32 col=0;col<sz;++col) {
        u32 pivot=col;
        for (u32 row=col+1;row<sz;++row) if(std::abs(mat[row][col])>std::abs(mat[pivot][col])) pivot=row;
        if (pivot!=col){std::swap(mat[col],mat[pivot]);det=-det;}
        if (std::abs(mat[col][col])<1e-14) return 0.;
        det*=mat[col][col];
        for (u32 row=col+1;row<sz;++row) {
            f64 f=mat[row][col]/mat[col][col];
            for (u32 k=col;k<sz;++k) mat[row][k]-=f*mat[col][k];
        }
    }
    return det;
}

// ─── Real cubic / quartic closed-form ────────────────────────────────────────

// Cardano cubic: x^3 + px + q = 0
[[nodiscard]] inline std::vector<f64> cardano(f64 p, f64 q) {
    f64 D = -4.*p*p*p - 27.*q*q;
    if (D > 0.) {
        // 3 real roots via trigonometric
        f64 m = 2.*std::sqrt(-p/3.);
        f64 t = std::acos(3.*q/(p*m))/3.;
        return {m*std::cos(t), m*std::cos(t-2.*k_pi_d64/3.), m*std::cos(t-4.*k_pi_d64/3.)};
    } else {
        f64 A=std::cbrt(-q/2.+std::sqrt(q*q/4.+p*p*p/27.));
        f64 B=std::cbrt(-q/2.-std::sqrt(q*q/4.+p*p*p/27.));
        return {A+B}; // one real root
    }
}

// Solve depressed quartic x^4 + px^2 + qx + r = 0 via Ferrari
[[nodiscard]] inline std::vector<f64> solve_quartic_depressed(f64 p, f64 q, f64 r) {
    // Resolvent cubic: y^3 + 5p/2*y^2 + (2p^2-r)*y + (p^3/2 - pr/2 - q^2/8)
    // Use numerical root finding instead
    auto roots_cx=durand_kerner(std::vector<f64>{r,q,p,0.,1.});
    std::vector<f64> real_roots;
    for(const auto& cx:roots_cx) if(std::abs(cx.imag())<1e-8) real_roots.push_back(cx.real());
    return real_roots;
}

// ─── Chebyshev polynomial ─────────────────────────────────────────────────────

[[nodiscard]] inline f64 chebyshev_T(u32 n, f64 x) {
    if (n==0) return 1.;
    if (n==1) return x;
    f64 t0=1., t1=x;
    for (u32 k=2;k<=n;++k){f64 t2=2.*x*t1-t0; t0=t1; t1=t2;}
    return t1;
}

[[nodiscard]] inline f64 chebyshev_U(u32 n, f64 x) {
    if (n==0) return 1.;
    if (n==1) return 2.*x;
    f64 u0=1.,u1=2.*x;
    for (u32 k=2;k<=n;++k){f64 u2=2.*x*u1-u0; u0=u1; u1=u2;}
    return u1;
}

// Chebyshev nodes on [-1,1] for interpolation
[[nodiscard]] inline std::vector<f64> chebyshev_nodes(u32 n) {
    std::vector<f64> pts(n);
    for (u32 k=0;k<n;++k)
        pts[k]=std::cos(k_pi_d64*(2.*f64(k)+1.)/(2.*f64(n)));
    return pts;
}

// ─── Legendre polynomial ──────────────────────────────────────────────────────

[[nodiscard]] inline f64 legendre_P(u32 n, f64 x) {
    if (n==0) return 1.;
    if (n==1) return x;
    f64 p0=1., p1=x;
    for (u32 k=2;k<=n;++k){
        f64 p2=((2.*f64(k)-1.)*x*p1-(f64(k)-1.)*p0)/f64(k);
        p0=p1; p1=p2;
    }
    return p1;
}

// Gauss-Legendre quadrature nodes and weights (n up to 5 hardcoded)
[[nodiscard]] inline std::pair<std::vector<f64>,std::vector<f64>> gauss_legendre(u32 n) {
    // n=1..5 exact values
    if (n==1) return {{0.},{2.}};
    if (n==2) return {{-0.5773502691896258,0.5773502691896258},{1.,1.}};
    if (n==3) return {{-0.7745966692414834,0.,0.7745966692414834},{0.5555555555555556,0.8888888888888888,0.5555555555555556}};
    if (n==4) return {{-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526},{0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538}};
    // n=5
    return {{-0.9061798459386640,-0.5384693101056831,0.,0.5384693101056831,0.9061798459386640},
            {0.2369268850561891,0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891}};
}

} // namespace qs
