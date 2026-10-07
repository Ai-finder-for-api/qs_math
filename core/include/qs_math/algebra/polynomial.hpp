#pragma once
// qs_math/algebra/polynomial.hpp — Polynomial algebra: evaluation, arithmetic,
//   GCD, resultant, Sturm sequences, real root isolation, Chebyshev/Legendre/
//   Hermite/Laguerre polynomials, polynomial interpolation, power sums (C++20)
#include "../arithmetic/scalar_types.hpp"
#include "../arithmetic/scalar_math.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <span>

namespace qs::math::algebra {

using Poly64  = std::vector<f64>;    // poly[i] = coefficient of x^i
using Cx64    = std::complex<f64>;
using CxPoly  = std::vector<Cx64>;

// ─── Evaluation ──────────────────────────────────────────────────────────
[[nodiscard]] inline f64 poly_eval(const Poly64& p, f64 x) noexcept {
    f64 r = p.empty() ? 0.0 : p.back();
    for (int i = int(p.size())-2; i >= 0; --i) r = r*x + p[i];
    return r;
}
[[nodiscard]] inline Cx64 poly_eval_cx(const Poly64& p, Cx64 z) noexcept {
    Cx64 r = p.empty() ? Cx64(0) : Cx64(p.back());
    for (int i = int(p.size())-2; i >= 0; --i) r = r*z + Cx64(p[i]);
    return r;
}

// Evaluate at multiple points (Horner batched)
[[nodiscard]] inline std::vector<f64> poly_eval_batch(
    const Poly64& p, std::span<const f64> xs) noexcept
{
    std::vector<f64> out(xs.size());
    for (int i = 0; i < int(xs.size()); ++i) out[i] = poly_eval(p, xs[i]);
    return out;
}

// ─── Degree / trim ───────────────────────────────────────────────────────
[[nodiscard]] inline int poly_degree(const Poly64& p) noexcept {
    int d = int(p.size())-1;
    while (d > 0 && std::abs(p[d]) < 1e-300) --d;
    return d;
}
inline void poly_trim(Poly64& p) noexcept {
    while (p.size() > 1 && std::abs(p.back()) < 1e-300) p.pop_back();
}

// ─── Arithmetic ───────────────────────────────────────────────────────────
[[nodiscard]] inline Poly64 poly_add(Poly64 a, const Poly64& b) noexcept {
    if (b.size() > a.size()) a.resize(b.size(), 0.0);
    for (int i = 0; i < int(b.size()); ++i) a[i] += b[i];
    poly_trim(a); return a;
}
[[nodiscard]] inline Poly64 poly_sub(Poly64 a, const Poly64& b) noexcept {
    if (b.size() > a.size()) a.resize(b.size(), 0.0);
    for (int i = 0; i < int(b.size()); ++i) a[i] -= b[i];
    poly_trim(a); return a;
}
[[nodiscard]] inline Poly64 poly_mul(const Poly64& a, const Poly64& b) noexcept {
    if (a.empty() || b.empty()) return {0.0};
    Poly64 c(a.size()+b.size()-1, 0.0);
    for (int i = 0; i < int(a.size()); ++i)
        for (int j = 0; j < int(b.size()); ++j)
            c[i+j] += a[i]*b[j];
    poly_trim(c); return c;
}
[[nodiscard]] inline Poly64 poly_scale(Poly64 p, f64 s) noexcept {
    for (auto& c : p) c *= s; return p;
}
[[nodiscard]] inline Poly64 poly_shift(const Poly64& p, f64 a) noexcept {
    // p(x+a) via repeated convolution
    int n = int(p.size());
    Poly64 result(n, 0.0);
    // Powers of a: a^k
    std::vector<f64> apow(n, 1.0);
    for (int k = 1; k < n; ++k) apow[k] = apow[k-1]*a;
    // Binomial expansion: p(x+a) = sum_i p[i] * sum_k C(i,k) * a^(i-k) * x^k
    for (int i = 0; i < n; ++i) {
        f64 binom = 1.0;
        for (int k = 0; k <= i; ++k) {
            result[k] += p[i] * binom * apow[i-k];
            binom *= f64(i-k) / f64(k+1);
        }
    }
    poly_trim(result); return result;
}

// ─── Division (Euclidean) ─────────────────────────────────────────────────
struct PolyDivResult { Poly64 quot, rem; };

[[nodiscard]] inline PolyDivResult poly_div(Poly64 a, const Poly64& b) noexcept {
    poly_trim(const_cast<Poly64&>(a));
    Poly64 bb = b; poly_trim(bb);
    int da = poly_degree(a), db = poly_degree(bb);
    if (da < db) return {{0.0}, a};
    Poly64 q(da-db+1, 0.0), r = a;
    for (int i = da; i >= db; --i) {
        f64 coeff = r[i] / bb[db];
        q[i-db] = coeff;
        for (int j = 0; j <= db; ++j) r[i-db+j] -= coeff * bb[j];
    }
    poly_trim(q); poly_trim(r); return {q, r};
}

// ─── GCD (subresultant) ───────────────────────────────────────────────────
[[nodiscard]] inline Poly64 poly_gcd(Poly64 a, Poly64 b) noexcept {
    poly_trim(a); poly_trim(b);
    while (poly_degree(b) > 0) {
        auto [q,r] = poly_div(a, b);
        a = b; b = r; poly_trim(b);
    }
    if (poly_degree(b) == 0 && std::abs(b[0]) > 1e-300) return {1.0};
    // Normalize leading coeff
    if (!a.empty() && std::abs(a.back()) > 1e-300)
        a = poly_scale(a, 1.0/a.back());
    return a;
}

// ─── Derivative & integral ────────────────────────────────────────────────
[[nodiscard]] inline Poly64 poly_deriv(const Poly64& p) noexcept {
    int n = int(p.size());
    if (n <= 1) return {0.0};
    Poly64 d(n-1);
    for (int i = 1; i < n; ++i) d[i-1] = p[i]*f64(i);
    poly_trim(d); return d;
}
[[nodiscard]] inline Poly64 poly_integral(const Poly64& p, f64 C=0.0) noexcept {
    Poly64 I(p.size()+1);
    I[0] = C;
    for (int i = 0; i < int(p.size()); ++i) I[i+1] = p[i]/f64(i+1);
    return I;
}

// ─── Resultant (from Sylvester matrix determinant — Bezout) ───────────────
[[nodiscard]] inline f64 poly_resultant(const Poly64& p, const Poly64& q) noexcept {
    // Via subresultant PRS (simplified)
    int m = poly_degree(p), n = poly_degree(q);
    if (m < n) return (n%2==0?1:-1)*poly_resultant(q,p)*std::pow(q.back(),m-n);
    if (n == 0) return std::pow(q[0], m);
    auto [d, r] = poly_div(p, q);
    f64 sign = ((m-n)%2==0) ? 1.0 : -1.0;
    f64 factor = std::pow(q.back(), m - poly_degree(r));
    return sign * factor * poly_resultant(q, r);
}

// ─── Sturm sequence & real root counting ─────────────────────────────────
[[nodiscard]] inline std::vector<Poly64> sturm_sequence(Poly64 p) noexcept {
    std::vector<Poly64> chain;
    chain.push_back(p);
    chain.push_back(poly_deriv(p));
    while (poly_degree(chain.back()) > 0) {
        auto [q, r] = poly_div(chain[chain.size()-2], chain.back());
        chain.push_back(poly_scale(r, -1.0));
        poly_trim(chain.back());
        if (chain.back().empty()) break;
    }
    return chain;
}

[[nodiscard]] inline int sturm_sign_changes(
    const std::vector<Poly64>& chain, f64 x) noexcept
{
    int changes = 0; f64 prev = 0;
    for (auto& p : chain) {
        f64 val = poly_eval(p, x);
        if (std::abs(val) < 1e-300) continue;
        if (prev != 0 && (val > 0) != (prev > 0)) ++changes;
        prev = val;
    }
    return changes;
}

[[nodiscard]] inline int count_real_roots(
    const std::vector<Poly64>& chain, f64 a, f64 b) noexcept
{
    return sturm_sign_changes(chain, a) - sturm_sign_changes(chain, b);
}

// ─── Real root isolation (bisection on Sturm) ─────────────────────────────
[[nodiscard]] inline std::vector<std::pair<f64,f64>> isolate_real_roots(
    const Poly64& p, f64 lo=-1e6, f64 hi=1e6, f64 tol=1e-10) noexcept
{
    auto chain = sturm_sequence(p);
    std::vector<std::pair<f64,f64>> intervals;
    // Recursive bisection to isolate each root
    std::function<void(f64,f64,int)> isolate = [&](f64 a, f64 b, int roots){
        if (roots == 0) return;
        if (roots == 1) { intervals.push_back({a,b}); return; }
        if (b-a < tol) { intervals.push_back({a,b}); return; }
        f64 m = 0.5*(a+b);
        int lo_roots = count_real_roots(chain, a, m);
        int hi_roots = count_real_roots(chain, m, b);
        isolate(a, m, lo_roots);
        isolate(m, b, hi_roots);
    };
    int total = count_real_roots(chain, lo, hi);
    isolate(lo, hi, total);
    std::sort(intervals.begin(), intervals.end());
    return intervals;
}

// Refine root in interval [a,b] to tolerance
[[nodiscard]] inline f64 refine_root(const Poly64& p, f64 a, f64 b, f64 tol=1e-12) noexcept {
    f64 fa = poly_eval(p, a);
    for (int i = 0; i < 100; ++i) {
        f64 m = 0.5*(a+b), fm = poly_eval(p, m);
        if (std::abs(fm) < tol || b-a < tol) return m;
        (fa*fm < 0) ? (b=m) : (a=m, fa=fm);
    }
    return 0.5*(a+b);
}

// ─── Chebyshev polynomials (first kind T_n) ───────────────────────────────
[[nodiscard]] inline f64 chebyshev_T(int n, f64 x) noexcept {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    f64 t0=1, t1=x;
    for (int k = 2; k <= n; ++k) { f64 t2=2*x*t1-t0; t0=t1; t1=t2; }
    return t1;
}
// Second kind U_n
[[nodiscard]] inline f64 chebyshev_U(int n, f64 x) noexcept {
    if (n == 0) return 1.0;
    if (n == 1) return 2*x;
    f64 u0=1, u1=2*x;
    for (int k = 2; k <= n; ++k) { f64 u2=2*x*u1-u0; u0=u1; u1=u2; }
    return u1;
}

// Chebyshev nodes (roots of T_n)
[[nodiscard]] inline std::vector<f64> chebyshev_nodes(int n) noexcept {
    std::vector<f64> nodes(n);
    for (int k = 0; k < n; ++k)
        nodes[k] = std::cos(f64(2*k+1)*PI/f64(2*n));
    return nodes;
}

// ─── Legendre polynomials P_n ─────────────────────────────────────────────
[[nodiscard]] inline f64 legendre_P(int n, f64 x) noexcept {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    f64 p0=1, p1=x;
    for (int k = 2; k <= n; ++k) {
        f64 p2 = ((2*k-1)*x*p1-(k-1)*p0)/k;
        p0=p1; p1=p2;
    }
    return p1;
}
// Gauss-Legendre quadrature nodes and weights
[[nodiscard]] inline std::pair<std::vector<f64>,std::vector<f64>> gauss_legendre(int n) noexcept {
    std::vector<f64> x(n), w(n);
    for (int i = 0; i < (n+1)/2; ++i) {
        // Initial guess from cos
        f64 xi = std::cos(PI*(i+0.75)/(n+0.5));
        for (int iter = 0; iter < 100; ++iter) {
            f64 p0=1, p1=xi;
            for (int k=2;k<=n;++k){f64 p2=((2*k-1)*xi*p1-(k-1)*p0)/k;p0=p1;p1=p2;}
            f64 dp = n*(p0-xi*p1)/(1-xi*xi);
            f64 dx = p1/dp; xi -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        x[i] = -xi; x[n-1-i] = xi;
        f64 p0=1,p1=xi;
        for(int k=2;k<=n;++k){f64 p2=((2*k-1)*xi*p1-(k-1)*p0)/k;p0=p1;p1=p2;}
        f64 dp = n*(p0-xi*p1)/(1-xi*xi);
        w[i] = w[n-1-i] = 2.0/((1-xi*xi)*dp*dp);
    }
    return {x, w};
}

// ─── Hermite polynomials H_n (probabilist's) ──────────────────────────────
[[nodiscard]] inline f64 hermite_He(int n, f64 x) noexcept {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    f64 h0=1, h1=x;
    for (int k=2;k<=n;++k){f64 h2=x*h1-(k-1)*h0;h0=h1;h1=h2;}
    return h1;
}
// Physicist's H_n
[[nodiscard]] inline f64 hermite_H(int n, f64 x) noexcept {
    if (n==0) return 1;
    if (n==1) return 2*x;
    f64 h0=1,h1=2*x;
    for(int k=2;k<=n;++k){f64 h2=2*x*h1-2*(k-1)*h0;h0=h1;h1=h2;}
    return h1;
}

// ─── Laguerre polynomials L_n ─────────────────────────────────────────────
[[nodiscard]] inline f64 laguerre_L(int n, f64 x) noexcept {
    if (n==0) return 1;
    if (n==1) return 1-x;
    f64 l0=1, l1=1-x;
    for (int k=2;k<=n;++k) {
        f64 l2=((2*k-1-x)*l1-(k-1)*l0)/k;
        l0=l1; l1=l2;
    }
    return l1;
}
// Generalized Laguerre L_n^alpha
[[nodiscard]] inline f64 laguerre_La(int n, f64 alpha, f64 x) noexcept {
    if (n==0) return 1;
    if (n==1) return 1+alpha-x;
    f64 l0=1, l1=1+alpha-x;
    for (int k=2;k<=n;++k) {
        f64 l2=((2*k-1+alpha-x)*l1-(k-1+alpha)*l0)/k;
        l0=l1; l1=l2;
    }
    return l1;
}

// ─── Jacobi polynomials P_n^(a,b) ────────────────────────────────────────
[[nodiscard]] inline f64 jacobi_P(int n, f64 a, f64 b, f64 x) noexcept {
    if (n==0) return 1;
    f64 p0=1, p1=0.5*(a-b+(a+b+2)*x);
    for (int k=2;k<=n;++k) {
        f64 k2=2.0*k, ab=a+b;
        f64 A=k2*(k+ab)*(k2+ab-2);
        f64 B=(k2+ab-1)*(a*a-b*b);
        f64 C=(k2+ab-1)*(k2+ab)*(k2+ab-2);
        f64 D=2*(k+a-1)*(k+b-1)*(k2+ab);
        f64 p2=((B+C*x)*p1-D*p0)/A;
        p0=p1; p1=p2;
    }
    return p1;
}

// ─── Power sums (Newton's identities) ────────────────────────────────────
// p_k = sum of k-th powers of roots, from elementary symmetric polynomials e_k
[[nodiscard]] inline std::vector<f64> power_sums(const Poly64& p) noexcept {
    // p(x) = x^n - e1*x^{n-1} + e2*x^{n-2} - ...
    int n = poly_degree(p);
    std::vector<f64> e(n+1,0), pw(n+1,0);
    for (int i=0;i<=n;++i) e[i]=(n-i<int(p.size()))?p[n-i]/p[n]:0;
    // Newton's identity: p_k = sum_{i=1}^{k} (-1)^{i-1} e_i p_{k-i} + (-1)^{k-1} k e_k
    pw[0] = f64(n);
    for (int k=1;k<=n;++k) {
        f64 s=0;
        for (int i=1;i<k;++i) s+=(i%2==0?-1.0:1.0)*e[i]*pw[k-i];
        s += (k%2==0?-1.0:1.0)*k*e[k];
        pw[k] = s;
    }
    return pw;
}

// ─── Monic polynomial from roots ─────────────────────────────────────────
[[nodiscard]] inline Poly64 poly_from_roots(const std::vector<f64>& roots) noexcept {
    Poly64 p = {1.0};
    for (f64 r : roots) {
        Poly64 factor = {-r, 1.0};
        p = poly_mul(p, factor);
    }
    return p;
}

// ─── Polynomial composition p(q(x)) ──────────────────────────────────────
[[nodiscard]] inline Poly64 poly_compose(const Poly64& p, const Poly64& q) noexcept {
    Poly64 result = {0.0}, qpow = {1.0};
    for (int i = 0; i < int(p.size()); ++i) {
        result = poly_add(result, poly_scale(qpow, p[i]));
        if (i < int(p.size())-1) qpow = poly_mul(qpow, q);
    }
    poly_trim(result); return result;
}

// ─── Interpolating polynomial (Lagrange basis) ────────────────────────────
[[nodiscard]] inline Poly64 lagrange_interp(
    std::span<const f64> xs, std::span<const f64> ys) noexcept
{
    int n = int(xs.size());
    Poly64 result(n, 0.0);
    for (int i = 0; i < n; ++i) {
        Poly64 basis = {1.0};
        f64 denom = 1.0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            basis = poly_mul(basis, {-xs[j], 1.0});
            denom *= (xs[i]-xs[j]);
        }
        result = poly_add(result, poly_scale(basis, ys[i]/denom));
    }
    poly_trim(result); return result;
}

// ─── Chebyshev series approximation ──────────────────────────────────────
// Approximate f on [-1,1] by n-term Chebyshev series
[[nodiscard]] inline std::vector<f64> chebyshev_coeffs(
    std::function<f64(f64)> f, int n) noexcept
{
    auto nodes = chebyshev_nodes(n);
    std::vector<f64> fvals(n);
    for (int k = 0; k < n; ++k) fvals[k] = f(nodes[k]);
    std::vector<f64> c(n, 0.0);
    for (int j = 0; j < n; ++j) {
        f64 sum = 0;
        for (int k = 0; k < n; ++k)
            sum += fvals[k] * std::cos(f64(j*(2*k+1))*PI/f64(2*n));
        c[j] = sum * (j==0?1.0:2.0)/n;
    }
    return c;
}

[[nodiscard]] inline f64 chebyshev_series_eval(
    const std::vector<f64>& c, f64 x) noexcept
{
    f64 b0=0,b1=0,b2=0;
    for (int i = int(c.size())-1; i >= 1; --i) {
        b2=b1; b1=b0; b0=2*x*b1-b2+c[i];
    }
    return 0.5*(b0-b2)+c[0];
}

} // namespace qs::math::algebra
