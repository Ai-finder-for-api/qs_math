#pragma once
// qs_math/polynomial/poly_arithmetic.hpp — Poly division, GCD, resultant, roots, multivariate
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>
#include <optional>

namespace qs::math {

// ─── Dense polynomial over field (coefficient array, degree N) ────────────────
template<int N>
struct Poly {
    std::array<f64,N> c{}; // c[0]=constant term
    [[nodiscard]] constexpr int degree() const noexcept {
        for(int i=N-1;i>=0;--i) if(std::abs(c[i])>1e-15) return i; return 0;
    }
    [[nodiscard]] constexpr f64 eval(f64 x) const noexcept {
        f64 r=c[N-1]; for(int i=N-2;i>=0;--i) r=r*x+c[i]; return r;
    }
    [[nodiscard]] constexpr f64 eval_deriv(f64 x) const noexcept {
        f64 r=c[N-1]*(N-1); for(int i=N-2;i>=1;--i) r=r*x+c[i]*i; return r;
    }
    [[nodiscard]] constexpr Poly operator+(const Poly& o) const noexcept {
        Poly r; for(int i=0;i<N;++i) r.c[i]=c[i]+o.c[i]; return r;
    }
    [[nodiscard]] constexpr Poly operator-(const Poly& o) const noexcept {
        Poly r; for(int i=0;i<N;++i) r.c[i]=c[i]-o.c[i]; return r;
    }
    [[nodiscard]] constexpr Poly operator*(f64 s) const noexcept {
        Poly r; for(int i=0;i<N;++i) r.c[i]=c[i]*s; return r;
    }
    [[nodiscard]] constexpr bool is_zero(f64 tol=1e-12) const noexcept {
        for(auto v:c) if(std::abs(v)>tol) return false; return true;
    }
    // Leading coefficient
    [[nodiscard]] constexpr f64 lc() const noexcept { return c[degree()]; }
};

// Polynomial multiplication (result degree up to N-1+M-1)
template<int N, int M>
[[nodiscard]] constexpr Poly<N+M-1> poly_mul(const Poly<N>& a, const Poly<M>& b) noexcept {
    Poly<N+M-1> r{};
    for(int i=0;i<N;++i) for(int j=0;j<M&&i+j<N+M-1;++j) r.c[i+j]+=a.c[i]*b.c[j];
    return r;
}

// Polynomial long division: a = q*b + r, returns (q,r)
template<int N>
[[nodiscard]] constexpr std::pair<Poly<N>,Poly<N>> poly_divmod(
    Poly<N> a, const Poly<N>& b) noexcept
{
    Poly<N> q{};
    int da=a.degree(),db=b.degree();
    if(da<db) return{q,a};
    for(int i=da;i>=db;--i) {
        if(std::abs(b.c[db])<1e-15) break;
        f64 coeff=a.c[i]/b.c[db];
        q.c[i-db]+=coeff;
        for(int j=0;j<=db;++j) a.c[i-db+j]-=coeff*b.c[j];
    }
    return{q,a};
}

// GCD via Euclidean algorithm
template<int N>
[[nodiscard]] constexpr Poly<N> poly_gcd(Poly<N> a, Poly<N> b) noexcept {
    while(!b.is_zero()) {
        auto [q,r]=poly_divmod(a,b);
        a=b; b=r;
    }
    // Normalize
    f64 lc=a.lc();
    if(std::abs(lc)>1e-15) a=a*(1/lc);
    return a;
}

// Resultant (via Sylvester matrix determinant)
template<int N>
[[nodiscard]] constexpr f64 resultant(const Poly<N>& a, const Poly<N>& b) noexcept {
    int da=a.degree(),db=b.degree();
    if(da==0) return std::pow(a.c[0],db);
    if(db==0) return std::pow(b.c[0],da);
    // Subresultant PRS (simplified: Euclidean + sign tracking)
    f64 res=1,s=1;
    Poly<N> f=a,g=b;
    while(g.degree()>0) {
        int df=f.degree(),dg=g.degree();
        res*=std::pow(g.lc(),df-dg+1)*s;
        s=(df%2==1&&dg%2==1)?-1:1;
        auto [q,r]=poly_divmod(f,g);
        f=g;
        // Pseudo-division correction
        f64 factor=std::pow(g.lc(),df-dg+1);
        for(auto& v:r.c) v/=(std::abs(factor)>1e-15?factor:1);
        g=r;
    }
    res*=std::pow(g.c[0],f.degree());
    return res;
}

// Discriminant: Res(f,f')/lc(f)^{...}
template<int N>
[[nodiscard]] constexpr f64 discriminant(const Poly<N>& f) noexcept {
    Poly<N> df{};
    for(int i=1;i<N;++i) df.c[i-1]=f.c[i]*i;
    return resultant(f,df);
}

// ─── Root finding ─────────────────────────────────────────────────────────────
// Aberth-Ehrlich iteration (all roots simultaneously)
template<int N>
[[nodiscard]] constexpr std::array<Cx,N-1> poly_roots_aberth(
    const Poly<N>& p, int max_iter=100) noexcept
{
    int deg=p.degree();
    std::array<Cx,N-1> z{};
    // Initial guess: roots of unity scaled by |c0/cN|^{1/n}
    f64 r=std::pow(std::abs(p.c[0]/p.c[deg]),1.0/deg);
    for(int k=0;k<deg;++k) {
        f64 theta=2*3.14159265358979323846*k/deg+0.5;
        z[k]={r*std::cos(theta),r*std::sin(theta)};
    }
    for(int iter=0;iter<max_iter;++iter) {
        f64 max_step=0;
        for(int k=0;k<deg;++k) {
            // Evaluate p(z_k) and p'(z_k)
            Cx pz{p.c[0],0},dpz{};
            for(int i=1;i<=deg;++i){
                dpz=dpz*z[k]+pz;
                pz=pz*z[k]+Cx{p.c[i],0};
            }
            // Weierstrass correction: sum 1/(z_k-z_j)
            Cx w{};
            for(int j=0;j<deg;++j) if(j!=k) {
                Cx dz=z[k]-z[j];
                f64 d2=dz.mag2();
                if(d2>1e-30) w=w+Cx{dz.re/d2,-dz.im/d2};
            }
            f64 pz2=pz.mag2();
            Cx ratio=pz2>1e-30?Cx{(pz*dpz.conj()).re/pz2,(pz*dpz.conj()).im/pz2}:Cx{};
            Cx step=cx_div(pz,dpz-pz*w);
            z[k]=z[k]-step;
            max_step=std::max(max_step,step.mag());
        }
        if(max_step<1e-12) break;
    }
    return z;
}

// Sturm sequence: count real roots in (a,b)
template<int N>
[[nodiscard]] constexpr int sturm_count(const Poly<N>& f, f64 a, f64 b) noexcept {
    // Build Sturm sequence: f0=f, f1=f', f2=-rem(f0,f1), ...
    std::array<Poly<N>,N> seq{};
    seq[0]=f;
    for(int i=1;i<N;++i) seq[i].c[i-1]=f.c[i]*i; // f'
    int n=2;
    while(n<N&&!seq[n-1].is_zero()) {
        auto [q,r]=poly_divmod(seq[n-2],seq[n-1]);
        for(int i=0;i<N;++i) seq[n].c[i]=-r.c[i];
        if(seq[n].is_zero()) break;
        ++n;
    }
    // Count sign changes at a and b
    auto sign_changes=[&](f64 x)->int{
        int changes=0;
        f64 prev=0;
        for(int i=0;i<n;++i){
            f64 v=seq[i].eval(x);
            if(std::abs(v)<1e-15) continue;
            if(prev!=0&&((v>0)!=(prev>0))) ++changes;
            prev=v;
        }
        return changes;
    };
    return sign_changes(a)-sign_changes(b);
}

// ─── Chebyshev polynomial Tn(x) ───────────────────────────────────────────────
[[nodiscard]] constexpr f64 chebyshev_T(int n, f64 x) noexcept {
    if(n==0) return 1; if(n==1) return x;
    f64 t0=1,t1=x;
    for(int k=2;k<=n;++k){f64 t2=2*x*t1-t0;t0=t1;t1=t2;}
    return t1;
}
[[nodiscard]] constexpr f64 chebyshev_U(int n, f64 x) noexcept {
    if(n==0) return 1; if(n==1) return 2*x;
    f64 u0=1,u1=2*x;
    for(int k=2;k<=n;++k){f64 u2=2*x*u1-u0;u0=u1;u1=u2;}
    return u1;
}
// Legendre polynomial P_n(x)
[[nodiscard]] constexpr f64 legendre_P(int n, f64 x) noexcept {
    if(n==0)return 1;if(n==1)return x;
    f64 p0=1,p1=x;
    for(int k=2;k<=n;++k){f64 p2=((2*k-1)*x*p1-(k-1)*p0)/k;p0=p1;p1=p2;}
    return p1;
}
// Hermite polynomial H_n(x) (physicist's)
[[nodiscard]] constexpr f64 hermite_H(int n, f64 x) noexcept {
    if(n==0)return 1;if(n==1)return 2*x;
    f64 h0=1,h1=2*x;
    for(int k=2;k<=n;++k){f64 h2=2*x*h1-2*(k-1)*h0;h0=h1;h1=h2;}
    return h1;
}
// Laguerre polynomial L_n(x)
[[nodiscard]] constexpr f64 laguerre_L(int n, f64 x) noexcept {
    if(n==0)return 1;if(n==1)return 1-x;
    f64 l0=1,l1=1-x;
    for(int k=2;k<=n;++k){f64 l2=((2*k-1-x)*l1-(k-1)*l0)/k;l0=l1;l1=l2;}
    return l1;
}
} // namespace qs::math
