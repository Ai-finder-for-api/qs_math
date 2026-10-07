#pragma once
// qs_math/series_math.hpp — Taylor, Laurent, power series truncation and evaluation (C++20)
#include "scalar_types.hpp"
#include "dual_number.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <functional>
#include <numbers>
#include <cassert>

namespace qs {

using Cx = std::complex<f64>;

// ─── Power series: f(x) = Σ c_n x^n ─────────────────────────────────────────
struct PowerSeries {
    std::vector<f64> coeffs; // coeffs[n] = coefficient of x^n

    PowerSeries() = default;
    explicit PowerSeries(std::vector<f64> c) : coeffs(std::move(c)) {}

    // Evaluate at x using Horner's method
    [[nodiscard]] f64 eval(f64 x) const {
        if(coeffs.empty()) return 0;
        f64 r=coeffs.back();
        for(i32 i=i32(coeffs.size())-2;i>=0;--i) r=r*x+coeffs[u32(i)];
        return r;
    }
    [[nodiscard]] Cx eval(Cx x) const {
        if(coeffs.empty()) return {};
        Cx r=coeffs.back();
        for(i32 i=i32(coeffs.size())-2;i>=0;--i) r=r*x+coeffs[u32(i)];
        return r;
    }

    // Differentiate: d/dx Σ c_n x^n = Σ n c_n x^{n-1}
    [[nodiscard]] PowerSeries differentiate() const {
        if(coeffs.size()<=1) return PowerSeries({0});
        std::vector<f64> dc(coeffs.size()-1);
        for(u32 n=1;n<u32(coeffs.size());++n) dc[n-1]=n*coeffs[n];
        return PowerSeries(dc);
    }
    // Integrate: ∫ Σ c_n x^n dx = C + Σ c_n/(n+1) x^{n+1}
    [[nodiscard]] PowerSeries integrate(f64 C=0) const {
        std::vector<f64> ic(coeffs.size()+1);
        ic[0]=C;
        for(u32 n=0;n<u32(coeffs.size());++n) ic[n+1]=coeffs[n]/(n+1);
        return PowerSeries(ic);
    }
    // Multiply two series (truncate to min degree)
    [[nodiscard]] PowerSeries operator*(const PowerSeries& b) const {
        u32 n=u32(coeffs.size()),m=u32(b.coeffs.size());
        std::vector<f64> r(n+m-1,0);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<m;++j) r[i+j]+=coeffs[i]*b.coeffs[j];
        return PowerSeries(r);
    }
    [[nodiscard]] PowerSeries operator+(const PowerSeries& b) const {
        u32 n=std::max(u32(coeffs.size()),u32(b.coeffs.size()));
        std::vector<f64> r(n,0);
        for(u32 i=0;i<u32(coeffs.size());++i) r[i]+=coeffs[i];
        for(u32 i=0;i<u32(b.coeffs.size());++i) r[i]+=b.coeffs[i];
        return PowerSeries(r);
    }
    // Truncate to n terms
    [[nodiscard]] PowerSeries truncate(u32 n) const {
        std::vector<f64> c(coeffs.begin(),coeffs.begin()+std::min(n,u32(coeffs.size())));
        c.resize(n,0); return PowerSeries(c);
    }
    // Compose: (this ∘ g)(x) = this(g(x)) — truncated
    [[nodiscard]] PowerSeries compose(const PowerSeries& g, u32 terms=10) const {
        PowerSeries r({0}); PowerSeries gn({1}); // g^0 = 1
        u32 N=std::min(terms,u32(coeffs.size()));
        for(u32 n=0;n<N;++n) {
            PowerSeries term({coeffs[n]});
            r=r+(term*gn).truncate(terms);
            gn=(gn*g).truncate(terms);
        }
        return r.truncate(terms);
    }
    // Radius of convergence estimate (ratio test on last few coefficients)
    [[nodiscard]] f64 radius_of_convergence() const {
        if(coeffs.size()<3) return std::numeric_limits<f64>::infinity();
        f64 L=0; u32 cnt=0;
        for(u32 i=u32(coeffs.size())-3;i<u32(coeffs.size())-1;++i) {
            if(std::abs(coeffs[i])>1e-15) { L+=std::abs(coeffs[i+1]/coeffs[i]); ++cnt; }
        }
        return cnt>0 ? 1.0/(L/cnt) : std::numeric_limits<f64>::infinity();
    }
};

// ─── Taylor series generators ─────────────────────────────────────────────────
// Taylor series of f around x0, N terms via repeated differentiation (using dual numbers)
[[nodiscard]] inline PowerSeries taylor_series(const std::function<f64(f64)>& f, f64 x0, u32 N) {
    // Use finite differences for each derivative
    std::vector<f64> c(N);
    f64 fac=1;
    c[0]=f(x0);
    std::function<f64(u32,f64,f64)> deriv=[&](u32 n,f64 x,f64 h)->f64 {
        if(n==0) return f(x);
        return (deriv(n-1,x+h,h)-deriv(n-1,x-h,h))/(2*h);
    };
    for(u32 n=1;n<N;++n) {
        if(n>1) fac*=n;
        f64 h=std::pow(1e-2,1.0/n); // adaptive step
        c[n]=deriv(n,x0,h)/fac;
    }
    return PowerSeries(c);
}

// Common Taylor series (exact coefficients)
namespace series {
    // exp(x): Σ x^n/n!
    [[nodiscard]] inline PowerSeries exp(u32 N=12) {
        std::vector<f64> c(N); f64 fac=1;
        for(u32 n=0;n<N;++n){c[n]=1.0/fac;if(n>0)fac*=(n+1);else fac=1;}
        // recompute properly
        c[0]=1; fac=1;
        for(u32 n=1;n<N;++n){fac*=n; c[n]=1.0/fac;}
        return PowerSeries(c);
    }
    // sin(x): Σ (-1)^n x^{2n+1}/(2n+1)!
    [[nodiscard]] inline PowerSeries sin(u32 N=12) {
        std::vector<f64> c(2*N+2,0); f64 fac=1;
        for(u32 n=0;n<N;++n){
            u32 k=2*n+1;
            fac=1; for(u32 i=1;i<=k;++i) fac*=i;
            c[k]=(n%2==0?1.0:-1.0)/fac;
        }
        return PowerSeries(c);
    }
    // cos(x): Σ (-1)^n x^{2n}/(2n)!
    [[nodiscard]] inline PowerSeries cos(u32 N=12) {
        std::vector<f64> c(2*N+1,0); f64 fac=1;
        for(u32 n=0;n<N;++n){
            u32 k=2*n;
            fac=1; for(u32 i=1;i<=k;++i) fac*=i;
            if(k==0) fac=1;
            c[k]=(n%2==0?1.0:-1.0)/fac;
        }
        return PowerSeries(c);
    }
    // ln(1+x): Σ (-1)^{n+1} x^n/n
    [[nodiscard]] inline PowerSeries log1p(u32 N=20) {
        std::vector<f64> c(N+1,0);
        for(u32 n=1;n<=N;++n) c[n]=(n%2==1?1.0:-1.0)/n;
        return PowerSeries(c);
    }
    // atan(x): Σ (-1)^n x^{2n+1}/(2n+1)
    [[nodiscard]] inline PowerSeries atan(u32 N=20) {
        std::vector<f64> c(2*N+2,0);
        for(u32 n=0;n<N;++n) c[2*n+1]=(n%2==0?1.0:-1.0)/(2*n+1);
        return PowerSeries(c);
    }
    // sqrt(1+x): Σ C(1/2,n) x^n
    [[nodiscard]] inline PowerSeries sqrt1p(u32 N=15) {
        std::vector<f64> c(N); c[0]=1;
        f64 coeff=1;
        for(u32 n=1;n<N;++n){
            coeff*=(0.5-(n-1))/n;
            c[n]=coeff;
        }
        return PowerSeries(c);
    }
}

// ─── Laurent series: f(x) = Σ c_n x^n, n from -M to N ───────────────────────
struct LaurentSeries {
    i32 min_power{0};      // lowest power
    std::vector<Cx> coeffs; // coeffs[k] ↔ power (min_power + k)

    LaurentSeries() = default;
    LaurentSeries(i32 min_p, std::vector<Cx> c) : min_power(min_p), coeffs(std::move(c)) {}

    [[nodiscard]] Cx eval(Cx z) const {
        Cx r(0,0);
        Cx zn=std::pow(z,min_power);
        Cx zinc=z; // multiplied each step
        for(u32 k=0;k<u32(coeffs.size());++k) {
            r+=coeffs[k]*std::pow(z,f64(min_power+i32(k)));
        }
        return r;
    }
    // Residue: coefficient of z^{-1} term
    [[nodiscard]] Cx residue() const {
        i32 neg1_idx=(-1-min_power); // index of z^{-1}
        if(neg1_idx<0||neg1_idx>=i32(coeffs.size())) return {0,0};
        return coeffs[u32(neg1_idx)];
    }
    // Principal part (negative powers)
    [[nodiscard]] LaurentSeries principal_part() const {
        if(min_power>=0) return LaurentSeries(0,{});
        u32 np=u32(-min_power);
        std::vector<Cx> c(np);
        for(u32 k=0;k<np&&k<u32(coeffs.size());++k) c[k]=coeffs[k];
        return LaurentSeries(min_power,c);
    }
};

// Laurent expansion around a pole via contour integral (numerical)
[[nodiscard]] inline LaurentSeries laurent_expand(const std::function<Cx(Cx)>& f, Cx z0,
    i32 min_pow=-3, i32 max_pow=5, u32 quad_pts=64)
{
    i32 N=max_pow-min_pow+1;
    std::vector<Cx> c(u32(N));
    f64 r=0.5; // contour radius
    for(i32 k=min_pow;k<=max_pow;++k) {
        Cx integral(0,0);
        for(u32 j=0;j<quad_pts;++j) {
            f64 theta=2*std::numbers::pi_v<f64>*j/quad_pts;
            Cx z=z0+Cx(r*std::cos(theta),r*std::sin(theta));
            Cx dz=Cx(-r*std::sin(theta),r*std::cos(theta))*Cx(0,2*std::numbers::pi_v<f64>/quad_pts);
            integral+=f(z)*std::pow(z-z0,f64(-k-1))*dz;
        }
        c[u32(k-min_pow)]=integral/(Cx(0,2*std::numbers::pi_v<f64>));
    }
    return LaurentSeries(min_pow,c);
}

// ─── Padé approximant [L/M] — rational function matching Taylor series ────────
struct PadeApprox {
    std::vector<f64> num, den; // numerator and denominator coefficients

    static PadeApprox from_series(const PowerSeries& s, u32 L, u32 M) {
        // Solve linear system for Padé coefficients
        // P(x) = Σ p_k x^k, Q(x) = 1 + Σ q_k x^k, P = Q * f(x) mod x^{L+M+1}
        u32 n=L+M+1;
        std::vector<f64> c=s.truncate(n).coeffs;
        c.resize(n,0);
        // Build system for q: Σ c_{k-j} q_j = -c_k for k=L+1..L+M
        DenseMatrix<f64> A(M,M); std::vector<f64> b(M);
        for(u32 k=0;k<M;++k) {
            b[k]=-c[L+1+k];
            for(u32 j=0;j<M;++j) {
                u32 idx=L+1+k-j-1;
                A.at(k,j)=idx<n?c[idx]:0;
            }
        }
        PadeApprox pa;
        pa.den.resize(M+1,0); pa.den[0]=1;
        auto q=solve(A,b);
        if(q) for(u32 j=0;j<M;++j) pa.den[j+1]=(*q)[j];
        // Compute numerator from P = Q*f truncated
        pa.num.resize(L+1,0);
        for(u32 k=0;k<=L;++k) {
            pa.num[k]=c[k];
            for(u32 j=1;j<=std::min(k,M);++j)
                pa.num[k]+=pa.den[j]*(k-j<n?c[k-j]:0);
        }
        return pa;
    }

    [[nodiscard]] f64 eval(f64 x) const {
        f64 n_val=0,d_val=0;
        f64 xn=1;
        for(auto v:num){n_val+=v*xn;xn*=x;}
        xn=1;
        for(auto v:den){d_val+=v*xn;xn*=x;}
        return std::abs(d_val)>1e-30?n_val/d_val:0;
    }
};

// ─── Series acceleration — Aitken's Δ² and Shanks transform ──────────────────
[[nodiscard]] inline f64 aitken_delta2(f64 s0, f64 s1, f64 s2) {
    f64 d1=s1-s0, d2=s2-s1;
    f64 dd=d2-d1;
    return std::abs(dd)<1e-20?s2:s2-d2*d2/dd;
}

// Apply Aitken repeatedly to a sequence of partial sums
[[nodiscard]] inline std::vector<f64> aitken_acceleration(const std::vector<f64>& sums) {
    u32 N=u32(sums.size());
    if(N<3) return sums;
    std::vector<f64> r(N-2);
    for(u32 i=0;i+2<N;++i) r[i]=aitken_delta2(sums[i],sums[i+1],sums[i+2]);
    return r;
}

// Euler transform for alternating series: Σ(-1)^n a_n
[[nodiscard]] inline f64 euler_transform(const std::vector<f64>& a) {
    u32 N=u32(a.size());
    std::vector<f64> diffs=a;
    f64 sum=0; f64 pow2=1;
    for(u32 k=0;k<N;++k) {
        sum+=diffs[0]/(2*pow2);
        // Forward differences
        for(u32 j=0;j+1<u32(diffs.size());++j) diffs[j]=diffs[j+1]-diffs[j];
        diffs.pop_back();
        pow2*=2;
        if(diffs.empty()) break;
    }
    return sum;
}

// Bernoulli numbers B_n (first N)
[[nodiscard]] inline std::vector<f64> bernoulli_numbers(u32 N) {
    std::vector<f64> B(N,0); B[0]=1;
    for(u32 m=1;m<N;++m) {
        f64 s=0; f64 binom=1;
        for(u32 k=0;k<m;++k) {
            s+=binom*B[k];
            binom*=f64(m-k)/f64(k+2);
        }
        B[m]=-s;
    }
    return B;
}

} // namespace qs
