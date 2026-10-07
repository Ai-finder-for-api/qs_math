#pragma once
// qs_math/statistics/probability.hpp — Probability distributions: PDF/CDF/quantile
//   for Normal, Poisson, Binomial, Beta, Gamma, Chi2, Student-t, F, Weibull,
//   Exponential, Uniform, LogNormal (C++20)
#include "../arithmetic/scalar_math.hpp"
#include "../arithmetic/constants.hpp"
#include <cmath>
#include <vector>

namespace qs::math::statistics {

// ─── Special functions ────────────────────────────────────────────────────
// Error function complement (Horner approximation, |err|<1.2e-7)
[[nodiscard]] inline f64 erfc_approx(f64 x) noexcept {
    if(x<0) return 2.0-erfc_approx(-x);
    f64 t=1.0/(1.0+0.3275911*x);
    f64 p=t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))));
    return p*std::exp(-x*x);
}

[[nodiscard]] inline f64 phi(f64 x) noexcept { return 0.5*std::erfc(-x*0.7071067811865476); }
[[nodiscard]] inline f64 phi_inv(f64 p) noexcept {
    // Rational approximation — Beasley-Springer-Moro
    if(p<=0) return -1e300; if(p>=1) return 1e300;
    if(p<0.02425||p>0.97575){
        f64 r=p<0.5?p:1-p;
        f64 q=std::sqrt(-2*std::log(r));
        f64 c[]={-7.784894002430293e-3,-0.3223964580411365,-2.400758277161838,
                 -2.549732539343734,4.374664141464968,2.938163982698783};
        f64 d[]={7.784695709041462e-3,0.3224671290700398,2.445134137142996,3.754408661907416};
        f64 x=(c[0]+q*(c[1]+q*(c[2]+q*(c[3]+q*(c[4]+q*c[5])))))
              /(1+q*(d[0]+q*(d[1]+q*(d[2]+q*d[3]))));
        return p<0.5?x:-x;
    }
    f64 q=p-0.5, r=q*q;
    f64 a[]={-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,
              1.383577518672690e2,-3.066479806614716e1,2.506628277459239};
    f64 b[]={-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,
              6.680131188771972e1,-1.328068155288572e1};
    return q*(a[0]+r*(a[1]+r*(a[2]+r*(a[3]+r*(a[4]+r*a[5])))))
            /(1+r*(b[0]+r*(b[1]+r*(b[2]+r*(b[3]+r*b[4])))));
}

// Log-gamma (Stirling series)
[[nodiscard]] inline f64 lgamma_approx(f64 x) noexcept { return std::lgamma(x); }

// Incomplete gamma (lower) via series — regularized P(a,x)
[[nodiscard]] inline f64 igamma_p(f64 a, f64 x) noexcept {
    if(x<0||a<=0) return 0;
    if(x==0) return 0;
    if(x<a+1){
        f64 ap=a,s=1.0/a,d=s;
        for(int n=1;n<200;++n){ap++;d*=x/ap;s+=d;if(std::abs(d)<std::abs(s)*1e-13)break;}
        return s*std::exp(-x+a*std::log(x)-lgamma_approx(a));
    }
    // Continued fraction (Lentz)
    f64 b=x+1-a,c=1e300,d=1.0/b,h=d,fpmin=1e-300;
    for(int n=1;n<=200;++n){
        f64 an=-n*(n-a); b+=2; d=an*d+b; if(std::abs(d)<fpmin) d=fpmin;
        c=b+an/c; if(std::abs(c)<fpmin) c=fpmin;
        d=1.0/d; f64 del=d*c; h*=del;
        if(std::abs(del-1)<1e-13) break;
    }
    return 1.0-std::exp(-x+a*std::log(x)-lgamma_approx(a))*h;
}
[[nodiscard]] inline f64 igamma_q(f64 a, f64 x) noexcept { return 1.0-igamma_p(a,x); }

// Regularized incomplete beta I_x(a,b) via continued fraction
[[nodiscard]] inline f64 ibeta(f64 x, f64 a, f64 b) noexcept {
    if(x<0||x>1) return 0; if(x==0) return 0; if(x==1) return 1;
    f64 lbeta=lgamma_approx(a)+lgamma_approx(b)-lgamma_approx(a+b);
    f64 front=std::exp(std::log(x)*a+std::log(1-x)*b-lbeta)/a;
    // Lentz
    f64 c=1,d=1.0-((a+b)*x/(a+1)),fpmin=1e-300;
    if(std::abs(d)<fpmin) d=fpmin; d=1.0/d; f64 h=d;
    for(int m=1;m<=200;++m){
        f64 m2=2*m;
        f64 aa=m*(b-m)*x/((a+m2-1)*(a+m2));
        d=1+aa*d; if(std::abs(d)<fpmin)d=fpmin; c=1+aa/c; if(std::abs(c)<fpmin)c=fpmin;
        d=1/d; h*=d*c;
        aa=-(a+m)*(a+b+m)*x/((a+m2)*(a+m2+1));
        d=1+aa*d; if(std::abs(d)<fpmin)d=fpmin; c=1+aa/c; if(std::abs(c)<fpmin)c=fpmin;
        d=1/d; f64 del=d*c; h*=del;
        if(std::abs(del-1)<1e-13) break;
    }
    f64 Ix=front*h;
    if(x>(a+1)/(a+b+2)) Ix=1.0-ibeta(1-x,b,a);
    return clamp(Ix,0.0,1.0);
}

// ─── Normal distribution ──────────────────────────────────────────────────
struct Normal {
    f64 mu=0, sigma=1;
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        f64 z=(x-mu)/sigma;
        return std::exp(-0.5*z*z)/(sigma*std::sqrt(2.0*PI));
    }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return phi((x-mu)/sigma); }
    [[nodiscard]] f64 quantile(f64 p) const noexcept { return mu+sigma*phi_inv(p); }
    [[nodiscard]] f64 mean() const noexcept { return mu; }
    [[nodiscard]] f64 variance() const noexcept { return sigma*sigma; }
    [[nodiscard]] f64 skewness() const noexcept { return 0; }
    [[nodiscard]] f64 excess_kurtosis() const noexcept { return 0; }
};

// ─── Log-Normal ───────────────────────────────────────────────────────────
struct LogNormal {
    f64 mu=0, sigma=1;
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<=0) return 0;
        f64 z=(std::log(x)-mu)/sigma;
        return std::exp(-0.5*z*z)/(x*sigma*std::sqrt(2.0*PI));
    }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return x>0?phi((std::log(x)-mu)/sigma):0; }
    [[nodiscard]] f64 mean() const noexcept { return std::exp(mu+0.5*sigma*sigma); }
    [[nodiscard]] f64 median() const noexcept { return std::exp(mu); }
};

// ─── Exponential distribution ─────────────────────────────────────────────
struct Exponential {
    f64 lambda=1;
    [[nodiscard]] f64 pdf(f64 x) const noexcept { return x>=0?lambda*std::exp(-lambda*x):0; }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return x>=0?1-std::exp(-lambda*x):0; }
    [[nodiscard]] f64 quantile(f64 p) const noexcept { return -std::log(1-p)/lambda; }
    [[nodiscard]] f64 mean() const noexcept { return 1/lambda; }
    [[nodiscard]] f64 variance() const noexcept { return 1/(lambda*lambda); }
};

// ─── Gamma distribution ───────────────────────────────────────────────────
struct Gamma {
    f64 alpha=1, beta=1; // shape, rate
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<=0) return 0;
        return std::exp((alpha-1)*std::log(x)-beta*x-lgamma_approx(alpha)+alpha*std::log(beta));
    }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return x>0?igamma_p(alpha,beta*x):0; }
    [[nodiscard]] f64 mean() const noexcept { return alpha/beta; }
    [[nodiscard]] f64 variance() const noexcept { return alpha/(beta*beta); }
};

// ─── Beta distribution ────────────────────────────────────────────────────
struct Beta {
    f64 alpha=1, beta=1;
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<=0||x>=1) return 0;
        return std::exp((alpha-1)*std::log(x)+(beta-1)*std::log(1-x)
               -lgamma_approx(alpha)-lgamma_approx(beta)+lgamma_approx(alpha+beta));
    }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return ibeta(x,alpha,beta); }
    [[nodiscard]] f64 mean() const noexcept { return alpha/(alpha+beta); }
    [[nodiscard]] f64 variance() const noexcept {
        f64 s=alpha+beta; return alpha*beta/(s*s*(s+1));
    }
    [[nodiscard]] f64 mode() const noexcept {
        return (alpha>1&&beta>1)?(alpha-1)/(alpha+beta-2):0;
    }
};

// ─── Poisson distribution ─────────────────────────────────────────────────
struct Poisson {
    f64 lambda=1;
    [[nodiscard]] f64 pmf(int k) const noexcept {
        if(k<0) return 0;
        return std::exp(k*std::log(lambda)-lambda-lgamma_approx(k+1));
    }
    [[nodiscard]] f64 cdf(int k) const noexcept { return igamma_q(k+1,lambda); }
    [[nodiscard]] f64 mean() const noexcept { return lambda; }
    [[nodiscard]] f64 variance() const noexcept { return lambda; }
};

// ─── Binomial distribution ────────────────────────────────────────────────
struct Binomial {
    int n=10; f64 p=0.5;
    [[nodiscard]] f64 pmf(int k) const noexcept {
        if(k<0||k>n) return 0;
        f64 lc=lgamma_approx(n+1)-lgamma_approx(k+1)-lgamma_approx(n-k+1);
        return std::exp(lc+k*std::log(p)+(n-k)*std::log(1-p));
    }
    [[nodiscard]] f64 cdf(int k) const noexcept { return ibeta(1-p,n-k,k+1); }
    [[nodiscard]] f64 mean() const noexcept { return n*p; }
    [[nodiscard]] f64 variance() const noexcept { return n*p*(1-p); }
};

// ─── Chi-squared distribution ─────────────────────────────────────────────
struct ChiSquared {
    int df=1;
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<=0) return 0;
        f64 k=0.5*df;
        return std::exp((k-1)*std::log(x)-0.5*x-lgamma_approx(k)-k*std::log(2.0));
    }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return x>0?igamma_p(0.5*df,0.5*x):0; }
    [[nodiscard]] f64 quantile(f64 p) const noexcept {
        // Newton on CDF
        f64 x=std::max(df-2.0+std::sqrt(2.0*df)*phi_inv(p),0.001);
        for(int i=0;i<50;++i){
            f64 c=cdf(x), d=pdf(x);
            if(d<1e-300) break;
            x-=(c-p)/d; x=std::max(x,1e-10);
        }
        return x;
    }
    [[nodiscard]] f64 mean() const noexcept { return df; }
    [[nodiscard]] f64 variance() const noexcept { return 2.0*df; }
};

// ─── Student's t-distribution ─────────────────────────────────────────────
struct StudentT {
    f64 nu=1; // degrees of freedom
    [[nodiscard]] f64 pdf(f64 t) const noexcept {
        f64 c=lgamma_approx(0.5*(nu+1))-lgamma_approx(0.5*nu)-0.5*std::log(nu*PI);
        return std::exp(c-(0.5*(nu+1))*std::log(1+t*t/nu));
    }
    [[nodiscard]] f64 cdf(f64 t) const noexcept {
        f64 x=nu/(nu+t*t);
        f64 p=0.5*ibeta(x,0.5*nu,0.5);
        return t>=0?1-p:p;
    }
    [[nodiscard]] f64 mean() const noexcept { return nu>1?0:std::numeric_limits<f64>::quiet_NaN(); }
    [[nodiscard]] f64 variance() const noexcept { return nu>2?nu/(nu-2):std::numeric_limits<f64>::infinity(); }
};

// ─── Weibull distribution ─────────────────────────────────────────────────
struct Weibull {
    f64 k=1, lambda=1; // shape, scale
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<0) return 0;
        f64 r=x/lambda;
        return (k/lambda)*std::pow(r,k-1)*std::exp(-std::pow(r,k));
    }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return x>=0?1-std::exp(-std::pow(x/lambda,k)):0; }
    [[nodiscard]] f64 quantile(f64 p) const noexcept { return lambda*std::pow(-std::log(1-p),1/k); }
    [[nodiscard]] f64 mean() const noexcept { return lambda*std::exp(lgamma_approx(1+1/k)); }
};

// ─── Uniform distribution ─────────────────────────────────────────────────
struct Uniform {
    f64 a=0, b=1;
    [[nodiscard]] f64 pdf(f64 x) const noexcept { return (x>=a&&x<=b)?1/(b-a):0; }
    [[nodiscard]] f64 cdf(f64 x) const noexcept { return x<a?0:(x>b?1:(x-a)/(b-a)); }
    [[nodiscard]] f64 quantile(f64 p) const noexcept { return a+p*(b-a); }
    [[nodiscard]] f64 mean() const noexcept { return 0.5*(a+b); }
    [[nodiscard]] f64 variance() const noexcept { f64 d=b-a; return d*d/12; }
};

// ─── F-distribution ───────────────────────────────────────────────────────
struct FDist {
    f64 d1=1, d2=1;
    [[nodiscard]] f64 pdf(f64 x) const noexcept {
        if(x<=0) return 0;
        f64 lc=lgamma_approx(0.5*(d1+d2))-lgamma_approx(0.5*d1)-lgamma_approx(0.5*d2)
               +0.5*d1*std::log(d1/d2);
        return std::exp(lc+(0.5*d1-1)*std::log(x)-(0.5*(d1+d2))*std::log(1+d1*x/d2));
    }
    [[nodiscard]] f64 cdf(f64 x) const noexcept {
        return x>0?ibeta(d1*x/(d1*x+d2),0.5*d1,0.5*d2):0;
    }
    [[nodiscard]] f64 mean() const noexcept { return d2>2?d2/(d2-2):1e300; }
};

// ─── Hypothesis tests ─────────────────────────────────────────────────────
// One-sample z-test
struct ZTestResult { f64 z_stat, p_value; bool reject; };
[[nodiscard]] inline ZTestResult z_test_one_sample(
    std::span<const f64> x, f64 mu0, f64 sigma, f64 alpha=0.05) noexcept
{
    int n=int(x.size());
    f64 m=0; for(auto v:x) m+=v; m/=n;
    f64 z=(m-mu0)/(sigma/std::sqrt(f64(n)));
    f64 p=2*(1-phi(std::abs(z)));
    return {z,p,p<alpha};
}

// Two-sample t-test (Welch)
struct TTestResult { f64 t_stat, p_value, df; bool reject; };
[[nodiscard]] inline TTestResult t_test_two_sample(
    std::span<const f64> x, std::span<const f64> y, f64 alpha=0.05) noexcept
{
    int nx=int(x.size()), ny=int(y.size());
    f64 mx=0,my=0; for(auto v:x) mx+=v; mx/=nx; for(auto v:y) my+=v; my/=ny;
    f64 sx2=0,sy2=0;
    for(auto v:x){f64 d=v-mx;sx2+=d*d;} sx2/=(nx-1);
    for(auto v:y){f64 d=v-my;sy2+=d*d;} sy2/=(ny-1);
    f64 se=std::sqrt(sx2/nx+sy2/ny);
    f64 t=(mx-my)/se;
    f64 nu=(sx2/nx+sy2/ny)*(sx2/nx+sy2/ny)/
            ((sx2/nx)*(sx2/nx)/(nx-1)+(sy2/ny)*(sy2/ny)/(ny-1));
    StudentT td{nu}; f64 p=2*(1-td.cdf(std::abs(t)));
    return {t,p,nu,p<alpha};
}

} // namespace qs::math::statistics
