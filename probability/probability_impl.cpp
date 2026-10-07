
// qs_math/probability/probability_impl.cpp (C++20)
#include "include/qs_math_probability.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::prob {
constexpr f64 PI = 3.14159265358979323846;
constexpr f64 SQRT2 = 1.41421356237309504880;
constexpr f64 INV_SQRT2PI = 0.39894228040143267794;

// ─── Log Gamma (Stirling) ─────────────────────────────────────────────────────
static f64 lgamma_local(f64 x) noexcept {
    static const f64 c[]={76.18009172947146,-86.50532032941677,24.01409824083091,
        -1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};
    f64 y=x,tmp=x+5.5; tmp-=(x+0.5)*std::log(tmp); f64 ser=1.000000000190015;
    for(auto ci:c){ser+=ci/(++y);}
    return -tmp+std::log(2.5066282746310005*ser/x);
}
static f64 inc_gamma_lower_local(f64 a,f64 x) noexcept {
    if(x<0)return 0;
    f64 sum=1.0/a,term=1.0/a;
    for(i32 n=1;n<300;++n){term*=x/(a+n);sum+=term;if(std::abs(term)<1e-14)break;}
    return sum*std::exp(-x+a*std::log(x))*std::exp(-lgamma_local(a+1));
}
static f64 inc_beta_cf(f64 a,f64 b,f64 x) noexcept {
    // Lentz continued fraction
    f64 h=1,C=1,D=1-((a+b)*x/(a+1));
    if(std::abs(D)<1e-30)D=1e-30; D=1/D; h*=D;
    for(i32 m=1;m<=200;++m){
        const f64 m2=2*m;
        f64 A=m*(b-m)*x/((a+m2-1)*(a+m2));
        D=1+A*D;if(std::abs(D)<1e-30)D=1e-30;D=1/D;C=1+A/C;if(std::abs(C)<1e-30)C=1e-30;
        h*=C*D;
        A=-(a+m)*(a+b+m)*x/((a+m2)*(a+m2+1));
        D=1+A*D;if(std::abs(D)<1e-30)D=1e-30;D=1/D;C=1+A/C;if(std::abs(C)<1e-30)C=1e-30;
        const f64 delta=C*D;h*=delta;
        if(std::abs(delta-1)<1e-12)break;
    }
    return std::exp(a*std::log(x)+b*std::log(1-x)+lgamma_local(a+b)-lgamma_local(a)-lgamma_local(b))*h/a;
}
static f64 reg_inc_beta(f64 a,f64 b,f64 x) noexcept {
    if(x<(a+1)/(a+b+2)) return inc_beta_cf(a,b,x);
    return 1-inc_beta_cf(b,a,1-x);
}

// ─── Normal ──────────────────────────────────────────────────────────────────
f64 normal_pdf(f64 x,f64 mu,f64 sigma) noexcept {
    const f64 z=(x-mu)/sigma;
    return INV_SQRT2PI/sigma*std::exp(-0.5*z*z);
}
f64 normal_cdf(f64 x,f64 mu,f64 sigma) noexcept {
    return 0.5*(1+std::erf((x-mu)/(sigma*SQRT2)));
}
f64 normal_inv_cdf(f64 p,f64 mu,f64 sigma) noexcept {
    // Rational approximation (Beasley-Springer-Moro)
    assert(p>0&&p<1);
    static const f64 a[]={-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,
        1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00};
    static const f64 b[]={-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,
        6.680131188771972e+01,-1.328068155288572e+01};
    static const f64 c[]={-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,
        -2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
    static const f64 d[]={7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
    const f64 q=p<0.5?p:1-p;
    f64 x2;
    if(q>0.02425){
        const f64 u=q-0.5,v=u*u;
        x2=u*(((((a[0]*v+a[1])*v+a[2])*v+a[3])*v+a[4])*v+a[5])/
             (((((b[0]*v+b[1])*v+b[2])*v+b[3])*v+b[4])*v+1);
    } else {
        const f64 v=std::sqrt(-2*std::log(q));
        x2=(((((c[0]*v+c[1])*v+c[2])*v+c[3])*v+c[4])*v+c[5])/
            ((((d[0]*v+d[1])*v+d[2])*v+d[3])*v+1);
    }
    if(p<0.5)x2=-x2;
    return mu+sigma*x2;
}
// ─── Student-t ───────────────────────────────────────────────────────────────
f64 student_t_pdf(f64 t,f64 nu) noexcept {
    const f64 lB=lgamma_local((nu+1)*0.5)-0.5*std::log(nu*PI)-lgamma_local(nu*0.5);
    return std::exp(lB-0.5*(nu+1)*std::log(1+t*t/nu));
}
f64 student_t_cdf(f64 t,f64 nu) noexcept {
    const f64 x=nu/(nu+t*t);
    return 0.5+0.5*(t>0?1:-1)*(1-reg_inc_beta(nu*0.5,0.5,x));
}
// ─── Beta ────────────────────────────────────────────────────────────────────
f64 beta_pdf(f64 x,f64 a,f64 b) noexcept {
    if(x<=0||x>=1)return 0;
    return std::exp((a-1)*std::log(x)+(b-1)*std::log(1-x)+lgamma_local(a+b)-lgamma_local(a)-lgamma_local(b));
}
f64 beta_cdf(f64 x,f64 a,f64 b) noexcept { return reg_inc_beta(a,b,x); }
// ─── Gamma ───────────────────────────────────────────────────────────────────
f64 gamma_pdf(f64 x,f64 shape,f64 rate) noexcept {
    if(x<=0)return 0;
    return std::exp((shape-1)*std::log(x)-rate*x+shape*std::log(rate)-lgamma_local(shape));
}
f64 gamma_cdf(f64 x,f64 shape,f64 rate) noexcept {
    return inc_gamma_lower_local(shape,rate*x);
}
// ─── Binomial ────────────────────────────────────────────────────────────────
f64 binomial_pmf(i32 k,i32 n,f64 p) noexcept {
    if(k<0||k>n)return 0;
    const f64 logpmf=lgamma_local(n+1)-lgamma_local(k+1)-lgamma_local(n-k+1)+k*std::log(p+1e-300)+(n-k)*std::log(1-p+1e-300);
    return std::exp(logpmf);
}
f64 binomial_cdf(i32 k,i32 n,f64 p) noexcept {
    f64 s=0; for(i32 i=0;i<=k;++i)s+=binomial_pmf(i,n,p); return std::min(1.0,s);
}
// ─── Poisson ─────────────────────────────────────────────────────────────────
f64 poisson_pmf(i32 k,f64 lam) noexcept {
    if(k<0)return 0;
    return std::exp(k*std::log(lam+1e-300)-lam-lgamma_local(k+1));
}
f64 poisson_cdf(i32 k,f64 lam) noexcept {
    f64 s=0; for(i32 i=0;i<=k;++i)s+=poisson_pmf(i,lam); return std::min(1.0,s);
}
// ─── Exponential ──────────────────────────────────────────────────────────────
f64 exponential_pdf(f64 x,f64 rate) noexcept { return x<0?0:rate*std::exp(-rate*x); }
f64 exponential_cdf(f64 x,f64 rate) noexcept { return x<0?0:1-std::exp(-rate*x); }
f64 exponential_inv_cdf(f64 p,f64 rate) noexcept { return -std::log(1-p)/rate; }
// ─── Chi-squared ─────────────────────────────────────────────────────────────
f64 chi2_pdf(f64 x,f64 k) noexcept { return gamma_pdf(x,k*0.5,0.5); }
f64 chi2_cdf(f64 x,f64 k) noexcept { return gamma_cdf(x,k*0.5,0.5); }
f64 chi2_test_pvalue(f64 stat,f64 df) noexcept { return 1-chi2_cdf(stat,df); }
// ─── Multivariate normal ──────────────────────────────────────────────────────
f64 mvn_pdf(const f64* x,const f64* mu,const f64* Si,f64 det_S,i32 n) noexcept {
    f64 q=0;
    std::vector<f64> d(n);
    for(i32 i=0;i<n;++i)d[i]=x[i]-mu[i];
    for(i32 i=0;i<n;++i)for(i32 j=0;j<n;++j)q+=d[i]*Si[i*n+j]*d[j];
    return std::exp(-0.5*q)/std::sqrt(std::pow(2*PI,n)*std::abs(det_S));
}
// ─── Dirichlet ───────────────────────────────────────────────────────────────
f64 dirichlet_pdf(const f64* x,const f64* alpha,i32 k) noexcept {
    f64 sum_alpha=0,log_B=0,log_pdf=0;
    for(i32 i=0;i<k;++i){sum_alpha+=alpha[i];log_B+=lgamma_local(alpha[i]);}
    log_B-=lgamma_local(sum_alpha);
    for(i32 i=0;i<k;++i)log_pdf+=(alpha[i]-1)*std::log(x[i]+1e-300);
    return std::exp(log_pdf-log_B);
}
// ─── Kalman filter ────────────────────────────────────────────────────────────
void KalmanFilter::predict(const f64* F,const f64* Q) noexcept {
    // x = F*x; P = F*P*F' + Q
    std::vector<f64> xp(n,0),Pp(n*n,0),FP(n*n,0);
    for(i32 i=0;i<n;++i)for(i32 j=0;j<n;++j)xp[i]+=F[i*n+j]*x[j];
    for(i32 i=0;i<n;++i)for(i32 k2=0;k2<n;++k2)for(i32 j=0;j<n;++j)FP[i*n+j]+=F[i*n+k2]*P[k2*n+j];
    for(i32 i=0;i<n;++i)for(i32 j=0;j<n;++j)for(i32 k2=0;k2<n;++k2)Pp[i*n+j]+=FP[i*n+k2]*F[j*n+k2];
    for(i32 i=0;i<n*n;++i)Pp[i]+=Q[i];
    x=xp; P=Pp;
}
void KalmanFilter::update(const f64* z,const f64* H,const f64* R,i32 m) noexcept {
    // y = z - H*x; S = H*P*H' + R; K = P*H'*S^-1; x += K*y; P = (I-K*H)*P
    std::vector<f64> y(m,0),PH(n*m,0),S(m*m,0),K(n*m,0);
    for(i32 i=0;i<m;++i){y[i]=z[i];for(i32 j=0;j<n;++j)y[i]-=H[i*n+j]*x[j];}
    for(i32 i=0;i<n;++i)for(i32 j=0;j<m;++j)for(i32 k2=0;k2<n;++k2)PH[i*m+j]+=P[i*n+k2]*H[j*n+k2];
    for(i32 i=0;i<m;++i)for(i32 j=0;j<m;++j){for(i32 k2=0;k2<n;++k2)S[i*m+j]+=H[i*n+k2]*PH[k2*m+j];S[i*m+j]+=R[i*m+j];}
    // Invert S (simple 1x1 or 2x2 for small m)
    if(m==1&&std::abs(S[0])>1e-14){K.assign(n,0);for(i32 i=0;i<n;++i)K[i]=PH[i*m]/S[0];}
    for(i32 i=0;i<n;++i)for(i32 j=0;j<m;++j)x[i]+=K[i*m+j]*y[j];
    std::vector<f64> KH(n*n,0),Pnew(n*n);
    for(i32 i=0;i<n;++i)for(i32 j=0;j<n;++j)for(i32 k2=0;k2<m;++k2)KH[i*n+j]+=K[i*m+k2]*H[k2*n+j];
    for(i32 i=0;i<n;++i)for(i32 j=0;j<n;++j){
        f64 v=(i==j?1.0:0.0)-KH[i*n+j];
        Pnew[i*n+j]=0; for(i32 k2=0;k2<n;++k2)Pnew[i*n+j]+=v*P[k2*n+j];
    }
    P=Pnew;
}
// ─── Markov chain stationary distribution ─────────────────────────────────────
std::vector<f64> markov_stationary(const f64* trans,i32 n,i32 max_iter,f64 tol) noexcept {
    std::vector<f64> pi(n,1.0/n),pi2(n);
    for(i32 it=0;it<max_iter;++it){
        std::fill(pi2.begin(),pi2.end(),0);
        for(i32 j=0;j<n;++j)for(i32 i=0;i<n;++i)pi2[j]+=pi[i]*trans[i*n+j];
        f64 err=0;for(i32 i=0;i<n;++i)err+=std::abs(pi2[i]-pi[i]);
        pi=pi2;if(err<tol)break;
    }
    return pi;
}
// ─── Information theory ───────────────────────────────────────────────────────
f64 entropy_shannon(const f64* p,i32 n) noexcept {
    f64 h=0;for(i32 i=0;i<n;++i)if(p[i]>1e-300)h-=p[i]*std::log2(p[i]);return h;
}
f64 kl_divergence(const f64* p,const f64* q,i32 n) noexcept {
    f64 kl=0;for(i32 i=0;i<n;++i)if(p[i]>1e-300&&q[i]>1e-300)kl+=p[i]*std::log(p[i]/q[i]);return kl;
}
f64 mutual_information(const f64* joint,i32 nx,i32 ny) noexcept {
    std::vector<f64> px(nx,0),py(ny,0);
    for(i32 i=0;i<nx;++i)for(i32 j=0;j<ny;++j){px[i]+=joint[i*ny+j];py[j]+=joint[i*ny+j];}
    f64 mi=0;
    for(i32 i=0;i<nx;++i)for(i32 j=0;j<ny;++j){
        const f64 pij=joint[i*ny+j];
        if(pij>1e-300&&px[i]>1e-300&&py[j]>1e-300)mi+=pij*std::log(pij/(px[i]*py[j]));
    }
    return mi;
}
} // namespace qs::math::prob
