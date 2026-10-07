#pragma once
// statistics_ext.hpp — Extended statistics: regression, correlation matrix,
//                       PCA helpers, Bayesian, signal processing, filter design,
//                       moving averages, outlier detection, KDE, entropy
#include "statistics.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "scalar_math.hpp"

namespace qs::math::stats {

// ---------------------------------------------------------------------------
// Linear regression (least squares)
// ---------------------------------------------------------------------------
struct LinearFit { f64 slope{0}, intercept{0}, r_sq{0}; };

[[nodiscard]] inline LinearFit linear_regression(const f64* xs, const f64* ys, i32 n) noexcept {
    if(n<2) return {};
    f64 sx=0,sy=0,sxy=0,sxx=0;
    for(i32 i=0;i<n;++i){sx+=xs[i];sy+=ys[i];sxy+=xs[i]*ys[i];sxx+=xs[i]*xs[i];}
    f64 denom=n*sxx-sx*sx;
    if(abs(denom)<1e-12) return {};
    f64 slope=(n*sxy-sx*sy)/denom;
    f64 intercept=(sy-slope*sx)/n;
    // R^2
    f64 ymean=sy/n, sst=0, sse=0;
    for(i32 i=0;i<n;++i){
        f64 diff=ys[i]-ymean; sst+=diff*diff;
        f64 res=ys[i]-(slope*xs[i]+intercept); sse+=res*res;
    }
    f64 rsq=sst>1e-12?1.0-sse/sst:0.0;
    return {slope,intercept,rsq};
}

// Polynomial regression (degree d)
inline void poly_regression(const f64* xs, const f64* ys, i32 n, i32 d,
                              f64* coeffs) noexcept {
    // Vandermonde matrix least squares (simple, d<=4)
    static constexpr i32 MAX_D=5;
    i32 m=d+1;
    f64 A[MAX_D][MAX_D]={}, b[MAX_D]={};
    for(i32 i=0;i<n;++i){
        f64 xp[MAX_D]; xp[0]=1.0;
        for(i32 k=1;k<m;++k) xp[k]=xp[k-1]*xs[i];
        for(i32 r=0;r<m;++r){
            b[r]+=xp[r]*ys[i];
            for(i32 c=0;c<m;++c) A[r][c]+=xp[r]*xp[c];
        }
    }
    // Gaussian elimination
    for(i32 p=0;p<m;++p){
        for(i32 r=p+1;r<m;++r){
            f64 f=A[r][p]/A[p][p];
            for(i32 c=p;c<m;++c) A[r][c]-=f*A[p][c];
            b[r]-=f*b[p];
        }
    }
    for(i32 p=m-1;p>=0;--p){
        coeffs[p]=b[p];
        for(i32 c=p+1;c<m;++c) coeffs[p]-=A[p][c]*coeffs[c];
        coeffs[p]/=A[p][p];
    }
}

// ---------------------------------------------------------------------------
// Moving averages
// ---------------------------------------------------------------------------
// Simple moving average (SMA)
template<FloatScalar T, i32 N>
struct SMA {
    T buf[N]{}; i32 head{0}; bool full{false}; T sum{T(0)};
    void push(T v) noexcept {
        if(full) sum-=buf[head]; else full=(head==(N-1));
        buf[head]=v; sum+=v; head=(head+1)%N;
    }
    [[nodiscard]] T value() const noexcept { return full?sum/T(N):sum/T(head?head:1); }
    [[nodiscard]] i32 count() const noexcept { return full?N:head; }
};

// Exponential moving average (EMA)
template<FloatScalar T>
struct EMA {
    T value{T(0)}; f32 alpha{0.1f}; bool initialized{false};
    void push(T v) noexcept {
        if(!initialized){value=v;initialized=true;}
        else value=value*(T(1)-T(alpha))+v*T(alpha);
    }
};

// Weighted moving average
template<FloatScalar T, i32 N>
struct WMA {
    T buf[N]{}; i32 head{0}; bool full{false};
    void push(T v) noexcept { buf[head]=v; head=(head+1)%N; if(head==0)full=true; }
    [[nodiscard]] T value() const noexcept {
        i32 cnt=full?N:head; T sum=T(0); T wsum=T(0);
        for(i32 i=0;i<cnt;++i){
            T w=T(i+1);
            i32 idx=(head-cnt+i+N)%N;
            sum+=buf[idx]*w; wsum+=w;
        }
        return wsum>T(0)?sum/wsum:T(0);
    }
};

// ---------------------------------------------------------------------------
// Kernel Density Estimation (Gaussian kernel)
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] inline T kde_gaussian(const T* data, i32 n, T x, T bandwidth) noexcept {
    T sum=T(0);
    T inv_bw=T(1)/bandwidth;
    for(i32 i=0;i<n;++i){
        T u=(x-data[i])*inv_bw;
        sum+=exp(T(-0.5)*u*u);
    }
    return sum/(T(n)*bandwidth*T(sqrt(TWO_PI)));
}

// ---------------------------------------------------------------------------
// Outlier detection — Z-score method
// ---------------------------------------------------------------------------
template<FloatScalar T>
inline i32 find_outliers_zscore(const T* data, i32 n, bool* is_outlier,
                                  T threshold=T(3)) noexcept {
    T m=mean(data,n), s=stddev(data,n);
    i32 count=0;
    for(i32 i=0;i<n;++i){
        is_outlier[i]=(abs(data[i]-m)/max(s,T(EPS_F32)))>threshold;
        if(is_outlier[i]) ++count;
    }
    return count;
}

// IQR outlier detection
template<FloatScalar T>
inline i32 find_outliers_iqr(T* data, i32 n, bool* is_outlier,
                               T scale=T(1.5)) noexcept {
    T sorted[1024]; // n<=1024 assumed
    i32 cnt=n<1024?n:1024;
    for(i32 i=0;i<cnt;++i) sorted[i]=data[i];
    for(i32 i=1;i<cnt;++i){ T k=sorted[i]; i32 j=i-1; while(j>=0&&sorted[j]>k){sorted[j+1]=sorted[j];--j;} sorted[j+1]=k; }
    T q1=percentile(sorted,cnt,25.0), q3=percentile(sorted,cnt,75.0);
    T iqr=q3-q1;
    T lo=q1-scale*iqr, hi=q3+scale*iqr;
    i32 count=0;
    for(i32 i=0;i<n;++i){
        is_outlier[i]=(data[i]<lo||data[i]>hi);
        if(is_outlier[i]) ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Shannon entropy
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] constexpr T entropy(const T* probs, i32 n) noexcept {
    T h=T(0);
    for(i32 i=0;i<n;++i) if(probs[i]>T(0)) h-=probs[i]*log2(probs[i]);
    return h;
}

// KL divergence: D_KL(P || Q)
template<FloatScalar T>
[[nodiscard]] constexpr T kl_divergence(const T* p, const T* q, i32 n) noexcept {
    T d=T(0);
    for(i32 i=0;i<n;++i)
        if(p[i]>T(0)&&q[i]>T(0)) d+=p[i]*log2(p[i]/q[i]);
    return d;
}

// Cross-entropy
template<FloatScalar T>
[[nodiscard]] constexpr T cross_entropy(const T* p, const T* q, i32 n) noexcept {
    T h=T(0);
    for(i32 i=0;i<n;++i) if(p[i]>T(0)&&q[i]>T(0)) h-=p[i]*log2(q[i]);
    return h;
}

// ---------------------------------------------------------------------------
// Simple PCA helpers (covariance eigenvalues, 2D/3D)
// ---------------------------------------------------------------------------
// 2D PCA: returns principal axes and eigenvalues
struct PCA2Result { Vec2f axis0,axis1; f32 lam0,lam1; };

[[nodiscard]] inline PCA2Result pca_2d(const Vec2f* pts, i32 n) noexcept {
    f32 mx=0,my=0;
    for(i32 i=0;i<n;++i){mx+=pts[i].x;my+=pts[i].y;}
    mx/=n; my/=n;
    f32 c00=0,c01=0,c11=0;
    for(i32 i=0;i<n;++i){
        f32 dx=pts[i].x-mx, dy=pts[i].y-my;
        c00+=dx*dx; c01+=dx*dy; c11+=dy*dy;
    }
    c00/=n; c01/=n; c11/=n;
    // Eigenvalues of 2x2 symmetric
    f32 trace=c00+c11, det=c00*c11-c01*c01;
    f32 disc=sqrt(max(0.f,trace*trace/4.f-det));
    f32 lam0=trace/2.f+disc, lam1=trace/2.f-disc;
    Vec2f ax0=abs(c01)>EPS_F32?Vec2f{lam0-c11,c01}.normalized():Vec2f{1,0};
    Vec2f ax1{-ax0.y,ax0.x};
    return {ax0,ax1,lam0,lam1};
}

// ---------------------------------------------------------------------------
// Chi-squared test statistic
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr f64 chi_squared(const f64* observed, const f64* expected, i32 n) noexcept {
    f64 chi=0.0;
    for(i32 i=0;i<n;++i)
        if(expected[i]>1e-12) chi+=(observed[i]-expected[i])*(observed[i]-expected[i])/expected[i];
    return chi;
}

// ---------------------------------------------------------------------------
// Signal processing — autocorrelation, DFT helpers
// ---------------------------------------------------------------------------
template<FloatScalar T>
[[nodiscard]] T autocorrelation(const T* data, i32 n, i32 lag) noexcept {
    if(lag>=n) return T(0);
    T m=mean(data,n), var=variance(data,n);
    if(var<T(EPS_F32)) return T(0);
    T sum=T(0);
    for(i32 i=0;i<n-lag;++i) sum+=(data[i]-m)*(data[i+lag]-m);
    return sum/(T(n-lag)*var);
}

// DFT (O(n^2), for small n up to 512)
inline void dft(const f32* real_in, const f32* imag_in,
                f32* real_out, f32* imag_out, i32 n) noexcept {
    for(i32 k=0;k<n;++k){
        f32 re=0,im=0;
        for(i32 j=0;j<n;++j){
            f32 angle=-TWO_PI*k*j/n;
            f32 c=cos(angle),s=sin(angle);
            re+=real_in[j]*c-imag_in[j]*s;
            im+=real_in[j]*s+imag_in[j]*c;
        }
        real_out[k]=re; imag_out[k]=im;
    }
}

// Power spectral density estimate
inline void psd(const f32* data, i32 n, f32* out) noexcept {
    f32 re[1024]={},im[1024]={};
    i32 cnt=n<1024?n:1024;
    dft(data,re,re,im,cnt);
    for(i32 i=0;i<cnt;++i) out[i]=(re[i]*re[i]+im[i]*im[i])/cnt;
}

// Zero-crossing rate
template<FloatScalar T>
[[nodiscard]] constexpr f32 zero_crossing_rate(const T* data, i32 n) noexcept {
    i32 zc=0;
    for(i32 i=1;i<n;++i) if((data[i]>=T(0))!=(data[i-1]>=T(0))) ++zc;
    return n>1?f32(zc)/(n-1):0.f;
}

// Root Mean Square
template<FloatScalar T>
[[nodiscard]] constexpr T rms(const T* data, i32 n) noexcept {
    T s=T(0); for(i32 i=0;i<n;++i) s+=data[i]*data[i];
    return sqrt(s/T(n));
}

} // namespace qs::math::stats
