
// qs_math/spline/spline_impl.cpp (C++20)
#include "include/qs_math_spline.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::spline {

// ─── Thomas algorithm ─────────────────────────────────────────────────────────
static void thomas(i32 n, f64* a, f64* b, f64* c, f64* d, f64* x) noexcept {
    for(i32 i=1;i<n;++i){const f64 m=a[i]/b[i-1];b[i]-=m*c[i-1];d[i]-=m*d[i-1];}
    x[n-1]=d[n-1]/b[n-1];
    for(i32 i=n-2;i>=0;--i)x[i]=(d[i]-c[i]*x[i+1])/b[i];
}
void NaturalSpline::fit(const f64* x,const f64* y,i32 n) noexcept {
    assert(n>=2);
    xs.assign(x,x+n); ys.assign(y,y+n); segs.resize(n-1);
    std::vector<f64> h(n-1),alpha(n-1),a(n,0),b(n,1),c(n,0),d(n,0),m(n);
    for(i32 i=0;i<n-1;++i)h[i]=x[i+1]-x[i];
    for(i32 i=1;i<n-1;++i)alpha[i]=3*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1]);
    for(i32 i=1;i<n-1;++i){a[i]=h[i-1];b[i]=2*(h[i-1]+h[i]);c[i]=h[i];d[i]=alpha[i];}
    thomas(n,a.data(),b.data(),c.data(),d.data(),m.data());
    for(i32 i=0;i<n-1;++i){
        segs[i].a=y[i];
        segs[i].b=(y[i+1]-y[i])/h[i]-h[i]*(2*m[i]+m[i+1])/3;
        segs[i].c=m[i];
        segs[i].d=(m[i+1]-m[i])/(3*h[i]);
    }
}
f64 NaturalSpline::eval(f64 x) const noexcept {
    if(xs.empty())return 0;
    i32 lo=0,hi=(i32)xs.size()-2;
    while(lo<hi){i32 mid=(lo+hi+1)/2;if(xs[mid]<=x)lo=mid;else hi=mid-1;}
    return segs[lo].eval(x-xs[lo]);
}
f64 NaturalSpline::deriv(f64 x) const noexcept {
    if(xs.empty())return 0;
    i32 lo=0,hi=(i32)xs.size()-2;
    while(lo<hi){i32 mid=(lo+hi+1)/2;if(xs[mid]<=x)lo=mid;else hi=mid-1;}
    return segs[lo].deriv(x-xs[lo]);
}
f64 NaturalSpline::integrate(f64 a,f64 b) const noexcept {
    if(xs.empty()||a>=b)return 0;
    i32 n=(i32)segs.size(); f64 sum=0;
    for(i32 i=0;i<n;++i){
        const f64 x0=xs[i],x1=xs[i+1];
        const f64 lo2=std::max(a,x0),hi2=std::min(b,x1);
        if(lo2>=hi2)continue;
        // Integral of (A+B*t+C*t^2+D*t^3) from lo2-x0 to hi2-x0
        const f64 t0=lo2-x0,t1=hi2-x0;
        auto F=[&](f64 t){return segs[i].a*t+segs[i].b*t*t/2+segs[i].c*t*t*t/3+segs[i].d*t*t*t*t/4;};
        sum+=F(t1)-F(t0);
    }
    return sum;
}
void ClampedSpline::fit(const f64* x,const f64* y,i32 n,f64 d0,f64 dn) noexcept {
    NaturalSpline::fit(x,y,n);
    // Override boundary slopes
    if(!segs.empty()){
        segs[0].b=d0;
        segs[n-2].b=dn;
    }
}
void PeriodicSpline::fit(const f64* x,const f64* y,i32 n) noexcept {
    NaturalSpline::fit(x,y,n); // simplified: just use natural for now
}
// ─── Parametric 3D spline ────────────────────────────────────────────────────
void Spline3D::fit(const Vec3<f64>* pts,i32 n) noexcept {
    ts.resize(n); ts[0]=0;
    for(i32 i=1;i<n;++i)ts[i]=ts[i-1]+(pts[i]-pts[i-1]).length();
    const f64 total=ts.back();
    if(total>1e-10)for(auto&t:ts)t/=total;
    std::vector<f64> xv(n),yv(n),zv(n);
    for(i32 i=0;i<n;++i){xv[i]=pts[i].x;yv[i]=pts[i].y;zv[i]=pts[i].z;}
    sx.fit(ts.data(),xv.data(),n);
    sy.fit(ts.data(),yv.data(),n);
    sz.fit(ts.data(),zv.data(),n);
}
Vec3<f64> Spline3D::eval(f64 t) const noexcept {
    return{sx.eval(t),sy.eval(t),sz.eval(t)};
}
Vec3<f64> Spline3D::tangent(f64 t) const noexcept {
    Vec3<f64> d{sx.deriv(t),sy.deriv(t),sz.deriv(t)};
    const f64 l=d.length(); return l>1e-10?d/l:Vec3<f64>{0,1,0};
}
Vec3<f64> Spline3D::normal(f64 t) const noexcept {
    const f64 h=1e-5;
    Vec3<f64> t1=tangent(t),t2=tangent(t+h);
    Vec3<f64> n=t2-t1; const f64 l=n.length();
    return l>1e-10?n/l:Vec3<f64>{0,0,1};
}
f64 Spline3D::arc_length(f64 t0,f64 t1,i32 steps) const noexcept {
    const f64 h=(t1-t0)/steps; f64 sum=0;
    Vec3<f64> prev=eval(t0);
    for(i32 i=1;i<=steps;++i){Vec3<f64> cur=eval(t0+i*h);sum+=(cur-prev).length();prev=cur;}
    return sum;
}
// ─── NURBS Curve ─────────────────────────────────────────────────────────────
static i32 nurbs_span(const std::vector<f64>& knots,i32 n,i32 p,f64 t) noexcept {
    if(t>=knots[n+1])return n;
    i32 lo=p,hi=n+1,mid=(lo+hi)/2;
    while(t<knots[mid]||t>=knots[mid+1]){if(t<knots[mid])hi=mid;else lo=mid;mid=(lo+hi)/2;}
    return mid;
}
static void nurbs_basis(const std::vector<f64>& k,i32 i,i32 p,f64 t,std::vector<f64>& N) noexcept {
    N.assign(p+1,0); N[0]=1;
    std::vector<f64> left(p+1),right(p+1);
    for(i32 j=1;j<=p;++j){
        left[j]=t-k[i+1-j]; right[j]=k[i+j]-t; f64 saved=0;
        for(i32 r=0;r<j;++r){
            const f64 denom=right[r+1]+left[j-r];
            if(std::abs(denom)<1e-14){N[r]=saved;saved=0;continue;}
            const f64 tmp=N[r]/denom;
            N[r]=saved+right[r+1]*tmp; saved=left[j-r]*tmp;
        }
        N[j]=saved;
    }
}
Vec3<f64> NURBSCurve::eval(f64 t) const noexcept {
    i32 n=(i32)ctrl.size()-1;
    const i32 span=nurbs_span(knots,n,degree,t);
    std::vector<f64> N;
    nurbs_basis(knots,span,degree,t,N);
    Vec4<f64> wpt{};
    for(i32 j=0;j<=degree;++j){
        const auto& c=ctrl[span-degree+j];
        wpt.x+=N[j]*c.x; wpt.y+=N[j]*c.y; wpt.z+=N[j]*c.z; wpt.w+=N[j]*c.w;
    }
    if(std::abs(wpt.w)<1e-14)return{wpt.x,wpt.y,wpt.z};
    return{wpt.x/wpt.w,wpt.y/wpt.w,wpt.z/wpt.w};
}
Vec3<f64> NURBSCurve::tangent(f64 t,f64 h) const noexcept {
    Vec3<f64> d=eval(std::min(1.0,t+h))-eval(std::max(0.0,t-h));
    const f64 l=d.length(); return l>1e-10?d/l:Vec3<f64>{0,1,0};
}
f64 NURBSCurve::arc_length(i32 steps) const noexcept {
    const f64 h=1.0/steps; f64 sum=0;
    Vec3<f64> prev=eval(0);
    for(i32 i=1;i<=steps;++i){Vec3<f64> cur=eval(i*h);sum+=(cur-prev).length();prev=cur;}
    return sum;
}
f64 NURBSCurve::t_at_arc_length(f64 s,i32 steps) const noexcept {
    const f64 total=arc_length(steps);
    if(total<1e-10)return 0;
    f64 target=s/total,accum=0;
    const f64 h=1.0/steps;
    Vec3<f64> prev=eval(0);
    for(i32 i=1;i<=steps;++i){
        Vec3<f64> cur=eval(i*h);
        accum+=(cur-prev).length()/total;
        if(accum>=target)return (i-1)*h+(target-(accum-(cur-prev).length()/total))/(accum-(accum-(cur-prev).length()/total))*h;
        prev=cur;
    }
    return 1.0;
}
NaturalSpline fit_spline_to_fn(std::function<f64(f64)> f,f64 a,f64 b,i32 n) noexcept {
    std::vector<f64> xs(n),ys(n);
    for(i32 i=0;i<n;++i){xs[i]=a+(b-a)*i/(n-1);ys[i]=f(xs[i]);}
    NaturalSpline s; s.fit(xs.data(),ys.data(),n); return s;
}
void bezier_to_bspline(const Vec3<f64>* bz,i32 ns,
    std::vector<Vec3<f64>>& ctrl,std::vector<f64>& knots) noexcept
{
    // Each cubic bezier segment: 4 ctrl pts. C2 join at knot.
    ctrl.clear(); knots.clear();
    for(i32 s=0;s<ns;++s){
        ctrl.push_back(bz[s*3+0]);
        ctrl.push_back(bz[s*3+1]);
        ctrl.push_back(bz[s*3+2]);
        if(s==ns-1)ctrl.push_back(bz[s*3+3]);
    }
    const i32 n=(i32)ctrl.size()-1;
    // Clamped cubic knot vector
    knots.assign(n+5,0);
    for(i32 i=0;i<=n-2;++i)knots[4+i]=(f64)(i+1)/(n-1);
    for(i32 i=n+1;i<=n+4;++i)knots[i]=1.0;
}
} // namespace qs::math::spline
