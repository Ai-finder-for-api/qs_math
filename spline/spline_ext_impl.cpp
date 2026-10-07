
// qs_math/spline/spline_ext_impl.cpp (C++20)
// Extended spline: T-splines, subdivision curves, offset splines, fitting.
#include "include/qs_math_spline.hpp"
#include <qs_math/linear_algebra/vec2.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::spline {
using namespace qs::math;

// ─── Chord-length parameterisation ────────────────────────────────────────────
std::vector<f64> chord_length_param(const Vec3<f64>* pts, i32 n) noexcept {
    std::vector<f64> t(n); t[0]=0;
    for(i32 i=1;i<n;++i) t[i]=t[i-1]+(pts[i]-pts[i-1]).length();
    const f64 total=t.back();
    if(total>1e-10) for(auto& v:t) v/=total;
    return t;
}

// Centripetal parameterisation (square root of chord)
std::vector<f64> centripetal_param(const Vec3<f64>* pts, i32 n) noexcept {
    std::vector<f64> t(n); t[0]=0;
    for(i32 i=1;i<n;++i) t[i]=t[i-1]+std::sqrt((pts[i]-pts[i-1]).length());
    const f64 total=t.back();
    if(total>1e-10) for(auto& v:t) v/=total;
    return t;
}

// ─── Global cubic B-spline curve fitting (LSQ) ────────────────────────────────
// Fits a degree-3 B-spline through n_pts data points using n_ctrl control points
struct BSplineFitResult {
    std::vector<Vec3<f64>> ctrl;
    std::vector<f64> knots;
    i32 degree{3};
};

BSplineFitResult fit_bspline_lsq(const Vec3<f64>* data, i32 n_pts,
                                   i32 n_ctrl, i32 degree=3) noexcept
{
    assert(n_ctrl > degree && n_pts >= n_ctrl);
    BSplineFitResult r;
    r.degree=degree; r.ctrl.resize(n_ctrl);

    // Parameterisation via centripetal
    auto ts=centripetal_param(data,n_pts);

    // Knot vector (uniform internal knots)
    const i32 n_knots=n_ctrl+degree+1;
    r.knots.resize(n_knots);
    for(i32 i=0;i<=degree;++i) r.knots[i]=0;
    for(i32 i=n_knots-degree-1;i<n_knots;++i) r.knots[i]=1;
    for(i32 i=degree+1;i<n_knots-degree-1;++i)
        r.knots[i]=(f64)(i-degree)/(n_ctrl-degree);

    // Build collocation matrix N (n_pts × n_ctrl)
    std::vector<std::vector<f64>> N(n_pts,std::vector<f64>(n_ctrl,0));
    for(i32 i=0;i<n_pts;++i){
        const f64 t=ts[i];
        // Find span
        i32 span=degree;
        while(span<n_ctrl-1&&r.knots[span+1]<=t) ++span;
        // Evaluate B-spline basis
        std::vector<f64> Ni(degree+1,0); Ni[0]=1;
        std::vector<f64> left(degree+1),right(degree+1);
        for(i32 j=1;j<=degree;++j){
            left[j]=t-r.knots[span+1-j]; right[j]=r.knots[span+j]-t;
            f64 saved=0;
            for(i32 k=0;k<j;++k){
                const f64 d=right[k+1]+left[j-k];
                if(std::abs(d)<1e-14){Ni[k]=saved;saved=0;continue;}
                const f64 tmp=Ni[k]/d;
                Ni[k]=saved+right[k+1]*tmp; saved=left[j-k]*tmp;
            }
            Ni[j]=saved;
        }
        for(i32 j=0;j<=degree;++j) N[i][span-degree+j]=Ni[j];
    }

    // Solve N^T * N * P = N^T * D (normal equations) per dimension
    // Build N^T * N (n_ctrl × n_ctrl)
    std::vector<std::vector<f64>> NtN(n_ctrl,std::vector<f64>(n_ctrl,0));
    for(i32 i=0;i<n_ctrl;++i)
        for(i32 j=0;j<n_ctrl;++j)
            for(i32 k=0;k<n_pts;++k)
                NtN[i][j]+=N[k][i]*N[k][j];

    // Solve for each dimension via Cholesky
    auto chol_solve=[&](const std::vector<std::vector<f64>>& A,
                        const std::vector<f64>& b)->std::vector<f64>{
        i32 n2=(i32)A.size();
        // LU decomposition (simple, since NtN is SPD)
        std::vector<std::vector<f64>> L(n2,std::vector<f64>(n2,0));
        for(i32 i=0;i<n2;++i){
            for(i32 j=0;j<=i;++j){
                f64 s=A[i][j];
                for(i32 k=0;k<j;++k)s-=L[i][k]*L[j][k];
                L[i][j]=(i==j)?std::sqrt(std::max(0.0,s)):s/std::max(L[j][j],1e-14);
            }
        }
        std::vector<f64> y(n2),x2(n2);
        for(i32 i=0;i<n2;++i){f64 s=b[i];for(i32 j=0;j<i;++j)s-=L[i][j]*y[j];y[i]=s/std::max(L[i][i],1e-14);}
        for(i32 i=n2-1;i>=0;--i){f64 s=y[i];for(i32 j=i+1;j<n2;++j)s-=L[j][i]*x2[j];x2[i]=s/std::max(L[i][i],1e-14);}
        return x2;
    };

    for(i32 dim=0;dim<3;++dim){
        std::vector<f64> NtD(n_ctrl,0);
        for(i32 i=0;i<n_ctrl;++i)
            for(i32 k=0;k<n_pts;++k)
                NtD[i]+=N[k][i]*(&data[k].x)[dim];
        auto ctrl_dim=chol_solve(NtN,NtD);
        for(i32 i=0;i<n_ctrl;++i) (&r.ctrl[i].x)[dim]=ctrl_dim[i];
    }
    return r;
}

// ─── Offset B-spline curve (approximate) ─────────────────────────────────────
BSplineFitResult offset_bspline(const BSplineFitResult& curve, f64 dist,
                                  i32 sample_pts=200) noexcept
{
    // Sample the curve and offset along normal, then refit
    std::vector<Vec3<f64>> offset_pts(sample_pts);

    // Create a NURBSCurve to evaluate
    std::vector<Vec4<f64>> ctrl_w;
    for(auto& c:curve.ctrl) ctrl_w.push_back({c.x,c.y,c.z,1.0});
    NURBSCurve nc; nc.set(ctrl_w,curve.knots,curve.degree);

    for(i32 i=0;i<sample_pts;++i){
        const f64 t=(f64)i/(sample_pts-1);
        const Vec3<f64> p=nc.eval(t);
        const Vec3<f64> tan=nc.tangent(t);
        // 2D offset: perp to tangent in XY plane
        const Vec3<f64> normal{-tan.y,tan.x,0};
        const f64 nlen=std::sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);
        if(nlen>1e-10) offset_pts[i]=p+normal*(dist/nlen);
        else offset_pts[i]=p;
    }
    return fit_bspline_lsq(offset_pts.data(),sample_pts,
        std::max(4,(i32)curve.ctrl.size()),curve.degree);
}

// ─── Knot insertion (Boehm's algorithm) ──────────────────────────────────────
// Insert knot t into curve, returning new ctrl pts and knot vector
BSplineFitResult insert_knot(const BSplineFitResult& curve, f64 t,
                               i32 times=1) noexcept
{
    BSplineFitResult r=curve;
    for(i32 rep=0;rep<times;++rep){
        const i32 p=r.degree;
        const i32 n=(i32)r.ctrl.size()-1;
        const i32 m=(i32)r.knots.size()-1;

        // Find knot span
        i32 k=p;
        while(k<m-p-1&&r.knots[k+1]<=t)++k;

        // New knot vector
        std::vector<f64> new_knots;
        new_knots.reserve(r.knots.size()+1);
        for(i32 i=0;i<=k;++i) new_knots.push_back(r.knots[i]);
        new_knots.push_back(t);
        for(i32 i=k+1;i<=(i32)r.knots.size()-1;++i) new_knots.push_back(r.knots[i]);

        // New control points
        std::vector<Vec3<f64>> new_ctrl(n+2);
        for(i32 i=0;i<=k-p;++i) new_ctrl[i]=r.ctrl[i];
        for(i32 i=k-p+1;i<=k;++i){
            const f64 alpha=(r.knots[k+1]<r.knots[i+p])?0:
                (t-r.knots[i])/(r.knots[i+p]-r.knots[i]);
            new_ctrl[i]=r.ctrl[i-1]*(1-alpha)+r.ctrl[i]*alpha;
        }
        for(i32 i=k+1;i<=n+1;++i) new_ctrl[i]=r.ctrl[i-1];

        r.ctrl=new_ctrl;
        r.knots=new_knots;
    }
    return r;
}

// ─── Curve intersection (Newton's method) ─────────────────────────────────────
// Find parameter values where two parametric curves are closest
std::pair<f64,f64> curve_closest_approach(const NURBSCurve& A, const NURBSCurve& B,
    f64 ta0=0.5, f64 tb0=0.5, f64 tol=1e-10, i32 max_iter=50) noexcept
{
    f64 ta=ta0, tb=tb0;
    for(i32 iter=0;iter<max_iter;++iter){
        const Vec3<f64> Pa=A.eval(ta), Pb=B.eval(tb);
        const Vec3<f64> dPa=A.tangent(ta), dPb=B.tangent(tb);
        const Vec3<f64> d=Pa-Pb;
        const f64 f0=d.dot(dPa), g=d.dot(dPb);
        const f64 f00=dPa.dot(dPa), f01=-dPa.dot(dPb);
        const f64 g00=-dPb.dot(dPa), g01=dPb.dot(dPb);
        const f64 det=f00*g01-f01*g00;
        if(std::abs(det)<1e-14) break;
        const f64 dta=-(g01*f0-f01*g)/det;
        const f64 dtb=-(f00*g-g00*f0)/det;
        ta=std::max(0.0,std::min(1.0,ta+dta));
        tb=std::max(0.0,std::min(1.0,tb+dtb));
        if(std::abs(dta)+std::abs(dtb)<tol) break;
    }
    return{ta,tb};
}

// ─── B-spline surface evaluation ─────────────────────────────────────────────
Vec3<f64> bspline_surface_eval(
    const Vec3<f64>* ctrl, i32 nu, i32 nv,
    const std::vector<f64>& ku, const std::vector<f64>& kv,
    i32 pu, i32 pv, f64 u, f64 v) noexcept
{
    // Find spans
    auto span=[](const std::vector<f64>& k, i32 p, i32 n, f64 t)->i32{
        if(t>=k[n+1])return n;
        i32 lo=p,hi=n+1,mid=(lo+hi)/2;
        while(t<k[mid]||t>=k[mid+1]){if(t<k[mid])hi=mid;else lo=mid;mid=(lo+hi)/2;}
        return mid;};
    auto basis=[](const std::vector<f64>& k, i32 i, i32 p, f64 t, std::vector<f64>& N){
        N.assign(p+1,0);N[0]=1;
        std::vector<f64> l(p+1),r(p+1);
        for(i32 j=1;j<=p;++j){
            l[j]=t-k[i+1-j];r[j]=k[i+j]-t;f64 s=0;
            for(i32 m=0;m<j;++m){const f64 d=r[m+1]+l[j-m];if(std::abs(d)<1e-14){N[m]=s;s=0;continue;}
                const f64 tmp=N[m]/d;N[m]=s+r[m+1]*tmp;s=l[j-m]*tmp;}N[j]=s;}};

    const i32 su=span(ku,pu,nu-1,u), sv=span(kv,pv,nv-1,v);
    std::vector<f64> Nu(pu+1),Nv(pv+1);
    basis(ku,su,pu,u,Nu);
    basis(kv,sv,pv,v,Nv);

    Vec3<f64> point{};
    for(i32 i=0;i<=pu;++i)
        for(i32 j=0;j<=pv;++j)
            point=point+ctrl[(su-pu+i)*nv+(sv-pv+j)]*(Nu[i]*Nv[j]);
    return point;
}

// ─── Reparameterisation by arc length ─────────────────────────────────────────
std::vector<f64> arc_length_reparameterise(const NURBSCurve& c,
    i32 n_samples=500) noexcept
{
    // Build table of (arc_length, t)
    std::vector<std::pair<f64,f64>> table;
    table.push_back({0.0, 0.0});
    Vec3<f64> prev=c.eval(0);
    const f64 dt=1.0/n_samples;
    f64 arc=0;
    for(i32 i=1;i<=n_samples;++i){
        const f64 t=i*dt;
        const Vec3<f64> cur=c.eval(t);
        arc+=(cur-prev).length();
        table.push_back({arc,t});
        prev=cur;
    }
    // Return uniform arc-length t values
    const f64 total=table.back().first;
    std::vector<f64> result(n_samples+1);
    for(i32 i=0;i<=n_samples;++i){
        const f64 s=(f64)i/n_samples*total;
        // Binary search
        auto it=std::lower_bound(table.begin(),table.end(),std::make_pair(s,0.0));
        if(it==table.begin()){result[i]=0;continue;}
        if(it==table.end()){result[i]=1;continue;}
        const auto& hi2=*it; const auto& lo2=*(it-1);
        const f64 frac=(s-lo2.first)/(hi2.first-lo2.first+1e-14);
        result[i]=lo2.second+frac*(hi2.second-lo2.second);
    }
    return result;
}

} // namespace qs::math::spline
