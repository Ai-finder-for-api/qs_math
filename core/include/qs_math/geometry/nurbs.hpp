#pragma once
// qs_math/geometry/nurbs.hpp — B-spline basis, NURBS curves/surfaces, knot insertion, evaluation
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── B-spline basis functions (Cox-de Boor recursion) ─────────────────────────
// N_{i,p}(u): degree-p basis function for knot span i
template<int NKNOTS>
[[nodiscard]] constexpr f64 bspline_basis(int i, int p,
    f64 u, const std::array<f64,NKNOTS>& knots) noexcept
{
    if(p==0) return (u>=knots[i]&&u<knots[i+1])?1.0:0.0;
    f64 left=0,right=0;
    f64 d1=knots[i+p]-knots[i];
    f64 d2=knots[i+p+1]-knots[i+1];
    if(d1>1e-15) left=(u-knots[i])/d1*bspline_basis<NKNOTS>(i,p-1,u,knots);
    if(d2>1e-15) right=(knots[i+p+1]-u)/d2*bspline_basis<NKNOTS>(i+1,p-1,u,knots);
    return left+right;
}

// Find knot span index (binary search)
template<int NKNOTS>
[[nodiscard]] constexpr int find_knot_span(int n, int p, f64 u,
    const std::array<f64,NKNOTS>& knots) noexcept
{
    if(u>=knots[n+1]) return n;
    if(u<=knots[p]) return p;
    int lo=p,hi=n+1,mid=(lo+hi)/2;
    while(u<knots[mid]||u>=knots[mid+1]) {
        if(u<knots[mid]) hi=mid; else lo=mid;
        mid=(lo+hi)/2;
    }
    return mid;
}

// ─── Evaluate all nonzero basis functions at u (de Boor) ──────────────────────
template<int P>
[[nodiscard]] constexpr std::array<f64,P+1> bspline_basis_all(
    int span, int p, f64 u, const f64* knots) noexcept
{
    std::array<f64,P+1> N{};
    std::array<f64,P+1> left{},right{};
    N[0]=1;
    for(int j=1;j<=p;++j) {
        left[j]=u-knots[span+1-j];
        right[j]=knots[span+j]-u;
        f64 saved=0;
        for(int r=0;r<j;++r) {
            f64 denom=right[r+1]+left[j-r];
            f64 tmp=(denom>1e-15)?N[r]/denom:0;
            N[r]=saved+right[r+1]*tmp;
            saved=left[j-r]*tmp;
        }
        N[j]=saved;
    }
    return N;
}

// ─── NURBS curve point evaluation ─────────────────────────────────────────────
template<int N, int P, int DIM>
struct NURBSCurve {
    std::array<std::array<f64,DIM>,N> ctrl{};  // control points
    std::array<f64,N>                 w{};      // weights
    std::array<f64,N+P+1>            knots{};  // knot vector

    // Evaluate curve at parameter u via de Boor algorithm
    [[nodiscard]] constexpr std::array<f64,DIM> eval(f64 u) const noexcept {
        // Clamp u to [knots[P], knots[N]]
        u=u<knots[P]?knots[P]:(u>knots[N]?knots[N]:u);
        int span=0;
        for(int i=P;i<N;++i) if(u>=knots[i]&&u<knots[i+1]){span=i;break;}
        if(u>=knots[N]) span=N-1;

        // Get basis functions
        auto Nf=bspline_basis_all<P>(span,P,u,knots.data());

        // Weighted sum of control points
        std::array<f64,DIM> num{}; f64 den=0;
        for(int j=0;j<=P;++j) {
            int idx=span-P+j;
            f64 Nw=Nf[j]*w[idx];
            den+=Nw;
            for(int d=0;d<DIM;++d) num[d]+=Nw*ctrl[idx][d];
        }
        std::array<f64,DIM> pt{};
        if(den>1e-15) for(int d=0;d<DIM;++d) pt[d]=num[d]/den;
        return pt;
    }

    // Sample curve at evenly spaced parameters
    template<int SAMPLES>
    [[nodiscard]] constexpr std::array<std::array<f64,DIM>,SAMPLES> sample() const noexcept {
        std::array<std::array<f64,DIM>,SAMPLES> pts{};
        f64 u0=knots[P],u1=knots[N];
        for(int i=0;i<SAMPLES;++i) pts[i]=eval(u0+f64(i)/(SAMPLES-1)*(u1-u0));
        return pts;
    }

    // Arc length approximation
    [[nodiscard]] constexpr f64 arc_length(int segs=100) const noexcept {
        f64 u0=knots[P],u1=knots[N],len=0;
        auto prev=eval(u0);
        for(int i=1;i<=segs;++i) {
            auto cur=eval(u0+f64(i)/segs*(u1-u0));
            f64 d=0; for(int j=0;j<DIM;++j) d+=(cur[j]-prev[j])*(cur[j]-prev[j]);
            len+=std::sqrt(d); prev=cur;
        }
        return len;
    }
};

// ─── NURBS surface ────────────────────────────────────────────────────────────
template<int NU, int NV, int PU, int PV, int DIM>
struct NURBSSurface {
    std::array<std::array<std::array<f64,DIM>,NV>,NU> ctrl{};
    std::array<std::array<f64,NV>,NU>                  w{};
    std::array<f64,NU+PU+1>                            ku{};
    std::array<f64,NV+PV+1>                            kv{};

    [[nodiscard]] constexpr std::array<f64,DIM> eval(f64 u, f64 v) const noexcept {
        // Clamp
        u=u<ku[PU]?ku[PU]:(u>ku[NU]?ku[NU]:u);
        v=v<kv[PV]?kv[PV]:(v>kv[NV]?kv[NV]:v);
        // Find spans
        int su=PU; for(int i=PU;i<NU;++i) if(u>=ku[i]&&u<ku[i+1]){su=i;break;}
        int sv=PV; for(int i=PV;i<NV;++i) if(v>=kv[i]&&v<kv[i+1]){sv=i;break;}
        auto Nu=bspline_basis_all<PU>(su,PU,u,ku.data());
        auto Nv=bspline_basis_all<PV>(sv,PV,v,kv.data());
        std::array<f64,DIM> num{}; f64 den=0;
        for(int i=0;i<=PU;++i) for(int j=0;j<=PV;++j) {
            int ui=su-PU+i, vi=sv-PV+j;
            f64 Nw=Nu[i]*Nv[j]*w[ui][vi];
            den+=Nw;
            for(int d=0;d<DIM;++d) num[d]+=Nw*ctrl[ui][vi][d];
        }
        std::array<f64,DIM> pt{};
        if(den>1e-15) for(int d=0;d<DIM;++d) pt[d]=num[d]/den;
        return pt;
    }

    // Normal vector at (u,v) via cross product of partial derivatives
    [[nodiscard]] constexpr std::array<f64,3> normal(f64 u, f64 v, f64 h=1e-5) const noexcept {
        auto pu=eval(u+h,v), pm_u=eval(u-h,v);
        auto pv=eval(u,v+h), pm_v=eval(u,v-h);
        std::array<f64,3> du{(pu[0]-pm_u[0])/(2*h),(pu[1]-pm_u[1])/(2*h),(pu[2]-pm_u[2])/(2*h)};
        std::array<f64,3> dv{(pv[0]-pm_v[0])/(2*h),(pv[1]-pm_v[1])/(2*h),(pv[2]-pm_v[2])/(2*h)};
        std::array<f64,3> n{du[1]*dv[2]-du[2]*dv[1],du[2]*dv[0]-du[0]*dv[2],du[0]*dv[1]-du[1]*dv[0]};
        f64 len=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if(len>1e-12) for(auto& x:n) x/=len;
        return n;
    }
};

// ─── Knot insertion (Boehm's algorithm) ───────────────────────────────────────
// Insert knot t into B-spline of degree p, n+1 control points
template<int N, int P, int DIM>
[[nodiscard]] constexpr std::pair<std::array<std::array<f64,DIM>,N+1>,
                                   std::array<f64,N+P+2>>
bspline_insert_knot(
    const std::array<std::array<f64,DIM>,N>& ctrl,
    const std::array<f64,N+P+1>& knots, f64 t) noexcept
{
    // Find span k: knot[k] <= t < knot[k+1]
    int k=P;
    for(int i=P;i<N;++i) if(t>=knots[i]&&t<knots[i+1]){k=i;break;}

    // New knot vector
    std::array<f64,N+P+2> new_knots{};
    for(int i=0;i<=k;++i) new_knots[i]=knots[i];
    new_knots[k+1]=t;
    for(int i=k+1;i<=N+P;++i) new_knots[i+1]=knots[i];

    // New control points
    std::array<std::array<f64,DIM>,N+1> new_ctrl{};
    for(int i=0;i<=k-P;++i) new_ctrl[i]=ctrl[i];
    for(int i=k-P+1;i<=k;++i) {
        f64 denom=knots[i+P]-knots[i];
        f64 alpha=(denom>1e-15)?(t-knots[i])/denom:0;
        for(int d=0;d<DIM;++d) new_ctrl[i][d]=(1-alpha)*ctrl[i-1][d]+alpha*ctrl[i][d];
    }
    for(int i=k+1;i<=N;++i) new_ctrl[i]=ctrl[i-1];
    return {new_ctrl,new_knots};
}

// ─── Bezier curve (special case p=n) ──────────────────────────────────────────
// De Casteljau algorithm
template<int N, int DIM>
[[nodiscard]] constexpr std::array<f64,DIM> bezier_eval(
    const std::array<std::array<f64,DIM>,N>& pts, f64 t) noexcept
{
    std::array<std::array<f64,DIM>,N> p=pts;
    for(int k=1;k<N;++k)
        for(int i=0;i<N-k;++i)
            for(int d=0;d<DIM;++d) p[i][d]=(1-t)*p[i][d]+t*p[i+1][d];
    return p[0];
}

// Bezier subdivision (de Casteljau)
template<int N, int DIM>
[[nodiscard]] constexpr std::pair<std::array<std::array<f64,DIM>,N>,
                                   std::array<std::array<f64,DIM>,N>>
bezier_subdivide(const std::array<std::array<f64,DIM>,N>& pts, f64 t=0.5) noexcept
{
    std::array<std::array<std::array<f64,DIM>,N>,N> tri{};
    tri[0]=pts;
    for(int k=1;k<N;++k)
        for(int i=0;i<N-k;++i)
            for(int d=0;d<DIM;++d) tri[k][i][d]=(1-t)*tri[k-1][i][d]+t*tri[k-1][i+1][d];
    std::array<std::array<f64,DIM>,N> left{},right{};
    for(int k=0;k<N;++k) { left[k]=tri[k][0]; right[N-1-k]=tri[k][N-1-k]; }
    return {left,right};
}

// ─── Hermite to Bezier conversion ─────────────────────────────────────────────
template<int DIM>
[[nodiscard]] constexpr std::array<std::array<f64,DIM>,4> hermite_to_bezier(
    const std::array<f64,DIM>& p0, const std::array<f64,DIM>& p1,
    const std::array<f64,DIM>& m0, const std::array<f64,DIM>& m1) noexcept
{
    std::array<std::array<f64,DIM>,4> bez{};
    for(int d=0;d<DIM;++d) {
        bez[0][d]=p0[d];
        bez[1][d]=p0[d]+m0[d]/3;
        bez[2][d]=p1[d]-m1[d]/3;
        bez[3][d]=p1[d];
    }
    return bez;
}
} // namespace qs::math
