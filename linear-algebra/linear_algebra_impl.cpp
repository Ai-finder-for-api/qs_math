
// qs_math/linear-algebra/linear_algebra_impl.cpp (C++20)
#include "include/qs_math_linear_algebra.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
namespace qs::math::linalg {

DMatrix DMatrix::identity(i32 n) noexcept {
    DMatrix r(n,n); for(i32 i=0;i<n;++i) r.at(i,i)=1; return r;
}
DMatrix DMatrix::operator*(const DMatrix& o) const noexcept {
    assert(n==o.m);
    DMatrix r(m,o.n);
    for(i32 i=0;i<m;++i) for(i32 k=0;k<n;++k) if(at(i,k)!=0)
        for(i32 j=0;j<o.n;++j) r.at(i,j)+=at(i,k)*o.at(k,j);
    return r;
}
DMatrix DMatrix::transposed() const noexcept {
    DMatrix r(n,m); for(i32 i=0;i<m;++i) for(i32 j=0;j<n;++j) r.at(j,i)=at(i,j); return r;
}
f64 DMatrix::frobenius_norm() const noexcept {
    f64 s=0; for(auto v:data) s+=v*v; return std::sqrt(s);
}
f64 DMatrix::trace() const noexcept {
    f64 s=0; i32 k=std::min(m,n); for(i32 i=0;i<k;++i)s+=at(i,i); return s;
}
// Householder reflection in-place on column [lo..hi] of A
static void householder_col(DMatrix& A,i32 col,i32 lo) {
    i32 m=A.m;
    f64 norm=0; for(i32 i=lo;i<m;++i) norm+=A.at(i,col)*A.at(i,col); norm=std::sqrt(norm);
    if(norm<1e-14) return;
    A.at(lo,col)+=A.at(lo,col)>=0?norm:-norm;
    const f64 inv=1.0/(norm*std::abs(A.at(lo,col)));
    for(i32 j=col;j<A.n;++j){
        f64 dot=0; for(i32 i=lo;i<m;++i) dot+=A.at(i,col)*A.at(i,j);
        dot*=inv;
        for(i32 i=lo;i<m;++i) A.at(i,j)-=A.at(i,col)*dot;
    }
}
SVDResult svd(DMatrix A) noexcept {
    // One-sided Jacobi SVD for small matrices
    i32 m=A.m,n=A.n;
    DMatrix U=DMatrix::identity(m),Vt=DMatrix::identity(n);
    // Bidiagonalization then QR: simplified here to 1-sided Jacobi
    for(i32 sweep=0;sweep<30;++sweep){
        f64 off=0;
        for(i32 p=0;p<n-1;++p) for(i32 q=p+1;q<n;++q){
            f64 app=0,apq=0,aqq=0;
            for(i32 i=0;i<m;++i){app+=A.at(i,p)*A.at(i,p);apq+=A.at(i,p)*A.at(i,q);aqq+=A.at(i,q)*A.at(i,q);}
            off+=2*apq*apq;
            if(std::abs(apq)<1e-14) continue;
            const f64 tau=(aqq-app)/(2*apq);
            const f64 t=tau>=0?1.0/(tau+std::sqrt(1+tau*tau)):1.0/(tau-std::sqrt(1+tau*tau));
            const f64 c=1.0/std::sqrt(1+t*t),s=c*t;
            for(i32 i=0;i<m;++i){f64 ai=A.at(i,p),aj=A.at(i,q);A.at(i,p)=c*ai+s*aj;A.at(i,q)=-s*ai+c*aj;}
            for(i32 i=0;i<n;++i){f64 vi=Vt.at(p,i),vj=Vt.at(q,i);Vt.at(p,i)=c*vi+s*vj;Vt.at(q,i)=-s*vi+c*vj;}
        }
        if(off<1e-20) break;
    }
    DMatrix S(std::min(m,n),std::min(m,n));
    for(i32 j=0;j<std::min(m,n);++j){
        f64 nrm=0; for(i32 i=0;i<m;++i) nrm+=A.at(i,j)*A.at(i,j); nrm=std::sqrt(nrm);
        S.at(j,j)=nrm;
        if(nrm>1e-14) for(i32 i=0;i<m;++i) A.at(i,j)/=nrm;
    }
    return{A,S,Vt};
}
bool cholesky(const DMatrix& A, DMatrix& L) noexcept {
    i32 n=A.n; L=DMatrix(n,n);
    for(i32 i=0;i<n;++i){
        for(i32 j=0;j<=i;++j){
            f64 s=A.at(i,j);
            for(i32 k=0;k<j;++k) s-=L.at(i,k)*L.at(j,k);
            if(i==j){ if(s<0)return false; L.at(i,j)=std::sqrt(s);}
            else L.at(i,j)=s/L.at(j,j);
        }
    }
    return true;
}
void cholesky_solve(const DMatrix& L,const f64* b,f64* x,i32 n) noexcept {
    std::vector<f64> y(n);
    for(i32 i=0;i<n;++i){f64 s=b[i];for(i32 j=0;j<i;++j)s-=L.at(i,j)*y[j];y[i]=s/L.at(i,i);}
    for(i32 i=n-1;i>=0;--i){f64 s=y[i];for(i32 j=i+1;j<n;++j)s-=L.at(j,i)*x[j];x[i]=s/L.at(i,i);}
}
void jacobi_iter(const DMatrix& A,const f64* b,f64* x,i32 max_iter,f64 tol) noexcept {
    i32 n=A.n; std::vector<f64> xn(n,0);
    std::fill(x,x+n,0.0);
    for(i32 it=0;it<max_iter;++it){
        f64 err=0;
        for(i32 i=0;i<n;++i){
            f64 s=b[i];
            for(i32 j=0;j<n;++j)if(j!=i)s-=A.at(i,j)*x[j];
            xn[i]=std::abs(A.at(i,i))<1e-14?0:s/A.at(i,i);
            err+=std::abs(xn[i]-x[i]);
        }
        std::copy(xn.begin(),xn.end(),x);
        if(err<tol) break;
    }
}
void gauss_seidel(const DMatrix& A,const f64* b,f64* x,i32 max_iter,f64 tol) noexcept {
    i32 n=A.n; std::fill(x,x+n,0.0);
    for(i32 it=0;it<max_iter;++it){
        f64 err=0;
        for(i32 i=0;i<n;++i){
            f64 s=b[i];
            for(i32 j=0;j<n;++j)if(j!=i)s-=A.at(i,j)*x[j];
            const f64 xn=std::abs(A.at(i,i))<1e-14?0:s/A.at(i,i);
            err+=std::abs(xn-x[i]); x[i]=xn;
        }
        if(err<tol) break;
    }
}
DMatrix outer_product(const f64* u,const f64* v,i32 m,i32 n) noexcept {
    DMatrix r(m,n);
    for(i32 i=0;i<m;++i)for(i32 j=0;j<n;++j)r.at(i,j)=u[i]*v[j];
    return r;
}
DMatrix low_rank_approx(const DMatrix& A,i32 k) noexcept {
    auto[U,S,Vt]=svd(A);
    i32 rank=std::min({A.m,A.n,k});
    DMatrix r(A.m,A.n);
    for(i32 i=0;i<rank;++i){
        const f64 sv=S.at(i,i);
        for(i32 row=0;row<A.m;++row)
            for(i32 col=0;col<A.n;++col)
                r.at(row,col)+=U.at(row,i)*sv*Vt.at(i,col);
    }
    return r;
}
std::array<f64,3> eigenvalues_3x3_sym(const Mat3<f64>& A) noexcept {
    // Cardano's formula for symmetric 3x3
    const f64 a=A[0][0],b=A[1][1],c=A[2][2];
    const f64 d=A[0][1],e=A[1][2],f=A[0][2];
    const f64 p1=d*d+e*e+f*f;
    if(p1<1e-14) return{{a,b,c}};
    const f64 q=(a+b+c)/3;
    const f64 p2=(a-q)*(a-q)+(b-q)*(b-q)+(c-q)*(c-q)+2*p1;
    const f64 p=std::sqrt(p2/6);
    const f64 B00=(a-q)/p,B11=(b-q)/p,B22=(c-q)/p,B01=d/p,B12=e/p,B02=f/p;
    const f64 r=B00*(B11*B22-B12*B12)-B01*(B01*B22-B12*B02)+B02*(B01*B12-B11*B02);
    const f64 phi=r<=−1?M_PI/3:(r>=1?0:std::acos(r)/3);
    return{{q+2*p*std::cos(phi),q+2*p*std::cos(phi+2*M_PI/3),q+2*p*std::cos(phi+4*M_PI/3)}};
}
EigenResult eigen_symmetric(DMatrix A,i32 max_iter) noexcept {
    i32 n=A.n;
    DMatrix V=DMatrix::identity(n);
    for(i32 it=0;it<max_iter;++it){
        // Find off-diagonal max
        i32 p=0,q=1; f64 mx=std::abs(A.at(0,1));
        for(i32 i=0;i<n-1;++i)for(i32 j=i+1;j<n;++j)
            if(std::abs(A.at(i,j))>mx){mx=std::abs(A.at(i,j));p=i;q=j;}
        if(mx<1e-12) break;
        const f64 tau=(A.at(q,q)-A.at(p,p))/(2*A.at(p,q));
        const f64 t=tau>=0?1.0/(tau+std::sqrt(1+tau*tau)):1.0/(tau-std::sqrt(1+tau*tau));
        const f64 c=1.0/std::sqrt(1+t*t),s=c*t;
        for(i32 r=0;r<n;++r){
            const f64 ar=A.at(r,p),aq=A.at(r,q);
            A.at(r,p)=c*ar-s*aq; A.at(r,q)=s*ar+c*aq;
        }
        for(i32 r=0;r<n;++r){
            const f64 ap=A.at(p,r),aq=A.at(q,r);
            A.at(p,r)=c*ap-s*aq; A.at(q,r)=s*ap+c*aq;
        }
        for(i32 r=0;r<n;++r){
            const f64 vp=V.at(r,p),vq=V.at(r,q);
            V.at(r,p)=c*vp-s*vq; V.at(r,q)=s*vp+c*vq;
        }
    }
    std::vector<f64> vals(n); for(i32 i=0;i<n;++i) vals[i]=A.at(i,i);
    return{vals,V};
}
} // namespace qs::math::linalg
