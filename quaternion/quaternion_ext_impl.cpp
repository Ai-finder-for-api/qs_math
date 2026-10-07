
// qs_math/quaternion/quaternion_ext_impl.cpp (C++20)
#include "include/qs_math_quaternion.hpp"
#include <cmath>
#include <algorithm>
namespace qs::math::quat_ext {
static Qf qmul(Qf a,Qf b){return{a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
    a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
    a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w};}
static Qf qconj(Qf q){return{q.w,-q.x,-q.y,-q.z};}
static f32 qdot(Qf a,Qf b){return a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;}
static Qf qnorm(Qf q){f32 l=std::sqrt(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z);if(l<1e-7f)return{1,0,0,0};return{q.w/l,q.x/l,q.y/l,q.z/l};}
static Qf qadd(Qf a,Qf b){return{a.w+b.w,a.x+b.x,a.y+b.y,a.z+b.z};}
static Qf qscale(Qf a,f32 s){return{a.w*s,a.x*s,a.y*s,a.z*s};}

Qf qlog(Qf q) noexcept {
    const f32 vn=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
    const f32 theta=vn<1e-7f?0.0f:std::atan2(vn,q.w)/vn;
    return{0,q.x*theta,q.y*theta,q.z*theta};
}
Qf qexp(Qf q) noexcept {
    const f32 vn=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
    const f32 s=vn<1e-7f?1.0f:std::sin(vn)/vn;
    return{std::cos(vn),q.x*s,q.y*s,q.z*s};
}
Qf qpow(Qf q,f32 t) noexcept { return qexp(qscale(qlog(q),t)); }

// ─── Dual Quaternion ──────────────────────────────────────────────────────────
DualQuat DualQuat::from_rt(Qf r,Vec3<f32> t) noexcept {
    DualQuat dq; dq.real=qnorm(r);
    Qf T{0,t.x*0.5f,t.y*0.5f,t.z*0.5f};
    dq.dual=qmul(T,dq.real);
    return dq;
}
Vec3<f32> DualQuat::transform(Vec3<f32> p) const noexcept {
    // Point as pure dual quaternion: [1+eps*p]
    // Result: dq * point * conj(dq)
    DualQuat pt{}; pt.real={1,0,0,0}; pt.dual={0,p.x,p.y,p.z};
    DualQuat res; res.real=qmul(real,pt.real);
    res.dual=qadd(qmul(real,pt.dual),qmul(dual,pt.real));
    DualQuat c; c.real=qconj(real); c.dual=qconj(dual);
    DualQuat fin; fin.real=qmul(res.real,c.real);
    fin.dual=qadd(qmul(res.real,c.dual),qmul(res.dual,c.real));
    return{fin.dual.x,fin.dual.y,fin.dual.z};
}
DualQuat DualQuat::operator*(const DualQuat& o) const noexcept {
    return{qmul(real,o.real), qadd(qmul(real,o.dual),qmul(dual,o.real))};
}
DualQuat DualQuat::conjugate() const noexcept { return{qconj(real),qconj(dual)}; }
void DualQuat::normalise() noexcept {
    const f32 len=qdot(real,real); if(len<1e-10f)return;
    const f32 inv=1.0f/std::sqrt(len);
    real=qscale(real,inv);
    const f32 dd=qdot(real,dual);
    dual=qscale(qadd(dual,qscale(real,-dd)),inv);
}
Mat4<f32> DualQuat::to_mat4() const noexcept {
    // Extract rotation matrix from real part
    const Qf& r=real;
    const f32 tx=2*(dual.x*r.w-dual.w*r.x+dual.y*r.z-dual.z*r.y);
    const f32 ty=2*(dual.y*r.w-dual.w*r.y+dual.z*r.x-dual.x*r.z);
    const f32 tz=2*(dual.z*r.w-dual.w*r.z+dual.x*r.y-dual.y*r.x);
    return Mat4<f32>::from_trs({tx,ty,tz},r,{1,1,1});
}
// ─── Weighted average ─────────────────────────────────────────────────────────
Qf weighted_average(const Qf* qs,const f32* ws,i32 n,i32 max_iter) noexcept {
    // Iterative geodesic mean on S3
    Qf mean=qnorm(qs[0]);
    for(i32 it=0;it<max_iter;++it){
        Qf delta{0,0,0,0};
        for(i32 i=0;i<n;++i){
            Qf qi=qs[i]; if(qdot(mean,qi)<0)qi=qscale(qi,-1);
            delta=qadd(delta,qscale(qlog(qmul(qconj(mean),qi)),ws[i]));
        }
        if(std::sqrt(delta.x*delta.x+delta.y*delta.y+delta.z*delta.z)<1e-7f)break;
        mean=qmul(mean,qexp(delta));
        mean=qnorm(mean);
    }
    return mean;
}
// ─── Rotation between vectors ─────────────────────────────────────────────────
Qf rotation_between(Vec3<f32> from,Vec3<f32> to) noexcept {
    const f32 fl=from.length(),tl=to.length();
    if(fl<1e-7f||tl<1e-7f)return{1,0,0,0};
    Vec3<f32> f=from/fl,t=to/tl;
    const f32 d=f.dot(t);
    if(d>0.9999f)return{1,0,0,0};
    if(d<-0.9999f){
        Vec3<f32> ax=Vec3<f32>{1,0,0}.cross(f);
        if(ax.length_sq()<1e-6f)ax=Vec3<f32>{0,1,0}.cross(f);
        ax=ax.normalized();
        return{0,ax.x,ax.y,ax.z};
    }
    Vec3<f32> ax=f.cross(t);
    return qnorm(Qf{1+d,ax.x,ax.y,ax.z});
}
// ─── Swing-twist decomposition ────────────────────────────────────────────────
void swing_twist(Qf q,Vec3<f32> axis,Qf& swing,Qf& twist) noexcept {
    Vec3<f32> ra{q.x,q.y,q.z};
    const f32 dot=ra.x*axis.x+ra.y*axis.y+ra.z*axis.z;
    Vec3<f32> proj=axis*dot;
    twist=qnorm(Qf{q.w,proj.x,proj.y,proj.z});
    swing=qmul(q,qconj(twist));
}
// ─── Angular velocity integration ────────────────────────────────────────────
Qf integrate_angular_vel(Qf q,Vec3<f32> omega,f32 dt) noexcept {
    const f32 ang=omega.length()*dt;
    if(ang<1e-7f)return q;
    const Vec3<f32> ax=omega.normalized();
    const f32 s=std::sin(ang*0.5f);
    Qf dq{std::cos(ang*0.5f),ax.x*s,ax.y*s,ax.z*s};
    return qnorm(qmul(q,dq));
}
// ─── Euler conversions ────────────────────────────────────────────────────────
Qf from_euler_xyz(f32 x,f32 y,f32 z) noexcept {
    const f32 cx=std::cos(x*0.5f),sx=std::sin(x*0.5f);
    const f32 cy=std::cos(y*0.5f),sy=std::sin(y*0.5f);
    const f32 cz=std::cos(z*0.5f),sz=std::sin(z*0.5f);
    return qnorm(Qf{cx*cy*cz-sx*sy*sz,sx*cy*cz+cx*sy*sz,cx*sy*cz-sx*cy*sz,cx*cy*sz+sx*sy*cz});
}
Qf from_euler_zyx(f32 z,f32 y,f32 x) noexcept {
    const f32 cz=std::cos(z*0.5f),sz=std::sin(z*0.5f);
    const f32 cy=std::cos(y*0.5f),sy=std::sin(y*0.5f);
    const f32 cx=std::cos(x*0.5f),sx=std::sin(x*0.5f);
    return qnorm(Qf{cx*cy*cz+sx*sy*sz,sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,cx*cy*sz-sx*sy*cz});
}
Vec3<f32> to_euler_xyz(Qf q) noexcept {
    return{std::atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y)),
           std::asin(std::max(-1.0f,std::min(1.0f,2*(q.w*q.y-q.z*q.x)))),
           std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))};
}
Vec3<f32> to_euler_zyx(Qf q) noexcept {
    return{std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),
           std::asin(std::max(-1.0f,std::min(1.0f,2*(q.w*q.y-q.z*q.x)))),
           std::atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y))};
}
Qf from_axis_angle(Vec3<f32> axis,f32 angle) noexcept {
    const f32 s=std::sin(angle*0.5f);
    return qnorm(Qf{std::cos(angle*0.5f),axis.x*s,axis.y*s,axis.z*s});
}
void to_axis_angle(Qf q,Vec3<f32>& axis,f32& angle) noexcept {
    q=qnorm(q);
    angle=2*std::acos(std::max(-1.0f,std::min(1.0f,q.w)));
    const f32 s=std::sqrt(1-q.w*q.w);
    if(s<1e-7f)axis={0,1,0};
    else axis={q.x/s,q.y/s,q.z/s};
}
Qf squad_eval(const Qf* ctrl,i32 n,f32 t) noexcept {
    // De Casteljau on S3
    std::vector<Qf> tmp(ctrl,ctrl+n);
    for(i32 r=1;r<n;++r)
        for(i32 i=0;i<n-r;++i){
            const f32 d=qdot(tmp[i],tmp[i+1]);
            Qf q2=d<0?qscale(tmp[i+1],-1):tmp[i+1];
            tmp[i]=qnorm(qmul(tmp[i],qexp(qscale(qlog(qmul(qconj(tmp[i]),q2)),t))));
        }
    return tmp[0];
}
} // namespace qs::math::quat_ext
