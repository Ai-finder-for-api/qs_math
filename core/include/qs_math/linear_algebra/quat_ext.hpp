#pragma once
// quat_ext.hpp — Extended quaternion ops: squad, exponential map, double cover,
//               averaging, derivative, angular velocity, gimbal detection
#include "quat.hpp"
#include "mat4.hpp"

namespace qs::math {

// Quaternion exponential map (axis-angle → quat)
template<FloatScalar T>
[[nodiscard]] inline Quat<T> quat_exp(Vec3<T> v) noexcept {
    T angle=v.length();
    if(angle<T(EPS_ANGLE)) return Quat<T>::identity();
    Vec3<T> axis=v*(T(1)/angle);
    return Quat<T>::from_axis_angle(axis,angle);
}

// Quaternion logarithm (quat → axis-angle vec3)
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> quat_log(Quat<T> q) noexcept {
    q=q.normalized();
    T vec_len=Vec3<T>{q.x,q.y,q.z}.length();
    if(vec_len<T(EPS_ANGLE)) return Vec3<T>{T(0)};
    T angle=T(2)*atan2(vec_len,q.w);
    return Vec3<T>{q.x,q.y,q.z}*(angle/vec_len);
}

// SQUAD tangent computation (for smooth path)
template<FloatScalar T>
[[nodiscard]] inline Quat<T> squad_tangent(Quat<T> q_prev, Quat<T> q, Quat<T> q_next) noexcept {
    Quat<T> qi=q.inverse();
    Vec3<T> log1=quat_log(qi*q_next);
    Vec3<T> log2=quat_log(qi*q_prev);
    Vec3<T> sum=(log1+log2)*T(-0.25);
    return q*quat_exp(sum);
}

// Quaternion angular velocity from two quats and delta time
template<FloatScalar T>
[[nodiscard]] inline Vec3<T> angular_velocity(Quat<T> q1, Quat<T> q2, T dt) noexcept {
    if(dt<T(EPS_LENGTH)) return Vec3<T>{T(0)};
    Quat<T> dq=q2*q1.inverse();
    return quat_log(dq)*(T(2)/dt);
}

// Integrate angular velocity: q_new = q * exp(omega * dt / 2)
template<FloatScalar T>
[[nodiscard]] inline Quat<T> integrate_angular_velocity(Quat<T> q, Vec3<T> omega, T dt) noexcept {
    return (q*quat_exp(omega*(dt*T(0.5)))).normalized();
}

// Average N quaternions (iterative geodesic mean)
template<FloatScalar T>
[[nodiscard]] inline Quat<T> quat_mean(const Quat<T>* quats, i32 n, i32 max_iter=10) noexcept {
    if(n<=0) return Quat<T>::identity();
    Quat<T> mean=quats[0];
    for(i32 iter=0;iter<max_iter;++iter){
        Vec3<T> error{T(0)};
        for(i32 i=0;i<n;++i) error=error+quat_log(mean.inverse()*quats[i]);
        error=error*(T(1)/T(n));
        if(error.length_sq()<T(EPS_ANGLE*EPS_ANGLE)) break;
        mean=(mean*quat_exp(error)).normalized();
    }
    return mean;
}

// Ensure shortest path (flip if dot < 0)
template<FloatScalar T>
[[nodiscard]] constexpr Quat<T> shortest_path(Quat<T> from, Quat<T> to) noexcept {
    return from.dot(to)<T(0) ? -to : to;
}

// Gimbal lock detection (threshold on dot of adjacent Euler axes)
template<FloatScalar T>
[[nodiscard]] inline bool has_gimbal_lock(Quat<T> q, T threshold=T(0.999)) noexcept {
    // Check |2(qx*qz - qw*qy)| ≈ 1 (XZY convention)
    T test=q.x*q.z-q.w*q.y;
    return abs(T(2)*test)>threshold;
}

// Double cover: ensure quaternion is in the "northern hemisphere" (w >= 0)
template<FloatScalar T>
[[nodiscard]] constexpr Quat<T> canonical_form(Quat<T> q) noexcept {
    return q.w<T(0)?-q:q;
}

// Constraint: restrict rotation around an axis to ±limit_rad
template<FloatScalar T>
[[nodiscard]] inline Quat<T> clamp_axis_rotation(Quat<T> q, Vec3<T> axis, T limit_rad) noexcept {
    Vec3<T> v; T angle;
    q.to_axis_angle(v,angle);
    // Project rotation onto axis
    T proj=dot(v,axis.normalized())*angle;
    proj=clamp(proj,-limit_rad,limit_rad);
    return Quat<T>::from_axis_angle(axis,proj);
}

// Twist-swing decomposition (separates rotation into swing around up and twist around axis)
template<FloatScalar T>
inline void twist_swing_decompose(Quat<T> q, Vec3<T> twist_axis,
                                   Quat<T>& twist, Quat<T>& swing) noexcept
{
    Vec3<T> qvec{q.x,q.y,q.z};
    Vec3<T> proj=twist_axis*dot(qvec,twist_axis);
    twist=Quat<T>{proj.x,proj.y,proj.z,q.w}.normalized();
    swing=(q*twist.inverse()).normalized();
}

// Kinematic chain: compose array of local quats
template<FloatScalar T>
[[nodiscard]] inline Quat<T> compose_local_quats(const Quat<T>* quats, i32 n) noexcept {
    Quat<T> result=Quat<T>::identity();
    for(i32 i=0;i<n;++i) result=result*quats[i];
    return result.normalized();
}

// Look-rotation quaternion (forward/up)
template<FloatScalar T>
[[nodiscard]] inline Quat<T> look_rotation(Vec3<T> forward, Vec3<T> up=Vec3<T>::up()) noexcept {
    Vec3<T> f=forward.normalized();
    Vec3<T> r=cross(up,f).normalized();
    Vec3<T> u=cross(f,r);
    Mat3<T> m{r,u,-f}; // note: -f for right-handed -Z forward
    return Quat<T>::from_mat3(m).normalized();
}

} // namespace qs::math
