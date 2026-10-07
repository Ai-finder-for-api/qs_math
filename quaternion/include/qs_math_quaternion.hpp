
#pragma once
// qs_math/quaternion/include/qs_math_quaternion.hpp (C++20)
// Extended quaternion: dual quaternion, double cover, averaging, splines.
#include <qs_math/arithmetic/scalar_types.hpp>
#include <qs_math/quaternion/quat.hpp>
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/linear_algebra/mat4.hpp>
#include <cmath>
#include <vector>
namespace qs::math::quat_ext {
using namespace qs::math;
using Qf = Quat<f32>;
using Qd = Quat<f64>;
// ─── Dual quaternion (rigid transform = rotation + translation) ───────────────
struct DualQuat {
    Qf real{1,0,0,0};  // rotation
    Qf dual{0,0,0,0};  // translation encoded
    [[nodiscard]] static DualQuat from_rt(Qf r, Vec3<f32> t) noexcept;
    [[nodiscard]] Vec3<f32> transform(Vec3<f32> p) const noexcept;
    [[nodiscard]] DualQuat operator*(const DualQuat& o) const noexcept;
    [[nodiscard]] DualQuat conjugate() const noexcept;
    void normalise() noexcept;
    [[nodiscard]] Mat4<f32> to_mat4() const noexcept;
};
// ─── Weighted quaternion average (Markley QUEST method) ─────────────────────
[[nodiscard]] Qf weighted_average(const Qf* quats, const f32* weights, i32 n,
    i32 max_iter=100) noexcept;
// ─── Quaternion spline interpolation (De Casteljau on S3) ────────────────────
[[nodiscard]] Qf squad_eval(const Qf* ctrl, i32 n, f32 t) noexcept;
// ─── Rotation between two vectors ─────────────────────────────────────────────
[[nodiscard]] Qf rotation_between(Vec3<f32> from, Vec3<f32> to) noexcept;
// ─── Decompose quaternion into swing-twist ────────────────────────────────────
void swing_twist(Qf q, Vec3<f32> twist_axis, Qf& swing_out, Qf& twist_out) noexcept;
// ─── Quaternion integration (angular velocity) ────────────────────────────────
[[nodiscard]] Qf integrate_angular_vel(Qf q, Vec3<f32> omega, f32 dt) noexcept;
// ─── Euler angle conversions ─────────────────────────────────────────────────
[[nodiscard]] Qf from_euler_xyz(f32 x, f32 y, f32 z) noexcept;
[[nodiscard]] Qf from_euler_zyx(f32 z, f32 y, f32 x) noexcept;
[[nodiscard]] Vec3<f32> to_euler_xyz(Qf q) noexcept;
[[nodiscard]] Vec3<f32> to_euler_zyx(Qf q) noexcept;
// ─── Axis-angle <-> quaternion ────────────────────────────────────────────────
[[nodiscard]] Qf from_axis_angle(Vec3<f32> axis, f32 angle) noexcept;
[[nodiscard]] void to_axis_angle(Qf q, Vec3<f32>& axis_out, f32& angle_out) noexcept;
// ─── Quaternion power and log/exp ─────────────────────────────────────────────
[[nodiscard]] Qf qpow(Qf q, f32 t) noexcept;
[[nodiscard]] Qf qlog(Qf q) noexcept;
[[nodiscard]] Qf qexp(Qf q) noexcept;
} // namespace qs::math::quat_ext
