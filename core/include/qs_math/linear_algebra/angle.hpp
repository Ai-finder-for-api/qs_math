#pragma once
// qs_math/angle.hpp — Strong angle types: Degrees and Radians
#include "scalar_types.hpp"
#include "constants.hpp"
#include "scalar_math.hpp"
#include <compare>

namespace qs::math {

template<FloatScalar T> struct Degrees;
template<FloatScalar T> struct Radians;

template<FloatScalar T = f32>
struct Radians {
    T value{T(0)};
    constexpr Radians() noexcept = default;
    constexpr explicit Radians(T v) noexcept : value(v) {}
    constexpr Radians(Degrees<T> d) noexcept;

    constexpr Radians  operator+(Radians r) const noexcept { return Radians{value+r.value}; }
    constexpr Radians  operator-(Radians r) const noexcept { return Radians{value-r.value}; }
    constexpr Radians  operator*(T s)       const noexcept { return Radians{value*s}; }
    constexpr Radians  operator/(T s)       const noexcept { return Radians{value/s}; }
    constexpr Radians  operator-()          const noexcept { return Radians{-value}; }
    constexpr Radians& operator+=(Radians r) noexcept { value+=r.value; return *this; }
    constexpr Radians& operator-=(Radians r) noexcept { value-=r.value; return *this; }
    constexpr Radians& operator*=(T s)       noexcept { value*=s; return *this; }
    constexpr Radians& operator/=(T s)       noexcept { value/=s; return *this; }
    constexpr auto     operator<=>(const Radians&) const noexcept = default;

    [[nodiscard]] constexpr Radians normalized() const noexcept {
        T v=value; const T tp=T(TWO_PI_D), p=T(PI_D);
        while(v> p) v-=tp; while(v<-p) v+=tp; return Radians{v};
    }
    [[nodiscard]] constexpr Radians normalized_positive() const noexcept {
        T v=value; const T tp=T(TWO_PI_D);
        while(v<T(0)) v+=tp; while(v>=tp) v-=tp; return Radians{v};
    }
    [[nodiscard]] constexpr Radians shortest_to(Radians t) const noexcept {
        T d=t.value-value; const T tp=T(TWO_PI_D),p=T(PI_D);
        while(d> p) d-=tp; while(d<-p) d+=tp; return Radians{d};
    }
    [[nodiscard]] constexpr Degrees<T> to_degrees() const noexcept;
    [[nodiscard]] constexpr T raw() const noexcept { return value; }
};

template<FloatScalar T = f32>
struct Degrees {
    T value{T(0)};
    constexpr Degrees() noexcept = default;
    constexpr explicit Degrees(T v) noexcept : value(v) {}
    constexpr Degrees(Radians<T> r) noexcept : value(r.value*T(RAD_TO_DEG_D)) {}

    constexpr Degrees  operator+(Degrees d) const noexcept { return Degrees{value+d.value}; }
    constexpr Degrees  operator-(Degrees d) const noexcept { return Degrees{value-d.value}; }
    constexpr Degrees  operator*(T s)       const noexcept { return Degrees{value*s}; }
    constexpr Degrees  operator/(T s)       const noexcept { return Degrees{value/s}; }
    constexpr Degrees  operator-()          const noexcept { return Degrees{-value}; }
    constexpr Degrees& operator+=(Degrees d) noexcept { value+=d.value; return *this; }
    constexpr Degrees& operator-=(Degrees d) noexcept { value-=d.value; return *this; }
    constexpr Degrees& operator*=(T s)       noexcept { value*=s; return *this; }
    constexpr Degrees& operator/=(T s)       noexcept { value/=s; return *this; }
    constexpr auto     operator<=>(const Degrees&) const noexcept = default;

    [[nodiscard]] constexpr Degrees normalized() const noexcept {
        T v=value; while(v>T(180)) v-=T(360); while(v<-T(180)) v+=T(360); return Degrees{v};
    }
    [[nodiscard]] constexpr Degrees normalized_positive() const noexcept {
        T v=value; while(v<T(0)) v+=T(360); while(v>=T(360)) v-=T(360); return Degrees{v};
    }
    [[nodiscard]] constexpr Degrees shortest_to(Degrees t) const noexcept {
        T d=t.value-value; while(d>T(180)) d-=T(360); while(d<-T(180)) d+=T(360); return Degrees{d};
    }
    [[nodiscard]] constexpr Radians<T> to_radians() const noexcept { return Radians<T>{value*T(DEG_TO_RAD_D)}; }
    [[nodiscard]] constexpr T raw() const noexcept { return value; }
};

template<FloatScalar T> constexpr Radians<T>::Radians(Degrees<T> d) noexcept : value(d.value*T(DEG_TO_RAD_D)) {}
template<FloatScalar T> constexpr Degrees<T> Radians<T>::to_degrees() const noexcept { return Degrees<T>{value*T(RAD_TO_DEG_D)}; }

namespace literals {
    constexpr Radians<f32> operator""_rad  (long double v) noexcept { return Radians<f32>{f32(v)}; }
    constexpr Degrees<f32> operator""_deg  (long double v) noexcept { return Degrees<f32>{f32(v)}; }
    constexpr Radians<f64> operator""_rad64(long double v) noexcept { return Radians<f64>{f64(v)}; }
    constexpr Degrees<f64> operator""_deg64(long double v) noexcept { return Degrees<f64>{f64(v)}; }
}

using Radiansf = Radians<f32>; using Radiansd = Radians<f64>;
using Degreesf = Degrees<f32>; using Degreesd = Degrees<f64>;

template<FloatScalar T>
[[nodiscard]] constexpr Radians<T> lerp_angle(Radians<T> a, Radians<T> b, T t) noexcept {
    return Radians<T>{a.value + a.shortest_to(b).value * t};
}

} // namespace qs::math

// ─── Angular arithmetic ────────────────────────────────────────────────────
// Shortest angular difference (wraps correctly across ±pi boundary)
template<FloatScalar T>
[[nodiscard]] inline Radians<T> angle_diff(Radians<T> a, Radians<T> b) noexcept {
    T d = b.value - a.value;
    // Wrap to [-pi, pi]
    d -= T(TWO_PI) * std::round(d * T(INV_TWO_PI));
    return Radians<T>{d};
}
template<FloatScalar T>
[[nodiscard]] inline Degrees<T> angle_diff(Degrees<T> a, Degrees<T> b) noexcept {
    T d = b.value - a.value;
    d -= T(360) * std::round(d / T(360));
    return Degrees<T>{d};
}

// Lerp angles (shortest path)
template<FloatScalar T>
[[nodiscard]] inline Radians<T> lerp_angle(Radians<T> a, Radians<T> b, T t) noexcept {
    return Radians<T>{a.value + angle_diff(a,b).value * t};
}
template<FloatScalar T>
[[nodiscard]] inline Degrees<T> lerp_angle(Degrees<T> a, Degrees<T> b, T t) noexcept {
    return Degrees<T>{a.value + angle_diff(a,b).value * t};
}

// Clamp angle to [-pi, pi]
template<FloatScalar T>
[[nodiscard]] inline Radians<T> wrap_pi(Radians<T> a) noexcept {
    T v = a.value;
    v -= T(TWO_PI) * std::round(v * T(INV_TWO_PI));
    return Radians<T>{v};
}

// Is angle in range [lo, hi] (CCW from lo to hi)
template<FloatScalar T>
[[nodiscard]] inline bool angle_in_range(Radians<T> a, Radians<T> lo, Radians<T> hi) noexcept {
    a = wrap_pi(a); lo = wrap_pi(lo); hi = wrap_pi(hi);
    if(lo.value <= hi.value) return a.value>=lo.value && a.value<=hi.value;
    return a.value>=lo.value || a.value<=hi.value;
}

// Angular velocity: clamp to max_speed
template<FloatScalar T>
[[nodiscard]] inline Radians<T> rotate_towards(Radians<T> from, Radians<T> to, T max_step) noexcept {
    T diff = angle_diff(from,to).value;
    T step = clamp(diff, -max_step, max_step);
    return Radians<T>{from.value + step};
}

// ─── Degree helpers ────────────────────────────────────────────────────────
[[nodiscard]] constexpr f32 wrap_degrees(f32 d) noexcept {
    d = fmod(d, 360.f);
    return d < 0.f ? d + 360.f : d;
}
[[nodiscard]] constexpr f32 wrap_degrees_pm180(f32 d) noexcept {
    d = fmod(d + 180.f, 360.f);
    return (d < 0.f ? d + 360.f : d) - 180.f;
}
