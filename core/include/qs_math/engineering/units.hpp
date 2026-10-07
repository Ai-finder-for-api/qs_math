#pragma once
// qs_math/units.hpp — Strong unit types for distances, time, mass, temperature
#include "scalar_types.hpp"
#include <compare>

namespace qs::math::units {

// CRTP base for all unit types
template<typename Derived, FloatScalar T>
struct Unit {
    T value{T(0)};
    constexpr explicit Unit(T v) noexcept : value(v) {}
    constexpr Unit() noexcept = default;
    [[nodiscard]] constexpr T raw() const noexcept { return value; }

    constexpr Derived  operator+(Derived o) const noexcept { return Derived{value+o.value}; }
    constexpr Derived  operator-(Derived o) const noexcept { return Derived{value-o.value}; }
    constexpr Derived  operator*(T s)       const noexcept { return Derived{value*s}; }
    constexpr Derived  operator/(T s)       const noexcept { return Derived{value/s}; }
    constexpr Derived  operator-()          const noexcept { return Derived{-value}; }
    constexpr Derived& operator+=(Derived o) noexcept { value+=o.value; return static_cast<Derived&>(*this); }
    constexpr Derived& operator-=(Derived o) noexcept { value-=o.value; return static_cast<Derived&>(*this); }
    constexpr Derived& operator*=(T s)       noexcept { value*=s; return static_cast<Derived&>(*this); }
    constexpr Derived& operator/=(T s)       noexcept { value/=s; return static_cast<Derived&>(*this); }
    constexpr auto     operator<=>(const Unit&) const noexcept = default;
};

// --- Distance ---
struct Meters    : Unit<Meters,    f32> { using Unit::Unit; };
struct Centimeters:Unit<Centimeters,f32>{ using Unit::Unit; };
struct Millimeters:Unit<Millimeters,f32>{ using Unit::Unit; };
struct Kilometers : Unit<Kilometers, f32> { using Unit::Unit; };
struct Inches    : Unit<Inches,    f32> { using Unit::Unit; };
struct Feet      : Unit<Feet,      f32> { using Unit::Unit; };
struct Miles     : Unit<Miles,     f32> { using Unit::Unit; };

constexpr Meters      to_meters(Centimeters v) noexcept { return Meters{v.value*0.01f}; }
constexpr Meters      to_meters(Millimeters v) noexcept { return Meters{v.value*0.001f}; }
constexpr Meters      to_meters(Kilometers  v) noexcept { return Meters{v.value*1000.0f}; }
constexpr Meters      to_meters(Inches      v) noexcept { return Meters{v.value*0.0254f}; }
constexpr Meters      to_meters(Feet        v) noexcept { return Meters{v.value*0.3048f}; }
constexpr Meters      to_meters(Miles       v) noexcept { return Meters{v.value*1609.344f}; }
constexpr Centimeters to_centimeters(Meters v) noexcept { return Centimeters{v.value*100.0f}; }
constexpr Millimeters to_millimeters(Meters v) noexcept { return Millimeters{v.value*1000.0f}; }
constexpr Inches      to_inches(Meters v)      noexcept { return Inches{v.value/0.0254f}; }
constexpr Feet        to_feet(Meters v)        noexcept { return Feet{v.value/0.3048f}; }

// --- Time ---
struct Seconds     : Unit<Seconds,     f64> { using Unit::Unit; };
struct Milliseconds: Unit<Milliseconds,f64> { using Unit::Unit; };
struct Microseconds: Unit<Microseconds,f64> { using Unit::Unit; };
struct Nanoseconds : Unit<Nanoseconds, f64> { using Unit::Unit; };
struct Minutes     : Unit<Minutes,     f64> { using Unit::Unit; };
struct Hours       : Unit<Hours,       f64> { using Unit::Unit; };

constexpr Seconds      to_seconds(Milliseconds v) noexcept { return Seconds{v.value*0.001}; }
constexpr Seconds      to_seconds(Microseconds v) noexcept { return Seconds{v.value*1e-6}; }
constexpr Seconds      to_seconds(Nanoseconds  v) noexcept { return Seconds{v.value*1e-9}; }
constexpr Seconds      to_seconds(Minutes      v) noexcept { return Seconds{v.value*60.0}; }
constexpr Seconds      to_seconds(Hours        v) noexcept { return Seconds{v.value*3600.0}; }
constexpr Milliseconds to_milliseconds(Seconds v) noexcept { return Milliseconds{v.value*1000.0}; }
constexpr Microseconds to_microseconds(Seconds v) noexcept { return Microseconds{v.value*1e6}; }
constexpr Nanoseconds  to_nanoseconds(Seconds  v) noexcept { return Nanoseconds{v.value*1e9}; }

// --- Mass ---
struct Kilograms: Unit<Kilograms,f32> { using Unit::Unit; };
struct Grams    : Unit<Grams,    f32> { using Unit::Unit; };
struct Pounds   : Unit<Pounds,   f32> { using Unit::Unit; };
struct Ounces   : Unit<Ounces,   f32> { using Unit::Unit; };

constexpr Kilograms to_kilograms(Grams   v) noexcept { return Kilograms{v.value*0.001f}; }
constexpr Kilograms to_kilograms(Pounds  v) noexcept { return Kilograms{v.value*0.453592f}; }
constexpr Kilograms to_kilograms(Ounces  v) noexcept { return Kilograms{v.value*0.0283495f}; }
constexpr Grams     to_grams(Kilograms   v) noexcept { return Grams{v.value*1000.0f}; }

// --- Temperature ---
struct Celsius   : Unit<Celsius,    f32> { using Unit::Unit; };
struct Fahrenheit: Unit<Fahrenheit, f32> { using Unit::Unit; };
struct Kelvin    : Unit<Kelvin,     f32> { using Unit::Unit; };

constexpr Celsius    to_celsius(Fahrenheit v)  noexcept { return Celsius{(v.value-32.0f)*5.0f/9.0f}; }
constexpr Celsius    to_celsius(Kelvin     v)  noexcept { return Celsius{v.value-273.15f}; }
constexpr Fahrenheit to_fahrenheit(Celsius v)  noexcept { return Fahrenheit{v.value*9.0f/5.0f+32.0f}; }
constexpr Kelvin     to_kelvin(Celsius     v)  noexcept { return Kelvin{v.value+273.15f}; }

// UDLs
namespace literals {
    constexpr Meters      operator""_m  (long double v) noexcept { return Meters{f32(v)}; }
    constexpr Centimeters operator""_cm (long double v) noexcept { return Centimeters{f32(v)}; }
    constexpr Millimeters operator""_mm (long double v) noexcept { return Millimeters{f32(v)}; }
    constexpr Kilometers  operator""_km (long double v) noexcept { return Kilometers{f32(v)}; }
    constexpr Seconds     operator""_s  (long double v) noexcept { return Seconds{f64(v)}; }
    constexpr Milliseconds operator""_ms(long double v) noexcept { return Milliseconds{f64(v)}; }
    constexpr Kilograms   operator""_kg (long double v) noexcept { return Kilograms{f32(v)}; }
    constexpr Celsius     operator""_C  (long double v) noexcept { return Celsius{f32(v)}; }
    constexpr Kelvin      operator""_K  (long double v) noexcept { return Kelvin{f32(v)}; }
}

} // namespace qs::math::units
