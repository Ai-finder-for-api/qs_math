#pragma once
// qs_math/units_ext.hpp — Extended unit system: SI derived, imperial, astronomical conversions (C++20)
#include "scalar_types.hpp"
#include "units.hpp"
#include <cmath>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qs {

// ─── Base SI units (extending units.hpp) ──────────────────────────────────────
// Fundamental: m, kg, s, A, K, mol, cd
// units.hpp covers Meters, Seconds, Kilograms, Celsius — we add the rest

template<typename T=f64>
using Amperes    = StrongUnit<T, struct AmpereTag>;
template<typename T=f64>
using Moles      = StrongUnit<T, struct MoleTag>;
template<typename T=f64>
using Candelas   = StrongUnit<T, struct CandelaTag>;

// ─── Derived SI units ─────────────────────────────────────────────────────────
template<typename T=f64> using Newtons   = StrongUnit<T, struct NewtonTag>;    // kg·m/s²
template<typename T=f64> using Joules    = StrongUnit<T, struct JouleTag>;     // kg·m²/s²
template<typename T=f64> using Watts     = StrongUnit<T, struct WattTag>;      // J/s
template<typename T=f64> using Pascals   = StrongUnit<T, struct PascalTag>;    // N/m²
template<typename T=f64> using Coulombs  = StrongUnit<T, struct CoulombTag>;   // A·s
template<typename T=f64> using Volts     = StrongUnit<T, struct VoltTag>;      // J/C
template<typename T=f64> using Ohms      = StrongUnit<T, struct OhmTag>;       // V/A
template<typename T=f64> using Farads    = StrongUnit<T, struct FaradTag>;     // C/V
template<typename T=f64> using Henrys    = StrongUnit<T, struct HenryTag>;     // V·s/A
template<typename T=f64> using Teslas    = StrongUnit<T, struct TeslaTag>;     // kg/(A·s²)
template<typename T=f64> using Webers    = StrongUnit<T, struct WeberTag>;     // V·s
template<typename T=f64> using Hertz     = StrongUnit<T, struct HertzTag>;     // 1/s
template<typename T=f64> using Becquerels= StrongUnit<T, struct BecquerelTag>; // 1/s (radioactivity)
template<typename T=f64> using Grays     = StrongUnit<T, struct GrayTag>;      // J/kg (absorbed dose)
template<typename T=f64> using Sieverts  = StrongUnit<T, struct SievertTag>;   // J/kg (dose equivalent)
template<typename T=f64> using Lumens    = StrongUnit<T, struct LumenTag>;     // cd·sr
template<typename T=f64> using Lux       = StrongUnit<T, struct LuxTag>;       // lm/m²

// ─── Imperial units ───────────────────────────────────────────────────────────
// Conversion constants to SI
static constexpr f64 INCH_TO_M    = 0.0254;
static constexpr f64 FOOT_TO_M    = 0.3048;
static constexpr f64 YARD_TO_M    = 0.9144;
static constexpr f64 MILE_TO_M    = 1609.344;
static constexpr f64 NAUTMI_TO_M  = 1852.0;
static constexpr f64 LB_TO_KG     = 0.45359237;
static constexpr f64 OZ_TO_KG     = 0.028349523125;
static constexpr f64 TON_US_TO_KG = 907.18474;
static constexpr f64 TON_UK_TO_KG = 1016.0469088;
static constexpr f64 GAL_US_TO_M3 = 0.003785411784;
static constexpr f64 GAL_UK_TO_M3 = 0.00454609;
static constexpr f64 BTU_TO_J     = 1055.05585262;
static constexpr f64 HP_TO_W      = 745.69987;
static constexpr f64 PSI_TO_PA    = 6894.757293168;
static constexpr f64 ATM_TO_PA    = 101325.0;
static constexpr f64 BAR_TO_PA    = 100000.0;
static constexpr f64 TORR_TO_PA   = 133.322;

// Imperial → SI
[[nodiscard]] constexpr f64 inches_to_m(f64 v) { return v*INCH_TO_M; }
[[nodiscard]] constexpr f64 feet_to_m(f64 v)   { return v*FOOT_TO_M; }
[[nodiscard]] constexpr f64 miles_to_m(f64 v)  { return v*MILE_TO_M; }
[[nodiscard]] constexpr f64 pounds_to_kg(f64 v){ return v*LB_TO_KG;  }
[[nodiscard]] constexpr f64 psi_to_pa(f64 v)   { return v*PSI_TO_PA; }
[[nodiscard]] constexpr f64 btu_to_j(f64 v)    { return v*BTU_TO_J;  }
[[nodiscard]] constexpr f64 hp_to_w(f64 v)     { return v*HP_TO_W;   }

// SI → Imperial
[[nodiscard]] constexpr f64 m_to_feet(f64 v)   { return v/FOOT_TO_M;  }
[[nodiscard]] constexpr f64 m_to_miles(f64 v)  { return v/MILE_TO_M;  }
[[nodiscard]] constexpr f64 kg_to_pounds(f64 v){ return v/LB_TO_KG;   }
[[nodiscard]] constexpr f64 pa_to_psi(f64 v)   { return v/PSI_TO_PA;  }

// Temperature conversions
[[nodiscard]] constexpr f64 celsius_to_fahrenheit(f64 c) { return c*9/5+32; }
[[nodiscard]] constexpr f64 fahrenheit_to_celsius(f64 f) { return (f-32)*5/9; }
[[nodiscard]] constexpr f64 celsius_to_kelvin(f64 c)    { return c+273.15; }
[[nodiscard]] constexpr f64 kelvin_to_celsius(f64 k)    { return k-273.15; }
[[nodiscard]] constexpr f64 fahrenheit_to_kelvin(f64 f) { return celsius_to_kelvin(fahrenheit_to_celsius(f)); }
[[nodiscard]] constexpr f64 rankine_to_kelvin(f64 r)    { return r*5/9; }
[[nodiscard]] constexpr f64 kelvin_to_rankine(f64 k)    { return k*9/5; }

// ─── Astronomical units ───────────────────────────────────────────────────────
static constexpr f64 AU_TO_M       = 1.495978707e11;   // astronomical unit
static constexpr f64 LY_TO_M       = 9.4607304725808e15; // light-year
static constexpr f64 PC_TO_M       = 3.085677581e16;   // parsec
static constexpr f64 PC_TO_LY      = 3.26156;
static constexpr f64 SOLAR_RADIUS  = 6.957e8;          // m
static constexpr f64 SOLAR_MASS    = 1.989e30;          // kg
static constexpr f64 SOLAR_LUM     = 3.828e26;          // W
static constexpr f64 EARTH_RADIUS  = 6.371e6;           // m
static constexpr f64 EARTH_MASS    = 5.972e24;          // kg
static constexpr f64 MOON_DIST     = 3.844e8;           // m (mean)

[[nodiscard]] constexpr f64 au_to_m(f64 v)  { return v*AU_TO_M; }
[[nodiscard]] constexpr f64 ly_to_m(f64 v)  { return v*LY_TO_M; }
[[nodiscard]] constexpr f64 pc_to_m(f64 v)  { return v*PC_TO_M; }
[[nodiscard]] constexpr f64 m_to_au(f64 v)  { return v/AU_TO_M; }
[[nodiscard]] constexpr f64 m_to_ly(f64 v)  { return v/LY_TO_M; }
[[nodiscard]] constexpr f64 m_to_pc(f64 v)  { return v/PC_TO_M; }

// Magnitudes: apparent magnitude m = M + 5 log10(d/10pc)
[[nodiscard]] inline f64 apparent_magnitude(f64 abs_mag, f64 distance_pc) {
    return abs_mag+5*std::log10(distance_pc/10);
}
[[nodiscard]] inline f64 absolute_magnitude(f64 app_mag, f64 distance_pc) {
    return app_mag-5*std::log10(distance_pc/10);
}
// Luminosity from absolute magnitude: L/L_sun = 10^(0.4*(M_sun - M))
[[nodiscard]] inline f64 luminosity_from_abs_mag(f64 M, f64 M_sun=4.83) {
    return std::pow(10,0.4*(M_sun-M));
}

// ─── Unit registry (runtime conversion) ──────────────────────────────────────
struct UnitRegistry {
    // All units stored as conversion factor to their SI base
    std::unordered_map<std::string,f64> to_si;

    UnitRegistry() {
        // Length
        to_si["m"]=1; to_si["km"]=1e3; to_si["cm"]=0.01; to_si["mm"]=0.001;
        to_si["in"]=INCH_TO_M; to_si["ft"]=FOOT_TO_M; to_si["yd"]=YARD_TO_M;
        to_si["mi"]=MILE_TO_M; to_si["nmi"]=NAUTMI_TO_M;
        to_si["au"]=AU_TO_M; to_si["ly"]=LY_TO_M; to_si["pc"]=PC_TO_M;
        to_si["nm"]=1e-9; to_si["um"]=1e-6; to_si["angstrom"]=1e-10;
        // Mass
        to_si["kg"]=1; to_si["g"]=0.001; to_si["mg"]=1e-6;
        to_si["lb"]=LB_TO_KG; to_si["oz"]=OZ_TO_KG;
        to_si["t"]=1000; to_si["ton"]=TON_US_TO_KG;
        // Time
        to_si["s"]=1; to_si["ms"]=0.001; to_si["us"]=1e-6; to_si["ns"]=1e-9;
        to_si["min"]=60; to_si["hr"]=3600; to_si["day"]=86400;
        to_si["yr"]=365.25*86400; to_si["week"]=7*86400;
        // Energy
        to_si["J"]=1; to_si["kJ"]=1e3; to_si["MJ"]=1e6;
        to_si["cal"]=4.184; to_si["kcal"]=4184; to_si["Wh"]=3600; to_si["kWh"]=3.6e6;
        to_si["eV"]=1.602176634e-19; to_si["BTU"]=BTU_TO_J;
        // Power
        to_si["W"]=1; to_si["kW"]=1e3; to_si["MW"]=1e6; to_si["hp"]=HP_TO_W;
        // Pressure
        to_si["Pa"]=1; to_si["kPa"]=1e3; to_si["MPa"]=1e6;
        to_si["bar"]=BAR_TO_PA; to_si["atm"]=ATM_TO_PA;
        to_si["psi"]=PSI_TO_PA; to_si["torr"]=TORR_TO_PA; to_si["mmHg"]=TORR_TO_PA;
        // Angle
        to_si["rad"]=1; to_si["deg"]=std::numbers::pi_v<f64>/180;
        to_si["arcmin"]=std::numbers::pi_v<f64>/10800; to_si["arcsec"]=std::numbers::pi_v<f64>/648000;
    }

    [[nodiscard]] f64 convert(f64 value, std::string_view from, std::string_view to_unit) const {
        auto it_f=to_si.find(std::string(from));
        auto it_t=to_si.find(std::string(to_unit));
        if(it_f==to_si.end()||it_t==to_si.end()) return value;
        return value*it_f->second/it_t->second;
    }

    [[nodiscard]] bool has_unit(std::string_view name) const {
        return to_si.count(std::string(name))>0;
    }
};

// Global registry singleton helper
[[nodiscard]] inline const UnitRegistry& unit_registry() {
    static UnitRegistry reg;
    return reg;
}

// ─── Dimensional analysis (compile-time dimensions) ──────────────────────────
// Represent dimensions as [L, M, T, I, Θ, N, J] exponents
template<i32 L,i32 M,i32 T,i32 I=0,i32 Theta=0,i32 N=0,i32 J=0>
struct Dimension {
    static constexpr i32 length=L,mass=M,time=T,current=I,temp=Theta,amount=N,intensity=J;
};

template<typename D1,typename D2>
using DimMul=Dimension<D1::length+D2::length,D1::mass+D2::mass,D1::time+D2::time,
                       D1::current+D2::current,D1::temp+D2::temp,D1::amount+D2::amount,D1::intensity+D2::intensity>;
template<typename D1,typename D2>
using DimDiv=Dimension<D1::length-D2::length,D1::mass-D2::mass,D1::time-D2::time,
                       D1::current-D2::current,D1::temp-D2::temp,D1::amount-D2::amount,D1::intensity-D2::intensity>;

// Common named dimensions
using Dimless  =Dimension<0,0,0>;
using DimLength=Dimension<1,0,0>;
using DimMass  =Dimension<0,1,0>;
using DimTime  =Dimension<0,0,1>;
using DimSpeed =Dimension<1,0,-1>;
using DimAccel =Dimension<1,0,-2>;
using DimForce =Dimension<1,1,-2>;
using DimEnergy=Dimension<2,1,-2>;
using DimPower =Dimension<2,1,-3>;
using DimPressure=Dimension<-1,1,-2>;
using DimFreq  =Dimension<0,0,-1>;

template<typename T,typename D>
struct DimValue {
    T value{};
    constexpr DimValue() = default;
    constexpr explicit DimValue(T v) : value(v) {}

    template<typename D2>
    [[nodiscard]] constexpr DimValue<T,DimMul<D,D2>> operator*(DimValue<T,D2> o) const { return DimValue<T,DimMul<D,D2>>(value*o.value); }
    template<typename D2>
    [[nodiscard]] constexpr DimValue<T,DimDiv<D,D2>> operator/(DimValue<T,D2> o) const { return DimValue<T,DimDiv<D,D2>>(value/o.value); }
    [[nodiscard]] constexpr DimValue operator+(DimValue o) const { return DimValue(value+o.value); }
    [[nodiscard]] constexpr DimValue operator-(DimValue o) const { return DimValue(value-o.value); }
};

// Type-safe quantity aliases
using Length_m   = DimValue<f64,DimLength>;
using Mass_kg    = DimValue<f64,DimMass>;
using Time_s     = DimValue<f64,DimTime>;
using Speed_ms   = DimValue<f64,DimSpeed>;
using Force_N    = DimValue<f64,DimForce>;
using Energy_J   = DimValue<f64,DimEnergy>;
using Power_W    = DimValue<f64,DimPower>;
using Pressure_Pa= DimValue<f64,DimPressure>;
using Frequency_Hz=DimValue<f64,DimFreq>;

} // namespace qs
