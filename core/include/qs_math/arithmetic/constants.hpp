#pragma once
// qs_math/constants.hpp — All mathematical constants, compile-time only
#include "scalar_types.hpp"

namespace qs::math {

// Pi
inline constexpr f64 PI_D          = 3.14159265358979323846264338327950288;
inline constexpr f64 TWO_PI_D      = 6.28318530717958647692528676655900576;
inline constexpr f64 HALF_PI_D     = 1.57079632679489661923132169163975144;
inline constexpr f64 QUARTER_PI_D  = 0.78539816339744830961566084581987572;
inline constexpr f64 INV_PI_D      = 0.31830988618379067153776752674502872;
inline constexpr f64 INV_TWO_PI_D  = 0.15915494309189533576888376337251436;
inline constexpr f64 SQRT_PI_D     = 1.77245385090551602729816748334114518;
inline constexpr f32 PI            = static_cast<f32>(PI_D);
inline constexpr f32 TWO_PI        = static_cast<f32>(TWO_PI_D);
inline constexpr f32 HALF_PI       = static_cast<f32>(HALF_PI_D);
inline constexpr f32 QUARTER_PI    = static_cast<f32>(QUARTER_PI_D);
inline constexpr f32 INV_PI        = static_cast<f32>(INV_PI_D);
inline constexpr f32 INV_TWO_PI    = static_cast<f32>(INV_TWO_PI_D);
inline constexpr f32 SQRT_PI       = static_cast<f32>(SQRT_PI_D);

// Euler / Log
inline constexpr f64 E_D           = 2.71828182845904523536028747135266250;
inline constexpr f64 LOG2E_D       = 1.44269504088896340735992468100189214;
inline constexpr f64 LOG10E_D      = 0.43429448190325182765112891891660508;
inline constexpr f64 LN2_D         = 0.69314718055994530941723212145817657;
inline constexpr f64 LN10_D        = 2.30258509299404568401799145468436421;
inline constexpr f32 E             = static_cast<f32>(E_D);
inline constexpr f32 LOG2E         = static_cast<f32>(LOG2E_D);
inline constexpr f32 LOG10E        = static_cast<f32>(LOG10E_D);
inline constexpr f32 LN2           = static_cast<f32>(LN2_D);
inline constexpr f32 LN10          = static_cast<f32>(LN10_D);

// Roots
inline constexpr f64 SQRT2_D       = 1.41421356237309504880168872420969808;
inline constexpr f64 SQRT3_D       = 1.73205080756887729352744634150587237;
inline constexpr f64 SQRT5_D       = 2.23606797749978969640917366873127624;
inline constexpr f64 INV_SQRT2_D   = 0.70710678118654752440084436210484904;
inline constexpr f64 INV_SQRT3_D   = 0.57735026918962576450914878050195746;
inline constexpr f32 SQRT2         = static_cast<f32>(SQRT2_D);
inline constexpr f32 SQRT3         = static_cast<f32>(SQRT3_D);
inline constexpr f32 SQRT5         = static_cast<f32>(SQRT5_D);
inline constexpr f32 INV_SQRT2     = static_cast<f32>(INV_SQRT2_D);
inline constexpr f32 INV_SQRT3     = static_cast<f32>(INV_SQRT3_D);

// Angle conversions
inline constexpr f64 DEG_TO_RAD_D  = PI_D / 180.0;
inline constexpr f64 RAD_TO_DEG_D  = 180.0 / PI_D;
inline constexpr f32 DEG_TO_RAD    = static_cast<f32>(DEG_TO_RAD_D);
inline constexpr f32 RAD_TO_DEG    = static_cast<f32>(RAD_TO_DEG_D);

// Golden ratio
inline constexpr f64 PHI_D         = 1.61803398874989484820458683436563812;
inline constexpr f32 PHI           = static_cast<f32>(PHI_D);
inline constexpr f64 INV_PHI_D     = 0.61803398874989484820458683436563812;
inline constexpr f32 INV_PHI       = static_cast<f32>(INV_PHI_D);

// Physics
inline constexpr f64 GRAVITY_D     = 9.80665;
inline constexpr f32 GRAVITY       = 9.80665f;
inline constexpr f64 SPEED_OF_LIGHT= 299792458.0;
inline constexpr f64 PLANCK_D      = 6.62607015e-34;

// Common fractions
inline constexpr f32 ONE_THIRD     = 1.0f/3.0f;
inline constexpr f32 TWO_THIRDS    = 2.0f/3.0f;
inline constexpr f32 ONE_SIXTH     = 1.0f/6.0f;
inline constexpr f32 ONE_SEVENTH   = 1.0f/7.0f;
inline constexpr f32 ONE_NINTH     = 1.0f/9.0f;

// Infinities
inline constexpr f32 F32_INF       = NumericLimits<f32>::inf;
inline constexpr f32 F32_NEG_INF   = -NumericLimits<f32>::inf;
inline constexpr f64 F64_INF       = NumericLimits<f64>::inf;

} // namespace qs::math

// ─── Additional math constants ─────────────────────────────────────────────
// Logarithms
inline constexpr f64 LN2          = 0.6931471805599453;    // ln(2)
inline constexpr f64 LN10         = 2.302585092994046;     // ln(10)
inline constexpr f64 LOG2E        = 1.4426950408889634;    // log2(e)
inline constexpr f64 LOG10E       = 0.4342944819032518;    // log10(e)
inline constexpr f64 SQRT2        = 1.4142135623730951;    // sqrt(2)
inline constexpr f64 INV_SQRT2    = 0.7071067811865476;    // 1/sqrt(2)
inline constexpr f64 SQRT3        = 1.7320508075688772;    // sqrt(3)
inline constexpr f64 INV_SQRT3    = 0.5773502691896258;    // 1/sqrt(3)
inline constexpr f64 SQRT5        = 2.2360679774997896;    // sqrt(5)
// Physics (SI)
inline constexpr f64 SPEED_OF_LIGHT   = 299792458.0;        // m/s
inline constexpr f64 PLANCK_H         = 6.62607015e-34;     // J·s
inline constexpr f64 BOLTZMANN_K      = 1.380649e-23;       // J/K
inline constexpr f64 AVOGADRO_N       = 6.02214076e23;      // mol^-1
inline constexpr f64 ELEMENTARY_CHARGE= 1.602176634e-19;    // C
inline constexpr f64 GRAVITATIONAL_G  = 6.674e-11;          // N·m^2/kg^2
inline constexpr f64 EARTH_GRAVITY    = 9.80665;            // m/s^2
// Geometry
inline constexpr f64 INV_PI           = 0.3183098861837907; // 1/pi
inline constexpr f64 INV_TWO_PI       = 0.15915494309189535;// 1/(2pi)
inline constexpr f64 FOUR_PI          = 12.566370614359172; // 4pi
inline constexpr f64 INV_FOUR_PI      = 0.07957747154594767;// 1/(4pi)
inline constexpr f64 PI_SQ            = 9.869604401089358;  // pi^2
inline constexpr f64 GOLDEN_RATIO     = 1.6180339887498949; // phi
inline constexpr f64 PLASTIC_NUMBER   = 1.3247179572447460; // rho
inline constexpr f64 SILVER_RATIO     = 2.4142135623730951; // 1+sqrt(2)
inline constexpr f64 EULER_MASCHERONI = 0.5772156649015329; // gamma
inline constexpr f64 CATALAN_G        = 0.9159655941772190; // Catalan's G
// f32 mirrors
inline constexpr f32 LN2_F            = f32(LN2);
inline constexpr f32 SQRT2_F          = f32(SQRT2);
inline constexpr f32 INV_SQRT2_F      = f32(INV_SQRT2);
inline constexpr f32 INV_PI_F         = f32(INV_PI);
inline constexpr f32 INV_TWO_PI_F     = f32(INV_TWO_PI);
inline constexpr f32 GOLDEN_RATIO_F   = f32(GOLDEN_RATIO);
