#pragma once
// qs_math/celestial_math.hpp — Kepler orbit, ephemeris, stellar coords, eclipse geometry (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"
#include "mat3.hpp"
#include <cmath>
#include <array>
#include <optional>

namespace qs {

// ─── Orbital elements ─────────────────────────────────────────────────────────

struct OrbitalElements {
    f64 semi_major_axis{};     // a (AU)
    f64 eccentricity{};        // e [0,1)
    f64 inclination{};         // i (radians)
    f64 lon_ascending_node{};  // Omega (radians)
    f64 arg_periapsis{};       // omega (radians)
    f64 mean_anomaly_epoch{};  // M0 (radians)
    f64 epoch_jd{};            // Julian date of epoch
    f64 period_years{};        // T
};

// Kepler's equation: M = E - e*sin(E), solve for E given M
[[nodiscard]] inline f64 solve_kepler(f64 M, f64 e, u32 max_iter=50, f64 tol=1e-12) {
    f64 E=M;
    for(u32 i=0;i<max_iter;++i) {
        f64 dE=(M-(E-e*std::sin(E)))/(1.-e*std::cos(E));
        E+=dE;
        if(std::abs(dE)<tol) break;
    }
    return E;
}

// Mean anomaly at time t (Julian date)
[[nodiscard]] inline f64 mean_anomaly(const OrbitalElements& o, f64 jd) {
    f64 dt=(jd-o.epoch_jd)/365.25;
    return std::fmod(o.mean_anomaly_epoch+k_tau_d64*dt/o.period_years, k_tau_d64);
}

// True anomaly from eccentric anomaly
[[nodiscard]] inline f64 true_anomaly(f64 E, f64 e) {
    f64 s=std::sqrt((1.+e)/(1.-e+1e-15))*std::tan(E/2.);
    return 2.*std::atan(s);
}

// Heliocentric ecliptic Cartesian (AU)
[[nodiscard]] inline Vec3d heliocentric_pos(const OrbitalElements& o, f64 jd) {
    f64 M=mean_anomaly(o,jd);
    f64 E=solve_kepler(M,o.eccentricity);
    f64 nu=true_anomaly(E,o.eccentricity);
    f64 r=o.semi_major_axis*(1.-o.eccentricity*std::cos(E));
    // Orbital plane coords
    f64 xo=r*std::cos(nu), yo=r*std::sin(nu);
    // Rotate to ecliptic
    f64 cO=std::cos(o.lon_ascending_node), sO=std::sin(o.lon_ascending_node);
    f64 co=std::cos(o.arg_periapsis),      so=std::sin(o.arg_periapsis);
    f64 ci=std::cos(o.inclination),        si=std::sin(o.inclination);
    return {
        (cO*co-sO*so*ci)*xo+(-cO*so-sO*co*ci)*yo,
        (sO*co+cO*so*ci)*xo+(-sO*so+cO*co*ci)*yo,
        (so*si)*xo+(co*si)*yo};
}

// ─── Julian date ──────────────────────────────────────────────────────────────

[[nodiscard]] inline f64 julian_date(i32 year, i32 month, i32 day,
                                      f64 hour=0., f64 min=0., f64 sec=0.)
{
    if(month<=2){--year;month+=12;}
    i32 A=year/100, B=2-A+A/4;
    return std::floor(365.25*(year+4716))+std::floor(30.6001*(month+1))
          +day+B-1524.5+(hour+min/60.+sec/3600.)/24.;
}

// J2000.0 epoch
static constexpr f64 k_J2000 = 2451545.0;

[[nodiscard]] inline f64 julian_centuries_j2000(f64 jd) { return (jd-k_J2000)/36525.; }

// ─── Equatorial <-> ecliptic ──────────────────────────────────────────────────

static constexpr f64 k_ecliptic_obliquity = 0.40909280; // radians (23.439 deg, J2000)

[[nodiscard]] inline Vec3d ecliptic_to_equatorial(const Vec3d& e) {
    f64 c=std::cos(k_ecliptic_obliquity), s=std::sin(k_ecliptic_obliquity);
    return {e.x, c*e.y-s*e.z, s*e.y+c*e.z};
}

[[nodiscard]] inline Vec3d equatorial_to_ecliptic(const Vec3d& q) {
    f64 c=std::cos(k_ecliptic_obliquity), s=std::sin(k_ecliptic_obliquity);
    return {q.x, c*q.y+s*q.z, -s*q.y+c*q.z};
}

// ─── Right ascension / declination ───────────────────────────────────────────

struct RaDec { f64 ra{}, dec{}; }; // both radians

[[nodiscard]] inline RaDec xyz_to_radec(const Vec3d& v) {
    f64 r=length(v);
    if(r<1e-15) return {};
    return {std::atan2(v.y,v.x), std::asin(v.z/r)};
}

[[nodiscard]] inline Vec3d radec_to_xyz(f64 ra, f64 dec, f64 r=1.) {
    return {r*std::cos(dec)*std::cos(ra), r*std::cos(dec)*std::sin(ra), r*std::sin(dec)};
}

// ─── Horizontal coordinates (alt/az) ─────────────────────────────────────────

struct AltAz { f64 altitude{}, azimuth{}; }; // radians

[[nodiscard]] inline AltAz equatorial_to_horizontal(
    f64 hour_angle, f64 dec, f64 latitude)
{
    f64 cd=std::cos(dec), sd=std::sin(dec);
    f64 cl=std::cos(latitude), sl=std::sin(latitude);
    f64 ch=std::cos(hour_angle), sh=std::sin(hour_angle);
    f64 alt=std::asin(sl*sd+cl*cd*ch);
    f64 az=std::atan2(-sh*cd, cl*sd-sl*cd*ch);
    if(az<0.) az+=k_tau_d64;
    return {alt, az};
}

// Greenwich Mean Sidereal Time (radians)
[[nodiscard]] inline f64 gmst(f64 jd) {
    f64 T=julian_centuries_j2000(jd);
    f64 theta=280.46061837+360.98564736629*(jd-k_J2000)+T*T*(0.000387933-T/38710000.);
    return std::fmod(theta*k_pi_d64/180., k_tau_d64);
}

// Local Sidereal Time
[[nodiscard]] inline f64 lst(f64 jd, f64 longitude) {
    return std::fmod(gmst(jd)+longitude, k_tau_d64);
}

// ─── Solar position (low precision) ──────────────────────────────────────────

struct SolarPosition { f64 ra{}, dec{}; f64 distance_au{}; };

[[nodiscard]] inline SolarPosition solar_position(f64 jd) {
    f64 T=julian_centuries_j2000(jd);
    // Mean longitude and anomaly
    f64 L0=std::fmod(280.46646+36000.76983*T,360.)*k_pi_d64/180.;
    f64 M=std::fmod(357.52911+35999.05029*T,360.)*k_pi_d64/180.;
    // Equation of center
    f64 C=(1.914602-0.004817*T-0.000014*T*T)*std::sin(M)*k_pi_d64/180.
         +(0.019993-0.000101*T)*std::sin(2.*M)*k_pi_d64/180.
         +0.000289*std::sin(3.*M)*k_pi_d64/180.;
    f64 sun_lon=L0+C;
    f64 R=1.000001018*(1.-0.0167086342*std::cos(M)-0.000139589*std::cos(2.*M));
    Vec3d ecl=radec_to_xyz(sun_lon,0.,R);
    Vec3d eq=ecliptic_to_equatorial(ecl);
    auto[ra,dec]=xyz_to_radec(eq);
    return {ra,dec,R};
}

// ─── Moon position (low precision, Chapront) ─────────────────────────────────

[[nodiscard]] inline Vec3d moon_position_ecl(f64 jd) {
    f64 T=julian_centuries_j2000(jd);
    f64 D=std::fmod(297.85036+445267.111480*T,360.)*k_pi_d64/180.;
    f64 M=std::fmod(357.52772+35999.050340*T,360.)*k_pi_d64/180.;
    f64 Mp=std::fmod(134.96298+477198.867398*T,360.)*k_pi_d64/180.;
    f64 F=std::fmod(93.27191+483202.017538*T,360.)*k_pi_d64/180.;
    f64 lon_deg=218.3165+481267.8813*T
        +6.288751*std::sin(Mp)+1.274018*std::sin(2.*D-Mp)
        +0.658309*std::sin(2.*D)+0.213616*std::sin(2.*Mp)
        -0.185596*std::sin(M)-0.114336*std::sin(2.*F);
    f64 lat_deg=5.128122*std::sin(F)+0.280602*std::sin(Mp+F)
        +0.277693*std::sin(Mp-F)+0.173237*std::sin(2.*D-F);
    f64 dist_km=385000.56+20905.355*std::cos(Mp)+3699.111*std::cos(2.*D-Mp)
        +2955.968*std::cos(2.*D)-569.925*std::cos(2.*Mp);
    f64 lon=lon_deg*k_pi_d64/180., lat=lat_deg*k_pi_d64/180.;
    return radec_to_xyz(lon,lat,dist_km/1.495978707e8); // convert to AU
}

// ─── Eclipse detection ────────────────────────────────────────────────────────

struct EclipseInfo {
    bool solar{}, lunar{};
    f32  obscuration{0.f};  // 0=none, 1=total
    f32  magnitude{0.f};
};

[[nodiscard]] inline EclipseInfo check_eclipse(f64 jd) {
    auto sun=solar_position(jd);
    Vec3d sun_eq=radec_to_xyz(sun.ra,sun.dec,sun.distance_au);
    Vec3d moon_ecl=moon_position_ecl(jd);
    Vec3d moon_eq=ecliptic_to_equatorial(moon_ecl);
    f64 sep=std::acos(std::clamp(dot(normalize(sun_eq),normalize(moon_eq)),-1.,1.));
    f64 sun_r=0.00465047/sun.distance_au;  // Sun angular radius (rad)
    f64 moon_r=0.00257/length(moon_eq);    // Moon angular radius
    EclipseInfo e;
    if(sep<sun_r+moon_r){
        e.solar=true;
        e.magnitude=f32((sun_r+moon_r-sep)/(2.*sun_r));
        e.obscuration=std::clamp(f32(1.-(sep/(sun_r+moon_r))*0.5),0.f,1.f);
    }
    return e;
}

// ─── Atmospheric refraction ───────────────────────────────────────────────────

[[nodiscard]] inline f64 atmospheric_refraction(f64 true_alt_deg,
    f64 pressure_mbar=1013.25, f64 temp_celsius=10.)
{
    if(true_alt_deg<-1.) return 0.;
    f64 A=1.02/(std::tan((true_alt_deg+10.3/(true_alt_deg+5.11))*k_pi_d64/180.));
    return A*(pressure_mbar/1010.)*(283./(273.+temp_celsius))/60.;
}

} // namespace qs
