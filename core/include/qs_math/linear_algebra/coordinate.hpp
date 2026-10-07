#pragma once
// qs_math/coordinate.hpp — Cartesian, Polar, Cylindrical, Spherical, Geographic
#include "vec2.hpp"
#include "vec3.hpp"
#include "trig.hpp"
#include "scalar_math.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// 2D: Cartesian <-> Polar
// ---------------------------------------------------------------------------
struct Polar2 {
    f32 r{0.f}, theta{0.f}; // theta in radians
    constexpr Polar2() noexcept = default;
    constexpr Polar2(f32 r, f32 theta) noexcept : r(r), theta(theta) {}

    [[nodiscard]] static Polar2 from_cartesian(Vec2f v) noexcept {
        return {v.length(), atan2(v.y,v.x)};
    }
    [[nodiscard]] Vec2f to_cartesian() const noexcept {
        return {r*cos(theta), r*sin(theta)};
    }
};

// ---------------------------------------------------------------------------
// 3D: Cartesian <-> Cylindrical (r, phi, z)
// ---------------------------------------------------------------------------
struct Cylindrical {
    f32 r{0.f}, phi{0.f}, z{0.f}; // phi in radians
    constexpr Cylindrical() noexcept = default;
    constexpr Cylindrical(f32 r, f32 phi, f32 z) noexcept : r(r),phi(phi),z(z) {}

    [[nodiscard]] static Cylindrical from_cartesian(Vec3f v) noexcept {
        return {sqrt(v.x*v.x+v.y*v.y), atan2(v.y,v.x), v.z};
    }
    [[nodiscard]] Vec3f to_cartesian() const noexcept {
        return {r*cos(phi), r*sin(phi), z};
    }
};

// ---------------------------------------------------------------------------
// 3D: Cartesian <-> Spherical (r, theta, phi) — physics convention
//     theta = polar angle from +Z [0,PI], phi = azimuth [0,2PI]
// ---------------------------------------------------------------------------
struct Spherical {
    f32 r{0.f}, theta{0.f}, phi{0.f};
    constexpr Spherical() noexcept = default;
    constexpr Spherical(f32 r, f32 theta, f32 phi) noexcept : r(r),theta(theta),phi(phi) {}

    [[nodiscard]] static Spherical from_cartesian(Vec3f v) noexcept {
        f32 r=v.length();
        if(r<EPS_LENGTH) return {};
        return {r, safe_acos(v.z/r), atan2(v.y,v.x)};
    }
    [[nodiscard]] Vec3f to_cartesian() const noexcept {
        f32 st=sin(theta);
        return {r*st*cos(phi), r*st*sin(phi), r*cos(theta)};
    }
    // Hemisphere sampling (cosine-weighted)
    [[nodiscard]] static Spherical hemisphere_sample(f32 u1, f32 u2) noexcept {
        f32 r=sqrt(u1);
        f32 theta=TWO_PI*u2;
        return {1.f, safe_acos(sqrt(max(0.f,1.f-u1))), theta};
    }
};

// ---------------------------------------------------------------------------
// Geographic: Lat/Lon <-> ECEF (Earth-Centered Earth-Fixed)
// ---------------------------------------------------------------------------
struct LatLon {
    f64 lat{0.0}, lon{0.0}; // degrees
    f64 alt{0.0};           // meters above ellipsoid
    constexpr LatLon() noexcept = default;
    constexpr LatLon(f64 lat, f64 lon, f64 alt=0.0) noexcept : lat(lat),lon(lon),alt(alt) {}
};

struct ECEF {
    f64 x{0.0},y{0.0},z{0.0};
    constexpr ECEF() noexcept = default;
    constexpr ECEF(f64 x,f64 y,f64 z) noexcept : x(x),y(y),z(z) {}
};

// WGS84 constants
inline constexpr f64 WGS84_A   = 6378137.0;          // semi-major axis (m)
inline constexpr f64 WGS84_F   = 1.0/298.257223563;  // flattening
inline constexpr f64 WGS84_E2  = 2*WGS84_F-WGS84_F*WGS84_F;

[[nodiscard]] inline ECEF latlon_to_ecef(LatLon ll) noexcept {
    f64 lat_r=ll.lat*DEG_TO_RAD_D, lon_r=ll.lon*DEG_TO_RAD_D;
    f64 slat=std::sin(lat_r), clat=std::cos(lat_r);
    f64 slon=std::sin(lon_r), clon=std::cos(lon_r);
    f64 N=WGS84_A/std::sqrt(1.0-WGS84_E2*slat*slat);
    return {(N+ll.alt)*clat*clon, (N+ll.alt)*clat*slon, (N*(1.0-WGS84_E2)+ll.alt)*slat};
}

[[nodiscard]] inline LatLon ecef_to_latlon(ECEF p) noexcept {
    // Bowring's method
    f64 lon=std::atan2(p.y,p.x);
    f64 b=WGS84_A*(1.0-WGS84_F);
    f64 ep2=(WGS84_A*WGS84_A-b*b)/(b*b);
    f64 D=std::sqrt(p.x*p.x+p.y*p.y);
    f64 theta=std::atan2(p.z*WGS84_A,D*b);
    f64 lat=std::atan2(p.z+ep2*b*std::pow(std::sin(theta),3.0),
                       D-WGS84_E2*WGS84_A*std::pow(std::cos(theta),3.0));
    f64 slat=std::sin(lat);
    f64 N=WGS84_A/std::sqrt(1.0-WGS84_E2*slat*slat);
    f64 alt=D/std::cos(lat)-N;
    return {lat*RAD_TO_DEG_D, lon*RAD_TO_DEG_D, alt};
}

// Great-circle distance (Haversine)
[[nodiscard]] inline f64 haversine_distance_m(LatLon a, LatLon b) noexcept {
    constexpr f64 R=6371000.0;
    f64 dlat=(b.lat-a.lat)*DEG_TO_RAD_D, dlon=(b.lon-a.lon)*DEG_TO_RAD_D;
    f64 slat=std::sin(dlat*0.5), slon=std::sin(dlon*0.5);
    f64 h=slat*slat+std::cos(a.lat*DEG_TO_RAD_D)*std::cos(b.lat*DEG_TO_RAD_D)*slon*slon;
    return 2.0*R*std::asin(std::sqrt(h));
}

} // namespace qs::math
