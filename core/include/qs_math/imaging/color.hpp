#pragma once
// qs_math/color.hpp — Mathematical color types: RGB, RGBA, HSV, HSL, XYZ, LAB, OKLAB, OKLCH, CMYK
#include "vec3.hpp"
#include "vec4.hpp"
#include "scalar_math.hpp"
#include "trig.hpp"

namespace qs::math {

// ---------------------------------------------------------------------------
// RGB / RGBA
// ---------------------------------------------------------------------------
struct ColorRGB {
    f32 r{0.f}, g{0.f}, b{0.f};
    constexpr ColorRGB() noexcept = default;
    constexpr ColorRGB(f32 r,f32 g,f32 b) noexcept : r(r),g(g),b(b) {}
    constexpr explicit ColorRGB(Vec3f v) noexcept : r(v.x),g(v.y),b(v.z) {}
    [[nodiscard]] constexpr Vec3f to_vec3() const noexcept { return {r,g,b}; }
    [[nodiscard]] constexpr ColorRGB clamped() const noexcept { return {saturate(r),saturate(g),saturate(b)}; }
    constexpr ColorRGB operator+(ColorRGB o) const noexcept { return {r+o.r,g+o.g,b+o.b}; }
    constexpr ColorRGB operator*(f32 s)      const noexcept { return {r*s,g*s,b*s}; }
    constexpr ColorRGB operator*(ColorRGB o) const noexcept { return {r*o.r,g*o.g,b*o.b}; }
    [[nodiscard]] constexpr f32 luminance()  const noexcept { return 0.2126f*r+0.7152f*g+0.0722f*b; }
    static constexpr ColorRGB black()  noexcept { return {0,0,0}; }
    static constexpr ColorRGB white()  noexcept { return {1,1,1}; }
    static constexpr ColorRGB red()    noexcept { return {1,0,0}; }
    static constexpr ColorRGB green()  noexcept { return {0,1,0}; }
    static constexpr ColorRGB blue()   noexcept { return {0,0,1}; }
};

struct ColorRGBA {
    f32 r{0.f}, g{0.f}, b{0.f}, a{1.f};
    constexpr ColorRGBA() noexcept = default;
    constexpr ColorRGBA(f32 r,f32 g,f32 b,f32 a=1.f) noexcept : r(r),g(g),b(b),a(a) {}
    constexpr ColorRGBA(ColorRGB c,f32 a=1.f) noexcept : r(c.r),g(c.g),b(c.b),a(a) {}
    constexpr explicit ColorRGBA(Vec4f v) noexcept : r(v.x),g(v.y),b(v.z),a(v.w) {}
    [[nodiscard]] constexpr Vec4f    to_vec4() const noexcept { return {r,g,b,a}; }
    [[nodiscard]] constexpr ColorRGB rgb()     const noexcept { return {r,g,b}; }
    [[nodiscard]] constexpr ColorRGBA premultiplied() const noexcept { return {r*a,g*a,b*a,a}; }
    static constexpr ColorRGBA transparent() noexcept { return {0,0,0,0}; }
    static constexpr ColorRGBA white()       noexcept { return {1,1,1,1}; }
};

// ---------------------------------------------------------------------------
// Linear <-> Gamma
// ---------------------------------------------------------------------------
[[nodiscard]] inline f32 linear_to_srgb(f32 v) noexcept {
    return v<=0.0031308f ? v*12.92f : 1.055f*pow(v,1.f/2.4f)-0.055f;
}
[[nodiscard]] inline f32 srgb_to_linear(f32 v) noexcept {
    return v<=0.04045f ? v/12.92f : pow((v+0.055f)/1.055f,2.4f);
}
[[nodiscard]] inline ColorRGB to_linear(ColorRGB c) noexcept {
    return {srgb_to_linear(c.r),srgb_to_linear(c.g),srgb_to_linear(c.b)};
}
[[nodiscard]] inline ColorRGB to_srgb(ColorRGB c) noexcept {
    return {linear_to_srgb(c.r),linear_to_srgb(c.g),linear_to_srgb(c.b)};
}

// ---------------------------------------------------------------------------
// HSV
// ---------------------------------------------------------------------------
struct ColorHSV {
    f32 h{0.f},s{0.f},v{0.f}; // h in [0,360), s,v in [0,1]
    constexpr ColorHSV() noexcept = default;
    constexpr ColorHSV(f32 h,f32 s,f32 v) noexcept : h(h),s(s),v(v) {}
    [[nodiscard]] ColorRGB to_rgb() const noexcept {
        if(s<=0.f) return {v,v,v};
        f32 hh=wrap(h,0.f,360.f)/60.f;
        i32 i=(i32)hh; f32 ff=hh-(f32)i;
        f32 p=v*(1.f-s),q=v*(1.f-(s*ff)),t=v*(1.f-(s*(1.f-ff)));
        switch(i){
            case 0: return {v,t,p}; case 1: return {q,v,p};
            case 2: return {p,v,t}; case 3: return {p,q,v};
            case 4: return {t,p,v}; default: return {v,p,q};
        }
    }
    static ColorHSV from_rgb(ColorRGB c) noexcept {
        f32 mx=max3(c.r,c.g,c.b), mn=min3(c.r,c.g,c.b), d=mx-mn;
        f32 h=0.f,s=mx<EPS_F32?0.f:d/mx, v=mx;
        if(d>EPS_F32){
            if(mx==c.r) h=60.f*frac((c.g-c.b)/d/6.f);
            else if(mx==c.g) h=60.f*((c.b-c.r)/d+2.f);
            else             h=60.f*((c.r-c.g)/d+4.f);
        }
        return {h<0.f?h+360.f:h,s,v};
    }
};

// ---------------------------------------------------------------------------
// HSL
// ---------------------------------------------------------------------------
struct ColorHSL {
    f32 h{0.f},s{0.f},l{0.f};
    constexpr ColorHSL() noexcept = default;
    constexpr ColorHSL(f32 h,f32 s,f32 l) noexcept : h(h),s(s),l(l) {}
    [[nodiscard]] ColorRGB to_rgb() const noexcept {
        if(s<=0.f) return {l,l,l};
        auto hue=[](f32 p,f32 q,f32 t)->f32{
            if(t<0.f)t+=1.f; if(t>1.f)t-=1.f;
            if(t<1.f/6.f) return p+(q-p)*6.f*t;
            if(t<0.5f) return q;
            if(t<2.f/3.f) return p+(q-p)*(2.f/3.f-t)*6.f;
            return p;
        };
        f32 q=l<0.5f?l*(1.f+s):l+s-l*s, p=2.f*l-q;
        f32 hn=h/360.f;
        return {hue(p,q,hn+1.f/3.f),hue(p,q,hn),hue(p,q,hn-1.f/3.f)};
    }
    static ColorHSL from_rgb(ColorRGB c) noexcept {
        f32 mx=max3(c.r,c.g,c.b),mn=min3(c.r,c.g,c.b);
        f32 l=(mx+mn)*0.5f, s=0.f, h=0.f;
        if(mx!=mn){ f32 d=mx-mn;
            s=l>0.5f?d/(2.f-mx-mn):d/(mx+mn);
            if(mx==c.r) h=60.f*frac((c.g-c.b)/d/6.f);
            else if(mx==c.g) h=60.f*((c.b-c.r)/d+2.f);
            else             h=60.f*((c.r-c.g)/d+4.f);
            if(h<0.f) h+=360.f;
        }
        return {h,s,l};
    }
};

// ---------------------------------------------------------------------------
// CIE XYZ (D65 illuminant)
// ---------------------------------------------------------------------------
struct ColorXYZ {
    f32 x{0.f},y{0.f},z{0.f};
    constexpr ColorXYZ() noexcept = default;
    constexpr ColorXYZ(f32 x,f32 y,f32 z) noexcept : x(x),y(y),z(z) {}
    static ColorXYZ from_linear_rgb(ColorRGB c) noexcept {
        return {
            0.4124564f*c.r+0.3575761f*c.g+0.1804375f*c.b,
            0.2126729f*c.r+0.7151522f*c.g+0.0721750f*c.b,
            0.0193339f*c.r+0.1191920f*c.g+0.9503041f*c.b
        };
    }
    [[nodiscard]] ColorRGB to_linear_rgb() const noexcept {
        return {
             3.2404542f*x-1.5371385f*y-0.4985314f*z,
            -0.9692660f*x+1.8760108f*y+0.0415560f*z,
             0.0556434f*x-0.2040259f*y+1.0572252f*z
        };
    }
};

// ---------------------------------------------------------------------------
// CIE LAB
// ---------------------------------------------------------------------------
struct ColorLAB {
    f32 L{0.f},a{0.f},b{0.f};
    constexpr ColorLAB() noexcept = default;
    constexpr ColorLAB(f32 L,f32 a,f32 b) noexcept : L(L),a(a),b(b) {}
    static ColorLAB from_xyz(ColorXYZ c) noexcept {
        auto f=[](f32 t)->f32{ return t>0.008856f?cbrt(t):(7.787f*t+16.f/116.f); };
        f32 fx=f(c.x/0.95047f),fy=f(c.y),fz=f(c.z/1.08883f);
        return {116.f*fy-16.f,500.f*(fx-fy),200.f*(fy-fz)};
    }
    [[nodiscard]] ColorXYZ to_xyz() const noexcept {
        f32 fy=(L+16.f)/116.f, fx=a/500.f+fy, fz=fy-b/200.f;
        auto f=[](f32 t)->f32{ return t>0.206897f?t*t*t:(t-16.f/116.f)/7.787f; };
        return {f(fx)*0.95047f, f(fy), f(fz)*1.08883f};
    }
    // Delta-E 2000 color difference
    [[nodiscard]] f32 delta_e(ColorLAB o) const noexcept {
        f32 dL=o.L-L, C1=sqrt(a*a+b*b), C2=sqrt(o.a*o.a+o.b*o.b);
        f32 dC=C2-C1, dH2=(o.a-a)*(o.a-a)+(o.b-b)*(o.b-b)-dC*dC;
        f32 dH=dH2>0.f?sqrt(dH2):0.f;
        f32 SL=1.f+0.015f*((L+o.L)*0.5f-50.f)*((L+o.L)*0.5f-50.f)/
               sqrt(20.f+((L+o.L)*0.5f-50.f)*((L+o.L)*0.5f-50.f));
        f32 SC=1.f+0.045f*(C1+C2)*0.5f, SH=1.f+0.015f*(C1+C2)*0.5f;
        return sqrt((dL/SL)*(dL/SL)+(dC/SC)*(dC/SC)+(dH/SH)*(dH/SH));
    }
};

// ---------------------------------------------------------------------------
// OKLAB (perceptual, Björn Ottosson 2020)
// ---------------------------------------------------------------------------
struct ColorOKLAB {
    f32 L{0.f},a{0.f},b{0.f};
    constexpr ColorOKLAB() noexcept = default;
    constexpr ColorOKLAB(f32 L,f32 a,f32 b) noexcept : L(L),a(a),b(b) {}
    static ColorOKLAB from_linear_rgb(ColorRGB c) noexcept {
        f32 l=0.4122214708f*c.r+0.5363325363f*c.g+0.0514459929f*c.b;
        f32 m=0.2119034982f*c.r+0.6806995451f*c.g+0.1073969566f*c.b;
        f32 s=0.0883024619f*c.r+0.2817188376f*c.g+0.6299787005f*c.b;
        f32 l_=cbrt(l),m_=cbrt(m),s_=cbrt(s);
        return {
            0.2104542553f*l_+0.7936177850f*m_-0.0040720468f*s_,
            1.9779984951f*l_-2.4285922050f*m_+0.4505937099f*s_,
            0.0259040371f*l_+0.7827717662f*m_-0.8086757660f*s_
        };
    }
    [[nodiscard]] ColorRGB to_linear_rgb() const noexcept {
        f32 l_=L+0.3963377774f*a+0.2158037573f*b;
        f32 m_=L-0.1055613458f*a-0.0638541728f*b;
        f32 s_=L-0.0894841775f*a-1.2914855480f*b;
        f32 l=l_*l_*l_,m=m_*m_*m_,s=s_*s_*s_;
        return {
             4.0767416621f*l-3.3077115913f*m+0.2309699292f*s,
            -1.2684380046f*l+2.6097574011f*m-0.3413193965f*s,
            -0.0041960863f*l-0.7034186147f*m+1.7076147010f*s
        };
    }
};

// ---------------------------------------------------------------------------
// OKLCH (cylindrical OKLAB)
// ---------------------------------------------------------------------------
struct ColorOKLCH {
    f32 L{0.f},C{0.f},h{0.f}; // h in radians
    constexpr ColorOKLCH() noexcept = default;
    constexpr ColorOKLCH(f32 L,f32 C,f32 h) noexcept : L(L),C(C),h(h) {}
    static ColorOKLCH from_oklab(ColorOKLAB c) noexcept {
        return {c.L, sqrt(c.a*c.a+c.b*c.b), atan2(c.b,c.a)};
    }
    [[nodiscard]] ColorOKLAB to_oklab() const noexcept {
        return {L, C*cos(h), C*sin(h)};
    }
    static ColorOKLCH from_linear_rgb(ColorRGB c) noexcept {
        return from_oklab(ColorOKLAB::from_linear_rgb(c));
    }
    [[nodiscard]] ColorRGB to_linear_rgb() const noexcept { return to_oklab().to_linear_rgb(); }
};

// ---------------------------------------------------------------------------
// CMYK
// ---------------------------------------------------------------------------
struct ColorCMYK {
    f32 c{0.f},m{0.f},y{0.f},k{0.f};
    constexpr ColorCMYK() noexcept = default;
    constexpr ColorCMYK(f32 c,f32 m,f32 y,f32 k) noexcept : c(c),m(m),y(y),k(k) {}
    static ColorCMYK from_rgb(ColorRGB rgb) noexcept {
        f32 k=1.f-max3(rgb.r,rgb.g,rgb.b);
        if(1.f-k<EPS_F32) return {0,0,0,1};
        f32 inv=1.f/(1.f-k);
        return {(1.f-rgb.r-k)*inv,(1.f-rgb.g-k)*inv,(1.f-rgb.b-k)*inv,k};
    }
    [[nodiscard]] ColorRGB to_rgb() const noexcept {
        return {(1.f-c)*(1.f-k),(1.f-m)*(1.f-k),(1.f-y)*(1.f-k)};
    }
};

// ---------------------------------------------------------------------------
// Color Temperature (blackbody approximation)
// ---------------------------------------------------------------------------
[[nodiscard]] inline ColorRGB blackbody_to_rgb(f32 kelvin) noexcept {
    kelvin=clamp(kelvin,1000.f,40000.f);
    f32 t=kelvin/100.f;
    f32 r,g,b;
    if(t<=66.f) {
        r=255.f;
        g=99.4708025861f*log(t)-161.1195681661f;
    } else {
        r=329.698727446f*pow(t-60.f,-0.1332047592f);
        g=288.1221695283f*pow(t-60.f,-0.0755148492f);
    }
    if(t>=66.f) b=255.f;
    else if(t<=19.f) b=0.f;
    else b=138.5177312231f*log(t-10.f)-305.0447927307f;
    return {saturate(r/255.f),saturate(g/255.f),saturate(b/255.f)};
}

// ---------------------------------------------------------------------------
// Color interpolation
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr ColorRGB lerp(ColorRGB a, ColorRGB b, f32 t) noexcept {
    return {qs::math::lerp(a.r,b.r,t),qs::math::lerp(a.g,b.g,t),qs::math::lerp(a.b,b.b,t)};
}
[[nodiscard]] constexpr ColorRGBA lerp(ColorRGBA a, ColorRGBA b, f32 t) noexcept {
    return {qs::math::lerp(a.r,b.r,t),qs::math::lerp(a.g,b.g,t),
            qs::math::lerp(a.b,b.b,t),qs::math::lerp(a.a,b.a,t)};
}
// Perceptual lerp in OKLAB space
[[nodiscard]] inline ColorRGB lerp_oklab(ColorRGB a, ColorRGB b, f32 t) noexcept {
    auto la=ColorOKLAB::from_linear_rgb(to_linear(a));
    auto lb=ColorOKLAB::from_linear_rgb(to_linear(b));
    ColorOKLAB lc{qs::math::lerp(la.L,lb.L,t),qs::math::lerp(la.a,lb.a,t),qs::math::lerp(la.b,lb.b,t)};
    return to_srgb(lc.to_linear_rgb());
}

// 32-bit packed RGBA (0xAARRGGBB)
[[nodiscard]] constexpr u32 pack_rgba(ColorRGBA c) noexcept {
    return (u32(saturate(c.a)*255.f)<<24)|(u32(saturate(c.r)*255.f)<<16)|
           (u32(saturate(c.g)*255.f)<<8)| u32(saturate(c.b)*255.f);
}
[[nodiscard]] constexpr ColorRGBA unpack_rgba(u32 v) noexcept {
    return {(v>>16&0xFF)/255.f,(v>>8&0xFF)/255.f,(v&0xFF)/255.f,(v>>24)/255.f};
}

} // namespace qs::math
