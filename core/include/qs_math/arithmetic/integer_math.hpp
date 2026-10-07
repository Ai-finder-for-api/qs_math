#pragma once
// integer_math.hpp — Integer mathematics: fixed-point, bit manipulation,
//                    integer sqrt/log, population count, bit tricks,
//                    LUT generation, integer SIMD helpers, modular arithmetic
#include "scalar_types.hpp"
#include <bit>

namespace qs::math {

// ---------------------------------------------------------------------------
// Bit manipulation
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u32 bit_reverse32(u32 v) noexcept {
    v=((v>>1)&0x55555555)|((v&0x55555555)<<1);
    v=((v>>2)&0x33333333)|((v&0x33333333)<<2);
    v=((v>>4)&0x0F0F0F0F)|((v&0x0F0F0F0F)<<4);
    v=((v>>8)&0x00FF00FF)|((v&0x00FF00FF)<<8);
    return (v>>16)|(v<<16);
}
[[nodiscard]] constexpr u64 bit_reverse64(u64 v) noexcept {
    return (u64(bit_reverse32(u32(v)))<<32)|bit_reverse32(u32(v>>32));
}

// Extract / deposit bits (BMI2-style)
[[nodiscard]] constexpr u32 pext32(u32 v, u32 mask) noexcept {
    u32 r=0; for(i32 i=0,o=0;i<32;++i) if((mask>>i)&1) r|=((v>>i)&1)<<o++;
    return r;
}
[[nodiscard]] constexpr u32 pdep32(u32 v, u32 mask) noexcept {
    u32 r=0; for(i32 i=0,o=0;i<32;++i) if((mask>>i)&1){ r|=((v>>o)&1)<<i; ++o; }
    return r;
}

// Gray code
[[nodiscard]] constexpr u32 to_gray(u32 n)   noexcept { return n^(n>>1); }
[[nodiscard]] constexpr u32 from_gray(u32 g) noexcept {
    u32 n=g;
    for(u32 m=n>>1;m;m>>=1) n^=m;
    return n;
}

// Integer sqrt (Newton's method, integer)
[[nodiscard]] constexpr u64 isqrt(u64 n) noexcept {
    if(n==0) return 0;
    u64 x=n, y=(x+1)/2;
    while(y<x){ x=y; y=(x+n/x)/2; }
    return x;
}

// Integer log2 (floor)
[[nodiscard]] constexpr i32 ilog2(u64 v) noexcept {
    return v?63-count_leading_zeros(v):0;
}
[[nodiscard]] constexpr i32 ilog2_ceil(u64 v) noexcept {
    i32 r=ilog2(v); return (v&(v-1))?r+1:r;
}

// Integer log10 (floor)
[[nodiscard]] constexpr i32 ilog10(u64 v) noexcept {
    if(v<10) return 0; if(v<100) return 1; if(v<1000) return 2;
    if(v<10000) return 3; if(v<100000) return 4; if(v<1000000) return 5;
    if(v<10000000) return 6; if(v<100000000) return 7; if(v<1000000000) return 8;
    if(v<10000000000ULL) return 9; if(v<100000000000ULL) return 10;
    return 18;
}

// Integer power
[[nodiscard]] constexpr u64 ipow(u64 base, u32 exp) noexcept {
    u64 r=1; while(exp>0){ if(exp&1) r*=base; base*=base; exp>>=1; } return r;
}

// GCD / LCM
[[nodiscard]] constexpr u64 gcd(u64 a, u64 b) noexcept {
    while(b){ u64 t=b; b=a%b; a=t; } return a;
}
[[nodiscard]] constexpr u64 lcm(u64 a, u64 b) noexcept {
    return a/gcd(a,b)*b;
}

// Extended GCD (returns gcd, sets x,y such that ax+by=gcd)
[[nodiscard]] constexpr i64 gcd_extended(i64 a, i64 b, i64& x, i64& y) noexcept {
    if(b==0){x=1;y=0;return a;}
    i64 x1,y1; i64 g=gcd_extended(b,a%b,x1,y1);
    x=y1; y=x1-(a/b)*y1; return g;
}

// Modular arithmetic
[[nodiscard]] constexpr u64 mod_add(u64 a, u64 b, u64 m) noexcept { return (a+b)%m; }
[[nodiscard]] constexpr u64 mod_mul(u64 a, u64 b, u64 m) noexcept {
    return (__uint128_t(a)*b)%m;
}
[[nodiscard]] constexpr u64 mod_pow(u64 base, u64 exp, u64 m) noexcept {
    u64 r=1; base%=m;
    while(exp>0){ if(exp&1) r=mod_mul(r,base,m); base=mod_mul(base,base,m); exp>>=1; }
    return r;
}
[[nodiscard]] constexpr u64 mod_inverse(u64 a, u64 m) noexcept {
    return mod_pow(a,m-2,m); // m must be prime
}

// ---------------------------------------------------------------------------
// Fixed-point arithmetic (Q-format)
// ---------------------------------------------------------------------------
// Q16.16 fixed-point
struct Fixed32 {
    i32 raw{0};
    static constexpr i32 FRAC_BITS = 16;
    static constexpr i32 SCALE = 1<<FRAC_BITS;

    constexpr Fixed32() noexcept = default;
    constexpr explicit Fixed32(i32 r) noexcept : raw(r) {}
    constexpr explicit Fixed32(f32 v) noexcept : raw(i32(v*SCALE)) {}
    constexpr explicit Fixed32(f64 v) noexcept : raw(i32(v*SCALE)) {}

    [[nodiscard]] constexpr f32 to_f32() const noexcept { return f32(raw)/SCALE; }
    [[nodiscard]] constexpr f64 to_f64() const noexcept { return f64(raw)/SCALE; }
    [[nodiscard]] constexpr i32 to_int() const noexcept { return raw>>FRAC_BITS; }

    constexpr Fixed32 operator+(Fixed32 o) const noexcept { return Fixed32{raw+o.raw}; }
    constexpr Fixed32 operator-(Fixed32 o) const noexcept { return Fixed32{raw-o.raw}; }
    constexpr Fixed32 operator*(Fixed32 o) const noexcept {
        return Fixed32{i32((i64(raw)*o.raw)>>FRAC_BITS)};
    }
    constexpr Fixed32 operator/(Fixed32 o) const noexcept {
        return Fixed32{i32((i64(raw)<<FRAC_BITS)/o.raw)};
    }
    constexpr Fixed32 operator-() const noexcept { return Fixed32{-raw}; }
    constexpr bool operator==(Fixed32 o) const noexcept { return raw==o.raw; }
    constexpr bool operator< (Fixed32 o) const noexcept { return raw< o.raw; }
    constexpr bool operator> (Fixed32 o) const noexcept { return raw> o.raw; }
    constexpr bool operator<=(Fixed32 o) const noexcept { return raw<=o.raw; }
    constexpr bool operator>=(Fixed32 o) const noexcept { return raw>=o.raw; }

    static constexpr Fixed32 zero()  noexcept { return Fixed32{0}; }
    static constexpr Fixed32 one()   noexcept { return Fixed32{SCALE}; }
    static constexpr Fixed32 half()  noexcept { return Fixed32{SCALE/2}; }
    static constexpr Fixed32 pi()    noexcept { return Fixed32{f32(PI)}; }
};

// Q8.24 for higher precision
struct Fixed64 {
    i64 raw{0};
    static constexpr i32 FRAC_BITS = 24;
    static constexpr i64 SCALE = 1LL<<FRAC_BITS;

    constexpr Fixed64() noexcept = default;
    constexpr explicit Fixed64(i64 r) noexcept : raw(r) {}
    constexpr explicit Fixed64(f64 v) noexcept : raw(i64(v*SCALE)) {}

    [[nodiscard]] constexpr f64 to_f64() const noexcept { return f64(raw)/f64(SCALE); }
    [[nodiscard]] constexpr i64 to_int() const noexcept { return raw>>FRAC_BITS; }

    constexpr Fixed64 operator+(Fixed64 o) const noexcept { return Fixed64{raw+o.raw}; }
    constexpr Fixed64 operator-(Fixed64 o) const noexcept { return Fixed64{raw-o.raw}; }
    constexpr Fixed64 operator*(Fixed64 o) const noexcept {
        return Fixed64{i64((__int128_t(raw)*o.raw)>>FRAC_BITS)};
    }
    constexpr Fixed64 operator/(Fixed64 o) const noexcept {
        return Fixed64{i64(((__int128_t(raw))<<FRAC_BITS)/o.raw)};
    }
    constexpr bool operator< (Fixed64 o) const noexcept { return raw< o.raw; }
    constexpr bool operator> (Fixed64 o) const noexcept { return raw> o.raw; }
};

// ---------------------------------------------------------------------------
// Integer vector helpers
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr i32 dot_i32(i32 ax,i32 ay,i32 az, i32 bx,i32 by,i32 bz) noexcept {
    return ax*bx+ay*by+az*bz;
}
[[nodiscard]] constexpr i64 cross_z_i32(i32 ax,i32 ay,i32 bx,i32 by) noexcept {
    return i64(ax)*by-i64(ay)*bx;
}

// Bresenham's line
inline i32 bresenham_line(i32 x0,i32 y0,i32 x1,i32 y1,
                           Vec2i* out, i32 max_pts) noexcept {
    i32 n=0;
    i32 dx=abs(x1-x0), sx=x0<x1?1:-1;
    i32 dy=-abs(y1-y0), sy=y0<y1?1:-1;
    i32 err=dx+dy;
    while(n<max_pts){
        out[n++]={x0,y0};
        if(x0==x1&&y0==y1) break;
        i32 e2=2*err;
        if(e2>=dy){ err+=dy; x0+=sx; }
        if(e2<=dx){ err+=dx; y0+=sy; }
    }
    return n;
}

// DDA (integer grid) traversal
inline i32 dda_line(i32 x0,i32 y0,i32 x1,i32 y1,
                    Vec2i* out, i32 max_pts) noexcept {
    i32 dx=x1-x0, dy=y1-y0;
    i32 steps=max(abs(dx),abs(dy));
    if(steps==0){if(max_pts>0){out[0]={x0,y0};return 1;}return 0;}
    f32 xi=f32(dx)/steps, yi=f32(dy)/steps;
    f32 x=f32(x0)+0.5f, y=f32(y0)+0.5f;
    i32 n=0;
    for(i32 i=0;i<=steps&&n<max_pts;++i,x+=xi,y+=yi)
        out[n++]={i32(x),i32(y)};
    return n;
}

// ---------------------------------------------------------------------------
// Hash and checksum
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u32 crc32_byte(u32 crc, u8 byte) noexcept {
    crc^=byte;
    for(i32 i=0;i<8;++i) crc=(crc&1)?(crc>>1)^0xEDB88320U:(crc>>1);
    return crc;
}
[[nodiscard]] constexpr u32 crc32(const u8* data, u64 len) noexcept {
    u32 crc=0xFFFFFFFF;
    for(u64 i=0;i<len;++i) crc=crc32_byte(crc,data[i]);
    return ~crc;
}

// MurmurHash3 (32-bit)
[[nodiscard]] inline u32 murmur3_32(const u8* data, u64 len, u32 seed=0) noexcept {
    u32 h=seed;
    u64 nblocks=len/4;
    for(u64 i=0;i<nblocks;++i){
        u32 k; __builtin_memcpy(&k,data+i*4,4);
        k*=0xcc9e2d51; k=(k<<15)|(k>>17); k*=0x1b873593;
        h^=k; h=(h<<13)|(h>>19); h=h*5+0xe6546b64;
    }
    u32 k=0;
    switch(len&3){
        case 3: k|=data[nblocks*4+2]<<16; [[fallthrough]];
        case 2: k|=data[nblocks*4+1]<<8;  [[fallthrough]];
        case 1: k|=data[nblocks*4]; k*=0xcc9e2d51; k=(k<<15)|(k>>17); k*=0x1b873593; h^=k;
    }
    h^=u32(len); h^=h>>16; h*=0x85ebca6b; h^=h>>13; h*=0xc2b2ae35; h^=h>>16;
    return h;
}

// FNV-1a (64-bit) — fast non-cryptographic
[[nodiscard]] constexpr u64 fnv1a(const u8* data, u64 len) noexcept {
    u64 h=14695981039346656037ULL;
    for(u64 i=0;i<len;++i){ h^=data[i]; h*=1099511628211ULL; }
    return h;
}

// xxHash32 (simplified)
[[nodiscard]] inline u32 xxhash32(const u8* data, u64 len, u32 seed=0) noexcept {
    static constexpr u32 P1=2654435761U,P2=2246822519U,P3=3266489917U,P4=668265263U,P5=374761393U;
    u32 h;
    if(len>=16){
        u32 v1=seed+P1+P2,v2=seed+P2,v3=seed,v4=seed-P1;
        const u8* end=data+len-16;
        while(data<=end){
            u32 k; 
            __builtin_memcpy(&k,data,4); v1+=k*P2; v1=(v1<<13)|(v1>>19); v1*=P1; data+=4;
            __builtin_memcpy(&k,data,4); v2+=k*P2; v2=(v2<<13)|(v2>>19); v2*=P1; data+=4;
            __builtin_memcpy(&k,data,4); v3+=k*P2; v3=(v3<<13)|(v3>>19); v3*=P1; data+=4;
            __builtin_memcpy(&k,data,4); v4+=k*P2; v4=(v4<<13)|(v4>>19); v4*=P1; data+=4;
        }
        h=((v1<<1)|(v1>>31))+((v2<<7)|(v2>>25))+((v3<<12)|(v3>>20))+((v4<<18)|(v4>>14));
    } else h=seed+P5;
    h+=u32(len);
    while(data+4<=(u8*)(data+len-1)+1){
        u32 k; __builtin_memcpy(&k,data,4);
        h+=k*P3; h=((h<<17)|(h>>15))*P4; data+=4;
    }
    while(data<(u8*)(data+len)){ h+=(*data++)*P5; h=((h<<11)|(h>>21))*P1; }
    h^=h>>15; h*=P2; h^=h>>13; h*=P3; h^=h>>16;
    return h;
}

// ---------------------------------------------------------------------------
// Integer sequences
// ---------------------------------------------------------------------------
// Fibonacci sequence (first 94 fit in u64)
[[nodiscard]] constexpr u64 fibonacci(i32 n) noexcept {
    if(n<=0) return 0; if(n==1) return 1;
    u64 a=0,b=1;
    for(i32 i=2;i<=n&&i<94;++i){ u64 c=a+b; a=b; b=c; }
    return b;
}

// Pascal's triangle (n choose k)
[[nodiscard]] constexpr u64 binomial(i32 n, i32 k) noexcept {
    if(k<0||k>n) return 0;
    if(k==0||k==n) return 1;
    if(k>n-k) k=n-k;
    u64 r=1;
    for(i32 i=0;i<k;++i){ r*=n-i; r/=i+1; }
    return r;
}

// Catalan numbers
[[nodiscard]] constexpr u64 catalan(i32 n) noexcept {
    return binomial(2*n,n)/(n+1);
}

// Stirling numbers of second kind
[[nodiscard]] constexpr i64 stirling2(i32 n, i32 k) noexcept {
    if(k==0) return n==0?1:0;
    if(k==n) return 1;
    if(k>n) return 0;
    // Explicit formula
    i64 r=0;
    for(i32 j=0;j<=k;++j){
        i64 sign=(j%2==0)?1:-1;
        r+=sign*binomial(k,j)*ipow(k-j,n);
    }
    for(i32 i=2;i<=k;++i) r/=i; // divide by k!
    return r;
}

} // namespace qs::math
