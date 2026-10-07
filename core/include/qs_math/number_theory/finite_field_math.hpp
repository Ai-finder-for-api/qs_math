#pragma once
// qs_math/finite_field_math.hpp — GF(p), GF(2^n), LFSR, BCH/Reed-Solomon syndrome (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <cassert>
#include <optional>
#include <numeric>

namespace qs {

// ─── GF(p) — prime field arithmetic ──────────────────────────────────────────

struct GFp {
    u64 p{};  // prime modulus

    [[nodiscard]] u64 add(u64 a, u64 b) const { return (a+b)%p; }
    [[nodiscard]] u64 sub(u64 a, u64 b) const { return (a+p-b%p)%p; }
    [[nodiscard]] u64 mul(u64 a, u64 b) const { return (a%p)*(b%p)%p; }
    [[nodiscard]] u64 neg(u64 a)        const { return (p-a%p)%p; }

    // Modular exponentiation
    [[nodiscard]] u64 pow(u64 base, u64 exp) const {
        u64 r=1; base%=p;
        while(exp>0){if(exp&1)r=r*base%p; base=base*base%p; exp>>=1;}
        return r;
    }

    // Modular inverse via Fermat's little theorem (p prime)
    [[nodiscard]] u64 inv(u64 a) const { return pow(a,p-2); }
    [[nodiscard]] u64 div(u64 a, u64 b) const { return mul(a,inv(b)); }

    // Extended Euclidean inverse
    [[nodiscard]] u64 inv_ext(u64 a) const {
        i64 t=0,newt=1,r=i64(p),newr=i64(a);
        while(newr){i64 q=r/newr;i64 tmp=t-q*newt;t=newt;newt=tmp;tmp=r-q*newr;r=newr;newr=tmp;}
        if(t<0) t+=i64(p);
        return u64(t);
    }

    // Tonelli-Shanks square root
    [[nodiscard]] std::optional<u64> sqrt(u64 n) const {
        n%=p;
        if(n==0) return 0;
        if(pow(n,(p-1)/2)!=1) return std::nullopt; // not a QR
        if(p%4==3) return pow(n,(p+1)/4);
        // Factor out 2s from p-1
        u64 Q=p-1, S=0;
        while(Q%2==0){Q/=2;++S;}
        // Find quadratic non-residue z
        u64 z=2;
        while(pow(z,(p-1)/2)==1) ++z;
        u64 M=S,c=pow(z,Q),t=pow(n,Q),R=pow(n,(Q+1)/2);
        while(true){
            if(t==1) return R;
            u64 i=0,tmp=t;
            while(tmp!=1){tmp=tmp*tmp%p;++i;}
            u64 b=pow(c,u64(1)<<(M-i-1));
            M=i; c=b*b%p; t=t*c%p; R=R*b%p;
        }
    }
};

// ─── GF(2) polynomial (bitwise) ───────────────────────────────────────────────

// GF(2)[x] polynomial stored as bitmask
[[nodiscard]] inline u64 gf2_poly_mul(u64 a, u64 b) {
    u64 r=0;
    while(b){if(b&1)r^=a; a<<=1; b>>=1;}
    return r;
}

[[nodiscard]] inline u64 gf2_poly_mod(u64 a, u64 m) {
    if(m==0) return a;
    int dm=63-__builtin_clzll(m);
    while(a>=(u64(1)<<dm)){
        int da=63-__builtin_clzll(a);
        a^=(m<<(da-dm));
    }
    return a;
}

[[nodiscard]] inline u64 gf2_poly_gcd(u64 a, u64 b) {
    while(b){u64 r=gf2_poly_mod(a,b);a=b;b=r;}
    return a;
}

// ─── GF(2^n) ─────────────────────────────────────────────────────────────────

struct GF2n {
    u32 n{8};           // field size GF(2^n)
    u64 primitive{};    // irreducible polynomial (defines the field)
    u64 size{};         // 2^n

    void init(u32 bits, u64 prim) {
        n=bits; primitive=prim; size=u64(1)<<n;
    }

    [[nodiscard]] u64 add(u64 a, u64 b) const { return a^b; }
    [[nodiscard]] u64 sub(u64 a, u64 b) const { return a^b; } // same in GF(2)

    [[nodiscard]] u64 mul(u64 a, u64 b) const {
        u64 r=0, mod=(u64(1)<<n);
        while(b){
            if(b&1)r^=a;
            a<<=1;
            if(a&mod) a^=primitive;
            b>>=1;
        }
        return r&(mod-1);
    }

    [[nodiscard]] u64 pow(u64 base, u64 exp) const {
        u64 r=1; base&=(size-1);
        while(exp>0){if(exp&1)r=mul(r,base); base=mul(base,base); exp>>=1;}
        return r;
    }

    [[nodiscard]] u64 inv(u64 a) const { return pow(a,size-2); }
    [[nodiscard]] u64 div(u64 a, u64 b) const { return mul(a,inv(b)); }

    // Build exp/log tables for fast arithmetic
    struct Tables {
        std::vector<u64> exp_t, log_t;
    };

    [[nodiscard]] Tables build_tables() const {
        Tables t;
        t.exp_t.resize(size*2,0); t.log_t.resize(size,0);
        u64 x=1;
        for (u64 i=0;i<size-1;++i) {
            t.exp_t[i]=x; t.log_t[x]=i;
            x=mul(x,2);
        }
        for (u64 i=0;i<size-1;++i) t.exp_t[size-1+i]=t.exp_t[i];
        return t;
    }
};

// ─── LFSR (Linear Feedback Shift Register) ───────────────────────────────────

struct LFSR {
    u64 state{1};
    u64 taps{};   // tap polynomial as bitmask
    u32 width{8};

    // Fibonacci LFSR step
    [[nodiscard]] u64 step() {
        u64 feedback = __builtin_parityll(state & taps);
        state = (state>>1) | (feedback << (width-1));
        return state & 1;
    }

    // Generate n bits
    [[nodiscard]] std::vector<u8> generate(u32 n_bits) {
        std::vector<u8> out(n_bits);
        for (auto& b : out) b=u8(step());
        return out;
    }

    // Maximum length sequence: tap polynomials for common widths
    static constexpr u64 MAXIMAL_TAPS[] = {
        0,0,0b11,0b110,0b1100,0b10100,
        0b110000,0b1100000,0b10111000,0b100010000};
};

// ─── BCH / Reed-Solomon syndrome computation ─────────────────────────────────

struct RSField {
    GF2n gf;
    std::vector<u64> alpha_powers; // powers of primitive element

    void init(u32 bits=8, u64 prim=0x11d) { // GF(256) default
        gf.init(bits,prim);
        alpha_powers.resize(gf.size);
        u64 a=1;
        for (auto& p : alpha_powers) {
            p=a; a=gf.mul(a,2);
        }
    }

    // Compute syndromes S_i = sum(r_j * alpha^(i*j)) for i=1..2t
    [[nodiscard]] std::vector<u64> syndromes(
        std::span<const u64> received, u32 num_syndromes) const
    {
        std::vector<u64> S(num_syndromes,0);
        u32 N=u32(received.size());
        for (u32 i=0;i<num_syndromes;++i) {
            for (u32 j=0;j<N;++j)
                S[i]=gf.add(S[i], gf.mul(received[j], alpha_powers[(i+1)*j%(gf.size-1)]));
        }
        return S;
    }

    // Check if codeword is error-free (all syndromes zero)
    [[nodiscard]] bool is_valid(std::span<const u64> received, u32 t) const {
        auto S=syndromes(received,2*t);
        for (auto s:S) if(s!=0) return false;
        return true;
    }
};

// ─── CRC computation ──────────────────────────────────────────────────────────

[[nodiscard]] inline u32 crc32(std::span<const u8> data, u32 poly=0x04C11DB7) {
    u32 crc=0xFFFFFFFFu;
    for (u8 b : data) {
        crc^=u32(b)<<24;
        for (u32 i=0;i<8;++i)
            crc=(crc&0x80000000u)?(crc<<1)^poly:(crc<<1);
    }
    return crc^0xFFFFFFFFu;
}

[[nodiscard]] inline u16 crc16(std::span<const u8> data, u16 poly=0x8005) {
    u16 crc=0xFFFFu;
    for (u8 b : data) {
        crc^=u16(b)<<8;
        for (u32 i=0;i<8;++i)
            crc=(crc&0x8000u)?(crc<<1)^poly:(crc<<1);
    }
    return crc;
}

} // namespace qs
