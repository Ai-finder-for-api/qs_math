#pragma once
// qs_math/arithmetic/bigint.hpp — Arbitrary-precision unsigned integer (C++20)
#include "scalar_types.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <compare>
#include <span>
#include <cmath>

namespace qs {

// BigUInt: arbitrary-precision non-negative integer stored in base 2^32 limbs (little-endian)
struct BigUInt {
    std::vector<u32> limbs; // limbs[0] = least significant

    BigUInt() : limbs{0} {}
    explicit BigUInt(u64 v){ limbs.push_back(u32(v)); if(v>>32) limbs.push_back(u32(v>>32)); normalize(); }
    explicit BigUInt(const std::string& dec){ *this=from_decimal(dec); }

    void normalize(){ while(limbs.size()>1&&limbs.back()==0) limbs.pop_back(); }

    [[nodiscard]] bool is_zero() const { return limbs.size()==1&&limbs[0]==0; }
    [[nodiscard]] u32 num_limbs() const { return u32(limbs.size()); }
    [[nodiscard]] u32 bit_length() const {
        if(is_zero()) return 0;
        u32 top=limbs.back();
        u32 bits=(u32(limbs.size())-1)*32;
        while(top){bits++;top>>=1;}
        return bits;
    }

    [[nodiscard]] bool bit(u32 i) const {
        u32 limb=i/32, bit2=i%32;
        return limb<limbs.size()&&((limbs[limb]>>bit2)&1);
    }

    // ── Comparison ────────────────────────────────────────────────────────────
    [[nodiscard]] std::strong_ordering operator<=>(const BigUInt& o) const {
        if(limbs.size()!=o.limbs.size()) return limbs.size()<=>o.limbs.size();
        for(i32 i=i32(limbs.size())-1;i>=0;--i)
            if(limbs[i]!=o.limbs[i]) return limbs[i]<=>o.limbs[i];
        return std::strong_ordering::equal;
    }
    [[nodiscard]] bool operator==(const BigUInt& o) const { return limbs==o.limbs; }

    // ── Addition ──────────────────────────────────────────────────────────────
    [[nodiscard]] BigUInt operator+(const BigUInt& o) const {
        BigUInt r;
        r.limbs.resize(std::max(limbs.size(),o.limbs.size())+1,0);
        u64 carry=0;
        for(u32 i=0;i<r.limbs.size();++i){
            u64 a=i<limbs.size()?limbs[i]:0;
            u64 b=i<o.limbs.size()?o.limbs[i]:0;
            u64 s=a+b+carry;
            r.limbs[i]=u32(s);
            carry=s>>32;
        }
        r.normalize();
        return r;
    }

    BigUInt& operator+=(const BigUInt& o){ *this=*this+o; return *this; }

    // ── Subtraction (assumes *this >= o) ──────────────────────────────────────
    [[nodiscard]] BigUInt operator-(const BigUInt& o) const {
        assert(*this>=o);
        BigUInt r; r.limbs.resize(limbs.size(),0);
        i64 borrow=0;
        for(u32 i=0;i<limbs.size();++i){
            i64 d=i64(limbs[i])-(i<o.limbs.size()?i64(o.limbs[i]):0)-borrow;
            r.limbs[i]=u32(d&0xFFFFFFFF);
            borrow=d<0?1:0;
        }
        r.normalize();
        return r;
    }

    BigUInt& operator-=(const BigUInt& o){ *this=*this-o; return *this; }

    // ── Multiplication (schoolbook O(n^2)) ─────────────────────────────────────
    [[nodiscard]] BigUInt operator*(const BigUInt& o) const {
        BigUInt r; r.limbs.assign(limbs.size()+o.limbs.size(),0);
        for(u32 i=0;i<limbs.size();++i){
            u64 carry=0;
            for(u32 j=0;j<o.limbs.size();++j){
                u64 cur=u64(limbs[i])*o.limbs[j]+r.limbs[i+j]+carry;
                r.limbs[i+j]=u32(cur);
                carry=cur>>32;
            }
            r.limbs[i+o.limbs.size()]+=u32(carry);
        }
        r.normalize();
        return r;
    }

    BigUInt& operator*=(const BigUInt& o){ *this=*this*o; return *this; }

    // ── Left/right shift ──────────────────────────────────────────────────────
    [[nodiscard]] BigUInt operator<<(u32 bits) const {
        u32 limb_shift=bits/32, bit_shift=bits%32;
        BigUInt r; r.limbs.assign(limbs.size()+limb_shift+1,0);
        for(u32 i=0;i<limbs.size();++i){
            r.limbs[i+limb_shift]|=limbs[i]<<bit_shift;
            if(bit_shift&&i+limb_shift+1<r.limbs.size())
                r.limbs[i+limb_shift+1]|=limbs[i]>>(32-bit_shift);
        }
        r.normalize();
        return r;
    }

    [[nodiscard]] BigUInt operator>>(u32 bits) const {
        u32 limb_shift=bits/32, bit_shift=bits%32;
        if(limb_shift>=limbs.size()) return BigUInt(0ull);
        BigUInt r; r.limbs.resize(limbs.size()-limb_shift,0);
        for(u32 i=limb_shift;i<limbs.size();++i){
            r.limbs[i-limb_shift]=limbs[i]>>bit_shift;
            if(bit_shift&&i+1<limbs.size())
                r.limbs[i-limb_shift]|=limbs[i+1]<<(32-bit_shift);
        }
        r.normalize();
        return r;
    }

    // ── Division + remainder ──────────────────────────────────────────────────
    [[nodiscard]] std::pair<BigUInt,BigUInt> divmod(const BigUInt& divisor) const {
        assert(!divisor.is_zero());
        if(*this<divisor) return {BigUInt(0ull),*this};
        BigUInt q, r;
        for(i32 i=i32(bit_length())-1;i>=0;--i){
            r=r<<1;
            if(bit(u32(i))) r.limbs[0]|=1;
            if(r>=divisor){r-=divisor;if(q.limbs.size()<=u32(i)/32)q.limbs.resize(u32(i)/32+1,0);q.limbs[u32(i)/32]|=1u<<(u32(i)%32);}
        }
        q.normalize(); r.normalize();
        return {q,r};
    }

    [[nodiscard]] BigUInt operator/(const BigUInt& o) const { return divmod(o).first; }
    [[nodiscard]] BigUInt operator%(const BigUInt& o) const { return divmod(o).second; }

    // ── GCD ───────────────────────────────────────────────────────────────────
    [[nodiscard]] BigUInt gcd(const BigUInt& o) const {
        BigUInt a=*this,b=o;
        while(!b.is_zero()){auto r=a%b;a=b;b=r;}
        return a;
    }

    // ── Modular exponentiation ────────────────────────────────────────────────
    [[nodiscard]] BigUInt powmod(const BigUInt& exp, const BigUInt& mod) const {
        BigUInt r(1ull), base=*this%mod, e=exp;
        while(!e.is_zero()){
            if(e.limbs[0]&1) r=(r*base)%mod;
            base=(base*base)%mod;
            e=e>>1;
        }
        return r;
    }

    // ── Decimal I/O ───────────────────────────────────────────────────────────
    [[nodiscard]] std::string to_decimal() const {
        if(is_zero()) return "0";
        BigUInt n=*this;
        std::string s;
        BigUInt ten(10ull);
        while(!n.is_zero()){
            auto [q,r]=n.divmod(ten);
            s+=char('0'+r.limbs[0]);
            n=q;
        }
        std::reverse(s.begin(),s.end());
        return s;
    }

    [[nodiscard]] static BigUInt from_decimal(const std::string& s) {
        BigUInt r(0ull), ten(10ull);
        for(char c:s){ r=r*ten; r+=BigUInt(u64(c-'0')); }
        return r;
    }

    [[nodiscard]] std::string to_hex() const {
        if(is_zero()) return "0";
        std::string s;
        static constexpr char hex[]="0123456789abcdef";
        for(i32 i=i32(limbs.size())-1;i>=0;--i){
            u32 v=limbs[i];
            for(i32 b=28;b>=0;b-=4) s+=hex[(v>>b)&0xF];
        }
        s.erase(0,s.find_first_not_of('0'));
        return s.empty()?"0":s;
    }

    [[nodiscard]] static BigUInt from_hex(const std::string& s){
        BigUInt r(0ull);
        for(char c:s){
            r=r<<4;
            u32 d=(c>='0'&&c<='9')?c-'0':(c>='a'&&c<='f')?c-'a'+10:(c>='A'&&c<='F')?c-'A'+10:0;
            r.limbs[0]|=d;
        }
        r.normalize();
        return r;
    }

    // ── Factorial (utility) ───────────────────────────────────────────────────
    [[nodiscard]] static BigUInt factorial(u32 n){
        BigUInt r(1ull);
        for(u32 i=2;i<=n;++i) r*=BigUInt(u64(i));
        return r;
    }

    // ── Fibonacci ─────────────────────────────────────────────────────────────
    [[nodiscard]] static BigUInt fibonacci(u32 n){
        if(n==0) return BigUInt(0ull);
        BigUInt a(0ull),b(1ull);
        for(u32 i=1;i<n;++i){auto c=a+b;a=b;b=c;}
        return b;
    }
};

// Signed big integer wrapper
struct BigInt {
    BigUInt mag;
    bool negative{false};

    BigInt()=default;
    explicit BigInt(i64 v):mag(v<0?u64(-v):u64(v)),negative(v<0){}
    BigInt(BigUInt m,bool neg):mag(std::move(m)),negative(neg){}

    [[nodiscard]] bool is_zero() const { return mag.is_zero(); }

    [[nodiscard]] BigInt operator-() const { return {mag,!negative}; }

    [[nodiscard]] BigInt operator+(const BigInt& o) const {
        if(negative==o.negative) return {mag+o.mag,negative};
        if(mag>=o.mag) return {mag-o.mag,negative};
        return {o.mag-mag,o.negative};
    }

    [[nodiscard]] BigInt operator*(const BigInt& o) const {
        return {mag*o.mag,negative!=o.negative};
    }

    [[nodiscard]] std::string to_string() const {
        return (negative&&!is_zero()?"-":"")+mag.to_decimal();
    }
};

} // namespace qs
