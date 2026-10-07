#pragma once
// qs_math/cryptographic_math.hpp — Modular arithmetic, RSA primitives, elliptic curve (C++20)
#include "scalar_types.hpp"
#include "finite_field_math.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <tuple>
#include <optional>
#include <random>

namespace qs {

// ─── Big integer (128-bit) helpers ───────────────────────────────────────────

using u128 = unsigned __int128;

[[nodiscard]] inline u128 mul_mod128(u64 a, u64 b, u64 m) {
    return (u128(a)*b)%u128(m);
}

[[nodiscard]] inline u64 pow_mod(u64 base, u64 exp, u64 mod) {
    u64 r=1; base%=mod;
    while(exp>0){if(exp&1)r=u64(mul_mod128(r,base,mod));base=u64(mul_mod128(base,base,mod));exp>>=1;}
    return r;
}

// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd
[[nodiscard]] inline std::tuple<i64,i64,i64> extended_gcd(i64 a, i64 b) {
    if(b==0) return {a,1,0};
    auto[g,x,y]=extended_gcd(b,a%b);
    return {g,y,x-(a/b)*y};
}

[[nodiscard]] inline std::optional<u64> mod_inverse(u64 a, u64 m) {
    auto[g,x,y]=extended_gcd(i64(a),i64(m));
    if(g!=1) return std::nullopt;
    return u64((x%i64(m)+i64(m))%i64(m));
}

// ─── Miller-Rabin primality test ─────────────────────────────────────────────

[[nodiscard]] inline bool miller_rabin_round(u64 n, u64 a) {
    if(n%a==0) return n==a;
    u64 d=n-1; u32 r=0;
    while(d%2==0){d>>=1;++r;}
    u64 x=pow_mod(a,d,n);
    if(x==1||x==n-1) return true;
    for(u32 i=0;i<r-1;++i){x=u64(mul_mod128(x,x,n));if(x==n-1)return true;}
    return false;
}

[[nodiscard]] inline bool is_prime(u64 n) {
    if(n<2) return false;
    if(n<4) return true;
    if(n%2==0||n%3==0) return false;
    // Deterministic for n < 3,215,031,751 with these witnesses
    for(u64 a:{2,3,5,7,11,13,17,19,23,29,31,37})
        if(n==a) return true;
        else if(!miller_rabin_round(n,a)) return false;
    return true;
}

// ─── RSA primitives ───────────────────────────────────────────────────────────

struct RSAKey {
    u64 n{},e{},d{};  // modulus, public exponent, private exponent
};

// Generate small RSA key (educational, not cryptographically safe)
[[nodiscard]] inline RSAKey rsa_keygen_small(u64 p, u64 q, u64 e=65537) {
    u64 n=p*q;
    u64 phi=(p-1)*(q-1);
    auto d_opt=mod_inverse(e,phi);
    if(!d_opt) return {};
    return {n,e,*d_opt};
}

[[nodiscard]] inline u64 rsa_encrypt(u64 msg, const RSAKey& key) {
    return pow_mod(msg,key.e,key.n);
}

[[nodiscard]] inline u64 rsa_decrypt(u64 cipher, const RSAKey& key) {
    return pow_mod(cipher,key.d,key.n);
}

// Chinese Remainder Theorem for RSA-CRT decryption
[[nodiscard]] inline u64 rsa_crt_decrypt(u64 c, u64 p, u64 q, u64 dp, u64 dq, u64 qinv) {
    u64 m1=pow_mod(c,dp,p);
    u64 m2=pow_mod(c,dq,q);
    i64 h=i64(mul_mod128(qinv,u64((m1+p-m2%p)%p),p));
    return m2+u64(h)*q;
}

// ─── Diffie-Hellman ───────────────────────────────────────────────────────────

struct DHParams { u64 p{}, g{}; };

[[nodiscard]] inline u64 dh_public_key(const DHParams& params, u64 private_key) {
    return pow_mod(params.g, private_key, params.p);
}

[[nodiscard]] inline u64 dh_shared_secret(const DHParams& params, u64 their_public, u64 my_private) {
    return pow_mod(their_public, my_private, params.p);
}

// ─── Elliptic curve (Weierstrass: y^2 = x^3 + ax + b mod p) ─────────────────

struct ECPoint { u64 x{}, y{}; bool infinity{true}; };

struct ECCurve {
    u64 p{}, a{}, b{};

    [[nodiscard]] bool on_curve(const ECPoint& pt) const {
        if(pt.infinity) return true;
        u64 lhs=u64(mul_mod128(pt.y,pt.y,p));
        u64 rhs=u64((u128(pow_mod(pt.x,3,p))+u128(a)*pt.x%p+b)%p);
        return lhs==rhs;
    }

    [[nodiscard]] ECPoint point_add(const ECPoint& P, const ECPoint& Q) const {
        if(P.infinity) return Q;
        if(Q.infinity) return P;
        if(P.x==Q.x&&(P.y!=Q.y||P.y==0)) return {0,0,true}; // point at infinity
        u64 lambda{};
        if(P.x==Q.x) {
            // Point doubling
            u64 num=u64((3*u128(P.x)%p*P.x%p+a)%p);
            u64 den=u64(2*u128(P.y)%p);
            auto inv=mod_inverse(den,p); if(!inv) return {0,0,true};
            lambda=u64(u128(num)**inv%p);
        } else {
            u64 num=u64((Q.y+p-P.y)%p);
            u64 den=u64((Q.x+p-P.x)%p);
            auto inv=mod_inverse(den,p); if(!inv) return {0,0,true};
            lambda=u64(u128(num)**inv%p);
        }
        u64 x3=u64((u128(lambda)*lambda%p+2*p-P.x-Q.x)%p);
        u64 y3=u64((u128(lambda)*(P.x+p-x3)%p+p-P.y)%p);
        return {x3,y3,false};
    }

    [[nodiscard]] ECPoint scalar_mul(const ECPoint& P, u64 k) const {
        ECPoint R{0,0,true}, Q=P;
        while(k>0){if(k&1)R=point_add(R,Q);Q=point_add(Q,Q);k>>=1;}
        return R;
    }
};

// ECDH shared secret
[[nodiscard]] inline ECPoint ecdh_public(const ECCurve& curve, const ECPoint& G, u64 priv) {
    return curve.scalar_mul(G,priv);
}

[[nodiscard]] inline ECPoint ecdh_shared(const ECCurve& curve, const ECPoint& their_pub, u64 my_priv) {
    return curve.scalar_mul(their_pub,my_priv);
}

// ─── Hash functions ───────────────────────────────────────────────────────────

// FNV-1a 64-bit
[[nodiscard]] inline u64 fnv1a_64(std::span<const u8> data) {
    u64 h=14695981039346656037ULL;
    for(u8 b:data) h=(h^b)*1099511628211ULL;
    return h;
}

// SipHash 1-3 (simplified)
[[nodiscard]] inline u64 siphash13(std::span<const u8> data, u64 k0=0, u64 k1=0) {
    auto rot=[](u64 x,u32 b)->u64{return (x<<b)|(x>>(64-b));};
    auto sipround=[&](u64&v0,u64&v1,u64&v2,u64&v3){
        v0+=v1;v1=rot(v1,13);v1^=v0;v0=rot(v0,32);
        v2+=v3;v3=rot(v3,16);v3^=v2;
        v0+=v3;v3=rot(v3,21);v3^=v0;
        v2+=v1;v1=rot(v1,17);v1^=v2;v2=rot(v2,32);};
    u64 v0=k0^0x736f6d6570736575ULL, v1=k1^0x646f72616e646f6dULL;
    u64 v2=k0^0x6c7967656e657261ULL, v3=k1^0x7465646279746573ULL;
    u32 n=u32(data.size());
    for(u32 i=0;i+8<=n;i+=8){
        u64 m=0; for(u32 j=0;j<8;++j) m|=u64(data[i+j])<<(8*j);
        v3^=m; sipround(v0,v1,v2,v3); v0^=m;
    }
    // Last block
    u64 last=u64(n&0xff)<<56;
    u32 rem=n&7;
    for(u32 i=0;i<rem;++i) last|=u64(data[n-rem+i])<<(8*i);
    v3^=last; sipround(v0,v1,v2,v3); v0^=last;
    v2^=0xff; for(u32 i=0;i<3;++i) sipround(v0,v1,v2,v3);
    return v0^v1^v2^v3;
}

// ─── Shamir secret sharing ────────────────────────────────────────────────────

// Split secret into n shares, need k to reconstruct (k <= n)
[[nodiscard]] inline std::vector<std::pair<u64,u64>> shamir_split(
    u64 secret, u32 k, u32 n, u64 prime, u64 seed=42)
{
    std::mt19937_64 rng(seed);
    std::vector<u64> coeffs={secret};
    for(u32 i=1;i<k;++i) coeffs.push_back(rng()%prime);
    std::vector<std::pair<u64,u64>> shares;
    for(u32 x=1;x<=n;++x){
        u64 y=0, xpow=1;
        for(u64 c:coeffs){y=u64((u128(y)+u128(c)*xpow)%prime);xpow=u64(u128(xpow)*x%prime);}
        shares.push_back({x,y});
    }
    return shares;
}

// Lagrange interpolation at x=0 to recover secret
[[nodiscard]] inline u64 shamir_reconstruct(
    std::span<const std::pair<u64,u64>> shares, u64 prime)
{
    u64 secret=0;
    u32 k=u32(shares.size());
    for(u32 i=0;i<k;++i){
        u64 num=1, den=1;
        for(u32 j=0;j<k;++j){
            if(i==j) continue;
            num=u64(u128(num)*shares[j].first%prime);
            u64 diff=(shares[j].first+prime-shares[i].first)%prime;
            den=u64(u128(den)*diff%prime);
        }
        auto inv=mod_inverse(den,prime); if(!inv) continue;
        secret=u64((u128(secret)+u128(shares[i].second)*num%prime**inv)%prime);
    }
    return secret;
}

} // namespace qs
