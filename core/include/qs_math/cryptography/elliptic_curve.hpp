#pragma once
// qs_math/cryptography/elliptic_curve.hpp — Generic EC over prime fields (short Weierstrass)
#include "../arithmetic/scalar_types.hpp"
#include "rsa_math.hpp"
#include <optional>

namespace qs::math {

// ─── Affine point on curve y^2 = x^3 + ax + b (mod p) ───────────────────────
struct EcPoint {
    BigInt2048 x{}, y{};
    bool inf{true}; // point at infinity
};

struct EcCurve {
    BigInt2048 p; // field prime
    BigInt2048 a; // curve coefficient a
    BigInt2048 b; // curve coefficient b
    BigInt2048 n; // group order
    EcPoint    G; // generator
    int        h; // cofactor
};

// ─── Point operations ─────────────────────────────────────────────────────────

[[nodiscard]] constexpr bool ec_point_eq(const EcPoint& A, const EcPoint& B) noexcept {
    if (A.inf && B.inf) return true;
    if (A.inf || B.inf) return false;
    return cmp(A.x,B.x)==0 && cmp(A.y,B.y)==0;
}

[[nodiscard]] constexpr EcPoint ec_neg(const EcPoint& P, const BigInt2048& p) noexcept {
    if (P.inf) return P;
    EcPoint R; R.inf=false; R.x=P.x;
    R.y=sub(p,P.y); return R;
}

// Modular inverse via Fermat (p prime): a^(p-2) mod p
[[nodiscard]] constexpr BigInt2048 ec_inv(const BigInt2048& a, const BigInt2048& p) noexcept {
    BigInt2048 exp=p;
    // exp = p-2
    BigInt2048 two{}; two.d[0]=2;
    exp=sub(p,two);
    return powmod(a,exp,p);
}

[[nodiscard]] constexpr EcPoint ec_add(const EcPoint& P, const EcPoint& Q,
                                        const EcCurve& curve) noexcept {
    if (P.inf) return Q;
    if (Q.inf) return P;
    const auto& p=curve.p;
    if (cmp(P.x,Q.x)==0) {
        if (cmp(P.y,Q.y)!=0) return EcPoint{}; // P = -Q
        // Point doubling
        BigInt2048 two{}; two.d[0]=2; BigInt2048 three{}; three.d[0]=3;
        // lambda = (3*x^2 + a) / (2*y) mod p
        BigInt2048 x2=mulmod(P.x,P.x,p);
        BigInt2048 num=add(mulmod(three,x2,p),curve.a);
        if (cmp(num,p)>=0) num=sub(num,p);
        BigInt2048 den=mulmod(two,P.y,p);
        BigInt2048 lam=mulmod(num,ec_inv(den,p),p);
        BigInt2048 x3=mulmod(lam,lam,p);
        BigInt2048 two_x=add(P.x,P.x); if(cmp(two_x,p)>=0) two_x=sub(two_x,p);
        if(cmp(x3,two_x)>=0) x3=sub(x3,two_x); else x3=add(sub(p,two_x),x3);
        if(cmp(x3,p)>=0) x3=sub(x3,p);
        BigInt2048 dx=sub(P.x,x3); if(cmp(P.x,x3)<0) dx=add(p,dx);
        BigInt2048 y3=sub(mulmod(lam,dx,p),P.y);
        if(cmp(y3,p)>=0) y3=sub(y3,p); // shouldn't happen but guard
        // y3 might be negative (below zero in modular sense)
        BigInt2048 zero{}; if(cmp(y3,p)>=0||cmp(y3,zero)<0) y3=mod(y3,p);
        EcPoint R; R.inf=false; R.x=x3; R.y=y3; return R;
    }
    // Point addition lambda = (y2-y1)/(x2-x1) mod p
    BigInt2048 dy=(cmp(Q.y,P.y)>=0)?sub(Q.y,P.y):add(p,sub(Q.y,P.y));
    BigInt2048 dx2=(cmp(Q.x,P.x)>=0)?sub(Q.x,P.x):add(p,sub(Q.x,P.x));
    BigInt2048 lam=mulmod(dy,ec_inv(dx2,p),p);
    BigInt2048 x3=mulmod(lam,lam,p);
    BigInt2048 sum_x=add(P.x,Q.x); if(cmp(sum_x,p)>=0) sum_x=sub(sum_x,p);
    x3=(cmp(x3,sum_x)>=0)?sub(x3,sum_x):add(p,sub(x3,sum_x));
    BigInt2048 dx3=(cmp(P.x,x3)>=0)?sub(P.x,x3):add(p,sub(P.x,x3));
    BigInt2048 y3=sub(mulmod(lam,dx3,p),P.y);
    if(cmp(y3,p)>=0) y3=mod(y3,p);
    EcPoint R; R.inf=false; R.x=x3; R.y=y3; return R;
}

// Scalar multiplication: k*P using double-and-add
[[nodiscard]] constexpr EcPoint ec_mul(const BigInt2048& k, const EcPoint& P,
                                        const EcCurve& curve) noexcept {
    EcPoint result{}; // identity
    EcPoint addend=P;
    for (int i=0;i<2048;++i) {
        if (k.bit(i)) result=ec_add(result,addend,curve);
        addend=ec_add(addend,addend,curve);
    }
    return result;
}

// ─── secp256k1 parameters (Bitcoin/Ethereum curve) ───────────────────────────
// p = 2^256 - 2^32 - 977
// Full constants stored as byte arrays for init
inline EcCurve make_secp256k1() noexcept {
    EcCurve c;
    // p in little-endian 64-bit limbs
    c.p.d[0]=0xFFFFFFFEFFFFFC2FULL; c.p.d[1]=0xFFFFFFFFFFFFFFFFULL;
    c.p.d[2]=0xFFFFFFFFFFFFFFFFULL; c.p.d[3]=0xFFFFFFFFFFFFFFFFULL;
    // a = 0 for secp256k1
    // b = 7
    c.b.d[0]=7;
    // n (group order)
    c.n.d[0]=0xBFD25E8CD0364141ULL; c.n.d[1]=0xBAAEDCE6AF48A03BULL;
    c.n.d[2]=0xFFFFFFFFFFFFFFFEULL; c.n.d[3]=0xFFFFFFFFFFFFFFFFULL;
    // G
    c.G.inf=false;
    c.G.x.d[0]=0x59F2815B16F81798ULL; c.G.x.d[1]=0x029BFCDB2DCE28D9ULL;
    c.G.x.d[2]=0x55A06295CE870B07ULL; c.G.x.d[3]=0x79BE667EF9DCBBACULL;
    c.G.y.d[0]=0x9C47D08FFB10D4B8ULL; c.G.y.d[1]=0xFD17B448A6855419ULL;
    c.G.y.d[2]=0x5DA4FBFC0E1108A8ULL; c.G.y.d[3]=0x483ADA7726A3C465ULL;
    c.h=1;
    return c;
}

// ─── EC key pair ─────────────────────────────────────────────────────────────
struct EcKeyPair {
    BigInt2048 priv_key;
    EcPoint    pub_key;
};

[[nodiscard]] constexpr EcKeyPair ec_make_keypair(const BigInt2048& priv,
                                                    const EcCurve& curve) noexcept {
    return { priv, ec_mul(priv, curve.G, curve) };
}

// Diffie-Hellman shared secret
[[nodiscard]] constexpr EcPoint ec_dh(const BigInt2048& priv,
                                       const EcPoint& peer_pub,
                                       const EcCurve& curve) noexcept {
    return ec_mul(priv, peer_pub, curve);
}

} // namespace qs::math
