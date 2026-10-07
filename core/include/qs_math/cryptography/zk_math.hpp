#pragma once
// qs_math/cryptography/zk_math.hpp — ZK proof helpers: Pedersen, Fiat-Shamir, Schnorr proofs
#include "elliptic_curve.hpp"
#include "sha_math.hpp"

namespace qs::math {

// ─── Pedersen commitment: C = r*G + m*H ──────────────────────────────────────
struct PedersenCommitment { EcPoint C; };
struct PedersenOpening    { BigInt2048 m, r; };

[[nodiscard]] inline PedersenCommitment pedersen_commit(
    const BigInt2048& m, const BigInt2048& r,
    const EcPoint& G, const EcPoint& H, const EcCurve& curve) noexcept
{
    EcPoint rG = ec_mul(r, G, curve);
    EcPoint mH = ec_mul(m, H, curve);
    return { ec_add(rG, mH, curve) };
}

[[nodiscard]] inline bool pedersen_verify(
    const PedersenCommitment& C, const PedersenOpening& o,
    const EcPoint& G, const EcPoint& H, const EcCurve& curve) noexcept
{
    auto expected = pedersen_commit(o.m, o.r, G, H, curve);
    return ec_point_eq(expected.C, C.C);
}

// ─── Fiat-Shamir transform: challenge = H(ctx || R) ──────────────────────────
[[nodiscard]] inline BigInt2048 fiat_shamir_challenge(
    const EcPoint& R, const u8* ctx, u64 ctx_len, const BigInt2048& n) noexcept
{
    // Serialize R.x (32 bytes big-endian) + ctx into hash input
    u8 buf[32 + ctx_len];
    for (int i = 0; i < 4; ++i) {
        u64 limb = R.x.d[3-i];
        for (int j = 7; j >= 0; --j) { buf[i*8+j] = u8(limb & 0xff); limb >>= 8; }
    }
    std::memcpy(buf + 32, ctx, ctx_len);
    auto h = sha256(buf, 32 + ctx_len);
    BigInt2048 c{};
    for (int i = 0; i < 32; ++i)
        c.d[(31-i)/8] |= (u64(h[i]) << (((31-i)%8)*8));
    while (cmp(c,n) >= 0) c = sub(c,n);
    return c;
}

// ─── Schnorr proof of knowledge of discrete log ───────────────────────────────
// Prover knows x such that P = x*G
struct SchnorrProof { EcPoint R; BigInt2048 s; };

[[nodiscard]] inline SchnorrProof schnorr_prove(
    const BigInt2048& x, const BigInt2048& k,  // k = random nonce
    const EcCurve& curve) noexcept
{
    EcPoint R = ec_mul(k, curve.G, curve);
    // c = H(R)
    BigInt2048 c = fiat_shamir_challenge(R, nullptr, 0, curve.n);
    // s = k + c*x mod n
    BigInt2048 cx = mulmod(c, x, curve.n);
    BigInt2048 s  = add(k, cx);
    if (cmp(s, curve.n) >= 0) s = sub(s, curve.n);
    return { R, s };
}

[[nodiscard]] inline bool schnorr_verify(
    const SchnorrProof& proof, const EcPoint& P, const EcCurve& curve) noexcept
{
    BigInt2048 c = fiat_shamir_challenge(proof.R, nullptr, 0, curve.n);
    EcPoint lhs  = ec_mul(proof.s, curve.G, curve);
    EcPoint rhs  = ec_add(proof.R, ec_mul(c, P, curve), curve);
    return ec_point_eq(lhs, rhs);
}

// ─── Sigma protocol: AND proof (both x0,x1 known) ────────────────────────────
struct AndProof { SchnorrProof p0, p1; BigInt2048 c_shared; };

[[nodiscard]] inline AndProof sigma_and_prove(
    const BigInt2048& x0, const BigInt2048& k0,
    const BigInt2048& x1, const BigInt2048& k1,
    const EcCurve& curve) noexcept
{
    EcPoint R0 = ec_mul(k0, curve.G, curve);
    EcPoint R1 = ec_mul(k1, curve.G, curve);
    // Joint challenge c = H(R0.x XOR R1.x)
    u8 xbuf[32];
    for (int i = 0; i < 4; ++i) {
        u64 v = R0.x.d[3-i] ^ R1.x.d[3-i];
        for (int j = 7; j >= 0; --j) { xbuf[i*8+j] = u8(v&0xff); v >>= 8; }
    }
    auto hv = sha256(xbuf, 32);
    BigInt2048 c{};
    for (int i = 0; i < 32; ++i)
        c.d[(31-i)/8] |= (u64(hv[i]) << (((31-i)%8)*8));
    while (cmp(c,curve.n) >= 0) c = sub(c,curve.n);
    BigInt2048 s0 = add(k0, mulmod(c,x0,curve.n));
    BigInt2048 s1 = add(k1, mulmod(c,x1,curve.n));
    if (cmp(s0,curve.n)>=0) s0=sub(s0,curve.n);
    if (cmp(s1,curve.n)>=0) s1=sub(s1,curve.n);
    return {{R0,s0},{R1,s1},c};
}

// ─── Bulletproofs inner product argument (structural skeleton) ────────────────
// Full recursive IPA — scalars over curve order
struct IpaProof {
    static constexpr int k_max_rounds = 8; // log2(256)
    EcPoint L[k_max_rounds], R_pts[k_max_rounds];
    BigInt2048 a_final, b_final;
    int rounds{0};
};

// Vector dot product mod n
[[nodiscard]] inline BigInt2048 inner_product(
    const BigInt2048* a, const BigInt2048* b, int n_elems,
    const BigInt2048& n) noexcept
{
    BigInt2048 acc{};
    for (int i = 0; i < n_elems; ++i) {
        BigInt2048 t = mulmod(a[i], b[i], n);
        acc = add(acc, t);
        if (cmp(acc,n) >= 0) acc = sub(acc,n);
    }
    return acc;
}

// Multi-scalar multiplication: sum(s[i]*P[i])
[[nodiscard]] inline EcPoint msm(
    const BigInt2048* scalars, const EcPoint* points, int count,
    const EcCurve& curve) noexcept
{
    EcPoint acc{};
    for (int i = 0; i < count; ++i)
        acc = ec_add(acc, ec_mul(scalars[i], points[i], curve), curve);
    return acc;
}

// ─── Hash-to-curve (simplified try-and-increment) ────────────────────────────
[[nodiscard]] inline EcPoint hash_to_curve(
    const u8* msg, u64 mlen, const EcCurve& curve) noexcept
{
    u8 buf[mlen + 4];
    std::memcpy(buf, msg, mlen);
    for (u32 ctr = 0; ctr < 1000; ++ctr) {
        buf[mlen+0]=u8(ctr>>24); buf[mlen+1]=u8(ctr>>16);
        buf[mlen+2]=u8(ctr>>8);  buf[mlen+3]=u8(ctr);
        auto h = sha256(buf, mlen+4);
        BigInt2048 x{};
        for (int i = 0; i < 32; ++i)
            x.d[(31-i)/8] |= (u64(h[i]) << (((31-i)%8)*8));
        if (cmp(x, curve.p) >= 0) continue;
        // Check if x^3 + ax + b is a QR mod p
        BigInt2048 x3  = mulmod(mulmod(x,x,curve.p),x,curve.p);
        BigInt2048 ax  = mulmod(curve.a,x,curve.p);
        BigInt2048 rhs = add(add(x3,ax,),curve.b);
        while(cmp(rhs,curve.p)>=0) rhs=sub(rhs,curve.p);
        // y = rhs^((p+1)/4) mod p for p≡3 mod 4 (secp256k1: p≡3 mod 4)
        BigInt2048 exp = add(curve.p,{});
        exp.d[0]+=1; // p+1
        // shift right 2: divide by 4
        for(int j=0;j<k_rsa_limbs-1;++j) exp.d[j]=(exp.d[j]>>2)|(exp.d[j+1]<<62);
        exp.d[k_rsa_limbs-1]>>=2;
        BigInt2048 y = powmod(rhs,exp,curve.p);
        // verify y^2 == rhs
        if(cmp(mulmod(y,y,curve.p),rhs)==0) {
            EcPoint P; P.inf=false; P.x=x; P.y=y; return P;
        }
    }
    return {};
}

} // namespace qs::math
