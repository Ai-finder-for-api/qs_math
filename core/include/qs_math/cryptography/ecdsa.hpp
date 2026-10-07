#pragma once
// qs_math/cryptography/ecdsa.hpp — ECDSA sign/verify, RFC 6979 deterministic k
#include "elliptic_curve.hpp"
#include "sha_math.hpp"
#include <array>
#include <cstring>

namespace qs::math {

// ─── RFC 6979: Deterministic k generation ────────────────────────────────────
// Produces a deterministic k in [1, n-1] from private key + message hash
[[nodiscard]] inline BigInt2048 rfc6979_k(const BigInt2048& priv,
                                            const u8* hash, u64 hlen,
                                            const BigInt2048& n) noexcept {
    // Serialize privkey (big-endian, 32 bytes for 256-bit curves)
    u8 x[32]{};
    for (int i=0;i<4;++i) {
        u64 limb=priv.d[3-i];
        for (int j=7;j>=0;--j) { x[i*8+j]=u8(limb&0xff); limb>>=8; }
    }
    // V = 0x01 * 32, K = 0x00 * 32
    u8 V[32]; std::memset(V,0x01,32);
    u8 K[32]; std::memset(K,0x00,32);

    // K = HMAC-SHA256(K, V || 0x00 || x || hash)
    u8 buf[32+1+32+hlen];
    std::memcpy(buf,V,32); buf[32]=0x00;
    std::memcpy(buf+33,x,32); std::memcpy(buf+65,hash,hlen);
    auto k_arr=hmac_sha256(K,32,buf,65+hlen);
    std::memcpy(K,k_arr.data(),32);

    // V = HMAC-SHA256(K, V)
    auto v_arr=hmac_sha256(K,32,V,32);
    std::memcpy(V,v_arr.data(),32);

    // K = HMAC-SHA256(K, V || 0x01 || x || hash)
    std::memcpy(buf,V,32); buf[32]=0x01;
    std::memcpy(buf+33,x,32); std::memcpy(buf+65,hash,hlen);
    k_arr=hmac_sha256(K,32,buf,65+hlen);
    std::memcpy(K,k_arr.data(),32);

    // V = HMAC-SHA256(K, V)
    v_arr=hmac_sha256(K,32,V,32);
    std::memcpy(V,v_arr.data(),32);

    // Generate candidate k from V (big-endian 32 bytes → BigInt2048)
    BigInt2048 k{};
    for (int i=0;i<32;++i) {
        int limb=(31-i)/8, bit_pos=((31-i)%8)*8;
        k.d[limb]|=(u64(V[i])<<bit_pos);
    }
    // Reduce mod n (simple subtraction)
    while(cmp(k,n)>=0) k=sub(k,n);
    return k;
}

// ─── ECDSA signature ──────────────────────────────────────────────────────────
struct EcdsaSig {
    BigInt2048 r, s;
    [[nodiscard]] bool valid() const noexcept { return !r.is_zero() && !s.is_zero(); }
};

[[nodiscard]] inline EcdsaSig ecdsa_sign(const u8* hash, u64 hlen,
                                           const BigInt2048& priv,
                                           const EcCurve& curve) noexcept {
    const auto& n=curve.n;
    BigInt2048 e{};
    // Load hash as big integer (leftmost hlen bytes, up to 32)
    u64 load=hlen<32?hlen:32;
    for (u64 i=0;i<load;++i) {
        e.d[(load-1-i)/8]|=(u64(hash[i])<<(((load-1-i)%8)*8));
    }
    // Reduce e mod n
    while(cmp(e,n)>=0) e=sub(e,n);

    for (int attempt=0;attempt<64;++attempt) {
        // k = RFC6979 deterministic nonce (retry with modified hash on failure)
        u8 mod_hash[hlen+1];
        std::memcpy(mod_hash,hash,hlen);
        mod_hash[hlen]=(u8)attempt;
        BigInt2048 k=rfc6979_k(priv, attempt==0?hash:mod_hash,
                                attempt==0?hlen:hlen+1, n);
        if (k.is_zero()) continue;

        // R = k*G
        EcPoint R=ec_mul(k,curve.G,curve);
        if (R.inf) continue;

        // r = R.x mod n
        BigInt2048 r=mod(R.x,n);
        if (r.is_zero()) continue;

        // s = k^-1 * (e + r*priv) mod n
        BigInt2048 k_inv=powmod(k,sub(n,{[0]=2ULL}),n); // k^(n-2) mod n (Fermat)
        // Workaround for sub with partial init:
        BigInt2048 n_minus_2=n; n_minus_2.d[0]-=2;
        k_inv=powmod(k,n_minus_2,n);

        BigInt2048 rp=mulmod(r,priv,n);
        BigInt2048 ep=add(e,rp); if(cmp(ep,n)>=0) ep=sub(ep,n);
        BigInt2048 s=mulmod(k_inv,ep,n);
        if (s.is_zero()) continue;

        return {r,s};
    }
    return {};
}

// ─── ECDSA verify ─────────────────────────────────────────────────────────────
[[nodiscard]] inline bool ecdsa_verify(const u8* hash, u64 hlen,
                                        const EcdsaSig& sig,
                                        const EcPoint& pub,
                                        const EcCurve& curve) noexcept {
    const auto& n=curve.n;
    BigInt2048 zero{};
    if (sig.r.is_zero()||sig.s.is_zero()) return false;
    if (cmp(sig.r,n)>=0||cmp(sig.s,n)>=0) return false;

    BigInt2048 e{};
    u64 load=hlen<32?hlen:32;
    for (u64 i=0;i<load;++i)
        e.d[(load-1-i)/8]|=(u64(hash[i])<<(((load-1-i)%8)*8));
    while(cmp(e,n)>=0) e=sub(e,n);

    BigInt2048 n2=n; n2.d[0]-=2;
    BigInt2048 s_inv=powmod(sig.s,n2,n);
    BigInt2048 u1=mulmod(e,s_inv,n);
    BigInt2048 u2=mulmod(sig.r,s_inv,n);

    // u1*G + u2*Q
    EcPoint P1=ec_mul(u1,curve.G,curve);
    EcPoint P2=ec_mul(u2,pub,curve);
    EcPoint X=ec_add(P1,P2,curve);
    if (X.inf) return false;

    BigInt2048 x_mod=mod(X.x,n);
    return cmp(x_mod,sig.r)==0;
}

// ─── DER encoding of ECDSA signature (compact, 256-bit curves) ───────────────
inline int ecdsa_der_encode(const EcdsaSig& sig, u8* out, int max_len) noexcept {
    // Encode r and s as big-endian 32-byte integers
    auto encode_int=[](const BigInt2048& v, u8* buf) -> int {
        u8 tmp[32];
        for (int i=0;i<4;++i) {
            u64 limb=v.d[3-i];
            for (int j=7;j>=0;--j) tmp[i*8+j]=u8(limb&0xff), limb>>=8;
        }
        int start=0;
        while (start<31&&tmp[start]==0) ++start;
        int len=32-start;
        bool pad=(tmp[start]&0x80)!=0;
        buf[0]=0x02; buf[1]=u8(len+(pad?1:0));
        if (pad) { buf[2]=0; std::memcpy(buf+3,tmp+start,len); return 2+1+len; }
        std::memcpy(buf+2,tmp+start,len); return 2+len;
    };
    u8 r_enc[35], s_enc[35];
    int rl=encode_int(sig.r,r_enc);
    int sl=encode_int(sig.s,s_enc);
    int total=2+rl+sl;
    if (total>max_len) return -1;
    out[0]=0x30; out[1]=u8(rl+sl);
    std::memcpy(out+2,r_enc,rl);
    std::memcpy(out+2+rl,s_enc,sl);
    return total;
}

// ─── Schnorr signature (BIP-340) ─────────────────────────────────────────────
struct SchnorrSig { u8 r_bytes[32]{}; u8 s_bytes[32]{}; };

// Tagged hash: SHA256(SHA256(tag) || SHA256(tag) || msg)
[[nodiscard]] inline std::array<u8,32> schnorr_tagged_hash(const char* tag,
                                                              const u8* msg, u64 mlen) noexcept {
    u64 tlen=0; while(tag[tlen]) ++tlen;
    auto t_hash=sha256(reinterpret_cast<const u8*>(tag),tlen);
    u8 buf[64+mlen];
    std::memcpy(buf,t_hash.data(),32);
    std::memcpy(buf+32,t_hash.data(),32);
    std::memcpy(buf+64,msg,mlen);
    return sha256(buf,64+mlen);
}

} // namespace qs::math
