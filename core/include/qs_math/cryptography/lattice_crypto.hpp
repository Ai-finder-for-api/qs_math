#pragma once
// qs_math/cryptography/lattice_crypto.hpp — Lattice crypto: NTT, RLWE, Kyber-style math
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cstring>

namespace qs::math {

// ─── NTT prime field parameters ───────────────────────────────────────────────
// Kyber uses q = 3329, n = 256
static constexpr i32 k_kyber_q   = 3329;
static constexpr i32 k_kyber_n   = 256;
static constexpr i32 k_kyber_n17 = 17; // primitive 512th root of unity mod q

[[nodiscard]] constexpr i32 mod_q(i32 a, i32 q=k_kyber_q) noexcept {
    a %= q; return a<0 ? a+q : a;
}

[[nodiscard]] constexpr i32 ntt_pow(i32 base, i32 exp, i32 q) noexcept {
    i32 r=1; base=mod_q(base,q);
    while (exp>0) { if(exp&1) r=mod_q(r*base,q); base=mod_q(base*base,q); exp>>=1; }
    return r;
}

// ─── Poly256: polynomial mod (x^256 + 1) over Z_q ────────────────────────────
struct Poly256 {
    std::array<i32,256> c{};
    [[nodiscard]] constexpr i32  operator[](int i) const noexcept { return c[i]; }
    [[nodiscard]] constexpr i32& operator[](int i)       noexcept { return c[i]; }
};

[[nodiscard]] constexpr Poly256 poly_add(const Poly256& a, const Poly256& b,
                                          i32 q=k_kyber_q) noexcept {
    Poly256 r;
    for (int i=0;i<256;++i) r[i]=mod_q(a[i]+b[i],q);
    return r;
}

[[nodiscard]] constexpr Poly256 poly_sub(const Poly256& a, const Poly256& b,
                                          i32 q=k_kyber_q) noexcept {
    Poly256 r;
    for (int i=0;i<256;++i) r[i]=mod_q(a[i]-b[i],q);
    return r;
}

// Schoolbook multiplication mod x^256+1 (NTT-domain multiply preferred)
[[nodiscard]] constexpr Poly256 poly_mul_schoolbook(const Poly256& a, const Poly256& b,
                                                     i32 q=k_kyber_q) noexcept {
    Poly256 r;
    for (int i=0;i<256;++i)
        for (int j=0;j<256;++j) {
            int idx=i+j;
            if (idx<256) r[idx]=mod_q(r[idx]+mod_q(a[i]*b[j],q),q);
            else         r[idx-256]=mod_q(r[idx-256]-mod_q(a[i]*b[j],q),q);
        }
    return r;
}

// ─── NTT over Z_3329 (Kyber-compatible) ──────────────────────────────────────
// Precompute zeta table: zeta[i] = 17^br(i) mod 3329, br = bit-reverse of 7-bit index
[[nodiscard]] constexpr u8 bit_rev7(u8 x) noexcept {
    u8 r=0;
    for (int i=0;i<7;++i) { r=(r<<1)|(x&1); x>>=1; }
    return r;
}

inline constexpr auto build_zeta_table() noexcept {
    std::array<i32,128> t{};
    for (int i=0;i<128;++i)
        t[i]=ntt_pow(k_kyber_n17, bit_rev7(static_cast<u8>(i)), k_kyber_q);
    return t;
}
inline constexpr auto k_zeta = build_zeta_table();

// In-place NTT on 256 coefficients (Kyber spec §3.8)
inline void ntt_forward(Poly256& f) noexcept {
    int k=1;
    for (int len=128;len>=2;len>>=1) {
        for (int start=0;start<256;start+=2*len) {
            i32 zeta=k_zeta[k++];
            for (int j=start;j<start+len;++j) {
                i32 t=mod_q(zeta*f[j+len],k_kyber_q);
                f[j+len]=mod_q(f[j]-t,k_kyber_q);
                f[j    ]=mod_q(f[j]+t,k_kyber_q);
            }
        }
    }
}

inline void ntt_inverse(Poly256& f) noexcept {
    int k=127;
    for (int len=2;len<=128;len<<=1) {
        for (int start=0;start<256;start+=2*len) {
            i32 zeta_inv=mod_q(-k_zeta[k--],k_kyber_q);
            for (int j=start;j<start+len;++j) {
                i32 t=f[j];
                f[j    ]=mod_q(t+f[j+len],k_kyber_q);
                f[j+len]=mod_q(zeta_inv*(f[j+len]-t),k_kyber_q);
            }
        }
    }
    // Multiply by n^-1 = 3303 mod 3329
    i32 n_inv=ntt_pow(k_kyber_n,k_kyber_q-2,k_kyber_q);
    for (int i=0;i<256;++i) f[i]=mod_q(f[i]*n_inv,k_kyber_q);
}

// NTT-domain pointwise multiplication
[[nodiscard]] constexpr Poly256 ntt_mul(const Poly256& a_hat, const Poly256& b_hat,
                                         i32 q=k_kyber_q) noexcept {
    Poly256 r;
    for (int i=0;i<128;++i) {
        i32 zeta=k_zeta[i+64<128?i+64:i]; // approximation; exact Kyber spec uses pair-wise
        i32 a0=a_hat[2*i], a1=a_hat[2*i+1], b0=b_hat[2*i], b1=b_hat[2*i+1];
        r[2*i  ]=mod_q(a0*b0 + mod_q(zeta*mod_q(a1*b1,q),q),q);
        r[2*i+1]=mod_q(a0*b1 + a1*b0,q);
    }
    return r;
}

// ─── RLWE encryption (simplified Kyber-like) ──────────────────────────────────
struct RlwePublicKey  { Poly256 a, b; }; // b = a*s + e
struct RlwePrivateKey { Poly256 s; };
struct RlweCiphertext { Poly256 u, v; }; // u=a*r+e1, v=b*r+e2+encode(m)

// Encode 1-bit message into polynomial coefficient (0 → 0, 1 → q/2)
[[nodiscard]] constexpr Poly256 encode_message(const u8* msg, int bits) noexcept {
    Poly256 m;
    for (int i=0;i<bits&&i<256;++i)
        m[i]=((msg[i/8]>>(i%8))&1)?(k_kyber_q/2):0;
    return m;
}

[[nodiscard]] constexpr u8 decode_coeff(i32 c) noexcept {
    // Round to nearest of 0 or q/2
    i32 threshold=k_kyber_q/4;
    i32 d=c%k_kyber_q; if(d<0) d+=k_kyber_q;
    return (d>threshold && d<k_kyber_q-threshold)?1:0;
}

// ─── LWE hardness parameters ─────────────────────────────────────────────────
struct LweParams {
    i32 n{256};    // dimension
    i32 q{3329};   // modulus
    f32 sigma{1.0f}; // Gaussian error std dev
};

// Centered binomial distribution sample (Kyber CBD eta=2/3)
[[nodiscard]] constexpr i32 cbd_sample(u8 byte_pair) noexcept {
    // CBD_eta2: sum of 2 bits minus sum of 2 bits
    i32 a=(byte_pair&1)+((byte_pair>>1)&1);
    i32 b=((byte_pair>>2)&1)+((byte_pair>>3)&1);
    return a-b;
}

// Sample polynomial from CBD using pseudorandom bytes
[[nodiscard]] constexpr Poly256 sample_poly_cbd(const u8* seed, int eta=2) noexcept {
    Poly256 p;
    for (int i=0;i<256;++i) {
        i32 v=0;
        for (int j=0;j<eta;++j)  v+=(seed[(i*2*eta+j)/8]>>((i*2*eta+j)%8))&1;
        for (int j=0;j<eta;++j)  v-=(seed[(i*2*eta+eta+j)/8]>>((i*2*eta+eta+j)%8))&1;
        p[i]=mod_q(v,k_kyber_q);
    }
    return p;
}

// ─── Matrix-vector product over ring (MLWE) ───────────────────────────────────
template<int K>
[[nodiscard]] constexpr std::array<Poly256,K>
mlwe_mat_vec(const std::array<std::array<Poly256,K>,K>& A,
             const std::array<Poly256,K>& s) noexcept {
    std::array<Poly256,K> result{};
    for (int i=0;i<K;++i)
        for (int j=0;j<K;++j) {
            auto prod=ntt_mul(A[i][j],s[j]);
            result[i]=poly_add(result[i],prod);
        }
    return result;
}

} // namespace qs::math
