#pragma once
// qs_math/cryptography/sha_math.hpp — SHA-2 family internals (SHA-256, SHA-512)
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bit>
#include <cstring>

namespace qs::math {

// ─── SHA-256 constants ────────────────────────────────────────────────────────
inline constexpr std::array<u32,64> k_sha256_k = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline constexpr std::array<u32,8> k_sha256_iv = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

// SHA-256 round functions
[[nodiscard]] constexpr u32 sha256_ch (u32 e,u32 f,u32 g) noexcept { return (e&f)^(~e&g); }
[[nodiscard]] constexpr u32 sha256_maj(u32 a,u32 b,u32 c) noexcept { return (a&b)^(a&c)^(b&c); }
[[nodiscard]] constexpr u32 sha256_S0 (u32 a) noexcept { return std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22); }
[[nodiscard]] constexpr u32 sha256_S1 (u32 e) noexcept { return std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25); }
[[nodiscard]] constexpr u32 sha256_s0 (u32 w) noexcept { return std::rotr(w,7) ^std::rotr(w,18)^(w>>3);  }
[[nodiscard]] constexpr u32 sha256_s1 (u32 w) noexcept { return std::rotr(w,17)^std::rotr(w,19)^(w>>10); }

struct Sha256State { std::array<u32,8> h = k_sha256_iv; u64 total_bits{0}; };

inline void sha256_compress(Sha256State& st, const u8 block[64]) noexcept {
    u32 w[64];
    for (int i=0;i<16;++i)
        w[i]=(u32(block[i*4])<<24)|(u32(block[i*4+1])<<16)|(u32(block[i*4+2])<<8)|block[i*4+3];
    for (int i=16;i<64;++i)
        w[i]=sha256_s1(w[i-2])+w[i-7]+sha256_s0(w[i-15])+w[i-16];
    u32 a=st.h[0],b=st.h[1],c=st.h[2],d=st.h[3],e=st.h[4],f=st.h[5],g=st.h[6],h=st.h[7];
    for (int i=0;i<64;++i) {
        u32 t1=h+sha256_S1(e)+sha256_ch(e,f,g)+k_sha256_k[i]+w[i];
        u32 t2=sha256_S0(a)+sha256_maj(a,b,c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    st.h[0]+=a;st.h[1]+=b;st.h[2]+=c;st.h[3]+=d;
    st.h[4]+=e;st.h[5]+=f;st.h[6]+=g;st.h[7]+=h;
}

// One-shot SHA-256
[[nodiscard]] inline std::array<u8,32> sha256(const u8* data, u64 len) noexcept {
    Sha256State st;
    u64 i=0;
    // Process full blocks
    for (; i+64<=len; i+=64) sha256_compress(st, data+i);
    // Padding
    u8 pad[128]{};
    u64 rem=len-i;
    std::memcpy(pad, data+i, rem);
    pad[rem]=0x80;
    u64 bits=len*8;
    int pad_len=(rem<56)?64:128;
    for (int j=7;j>=0;--j) { pad[pad_len-1-j]=u8(bits&0xff); bits>>=8; }
    for (int b=0;b<pad_len;b+=64) sha256_compress(st, pad+b);
    std::array<u8,32> out{};
    for (int j=0;j<8;++j) {
        out[j*4+0]=u8(st.h[j]>>24);out[j*4+1]=u8(st.h[j]>>16);
        out[j*4+2]=u8(st.h[j]>>8); out[j*4+3]=u8(st.h[j]);
    }
    return out;
}

// ─── SHA-512 constants ────────────────────────────────────────────────────────
inline constexpr std::array<u64,80> k_sha512_k = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL
};

inline constexpr std::array<u64,8> k_sha512_iv = {
    0x6a09e667f3bcc908ULL,0xbb67ae8584caa73bULL,0x3c6ef372fe94f82bULL,0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL,0x9b05688c2b3e6c1fULL,0x1f83d9abfb41bd6bULL,0x5be0cd19137e2179ULL
};

[[nodiscard]] constexpr u64 sha512_ch (u64 e,u64 f,u64 g) noexcept { return (e&f)^(~e&g); }
[[nodiscard]] constexpr u64 sha512_maj(u64 a,u64 b,u64 c) noexcept { return (a&b)^(a&c)^(b&c); }
[[nodiscard]] constexpr u64 sha512_S0 (u64 a) noexcept { return std::rotr(a,28)^std::rotr(a,34)^std::rotr(a,39); }
[[nodiscard]] constexpr u64 sha512_S1 (u64 e) noexcept { return std::rotr(e,14)^std::rotr(e,18)^std::rotr(e,41); }
[[nodiscard]] constexpr u64 sha512_s0 (u64 w) noexcept { return std::rotr(w,1) ^std::rotr(w,8) ^(w>>7);  }
[[nodiscard]] constexpr u64 sha512_s1 (u64 w) noexcept { return std::rotr(w,19)^std::rotr(w,61)^(w>>6);  }

struct Sha512State { std::array<u64,8> h = k_sha512_iv; };

inline void sha512_compress(Sha512State& st, const u8 block[128]) noexcept {
    u64 w[80];
    for (int i=0;i<16;++i) {
        w[i]=0;
        for (int j=0;j<8;++j) w[i]=(w[i]<<8)|block[i*8+j];
    }
    for (int i=16;i<80;++i)
        w[i]=sha512_s1(w[i-2])+w[i-7]+sha512_s0(w[i-15])+w[i-16];
    u64 a=st.h[0],b=st.h[1],c=st.h[2],d=st.h[3],e=st.h[4],f=st.h[5],g=st.h[6],h=st.h[7];
    for (int i=0;i<80;++i) {
        u64 t1=h+sha512_S1(e)+sha512_ch(e,f,g)+k_sha512_k[i]+w[i];
        u64 t2=sha512_S0(a)+sha512_maj(a,b,c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    st.h[0]+=a;st.h[1]+=b;st.h[2]+=c;st.h[3]+=d;
    st.h[4]+=e;st.h[5]+=f;st.h[6]+=g;st.h[7]+=h;
}

[[nodiscard]] inline std::array<u8,64> sha512(const u8* data, u64 len) noexcept {
    Sha512State st;
    u64 i=0;
    for (; i+128<=len; i+=128) sha512_compress(st, data+i);
    u8 pad[256]{};
    u64 rem=len-i;
    std::memcpy(pad,data+i,rem);
    pad[rem]=0x80;
    u64 bits=len*8;
    int pad_len=(rem<112)?128:256;
    // 128-bit length (we only use lower 64 bits here)
    for (int j=7;j>=0;--j) { pad[pad_len-1-j]=u8(bits&0xff); bits>>=8; }
    for (int b=0;b<pad_len;b+=128) sha512_compress(st,pad+b);
    std::array<u8,64> out{};
    for (int j=0;j<8;++j)
        for (int k=7;k>=0;--k) { out[j*8+7-k]=u8(st.h[j]&0xff); st.h[j]>>=8; }
    return out;
}

// ─── HMAC-SHA256 ──────────────────────────────────────────────────────────────
[[nodiscard]] inline std::array<u8,32> hmac_sha256(const u8* key, u64 klen,
                                                     const u8* msg, u64 mlen) noexcept {
    u8 k[64]{};
    if (klen>64) { auto h=sha256(key,klen); std::memcpy(k,h.data(),32); }
    else std::memcpy(k,key,klen);
    u8 ipad[64], opad[64];
    for (int i=0;i<64;++i){ipad[i]=k[i]^0x36;opad[i]=k[i]^0x5c;}
    // inner: sha256(ipad||msg)
    u8 inner_data[64+mlen]; // VLA for simplicity (bounded by caller)
    std::memcpy(inner_data,ipad,64);
    std::memcpy(inner_data+64,msg,mlen);
    auto inner=sha256(inner_data,64+mlen);
    // outer: sha256(opad||inner)
    u8 outer_data[64+32];
    std::memcpy(outer_data,opad,64);
    std::memcpy(outer_data+64,inner.data(),32);
    return sha256(outer_data,64+32);
}

// ─── HKDF-SHA256 ──────────────────────────────────────────────────────────────
[[nodiscard]] inline std::array<u8,32> hkdf_extract(const u8* salt, u64 slen,
                                                      const u8* ikm,  u64 ilen) noexcept {
    static const u8 zero_salt[32]{};
    if (!salt||slen==0) { salt=zero_salt; slen=32; }
    return hmac_sha256(salt,slen,ikm,ilen);
}

} // namespace qs::math
