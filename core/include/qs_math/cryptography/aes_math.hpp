#pragma once
// qs_math/cryptography/aes_math.hpp — AES GF(2^8) field, S-box, key schedule, round math (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bit>
#include <cstring>

namespace qs::math {

// ─── GF(2^8) with AES polynomial x^8+x^4+x^3+x+1 (0x11b) ───────────────────

[[nodiscard]] constexpr u8 gf_mul(u8 a, u8 b) noexcept {
    u8 p = 0;
    for (int i = 0; i < 8; ++i) {
        if (b & 1) p ^= a;
        bool hi = (a & 0x80) != 0;
        a = static_cast<u8>(a << 1);
        if (hi) a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

[[nodiscard]] constexpr u8 gf_pow(u8 base, u8 exp) noexcept {
    u8 result = 1;
    for (int i = 0; i < exp; ++i) result = gf_mul(result, base);
    return result;
}

[[nodiscard]] constexpr u8 gf_inv(u8 a) noexcept {
    if (a == 0) return 0;
    u8 r = 1;
    for (int i = 0; i < 254; ++i) r = gf_mul(r, a); // Fermat: a^(256-2)
    return r;
}

// ─── S-box generation ─────────────────────────────────────────────────────────

[[nodiscard]] constexpr u8 aes_sbox_entry(u8 a) noexcept {
    u8 b = gf_inv(a);
    u8 c = 0x63;
    for (int i = 0; i < 8; ++i)
        c ^= static_cast<u8>(((b >> i) & 1) << i)
           ^ static_cast<u8>(((b >> ((i+4)%8)) & 1) << i)
           ^ static_cast<u8>(((b >> ((i+5)%8)) & 1) << i)
           ^ static_cast<u8>(((b >> ((i+6)%8)) & 1) << i)
           ^ static_cast<u8>(((b >> ((i+7)%8)) & 1) << i);
    // Standard affine: s = b XOR rot(b,1) XOR rot(b,2) XOR rot(b,3) XOR rot(b,4) XOR 0x63
    u8 s = b ^ std::rotl(b,1) ^ std::rotl(b,2) ^ std::rotl(b,3) ^ std::rotl(b,4) ^ u8(0x63);
    return s;
}

using AesSbox    = std::array<u8,256>;
using AesSboxInv = std::array<u8,256>;

[[nodiscard]] constexpr AesSbox build_sbox() noexcept {
    AesSbox t{};
    for (int i = 0; i < 256; ++i)
        t[i] = aes_sbox_entry(static_cast<u8>(i));
    return t;
}

[[nodiscard]] constexpr AesSboxInv build_sbox_inv(const AesSbox& s) noexcept {
    AesSboxInv t{};
    for (int i = 0; i < 256; ++i) t[s[i]] = static_cast<u8>(i);
    return t;
}

inline constexpr AesSbox    k_aes_sbox     = build_sbox();
inline constexpr AesSboxInv k_aes_sbox_inv = build_sbox_inv(k_aes_sbox);

// ─── Round constant (RCON) ────────────────────────────────────────────────────

[[nodiscard]] constexpr u8 aes_rcon(u8 i) noexcept {
    u8 c = 1;
    for (u8 j = 1; j < i; ++j) c = gf_mul(c, 2);
    return (i == 0) ? 0 : c;
}

// ─── AES state (4×4 bytes, column-major) ──────────────────────────────────────

struct AesState {
    u8 s[4][4]{};  // s[row][col]

    [[nodiscard]] constexpr u8  at(int r, int c) const noexcept { return s[r][c]; }
    [[nodiscard]] constexpr u8& at(int r, int c)       noexcept { return s[r][c]; }

    // Load 16-byte block (row-major input)
    constexpr void load(const u8* in) noexcept {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                s[r][c] = in[r*4+c];
    }
    constexpr void store(u8* out) const noexcept {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                out[r*4+c] = s[r][c];
    }
};

// ─── SubBytes ─────────────────────────────────────────────────────────────────

[[nodiscard]] inline AesState sub_bytes(AesState st) noexcept {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            st.s[r][c] = k_aes_sbox[st.s[r][c]];
    return st;
}

[[nodiscard]] inline AesState sub_bytes_inv(AesState st) noexcept {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            st.s[r][c] = k_aes_sbox_inv[st.s[r][c]];
    return st;
}

// ─── ShiftRows ────────────────────────────────────────────────────────────────

[[nodiscard]] constexpr AesState shift_rows(AesState st) noexcept {
    for (int r = 1; r < 4; ++r) {
        u8 tmp[4];
        for (int c = 0; c < 4; ++c) tmp[c] = st.s[r][(c+r)%4];
        for (int c = 0; c < 4; ++c) st.s[r][c] = tmp[c];
    }
    return st;
}

[[nodiscard]] constexpr AesState shift_rows_inv(AesState st) noexcept {
    for (int r = 1; r < 4; ++r) {
        u8 tmp[4];
        for (int c = 0; c < 4; ++c) tmp[c] = st.s[r][(c+4-r)%4];
        for (int c = 0; c < 4; ++c) st.s[r][c] = tmp[c];
    }
    return st;
}

// ─── MixColumns ───────────────────────────────────────────────────────────────

[[nodiscard]] inline AesState mix_columns(AesState st) noexcept {
    for (int c = 0; c < 4; ++c) {
        u8 a = st.s[0][c], b = st.s[1][c], d = st.s[2][c], e = st.s[3][c];
        st.s[0][c] = gf_mul(2,a) ^ gf_mul(3,b) ^ d ^ e;
        st.s[1][c] = a ^ gf_mul(2,b) ^ gf_mul(3,d) ^ e;
        st.s[2][c] = a ^ b ^ gf_mul(2,d) ^ gf_mul(3,e);
        st.s[3][c] = gf_mul(3,a) ^ b ^ d ^ gf_mul(2,e);
    }
    return st;
}

[[nodiscard]] inline AesState mix_columns_inv(AesState st) noexcept {
    for (int c = 0; c < 4; ++c) {
        u8 a = st.s[0][c], b = st.s[1][c], d = st.s[2][c], e = st.s[3][c];
        st.s[0][c] = gf_mul(0x0e,a) ^ gf_mul(0x0b,b) ^ gf_mul(0x0d,d) ^ gf_mul(0x09,e);
        st.s[1][c] = gf_mul(0x09,a) ^ gf_mul(0x0e,b) ^ gf_mul(0x0b,d) ^ gf_mul(0x0d,e);
        st.s[2][c] = gf_mul(0x0d,a) ^ gf_mul(0x09,b) ^ gf_mul(0x0e,d) ^ gf_mul(0x0b,e);
        st.s[3][c] = gf_mul(0x0b,a) ^ gf_mul(0x0d,b) ^ gf_mul(0x09,d) ^ gf_mul(0x0e,e);
    }
    return st;
}

// ─── AddRoundKey ──────────────────────────────────────────────────────────────

[[nodiscard]] constexpr AesState add_round_key(AesState st, const u8 rk[4][4]) noexcept {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            st.s[r][c] ^= rk[r][c];
    return st;
}

// ─── AES-128 Key Schedule ─────────────────────────────────────────────────────

struct Aes128KeySchedule {
    static constexpr int k_rounds = 10;
    u8 rk[k_rounds+1][4][4]{};
};

[[nodiscard]] inline Aes128KeySchedule aes128_key_schedule(const u8 key[16]) noexcept {
    Aes128KeySchedule ks;
    // Copy original key into round 0
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            ks.rk[0][r][c] = key[r*4+c];

    // 4-word key words
    u8 w[44][4];
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            w[c][r] = key[r + 4*c];

    for (int i = 4; i < 44; ++i) {
        u8 tmp[4];
        for (int j = 0; j < 4; ++j) tmp[j] = w[i-1][j];
        if (i % 4 == 0) {
            // RotWord + SubWord + Rcon
            u8 t = tmp[0];
            tmp[0] = k_aes_sbox[tmp[1]] ^ aes_rcon(static_cast<u8>(i/4));
            tmp[1] = k_aes_sbox[tmp[2]];
            tmp[2] = k_aes_sbox[tmp[3]];
            tmp[3] = k_aes_sbox[t];
        }
        for (int j = 0; j < 4; ++j) w[i][j] = w[i-4][j] ^ tmp[j];
    }

    // Map words back into round-key matrices [round][row][col]
    for (int rnd = 0; rnd <= k_rounds; ++rnd)
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                ks.rk[rnd][r][c] = w[rnd*4+c][r];

    return ks;
}

// ─── AES-128 Block Encrypt / Decrypt ──────────────────────────────────────────

inline void aes128_encrypt_block(const u8 plain[16], const Aes128KeySchedule& ks, u8 cipher[16]) noexcept {
    AesState st;
    st.load(plain);
    st = add_round_key(st, ks.rk[0]);
    for (int rnd = 1; rnd < Aes128KeySchedule::k_rounds; ++rnd) {
        st = sub_bytes(st);
        st = shift_rows(st);
        st = mix_columns(st);
        st = add_round_key(st, ks.rk[rnd]);
    }
    st = sub_bytes(st);
    st = shift_rows(st);
    st = add_round_key(st, ks.rk[Aes128KeySchedule::k_rounds]);
    st.store(cipher);
}

inline void aes128_decrypt_block(const u8 cipher[16], const Aes128KeySchedule& ks, u8 plain[16]) noexcept {
    AesState st;
    st.load(cipher);
    st = add_round_key(st, ks.rk[Aes128KeySchedule::k_rounds]);
    for (int rnd = Aes128KeySchedule::k_rounds - 1; rnd >= 1; --rnd) {
        st = shift_rows_inv(st);
        st = sub_bytes_inv(st);
        st = add_round_key(st, ks.rk[rnd]);
        st = mix_columns_inv(st);
    }
    st = shift_rows_inv(st);
    st = sub_bytes_inv(st);
    st = add_round_key(st, ks.rk[0]);
    st.store(plain);
}

// ─── CTR mode helper ──────────────────────────────────────────────────────────

inline void aes128_ctr_keystream(const Aes128KeySchedule& ks, u8 nonce[16], u8 out[16]) noexcept {
    aes128_encrypt_block(nonce, ks, out);
    // Increment nonce (big-endian 128-bit counter)
    for (int i = 15; i >= 0; --i) {
        if (++nonce[i]) break;
    }
}

inline void aes128_ctr_process(const Aes128KeySchedule& ks, u8 nonce[16],
                                const u8* in, u8* out, u64 len) noexcept {
    u64 i = 0;
    while (i + 16 <= len) {
        u8 ks_block[16];
        aes128_ctr_keystream(ks, nonce, ks_block);
        for (int j = 0; j < 16; ++j) out[i+j] = in[i+j] ^ ks_block[j];
        i += 16;
    }
    if (i < len) {
        u8 ks_block[16];
        aes128_ctr_keystream(ks, nonce, ks_block);
        for (u64 j = 0; j < len - i; ++j) out[i+j] = in[i+j] ^ ks_block[j];
    }
}

// ─── GCM GHASH (GF(2^128)) ────────────────────────────────────────────────────

struct GF128 { u64 hi{0}, lo{0}; };

[[nodiscard]] inline GF128 gf128_mul(GF128 x, GF128 y) noexcept {
    // Shoup's table-free multiplication, reduction poly = x^128+x^7+x^2+x+1
    GF128 z{};
    for (int i = 127; i >= 0; --i) {
        u64 bit = (i >= 64) ? ((y.hi >> (i-64)) & 1) : ((y.lo >> i) & 1);
        // z = z XOR (x * bit)
        if (bit) { z.hi ^= x.hi; z.lo ^= x.lo; }
        // x = x * alpha (shift right by 1 in GF(2^128))
        bool low_bit = (x.lo & 1) != 0;
        x.lo = (x.lo >> 1) | (x.hi << 63);
        x.hi >>= 1;
        if (low_bit) x.hi ^= 0xE100000000000000ULL;
    }
    return z;
}

inline void ghash(const u8* h, const u8* data, u64 len, u8 tag[16]) noexcept {
    GF128 H{};
    for (int i = 0; i < 8; ++i) H.hi = (H.hi << 8) | h[i];
    for (int i = 0; i < 8; ++i) H.lo = (H.lo << 8) | h[i+8];
    GF128 X{};
    u64 i = 0;
    while (i + 16 <= len) {
        GF128 b{};
        for (int j = 0; j < 8; ++j) b.hi = (b.hi << 8) | data[i+j];
        for (int j = 0; j < 8; ++j) b.lo = (b.lo << 8) | data[i+j+8];
        X.hi ^= b.hi; X.lo ^= b.lo;
        X = gf128_mul(X, H);
        i += 16;
    }
    // Store X into tag
    for (int j = 7; j >= 0; --j) { tag[j]   = static_cast<u8>(X.hi & 0xff); X.hi >>= 8; }
    for (int j = 7; j >= 0; --j) { tag[j+8] = static_cast<u8>(X.lo & 0xff); X.lo >>= 8; }
}

} // namespace qs::math
