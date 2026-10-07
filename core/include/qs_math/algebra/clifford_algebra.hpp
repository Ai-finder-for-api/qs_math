#pragma once
// qs_math/clifford_algebra.hpp — Clifford Cl(p,q) basis elements and products (C++20)
// General Clifford algebra over a vector space with signature (p,q): p positive, q negative
// basis vectors. Covers all special cases: complex, quaternions, Cl(3,0), PGA, etc.

#include "scalar_types.hpp"
#include <array>
#include <bitset>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

namespace qs {

// ─────────────────────────────────────────────────────────────────────────────
// BasisBlade — a single basis blade represented as a bitmask over basis vectors
// e.g. in Cl(3,0): e1=0b001, e2=0b010, e3=0b100, e12=0b011, e123=0b111
// ─────────────────────────────────────────────────────────────────────────────
struct BasisBlade {
    u32 bitmap{0}; // which basis vectors are in this blade
    i8  sign{1};   // +1 or -1

    constexpr BasisBlade() = default;
    constexpr BasisBlade(u32 bm, i8 s = 1) : bitmap(bm), sign(s) {}

    [[nodiscard]] constexpr u32 grade() const { return u32(__builtin_popcount(bitmap)); }

    // Compute the sign from reordering to canonical form using bubble sort count
    [[nodiscard]] static constexpr i8 canonical_reorder_sign(u32 a, u32 b) {
        // Count the number of swaps needed to merge blade a (bits) past blade b (bits)
        u32 sum = 0;
        a >>= 1;
        while (a) { sum += u32(__builtin_popcount(a & b)); a >>= 1; }
        return (sum & 1u) ? -1 : 1;
    }

    // Geometric product of two basis blades in Cl(p,q)
    // p = number of basis vectors squaring to +1 (indices 0..p-1)
    // q = number squaring to -1 (indices p..p+q-1)
    [[nodiscard]] static constexpr BasisBlade geo_product(BasisBlade a, BasisBlade b, u32 p, u32 q) {
        i8 s = i8(a.sign * b.sign * canonical_reorder_sign(a.bitmap, b.bitmap));
        u32 result = a.bitmap ^ b.bitmap; // XOR gives the surviving basis vectors
        // For bits that were in both (AND mask) — they square; apply metric signature
        u32 shared = a.bitmap & b.bitmap;
        while (shared) {
            u32 bit = shared & (-i32(shared)); // LSB
            u32 idx = u32(__builtin_ctz(bit));
            if (idx >= p && idx < p + q) s = i8(-s); // negative-squaring basis vector
            shared &= shared - 1;
        }
        return {result, s};
    }

    // Outer (wedge) product: zero if blades share a basis vector
    [[nodiscard]] static constexpr BasisBlade outer_product(BasisBlade a, BasisBlade b) {
        if (a.bitmap & b.bitmap) return {0, 0}; // shared basis vector → zero
        i8 s = i8(a.sign * b.sign * canonical_reorder_sign(a.bitmap, b.bitmap));
        return {a.bitmap | b.bitmap, s};
    }

    // String representation e.g. "e13" or "1" or "e123"
    [[nodiscard]] std::string to_string(u32 n_dims) const {
        if (bitmap == 0) return "1";
        std::string s = "e";
        for (u32 i = 0; i < n_dims; ++i)
            if (bitmap & (1u << i)) s += char('1' + i);
        return s;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CliffordAlgebra<N,T> — general Cl(p,q) multivector with N=p+q dimensions
// 2^N basis blades total.
// ─────────────────────────────────────────────────────────────────────────────
template<u32 N, typename T = f32>
struct CliffordMV {
    static_assert(N >= 1 && N <= 8, "CliffordMV supports N=1..8 dimensions");
    static constexpr u32 DIM    = N;
    static constexpr u32 BLADES = 1u << N;

    std::array<T, (1u << N)> c{}; // indexed by blade bitmap

    constexpr CliffordMV() { c.fill(T(0)); }

    [[nodiscard]] constexpr T& operator[](u32 bitmap)       { return c[bitmap]; }
    [[nodiscard]] constexpr T  operator[](u32 bitmap) const { return c[bitmap]; }

    // Grade extraction
    [[nodiscard]] constexpr CliffordMV grade(u32 g) const {
        CliffordMV r;
        for (u32 i = 0; i < BLADES; ++i)
            if (BasisBlade(i).grade() == g) r.c[i] = c[i];
        return r;
    }

    // Max grade with nonzero coefficient
    [[nodiscard]] constexpr u32 top_grade() const {
        for (i32 g = i32(N); g >= 0; --g)
            for (u32 i = 0; i < BLADES; ++i)
                if (BasisBlade(i).grade() == u32(g) && c[i] != T(0)) return u32(g);
        return 0;
    }

    // Reverse: negate grade-2 and grade-3 blades (grade k → sign (-1)^{k(k-1)/2})
    [[nodiscard]] constexpr CliffordMV reverse() const {
        CliffordMV r = *this;
        for (u32 i = 0; i < BLADES; ++i) {
            u32 g = BasisBlade(i).grade();
            u32 flips = g * (g - 1) / 2;
            if (flips & 1u) r.c[i] = -r.c[i];
        }
        return r;
    }

    // Grade involution: negate odd grades
    [[nodiscard]] constexpr CliffordMV grade_involution() const {
        CliffordMV r = *this;
        for (u32 i = 0; i < BLADES; ++i)
            if (BasisBlade(i).grade() & 1u) r.c[i] = -r.c[i];
        return r;
    }

    // Clifford conjugate
    [[nodiscard]] constexpr CliffordMV clifford_conj() const { return grade_involution().reverse(); }

    // Norm squared: <M ~M>_0 (scalar part of M * reverse(M))
    [[nodiscard]] constexpr T norm_sq(u32 p, u32 q) const {
        return geo_product(*this, reverse(), p, q).c[0];
    }
    [[nodiscard]] T norm(u32 p, u32 q) const { return std::sqrt(std::abs(norm_sq(p, q))); }

    // Basic arithmetic
    [[nodiscard]] constexpr CliffordMV operator+(const CliffordMV& b) const {
        CliffordMV r; for(u32 i=0;i<BLADES;++i) r.c[i]=c[i]+b.c[i]; return r;
    }
    [[nodiscard]] constexpr CliffordMV operator-(const CliffordMV& b) const {
        CliffordMV r; for(u32 i=0;i<BLADES;++i) r.c[i]=c[i]-b.c[i]; return r;
    }
    [[nodiscard]] constexpr CliffordMV operator*(T s) const {
        CliffordMV r; for(u32 i=0;i<BLADES;++i) r.c[i]=c[i]*s; return r;
    }
    [[nodiscard]] constexpr CliffordMV operator-() const {
        CliffordMV r; for(u32 i=0;i<BLADES;++i) r.c[i]=-c[i]; return r;
    }

    // Geometric product with signature (p,q)
    [[nodiscard]] static CliffordMV geo_product(const CliffordMV& A, const CliffordMV& B, u32 p, u32 q) {
        CliffordMV r;
        for (u32 i = 0; i < BLADES; ++i) {
            if (A.c[i] == T(0)) continue;
            for (u32 j = 0; j < BLADES; ++j) {
                if (B.c[j] == T(0)) continue;
                auto blade = BasisBlade::geo_product(BasisBlade(i), BasisBlade(j), p, q);
                if (blade.sign != 0)
                    r.c[blade.bitmap] += T(blade.sign) * A.c[i] * B.c[j];
            }
        }
        return r;
    }

    // Outer (wedge) product
    [[nodiscard]] static CliffordMV outer_product(const CliffordMV& A, const CliffordMV& B) {
        CliffordMV r;
        for (u32 i = 0; i < BLADES; ++i) {
            if (A.c[i] == T(0)) continue;
            for (u32 j = 0; j < BLADES; ++j) {
                if (B.c[j] == T(0)) continue;
                auto blade = BasisBlade::outer_product(BasisBlade(i), BasisBlade(j));
                if (blade.sign != 0)
                    r.c[blade.bitmap] += T(blade.sign) * A.c[i] * B.c[j];
            }
        }
        return r;
    }

    // Left contraction
    [[nodiscard]] static CliffordMV left_contract(const CliffordMV& A, const CliffordMV& B, u32 p, u32 q) {
        CliffordMV r;
        for (u32 i = 0; i < BLADES; ++i) {
            if (A.c[i] == T(0)) continue;
            u32 ga = BasisBlade(i).grade();
            for (u32 j = 0; j < BLADES; ++j) {
                if (B.c[j] == T(0)) continue;
                u32 gb = BasisBlade(j).grade();
                if (gb < ga) continue;
                auto blade = BasisBlade::geo_product(BasisBlade(i), BasisBlade(j), p, q);
                if (blade.sign != 0 && BasisBlade(blade.bitmap).grade() == gb - ga)
                    r.c[blade.bitmap] += T(blade.sign) * A.c[i] * B.c[j];
            }
        }
        return r;
    }

    // Dual (Hodge): *A = A · I^{-1}
    [[nodiscard]] CliffordMV dual(u32 p, u32 q) const {
        u32 pseudo_bitmap = BLADES - 1u; // all bits set = e12...N
        CliffordMV I;
        I.c[pseudo_bitmap] = T(1);
        // Compute I^{-1}
        auto Irev = I.reverse();
        T Inorm = geo_product(I, Irev, p, q).c[0];
        CliffordMV Iinv = Irev * (T(1) / Inorm);
        return geo_product(*this, Iinv, p, q);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pre-defined algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

// Cl(0,1) = complex numbers: signature (0,1), basis {1, e1} with e1²=-1
using ClComplex = CliffordMV<1, f32>;
[[nodiscard]] inline ClComplex make_complex(f32 re, f32 im) {
    ClComplex c; c[0b00] = re; c[0b01] = im; return c;
}

// Cl(0,2) = quaternions: signature (0,2), basis {1,e1,e2,e12} with e1²=e2²=-1
using ClQuat = CliffordMV<2, f32>;
[[nodiscard]] inline ClQuat make_clquat(f32 s, f32 i, f32 j, f32 k) {
    ClQuat q; q[0b00]=s; q[0b01]=i; q[0b10]=j; q[0b11]=k; return q;
}

// Cl(3,0) = 3D VGA: signature (3,0)
using ClVGA3 = CliffordMV<3, f32>;

// Cl(3,1) = STA (Space-Time Algebra / Minkowski): signature (3,1), dim=4, 16 blades
using ClSTA = CliffordMV<4, f32>;

// Cl(3,0,1) surrogate — use CliffordMV<4> with idx 3 as the degenerate direction
// (full PGA requires different metric handling — here we provide the structure)
using ClPGA = CliffordMV<4, f32>;

// ─────────────────────────────────────────────────────────────────────────────
// Structure constants: generate the full multiplication table for Cl(p,q)
// Useful for exporting to other systems or for JIT code generation
// ─────────────────────────────────────────────────────────────────────────────
template<u32 N>
struct CliffordTable {
    static constexpr u32 BLADES = 1u << N;
    // result_index[i][j] and result_sign[i][j] for geo product
    std::array<std::array<u32, 1<<N>, 1<<N> result_index{};
    std::array<std::array<i8,  1<<N>, 1<<N> result_sign{};

    CliffordTable(u32 p, u32 q) {
        for (u32 i = 0; i < BLADES; ++i)
            for (u32 j = 0; j < BLADES; ++j) {
                auto b = BasisBlade::geo_product(BasisBlade(i), BasisBlade(j), p, q);
                result_index[i][j] = b.bitmap;
                result_sign[i][j]  = b.sign;
            }
    }
};

} // namespace qs
