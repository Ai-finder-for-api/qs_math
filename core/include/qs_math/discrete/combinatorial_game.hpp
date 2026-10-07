#pragma once
// qs_math/discrete/combinatorial_game.hpp — Sprague-Grundy, Nim, game sums
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bitset>

namespace qs::math {

// ─── Nim value (Grundy number) ────────────────────────────────────────────────
// mex: minimum excludant of a set encoded as bitmask
[[nodiscard]] constexpr i32 mex(u64 reachable_mask) noexcept {
    for(int i=0;i<64;++i) if(!((reachable_mask>>i)&1)) return i;
    return 64;
}

// Grundy for single Nim heap of size n
[[nodiscard]] constexpr i32 nim_grundy(i32 n) noexcept { return n; } // G(n)=n

// Nim game: XOR of all heap sizes
[[nodiscard]] constexpr i32 nim_value(const i32* heaps, int count) noexcept {
    i32 xorsum=0;
    for(int i=0;i<count;++i) xorsum^=heaps[i];
    return xorsum;
}

[[nodiscard]] constexpr bool nim_first_player_wins(const i32* heaps, int count) noexcept {
    return nim_value(heaps,count)!=0;
}

// Optimal nim move: find heap to reduce so xorsum becomes 0
[[nodiscard]] constexpr int nim_winning_move(const i32* heaps, int count,
                                              i32* new_heaps) noexcept {
    i32 xsum=nim_value(heaps,count);
    for(int i=0;i<count;++i) new_heaps[i]=heaps[i];
    if(xsum==0) return -1; // losing position
    for(int i=0;i<count;++i) {
        i32 target=heaps[i]^xsum;
        if(target<heaps[i]){ new_heaps[i]=target; return i; }
    }
    return -1;
}

// ─── Sprague-Grundy for generic impartial games ───────────────────────────────
// Game state precomputed: grundy[state] = Grundy number
// Subtraction game S = {s1,s2,...} from heap of size n
template<int MAXN, int K>
[[nodiscard]] constexpr std::array<i32,MAXN>
subtraction_game_grundy(const std::array<i32,K>& moves) noexcept {
    std::array<i32,MAXN> G{};
    G[0]=0; // losing position (no moves)
    for(int n=1;n<MAXN;++n) {
        u64 reach=0;
        for(int k=0;k<K;++k) if(n>=moves[k]) reach|=(1ULL<<G[n-moves[k]]);
        G[n]=mex(reach);
    }
    return G;
}

// Game sum: G(A+B) = G(A) XOR G(B)
[[nodiscard]] constexpr i32 game_sum_grundy(i32 gA, i32 gB) noexcept { return gA^gB; }

// ─── Wythoff's game: two-heap game with L-moves ───────────────────────────────
// P-position: (floor(k*phi), floor(k*phi^2)) for k=0,1,2,...
// phi = (1+sqrt(5))/2, phi^2 = phi+1
[[nodiscard]] constexpr bool wythoff_losing(i32 a, i32 b) noexcept {
    if(a>b){i32 t=a;a=b;b=t;}
    // Use integer approximation: k = b-a, check if a == floor(k*phi)
    i32 k=b-a;
    // phi ≈ 1.6180339887, use rational approx 89/55 = 1.6181...
    // Better: (a,b) is P-pos iff a = floor(k*(1+sqrt5)/2)
    // We implement via: a*(a+1)*2 vs k*(k+1)+k*k (avoids float for small n)
    // For exactness with small values, use direct golden ratio check:
    // floor(k*phi) = floor((k*(1+sqrt5))/2)
    // Using: floor(k*phi) can be computed as (k + isqrt(5*k*k))/2
    i64 k2=(i64)k*k;
    i64 s5k2=5*k2;
    // Integer sqrt of s5k2
    i64 sq=(i64)(f64(s5k2));
    while(sq*sq>s5k2) --sq;
    while((sq+1)*(sq+1)<=s5k2) ++sq;
    i64 approx=(k+sq)/2;
    return a==approx;
}

// ─── Kayles (peg solitaire variant) Grundy values ────────────────────────────
// Kayles on a row of n pins: can knock 1 or 2 adjacent pins
template<int MAXN>
[[nodiscard]] constexpr std::array<i32,MAXN> kayles_grundy() noexcept {
    std::array<i32,MAXN> G{};
    // Known periodic with period 12 after n=71
    // Compute directly for all n up to MAXN
    for(int n=0;n<MAXN;++n) {
        u64 reach=0;
        // Move: remove pin i alone (splits into two sub-rows)
        for(int i=0;i<n;++i) {
            i32 left=i, right=n-i-1;
            reach|=(1ULL<<(G[left]^G[right]));
        }
        // Move: remove pins i,i+1 together
        for(int i=0;i<n-1;++i) {
            i32 left=i, right=n-i-2;
            reach|=(1ULL<<(G[left]^G[right]));
        }
        G[n]=mex(reach);
    }
    return G;
}

// ─── Hackenbush (combinatorial game theory) ──────────────────────────────────
// Blue-Red Hackenbush edge game value (integer for all-blue graphs)
// For a path of k blue edges: value = k (integer surreal number)
// For a path with sign alternation: standard CGT temperature analysis

struct HackenbushValue {
    i32 numerator{0};
    i32 denominator{1}; // power of 2 (dyadic rational)

    [[nodiscard]] constexpr f64 to_f64() const noexcept {
        return f64(numerator)/f64(denominator);
    }
};

[[nodiscard]] constexpr HackenbushValue hackenbush_add(
    HackenbushValue a, HackenbushValue b) noexcept
{
    // Common denominator (LCM of powers of 2 = max)
    i32 denom = a.denominator > b.denominator ? a.denominator : b.denominator;
    i32 num   = a.numerator*(denom/a.denominator) + b.numerator*(denom/b.denominator);
    // Simplify
    while(denom>1 && num%2==0){num/=2;denom/=2;}
    return {num,denom};
}

// ─── Combinatorial game temperature ───────────────────────────────────────────
// Temperature of G = {GL|GR}: t = (value(GR) - value(GL))/2 at thermograph intersection
[[nodiscard]] constexpr f64 game_temperature(f64 left_val, f64 right_val) noexcept {
    return (right_val - left_val) / 2.0;
}

// Mean value of game G = {GL|GR}
[[nodiscard]] constexpr f64 game_mean(f64 left_val, f64 right_val) noexcept {
    return (left_val + right_val) / 2.0;
}

// ─── Surreal numbers (Conway) — finite representation ────────────────────────
struct Surreal {
    i32 num{0}, den{1}; // dyadic rational p/2^k, den is power of 2

    static constexpr Surreal zero()     noexcept { return {0,1}; }
    static constexpr Surreal one()      noexcept { return {1,1}; }
    static constexpr Surreal neg_one()  noexcept { return {-1,1}; }
    static constexpr Surreal half()     noexcept { return {1,2}; }

    [[nodiscard]] constexpr f64 to_f64() const noexcept { return f64(num)/f64(den); }
    [[nodiscard]] constexpr bool operator==(const Surreal& o) const noexcept {
        return num*o.den == o.num*den;
    }
    [[nodiscard]] constexpr bool operator<(const Surreal& o) const noexcept {
        return num*o.den < o.num*den;
    }
};

[[nodiscard]] constexpr Surreal surreal_add(Surreal a, Surreal b) noexcept {
    i32 d=a.den>b.den?a.den:b.den;
    i32 n=a.num*(d/a.den)+b.num*(d/b.den);
    while(d>1&&n%2==0){n/=2;d/=2;}
    return {n,d};
}

[[nodiscard]] constexpr Surreal surreal_neg(Surreal a) noexcept { return {-a.num,a.den}; }

[[nodiscard]] constexpr Surreal surreal_mul(Surreal a, Surreal b) noexcept {
    return {a.num*b.num, a.den*b.den}; // may overflow for large; fine for games
}

// Simplest surreal in (lo, hi)
[[nodiscard]] constexpr Surreal surreal_simplest(Surreal lo, Surreal hi) noexcept {
    // If an integer is in range, return smallest magnitude integer in (lo,hi)
    i32 lo_ceil = (lo.num + lo.den) / lo.den; // ceil(lo)
    i32 hi_floor= hi.num / hi.den;            // floor(hi)
    if(lo_ceil <= hi_floor) {
        i32 best = (lo_ceil<=0 && hi_floor>=0)?0:(lo_ceil>0?lo_ceil:hi_floor);
        return {best,1};
    }
    // Otherwise recurse with halved interval (birthday argument)
    return {(lo.num*hi.den + hi.num*lo.den)/(2*lo.den*hi.den), 1};
}

} // namespace qs::math
