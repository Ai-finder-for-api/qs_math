#pragma once
// qs_math/combinatorics/counting.hpp — Permutations, combinations, partitions,
//   Catalan/Bell/Stirling numbers, Burnside's lemma, generating functions (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace qs::math::combinatorics {

// ─── Factorial & binomial (exact, u64) ───────────────────────────────────
[[nodiscard]] constexpr u64 factorial(int n) noexcept {
    u64 r = 1;
    for (int i = 2; i <= n; ++i) r *= u64(i);
    return r;
}

// Pascal's triangle up to n=66 (before u64 overflow)
[[nodiscard]] constexpr u64 binomial(int n, int k) noexcept {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n-k) k = n-k;
    u64 r = 1;
    for (int i = 0; i < k; ++i) {
        r *= u64(n-i);
        r /= u64(i+1);
    }
    return r;
}

// Multinomial coefficient: n! / (k1! * k2! * ... km!)
[[nodiscard]] inline u64 multinomial(int n, const std::vector<int>& ks) noexcept {
    u64 r = factorial(n);
    for (int k : ks) r /= factorial(k);
    return r;
}

// ─── Falling / rising factorials (Pochhammer) ────────────────────────────
[[nodiscard]] constexpr u64 falling_factorial(int n, int k) noexcept {
    u64 r = 1;
    for (int i = 0; i < k; ++i) r *= u64(n-i);
    return r;
}
[[nodiscard]] constexpr u64 rising_factorial(int n, int k) noexcept {
    u64 r = 1;
    for (int i = 0; i < k; ++i) r *= u64(n+i);
    return r;
}

// ─── Catalan numbers C_n = C(2n,n)/(n+1) ────────────────────────────────
[[nodiscard]] constexpr u64 catalan(int n) noexcept {
    return binomial(2*n, n) / u64(n+1);
}

// ─── Bell numbers (number of set partitions of n elements) ────────────────
[[nodiscard]] inline u64 bell_number(int n) noexcept {
    // Bell triangle
    std::vector<std::vector<u64>> tri(n+1, std::vector<u64>(n+1, 0));
    tri[0][0] = 1;
    for (int i = 1; i <= n; ++i) {
        tri[i][0] = tri[i-1][i-1];
        for (int j = 1; j <= i; ++j)
            tri[i][j] = tri[i][j-1] + tri[i-1][j-1];
    }
    return tri[n][0];
}

// ─── Stirling numbers of the second kind S(n,k) ───────────────────────────
// Number of ways to partition n elements into k non-empty subsets
[[nodiscard]] inline u64 stirling2(int n, int k) noexcept {
    if (k == 0) return n == 0 ? 1 : 0;
    if (k > n) return 0;
    std::vector<std::vector<u64>> S(n+1, std::vector<u64>(n+1, 0));
    S[0][0] = 1;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= i; ++j)
            S[i][j] = u64(j)*S[i-1][j] + S[i-1][j-1];
    return S[n][k];
}

// Stirling numbers of the first kind s(n,k) (unsigned)
// Number of permutations of n with exactly k cycles
[[nodiscard]] inline u64 stirling1_unsigned(int n, int k) noexcept {
    if (k == n) return 1;
    if (k == 0) return n == 0 ? 1 : 0;
    std::vector<std::vector<u64>> s(n+1, std::vector<u64>(n+1, 0));
    s[0][0] = 1;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= i; ++j)
            s[i][j] = u64(i-1)*s[i-1][j] + s[i-1][j-1];
    return s[n][k];
}

// ─── Integer partitions ───────────────────────────────────────────────────
// Count partitions of n (partition function p(n))
[[nodiscard]] inline u64 partition_count(int n) noexcept {
    std::vector<u64> p(n+1, 0);
    p[0] = 1;
    for (int k = 1; k <= n; ++k)
        for (int i = k; i <= n; ++i)
            p[i] += p[i-k];
    return p[n];
}

// Enumerate all partitions of n (as sorted vectors, descending)
[[nodiscard]] inline std::vector<std::vector<int>> enumerate_partitions(int n) noexcept {
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    std::function<void(int, int)> gen = [&](int remaining, int max_part) {
        if (remaining == 0) { result.push_back(current); return; }
        for (int p = min(remaining, max_part); p >= 1; --p) {
            current.push_back(p);
            gen(remaining - p, p);
            current.pop_back();
        }
    };
    gen(n, n);
    return result;
}

// ─── Permutation operations ───────────────────────────────────────────────
// Enumerate all permutations of [0..n-1] via Heap's algorithm
inline void heap_permutations(int n, std::vector<std::vector<int>>& out) noexcept {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<int> c(n, 0);
    out.push_back(perm);
    for (int i = 0; i < n; ) {
        if (c[i] < i) {
            if (i%2==0) std::swap(perm[0], perm[i]);
            else        std::swap(perm[c[i]], perm[i]);
            out.push_back(perm);
            ++c[i]; i = 0;
        } else { c[i++] = 0; }
    }
}

// Permutation rank (Lehmer code -> index) and unrank
[[nodiscard]] inline u64 permutation_rank(const std::vector<int>& perm) noexcept {
    int n = int(perm.size());
    u64 rank = 0;
    std::vector<bool> used(n, false);
    for (int i = 0; i < n; ++i) {
        int cnt = 0;
        for (int j = 0; j < perm[i]; ++j) if (!used[j]) ++cnt;
        rank += u64(cnt) * factorial(n-1-i);
        used[perm[i]] = true;
    }
    return rank;
}

[[nodiscard]] inline std::vector<int> permutation_unrank(int n, u64 rank) noexcept {
    std::vector<int> elems(n), perm(n);
    std::iota(elems.begin(), elems.end(), 0);
    for (int i = 0; i < n; ++i) {
        u64 f = factorial(n-1-i);
        int idx = int(rank / f);
        rank %= f;
        perm[i] = elems[idx];
        elems.erase(elems.begin()+idx);
    }
    return perm;
}

// Permutation sign (+1 or -1) via cycle decomposition
[[nodiscard]] inline int permutation_sign(std::vector<int> p) noexcept {
    int n = int(p.size()), sign = 1;
    std::vector<bool> visited(n, false);
    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        int len = 0;
        for (int j = i; !visited[j]; j = p[j]) { visited[j] = true; ++len; }
        if (len % 2 == 0) sign = -sign;
    }
    return sign;
}

// ─── Combinations ─────────────────────────────────────────────────────────
// Enumerate all k-combinations of {0,..,n-1}
[[nodiscard]] inline std::vector<std::vector<int>> combinations(int n, int k) noexcept {
    std::vector<std::vector<int>> result;
    std::vector<int> idx(k);
    std::iota(idx.begin(), idx.end(), 0);
    while (idx[0] <= n-k) {
        result.push_back(idx);
        int i = k-1;
        while (i > 0 && idx[i] == n-k+i) --i;
        ++idx[i];
        for (int j = i+1; j < k; ++j) idx[j] = idx[j-1]+1;
    }
    return result;
}

// ─── Derangements D_n = n! * sum_{k=0}^{n} (-1)^k/k! ───────────────────
[[nodiscard]] constexpr u64 derangements(int n) noexcept {
    if (n == 0) return 1;
    if (n == 1) return 0;
    u64 a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        u64 c = u64(i-1)*(a+b);
        a = b; b = c;
    }
    return b;
}

// ─── Burnside's lemma (cycle index, group action) ────────────────────────
// Count distinct colorings of n positions under rotation group of order g
// fixed[i] = number of colorings fixed by rotation i (given as input)
[[nodiscard]] inline u64 burnside_count(const std::vector<u64>& fixed_by_rotation) noexcept {
    u64 sum = 0;
    for (auto f : fixed_by_rotation) sum += f;
    return sum / u64(fixed_by_rotation.size());
}

// Rotational symmetry of n-bead necklace with k colors: fixed by rotation r is k^gcd(n,r)
[[nodiscard]] inline u64 necklace_count(int n, int k) noexcept {
    u64 sum = 0;
    for (int r = 0; r < n; ++r) {
        int g = std::gcd(n, r==0?n:r);
        u64 pow = 1;
        for (int i = 0; i < g; ++i) pow *= u64(k);
        sum += pow;
    }
    return sum / u64(n);
}

// ─── Generating functions (polynomial multiplication) ─────────────────────
using Poly = std::vector<i64>;

[[nodiscard]] inline Poly poly_multiply(const Poly& a, const Poly& b) noexcept {
    if (a.empty() || b.empty()) return {};
    Poly c(a.size()+b.size()-1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            c[i+j] += a[i]*b[j];
    return c;
}

// Product of (1 + x + x^2 + ... + x^{k_i}) for i=1..n  (generating fn for subset sums)
[[nodiscard]] inline Poly subset_sum_gf(const std::vector<int>& max_counts) noexcept {
    Poly result = {1};
    for (int m : max_counts) {
        Poly factor(m+1, 1);
        result = poly_multiply(result, factor);
    }
    return result;
}

} // namespace qs::math::combinatorics
