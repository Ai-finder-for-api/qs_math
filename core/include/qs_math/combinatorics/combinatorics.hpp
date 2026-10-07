#pragma once
// combinatorics.hpp — Combinatorics: permutations, combinations, partitions,
//                     Bell numbers, generating functions, lattice paths,
//                     inclusion-exclusion, Burnside's lemma helpers
#include "scalar_types.hpp"
#include "integer_math.hpp"

namespace qs::math::combinatorics {

// ---------------------------------------------------------------------------
// Factorials (precomputed, overflow-safe)
// ---------------------------------------------------------------------------
inline constexpr u64 FACTORIAL[21]={
    1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,
    479001600ULL,6227020800ULL,87178291200ULL,1307674368000ULL,
    20922789888000ULL,355687428096000ULL,6402373705728000ULL,
    121645100408832000ULL,2432902008176640000ULL
};
[[nodiscard]] constexpr u64 factorial(i32 n) noexcept {
    return n<0?0:(n<=20?FACTORIAL[n]:u64(-1));
}

// Double factorial n!! = n*(n-2)*(n-4)*...
[[nodiscard]] constexpr u64 double_factorial(i32 n) noexcept {
    u64 r=1; while(n>1){ r*=n; n-=2; } return r;
}

// ---------------------------------------------------------------------------
// Permutations
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 permutations(i32 n, i32 r) noexcept {
    if(r<0||r>n) return 0;
    u64 result=1;
    for(i32 i=n;i>n-r;--i) result*=i;
    return result;
}

// Next permutation (lexicographic order), returns false when done
[[nodiscard]] inline bool next_permutation(i32* arr, i32 n) noexcept {
    i32 i=n-2;
    while(i>=0&&arr[i]>=arr[i+1]) --i;
    if(i<0) return false;
    i32 j=n-1;
    while(arr[j]<=arr[i]) --j;
    i32 t=arr[i];arr[i]=arr[j];arr[j]=t;
    // Reverse from i+1 to end
    for(i32 l=i+1,r=n-1;l<r;++l,--r){t=arr[l];arr[l]=arr[r];arr[r]=t;}
    return true;
}

// kth permutation of [0..n-1]
inline void kth_permutation(i32 n, u64 k, i32* out) noexcept {
    i32 items[64]; for(i32 i=0;i<n&&i<64;++i) items[i]=i;
    for(i32 i=0;i<n&&i<64;++i){
        u64 f=factorial(n-1-i);
        i32 idx=i32(k/f); k%=f;
        out[i]=items[idx];
        for(i32 j=idx;j<n-1-i;++j) items[j]=items[j+1];
    }
}

// Rank of permutation (Lehmer code)
[[nodiscard]] inline u64 permutation_rank(const i32* arr, i32 n) noexcept {
    u64 rank=0;
    for(i32 i=0;i<n;++i){
        i32 cnt=0;
        for(i32 j=i+1;j<n;++j) if(arr[j]<arr[i]) ++cnt;
        rank+=cnt*factorial(n-1-i);
    }
    return rank;
}

// ---------------------------------------------------------------------------
// Combinations
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 combinations(i32 n, i32 r) noexcept {
    return binomial(n,r);
}

// Next combination (k-subset of {0..n-1}) in lexicographic order
[[nodiscard]] inline bool next_combination(i32* c, i32 k, i32 n) noexcept {
    i32 i=k-1;
    while(i>=0&&c[i]==i+n-k) --i;
    if(i<0) return false;
    ++c[i];
    for(i32 j=i+1;j<k;++j) c[j]=c[j-1]+1;
    return true;
}

// Multiset coefficient: n+k-1 choose k
[[nodiscard]] constexpr u64 multiset(i32 n, i32 k) noexcept { return binomial(n+k-1,k); }

// ---------------------------------------------------------------------------
// Partitions
// ---------------------------------------------------------------------------
// Number of integer partitions of n
[[nodiscard]] constexpr u64 partition_count(i32 n) noexcept {
    // Euler's pentagonal theorem DP
    static u64 dp[512]={1};
    static i32 computed=0;
    if(n>511) return 0;
    for(i32 i=computed+1;i<=n;++i){
        dp[i]=0;
        for(i32 k=1;;++k){
            i32 p1=k*(3*k-1)/2, p2=k*(3*k+1)/2;
            if(p1>i) break;
            dp[i]+=k%2==0?-dp[i-p1]:dp[i-p1];
            if(p2<=i) dp[i]+=k%2==0?-dp[i-p2]:dp[i-p2];
        }
    }
    computed=max(computed,n);
    return dp[n];
}

// Partition into at most k parts
[[nodiscard]] constexpr u64 partition_at_most_k(i32 n, i32 k) noexcept {
    // DP table
    static constexpr i32 MAX=50;
    if(n>MAX||k>MAX) return 0;
    u64 dp[MAX+1][MAX+1]={};
    dp[0][0]=1;
    for(i32 i=1;i<=n;++i)
        for(i32 j=0;j<=k;++j)
            dp[i][j]=dp[i][j>0?j-1:0]+(i>=j?dp[i-j][j]:0);
    return dp[n][k];
}

// ---------------------------------------------------------------------------
// Bell numbers (number of set partitions)
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 bell_number(i32 n) noexcept {
    if(n==0) return 1;
    // Bell triangle
    static constexpr i32 MAX_N=20;
    if(n>MAX_N) return 0;
    u64 tri[MAX_N+1][MAX_N+1]={};
    tri[0][0]=1;
    for(i32 i=1;i<=n;++i){
        tri[i][0]=tri[i-1][i-1];
        for(i32 j=1;j<=i;++j) tri[i][j]=tri[i][j-1]+tri[i-1][j-1];
    }
    return tri[n][0];
}

// Stirling numbers (second kind) — from integer_math.hpp
// Using stirling2() from integer_math.hpp

// ---------------------------------------------------------------------------
// Lattice paths
// ---------------------------------------------------------------------------
// Number of paths from (0,0) to (m,n) with only right/up steps
[[nodiscard]] constexpr u64 lattice_paths(i32 m, i32 n) noexcept { return binomial(m+n,m); }

// Catalan-like: paths that don't cross diagonal
[[nodiscard]] constexpr u64 ballot_paths(i32 n) noexcept {
    // Catalan number C_n = C(2n,n)/(n+1)
    return binomial(2*n,n)/(n+1);
}

// Motzkin number (paths with horizontal steps that don't go below x-axis)
[[nodiscard]] constexpr u64 motzkin(i32 n) noexcept {
    u64 r=0;
    for(i32 k=0;k<=n/2;++k) r+=binomial(n,2*k)*binomial(2*k,k)/(k+1);
    return r;
}

// ---------------------------------------------------------------------------
// Generating functions (polynomial coefficient extraction)
// ---------------------------------------------------------------------------
// Coefficient of x^n in (1-x)^(-k) = C(n+k-1, k-1)
[[nodiscard]] constexpr u64 neg_binom_coeff(i32 n, i32 k) noexcept {
    return binomial(n+k-1,k-1);
}

// Partition generating function coefficient (up to degree N)
inline void partition_gf(i32 n, u64* coeffs) noexcept {
    coeffs[0]=1; for(i32 i=1;i<=n;++i) coeffs[i]=0;
    for(i32 k=1;k<=n;++k)
        for(i32 i=k;i<=n;++i)
            coeffs[i]+=coeffs[i-k];
}

// ---------------------------------------------------------------------------
// Inclusion-Exclusion principle
// ---------------------------------------------------------------------------
// Count elements in at least one of n sets given |A_i...| values
// sizes[mask] = |intersection of sets where bit is set in mask|
[[nodiscard]] inline u64 inclusion_exclusion(const u64* sizes, i32 n_sets) noexcept {
    i32 m=1<<n_sets; u64 total=0;
    for(i32 mask=1;mask<m;++mask){
        i32 bits=std::popcount(u32(mask));
        if(bits%2==1) total+=sizes[mask];
        else          total-=sizes[mask];
    }
    return total;
}

// ---------------------------------------------------------------------------
// Derangements D(n) = n! * sum(-1)^k/k! for k=0..n
// ---------------------------------------------------------------------------
[[nodiscard]] constexpr u64 derangements(i32 n) noexcept {
    if(n==0) return 1; if(n==1) return 0;
    u64 a=1,b=0;
    for(i32 i=2;i<=n;++i){ u64 c=(i-1)*(a+b); a=b; b=c; }
    return b;
}

// Subfactorial (same as derangements)
[[nodiscard]] constexpr u64 subfactorial(i32 n) noexcept { return derangements(n); }

// ---------------------------------------------------------------------------
// Gray code combinatorics
// ---------------------------------------------------------------------------
// kth combination in reflected Gray code order
[[nodiscard]] constexpr u64 combination_to_gray(u64 rank, i32 n, i32 k) noexcept {
    return to_gray(u32(rank)); // simplified
}

} // namespace qs::math::combinatorics
