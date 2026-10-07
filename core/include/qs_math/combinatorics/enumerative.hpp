#pragma once
// qs_math/combinatorics/enumerative.hpp — Permutations, combinations, Catalan, Stirling, Young
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Factorials and binomials ─────────────────────────────────────────────────
[[nodiscard]] constexpr i64 factorial(int n) noexcept {
    i64 r=1; for(int i=2;i<=n;++i) r*=i; return r;
}
[[nodiscard]] constexpr i64 binomial(int n, int k) noexcept {
    if(k<0||k>n) return 0;
    if(k==0||k==n) return 1;
    k=k<n-k?k:n-k;
    i64 r=1;
    for(int i=0;i<k;++i){r*=(n-i);r/=(i+1);}
    return r;
}
[[nodiscard]] constexpr i64 multinomial(int n, const int* ks, int m) noexcept {
    i64 r=factorial(n);
    for(int i=0;i<m;++i) r/=factorial(ks[i]);
    return r;
}
// Falling factorial: (n)_k = n*(n-1)*...*(n-k+1)
[[nodiscard]] constexpr i64 falling_factorial(int n, int k) noexcept {
    i64 r=1; for(int i=0;i<k;++i) r*=(n-i); return r;
}
// Rising factorial (Pochhammer): (n)^k = n*(n+1)*...*(n+k-1)
[[nodiscard]] constexpr i64 rising_factorial(int n, int k) noexcept {
    i64 r=1; for(int i=0;i<k;++i) r*=(n+i); return r;
}

// ─── Pascal's triangle ───────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr std::array<std::array<i64,N>,N> pascal_triangle() noexcept {
    std::array<std::array<i64,N>,N> T{};
    for(int i=0;i<N;++i){T[i][0]=1;for(int j=1;j<=i;++j)T[i][j]=T[i-1][j-1]+T[i-1][j];}
    return T;
}

// ─── Catalan numbers ─────────────────────────────────────────────────────────
[[nodiscard]] constexpr i64 catalan(int n) noexcept {
    return binomial(2*n,n)/(n+1);
}
// Generalized Catalan: ballot numbers, Motzkin, etc.
[[nodiscard]] constexpr i64 motzkin(int n) noexcept {
    if(n<=1) return 1;
    // M(n) = M(n-1) + sum_{k=0}^{n-2} M(k)*M(n-2-k)
    std::array<i64,32> M{};
    M[0]=1;M[1]=1;
    for(int i=2;i<=n&&i<32;++i){
        M[i]=M[i-1];
        for(int k=0;k<=i-2;++k) M[i]+=M[k]*M[i-2-k];
    }
    return n<32?M[n]:0;
}
// Narayana number N(n,k) = (1/n)*C(n,k)*C(n,k-1)
[[nodiscard]] constexpr i64 narayana(int n, int k) noexcept {
    if(n<=0||k<=0||k>n) return 0;
    return binomial(n,k)*binomial(n,k-1)/n;
}

// ─── Stirling numbers ─────────────────────────────────────────────────────────
// Stirling 1st kind: s(n,k) = unsigned, S(n,k) signed
[[nodiscard]] constexpr i64 stirling1(int n, int k) noexcept {
    // |s(n,k)| = |s(n-1,k-1)| + (n-1)*|s(n-1,k)|
    if(k==n) return 1; if(k==0||n==0) return 0;
    std::array<std::array<i64,16>,16> s{};
    s[0][0]=1;
    for(int i=1;i<=n&&i<16;++i)
        for(int j=1;j<=i&&j<16;++j)
            s[i][j]=s[i-1][j-1]+(i-1)*s[i-1][j];
    return (n<16&&k<16)?s[n][k]:0;
}
// Stirling 2nd kind: S(n,k) = S(n-1,k-1) + k*S(n-1,k)
[[nodiscard]] constexpr i64 stirling2(int n, int k) noexcept {
    if(k==n) return 1; if(k==0||n==0) return 0;
    std::array<std::array<i64,16>,16> S{};
    S[0][0]=1;
    for(int i=1;i<=n&&i<16;++i)
        for(int j=1;j<=i&&j<16;++j)
            S[i][j]=S[i-1][j-1]+j*S[i-1][j];
    return (n<16&&k<16)?S[n][k]:0;
}
// Bell numbers: B(n) = sum_{k=0}^{n} S(n,k)
[[nodiscard]] constexpr i64 bell(int n) noexcept {
    i64 s=0; for(int k=0;k<=n;++k) s+=stirling2(n,k); return s;
}

// ─── Partition numbers ────────────────────────────────────────────────────────
[[nodiscard]] constexpr i64 partition_count(int n) noexcept {
    std::array<i64,64> p{};p[0]=1;
    for(int k=1;k<=n&&k<64;++k)
        for(int i=k;i<=n&&i<64;++i)
            p[i]+=p[i-k];
    return n<64?p[n]:0;
}

// ─── Young tableaux (hook length formula) ────────────────────────────────────
// Number of standard Young tableaux of shape λ: n! / product of hook lengths
template<int K>
[[nodiscard]] constexpr i64 young_tableaux_count(const std::array<int,K>& lambda) noexcept {
    int n=0; for(int l:lambda) n+=l;
    i64 num=factorial(n), den=1;
    for(int i=0;i<K;++i) {
        for(int j=0;j<lambda[i];++j) {
            // Hook length: (lambda[i]-j) + (number of rows > i with lambda[r]>j) - 1 + 1
            int hook=lambda[i]-j;
            for(int r=i+1;r<K;++r) if(lambda[r]>j) ++hook;
            den*=hook;
        }
    }
    return num/den;
}

// ─── Permutation rank/unrank ───────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr i64 permutation_rank(const std::array<int,N>& perm) noexcept {
    std::array<bool,N> used{};
    i64 rank=0;
    for(int i=0;i<N;++i) {
        int cnt=0;
        for(int j=0;j<perm[i];++j) if(!used[j]) ++cnt;
        rank+=cnt*factorial(N-1-i);
        used[perm[i]]=true;
    }
    return rank;
}

template<int N>
[[nodiscard]] constexpr std::array<int,N> permutation_unrank(i64 rank) noexcept {
    std::array<int,N> perm{};
    std::array<bool,N> used{};
    for(int i=0;i<N;++i) {
        i64 f=factorial(N-1-i);
        int cnt=int(rank/f); rank%=f;
        for(int j=0,found=0;;++j) if(!used[j]&&found++==cnt){perm[i]=j;used[j]=true;break;}
    }
    return perm;
}

// ─── Inclusion-exclusion ──────────────────────────────────────────────────────
// |A1 ∪ ... ∪ An| = Σ|Ai| - Σ|Ai∩Aj| + ...
template<int N>
[[nodiscard]] constexpr i64 inclusion_exclusion(
    const std::array<i64,(1<<N)>& set_sizes) noexcept
{
    i64 total=0;
    for(int mask=1;mask<(1<<N);++mask) {
        int bits=0; for(int b=mask;b;b&=b-1)++bits;
        total+=(bits%2==1?1:-1)*set_sizes[mask];
    }
    return total;
}

// ─── Eulerian numbers ─────────────────────────────────────────────────────────
// A(n,k) = number of permutations of [n] with exactly k ascents
[[nodiscard]] constexpr i64 eulerian(int n, int k) noexcept {
    if(k<0||k>=n) return k==0&&n==0?1:0;
    std::array<std::array<i64,16>,16> A{};
    A[0][0]=1;
    for(int i=1;i<=n&&i<16;++i)
        for(int j=0;j<i&&j<16;++j)
            A[i][j]=(j+1)*A[i-1][j]+(i-j)*A[i-1][j-1<0?0:j-1];
    return n<16&&k<16?A[n][k]:0;
}
} // namespace qs::math
