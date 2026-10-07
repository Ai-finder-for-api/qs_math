#pragma once
// qs_math/discrete/recurrence.hpp — Linear recurrences, Berlekamp-Massey, Kitamasa
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <vector>
#include <optional>

namespace qs::math {

// ─── Matrix over Z (for recurrence via matrix exponentiation) ─────────────────
template<int N>
struct MatNx [[nodiscard]] {
    std::array<std::array<i64,N>,N> m{};
    [[nodiscard]] constexpr i64  at(int r,int c) const noexcept { return m[r][c]; }
    [[nodiscard]] constexpr i64& at(int r,int c)       noexcept { return m[r][c]; }
};

template<int N>
[[nodiscard]] constexpr MatNx<N> mat_mul(const MatNx<N>& A, const MatNx<N>& B) noexcept {
    MatNx<N> C{};
    for (int i=0;i<N;++i)
        for (int k=0;k<N;++k) if (A.at(i,k))
            for (int j=0;j<N;++j)
                C.at(i,j) += A.at(i,k)*B.at(k,j);
    return C;
}

template<int N>
[[nodiscard]] constexpr MatNx<N> mat_mul_mod(const MatNx<N>& A, const MatNx<N>& B,
                                               i64 mod) noexcept {
    MatNx<N> C{};
    for (int i=0;i<N;++i)
        for (int k=0;k<N;++k) if (A.at(i,k))
            for (int j=0;j<N;++j)
                C.at(i,j) = (C.at(i,j) + A.at(i,k)%mod * (B.at(k,j)%mod)) % mod;
    return C;
}

template<int N>
[[nodiscard]] constexpr MatNx<N> mat_identity() noexcept {
    MatNx<N> I{};
    for (int i=0;i<N;++i) I.at(i,i)=1;
    return I;
}

template<int N>
[[nodiscard]] constexpr MatNx<N> mat_pow_mod(MatNx<N> A, i64 exp, i64 mod) noexcept {
    auto R = mat_identity<N>();
    while (exp > 0) {
        if (exp & 1) R = mat_mul_mod(R,A,mod);
        A = mat_mul_mod(A,A,mod);
        exp >>= 1;
    }
    return R;
}

// ─── Linear recurrence: a[n] = c[0]*a[n-1] + c[1]*a[n-2] + ... ──────────────
template<int K>
struct LinearRecurrence {
    std::array<i64,K> coeffs{};   // c[0..K-1]
    std::array<i64,K> init{};     // a[0..K-1]

    // Evaluate a[n] mod m via matrix exponentiation, O(K^3 log n)
    [[nodiscard]] constexpr i64 eval_mod(i64 n, i64 m) const noexcept {
        if (n < K) return ((init[n] % m) + m) % m;
        // Companion matrix
        MatNx<K> C{};
        for (int j=0;j<K;++j) C.at(0,j) = ((coeffs[j]%m)+m)%m;
        for (int i=1;i<K;++i) C.at(i,i-1) = 1;
        auto Cn = mat_pow_mod<K>(C, n-K+1, m);
        i64 result = 0;
        for (int j=0;j<K;++j)
            result = (result + Cn.at(0,K-1-j) % m * (((init[K-1-j]%m)+m)%m)) % m;
        return result;
    }
};

// Fibonacci as a LinearRecurrence<2>
[[nodiscard]] constexpr LinearRecurrence<2> fibonacci_recurrence() noexcept {
    return { {1,1}, {0,1} };
}

// Tribonacci
[[nodiscard]] constexpr LinearRecurrence<3> tribonacci_recurrence() noexcept {
    return { {1,1,1}, {0,0,1} };
}

// ─── Berlekamp-Massey: find shortest LFSR for sequence ───────────────────────
// Returns minimal polynomial coefficients over integers (mod p)
[[nodiscard]] inline std::vector<i64> berlekamp_massey(
    const std::vector<i64>& s, i64 p) noexcept
{
    int n = static_cast<int>(s.size());
    std::vector<i64> C(1,1), B(1,1), T;
    int L=0, x=1;
    i64 b=1;
    auto mod_p=[&](i64 v){ return ((v%p)+p)%p; };
    auto inv_p=[&](i64 a) -> i64 {
        // Fermat inverse mod prime p
        i64 r=1,e=p-2; a=mod_p(a);
        while(e>0){if(e&1)r=r*a%p;a=a*a%p;e>>=1;}
        return r;
    };
    for (int i=0;i<n;++i) {
        i64 d=mod_p(s[i]);
        for (int j=1;j<(int)C.size();++j) d=mod_p(d+C[j]*s[i-j]);
        if (d==0){++x;continue;}
        T=C;
        i64 coef=mod_p(d*inv_p(b));
        if ((int)C.size()<(int)B.size()+x) C.resize(B.size()+x,0);
        for (int j=x;j<(int)B.size()+x;++j) C[j]=mod_p(C[j]-coef*B[j-x]);
        if (2*L<=i){L=i+1-L;B=T;b=d;x=1;}else ++x;
    }
    return C; // C[0]=1, LFSR polynomial
}

// ─── Kitamasa's method: direct nth term of k-order linear recurrence ──────────
// Reduces x^n mod minimal polynomial, then evaluates using initial values
[[nodiscard]] inline i64 kitamasa(const std::vector<i64>& c,   // recurrence coeffs
                                   const std::vector<i64>& init, // initial values
                                   i64 n, i64 mod) noexcept
{
    int k = static_cast<int>(c.size());
    if (n < k) return ((init[n]%mod)+mod)%mod;
    auto poly_mod=[&](std::vector<i64> a) -> std::vector<i64> {
        while ((int)a.size() > k) {
            i64 lead = ((a.back()%mod)+mod)%mod;
            int deg   = (int)a.size()-1;
            a.pop_back();
            if (!lead) continue;
            for (int i=0;i<k;++i)
                a[deg-k+i] = ((a[deg-k+i]+lead*c[i])%mod+mod)%mod;
        }
        return a;
    };
    auto poly_mul=[&](const std::vector<i64>& a, const std::vector<i64>& b)
        -> std::vector<i64>
    {
        std::vector<i64> res(a.size()+b.size()-1,0);
        for (int i=0;i<(int)a.size();++i)
            for (int j=0;j<(int)b.size();++j)
                res[i+j]=(res[i+j]+a[i]*b[j])%mod;
        return poly_mod(res);
    };
    // x^n mod minimal poly via repeated squaring
    std::vector<i64> result(k,0), base(std::min(2,k),0);
    result[0]=1; if(k>1)base[1]=1; else base[0]=1; // base = x
    i64 exp=n;
    while (exp>0){
        if(exp&1) result=poly_mul(result,base);
        base=poly_mul(base,base);
        exp>>=1;
    }
    i64 ans=0;
    for (int i=0;i<k;++i) ans=(ans+result[i]%mod*init[i])%mod;
    return (ans+mod)%mod;
}

// ─── Generating function evaluation ──────────────────────────────────────────
// Evaluate sum_{k=0}^{N-1} a[k]*x^k at point x=val (Horner)
[[nodiscard]] constexpr f64 eval_generating_function(
    const f64* coeffs, int N, f64 val) noexcept
{
    f64 r=0;
    for (int i=N-1;i>=0;--i) r=r*val+coeffs[i];
    return r;
}

// ─── Closed forms for common recurrences ─────────────────────────────────────
// Fibonacci: F(n) = round(phi^n / sqrt(5)), exact for moderate n
[[nodiscard]] constexpr i64 fibonacci(i64 n) noexcept {
    if (n<=0) return 0; if (n==1) return 1;
    // Matrix method for exact i64 result
    i64 a=0,b=1;
    for (i64 i=2;i<=n;++i){i64 t=a+b;a=b;b=t;}
    return b;
}

[[nodiscard]] constexpr i64 fibonacci_mod(i64 n, i64 m) noexcept {
    return fibonacci_recurrence().eval_mod(n,m);
}

// Lucas numbers: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2)
[[nodiscard]] constexpr i64 lucas(i64 n) noexcept {
    if (n==0) return 2; if (n==1) return 1;
    i64 a=2,b=1;
    for (i64 i=2;i<=n;++i){i64 t=a+b;a=b;b=t;}
    return b;
}

// Padovan: P(n)=P(n-2)+P(n-3), P(0)=P(1)=P(2)=1
[[nodiscard]] constexpr i64 padovan(i64 n) noexcept {
    if(n<3) return 1;
    i64 a=1,b=1,c=1;
    for(i64 i=3;i<=n;++i){i64 t=a+b;a=b;b=c;c=t;}
    return c;
}

} // namespace qs::math
