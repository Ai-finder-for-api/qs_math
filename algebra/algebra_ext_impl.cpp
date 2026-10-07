
// qs_math/algebra/algebra_ext_impl.cpp (C++20)
// Extended algebra: lattice theory, abstract algebra, category theory helpers,
// p-adic numbers, modular arithmetic, combinatorics.
#include "include/qs_math_algebra.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace qs::math::algebra {

// ─── Modular arithmetic ───────────────────────────────────────────────────────
[[nodiscard]] i64 mod_inverse(i64 a, i64 m) noexcept {
    // Extended Euclidean algorithm
    i64 g=m,x=0,y=1;
    if(a==0)return 0;
    i64 a0=a,g0=g;(void)g0;
    while(a!=0){const i64 q=g/a;g-=q*a;std::swap(g,a);x-=q*y;std::swap(x,y);}
    if(g!=1)return 0; // no inverse
    return (x%m+m)%m;
}

[[nodiscard]] i64 mod_add(i64 a, i64 b, i64 m) noexcept { return (a+b)%m; }
[[nodiscard]] i64 mod_sub(i64 a, i64 b, i64 m) noexcept { return ((a-b)%m+m)%m; }
[[nodiscard]] i64 mod_mul(i64 a, i64 b, i64 m) noexcept { return (__int128)a*b%m; }

// Discrete logarithm (baby-step giant-step)
[[nodiscard]] i64 discrete_log(i64 g, i64 h, i64 p) noexcept {
    const i64 n = (i64)std::ceil(std::sqrt((f64)p));
    // Baby steps: store g^j mod p
    std::vector<std::pair<i64,i64>> table;
    i64 gj = 1;
    for (i64 j=0;j<n;++j){ table.push_back({gj,j}); gj=mod_mul(gj,g,p); }
    std::sort(table.begin(),table.end());
    // Giant steps: g^(-n) mod p
    const i64 gn_inv = [&](){
        // g^(p-1-n) mod p via fast exponentiation (Fermat's little theorem)
        i64 base=g,exp=(p-1-n%((p-1))),result=1;
        while(exp>0){if(exp&1)result=mod_mul(result,base,p);base=mod_mul(base,base,p);exp>>=1;}
        return result;
    }();
    i64 giant = h;
    for (i64 i=0;i<n;++i){
        auto it = std::lower_bound(table.begin(),table.end(),std::make_pair(giant,(i64)0));
        if(it!=table.end()&&it->first==giant)return i*n+it->second;
        giant=mod_mul(giant,gn_inv,p);
    }
    return -1; // not found
}

// ─── Combinatorics ────────────────────────────────────────────────────────────
// Pascal's triangle (binomial coefficients) modulo prime p
struct PascalMod {
    i64 p;
    std::vector<i64> fact, inv_fact;
    PascalMod(i32 n, i64 prime) : p(prime), fact(n+1), inv_fact(n+1) {
        fact[0]=1;
        for(i32 i=1;i<=n;++i)fact[i]=mod_mul(fact[i-1],i,p);
        // Fermat's little theorem for inverse
        auto mpow=[&](i64 base,i64 exp)->i64{
            i64 r=1;while(exp>0){if(exp&1)r=mod_mul(r,base,p);base=mod_mul(base,base,p);exp>>=1;}return r;};
        inv_fact[n]=mpow(fact[n],p-2);
        for(i32 i=n-1;i>=0;--i)inv_fact[i]=mod_mul(inv_fact[i+1],i+1,p);
    }
    [[nodiscard]] i64 C(i32 n, i32 k) const noexcept {
        if(k<0||k>n)return 0;
        return mod_mul(mod_mul(fact[n],inv_fact[k],p),inv_fact[n-k],p);
    }
    [[nodiscard]] i64 P(i32 n, i32 k) const noexcept {
        if(k<0||k>n)return 0;
        return mod_mul(fact[n],inv_fact[n-k],p);
    }
};

// Generating function: multiply two polynomial arrays mod prime
std::vector<i64> poly_mul_mod(const std::vector<i64>& a,
    const std::vector<i64>& b, i64 mod) noexcept
{
    i32 na=(i32)a.size(),nb=(i32)b.size();
    std::vector<i64> c(na+nb-1,0);
    for(i32 i=0;i<na;++i)for(i32 j=0;j<nb;++j)
        c[i+j]=(c[i+j]+mod_mul(a[i],b[j],mod))%mod;
    return c;
}

// Stirling numbers of the second kind S(n,k): partitions of [n] into k blocks
std::vector<std::vector<i64>> stirling2_table(i32 max_n) noexcept {
    std::vector<std::vector<i64>> S(max_n+1, std::vector<i64>(max_n+1,0));
    S[0][0]=1;
    for(i32 n=1;n<=max_n;++n)
        for(i32 k=1;k<=n;++k)
            S[n][k]=k*S[n-1][k]+S[n-1][k-1];
    return S;
}

// Bell numbers (total number of set partitions)
std::vector<i64> bell_numbers(i32 n_max) noexcept {
    std::vector<i64> B(n_max+1,0); B[0]=1;
    for(i32 n=1;n<=n_max;++n){
        std::vector<i64> row(n+1,0); row[0]=B[n-1];
        for(i32 k=1;k<=n;++k)row[k]=row[k-1]+B[k-1];
        B[n]=row[n];
    }
    return B;
}

// Inclusion-exclusion: count surjections from [n] to [k]
[[nodiscard]] i64 surjections(i32 n, i32 k) noexcept {
    i64 result=0;
    for(i32 j=0;j<=k;++j){
        const i64 sign=(j%2==0)?1:-1;
        i64 binom=1;for(i32 m=0;m<j;++m)binom=binom*(k-m)/(m+1);
        result+=sign*binom*(i64)std::round(std::pow(k-j,n));
    }
    return result;
}

// ─── Group theory ─────────────────────────────────────────────────────────────
// Check if permutation is even (sign = +1 → even, -1 → odd)
[[nodiscard]] i32 permutation_sign(const std::vector<i32>& perm) noexcept {
    i32 n=(i32)perm.size();
    std::vector<bool> visited(n,false);
    i32 sign=1;
    for(i32 i=0;i<n;++i){
        if(visited[i])continue;
        i32 cycle_len=0, j=i;
        while(!visited[j]){visited[j]=true;j=perm[j];++cycle_len;}
        if(cycle_len%2==0)sign=-sign;
    }
    return sign;
}

// Compose two permutations: result[i] = p2[p1[i]]
[[nodiscard]] std::vector<i32> compose_perms(const std::vector<i32>& p1,
    const std::vector<i32>& p2) noexcept
{
    assert(p1.size()==p2.size());
    std::vector<i32> r(p1.size());
    for(i32 i=0;i<(i32)p1.size();++i)r[i]=p2[p1[i]];
    return r;
}

// Generate all permutations of [0..n-1] via Heap's algorithm
std::vector<std::vector<i32>> all_permutations(i32 n) noexcept {
    std::vector<i32> perm(n); std::iota(perm.begin(),perm.end(),0);
    std::vector<std::vector<i32>> result;
    std::vector<i32> c(n,0);
    result.push_back(perm);
    for(i32 i=0;i<n;){
        if(c[i]<i){
            if(i%2==0)std::swap(perm[0],perm[i]);
            else std::swap(perm[c[i]],perm[i]);
            result.push_back(perm);
            ++c[i];i=0;
        }else{c[i]=0;++i;}
    }
    return result;
}

// ─── Tensor operations ────────────────────────────────────────────────────────
// Kronecker product A⊗B
std::vector<f64> kronecker_product(const f64* A,i32 ma,i32 na,
    const f64* B,i32 mb,i32 nb) noexcept
{
    const i32 mc=ma*mb,nc=na*nb;
    std::vector<f64> C(mc*nc);
    for(i32 ia=0;ia<ma;++ia)for(i32 ja=0;ja<na;++ja)
        for(i32 ib=0;ib<mb;++ib)for(i32 jb=0;jb<nb;++jb)
            C[(ia*mb+ib)*nc+(ja*nb+jb)]=A[ia*na+ja]*B[ib*nb+jb];
    return C;
}

// Tensor contraction: contract index i of A with index j of B
std::vector<f64> tensor_contract(const f64* A,const std::vector<i32>& dims_A,i32 idx_A,
    const f64* B,const std::vector<i32>& dims_B,i32 idx_B) noexcept
{
    // Simplified: contract last index of A with first index of B (matrix multiply)
    i32 k=dims_A.back(),m=1,n=1;
    for(i32 i=0;i<(i32)dims_A.size()-1;++i)m*=dims_A[i];
    for(i32 i=1;i<(i32)dims_B.size();++i)n*=dims_B[i];
    std::vector<f64> C(m*n,0);
    for(i32 i=0;i<m;++i)for(i32 j=0;j<n;++j)
        for(i32 kk=0;kk<k;++kk)C[i*n+j]+=A[i*k+kk]*B[kk*n+j];
    (void)idx_A;(void)idx_B;
    return C;
}

} // namespace qs::math::algebra
