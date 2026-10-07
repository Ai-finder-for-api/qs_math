#pragma once
// qs_math/algebra/group_theory.hpp — Abstract algebra: groups, rings, fields,
//   permutation groups, Cayley tables, cosets, normal subgroups, quotient groups,
//   cyclic groups, dihedral groups, symmetric group S_n (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <unordered_map>
#include <cmath>

namespace qs::math::algebra {

// ─── Finite group (Cayley table) ──────────────────────────────────────────
struct FiniteGroup {
    int order;
    std::vector<int> table; // table[i*order+j] = i*j

    [[nodiscard]] int op(int a, int b) const noexcept { return table[a*order+b]; }
    [[nodiscard]] int identity() const noexcept {
        for(int e=0;e<order;++e){
            bool ok=true;
            for(int g=0;g<order;++g) if(op(e,g)!=g){ok=false;break;}
            if(ok) return e;
        }
        return 0;
    }
    [[nodiscard]] int inverse(int g) const noexcept {
        int e=identity();
        for(int h=0;h<order;++h) if(op(g,h)==e) return h;
        return -1;
    }
    [[nodiscard]] bool is_abelian() const noexcept {
        for(int i=0;i<order;++i) for(int j=0;j<order;++j)
            if(op(i,j)!=op(j,i)) return false;
        return true;
    }
    // Order of element g
    [[nodiscard]] int element_order(int g) const noexcept {
        int e=identity(), cur=g;
        for(int k=1;k<=order;++k){if(cur==e) return k; cur=op(cur,g);}
        return -1;
    }
    // Generate subgroup from set of generators
    [[nodiscard]] std::vector<int> generate(std::vector<int> gens) const noexcept {
        std::vector<bool> in(order,false);
        std::vector<int> subgp;
        int e=identity(); in[e]=true; subgp.push_back(e);
        for(int g:gens){in[g]=true;subgp.push_back(g);}
        bool changed=true;
        while(changed){
            changed=false;
            for(int i:subgp) for(int j:subgp){
                int k=op(i,j);
                if(!in[k]){in[k]=true;subgp.push_back(k);changed=true;}
            }
        }
        std::sort(subgp.begin(),subgp.end());
        return subgp;
    }
    // Left cosets of subgroup H
    [[nodiscard]] std::vector<std::vector<int>> left_cosets(
        const std::vector<int>& H) const noexcept
    {
        std::vector<bool> used(order,false);
        std::vector<std::vector<int>> cosets;
        for(int g=0;g<order;++g){
            if(used[g]) continue;
            std::vector<int> coset;
            for(int h:H){int gh=op(g,h);if(!used[gh]){used[gh]=true;coset.push_back(gh);}}
            std::sort(coset.begin(),coset.end());
            cosets.push_back(coset);
        }
        return cosets;
    }
    // Check if H is normal subgroup
    [[nodiscard]] bool is_normal(const std::vector<int>& H) const noexcept {
        for(int g=0;g<order;++g){
            int ginv=inverse(g);
            for(int h:H){
                int conj=op(g,op(h,ginv));
                if(!std::count(H.begin(),H.end(),conj)) return false;
            }
        }
        return true;
    }
    // Center Z(G)
    [[nodiscard]] std::vector<int> center() const noexcept {
        std::vector<int> Z;
        for(int g=0;g<order;++g){
            bool comm=true;
            for(int h=0;h<order;++h) if(op(g,h)!=op(h,g)){comm=false;break;}
            if(comm) Z.push_back(g);
        }
        return Z;
    }
    // Commutator subgroup [G,G]
    [[nodiscard]] std::vector<int> commutator_subgroup() const noexcept {
        std::vector<int> gens;
        for(int a=0;a<order;++a) for(int b=0;b<order;++b){
            int com=op(op(a,b),op(inverse(a),inverse(b)));
            if(!std::count(gens.begin(),gens.end(),com)) gens.push_back(com);
        }
        return generate(gens);
    }
};

// ─── Cyclic group Z_n ─────────────────────────────────────────────────────
[[nodiscard]] inline FiniteGroup cyclic_group(int n) noexcept {
    FiniteGroup G; G.order=n; G.table.resize(n*n);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) G.table[i*n+j]=(i+j)%n;
    return G;
}

// ─── Dihedral group D_n (symmetries of regular n-gon) ────────────────────
// Elements: r^k (rotations, k=0..n-1) and s*r^k (reflections)
// r^k has index k, s*r^k has index n+k
[[nodiscard]] inline FiniteGroup dihedral_group(int n) noexcept {
    int order=2*n; FiniteGroup G; G.order=order; G.table.resize(order*order);
    // r^a * r^b = r^{(a+b)%n},  r^a * s*r^b = s*r^{(b-a+n)%n}
    // s*r^a * r^b = s*r^{(a+b)%n}, s*r^a * s*r^b = r^{(b-a+n)%n}
    for(int i=0;i<order;++i) for(int j=0;j<order;++j){
        bool si=i>=n, sj=j>=n;
        int ri=i%n, rj=j%n;
        int res_rot, res_ref;
        if(!si&&!sj){G.table[i*order+j]=((ri+rj)%n);}
        else if(!si&&sj){G.table[i*order+j]=n+((rj-ri%n+n)%n);}
        else if(si&&!sj){G.table[i*order+j]=n+((ri+rj)%n);}
        else{G.table[i*order+j]=((rj-ri+n)%n);}
    }
    return G;
}

// ─── Symmetric group S_n (permutations) ───────────────────────────────────
// Represent permutations as vectors of size n
using Perm = std::vector<int>;

[[nodiscard]] inline Perm perm_compose(const Perm& a, const Perm& b) noexcept {
    int n=int(a.size()); Perm c(n);
    for(int i=0;i<n;++i) c[i]=a[b[i]];
    return c;
}

[[nodiscard]] inline Perm perm_inverse(const Perm& p) noexcept {
    int n=int(p.size()); Perm inv(n);
    for(int i=0;i<n;++i) inv[p[i]]=i;
    return inv;
}

[[nodiscard]] inline int perm_order(Perm p) noexcept {
    int n=int(p.size()), ord=1;
    std::vector<bool> vis(n,false);
    for(int i=0;i<n;++i){
        if(vis[i]) continue;
        int cycle=0; int j=i;
        while(!vis[j]){vis[j]=true;j=p[j];++cycle;}
        ord=ord*cycle/std::gcd(ord,cycle);
    }
    return ord;
}

// All permutations of {0,..,n-1}
[[nodiscard]] inline std::vector<Perm> symmetric_group(int n) noexcept {
    Perm p(n); std::iota(p.begin(),p.end(),0);
    std::vector<Perm> all;
    do{ all.push_back(p); }while(std::next_permutation(p.begin(),p.end()));
    return all;
}

// Conjugacy class of permutation p
[[nodiscard]] inline std::vector<Perm> conjugacy_class(
    const Perm& p, const std::vector<Perm>& group) noexcept
{
    std::vector<Perm> cl;
    for(auto& g:group){
        Perm conj=perm_compose(g,perm_compose(p,perm_inverse(g)));
        if(!std::count(cl.begin(),cl.end(),conj)) cl.push_back(conj);
    }
    return cl;
}

// ─── Ring operations (Z/nZ) ───────────────────────────────────────────────
struct ZnZ {
    int n;
    [[nodiscard]] int add(int a, int b) const noexcept { return (a+b)%n; }
    [[nodiscard]] int mul(int a, int b) const noexcept { return (a*b)%n; }
    [[nodiscard]] int neg(int a)        const noexcept { return (n-a)%n; }
    [[nodiscard]] std::optional<int> inv(int a) const noexcept {
        // Extended Euclidean
        int x=0,y=1,r0=n,r1=a;
        while(r1){int q=r0/r1;int tmp=r1;r1=r0-q*r1;r0=tmp;tmp=y;y=x-q*y;x=tmp;}
        if(r0!=1) return std::nullopt;
        return ((x%n)+n)%n;
    }
    [[nodiscard]] bool is_unit(int a) const noexcept { return inv(a).has_value(); }
    [[nodiscard]] bool is_field() const noexcept {
        // Z/nZ is a field iff n is prime
        if(n<2) return false;
        for(int i=2;i*i<=n;++i) if(n%i==0) return false;
        return true;
    }
};

// ─── Polynomial ring over Z_p ─────────────────────────────────────────────
struct PolyZp {
    int p; // prime
    std::vector<int> coeffs; // coeffs[i] = coeff of x^i

    [[nodiscard]] int eval(int x) const noexcept {
        int r=0,xpow=1;
        for(auto c:coeffs){r=(r+c*xpow)%p;xpow=xpow*x%p;}
        return r;
    }
    [[nodiscard]] PolyZp add(const PolyZp& o) const noexcept {
        PolyZp r{p};
        int sz=std::max(coeffs.size(),o.coeffs.size());
        r.coeffs.resize(sz,0);
        for(int i=0;i<int(coeffs.size());++i) r.coeffs[i]=(r.coeffs[i]+coeffs[i])%p;
        for(int i=0;i<int(o.coeffs.size());++i) r.coeffs[i]=(r.coeffs[i]+o.coeffs[i])%p;
        return r;
    }
    [[nodiscard]] PolyZp mul(const PolyZp& o) const noexcept {
        PolyZp r{p};
        if(coeffs.empty()||o.coeffs.empty()) return r;
        r.coeffs.resize(coeffs.size()+o.coeffs.size()-1,0);
        for(int i=0;i<int(coeffs.size());++i)
            for(int j=0;j<int(o.coeffs.size());++j)
                r.coeffs[i+j]=(r.coeffs[i+j]+coeffs[i]*o.coeffs[j])%p;
        return r;
    }
    // Count roots in Z_p
    [[nodiscard]] std::vector<int> roots() const noexcept {
        std::vector<int> r;
        for(int x=0;x<p;++x) if(eval(x)==0) r.push_back(x);
        return r;
    }
};

// ─── Group homomorphism ───────────────────────────────────────────────────
// Check if phi:G->H is a group homomorphism (phi represented as array)
[[nodiscard]] inline bool is_homomorphism(
    const FiniteGroup& G, const FiniteGroup& H,
    std::span<const int> phi) noexcept
{
    for(int a=0;a<G.order;++a) for(int b=0;b<G.order;++b){
        int ab=G.op(a,b);
        if(H.op(phi[a],phi[b])!=phi[ab]) return false;
    }
    return true;
}

// Kernel of homomorphism phi
[[nodiscard]] inline std::vector<int> kernel(
    const FiniteGroup& H, std::span<const int> phi) noexcept
{
    int e_H=H.identity();
    std::vector<int> K;
    for(int i=0;i<int(phi.size());++i) if(phi[i]==e_H) K.push_back(i);
    return K;
}

// ─── Lagrange's theorem (index of subgroup) ───────────────────────────────
[[nodiscard]] constexpr int lagrange_index(int group_order, int subgroup_order) noexcept {
    return group_order/subgroup_order;
}

// ─── Sylow subgroups (count) ──────────────────────────────────────────────
// Number of Sylow p-subgroups n_p must satisfy: n_p ≡ 1 (mod p) and n_p | m
// where |G| = p^k * m, gcd(p,m)=1
[[nodiscard]] inline std::vector<int> sylow_count_possibilities(int order, int p) noexcept {
    // Factor out p from order
    int pk=1; while(order%p==0){order/=p;pk*=p;}
    int m=order; // now gcd(m,p)=1
    std::vector<int> candidates;
    for(int n=1;n<=m;n+=p){if(m%n==0) candidates.push_back(n);}
    return candidates;
}

} // namespace qs::math::algebra
