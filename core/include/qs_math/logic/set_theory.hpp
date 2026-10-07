#pragma once
// qs_math/logic/set_theory.hpp — Formal set operations, relations, ZFC axioms (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <optional>
#include <algorithm>
#include <string>
#include <cassert>
#include <span>

namespace qs {

// ─── Finite set ───────────────────────────────────────────────────────────────

template<typename T>
struct FiniteSet {
    std::vector<T> elements;

    FiniteSet() = default;
    explicit FiniteSet(std::initializer_list<T> il) : elements(il) { deduplicate(); }
    explicit FiniteSet(std::vector<T> v) : elements(std::move(v)) { deduplicate(); }

    void deduplicate() {
        std::sort(elements.begin(),elements.end());
        elements.erase(std::unique(elements.begin(),elements.end()),elements.end());
    }

    void insert(const T& x) {
        if(!contains(x)){elements.push_back(x);std::sort(elements.begin(),elements.end());}
    }

    void remove(const T& x) {
        elements.erase(std::remove(elements.begin(),elements.end(),x),elements.end());
    }

    [[nodiscard]] bool contains(const T& x) const {
        return std::binary_search(elements.begin(),elements.end(),x);
    }

    [[nodiscard]] u32 cardinality() const { return u32(elements.size()); }
    [[nodiscard]] bool empty() const { return elements.empty(); }

    // Equality
    [[nodiscard]] bool operator==(const FiniteSet& o) const { return elements==o.elements; }
    [[nodiscard]] bool operator!=(const FiniteSet& o) const { return !(*this==o); }

    // Subset: A ⊆ B
    [[nodiscard]] bool is_subset_of(const FiniteSet& B) const {
        for(const auto& x:elements) if(!B.contains(x)) return false;
        return true;
    }

    // Proper subset: A ⊂ B
    [[nodiscard]] bool is_proper_subset_of(const FiniteSet& B) const {
        return is_subset_of(B) && !(*this==B);
    }

    // Union: A ∪ B
    [[nodiscard]] FiniteSet set_union(const FiniteSet& B) const {
        FiniteSet r=*this;
        for(const auto& x:B.elements) r.insert(x);
        return r;
    }

    // Intersection: A ∩ B
    [[nodiscard]] FiniteSet set_intersection(const FiniteSet& B) const {
        FiniteSet r;
        for(const auto& x:elements) if(B.contains(x)) r.elements.push_back(x);
        return r;
    }

    // Difference: A \ B
    [[nodiscard]] FiniteSet set_difference(const FiniteSet& B) const {
        FiniteSet r;
        for(const auto& x:elements) if(!B.contains(x)) r.elements.push_back(x);
        return r;
    }

    // Symmetric difference: A △ B = (A\B) ∪ (B\A)
    [[nodiscard]] FiniteSet symmetric_difference(const FiniteSet& B) const {
        return set_difference(B).set_union(B.set_difference(*this));
    }

    // Power set: P(A)
    [[nodiscard]] FiniteSet<FiniteSet<T>> power_set() const {
        FiniteSet<FiniteSet<T>> P;
        u32 n=cardinality();
        for(u32 mask=0;mask<(1u<<n);++mask){
            FiniteSet<T> sub;
            for(u32 i=0;i<n;++i) if(mask>>i&1) sub.elements.push_back(elements[i]);
            P.elements.push_back(sub);
        }
        return P;
    }

    // Cartesian product: A × B
    template<typename U>
    [[nodiscard]] FiniteSet<std::pair<T,U>> cartesian_product(const FiniteSet<U>& B) const {
        FiniteSet<std::pair<T,U>> r;
        for(const auto& a:elements) for(const auto& b:B.elements)
            r.elements.push_back({a,b});
        return r;
    }

    // Partition: collection of disjoint sets covering this set
    [[nodiscard]] bool is_partition(const std::vector<FiniteSet<T>>& parts) const {
        // Check disjointness
        for(u32 i=0;i<parts.size();++i)
        for(u32 j=i+1;j<parts.size();++j)
            if(!parts[i].set_intersection(parts[j]).empty()) return false;
        // Check coverage
        FiniteSet<T> union_all;
        for(const auto& p:parts) union_all=union_all.set_union(p);
        return union_all==*this;
    }
};

// ─── Relations ────────────────────────────────────────────────────────────────

// Binary relation R ⊆ A × A
template<typename T>
struct BinaryRelation {
    std::vector<std::pair<T,T>> pairs;
    FiniteSet<T> domain;

    void add(const T& a, const T& b) { pairs.push_back({a,b}); }

    [[nodiscard]] bool relates(const T& a, const T& b) const {
        for(const auto& [x,y]:pairs) if(x==a&&y==b) return true;
        return false;
    }

    // Reflexive: (a,a) ∈ R for all a ∈ A
    [[nodiscard]] bool is_reflexive() const {
        for(const auto& a:domain.elements) if(!relates(a,a)) return false;
        return true;
    }

    // Symmetric: (a,b) ∈ R => (b,a) ∈ R
    [[nodiscard]] bool is_symmetric() const {
        for(const auto& [a,b]:pairs) if(!relates(b,a)) return false;
        return true;
    }

    // Antisymmetric: (a,b)∈R ∧ (b,a)∈R => a=b
    [[nodiscard]] bool is_antisymmetric() const {
        for(const auto& [a,b]:pairs)
            if(a!=b&&relates(b,a)) return false;
        return true;
    }

    // Transitive: (a,b)∈R ∧ (b,c)∈R => (a,c)∈R
    [[nodiscard]] bool is_transitive() const {
        for(const auto& [a,b]:pairs)
        for(const auto& [b2,c]:pairs)
            if(b==b2&&!relates(a,c)) return false;
        return true;
    }

    [[nodiscard]] bool is_equivalence_relation() const {
        return is_reflexive()&&is_symmetric()&&is_transitive();
    }

    [[nodiscard]] bool is_partial_order() const {
        return is_reflexive()&&is_antisymmetric()&&is_transitive();
    }

    // Equivalence class of element a
    [[nodiscard]] FiniteSet<T> equivalence_class(const T& a) const {
        FiniteSet<T> cls;
        for(const auto& x:domain.elements) if(relates(a,x)) cls.insert(x);
        return cls;
    }

    // Quotient set A/R
    [[nodiscard]] std::vector<FiniteSet<T>> quotient_set() const {
        std::vector<FiniteSet<T>> classes;
        std::vector<bool> assigned(domain.cardinality(),false);
        for(u32 i=0;i<domain.cardinality();++i){
            if(assigned[i]) continue;
            FiniteSet<T> cls=equivalence_class(domain.elements[i]);
            for(u32 j=0;j<domain.cardinality();++j)
                if(cls.contains(domain.elements[j])) assigned[j]=true;
            classes.push_back(cls);
        }
        return classes;
    }

    // Transitive closure (Floyd-Warshall on finite set)
    [[nodiscard]] BinaryRelation transitive_closure() const {
        BinaryRelation r=*this;
        for(const auto& b:domain.elements)
        for(const auto& a:domain.elements)
        for(const auto& c:domain.elements)
            if(r.relates(a,b)&&r.relates(b,c)) r.add(a,c);
        return r;
    }

    // Composition R ∘ S: (a,c) if exists b: (a,b)∈S ∧ (b,c)∈R
    [[nodiscard]] BinaryRelation compose(const BinaryRelation& S) const {
        BinaryRelation comp; comp.domain=domain;
        for(const auto& [a,b]:S.pairs)
        for(const auto& [b2,c]:pairs)
            if(b==b2) comp.add(a,c);
        return comp;
    }
};

// ─── Functions (as relations) ─────────────────────────────────────────────────

template<typename A, typename B>
struct SetFunction {
    std::vector<std::pair<A,B>> graph; // set of (a,b) pairs
    FiniteSet<A> domain_set;
    FiniteSet<B> codomain_set;

    void define(const A& a, const B& b) { graph.push_back({a,b}); }

    [[nodiscard]] std::optional<B> apply(const A& a) const {
        for(const auto& [x,y]:graph) if(x==a) return y;
        return std::nullopt;
    }

    [[nodiscard]] bool is_function() const {
        // Every element of domain has exactly one image
        for(const auto& a:domain_set.elements){
            u32 count=0;
            for(const auto& [x,y]:graph) if(x==a) ++count;
            if(count!=1) return false;
        }
        return true;
    }

    [[nodiscard]] bool is_injective() const {
        // Different inputs => different outputs
        for(u32 i=0;i<graph.size();++i)
        for(u32 j=i+1;j<graph.size();++j)
            if(graph[i].second==graph[j].second&&graph[i].first!=graph[j].first) return false;
        return true;
    }

    [[nodiscard]] bool is_surjective() const {
        for(const auto& b:codomain_set.elements){
            bool found=false;
            for(const auto& [a,y]:graph) if(y==b){found=true;break;}
            if(!found) return false;
        }
        return true;
    }

    [[nodiscard]] bool is_bijective() const { return is_injective()&&is_surjective(); }

    // Image of subset S
    [[nodiscard]] FiniteSet<B> image(const FiniteSet<A>& S) const {
        FiniteSet<B> img;
        for(const auto& [a,b]:graph) if(S.contains(a)) img.insert(b);
        return img;
    }

    // Preimage of subset T
    [[nodiscard]] FiniteSet<A> preimage(const FiniteSet<B>& T) const {
        FiniteSet<A> pre;
        for(const auto& [a,b]:graph) if(T.contains(b)) pre.insert(a);
        return pre;
    }
};

// ─── Ordinals (Von Neumann, finite) ──────────────────────────────────────────

// In ZFC: 0={}, 1={0}, 2={0,1}, 3={0,1,2}, ...
struct VonNeumannOrdinal {
    u64 n{};  // represents ordinal n

    [[nodiscard]] bool operator<(const VonNeumannOrdinal& o) const { return n<o.n; }
    [[nodiscard]] bool operator==(const VonNeumannOrdinal& o) const { return n==o.n; }

    // Successor: S(n) = n+1
    [[nodiscard]] VonNeumannOrdinal successor() const { return {n+1}; }

    // Cardinal number (for finite ordinals = ordinal itself)
    [[nodiscard]] u64 cardinal() const { return n; }
};

// ─── ZFC Axiom verification helpers ──────────────────────────────────────────

// Axiom of Extensionality: A=B iff ∀x(x∈A ↔ x∈B)
template<typename T>
[[nodiscard]] bool axiom_extensionality(const FiniteSet<T>& A, const FiniteSet<T>& B) {
    return A==B;
}

// Axiom of Pairing: {a,b} exists
template<typename T>
[[nodiscard]] FiniteSet<T> axiom_pairing(const T& a, const T& b) {
    return FiniteSet<T>({a,b});
}

// Axiom of Union: ∪A exists (union of all sets in a collection)
template<typename T>
[[nodiscard]] FiniteSet<T> axiom_union(const std::vector<FiniteSet<T>>& collection) {
    FiniteSet<T> result;
    for(const auto& S:collection) result=result.set_union(S);
    return result;
}

// Axiom of Power Set: P(A) exists
template<typename T>
[[nodiscard]] auto axiom_power_set(const FiniteSet<T>& A) {
    return A.power_set();
}

// Axiom Schema of Separation: {x ∈ A : φ(x)}
template<typename T>
[[nodiscard]] FiniteSet<T> separation(const FiniteSet<T>& A, std::function<bool(const T&)> phi) {
    FiniteSet<T> result;
    for(const auto& x:A.elements) if(phi(x)) result.elements.push_back(x);
    return result;
}

// ─── Well-ordering ────────────────────────────────────────────────────────────

template<typename T>
[[nodiscard]] bool is_well_ordered(const FiniteSet<T>& S, const BinaryRelation<T>& order) {
    if(!order.is_partial_order()) return false;
    // Every non-empty subset has a minimum element
    for(u32 mask=1;mask<(1u<<S.cardinality());++mask){
        FiniteSet<T> sub;
        for(u32 i=0;i<S.cardinality();++i) if(mask>>i&1) sub.insert(S.elements[i]);
        bool has_min=false;
        for(const auto& a:sub.elements){
            bool is_min=true;
            for(const auto& b:sub.elements)
                if(b!=a&&!order.relates(a,b)){ is_min=false; break; }
            if(is_min){has_min=true;break;}
        }
        if(!has_min) return false;
    }
    return true;
}

} // namespace qs
