#pragma once
// qs_math/logic/propositional.hpp — Propositional logic, truth tables, normal forms (C++20)
#include "../arithmetic/scalar_types.hpp"
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <optional>
#include <memory>
#include <algorithm>
#include <cassert>

namespace qs {

// ─── Logical connectives ──────────────────────────────────────────────────────

[[nodiscard]] constexpr bool logical_and(bool a, bool b)  { return a && b; }
[[nodiscard]] constexpr bool logical_or(bool a, bool b)   { return a || b; }
[[nodiscard]] constexpr bool logical_not(bool a)           { return !a; }
[[nodiscard]] constexpr bool logical_xor(bool a, bool b)  { return a != b; }
[[nodiscard]] constexpr bool logical_nand(bool a, bool b) { return !(a && b); }
[[nodiscard]] constexpr bool logical_nor(bool a, bool b)  { return !(a || b); }
[[nodiscard]] constexpr bool logical_implies(bool a, bool b)  { return !a || b; }   // a → b
[[nodiscard]] constexpr bool logical_iff(bool a, bool b)      { return a == b; }     // a ↔ b

// ─── Proposition AST ─────────────────────────────────────────────────────────

enum class PropKind { Var, Const, Not, And, Or, Implies, Iff, Xor, Nand, Nor };

struct Prop {
    PropKind kind{PropKind::Const};
    bool const_val{false};
    std::string var_name{};
    std::vector<std::shared_ptr<Prop>> children;

    [[nodiscard]] static std::shared_ptr<Prop> make_var(const std::string& name) {
        auto p=std::make_shared<Prop>(); p->kind=PropKind::Var; p->var_name=name; return p; }
    [[nodiscard]] static std::shared_ptr<Prop> make_const(bool v) {
        auto p=std::make_shared<Prop>(); p->kind=PropKind::Const; p->const_val=v; return p; }
    [[nodiscard]] static std::shared_ptr<Prop> make_not(std::shared_ptr<Prop> a) {
        auto p=std::make_shared<Prop>(); p->kind=PropKind::Not; p->children={a}; return p; }
    [[nodiscard]] static std::shared_ptr<Prop> make_and(std::shared_ptr<Prop> a, std::shared_ptr<Prop> b) {
        auto p=std::make_shared<Prop>(); p->kind=PropKind::And; p->children={a,b}; return p; }
    [[nodiscard]] static std::shared_ptr<Prop> make_or(std::shared_ptr<Prop> a, std::shared_ptr<Prop> b) {
        auto p=std::make_shared<Prop>(); p->kind=PropKind::Or; p->children={a,b}; return p; }
    [[nodiscard]] static std::shared_ptr<Prop> make_implies(std::shared_ptr<Prop> a, std::shared_ptr<Prop> b) {
        auto p=std::make_shared<Prop>(); p->kind=PropKind::Implies; p->children={a,b}; return p; }
    [[nodiscard]] static std::shared_ptr<Prop> make_iff(std::shared_ptr<Prop> a, std::shared_ptr<Prop> b) {
        auto p=std::make_shared<Prop>(); p->kind=PropKind::Iff; p->children={a,b}; return p; }
};

using PropPtr = std::shared_ptr<Prop>;
using Valuation = std::unordered_map<std::string, bool>;

// ─── Evaluation ───────────────────────────────────────────────────────────────

[[nodiscard]] inline bool eval_prop(const PropPtr& p, const Valuation& v) {
    switch(p->kind){
    case PropKind::Const:   return p->const_val;
    case PropKind::Var:     { auto it=v.find(p->var_name); return it!=v.end()?it->second:false; }
    case PropKind::Not:     return !eval_prop(p->children[0],v);
    case PropKind::And:     return eval_prop(p->children[0],v)&&eval_prop(p->children[1],v);
    case PropKind::Or:      return eval_prop(p->children[0],v)||eval_prop(p->children[1],v);
    case PropKind::Implies: return !eval_prop(p->children[0],v)||eval_prop(p->children[1],v);
    case PropKind::Iff:     return eval_prop(p->children[0],v)==eval_prop(p->children[1],v);
    case PropKind::Xor:     return eval_prop(p->children[0],v)!=eval_prop(p->children[1],v);
    case PropKind::Nand:    return !(eval_prop(p->children[0],v)&&eval_prop(p->children[1],v));
    case PropKind::Nor:     return !(eval_prop(p->children[0],v)||eval_prop(p->children[1],v));
    }
    return false;
}

// ─── Variable collection ──────────────────────────────────────────────────────

inline void collect_vars(const PropPtr& p, std::vector<std::string>& vars) {
    if(p->kind==PropKind::Var){
        if(std::find(vars.begin(),vars.end(),p->var_name)==vars.end())
            vars.push_back(p->var_name);
    }
    for(const auto& c:p->children) collect_vars(c,vars);
}

// ─── Truth table ─────────────────────────────────────────────────────────────

struct TruthTable {
    std::vector<std::string> vars;
    std::vector<std::pair<Valuation,bool>> rows;

    [[nodiscard]] static TruthTable build(const PropPtr& p) {
        TruthTable tt;
        collect_vars(p,tt.vars);
        u32 n=u32(tt.vars.size());
        for(u32 mask=0;mask<(1u<<n);++mask){
            Valuation v;
            for(u32 i=0;i<n;++i) v[tt.vars[i]]=(mask>>i)&1;
            tt.rows.push_back({v,eval_prop(p,v)});
        }
        return tt;
    }

    [[nodiscard]] bool is_tautology() const {
        for(const auto& [v,r]:rows) if(!r) return false;
        return true;
    }

    [[nodiscard]] bool is_contradiction() const {
        for(const auto& [v,r]:rows) if(r) return false;
        return true;
    }

    [[nodiscard]] bool is_satisfiable() const {
        for(const auto& [v,r]:rows) if(r) return true;
        return false;
    }

    [[nodiscard]] std::optional<Valuation> satisfying_assignment() const {
        for(const auto& [v,r]:rows) if(r) return v;
        return std::nullopt;
    }
};

// ─── Logical equivalence check ────────────────────────────────────────────────

[[nodiscard]] inline bool logically_equivalent(const PropPtr& a, const PropPtr& b) {
    auto iff=Prop::make_iff(a,b);
    auto tt=TruthTable::build(iff);
    return tt.is_tautology();
}

// ─── Normal forms ─────────────────────────────────────────────────────────────

// Push negation inward (NNF: Negation Normal Form)
[[nodiscard]] PropPtr to_nnf(const PropPtr& p, bool negated=false);
[[nodiscard]] inline PropPtr to_nnf(const PropPtr& p, bool negated) {
    switch(p->kind){
    case PropKind::Const:
        return Prop::make_const(negated?!p->const_val:p->const_val);
    case PropKind::Var:
        return negated?Prop::make_not(p):p;
    case PropKind::Not:
        return to_nnf(p->children[0],!negated);
    case PropKind::And:
        if(negated) return Prop::make_or(to_nnf(p->children[0],true),to_nnf(p->children[1],true));
        return Prop::make_and(to_nnf(p->children[0],false),to_nnf(p->children[1],false));
    case PropKind::Or:
        if(negated) return Prop::make_and(to_nnf(p->children[0],true),to_nnf(p->children[1],true));
        return Prop::make_or(to_nnf(p->children[0],false),to_nnf(p->children[1],false));
    case PropKind::Implies:
        // a→b ≡ ¬a∨b
        return to_nnf(Prop::make_or(Prop::make_not(p->children[0]),p->children[1]),negated);
    case PropKind::Iff:
        // a↔b ≡ (a∧b)∨(¬a∧¬b)
        return to_nnf(Prop::make_or(
            Prop::make_and(p->children[0],p->children[1]),
            Prop::make_and(Prop::make_not(p->children[0]),Prop::make_not(p->children[1]))),negated);
    default:
        return p;
    }
}

// Distribute OR over AND to get CNF clauses
[[nodiscard]] inline std::vector<std::vector<std::pair<std::string,bool>>>
collect_cnf_clauses(const PropPtr& p) {
    std::vector<std::vector<std::pair<std::string,bool>>> clauses;
    if(p->kind==PropKind::Var)
        clauses.push_back({{p->var_name,true}});
    else if(p->kind==PropKind::Not&&p->children[0]->kind==PropKind::Var)
        clauses.push_back({{p->children[0]->var_name,false}});
    else if(p->kind==PropKind::And){
        auto L=collect_cnf_clauses(p->children[0]);
        auto R=collect_cnf_clauses(p->children[1]);
        for(auto& c:L) clauses.push_back(c);
        for(auto& c:R) clauses.push_back(c);
    } else if(p->kind==PropKind::Or){
        // Distribute: flatten OR into one clause
        std::vector<std::pair<std::string,bool>> clause;
        std::function<void(const PropPtr&)> flatten=[&](const PropPtr& q){
            if(q->kind==PropKind::Var) clause.push_back({q->var_name,true});
            else if(q->kind==PropKind::Not&&q->children[0]->kind==PropKind::Var)
                clause.push_back({q->children[0]->var_name,false});
            else if(q->kind==PropKind::Or){flatten(q->children[0]);flatten(q->children[1]);}
        };
        flatten(p);
        clauses.push_back(clause);
    }
    return clauses;
}

// ─── Resolution (propositional) ───────────────────────────────────────────────

using Clause = std::vector<std::pair<std::string,bool>>; // literal = (var, positive)

[[nodiscard]] inline std::optional<Clause> resolve(const Clause& C1, const Clause& C2) {
    std::optional<std::string> pivot;
    for(const auto& [v,p]:C1)
    for(const auto& [v2,p2]:C2)
        if(v==v2&&p!=p2){ if(pivot) return std::nullopt; pivot=v; }
    if(!pivot) return std::nullopt;
    Clause resolvent;
    for(const auto& lit:C1) if(lit.first!=*pivot) resolvent.push_back(lit);
    for(const auto& lit:C2) if(lit.first!=*pivot) resolvent.push_back(lit);
    // Remove duplicates
    std::sort(resolvent.begin(),resolvent.end());
    resolvent.erase(std::unique(resolvent.begin(),resolvent.end()),resolvent.end());
    return resolvent;
}

// Check unsatisfiability via resolution (simple, exponential in worst case)
[[nodiscard]] inline bool resolution_refutation(std::vector<Clause> clauses) {
    bool changed=true;
    while(changed){
        changed=false;
        std::vector<Clause> new_clauses;
        for(u32 i=0;i<clauses.size();++i)
        for(u32 j=i+1;j<clauses.size();++j){
            auto r=resolve(clauses[i],clauses[j]);
            if(r){
                if(r->empty()) return true; // derived empty clause = unsatisfiable
                bool dup=false;
                for(const auto& c:clauses) if(c==*r){dup=true;break;}
                if(!dup){new_clauses.push_back(*r);changed=true;}
            }
        }
        for(auto& c:new_clauses) clauses.push_back(c);
    }
    return false;
}

// ─── DPLL SAT solver (Davis-Putnam-Logemann-Loveland) ─────────────────────────

[[nodiscard]] bool dpll(std::vector<Clause> clauses, Valuation& assignment);

inline void unit_propagate(std::vector<Clause>& clauses, Valuation& assign) {
    bool changed=true;
    while(changed){
        changed=false;
        for(auto& c:clauses){
            if(c.size()==1){
                auto [v,p]=c[0];
                if(assign.find(v)==assign.end()){assign[v]=p;changed=true;}
            }
        }
        if(!changed) break;
        // Apply assignment
        std::vector<Clause> new_clauses;
        for(auto& c:clauses){
            bool sat=false;
            Clause nc;
            for(auto& [v,p]:c){
                auto it=assign.find(v);
                if(it!=assign.end()){if(it->second==p){sat=true;break;}}
                else nc.push_back({v,p});
            }
            if(!sat) new_clauses.push_back(nc);
        }
        clauses=new_clauses;
    }
}

[[nodiscard]] inline bool dpll(std::vector<Clause> clauses, Valuation& assignment) {
    unit_propagate(clauses,assignment);
    if(clauses.empty()) return true;
    for(const auto& c:clauses) if(c.empty()) return false;
    // Choose variable
    std::string var=clauses[0][0].first;
    for(bool val:{true,false}){
        Valuation a2=assignment; a2[var]=val;
        std::vector<Clause> c2;
        for(auto& c:clauses){
            bool sat=false; Clause nc;
            for(auto& [v,p]:c){
                auto it=a2.find(v);
                if(it!=a2.end()){if(it->second==p){sat=true;break;}}
                else nc.push_back({v,p});
            }
            if(!sat) c2.push_back(nc);
        }
        if(dpll(c2,a2)){assignment=a2;return true;}
    }
    return false;
}

} // namespace qs
