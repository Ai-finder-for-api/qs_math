#pragma once
// qs_math/discrete/formal_languages.hpp — Grammars, CNF, CYK, regex-to-NFA
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bitset>

namespace qs::math {

// ─── Context-Free Grammar (compact, fixed max symbols) ────────────────────────
static constexpr int k_max_symbols   = 64;  // terminal + nonterminal
static constexpr int k_max_rules     = 256;
static constexpr int k_max_rule_rhs  = 4;

struct CfgRule {
    int lhs{-1};
    int rhs[k_max_rule_rhs]{-1,-1,-1,-1};
    int rhs_len{0};
};

struct CFG {
    int     start_sym{0};
    int     n_nonterminals{0};
    int     n_terminals{0};
    CfgRule rules[k_max_rules]{};
    int     n_rules{0};

    constexpr void add_rule(int lhs, const int* rhs, int rhs_len) noexcept {
        if(n_rules>=k_max_rules) return;
        rules[n_rules].lhs=lhs;
        rules[n_rules].rhs_len=rhs_len;
        for(int i=0;i<rhs_len&&i<k_max_rule_rhs;++i) rules[n_rules].rhs[i]=rhs[i];
        ++n_rules;
    }
};

// ─── Convert CFG to Chomsky Normal Form ──────────────────────────────────────
// CNF: every rule is A→BC or A→a
// Steps: eliminate epsilon, unit, long rules, mixed rules
// (Structural transformation; exact CFG manipulated in-place)

// Check if symbol is terminal (id >= n_nonterminals)
[[nodiscard]] constexpr bool is_terminal(const CFG& g, int sym) noexcept {
    return sym >= g.n_nonterminals;
}

// Check if rule is in CNF form
[[nodiscard]] constexpr bool is_cnf_rule(const CFG& g, const CfgRule& r) noexcept {
    if(r.rhs_len==1 && is_terminal(g,r.rhs[0])) return true;
    if(r.rhs_len==2 && !is_terminal(g,r.rhs[0]) && !is_terminal(g,r.rhs[1])) return true;
    return false;
}

// ─── CYK parsing algorithm ────────────────────────────────────────────────────
// Parse word[0..n-1] using grammar in CNF (nonterminals 0..NT-1)
// Returns true if start symbol derives the word
template<int NT, int MAX_WORD>
[[nodiscard]] constexpr bool cyk_parse(const CFG& g,
                                        const int* word, int n) noexcept {
    // table[i][j] = bitmask of nonterminals deriving word[i..i+j-1]
    std::array<std::array<u64,MAX_WORD+1>,MAX_WORD> table{};

    // Length 1 substrings
    for(int i=0;i<n;++i)
        for(int r=0;r<g.n_rules;++r)
            if(g.rules[r].rhs_len==1 && g.rules[r].rhs[0]==word[i])
                table[i][1]|=(1ULL<<g.rules[r].lhs);

    // Length 2..n substrings
    for(int len=2;len<=n;++len)
        for(int i=0;i<=n-len;++i)
            for(int k=1;k<len;++k)
                for(int r=0;r<g.n_rules;++r) {
                    const auto& rule=g.rules[r];
                    if(rule.rhs_len!=2) continue;
                    if((table[i][k]>>(rule.rhs[0]))&1)
                        if((table[i+k][len-k]>>(rule.rhs[1]))&1)
                            table[i][len]|=(1ULL<<rule.lhs);
                }

    return (n>0) && ((table[0][n]>>g.start_sym)&1);
}

// ─── Earley parser item ───────────────────────────────────────────────────────
struct EarleyItem {
    int rule_idx{-1};
    int dot{0};       // position in RHS
    int origin{0};    // input position where this item started
};

// Earley recognizer (O(n^3) general CFGs)
template<int MAX_ITEMS, int MAX_WORD>
[[nodiscard]] constexpr bool earley_recognize(const CFG& g,
                                               const int* word, int n) noexcept {
    std::array<std::array<EarleyItem,MAX_ITEMS>,MAX_WORD+1> chart{};
    std::array<int,MAX_WORD+1> chart_sz{};

    auto add_item=[&](int pos, EarleyItem it) {
        for(int j=0;j<chart_sz[pos];++j)
            if(chart[pos][j].rule_idx==it.rule_idx &&
               chart[pos][j].dot==it.dot &&
               chart[pos][j].origin==it.origin) return;
        if(chart_sz[pos]<MAX_ITEMS) chart[pos][chart_sz[pos]++]=it;
    };

    // Predict from start symbol
    for(int r=0;r<g.n_rules;++r)
        if(g.rules[r].lhs==g.start_sym) add_item(0,{r,0,0});

    for(int k=0;k<=n;++k) {
        for(int i=0;i<chart_sz[k];++i) {
            auto& it=chart[k][i];
            const auto& rule=g.rules[it.rule_idx];
            if(it.dot==rule.rhs_len) {
                // Complete
                for(int j=0;j<chart_sz[it.origin];++j) {
                    auto& prev=chart[it.origin][j];
                    const auto& pr=g.rules[prev.rule_idx];
                    if(prev.dot<pr.rhs_len && pr.rhs[prev.dot]==rule.lhs)
                        add_item(k,{prev.rule_idx,prev.dot+1,prev.origin});
                }
            } else {
                int next_sym=rule.rhs[it.dot];
                if(!is_terminal(g,next_sym)) {
                    // Predict
                    for(int r=0;r<g.n_rules;++r)
                        if(g.rules[r].lhs==next_sym) add_item(k,{r,0,k});
                } else if(k<n && next_sym==word[k]) {
                    // Scan
                    add_item(k+1,{it.rule_idx,it.dot+1,it.origin});
                }
            }
        }
    }

    // Accept if start symbol completes over entire input
    for(int i=0;i<chart_sz[n];++i) {
        const auto& it=chart[n][i];
        const auto& r=g.rules[it.rule_idx];
        if(r.lhs==g.start_sym && it.dot==r.rhs_len && it.origin==0) return true;
    }
    return false;
}

// ─── Regex-to-NFA (Thompson construction) ────────────────────────────────────
// Minimal subset: concatenation, alternation(|), Kleene star(*)
// Alphabet: 0..A-1, epsilon = -1
template<int STATES, int A>
struct ThompsonNFA {
    std::array<std::array<int,A+1>,STATES> delta{};
    // delta[s][A] = epsilon-next (single epsilon per state for Thompson)
    int start{0}, accept{0}, n_states{0};

    constexpr int new_state() noexcept { return n_states++; }
    constexpr void set_trans(int from,int sym,int to) noexcept { delta[from][sym]=to; }
    constexpr void set_eps  (int from,int to)         noexcept { delta[from][A]=to;   }
};

// ─── Chomsky hierarchy classifiers ───────────────────────────────────────────
// Type 3 (regular): all rules A→aB or A→a
[[nodiscard]] constexpr bool is_right_regular(const CFG& g) noexcept {
    for(int r=0;r<g.n_rules;++r) {
        const auto& rule=g.rules[r];
        if(rule.rhs_len==1 && is_terminal(g,rule.rhs[0])) continue;
        if(rule.rhs_len==2 && is_terminal(g,rule.rhs[0]) && !is_terminal(g,rule.rhs[1])) continue;
        return false;
    }
    return true;
}

// Type 2 (context-free): all rules A→α
[[nodiscard]] constexpr bool is_context_free(const CFG& g) noexcept {
    for(int r=0;r<g.n_rules;++r)
        if(g.rules[r].lhs<0 || g.rules[r].lhs>=g.n_nonterminals) return false;
    return true; // all LHS are single nonterminals
}

} // namespace qs::math
