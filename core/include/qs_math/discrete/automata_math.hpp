#pragma once
// qs_math/discrete/automata_math.hpp — DFA/NFA math, language ops, pumping lemma tools
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bitset>

namespace qs::math {

// ─── DFA with N states, A alphabet symbols ────────────────────────────────────
template<int N, int A>
struct DFA {
    std::array<std::array<int,A>,N> delta{};  // transition function
    std::array<bool,N>              accept{};
    int                             start{0};

    [[nodiscard]] constexpr int run(const int* word, int len) const noexcept {
        int q=start;
        for(int i=0;i<len;++i) {
            if(word[i]<0||word[i]>=A) return -1;
            q=delta[q][word[i]];
        }
        return q;
    }

    [[nodiscard]] constexpr bool accepts(const int* word, int len) const noexcept {
        int q=run(word,len);
        return q>=0 && accept[q];
    }

    // Number of strings of length exactly L accepted
    [[nodiscard]] constexpr i64 count_accepted(int L) const noexcept {
        // DP: dp[q] = number of length-L strings reaching state q
        std::array<i64,N> dp{}, ndp{};
        dp[start]=1;
        for(int step=0;step<L;++step) {
            ndp={};
            for(int q=0;q<N;++q) if(dp[q])
                for(int c=0;c<A;++c) ndp[delta[q][c]]+=dp[q];
            dp=ndp;
        }
        i64 total=0;
        for(int q=0;q<N;++q) if(accept[q]) total+=dp[q];
        return total;
    }

    // Reachable states via BFS bitmask
    [[nodiscard]] constexpr std::bitset<N> reachable() const noexcept {
        std::bitset<N> vis; vis.set(start);
        bool changed=true;
        while(changed) {
            changed=false;
            for(int q=0;q<N;++q) if(vis[q])
                for(int c=0;c<A;++c)
                    if(!vis[delta[q][c]]){vis.set(delta[q][c]);changed=true;}
        }
        return vis;
    }

    // Minimization via Hopcroft (partition refinement, N≤64)
    [[nodiscard]] constexpr DFA<N,A> minimize() const noexcept {
        // Myhill-Nerode partition: start with {accept, non-accept}
        std::array<int,N> part{};
        for(int q=0;q<N;++q) part[q]=accept[q]?1:0;
        bool changed=true;
        while(changed) {
            changed=false;
            for(int q=0;q<N;++q)
                for(int c=0;c<A;++c) {
                    int target_cls=part[delta[q][c]];
                    // Split class of q if successor class differs from same-class peer
                    for(int p=0;p<N;++p)
                        if(part[p]==part[q]&&part[delta[p][c]]!=target_cls){
                            // Assign new class to p
                            int new_cls=0;
                            for(int x=0;x<N;++x) if(part[x]>new_cls) new_cls=part[x];
                            part[p]=new_cls+1; changed=true;
                        }
                }
        }
        // Build minimized DFA from equivalence classes
        DFA<N,A> min_dfa{};
        for(int q=0;q<N;++q) {
            for(int c=0;c<A;++c) min_dfa.delta[part[q]][c]=part[delta[q][c]];
            if(accept[q]) min_dfa.accept[part[q]]=true;
        }
        min_dfa.start=part[start];
        return min_dfa;
    }
};

// ─── NFA with epsilon transitions ────────────────────────────────────────────
template<int N, int A>
struct NFA {
    // delta[state][symbol] = bitmask of successor states
    // delta[state][A]      = epsilon closure
    std::array<std::array<u64,A+1>,N> delta{};
    u64  accept_mask{0};
    int  start{0};

    constexpr void add_transition(int from, int sym, int to) noexcept {
        delta[from][sym] |= (1ULL<<to);
    }
    constexpr void add_epsilon(int from, int to) noexcept {
        delta[from][A] |= (1ULL<<to);
    }
    constexpr void set_accept(int q) noexcept { accept_mask|=(1ULL<<q); }

    [[nodiscard]] constexpr u64 epsilon_closure(u64 states) const noexcept {
        u64 closure=states;
        bool changed=true;
        while(changed) {
            changed=false;
            u64 tmp=closure;
            for(int q=0;q<N;++q) if(closure>>q&1) {
                u64 eps=delta[q][A];
                if((closure|eps)!=closure){closure|=eps;changed=true;}
            }
        }
        return closure;
    }

    [[nodiscard]] constexpr bool accepts(const int* word, int len) const noexcept {
        u64 current=epsilon_closure(1ULL<<start);
        for(int i=0;i<len;++i) {
            u64 next=0;
            for(int q=0;q<N;++q) if(current>>q&1) next|=delta[q][word[i]];
            current=epsilon_closure(next);
        }
        return (current&accept_mask)!=0;
    }

    // Subset construction: NFA→DFA
    [[nodiscard]] constexpr DFA<(1<<N),A> to_dfa() const noexcept {
        DFA<(1<<N),A> dfa{};
        dfa.start = static_cast<int>(epsilon_closure(1ULL<<start));
        // BFS over subset states
        u64 worklist[1<<N]; int wl_sz=0;
        worklist[wl_sz++]=(u64)dfa.start;
        bool visited[1<<N]{};
        visited[dfa.start]=true;
        while(wl_sz>0) {
            u64 S=worklist[--wl_sz];
            if(S&accept_mask) dfa.accept[S]=true;
            for(int c=0;c<A;++c) {
                u64 T=0;
                for(int q=0;q<N;++q) if(S>>q&1) T|=delta[q][c];
                T=epsilon_closure(T);
                dfa.delta[S][c]=(int)T;
                if(!visited[T]){visited[T]=true;worklist[wl_sz++]=T;}
            }
        }
        return dfa;
    }
};

// ─── Pumping lemma witness finder ────────────────────────────────────────────
// For a DFA, find pumping length p and a witness string of length ≥ p
template<int N, int A>
[[nodiscard]] constexpr int pumping_length(const DFA<N,A>& dfa) noexcept {
    return N; // By pigeonhole, pumping length = number of states
}

// ─── Myhill-Nerode equivalence class count ────────────────────────────────────
template<int N, int A>
[[nodiscard]] constexpr int nerode_index(const DFA<N,A>& dfa) noexcept {
    // Count distinguishable state classes = number of classes in minimized DFA
    std::array<int,N> part{};
    for(int q=0;q<N;++q) part[q]=dfa.accept[q]?1:0;
    int num_classes=2;
    bool changed=true;
    while(changed){
        changed=false;
        for(int q=0;q<N&&!changed;++q)
            for(int c=0;c<A&&!changed;++c)
                for(int p=q+1;p<N&&!changed;++p)
                    if(part[p]==part[q]&&part[dfa.delta[p][c]]!=part[dfa.delta[q][c]])
                    { part[p]=num_classes++;changed=true; }
    }
    return num_classes;
}

// ─── Regular language intersection (product construction) ─────────────────────
template<int N1, int N2, int A>
[[nodiscard]] constexpr DFA<N1*N2,A> dfa_intersect(
    const DFA<N1,A>& d1, const DFA<N2,A>& d2) noexcept
{
    DFA<N1*N2,A> prod{};
    prod.start = d1.start*N2 + d2.start;
    for(int q1=0;q1<N1;++q1)
        for(int q2=0;q2<N2;++q2) {
            int q=q1*N2+q2;
            for(int c=0;c<A;++c)
                prod.delta[q][c]=d1.delta[q1][c]*N2+d2.delta[q2][c];
            prod.accept[q]=(d1.accept[q1]&&d2.accept[q2]);
        }
    return prod;
}

} // namespace qs::math
