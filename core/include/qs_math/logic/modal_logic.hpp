#pragma once
// qs_math/logic/modal_logic.hpp — Kripke frames, accessibility, S4/S5, temporal logic
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <bitset>

namespace qs::math {

// ─── Kripke frame: worlds W and accessibility relation R ──────────────────────
template<int N>
struct KripkeFrame {
    std::array<u64,N> R{}; // R[w] = bitmask of worlds accessible from w

    constexpr void add_access(int w, int v) noexcept { R[w]|=(1ULL<<v); }
    [[nodiscard]] constexpr bool accessible(int w, int v) const noexcept {
        return (R[w]>>v)&1;
    }
    // Reflexive: R(w,w) for all w
    [[nodiscard]] constexpr bool is_reflexive(int n=N) const noexcept {
        for(int w=0;w<n;++w) if(!accessible(w,w)) return false;
        return true;
    }
    // Transitive: R(w,u) and R(u,v) → R(w,v)
    [[nodiscard]] constexpr bool is_transitive(int n=N) const noexcept {
        for(int w=0;w<n;++w) for(int u=0;u<n;++u) if(accessible(w,u))
            for(int v=0;v<n;++v) if(accessible(u,v)&&!accessible(w,v)) return false;
        return true;
    }
    // Symmetric: R(w,v) → R(v,w)
    [[nodiscard]] constexpr bool is_symmetric(int n=N) const noexcept {
        for(int w=0;w<n;++w) for(int v=0;v<n;++v)
            if(accessible(w,v)&&!accessible(v,w)) return false;
        return true;
    }
    // Euclidean: R(w,u) and R(w,v) → R(u,v)
    [[nodiscard]] constexpr bool is_euclidean(int n=N) const noexcept {
        for(int w=0;w<n;++w) for(int u=0;u<n;++u) if(accessible(w,u))
            for(int v=0;v<n;++v) if(accessible(w,v)&&!accessible(u,v)) return false;
        return true;
    }
    // Serial: every world has at least one accessible world
    [[nodiscard]] constexpr bool is_serial(int n=N) const noexcept {
        for(int w=0;w<n;++w) if(!R[w]) return false;
        return true;
    }
    // S4 frame = reflexive + transitive
    [[nodiscard]] constexpr bool is_S4() const noexcept {
        return is_reflexive()&&is_transitive();
    }
    // S5 frame = reflexive + transitive + symmetric = equivalence relation
    [[nodiscard]] constexpr bool is_S5() const noexcept {
        return is_reflexive()&&is_transitive()&&is_symmetric();
    }
    // Transitive closure (Floyd-Warshall on bitmasks)
    [[nodiscard]] constexpr KripkeFrame transitive_closure(int n=N) const noexcept {
        KripkeFrame f=*this;
        for(int k=0;k<n;++k) for(int i=0;i<n;++i) if((f.R[i]>>k)&1)
            f.R[i]|=f.R[k];
        return f;
    }
};

// ─── Kripke model: frame + valuation ─────────────────────────────────────────
template<int N, int NPROPS>
struct KripkeModel {
    KripkeFrame<N> frame{};
    std::array<u64,NPROPS> val{}; // val[p] = bitmask of worlds where p is true

    [[nodiscard]] constexpr bool prop_true(int p, int w) const noexcept {
        return (val[p]>>w)&1;
    }
    // □φ (box): φ true at all accessible worlds
    [[nodiscard]] constexpr bool box(u64 phi_worlds, int w) const noexcept {
        // all accessible from w must be in phi_worlds
        return (frame.R[w]&phi_worlds)==frame.R[w];
    }
    // ◇φ (diamond): φ true at some accessible world
    [[nodiscard]] constexpr bool diamond(u64 phi_worlds, int w) const noexcept {
        return (frame.R[w]&phi_worlds)!=0;
    }
    // Evaluate □φ as bitmask over all worlds
    [[nodiscard]] constexpr u64 box_worlds(u64 phi_worlds, int n=N) const noexcept {
        u64 result=0;
        for(int w=0;w<n;++w) if(box(phi_worlds,w)) result|=(1ULL<<w);
        return result;
    }
    [[nodiscard]] constexpr u64 diamond_worlds(u64 phi_worlds, int n=N) const noexcept {
        u64 result=0;
        for(int w=0;w<n;++w) if(diamond(phi_worlds,w)) result|=(1ULL<<w);
        return result;
    }
};

// ─── Linear Temporal Logic (LTL) operators ────────────────────────────────────
// Over a sequence of world-states (discrete time)
template<int T, int NPROPS>
struct LTLTrace {
    std::array<std::array<bool,NPROPS>,T> states{};

    [[nodiscard]] constexpr bool prop(int t, int p) const noexcept {
        return t>=0&&t<T?states[t][p]:false;
    }
    // X φ (next): φ at t+1
    [[nodiscard]] constexpr bool X(int t, auto phi) const noexcept {
        return t+1<T&&phi(t+1);
    }
    // G φ (globally): φ at all t'≥t
    [[nodiscard]] constexpr bool G(int t, auto phi) const noexcept {
        for(int s=t;s<T;++s) if(!phi(s)) return false; return true;
    }
    // F φ (finally): φ at some t'≥t
    [[nodiscard]] constexpr bool F(int t, auto phi) const noexcept {
        for(int s=t;s<T;++s) if(phi(s)) return true; return false;
    }
    // φ U ψ (until): φ holds until ψ becomes true
    [[nodiscard]] constexpr bool U(int t, auto phi, auto psi) const noexcept {
        for(int s=t;s<T;++s){if(psi(s))return true;if(!phi(s))return false;}
        return false;
    }
    // φ W ψ (weak until): like U but ψ need not happen
    [[nodiscard]] constexpr bool W(int t, auto phi, auto psi) const noexcept {
        for(int s=t;s<T;++s){if(psi(s))return true;if(!phi(s))return false;}
        return true;
    }
    // φ R ψ (release): ψ until (φ and ψ), or ψ forever
    [[nodiscard]] constexpr bool R(int t, auto phi, auto psi) const noexcept {
        for(int s=t;s<T;++s){if(!psi(s))return false;if(phi(s))return true;}
        return true;
    }
};

// ─── Propositional dynamic logic (PDL) programs ───────────────────────────────
enum class PDLOp { Sequence, Choice, Star, Test };
struct PDLProgram { PDLOp op; int left{-1},right{-1}; int prop{-1}; };

// Compute reachability for program α from world w (bitmask of reachable worlds)
template<int N, int NPROPS>
[[nodiscard]] constexpr u64 pdl_reach(
    const KripkeModel<N,NPROPS>& M,
    const PDLProgram* progs, int prog_idx, int w, int n=N) noexcept
{
    const PDLProgram& p=progs[prog_idx];
    switch(p.op) {
        case PDLOp::Sequence: {
            u64 mid=pdl_reach(M,progs,p.left,w,n);
            u64 result=0;
            for(int v=0;v<n;++v) if((mid>>v)&1)
                result|=pdl_reach(M,progs,p.right,v,n);
            return result;
        }
        case PDLOp::Choice:
            return pdl_reach(M,progs,p.left,w,n)|pdl_reach(M,progs,p.right,w,n);
        case PDLOp::Star: {
            u64 reach=(1ULL<<w),prev=0;
            while(reach!=prev){prev=reach;
                for(int v=0;v<n;++v) if((reach>>v)&1) reach|=pdl_reach(M,progs,p.left,v,n);}
            return reach;
        }
        case PDLOp::Test:
            return M.prop_true(p.prop,w)?(1ULL<<w):0;
    }
    return 0;
}

// ─── Doxastic logic (belief) and epistemic logic (knowledge) ──────────────────
// K_i φ (agent i knows φ): φ is true in all worlds indistinguishable from current
template<int N, int N_AGENTS>
struct EpistemicModel {
    std::array<KripkeFrame<N>,N_AGENTS> knowledge_frames{}; // S5 frames per agent
    u64 actual_world{0}; // index of actual world

    // Agent i knows φ at world w
    [[nodiscard]] constexpr bool knows(int agent, u64 phi_worlds, int w) const noexcept {
        return knowledge_frames[agent].box(phi_worlds,w);
    }
    // Common knowledge: CK_G φ = greatest fixed point of E_G
    [[nodiscard]] constexpr u64 common_knowledge(
        const std::array<bool,N_AGENTS>& group, u64 phi_worlds, int n=N) const noexcept
    {
        u64 ck=phi_worlds;
        for(int iter=0;iter<n;++iter) {
            u64 prev=ck;
            for(int a=0;a<N_AGENTS;++a) if(group[a])
                ck&=knowledge_frames[a].box_worlds(ck,n);
            if(ck==prev) break;
        }
        return ck;
    }
};
} // namespace qs::math
