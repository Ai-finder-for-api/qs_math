#pragma once
// qs_math/markov_math.hpp — Transition matrices, steady state, absorbing chains, HMM (C++20)
#include "scalar_types.hpp"
#include "linear_algebra.hpp"
#include "eigensolvers.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <optional>
#include <numeric>

namespace qs {

// ─── Markov Chain ─────────────────────────────────────────────────────────────
struct MarkovChain {
    DenseMatrix<f64> P; // row-stochastic transition matrix: P[i][j] = p(j|i)
    u32 n{0};

    explicit MarkovChain(u32 states) : P(states,states,0), n(states) {}
    explicit MarkovChain(DenseMatrix<f64> tp) : P(std::move(tp)), n(tp.rows) {}

    // Validate: rows sum to 1
    [[nodiscard]] bool is_valid(f64 tol=1e-8) const {
        for(u32 i=0;i<n;++i){ f64 s=0; for(u32 j=0;j<n;++j) s+=P.at(i,j); if(std::abs(s-1)>tol) return false; }
        return true;
    }

    // n-step transition matrix P^n via repeated squaring
    [[nodiscard]] DenseMatrix<f64> power(u32 steps) const {
        DenseMatrix<f64> R=DenseMatrix<f64>::identity(n), base=P;
        while(steps>0){ if(steps&1) R=R*base; base=base*base; steps>>=1; }
        return R;
    }

    // State distribution after k steps from initial dist π_0
    [[nodiscard]] std::vector<f64> evolve(std::vector<f64> pi, u32 steps) const {
        for(u32 s=0;s<steps;++s){
            std::vector<f64> pi_new(n,0);
            for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) pi_new[j]+=pi[i]*P.at(i,j);
            pi=pi_new;
        }
        return pi;
    }

    // Steady-state distribution: π P = π, |π|=1
    // Solve (P^T - I)π = 0 with Σπ_i=1
    [[nodiscard]] std::vector<f64> steady_state() const {
        DenseMatrix<f64> A(n,n);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) A.at(j,i)=P.at(i,j)-f64(i==j);
        // Replace last row with normalisation constraint
        for(u32 j=0;j<n;++j) A.at(n-1,j)=1;
        std::vector<f64> b(n,0); b[n-1]=1;
        auto sol=solve(A,b);
        if(!sol){ std::vector<f64> pi(n,1.0/n); return pi; }
        return *sol;
    }

    // Mixing time: smallest t such that ||P^t - π1^T||_∞ < ε
    [[nodiscard]] u32 mixing_time(f64 eps=0.01) const {
        auto pi=steady_state();
        for(u32 t=1;t<=1000;++t) {
            auto Pt=power(t);
            f64 maxdiff=0;
            for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
                maxdiff=std::max(maxdiff,std::abs(Pt.at(i,j)-pi[j]));
            if(maxdiff<eps) return t;
        }
        return 1000;
    }

    // Mean first passage time m[i][j]: E[T_j | start at i]
    [[nodiscard]] DenseMatrix<f64> mean_first_passage_times() const {
        auto pi=steady_state();
        DenseMatrix<f64> Z(n,n); // fundamental matrix
        // Z = (I - P + 1π^T)^{-1}  (Kemeny-Snell)
        DenseMatrix<f64> A(n,n);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j) A.at(i,j)=f64(i==j)-P.at(i,j)+pi[j];
        auto Zinv=inverse(A);
        if(!Zinv) return DenseMatrix<f64>(n,n);
        Z=*Zinv;
        DenseMatrix<f64> M(n,n);
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
            M.at(i,j)=(Z.at(j,j)-Z.at(i,j))/(pi[j]+1e-30);
        return M;
    }

    // Stationary distribution entropy rate H = -Σ π_i Σ_j P_ij log P_ij
    [[nodiscard]] f64 entropy_rate() const {
        auto pi=steady_state(); f64 H=0;
        for(u32 i=0;i<n;++i) for(u32 j=0;j<n;++j)
            if(P.at(i,j)>1e-15) H-=pi[i]*P.at(i,j)*std::log(P.at(i,j));
        return H;
    }

    // Spectral gap = 1 - |λ_2| (second-largest eigenvalue magnitude)
    [[nodiscard]] f64 spectral_gap() const {
        auto eig=qr_eigen(P);
        std::vector<f64> evs=eig.eigenvalues;
        std::sort(evs.begin(),evs.end(),[](f64 a,f64 b){return std::abs(a)>std::abs(b);});
        return evs.size()>1?1-std::abs(evs[1]):1;
    }
};

// ─── Absorbing Markov chains ───────────────────────────────────────────────────
struct AbsorbingChain {
    MarkovChain chain;
    std::vector<bool> absorbing; // which states are absorbing

    // Canonical form: Q (transient→transient), R (transient→absorbing)
    struct CanonicalForm { DenseMatrix<f64> Q, R; std::vector<u32> transient, absorb; };

    [[nodiscard]] CanonicalForm canonical() const {
        CanonicalForm cf;
        for(u32 i=0;i<chain.n;++i) {
            if(absorbing[i]) cf.absorb.push_back(i);
            else             cf.transient.push_back(i);
        }
        u32 t=u32(cf.transient.size()), a=u32(cf.absorb.size());
        cf.Q=DenseMatrix<f64>(t,t); cf.R=DenseMatrix<f64>(t,a);
        for(u32 i=0;i<t;++i) {
            for(u32 j=0;j<t;++j) cf.Q.at(i,j)=chain.P.at(cf.transient[i],cf.transient[j]);
            for(u32 j=0;j<a;++j) cf.R.at(i,j)=chain.P.at(cf.transient[i],cf.absorb[j]);
        }
        return cf;
    }

    // Fundamental matrix N = (I-Q)^{-1}; N[i][j] = expected visits to j before absorption
    [[nodiscard]] DenseMatrix<f64> fundamental_matrix() const {
        auto [Q,R,t_states,a_states]=canonical();
        u32 t=Q.rows;
        DenseMatrix<f64> IQ=DenseMatrix<f64>::identity(t);
        for(u32 i=0;i<t;++i) for(u32 j=0;j<t;++j) IQ.at(i,j)-=Q.at(i,j);
        auto N=inverse(IQ);
        return N?*N:DenseMatrix<f64>(t,t);
    }

    // Absorption probabilities B = N R; B[i][j] = P(absorbed into j | start at i)
    [[nodiscard]] DenseMatrix<f64> absorption_probabilities() const {
        auto [Q,R,ts,as]=canonical();
        return fundamental_matrix()*R;
    }

    // Expected steps to absorption: t = N 1 (row sums of fundamental matrix)
    [[nodiscard]] std::vector<f64> expected_steps_to_absorption() const {
        auto N=fundamental_matrix();
        std::vector<f64> t(N.rows,0);
        for(u32 i=0;i<N.rows;++i) for(u32 j=0;j<N.cols;++j) t[i]+=N.at(i,j);
        return t;
    }
};

// ─── Hidden Markov Model ───────────────────────────────────────────────────────
struct HMM {
    u32 K{2}; // number of hidden states
    u32 M{3}; // number of observation symbols
    std::vector<f64> pi;              // initial distribution π_i
    DenseMatrix<f64> A;               // transition A[i][j] = p(z_t=j|z_{t-1}=i)
    DenseMatrix<f64> B;               // emission B[i][k] = p(x_t=k|z_t=i)

    HMM(u32 K,u32 M) : K(K),M(M),pi(K,1.0/K),A(K,K,1.0/K),B(K,M,1.0/M) {}

    // Forward algorithm: α[t][i] = p(x_1..x_t, z_t=i)
    [[nodiscard]] std::vector<std::vector<f64>> forward(const std::vector<u32>& obs) const {
        u32 T=u32(obs.size());
        std::vector<std::vector<f64>> alpha(T,std::vector<f64>(K,0));
        for(u32 i=0;i<K;++i) alpha[0][i]=pi[i]*B.at(i,obs[0]);
        for(u32 t=1;t<T;++t) for(u32 j=0;j<K;++j) {
            for(u32 i=0;i<K;++i) alpha[t][j]+=alpha[t-1][i]*A.at(i,j);
            alpha[t][j]*=B.at(j,obs[t]);
        }
        return alpha;
    }

    // Backward algorithm: β[t][i] = p(x_{t+1}..x_T | z_t=i)
    [[nodiscard]] std::vector<std::vector<f64>> backward(const std::vector<u32>& obs) const {
        u32 T=u32(obs.size());
        std::vector<std::vector<f64>> beta(T,std::vector<f64>(K,0));
        for(u32 i=0;i<K;++i) beta[T-1][i]=1;
        for(i32 t=i32(T)-2;t>=0;--t) for(u32 i=0;i<K;++i)
            for(u32 j=0;j<K;++j) beta[u32(t)][i]+=A.at(i,j)*B.at(j,obs[u32(t)+1])*beta[u32(t)+1][j];
        return beta;
    }

    // Log-likelihood: log p(obs | model)
    [[nodiscard]] f64 log_likelihood(const std::vector<u32>& obs) const {
        auto alpha=forward(obs);
        f64 L=0; for(u32 i=0;i<K;++i) L+=alpha.back()[i];
        return std::log(L+1e-300);
    }

    // Viterbi: most likely state sequence
    [[nodiscard]] std::vector<u32> viterbi(const std::vector<u32>& obs) const {
        u32 T=u32(obs.size());
        DenseMatrix<f64> delta(T,K); DenseMatrix<f64> psi(T,K);
        for(u32 i=0;i<K;++i){ delta.at(0,i)=std::log(pi[i]+1e-300)+std::log(B.at(i,obs[0])+1e-300); psi.at(0,i)=0; }
        for(u32 t=1;t<T;++t) for(u32 j=0;j<K;++j) {
            f64 best=-1e300; u32 best_i=0;
            for(u32 i=0;i<K;++i){ f64 v=delta.at(t-1,i)+std::log(A.at(i,j)+1e-300); if(v>best){best=v;best_i=i;} }
            delta.at(t,j)=best+std::log(B.at(j,obs[t])+1e-300);
            psi.at(t,j)=f64(best_i);
        }
        // Backtrack
        std::vector<u32> seq(T);
        f64 best=-1e300;
        for(u32 i=0;i<K;++i) if(delta.at(T-1,i)>best){best=delta.at(T-1,i);seq[T-1]=i;}
        for(i32 t=i32(T)-2;t>=0;--t) seq[u32(t)]=u32(psi.at(u32(t)+1,seq[u32(t)+1]));
        return seq;
    }

    // Baum-Welch EM training (one iteration)
    void baum_welch_step(const std::vector<u32>& obs) {
        u32 T=u32(obs.size());
        auto alpha=forward(obs);
        auto beta=backward(obs);
        f64 LL=0; for(u32 i=0;i<K;++i) LL+=alpha.back()[i];
        if(LL<1e-300) return;
        // γ[t][i] = p(z_t=i | obs, model)
        std::vector<std::vector<f64>> gamma(T,std::vector<f64>(K,0));
        for(u32 t=0;t<T;++t){ for(u32 i=0;i<K;++i) gamma[t][i]=alpha[t][i]*beta[t][i]/LL; }
        // ξ[t][i][j] = p(z_t=i,z_{t+1}=j | obs, model)
        // Update pi
        pi=gamma[0];
        // Update A
        for(u32 i=0;i<K;++i) for(u32 j=0;j<K;++j) {
            f64 num=0,denom=0;
            for(u32 t=0;t<T-1;++t){
                f64 xi=alpha[t][i]*A.at(i,j)*B.at(j,obs[t+1])*beta[t+1][j]/LL;
                num+=xi; denom+=gamma[t][i];
            }
            A.at(i,j)=denom>1e-30?num/denom:A.at(i,j);
        }
        // Normalise A rows
        for(u32 i=0;i<K;++i){f64 s=0;for(u32 j=0;j<K;++j)s+=A.at(i,j);if(s>0)for(u32 j=0;j<K;++j)A.at(i,j)/=s;}
        // Update B
        for(u32 i=0;i<K;++i) for(u32 k=0;k<M;++k) {
            f64 num=0,denom=0;
            for(u32 t=0;t<T;++t){ if(obs[t]==k) num+=gamma[t][i]; denom+=gamma[t][i]; }
            B.at(i,k)=denom>1e-30?num/denom:B.at(i,k);
        }
        // Normalise B rows
        for(u32 i=0;i<K;++i){f64 s=0;for(u32 k=0;k<M;++k)s+=B.at(i,k);if(s>0)for(u32 k=0;k<M;++k)B.at(i,k)/=s;}
    }
};

} // namespace qs
