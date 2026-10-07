#pragma once
// qs_math/physics/quantum_math.hpp — Quantum operators, wavefunctions, Dirac notation, Pauli
#include "../arithmetic/scalar_types.hpp"
#include "../signal/fft.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Complex matrix operations ────────────────────────────────────────────────
template<int N>
using CxMat = std::array<std::array<Cx,N>,N>;
template<int N>
using CxVec = std::array<Cx,N>;

template<int N>
[[nodiscard]] constexpr CxMat<N> mat_mul_cx(const CxMat<N>& A, const CxMat<N>& B) noexcept {
    CxMat<N> C{};
    for(int i=0;i<N;++i) for(int k=0;k<N;++k) if(A[i][k].mag2()>1e-30)
        for(int j=0;j<N;++j) C[i][j]=C[i][j]+A[i][k]*B[k][j];
    return C;
}
template<int N>
[[nodiscard]] constexpr CxVec<N> mat_vec_cx(const CxMat<N>& A, const CxVec<N>& v) noexcept {
    CxVec<N> w{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) w[i]=w[i]+A[i][j]*v[j];
    return w;
}
template<int N>
[[nodiscard]] constexpr CxMat<N> mat_dagger(const CxMat<N>& A) noexcept {
    CxMat<N> Ad{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) Ad[j][i]=A[i][j].conj();
    return Ad;
}
template<int N>
[[nodiscard]] constexpr bool is_unitary(const CxMat<N>& U, f64 tol=1e-9) noexcept {
    auto UdU=mat_mul_cx(mat_dagger(U),U);
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
        f64 expect=(i==j)?1.0:0.0;
        if(std::abs(UdU[i][j].re-expect)>tol||std::abs(UdU[i][j].im)>tol) return false;
    }
    return true;
}
template<int N>
[[nodiscard]] constexpr bool is_hermitian(const CxMat<N>& H, f64 tol=1e-9) noexcept {
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) {
        if(std::abs(H[i][j].re-H[j][i].re)>tol||std::abs(H[i][j].im+H[j][i].im)>tol) return false;
    }
    return true;
}

// ─── Pauli matrices ───────────────────────────────────────────────────────────
inline constexpr CxMat<2> k_pauli_x={{{{{0,0},{1,0}}},{{{1,0},{0,0}}}}};
inline constexpr CxMat<2> k_pauli_y={{{{{0,0},{0,-1}}},{{{0,1},{0,0}}}}};
inline constexpr CxMat<2> k_pauli_z={{{{{1,0},{0,0}}},{{{0,0},{-1,0}}}}};
inline constexpr CxMat<2> k_identity2={{{{{1,0},{0,0}}},{{{0,0},{1,0}}}}};
inline constexpr CxMat<2> k_hadamard={{{{{0.7071067811865476,0},{0.7071067811865476,0}}},
                                        {{{0.7071067811865476,0},{-0.7071067811865476,0}}}}};

// ─── Qubit state |ψ⟩ = α|0⟩ + β|1⟩ ─────────────────────────────────────────
struct Qubit { Cx alpha{1,0},beta{0,0};
    [[nodiscard]] constexpr f64 prob0() const noexcept { return alpha.mag2(); }
    [[nodiscard]] constexpr f64 prob1() const noexcept { return beta.mag2(); }
    [[nodiscard]] constexpr bool is_normalized(f64 tol=1e-9) const noexcept {
        return std::abs(alpha.mag2()+beta.mag2()-1)<tol;
    }
    [[nodiscard]] constexpr Qubit apply(const CxMat<2>& U) const noexcept {
        return{U[0][0]*alpha+U[0][1]*beta, U[1][0]*alpha+U[1][1]*beta};
    }
    // Bloch sphere angles: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
    [[nodiscard]] f64 bloch_theta() const noexcept { return 2*std::acos(std::min(1.0,alpha.mag())); }
    [[nodiscard]] f64 bloch_phi()   const noexcept { return std::atan2(beta.im,beta.re)-std::atan2(alpha.im,alpha.re); }
};

// ─── Quantum gates ────────────────────────────────────────────────────────────
// T gate: diag(1, e^{iπ/4})
inline constexpr CxMat<2> k_T_gate={{{{{1,0},{0,0}}},{{{0,0},{0.7071067811865476,0.7071067811865476}}}}};
// S gate: diag(1, i)
inline constexpr CxMat<2> k_S_gate={{{{{1,0},{0,0}}},{{{0,0},{0,1}}}}};
// CNOT (4x4): |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
inline constexpr CxMat<4> k_CNOT={{{{{1,0},{0,0},{0,0},{0,0}}},
                                     {{{0,0},{1,0},{0,0},{0,0}}},
                                     {{{0,0},{0,0},{0,0},{1,0}}},
                                     {{{0,0},{0,0},{1,0},{0,0}}}}};

// Rotation gates
[[nodiscard]] constexpr CxMat<2> Rx(f64 theta) noexcept {
    f64 c=std::cos(theta/2),s=std::sin(theta/2);
    return{{{{{c,0},{0,-s}}},{{{0,-s},{c,0}}}}};
}
[[nodiscard]] constexpr CxMat<2> Ry(f64 theta) noexcept {
    f64 c=std::cos(theta/2),s=std::sin(theta/2);
    return{{{{{c,0},{-s,0}}},{{{s,0},{c,0}}}}};
}
[[nodiscard]] constexpr CxMat<2> Rz(f64 theta) noexcept {
    return{{{{{std::cos(theta/2),-std::sin(theta/2)},{0,0}}},
            {{{0,0},{std::cos(theta/2),std::sin(theta/2)}}}}};
}

// ─── Density matrix ───────────────────────────────────────────────────────────
template<int N>
[[nodiscard]] constexpr CxMat<N> density_matrix(const CxVec<N>& psi) noexcept {
    CxMat<N> rho{};
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) rho[i][j]=psi[i]*psi[j].conj();
    return rho;
}
template<int N>
[[nodiscard]] constexpr f64 trace_density(const CxMat<N>& rho) noexcept {
    f64 tr=0; for(int i=0;i<N;++i) tr+=rho[i][i].re; return tr;
}
template<int N>
[[nodiscard]] constexpr f64 purity(const CxMat<N>& rho) noexcept {
    // Tr(ρ²)
    auto rho2=mat_mul_cx(rho,rho); return trace_density(rho2);
}
template<int N>
[[nodiscard]] constexpr f64 von_neumann_entropy(const CxMat<N>& rho, f64 eps=1e-12) noexcept {
    // S = -Tr(ρ log ρ) — approximate via diagonal (works for diagonal ρ)
    f64 S=0;
    for(int i=0;i<N;++i) {
        f64 p=rho[i][i].re;
        if(p>eps) S-=p*std::log(p);
    }
    return S;
}

// ─── Wigner function (qubit, SU(2) Wigner) ───────────────────────────────────
// Wigner quasi-probability on Bloch sphere (discrete)
[[nodiscard]] constexpr f64 wigner_qubit(const CxMat<2>& rho, f64 theta, f64 phi) noexcept {
    // W(θ,φ) = Tr(ρ Δ(θ,φ)) where Δ = I/2 + (3/2)(n·σ)
    f64 nx=std::sin(theta)*std::cos(phi);
    f64 ny=std::sin(theta)*std::sin(phi);
    f64 nz=std::cos(theta);
    // n·σ = nz*σz + nx*σx + ny*σy
    f64 w=0.5*(rho[0][0].re+rho[1][1].re)
         +1.5*(nz*(rho[0][0].re-rho[1][1].re)
              +nx*(rho[0][1].re+rho[1][0].re)
              +ny*(rho[1][0].im-rho[0][1].im));
    return w;
}

// ─── Quantum harmonic oscillator ──────────────────────────────────────────────
// Ladder operators a and a† truncated to N levels
template<int N>
[[nodiscard]] constexpr CxMat<N> annihilation_op() noexcept {
    CxMat<N> a{};
    for(int i=0;i<N-1;++i) a[i][i+1]={std::sqrt(f64(i+1)),0};
    return a;
}
template<int N>
[[nodiscard]] constexpr CxMat<N> creation_op() noexcept {
    return mat_dagger(annihilation_op<N>());
}
template<int N>
[[nodiscard]] constexpr CxMat<N> number_op() noexcept {
    CxMat<N> n{};
    for(int i=0;i<N;++i) n[i][i]={f64(i),0};
    return n;
}
// Fock state |n⟩
template<int N>
[[nodiscard]] constexpr CxVec<N> fock_state(int n) noexcept {
    CxVec<N> psi{};
    if(n<N) psi[n]={1,0};
    return psi;
}
// Coherent state |α⟩ = exp(-|α|²/2) Σ αⁿ/√(n!) |n⟩
template<int N>
[[nodiscard]] constexpr CxVec<N> coherent_state(Cx alpha) noexcept {
    CxVec<N> psi{};
    f64 norm=std::exp(-0.5*alpha.mag2());
    Cx an{1,0}; f64 fact=1;
    for(int n=0;n<N;++n) {
        if(n>0){an=an*alpha;fact*=n;}
        psi[n]=an*(norm/std::sqrt(fact));
    }
    return psi;
}
} // namespace qs::math
