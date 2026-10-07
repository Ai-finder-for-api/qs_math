#pragma once
// qs_math/algebra/representation_theory.hpp — Group reps, characters, Schur lemma, decomposition
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── Character table (finite group, N irreps) ─────────────────────────────────
template<int N_CLASSES, int N_IRREPS>
struct CharacterTable {
    std::array<std::array<f64,N_CLASSES>,N_IRREPS> chi{};
    std::array<i64,N_CLASSES> class_sizes{};
    std::array<const char*,N_IRREPS> irrep_labels{};
    i64 group_order{0};

    // Inner product of characters: <χ_i,χ_j> = (1/|G|) Σ_C |C| χ_i(C)*χ_j(C)
    [[nodiscard]] constexpr f64 inner_product(int i, int j) const noexcept {
        f64 s=0;
        for(int c=0;c<N_CLASSES;++c) s+=class_sizes[c]*chi[i][c]*chi[j][c];
        return s/(group_order+1e-15);
    }
    // Verify orthogonality: <χ_i,χ_j> = δ_{ij}
    [[nodiscard]] constexpr bool check_orthogonality(f64 tol=1e-9) const noexcept {
        for(int i=0;i<N_IRREPS;++i) for(int j=0;j<N_IRREPS;++j) {
            f64 ip=inner_product(i,j);
            f64 expected=(i==j)?1.0:0.0;
            if(std::abs(ip-expected)>tol) return false;
        }
        return true;
    }
    // Burnside's lemma: |G| = Σ dim(V_i)²
    [[nodiscard]] constexpr bool check_burnside() const noexcept {
        i64 sum=0;
        for(int i=0;i<N_IRREPS;++i){i64 d=i64(chi[i][0]+0.5);sum+=d*d;}
        return sum==group_order;
    }
    // Decompose representation with character chi_rep
    [[nodiscard]] constexpr std::array<f64,N_IRREPS> decompose(
        const std::array<f64,N_CLASSES>& chi_rep) const noexcept
    {
        std::array<f64,N_IRREPS> mult{};
        for(int i=0;i<N_IRREPS;++i) {
            f64 s=0;
            for(int c=0;c<N_CLASSES;++c) s+=class_sizes[c]*chi[i][c]*chi_rep[c];
            mult[i]=s/group_order;
        }
        return mult;
    }
    // Regular representation character: χ_reg(e)=|G|, χ_reg(g)=0 for g≠e
    [[nodiscard]] constexpr std::array<f64,N_CLASSES> regular_character() const noexcept {
        std::array<f64,N_CLASSES> chi_reg{};
        chi_reg[0]=f64(group_order); // identity class
        return chi_reg;
    }
};

// ─── Symmetric group S_N character table (Young tableaux) ─────────────────────
// Character of S_3 (3 conjugacy classes: {e},{(12)},{(123)})
[[nodiscard]] constexpr CharacterTable<3,3> S3_character_table() noexcept {
    CharacterTable<3,3> ct;
    ct.group_order=6;
    ct.class_sizes={1,3,2};
    // Trivial: χ=(1,1,1), Sign: χ=(1,-1,1), Standard: χ=(2,0,-1)
    ct.chi[0]={1,1,1};
    ct.chi[1]={1,-1,1};
    ct.chi[2]={2,0,-1};
    return ct;
}

// Character of S_4 (5 conjugacy classes)
[[nodiscard]] constexpr CharacterTable<5,5> S4_character_table() noexcept {
    CharacterTable<5,5> ct;
    ct.group_order=24;
    ct.class_sizes={1,6,3,8,6};
    ct.chi[0]={1,1,1,1,1};      // trivial
    ct.chi[1]={1,-1,1,1,-1};    // sign
    ct.chi[2]={2,0,2,-1,0};     // standard 2D
    ct.chi[3]={3,1,-1,0,-1};    // standard 3D
    ct.chi[4]={3,-1,-1,0,1};    // standard 3D (sign twist)
    return ct;
}

// ─── Schur's lemma ─────────────────────────────────────────────────────────────
// If ρ:G→GL(V) irreducible and T:V→V commutes with ρ(g) for all g, then T=λI
// Verify: T*rho(g) = rho(g)*T for all group elements
template<int DIM, int N_GROUP>
[[nodiscard]] constexpr bool schur_commutes(
    const std::array<std::array<std::array<f64,DIM>,DIM>,N_GROUP>& rho,
    const std::array<std::array<f64,DIM>,DIM>& T, f64 tol=1e-9) noexcept
{
    for(int g=0;g<N_GROUP;++g) {
        for(int i=0;i<DIM;++i) for(int j=0;j<DIM;++j) {
            f64 Trho=0,rhoT=0;
            for(int k=0;k<DIM;++k) {Trho+=T[i][k]*rho[g][k][j];rhoT+=rho[g][i][k]*T[k][j];}
            if(std::abs(Trho-rhoT)>tol) return false;
        }
    }
    return true;
}

// ─── Tensor product representation ────────────────────────────────────────────
// (ρ₁⊗ρ₂)(g) = ρ₁(g) ⊗ ρ₂(g)
template<int D1, int D2, int NG>
[[nodiscard]] constexpr std::array<std::array<std::array<f64,D1*D2>,D1*D2>,NG>
tensor_product_rep(
    const std::array<std::array<std::array<f64,D1>,D1>,NG>& rho1,
    const std::array<std::array<std::array<f64,D2>,D2>,NG>& rho2) noexcept
{
    std::array<std::array<std::array<f64,D1*D2>,D1*D2>,NG> result{};
    for(int g=0;g<NG;++g)
        for(int i=0;i<D1;++i) for(int j=0;j<D2;++j)
            for(int k=0;k<D1;++k) for(int l=0;l<D2;++l)
                result[g][i*D2+j][k*D2+l]=rho1[g][i][k]*rho2[g][j][l];
    return result;
}

// Character of tensor product rep
template<int D1, int D2, int NC>
[[nodiscard]] constexpr std::array<f64,NC> tensor_product_chi(
    const std::array<f64,NC>& chi1,
    const std::array<f64,NC>& chi2) noexcept
{
    std::array<f64,NC> chi{};
    for(int c=0;c<NC;++c) chi[c]=chi1[c]*chi2[c];
    return chi;
}

// ─── Induced representation ───────────────────────────────────────────────────
// Ind_H^G ρ has dimension [G:H]*dim(ρ), character: χ_ind(g)=(1/|H|)Σ_{x:xgx⁻¹∈H} ρ(xgx⁻¹)
template<int NG, int NH, int NC>
[[nodiscard]] constexpr std::array<f64,NC> induced_character(
    const std::array<f64,NC>& chi_H,
    const std::array<i64,NC>& class_sizes_G,
    const std::array<i64,NC>& class_sizes_H,
    i64 order_H) noexcept
{
    std::array<f64,NC> chi_ind{};
    for(int c=0;c<NC;++c)
        chi_ind[c]=chi_H[c]*f64(class_sizes_G[c])/f64(order_H);
    return chi_ind;
}

// ─── Clebsch-Gordan coefficients (SU(2)) ─────────────────────────────────────
// |j1,m1⟩⊗|j2,m2⟩ = Σ_{j,m} <j1,m1;j2,m2|j,m> |j,m⟩
// For half-integer j: use Wigner 3j symbols
// Simple case: j1=j2=1/2
[[nodiscard]] constexpr f64 cg_half_half(int m1_2, int m2_2, int J2, int M2) noexcept {
    // j1=j2=1/2, m=±1/2 (stored as ±1 for 2*m)
    // J=0 (singlet) or J=1 (triplet)
    if(m1_2+m2_2!=M2) return 0;
    if(J2==0){
        if(M2!=0) return 0;
        return m1_2==1?1.0/std::sqrt(2.0):-1.0/std::sqrt(2.0);
    }
    if(J2==2){
        if(m1_2==1&&m2_2==1&&M2==2) return 1;
        if(m1_2==-1&&m2_2==-1&&M2==-2) return 1;
        if(M2==0) return 1.0/std::sqrt(2.0);
        return 0;
    }
    return 0;
}

// ─── Wigner D-matrix (rotation matrix in |j,m⟩ basis) ────────────────────────
// D^j_{m'm}(α,β,γ) = e^{-im'α} d^j_{m'm}(β) e^{-imγ}
// Small Wigner d-matrix (β rotation)
[[nodiscard]] constexpr f64 wigner_d_half(f64 beta, int m1_2, int m2_2) noexcept {
    // j=1/2: d^{1/2}_{m'm}(β)
    f64 c=std::cos(beta*0.5),s=std::sin(beta*0.5);
    if(m1_2==1&&m2_2==1) return c;
    if(m1_2==1&&m2_2==-1) return -s;
    if(m1_2==-1&&m2_2==1) return s;
    if(m1_2==-1&&m2_2==-1) return c;
    return 0;
}
[[nodiscard]] constexpr f64 wigner_d_1(f64 beta, int m1, int m2) noexcept {
    // j=1
    f64 c=std::cos(beta),s=std::sin(beta);
    f64 c2=std::cos(beta*0.5),s2=std::sin(beta*0.5);
    if(m1==1&&m2==1) return c2*c2;
    if(m1==1&&m2==0) return -s/std::sqrt(2.0);
    if(m1==1&&m2==-1) return s2*s2;
    if(m1==0&&m2==1) return s/std::sqrt(2.0);
    if(m1==0&&m2==0) return c;
    if(m1==0&&m2==-1) return -s/std::sqrt(2.0);
    if(m1==-1&&m2==1) return s2*s2;
    if(m1==-1&&m2==0) return s/std::sqrt(2.0);
    if(m1==-1&&m2==-1) return c2*c2;
    return 0;
}

// ─── Fourier transform on finite groups (G-FT) ───────────────────────────────
// f̂(ρ) = Σ_{g∈G} f(g) ρ(g) (matrix-valued Fourier coefficient)
template<int NG, int DIM>
[[nodiscard]] constexpr std::array<std::array<f64,DIM>,DIM> group_ft(
    const std::array<f64,NG>& f,
    const std::array<std::array<std::array<f64,DIM>,DIM>,NG>& rho) noexcept
{
    std::array<std::array<f64,DIM>,DIM> hat{};
    for(int g=0;g<NG;++g) if(f[g]!=0)
        for(int i=0;i<DIM;++i) for(int j=0;j<DIM;++j)
            hat[i][j]+=f[g]*rho[g][i][j];
    return hat;
}
} // namespace qs::math
