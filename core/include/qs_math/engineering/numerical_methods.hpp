#pragma once
// qs_math/engineering/numerical_methods.hpp — FEM basis, shape functions, integration rules
#include "../arithmetic/scalar_types.hpp"
#include <array>
#include <cmath>

namespace qs::math {

// ─── 1D FEM shape functions (Lagrange) ────────────────────────────────────────
// Linear element (2-node): N1(ξ)=(1-ξ)/2, N2(ξ)=(1+ξ)/2 on [-1,1]
[[nodiscard]] constexpr std::array<f64,2> shape_linear1d(f64 xi) noexcept {
    return{(1-xi)*0.5,(1+xi)*0.5};
}
[[nodiscard]] constexpr std::array<f64,2> dshape_linear1d(f64) noexcept {
    return{-0.5,0.5};
}
// Quadratic element (3-node): N1=ξ(ξ-1)/2, N2=1-ξ², N3=ξ(ξ+1)/2
[[nodiscard]] constexpr std::array<f64,3> shape_quad1d(f64 xi) noexcept {
    return{xi*(xi-1)*0.5,1-xi*xi,xi*(xi+1)*0.5};
}
[[nodiscard]] constexpr std::array<f64,3> dshape_quad1d(f64 xi) noexcept {
    return{xi-0.5,-2*xi,xi+0.5};
}
// Cubic Hermite (4-DOF): position and derivative at endpoints
[[nodiscard]] constexpr std::array<f64,4> shape_hermite1d(f64 xi) noexcept {
    f64 xi2=xi*xi,xi3=xi2*xi;
    return{1-3*xi2+2*xi3, xi-2*xi2+xi3, 3*xi2-2*xi3, -xi2+xi3};
}

// ─── 2D triangular element shape functions ────────────────────────────────────
// Linear (T3): N_i = barycentric coordinates L_i
[[nodiscard]] constexpr std::array<f64,3> shape_T3(f64 L1, f64 L2) noexcept {
    return{L1,L2,1-L1-L2};
}
// Quadratic (T6): N = [L1(2L1-1), L2(2L2-1), L3(2L3-1), 4L1L2, 4L2L3, 4L1L3]
[[nodiscard]] constexpr std::array<f64,6> shape_T6(f64 L1, f64 L2) noexcept {
    f64 L3=1-L1-L2;
    return{L1*(2*L1-1),L2*(2*L2-1),L3*(2*L3-1),4*L1*L2,4*L2*L3,4*L1*L3};
}

// ─── 2D quadrilateral element ─────────────────────────────────────────────────
// Bilinear Q4: N_i = (1±ξ)(1±η)/4
[[nodiscard]] constexpr std::array<f64,4> shape_Q4(f64 xi, f64 eta) noexcept {
    return{(1-xi)*(1-eta)*0.25,(1+xi)*(1-eta)*0.25,
           (1+xi)*(1+eta)*0.25,(1-xi)*(1+eta)*0.25};
}
[[nodiscard]] constexpr std::array<std::array<f64,4>,2> dshape_Q4(f64 xi, f64 eta) noexcept {
    return{{{-(1-eta)*0.25,(1-eta)*0.25,(1+eta)*0.25,-(1+eta)*0.25}},
           {{-(1-xi)*0.25,-(1+xi)*0.25,(1+xi)*0.25,(1-xi)*0.25}}};
}
// Serendipity Q8 (8-node): add midside nodes
[[nodiscard]] constexpr std::array<f64,8> shape_Q8(f64 xi, f64 eta) noexcept {
    f64 xi2=xi*xi,eta2=eta*eta;
    return{(1-xi)*(1-eta)*(-xi-eta-1)*0.25,(1+xi)*(1-eta)*(xi-eta-1)*0.25,
           (1+xi)*(1+eta)*(xi+eta-1)*0.25,(1-xi)*(1+eta)*(-xi+eta-1)*0.25,
           (1-xi2)*(1-eta)*0.5,(1+xi)*(1-eta2)*0.5,
           (1-xi2)*(1+eta)*0.5,(1-xi)*(1-eta2)*0.5};
}

// ─── 3D tetrahedral element ───────────────────────────────────────────────────
// Linear Tet4: N_i = L_i (barycentric)
[[nodiscard]] constexpr std::array<f64,4> shape_Tet4(f64 L1,f64 L2,f64 L3) noexcept {
    return{L1,L2,L3,1-L1-L2-L3};
}
// Jacobian of isoparametric map (2D triangle)
[[nodiscard]] constexpr f64 triangle_jacobian(
    f64 x1,f64 y1,f64 x2,f64 y2,f64 x3,f64 y3) noexcept
{
    return (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
}

// ─── Gauss quadrature rules (for reference elements) ──────────────────────────
// 1D: 3-point Gauss-Legendre on [-1,1]
struct GQ1D { std::array<f64,5> pts; std::array<f64,5> wts; int n; };
[[nodiscard]] constexpr GQ1D gauss_1d_5pt() noexcept {
    return{{-0.9061798459,-0.5384693101,0,0.5384693101,0.9061798459},
           {0.2369268851,0.4786286705,0.5688888889,0.4786286705,0.2369268851},5};
}
// 2D triangle: 7-point rule (degree 5)
struct GQTri { std::array<f64,7> L1,L2,wt; int n; };
[[nodiscard]] constexpr GQTri gauss_tri_7pt() noexcept {
    return{{0.1012865073,0.7974269853,0.1012865073,0.4701420641,0.4701420641,0.0597158717,0.3333333333},
           {0.1012865073,0.1012865073,0.7974269853,0.4701420641,0.0597158717,0.4701420641,0.3333333333},
           {0.1259391805,0.1259391805,0.1259391805,0.1323941527,0.1323941527,0.1323941527,0.2250000000},7};
}

// ─── Element stiffness matrix (1D Poisson: -u''=f) ────────────────────────────
// K_ij = ∫ N'_i * N'_j dx (on element of length h)
[[nodiscard]] constexpr std::array<std::array<f64,2>,2> stiffness_1d_linear(f64 h) noexcept {
    return{{{{1/h,-1/h}},{{-1/h,1/h}}}};
}
[[nodiscard]] constexpr std::array<std::array<f64,2>,2> mass_1d_linear(f64 h) noexcept {
    return{{{{h/3,h/6}},{{h/6,h/3}}}};
}

// ─── Assembly (2-element example) ────────────────────────────────────────────
template<int NEL, int NDOF>
constexpr void assemble_1d(
    std::array<std::array<f64,NDOF>,NDOF>& K_global,
    const std::array<std::array<f64,2>,2>& K_el,
    int i, int j) noexcept
{
    K_global[i][i]+=K_el[0][0]; K_global[i][j]+=K_el[0][1];
    K_global[j][i]+=K_el[1][0]; K_global[j][j]+=K_el[1][1];
}

// ─── Boundary conditions (essential/Dirichlet) ────────────────────────────────
template<int N>
constexpr void apply_dirichlet(
    std::array<std::array<f64,N>,N>& K,
    std::array<f64,N>& f,
    int dof, f64 value) noexcept
{
    for(int j=0;j<N;++j){f[j]-=K[j][dof]*value;K[j][dof]=0;K[dof][j]=0;}
    K[dof][dof]=1; f[dof]=value;
}

// ─── Isoparametric map Jacobian (2D Q4) ──────────────────────────────────────
// Given element coordinates [(x1,y1),...,(x4,y4)] and (ξ,η), compute J
[[nodiscard]] constexpr std::array<std::array<f64,2>,2> iso_jacobian_Q4(
    const std::array<std::array<f64,2>,4>& coords, f64 xi, f64 eta) noexcept
{
    auto dN=dshape_Q4(xi,eta);
    std::array<std::array<f64,2>,2> J{};
    for(int a=0;a<4;++a) {
        J[0][0]+=dN[0][a]*coords[a][0]; J[0][1]+=dN[0][a]*coords[a][1];
        J[1][0]+=dN[1][a]*coords[a][0]; J[1][1]+=dN[1][a]*coords[a][1];
    }
    return J;
}
[[nodiscard]] constexpr f64 det2x2(const std::array<std::array<f64,2>,2>& J) noexcept {
    return J[0][0]*J[1][1]-J[0][1]*J[1][0];
}
} // namespace qs::math
