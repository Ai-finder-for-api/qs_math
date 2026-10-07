# qs_math

**Quantum Suite Math Library** — a comprehensive C++20 mathematics library for
real-time graphics, game development, simulation, and scientific computing.

Part of the **Quantum Suite** (QS) ecosystem. Built and consumed via `qs_build`
(the QS native build system) and also supports standard CMake.

---

## Overview

| Metric | Value |
|--------|-------|
| Total LOC | ~84,000 |
| Source files | 75 `.cpp` + 362 `.hpp` |
| External modules | 27 |
| Core subsystems | 15 |
| C++ standard | C++20 |
| Build systems | `qs_build` (primary), CMake 3.20+ |

---

## Architecture

```
qs_math/
├── build.qs              ← qs_build manifest (level-based, PCH-enabled)
├── CMakeLists.txt        ← Full CMake build (companion)
├── core/                 ← Level-0 foundation library
│   ├── include/
│   │   └── qs_math_pch.hpp   ← Precompiled header
│   └── src/
│       ├── vectors/           vec2, vec3, vec4 (f32/f64/i32)
│       ├── matrices/          mat3, mat4
│       ├── quaternion/        Quat<T>
│       ├── scalar/            scalar types, constants
│       ├── geometry/          ray, plane, frustum primitives
│       ├── color/             RGBA, color ops
│       ├── transform/         TRS, SRT, local/world
│       ├── camera/            perspective, orthographic
│       ├── noise/             Perlin, simplex, FBM
│       ├── hashing/           spatial hash, Morton encoding
│       ├── serialization/     binary + JSON serialisation
│       ├── interpolation/     Hermite, Catmull-Rom, bicubic, Akima
│       ├── numerical/         root finding, quadrature, ODE, linear solvers
│       ├── intersection/      ray–geometry, swept, CCD
│       ├── curves/            Bézier, B-spline, NURBS, Frenet frames
│       ├── easing/            30 Penner easings + spring/elastic/bounce
│       ├── random/            PCG64, Xoshiro256**, Halton, Sobol
│       ├── statistics/        descriptive stats, Welch t-test, OLS, KDE
│       ├── physics_helpers/   inertia tensors, Verlet, XPBD
│       ├── mesh_math/         normals, tangents, Laplacian smoothing
│       ├── simd/              AVX2/SSE4/NEON abstraction layer
│       ├── spatial/           KD-tree, spatial hash, octree
│       ├── bounding/          AABB, sphere, capsule, frustum
│       └── coordinate/        spherical, cylindrical, ECEF, geodetic, UTM
│
└── <module>/             ← Level-1 leaf modules (each depends on core)
    ├── include/qs_math_<module>.hpp
    └── <module>_impl.cpp (+ *_ext_impl.cpp for larger modules)
```

### Dependency graph

```
Level 0:  qs_math_core          (no qs_math dependencies)
          └─ PCH: core/include/qs_math_pch.hpp
          └─ Unity chunks: 50 TUs per chunk

Level 1:  all leaf modules      (each: dependencies = ["qs_math_core"])
          algebra  arithmetic  calculus  collision  color
          editor   game        geometry  graph-lib  interpolation
          linear-algebra  media  noise  numerical  optimization
          physics  probability  procedural  quaternion  random
          rendering  sculpting  spline  statistics  transforms
          utilities  uv

Level 2:  qs_math_tests         (depends on core + all modules)
```

---

## Modules

### Core (`core/`)

The foundation. Always compiled first. Provides all primitive types, the PCH,
and the bulk of everyday math.

| Subsystem | Key types / functions |
|-----------|----------------------|
| `vectors` | `Vec2<T>`, `Vec3<T>`, `Vec4<T>` — dot, cross, length, swizzle |
| `matrices` | `Mat3<T>`, `Mat4<T>` — multiply, transpose, inverse, LU |
| `quaternion` | `Quat<T>` — SLERP, rotate, to_mat4, from_euler |
| `geometry` | `Ray`, `Plane`, `AABB3`, `Frustum` |
| `transform` | `Transform` — TRS decompose/compose, world↔local |
| `camera` | `PerspectiveCamera`, `OrthoCamera`, view/proj matrices |
| `noise` | `perlin_2d/3d`, `simplex_2d/3d`, `fbm`, `ridged_fbm` |
| `curves` | Bézier (quad/cubic), B-spline, NURBS, arc-length param |
| `easing` | 30 Penner easing functions + spring, elastic, bounce, back |
| `interpolation` | Hermite, Catmull-Rom (uniform + centripetal), barycentric, bilinear, bicubic, trilinear, Akima, rational, Shepard, monotone cubic |
| `numerical` | Brent/Halley/Ridder root finding; Gauss-Legendre, adaptive quadrature, Romberg; RK4, RKDP45; Gauss elim, Thomas, CG, LU, QR, power iteration |
| `intersection` | Ray–sphere/AABB/triangle/capsule/cylinder/disk/cone; OBB–OBB SAT; Sutherland-Hodgman clipping; swept-sphere CCD |
| `random` | PCG64, Xoshiro256**, Halton, Sobol, Box-Muller, reservoir sampling |
| `simd` | AVX2/SSE4/NEON: dot4, add8/mul8, mat4_mul, normalize4 |
| `spatial` | KD-tree, `SpatialHash`, Octree (stub) |
| `bounding` | AABB from points/transform, Ritter sphere, capsule distance, frustum planes |
| `coordinate` | Spherical↔Cartesian, cylindrical, ECEF↔geodetic, UTM |
| `statistics` | Mean, variance, Spearman, Welch t-test, OLS, KDE |
| `mesh_math` | Face normals, area, tangent frames (Lengyel), Laplacian smooth, quadric error |
| `physics_helpers` | Inertia tensors, impulse resolution, Verlet, XPBD constraints, drag |

### External Modules

Each module lives in its own directory and is compiled as a separate static
library (Level 1). Link only what you need, or use `qs_math_all` to pull
everything in.

| Module | Description |
|--------|-------------|
| `algebra` | Polynomials, GF(256), octonions, Clifford algebra Cl(3), Lie algebra so(3), NTT, Berlekamp-Massey, Kronecker product, tensor contraction, permutation groups, combinatorics |
| `arithmetic` | Fixed-point Q16.16/Q8.24, saturating arithmetic, bit ops, u128, double-double (DDFloat), CORDIC, modular arithmetic, Miller-Rabin, Pollard ρ |
| `calculus` | Symbolic derivatives (5-pt), adaptive Simpson, partial/Laplacian operators, Fourier coefficients, Euler/midpoint ODE, cotan Laplacian |
| `collision` | Contact manifold, GJK, AABB BVH (top-down), support functions, sphere/box primitives |
| `color` | CIE ΔE 2000, sRGB↔linear, XYZ↔Lab (D65), Oklab, Lab↔LCh, ACES/Reinhard tone curves |
| `editor` | Screen↔world projection, axis-handle hit testing, arcball rotation, AABB screen rect |
| `game` | Tile/grid helpers, isometric projection, Camera2D, Bresenham, BFS flood fill, A\* pathfinding, orbit/FPS cameras, animation blending, FABRIK IK, loot tables, line-of-sight DDA, ballistic angle, damage falloff, hitbox OBB test, XP curves, cooldown, minimap, FSM, autotiling |
| `geometry` | 2D convex hull (Graham), polygon area/centroid, Sutherland-Hodgman clipping, Minkowski sum, 3D convex hull, ear-clipping triangulation, Bowyer-Watson Delaunay, SAT polygon, Voronoi (stub) |
| `graph-lib` | Dijkstra, Bellman-Ford, Floyd-Warshall, Prim/Kruskal MST, Kahn topo sort, Tarjan SCC, BFS/DFS, Hopcroft-Karp bipartite matching |
| `interpolation` | SLERP/NLERP/SQUAD, RBF (Gaussian/MQ/IMQ/ThinPlate), trilinear field, Bézier surface, B-spline surface |
| `linear-algebra` | Dense heap matrix (`DMatrix`), 1-sided Jacobi SVD, Cholesky, Jacobi/Gauss-Seidel, outer product, low-rank approx, 3×3 Cardano eigenvalues |
| `media` | FFT (Cooley-Tukey), RFFT/IRFFT, window functions (Hann/Hamming/Blackman/FlatTop/Kaiser/Triangular), PSD, FFT-based convolution/cross-correlation, biquad filter (LPF/HPF/BPF/peak), sinc resampler, Mel filterbank |
| `noise` | Value noise 2D/3D, Worley 2D/3D F1, domain-warped FBM, curl noise 2D/3D, Mitchell blue noise, 1/f^β spectral synthesis, FBM with analytical derivative |
| `numerical` | Gamma/beta/erf/erfc/digamma, Bessel J0/J1/Jn, Legendre/Chebyshev/Laguerre/Hermite polynomials, regularised incomplete beta/gamma, Riemann zeta, Fibonacci/Catalan; **sparse**: CSR matrix, Arnoldi, GMRES (restarted), Lanczos, ILU0, randomised SVD, NMF, finite-difference matrices, Chebyshev nodes/barycentric interpolation |
| `optimization` | Gradient descent + backtracking, Adam, L-BFGS, Nelder-Mead, Simulated Annealing, Differential Evolution, Golden section, Polak-Ribière CG; **extended**: CMA-ES, PSO, NSGA-II (Pareto front), Trust region (dogleg), Augmented Lagrangian |
| `physics` | `RigidBody` (semi-implicit Euler, impulse), `Projectile`, orbital mechanics, fluid helpers (Re/Ma/Bernoulli/Cd), EM (Lorentz force), optics (Snell/Fresnel), special relativity; **extended**: PBD cloth, SPH fluid (poly6/spiky/viscosity kernels), contact constraint solver (Baumgarte), buoyancy, hinge constraint |
| `probability` | Normal/Student-t/Beta/Gamma/Binomial/Poisson/Exponential/Chi-squared PDF+CDF+inverse CDF, multivariate normal, Dirichlet, Kalman filter, Markov stationary, Shannon entropy, KL divergence, mutual information; **extended**: Metropolis-Hastings MCMC, HMC, mean-field variational inference, discrete factor graph + loopy belief propagation |
| `procedural` | Heightmap (FBM + diamond-square + thermal/hydraulic erosion + normals), L-system iterate + turtle-to-lines, maze (recursive backtracker + BFS solve), Poisson disk sampling, WFC (stub) |
| `quaternion` | Dual quaternion (from_rt, transform, multiply, normalise, to_mat4), weighted average (geodesic mean), rotation-between-vectors, swing-twist decomposition, angular velocity integration, Euler XYZ/ZYX, axis-angle, qpow/qlog/qexp, De Casteljau squad |
| `random` | LCG, stratified 1D/2D, Latin hypercube, Hammersley, Fibonacci sphere, rejection sampling, cosine hemisphere/uniform sphere importance sampling, alias table O(1), Sobol 4D + scrambling |
| `rendering` | GGX NDF, Smith G, Fresnel-Schlick, Cook-Torrance, Lambertian, Disney diffuse, Oren-Nayar, VNDF sampling, MIS (balance/power²), Beer-Lambert, Henyey-Greenstein, dipole SSS, SSAO kernel, CSM cascades, G-Buffer oct normal pack/unpack, TAA Halton jitter, VSM, SH irradiance, spatiotemporal blue noise, AgX tone map, sRGB/HSV/HSL/YCbCr/XYZ conversions, Bayer dither, equirect/octahedron env mapping, PCF/VSM shadow |
| `sculpting` | Brush falloffs (6 types), draw/smooth/flatten/pinch/crease brushes, Loop subdivision, Catmull-Clark limit position/normal, voxel SDF compute/gradient/sample, edge collapse (half-edge mesh) |
| `spline` | Natural/Clamped/Periodic cubic splines (Thomas algo), Spline3D (chord-length param), NURBS curve (de Boor, tangent, arc-length, t@arc-length), bezier-to-bspline; **extended**: centripetal param, global cubic B-spline LSQ fitting, offset B-spline, knot insertion (Boehm), curve closest-approach (Newton), B-spline surface, arc-length reparameterisation |
| `statistics` | Bootstrap CI, one-way ANOVA, Mann-Whitney U, AR model (Durbin-Levinson), moving average, exponential smoothing, Holt-Winters, PCA (power iteration), K-means (K-means++), correlation matrix, z-score; **extended**: GMM (EM), MAD, Huber M-estimator, Theil-Sen slope, KS test vs normal, Anderson-Darling, Benjamini-Hochberg FDR correction, BIC/AIC/AICc, k-fold CV |
| `transforms` | FFT (Cooley-Tukey in-place), RFFT/IRFFT, DCT-II/III/IV, 8×8 DCT/IDCT, WHT/IWHT, Haar fwd/inv, Daubechies D4 fwd/inv, CZT, Z-transform eval, Hilbert envelope, instantaneous phase, STFT/ISTFT; **extended**: Radon transform, filtered backprojection, Abel forward, Hankel transform (J0), Mellin transform, real cepstrum, CQT, MFCC (13-coeff), wavelet scattering L1 (Morlet) |
| `utilities` | Morton encode 2D/3D, bit ops (rotl/rotr, spread_bits, is_pow2, next_pow2, ilog2, popcount), rsqrt_fast (Quake III), nearly_equal f32/f64, deg↔rad, clamp/saturate/remap templates, SinTable<N> with lerp, prefix_sum, exclusive_scan, RunningStats (Welford), KahanSum, PiecewiseLinear, compile-time Factorial/Binomial |
| `uv` | Planar/cylindrical/spherical/cube projection, UV↔barycentric, area/angle distortion metrics, LSCM unwrap, UV island bin packing (shelf), seam relaxation (Laplacian), TangentFrame (Lengyel), mip level computation |

---

## Building

### With qs_build (primary)

```bash
# Debug
qs_build

# Release + LTO
qs_build --profile release

# Build and run tests
qs_build --test

# Dry run
qs_build --dry-run --verbose
```

### With CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
```

#### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `QS_MATH_BUILD_TESTS` | `ON` | Build unit tests |
| `QS_MATH_BUILD_MODULES` | `ON` | Build leaf module libs |
| `QS_MATH_ENABLE_LTO` | `OFF` | Link-time optimisation |
| `QS_MATH_ENABLE_SIMD` | `ON` | AVX2 + FMA intrinsics |

---

## Usage

### Link everything (umbrella target)

```cmake
target_link_libraries(my_app PRIVATE qs_math_all)
```

### Link only what you need

```cmake
target_link_libraries(my_app PRIVATE
    qs_math_core
    qs_math_rendering
    qs_math_physics)
```

### Quick example

```cpp
#include <qs_math/linear_algebra/vec3.hpp>
#include <qs_math/quaternion/quat.hpp>
#include <qs_math/rendering/rendering.hpp>
using namespace qs::math;

// Rotate a direction vector
Vec3<f32> dir = {0, 0, -1};
Quat<f32> q   = Quat<f32>::from_axis_angle({0,1,0}, 0.785f); // 45°
Vec3<f32> out = q.rotate(dir);

// Cook-Torrance BRDF
Vec3<f32> F0  = {0.04f, 0.04f, 0.04f};
Vec3<f32> brdf = render::cook_torrance(N, V, L, F0, roughness);

// Solve Ax=b with GMRES
numerical::CSRMatrix A = numerical::laplacian_matrix_1d(256, 1.0/255);
std::vector<f64> b(256, 1.0), x(256, 0.0);
numerical::gmres(A, b.data(), x.data(), 256);
```

---

## PCH (Precompiled Header)

The PCH at `core/include/qs_math_pch.hpp` includes:

- All C++20 standard library headers used across the codebase
- Core QS type headers (`vec2/3/4`, `mat3/4`, `quat`, scalar types)

`qs_build` auto-enables PCH when a module has ≥ 30 sources and uses
`REUSE_FROM qs_math_core` in CMake for all leaf modules — no per-module PCH
rebuild cost.

---

## Platform Support

| Platform | Compiler | SIMD |
|----------|----------|------|
| Linux x86-64 | GCC 12+, Clang 15+ | AVX2 + FMA |
| macOS arm64 | Apple Clang 14+ | NEON |
| Windows x64 | MSVC 19.34+ | AVX2 (/arch:AVX2) |
| WebAssembly | Emscripten 3.1+ | SIMD128 (partial) |

---

## Tests

Tests live in `tests/`. Each test file maps to one functional area:

```
tests/
├── test_vectors.cpp
├── test_matrices.cpp
├── test_quat.cpp
├── test_color.cpp
├── test_curves.cpp
├── test_geometry.cpp
├── test_hashing_spatial.cpp
├── test_noise.cpp
├── test_numerical.cpp
├── test_random_stats.cpp
├── test_scalars.cpp
├── test_serialization.cpp
└── test_transform.cpp
```

Run all tests:

```bash
qs_build --test
# or
ctest --test-dir build --output-on-failure
```

---

## License

See `LICENSE`. Part of the Quantum Suite project.
