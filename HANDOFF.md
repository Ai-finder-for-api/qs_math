# qs_math — Quantum Suite Math Library: Handoff Document
## For the next Claude session continuing this work

---

## Repository
**GitHub:** https://github.com/Quantum-Suite-org/Quantum-suite-math  
**Local path:** `/home/claude/qs_math/`  
**Language:** C++20, header-only core  
**Build system:** No CMake in core — qs_build (Zig) handles everything  
**vcpkg packaging:** In progress (see below for what still needs doing)

---

## Current State (end of this session)

| Metric | Value |
|--------|-------|
| Total headers in core | **166 .hpp files** |
| Total lines of code | **~34,812 LOC** |
| LOC target | **70,000+** |
| LOC remaining | **~35,188** |
| New files written this multi-session run | **77 files** (started from 89 originals) |

### Progress summary across all sessions:
- Session 1–2: Original 89 headers (~16k LOC) established  
- Sessions 3–7 (this context): +77 new headers, +18,716 LOC  
- **Still need ~35k more LOC** to hit the 70k target

---

## Architecture

```
qs_math/
├── core/
│   ├── include/
│   │   └── qs_math/          ← ALL 166 headers live here (flat structure)
│   │       ├── qs_math.hpp   ← Master include
│   │       └── *.hpp         ← Individual subsystems
│   └── src/                  ← (empty — header-only)
├── collision/                 ← stub
├── graph-lib/                 ← stub
├── physics/                   ← stub
├── [25+ other subsystem stubs]
├── benchmarks/
├── tests/
├── docs/
├── vcpkg.json                 ← ⚠️ NEEDS CREATION (see below)
├── CMakeLists.txt             ← ⚠️ NEEDS CREATION (see below)
└── cmake/
    └── qs_math-config.cmake   ← ⚠️ NEEDS CREATION (see below)
```

### Design invariants (DO NOT BREAK):
- **No allocations in hot paths** — all heap via STL containers in non-critical paths only
- **`[[nodiscard]]` everywhere** on pure functions
- **`constexpr` where possible**
- **`extern "C"` ABI at language seams** (Rust/Go/C# boundary)
- **No duplicate math across subsystems** — subsystems call into core
- **No CMake in core** — qs_build (Zig) handles the Quantum Suite build
- **`namespace qs {}`** — everything lives in the `qs` namespace
- **Header-only** — all implementation in `.hpp`, no `.cpp` files in core

---

## All 166 Headers (alphabetical)

```
angle.hpp               animation_math.hpp      audio_codec_math.hpp
audio_math.hpp          bayesian_math.hpp       benchmarks_math.hpp
big_integer.hpp         bounding_volume.hpp     bvec.hpp
calculus.hpp            camera_ext.hpp          camera_math.hpp
clifford_algebra.hpp    collision_broad.hpp     collision_narrow.hpp
collision_shapes.hpp    color.hpp               color_appearance.hpp
color_calibration.hpp   color_grading.hpp       color_naming.hpp
color_science.hpp       color_vision.hpp        combinatorics.hpp
complex_math.hpp        compression_math.hpp    constants.hpp
constexpr_math.hpp      continuum_mechanics.hpp convex_hull.hpp
convex_math.hpp         coordinate.hpp          csg_math.hpp
curves.hpp              curves_ext.hpp          delaunay_math.hpp
differential_geom.hpp   displacement_brush.hpp  distance_queries.hpp
distributions.hpp       dual_number.hpp         eigensolvers.hpp
electromagnetism.hpp    evolutionary.hpp        expression_tree.hpp
fluid_helpers.hpp       forge_visuals_math.hpp  format_math.hpp
fourier_math.hpp        fp16.hpp                game_math.hpp
geodesic_math.hpp       geometric_algebra.hpp   geometry2d.hpp
geometry3d.hpp          geometry_utils.hpp      gizmo_math.hpp
gjk.hpp                 graph_theory.hpp        grid_math.hpp
hashing.hpp             hypercomplex.hpp        hypothesis_testing.hpp
image_compositing.hpp   image_filters.hpp       image_processing.hpp
information_theory.hpp  integer_math.hpp        interpolation.hpp
intersection.hpp        interval_arithmetic.hpp iterative_solvers.hpp
least_squares.hpp       lie_algebra.hpp         lighting_math.hpp
linear_algebra.hpp      linear_programming.hpp  markov_math.hpp
mat2.hpp                mat2_ext.hpp            mat3.hpp
mat3_ext.hpp            mat4.hpp                mat4_ext.hpp
material_math.hpp       math_constants_ext.hpp  matrix_decomp.hpp
mesh_boolean.hpp        mesh_math.hpp           mesh_math_ext.hpp
monte_carlo.hpp         network_math.hpp        noise.hpp
noise_ext.hpp           nonlinear_opt.hpp       number_theory.hpp
numerical.hpp           numerical_ext.hpp       ode_systems.hpp
optimization.hpp        orbital_mechanics.hpp   parameterization.hpp
pathfinding_math.hpp    pde_helpers.hpp         physics_dynamics.hpp
physics_helpers.hpp     polygon.hpp             precision.hpp
probability.hpp         procedural_math.hpp     qs_math.hpp
quadratic_programming.hpp  quat.hpp             quat_ext.hpp
random.hpp              random_ext.hpp          rational.hpp
ray_tracing_math.hpp    remesh_math.hpp         render_math.hpp
rigid_body_ext.hpp      scalar_field.hpp        scalar_math.hpp
scalar_types.hpp        sculpt_math.hpp         serialization_helpers.hpp
series_math.hpp         shader_math.hpp         signal_processing.hpp
simd_math.hpp           simd_stub.hpp           sparse_matrix.hpp
spatial.hpp             spatial_audio_math.hpp  spline_editor_math.hpp
statistics.hpp          statistics_ext.hpp      sweep_tests.hpp
tensor_math.hpp         terrain_math.hpp        testing_utils.hpp
thermodynamics.hpp      topology_math.hpp       transform.hpp
trig.hpp                units.hpp               units_ext.hpp
uv_distortion.hpp       uv_math.hpp             uv_pack.hpp
uv_seam.hpp             vec2.hpp                vec2_ext.hpp
vec3.hpp                vec3_ext.hpp            vec4.hpp
vec4_ext.hpp            vec_compare.hpp         vec_ops.hpp
vec_pack.hpp            vector_field.hpp        video_math.hpp
viewport_math.hpp       volumetric_math.hpp     voronoi_math.hpp
wavelet_math.hpp
```

---

## Headers written THIS session (new, ~18k LOC added)

### Batch 5
- `evolutionary.hpp` — GA, DE/rand/1/bin, CMA-ES, PSO
- `convex_math.hpp` — Projections, ADMM, Frank-Wolfe, Moreau, Bregman divergence
- `voronoi_math.hpp` — Fortune's algo helpers, Lloyd relaxation, power diagram
- `delaunay_math.hpp` — Bowyer-Watson, circumcircle, Lawson flip
- `convex_hull.hpp` — Quickhull 3D full, gift-wrapping, incremental hull
- `csg_math.hpp` — SDF boolean ops, BSP-tree polygon CSG (union/subtract/intersect)
- `topology_math.hpp` — Winding numbers, Euler characteristic, genus, persistence
- `continuum_mechanics.hpp` — Strain tensors, hyperelastic (Neo-Hookean, Mooney-Rivlin), von Mises

### Batch 6
- `fluid_helpers.hpp` — SPH kernels (poly6/spiky/cubic/Wendland), NS grid, LBM D2Q9
- `thermodynamics.hpp` — EOS (ideal/VdW/Peng-Robinson), blackbody, Carnot cycles
- `electromagnetism.hpp` — Maxwell helpers, Biot-Savart, Fresnel, Lorentz, transmission line
- `orbital_mechanics.hpp` — Kepler equation, Hohmann transfer, J2 perturbation, tisserand
- `ray_tracing_math.hpp` — MIS weights, BVH SAH, hemisphere sampling, photon mapping
- `volumetric_math.hpp` — Phase functions (HG/Rayleigh/Mie), delta tracking, atmosphere model, SSS
- `color_grading.hpp` — LGG, ASC CDL, 3D LUT, color wheels, film emulation, ACES
- `pathfinding_math.hpp` — A* heuristics, grid A*, NavMesh, steering behaviours (flock/pursuit/evade), potential fields

### Batch 7
- `game_math.hpp` — Damage formulas, loot tables, alias sampling, stat blocks, fog-of-war
- `network_math.hpp` — Dead reckoning, snapshot buffer, lag compensation, delta compression, jitter buffer
- `gizmo_math.hpp` — Translate/rotate/scale hit testing, screen-space picking, snapping
- `grid_math.hpp` — Cartesian, polar, isometric, hex (axial coords), voxel DDA
- `quadratic_programming.hpp` — Active set, OSQP-style ADMM, equality-constrained
- `spline_editor_math.hpp` — Tangent constraints, arc-length reparam, knot insertion, polyline→spline
- `expression_tree.hpp` — AST, eval, symbolic differentiation, recursive descent parser

### Batch 8
- `mesh_boolean.hpp` — Triangle-triangle intersection, BSP-based union/subtract/intersect
- `geodesic_math.hpp` — Dijkstra geodesic, MMP approximation, heat method, isoline extraction
- `rigid_body_ext.hpp` — Articulated bodies, Jacobians, Lagrangian mechanics, constraint dynamics
- `units_ext.hpp` — Full SI/Imperial/astronomical conversion registry, dimensional analysis
- `math_constants_ext.hpp` — 50+ mathematical & physical constants (CODATA 2018)
- `viewport_math.hpp` — Orbit/arcball/fly cameras, marquee selection, depth linearization
- `benchmarks_math.hpp` — Timer, benchmark runner, FLOP counting, roofline model

### Batch 9
- `parameterization.hpp` — Tutte, harmonic, LSCM, cotangent Laplacian, UV stretch metrics
- `remesh_math.hpp` — Half-edge mesh, isotropic remesh ops, quadric error metrics
- `sculpt_math.hpp` — Brush falloffs, draw/smooth/flatten/crease/pinch/grab/clay, multi-res
- `displacement_brush.hpp` — Brush params, stroke path/dab placement, PN triangles, disp layers
- `spatial_audio_math.hpp` — HRTF, ambisonics FOA/HOA, reverb zones, acoustic occlusion
- `color_naming.hpp` — 148 CSS colors, nearest-color, palette generation, dominant colors

### Batch 10
- `image_filters.hpp` — Bilateral, guided, anisotropic diffusion, TV denoising, morphological, Canny, CLAHE
- `compression_math.hpp` — Huffman, arithmetic, ANS, DCT 8×8, LZ77, wavelet EZW/SPIHT
- `video_math.hpp` — Frame timing, motion vectors (full/diamond/subpixel), YUV420, PSNR/SSIM, LUFS
- `audio_codec_math.hpp` — PCM/μ-law/A-law, sinc resampling, LUFS loudness, psychoacoustic masking

### Batch 11 (partial — this is where session ended)
- `color_vision.hpp` — CVD simulation (protan/deutan/tritan), CIECAM02, CAM16, Daltonize
- `color_calibration.hpp` — White balance, Delta-E 76/94/2000, gamut mapping, ICC math
- `color_appearance.hpp` — CAT02/CAT16/CMCCAT2000, ICtCp, OKLCH interpolation, APCA contrast
- `uv_pack.hpp` — Island detection, border tracing, shelf-bin packing, lightmap packing
- `uv_distortion.hpp` — Jacobian, SVD, L2/MIPS/angle/area distortion, symmetric Dirichlet
- `uv_seam.hpp` — Seam candidates, cut-to-disk, virtual node splitting, cone singularities

---

## What Still Needs Doing

### 1. LOC — **HIGHEST PRIORITY** (~35k more needed)
Currently at **34,812 LOC**, need **70,000+**.

Suggested areas for new dense headers (aim for 300–500 LOC each):

**Rendering / GI:**
- `occlusion_math.hpp` — SSAO, HBAO, bent normals, voxel AO, horizon-based AO
- `spectral_render.hpp` — Hero wavelength, spectral upsampling (Smits), fluorescence
- `denoising_math.hpp` — Bilateral, NL-means, spatio-temporal, SVGF, A-SVGF kernels
- `displacement_math.hpp` — PN triangles refinement, hardware tessellation math, micro-displacement

**Physics extension:**
- `cloth_math.hpp` — Position-based dynamics, XPBD constraints, bending stiffness
- `fracture_math.hpp` — Voronoi fracture, crack propagation, brittle/ductile
- `soft_body_math.hpp` — FEM tetrahedra, co-rotational elasticity

**Signal/Audio:**
- `convolution_reverb.hpp` — FFT convolution, partitioned convolution, Schroeder reverb
- `pitch_math.hpp` — YIN algorithm, autocorrelation, pitch shifting, phase vocoder
- `spectral_analysis.hpp` — Short-time Fourier, CQT, mel spectrogram, MFCC

**Procedural:**
- `l_systems.hpp` — L-system grammar, turtle geometry, fractal plants
- `city_gen_math.hpp` — Lot subdivision, road networks, block generation
- `terrain_erosion.hpp` — Hydraulic erosion, thermal erosion, wind erosion

**AI/ML math:**
- `neural_math.hpp` — Forward/backprop primitives, activations, batch norm, attention math
- `loss_functions.hpp` — Cross-entropy, focal loss, Huber, triplet, contrastive
- `optimizer_math.hpp` — SGD/Adam/AdaGrad/RMSProp math

**Cryptography / hash:**
- `hash_math.hpp` — MurmurHash3, xxHash, FNV, SipHash, Poly1305 field math

**More algebra:**
- `polynomial_math.hpp` — Polynomial roots (Durand-Kerner), Sturm chains, resultants
- `continued_fractions.hpp` — CF convergents, Stern-Brocot, best rational approximation
- `finite_field_math.hpp` — GF(p), GF(2^n), LFSR, BCH/Reed-Solomon syndromes

**More geometry:**
- `voronoi_3d.hpp` — 3D Voronoi, power diagram, Apollonius
- `medial_axis.hpp` — MAT, Voronoi-based skeleton, thinning
- `implicit_surfaces.hpp` — Metaballs, RBF implicit, level-set operations

Each header above should target **300–500 LOC** of real math implementations.

### 2. vcpkg Packaging — **NEEDS IMPLEMENTATION**

Files to create (none exist yet):

```
qs_math/
├── vcpkg.json                     ← package manifest
├── CMakeLists.txt                 ← header-only INTERFACE target
├── cmake/
│   └── qs_math-config.cmake       ← find_package() support
└── ports/qs-math/                 ← vcpkg port files (for registry)
    ├── portfile.cmake
    └── vcpkg.json
```

**vcpkg.json content needed:**
```json
{
  "name": "qs-math",
  "version": "0.1.0",
  "description": "Quantum Suite math library — 166-header C++20 header-only core",
  "homepage": "https://github.com/Quantum-Suite-org/Quantum-suite-math",
  "license": "MIT",
  "supports": "!uwp",
  "dependencies": []
}
```

**CMakeLists.txt needed:**
```cmake
cmake_minimum_required(VERSION 3.20)
project(qs_math VERSION 0.1.0 LANGUAGES CXX)
add_library(qs_math INTERFACE)
target_include_directories(qs_math INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(qs_math INTERFACE cxx_std_20)
install(TARGETS qs_math EXPORT qs_math-targets)
install(DIRECTORY core/include/ DESTINATION include)
install(EXPORT qs_math-targets FILE qs_math-config.cmake
        NAMESPACE qs_math:: DESTINATION share/qs_math)
```

**portfile.cmake needed:**
```cmake
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Quantum-Suite-org/Quantum-suite-math
    REF v0.1.0
    SHA512 <FILL-AFTER-TAGGING>
    HEAD_REF main
)
vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME qs-math)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
set(VCPKG_BUILD_TYPE release)
```

### 3. Python / Java / Rust bindings — **DEFERRED (user confirmed)**
Will be separate repos under Quantum-Suite-org. Math core is C++ only for now.

### 4. Test suite — **STUB EXISTS, needs content**
`tests/` directory exists but is empty. Should add:
- `tests/CMakeLists.txt`
- `tests/test_vec3.cpp`
- `tests/test_mat4.cpp`
- `tests/test_color.cpp`
- etc.

### 5. Documentation — **STUB EXISTS**
`docs/` is empty. Doxygen config needed.

---

## Key Cross-File Dependencies

Headers that almost everything depends on (touch carefully):
```
scalar_types.hpp  ← defines f32/f64/u32/i32/u64/i64/Vec2i etc.
vec2.hpp, vec3.hpp, vec4.hpp  ← used by ~80% of headers
mat3.hpp, mat4.hpp  ← transforms, physics, rendering
quat.hpp  ← animation, physics, camera
geometry3d.hpp  ← intersection, collision, rendering
intersection.hpp  ← ray tracing, collision, gizmo
random.hpp  ← PCG32, noise, Monte Carlo
noise.hpp  ← terrain, procedural, sculpt
linear_algebra.hpp  ← DenseMatrix, solve(); used by optimization, FEM, parameterization
sparse_matrix.hpp  ← SparseCSR; used by parameterization, remesh, FEM
iterative_solvers.hpp  ← cg_solve(); used by parameterization, ADMM
color.hpp  ← rgb/hsv/lab/xyz conversions; used by all color_* headers
```

---

## Session History (what each session did)

| Session | Files Added | LOC Added | Focus |
|---------|-------------|-----------|-------|
| 1–2 | 89 originals | ~16,096 | Core vectors/matrices/noise/geometry/physics/render |
| 3 (batch 1–4) | +40 | +9,174 | Algebra (Clifford, Lie, Tensor), solvers, probability, optimization |
| 4 (batch 5–7) | +27 | +6,800 | Geometry (Voronoi/Delaunay/hull/CSG/topology), physics (continuum/fluid/thermo/EM), raytracing, game/network |
| 5 (batch 8–9) | +20 | +3,000 | Mesh tools, UV, sculpt, audio spatial, color deep stack |
| 6 (batch 10–11) | +10 | ~2,742 | Image filters, compression, video, audio codec, color calibration/appearance |

**Total: 166 headers, 34,812 LOC**

---

## How to Continue

```bash
# Resume in container
DIR=/home/claude/qs_math/core/include/qs_math

# Check current count
find $DIR -name "*.hpp" | wc -l
find $DIR -name "*.hpp" -exec wc -l {} + | tail -1

# Write new headers (same pattern as all previous batches)
# Each should be 300-500 LOC, self-contained, use namespace qs {}
# Follow existing style: [[nodiscard]], constexpr, no allocations in hot paths

# After reaching 70k LOC, create vcpkg files:
# - /home/claude/qs_math/vcpkg.json
# - /home/claude/qs_math/CMakeLists.txt
# - /home/claude/qs_math/cmake/qs_math-config.cmake
# - /home/claude/qs_math/ports/qs-math/portfile.cmake
# - /home/claude/qs_math/ports/qs-math/vcpkg.json
# - /home/claude/qs_math/LICENSE (MIT)
# - /home/claude/qs_math/README.md (user-facing)

# Then zip:
cd /home/claude && zip -r qs_math_vX.zip qs_math/
```

---

## Style Reference (copy this pattern for new headers)

```cpp
#pragma once
// qs_math/your_header.hpp — Brief description (C++20)
#include "scalar_types.hpp"
#include "vec3.hpp"          // include only what you need
#include <vector>
#include <cmath>
#include <cassert>

namespace qs {

// ─── Section name ─────────────────────────────────────────────────────────────

[[nodiscard]] inline f32 some_function(f32 a, f32 b) {
    return a + b;
}

struct SomeStruct {
    f32 field{0};
    [[nodiscard]] f32 method() const { return field; }
};

} // namespace qs
```

---

*Generated at end of session — 166 headers, 34,812 LOC, ~35k LOC remaining to target.*
