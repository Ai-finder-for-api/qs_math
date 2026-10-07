#pragma once
// qs_math/qs_math.hpp — Master include for all of qs_math/core (C++20)
// Include this single header to pull in the full qs_math library.
// Alternatively include individual headers for faster compilation.

// ── Foundation ──────────────────────────────────────────────────────────────
#include "scalar_types.hpp"
#include "constants.hpp"
#include "scalar_math.hpp"
#include "precision.hpp"
#include "angle.hpp"
#include "trig.hpp"
#include "units.hpp"
#include "constexpr_math.hpp"
#include "integer_math.hpp"
#include "big_integer.hpp"
#include "rational.hpp"
#include "fp16.hpp"
#include "simd_math.hpp"

// ── Vectors ──────────────────────────────────────────────────────────────────
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "bvec.hpp"
#include "vec2_ext.hpp"
#include "vec3_ext.hpp"
#include "vec4_ext.hpp"
#include "vec_ops.hpp"
#include "vec_compare.hpp"
#include "vec_pack.hpp"

// ── Matrices ─────────────────────────────────────────────────────────────────
#include "mat2.hpp"
#include "mat3.hpp"
#include "mat4.hpp"
#include "mat2_ext.hpp"
#include "mat3_ext.hpp"
#include "mat4_ext.hpp"
#include "matrix_decomp.hpp"

// ── Quaternion & Transform ────────────────────────────────────────────────────
#include "quat.hpp"
#include "quat_ext.hpp"
#include "transform.hpp"

// ── Geometry ─────────────────────────────────────────────────────────────────
#include "geometry2d.hpp"
#include "geometry3d.hpp"
#include "intersection.hpp"
#include "bounding_volume.hpp"
#include "polygon.hpp"
#include "distance_queries.hpp"
#include "gjk.hpp"
#include "sweep_tests.hpp"
#include "geometry_utils.hpp"

// ── Curves & Interpolation ────────────────────────────────────────────────────
#include "curves.hpp"
#include "curves_ext.hpp"
#include "interpolation.hpp"

// ── Noise & Random ────────────────────────────────────────────────────────────
#include "noise.hpp"
#include "noise_ext.hpp"
#include "random.hpp"
#include "random_ext.hpp"

// ── Statistics & Numerical ────────────────────────────────────────────────────
#include "statistics.hpp"
#include "statistics_ext.hpp"
#include "numerical.hpp"
#include "numerical_ext.hpp"

// ── Color ─────────────────────────────────────────────────────────────────────
#include "color.hpp"
#include "color_science.hpp"

// ── Camera ────────────────────────────────────────────────────────────────────
#include "camera_math.hpp"
#include "camera_ext.hpp"
#include "coordinate.hpp"

// ── Physics & Material & Lighting ────────────────────────────────────────────
#include "physics_helpers.hpp"
#include "physics_dynamics.hpp"
#include "material_math.hpp"
#include "lighting_math.hpp"

// ── Mesh & Animation ──────────────────────────────────────────────────────────
#include "mesh_math.hpp"
#include "mesh_math_ext.hpp"
#include "animation_math.hpp"

// ── Shader & Terrain ──────────────────────────────────────────────────────────
#include "shader_math.hpp"
#include "terrain_math.hpp"

// ── Collision ─────────────────────────────────────────────────────────────────
#include "collision_broad.hpp"
#include "collision_narrow.hpp"
#include "collision_shapes.hpp"

// ── Image & Compositing ───────────────────────────────────────────────────────
#include "image_processing.hpp"
#include "image_compositing.hpp"

// ── UV ────────────────────────────────────────────────────────────────────────
#include "uv_math.hpp"

// ── Audio & DSP ───────────────────────────────────────────────────────────────
#include "audio_math.hpp"
#include "signal_processing.hpp"

// ── Render & Procedural ───────────────────────────────────────────────────────
#include "render_math.hpp"
#include "procedural_math.hpp"

// ── Deep Math ─────────────────────────────────────────────────────────────────
#include "complex_math.hpp"
#include "dual_number.hpp"
#include "interval_arithmetic.hpp"
#include "number_theory.hpp"
#include "combinatorics.hpp"
#include "graph_theory.hpp"
#include "vector_field.hpp"
#include "scalar_field.hpp"
#include "differential_geom.hpp"

// ── Format & Editor Math ──────────────────────────────────────────────────────
#include "format_math.hpp"
#include "forge_visuals_math.hpp"

// ── Misc ──────────────────────────────────────────────────────────────────────
#include "hashing.hpp"
#include "spatial.hpp"
#include "serialization_helpers.hpp"
#include "testing_utils.hpp"
#include "simd_stub.hpp"

// ── New Core Files (session additions) ───────────────────────────────────────
#include "big_integer.hpp"
#include "rational.hpp"
#include "fp16.hpp"
#include "simd_math.hpp"
#include "hypercomplex.hpp"
#include "geometric_algebra.hpp"
#include "clifford_algebra.hpp"
#include "lie_algebra.hpp"
#include "tensor_math.hpp"
#include "linear_algebra.hpp"
#include "sparse_matrix.hpp"
#include "iterative_solvers.hpp"
#include "eigensolvers.hpp"
#include "least_squares.hpp"
#include "calculus.hpp"
#include "ode_systems.hpp"
#include "pde_helpers.hpp"
#include "fourier_math.hpp"
#include "wavelet_math.hpp"
#include "series_math.hpp"
#include "probability.hpp"
#include "distributions.hpp"
#include "hypothesis_testing.hpp"
#include "bayesian_math.hpp"
#include "information_theory.hpp"
#include "markov_math.hpp"
#include "monte_carlo.hpp"
#include "optimization.hpp"
#include "nonlinear_opt.hpp"
#include "linear_programming.hpp"
#include "quadratic_programming.hpp"
#include "evolutionary.hpp"
#include "convex_math.hpp"
#include "voronoi_math.hpp"
#include "delaunay_math.hpp"
#include "convex_hull.hpp"
#include "csg_math.hpp"
#include "topology_math.hpp"
#include "continuum_mechanics.hpp"
#include "fluid_helpers.hpp"
#include "thermodynamics.hpp"
#include "electromagnetism.hpp"
#include "orbital_mechanics.hpp"
#include "ray_tracing_math.hpp"
#include "volumetric_math.hpp"
#include "color_grading.hpp"
#include "pathfinding_math.hpp"
#include "game_math.hpp"
#include "network_math.hpp"
#include "gizmo_math.hpp"
#include "grid_math.hpp"
#include "spline_editor_math.hpp"
#include "expression_tree.hpp"

// Still-stubbed subsystem includes (add as they are implemented):
// #include "mesh_boolean.hpp"
// #include "geodesic_math.hpp"
// #include "parameterization.hpp"
// #include "remesh_math.hpp"
// #include "rigid_body_ext.hpp"
// #include "displacement_math.hpp"
// #include "occlusion_math.hpp"
// #include "spectral_render.hpp"
// #include "denoising_math.hpp"
// #include "color_vision.hpp"
// #include "color_calibration.hpp"
// #include "color_appearance.hpp"
// #include "uv_pack.hpp"
// #include "uv_distortion.hpp"
// #include "uv_seam.hpp"
// #include "video_math.hpp"
// #include "audio_codec_math.hpp"
// #include "image_filters.hpp"
// #include "compression_math.hpp"
// #include "sculpt_math.hpp"
// #include "displacement_brush.hpp"
// #include "spatial_audio_math.hpp"
// #include "viewport_math.hpp"
// #include "units_ext.hpp"
// #include "coordinate_ext.hpp"
// #include "color_naming.hpp"
// #include "math_constants_ext.hpp"
// #include "benchmarks_math.hpp"

