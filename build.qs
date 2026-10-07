// qs_math — Quantum Suite Math Library
// build.qs — qs_build manifest
// Level-based ordering: the dependency graph drives topological compilation order.
//   Level 0: qs_math_core        (no qs_math dependencies)
//   Level 1: all leaf modules    (depend only on qs_math_core)
//   Level 2: qs_math_tests       (depends on qs_math_core + all modules)
//
// Run:  qs_build                          (debug)
//       qs_build --profile release        (release + LTO)
//       qs_build --test                   (build + run tests)
//       qs_build --dry-run --verbose      (plan without compiling)

// ════════════════════════════════════════════════════════════════
// LEVEL 0 — Core library (no qs_math deps)
// ════════════════════════════════════════════════════════════════
module qs_math_core {
    name         = "qs_math_core";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    pch_header   = "core/include/qs_math_pch.hpp";
    pch_source   = "core/src/pch/pch_source.cpp";

    sources = [
        "core/src/pch/pch_source.cpp",
        "core/src/vectors/vec_impl.cpp",
        "core/src/scalar/scalar_impl.cpp",
        "core/src/scalar/random_global.cpp",
        "core/src/matrices/mat_impl.cpp",
        "core/src/noise/noise_impl.cpp",
        "core/src/hashing/hashing_impl.cpp",
        "core/src/quaternion/quat_impl.cpp",
        "core/src/geometry/geometry_impl.cpp",
        "core/src/color/color_impl.cpp",
        "core/src/transform/transform_impl.cpp",
        "core/src/camera/camera_impl.cpp",
        "core/src/serialization/serial_impl.cpp",
        "core/src/interpolation/interpolation_impl.cpp",
        "core/src/numerical/numerical_impl.cpp",
        "core/src/intersection/intersection_impl.cpp",
        "core/src/curves/curves_impl.cpp",
        "core/src/easing/easing_impl.cpp",
        "core/src/random/random_impl.cpp",
        "core/src/statistics/statistics_impl.cpp",
        "core/src/physics_helpers/physics_helpers_impl.cpp",
        "core/src/mesh_math/mesh_math_impl.cpp",
        "core/src/simd/simd_impl.cpp",
        "core/src/spatial/spatial_impl.cpp",
        "core/src/bounding/bounding_impl.cpp",
        "core/src/coordinate/coordinate_impl.cpp",
        "core/src/ml/ml_impl.cpp",
        "core/src/signal/signal_impl.cpp",
        "core/src/analysis/analysis_impl.cpp",
        "core/src/differential/differential_impl.cpp",
        "core/src/algebra/algebra_impl.cpp",
        "core/src/physics/physics_impl.cpp",
        "core/src/rendering/rendering_impl.cpp"
    ];

    include_dirs = [ "core/include", "../qs_core/include" ];
    defines      = [ "QS_MATH_BUILD", "QS_CPP20" ];
    enable_lto   = false;
}

// ════════════════════════════════════════════════════════════════
// LEVEL 1 — Leaf modules (each depends only on qs_math_core)
// ════════════════════════════════════════════════════════════════

module qs_math_algebra {
    name         = "qs_math_algebra";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "algebra/algebra_impl.cpp" ];
    include_dirs = [ "core/include", "algebra/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_arithmetic {
    name         = "qs_math_arithmetic";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "arithmetic/arithmetic_impl.cpp" ];
    include_dirs = [ "core/include", "arithmetic/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_calculus {
    name         = "qs_math_calculus";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "calculus/calculus_impl.cpp" ];
    include_dirs = [ "core/include", "calculus/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_collision {
    name         = "qs_math_collision";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "collision/collision_impl.cpp" ];
    include_dirs = [ "core/include", "collision/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_color {
    name         = "qs_math_color";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "color/color_ext_impl.cpp" ];
    include_dirs = [ "core/include", "color/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_editor {
    name         = "qs_math_editor";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "editor/editor_impl.cpp" ];
    include_dirs = [ "core/include", "editor/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_game {
    name         = "qs_math_game";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "game/game_impl.cpp" ];
    include_dirs = [ "core/include", "game/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_geometry {
    name         = "qs_math_geometry";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "geometry/geometry_ext_impl.cpp" ];
    include_dirs = [ "core/include", "geometry/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_graph_lib {
    name         = "qs_math_graph_lib";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "graph-lib/graph_lib_impl.cpp" ];
    include_dirs = [ "core/include", "graph-lib/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_interpolation {
    name         = "qs_math_interpolation";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "interpolation/interpolation_ext_impl.cpp" ];
    include_dirs = [ "core/include", "interpolation/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_linear_algebra {
    name         = "qs_math_linear_algebra";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "linear-algebra/linear_algebra_impl.cpp" ];
    include_dirs = [ "core/include", "linear-algebra/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_media {
    name         = "qs_math_media";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "media/media_impl.cpp" ];
    include_dirs = [ "core/include", "media/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_noise {
    name         = "qs_math_noise";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "noise/noise_ext_impl.cpp" ];
    include_dirs = [ "core/include", "noise/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_numerical {
    name         = "qs_math_numerical";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "numerical/numerical_module_impl.cpp" ];
    include_dirs = [ "core/include", "numerical/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_optimization {
    name         = "qs_math_optimization";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "optimization/optimization_impl.cpp" ];
    include_dirs = [ "core/include", "optimization/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_physics {
    name         = "qs_math_physics";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "physics/physics_ext_impl.cpp" ];
    include_dirs = [ "core/include", "physics/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_probability {
    name         = "qs_math_probability";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "probability/probability_impl.cpp" ];
    include_dirs = [ "core/include", "probability/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_procedural {
    name         = "qs_math_procedural";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "procedural/procedural_impl.cpp" ];
    include_dirs = [ "core/include", "procedural/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_quaternion {
    name         = "qs_math_quaternion";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "quaternion/quaternion_ext_impl.cpp" ];
    include_dirs = [ "core/include", "quaternion/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_random {
    name         = "qs_math_random";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "random/random_ext_impl.cpp" ];
    include_dirs = [ "core/include", "random/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_rendering {
    name         = "qs_math_rendering";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "rendering/rendering_impl.cpp" ];
    include_dirs = [ "core/include", "rendering/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_sculpting {
    name         = "qs_math_sculpting";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "sculpting/sculpting_impl.cpp" ];
    include_dirs = [ "core/include", "sculpting/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_spline {
    name         = "qs_math_spline";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "spline/spline_impl.cpp" ];
    include_dirs = [ "core/include", "spline/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_statistics {
    name         = "qs_math_statistics";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "statistics/statistics_ext_impl.cpp" ];
    include_dirs = [ "core/include", "statistics/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_transforms {
    name         = "qs_math_transforms";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "transforms/transforms_impl.cpp" ];
    include_dirs = [ "core/include", "transforms/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_utilities {
    name         = "qs_math_utilities";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "utilities/utilities_impl.cpp" ];
    include_dirs = [ "core/include", "utilities/include" ];
    dependencies = [ "qs_math_core" ];
}

module qs_math_uv {
    name         = "qs_math_uv";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "lib";
    output       = "lib";
    kind         = "lib";
    sources      = [ "uv/uv_impl.cpp" ];
    include_dirs = [ "core/include", "uv/include" ];
    dependencies = [ "qs_math_core" ];
}

// ════════════════════════════════════════════════════════════════
// LEVEL 2 — Tests (depend on core + leaf modules)
// ════════════════════════════════════════════════════════════════
module qs_math_tests {
    name         = "qs_math_tests";
    version      = "0.1.0";
    language     = "cpp";
    standard     = "c++20";
    type         = "exe";
    output       = "exe";
    kind         = "exe";
    sources = [
        "tests/test_vectors.cpp",
        "tests/test_matrices.cpp",
        "tests/test_quat.cpp",
        "tests/test_color.cpp",
        "tests/test_curves.cpp",
        "tests/test_geometry.cpp",
        "tests/test_hashing_spatial.cpp",
        "tests/test_noise.cpp",
        "tests/test_numerical.cpp",
        "tests/test_random_stats.cpp",
        "tests/test_scalars.cpp",
        "tests/test_serialization.cpp",
        "tests/test_transform.cpp"
    ];
    include_dirs = [ "core/include" ];
    dependencies = [
        "qs_math_core",
        "qs_math_algebra",
        "qs_math_arithmetic",
        "qs_math_calculus",
        "qs_math_collision",
        "qs_math_color",
        "qs_math_editor",
        "qs_math_game",
        "qs_math_geometry",
        "qs_math_graph_lib",
        "qs_math_interpolation",
        "qs_math_linear_algebra",
        "qs_math_media",
        "qs_math_noise",
        "qs_math_numerical",
        "qs_math_optimization",
        "qs_math_physics",
        "qs_math_probability",
        "qs_math_procedural",
        "qs_math_quaternion",
        "qs_math_random",
        "qs_math_rendering",
        "qs_math_sculpting",
        "qs_math_spline",
        "qs_math_statistics",
        "qs_math_transforms",
        "qs_math_utilities",
        "qs_math_uv"
    ];
    defines = [ "QS_MATH_TESTING" ];
}
