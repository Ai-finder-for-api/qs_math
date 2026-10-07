#pragma once
// qs_math.hpp — Master include for the Quantum Suite math library
// Include this header to get everything, or include individual headers selectively.

// Foundation
#include "qs_math/scalar_types.hpp"
#include "qs_math/constants.hpp"
#include "qs_math/scalar_math.hpp"
#include "qs_math/trig.hpp"
#include "qs_math/angle.hpp"
#include "qs_math/units.hpp"
#include "qs_math/precision.hpp"

// Vectors
#include "qs_math/vec2.hpp"
#include "qs_math/vec3.hpp"
#include "qs_math/vec4.hpp"
#include "qs_math/bvec.hpp"
#include "qs_math/vec_ops.hpp"

// Matrices
#include "qs_math/mat2.hpp"
#include "qs_math/mat3.hpp"
#include "qs_math/mat4.hpp"

// Quaternion
#include "qs_math/quat.hpp"

// Transform
#include "qs_math/transform.hpp"

// Geometry
#include "qs_math/geometry2d.hpp"
#include "qs_math/geometry3d.hpp"
#include "qs_math/intersection.hpp"
#include "qs_math/bounding_volume.hpp"

// Color
#include "qs_math/color.hpp"

// Curves & Interpolation
#include "qs_math/curves.hpp"
#include "qs_math/interpolation.hpp"

// Noise & Random
#include "qs_math/noise.hpp"
#include "qs_math/random.hpp"

// Statistics & Numerical
#include "qs_math/statistics.hpp"
#include "qs_math/numerical.hpp"

// Coordinate systems & Camera
#include "qs_math/coordinate.hpp"
#include "qs_math/camera_math.hpp"

// Physics & Mesh helpers
#include "qs_math/physics_helpers.hpp"
#include "qs_math/mesh_math.hpp"

// Spatial & Hashing
#include "qs_math/hashing.hpp"
#include "qs_math/spatial.hpp"

// SIMD stubs
#include "qs_math/simd_stub.hpp"

// Serialization & Testing
#include "qs_math/serialization_helpers.hpp"
#include "qs_math/testing_utils.hpp"
