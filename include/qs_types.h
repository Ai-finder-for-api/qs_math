#pragma once
// qs_math/include/qs_types.h
// Forwarding shim — resolves the relative include from scalar_types.hpp
// The canonical file lives in qs_core/include/qs_types.h
// This shim exists because scalar_types.hpp uses ../../../../include/qs_types.h
// which resolves here from core/include/qs_math/arithmetic/

// Forward to qs_core
#if __has_include("../qs_core/include/qs_types.h")
    #include "../qs_core/include/qs_types.h"
#elif __has_include("../../qs_core/include/qs_types.h")
    #include "../../qs_core/include/qs_types.h"
#else
    // Fallback: define the types directly
    #include <stdint.h>
    #include <stddef.h>
    typedef float           qs_f32;
    typedef double          qs_f64;
    typedef int8_t          qs_i8;
    typedef int16_t         qs_i16;
    typedef int32_t         qs_i32;
    typedef int64_t         qs_i64;
    typedef uint8_t         qs_u8;
    typedef uint16_t        qs_u16;
    typedef uint32_t        qs_u32;
    typedef uint64_t        qs_u64;
    typedef size_t          qs_usize;
    typedef ptrdiff_t       qs_isize;
    typedef unsigned char   qs_byte;
    typedef int             qs_bool;
    #define QS_TRUE  1
    #define QS_FALSE 0
#endif
