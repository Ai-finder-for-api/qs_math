# qs_math — Quantum Suite Math Library
# CMakeLists.txt — full CMake build (companion to build.qs)
# cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

cmake_minimum_required(VERSION 3.20)
project(qs_math VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ───────────────────────────────────────────────────────────────────
option(QS_MATH_BUILD_TESTS   "Build unit tests"          ON)
option(QS_MATH_BUILD_MODULES "Build leaf module libs"    ON)
option(QS_MATH_ENABLE_LTO    "Enable link-time optimisation" OFF)
option(QS_MATH_ENABLE_SIMD   "Enable SIMD intrinsics"   ON)

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /MP /utf-8 /Zc:__cplusplus)
    if(QS_MATH_ENABLE_SIMD)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -fno-exceptions)
    if(QS_MATH_ENABLE_SIMD)
        add_compile_options(-mavx2 -mfma)
    endif()
    if(QS_MATH_ENABLE_LTO)
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

# ── PCH ───────────────────────────────────────────────────────────────────────
# Shared PCH for all targets — matches pch_header in build.qs
set(QS_PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/core/include/qs_math_pch.hpp")

# ── Core include directory (used by every target) ─────────────────────────────
set(QS_CORE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/core/include")

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 0 — qs_math_core
# ─────────────────────────────────────────────────────────────────────────────
file(GLOB_RECURSE QS_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/core/src/*.cpp")

add_library(qs_math_core STATIC ${QS_CORE_SOURCES})
target_include_directories(qs_math_core PUBLIC ${QS_CORE_INCLUDE})
target_compile_definitions(qs_math_core PRIVATE QS_MATH_BUILD QS_CPP20)
target_precompile_headers(qs_math_core PRIVATE ${QS_PCH_HEADER})
set_target_properties(qs_math_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ─────────────────────────────────────────────────────────────────────────────
# Helper macro: add_qs_module(NAME dir [EXTRA_SOURCES...])
# Builds a STATIC lib at LEVEL 1 (depends on qs_math_core).
# ─────────────────────────────────────────────────────────────────────────────
macro(add_qs_module TARGET_NAME MODULE_DIR)
    set(_extra_src ${ARGN})
    file(GLOB _mod_src CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR}/src/*.cpp")
    list(APPEND _mod_src ${_extra_src})
    if(_mod_src)
        add_library(${TARGET_NAME} STATIC ${_mod_src})
    else()
        # Header-only shim — still register the target for dependency tracking
        add_library(${TARGET_NAME} INTERFACE)
        target_include_directories(${TARGET_NAME} INTERFACE ${QS_CORE_INCLUDE})
        return()
    endif()
    target_include_directories(${TARGET_NAME}
        PUBLIC  ${QS_CORE_INCLUDE}
                "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR}/include"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR}")
    target_link_libraries(${TARGET_NAME} PUBLIC qs_math_core)
    target_precompile_headers(${TARGET_NAME} REUSE_FROM qs_math_core)
    set_target_properties(${TARGET_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endmacro()

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — Leaf modules
# ─────────────────────────────────────────────────────────────────────────────
if(QS_MATH_BUILD_MODULES)
    add_qs_module(qs_math_algebra        algebra)
    add_qs_module(qs_math_arithmetic     arithmetic)
    add_qs_module(qs_math_calculus       calculus)
    add_qs_module(qs_math_collision      collision)
    add_qs_module(qs_math_color          color)
    add_qs_module(qs_math_editor         editor)
    add_qs_module(qs_math_game           game)
    add_qs_module(qs_math_geometry       geometry)
    add_qs_module(qs_math_graph_lib      graph-lib)
    add_qs_module(qs_math_interpolation  interpolation)
    add_qs_module(qs_math_linear_algebra linear-algebra)
    add_qs_module(qs_math_media          media)
    add_qs_module(qs_math_noise          noise)
    add_qs_module(qs_math_numerical      numerical)
    add_qs_module(qs_math_optimization   optimization)
    add_qs_module(qs_math_physics        physics)
    add_qs_module(qs_math_probability    probability)
    add_qs_module(qs_math_procedural     procedural)
    add_qs_module(qs_math_quaternion     quaternion)
    add_qs_module(qs_math_random         random)
    add_qs_module(qs_math_rendering      rendering)
    add_qs_module(qs_math_sculpting      sculpting)
    add_qs_module(qs_math_spline         spline)
    add_qs_module(qs_math_statistics     statistics)
    add_qs_module(qs_math_transforms     transforms)
    add_qs_module(qs_math_utilities      utilities)
    add_qs_module(qs_math_uv             uv)

    # Umbrella interface library — link this to get everything
    add_library(qs_math_all INTERFACE)
    target_link_libraries(qs_math_all INTERFACE
        qs_math_core
        qs_math_algebra qs_math_arithmetic qs_math_calculus
        qs_math_collision qs_math_color qs_math_editor
        qs_math_game qs_math_geometry qs_math_graph_lib
        qs_math_interpolation qs_math_linear_algebra qs_math_media
        qs_math_noise qs_math_numerical qs_math_optimization
        qs_math_physics qs_math_probability qs_math_procedural
        qs_math_quaternion qs_math_random qs_math_rendering
        qs_math_sculpting qs_math_spline qs_math_statistics
        qs_math_transforms qs_math_utilities qs_math_uv)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — Tests
# ─────────────────────────────────────────────────────────────────────────────
if(QS_MATH_BUILD_TESTS)
    enable_testing()

    file(GLOB QS_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

    foreach(_test_src ${QS_TEST_SOURCES})
        get_filename_component(_test_name ${_test_src} NAME_WE)
        add_executable(${_test_name} ${_test_src})
        target_include_directories(${_test_name} PRIVATE ${QS_CORE_INCLUDE})
        target_compile_definitions(${_test_name} PRIVATE QS_MATH_TESTING)
        target_precompile_headers(${_test_name} REUSE_FROM qs_math_core)
        if(TARGET qs_math_all)
            target_link_libraries(${_test_name} PRIVATE qs_math_all)
        else()
            target_link_libraries(${_test_name} PRIVATE qs_math_core)
        endif()
        set_target_properties(${_test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests")
        add_test(NAME ${_test_name}
                 COMMAND ${_test_name}
                 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests")
    endforeach()
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(DIRECTORY core/include/ DESTINATION include/qs_math)
install(TARGETS qs_math_core ARCHIVE DESTINATION lib)
install(FILES build.qs DESTINATION share/qs_math)
