# vcpkg portfile for qs-math
# Update SHA512 after tagging v0.1.0 on GitHub

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Quantum-Suite-org/Quantum-suite-math
    REF "v${VERSION}"
    SHA512 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DQS_MATH_BUILD_TESTS=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME qs-math CONFIG_PATH share/qs_math)

# Header-only: no debug/release split needed
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

# Copy license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# Declare header-only
set(VCPKG_BUILD_TYPE release)
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage"
"qs-math provides CMake targets:

    find_package(qs_math CONFIG REQUIRED)
    target_link_libraries(main PRIVATE qs_math::qs_math)
")
