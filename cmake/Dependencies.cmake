find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if(WIN32)
  # Native Windows takes hwloc, FFmpeg and libcurl from vcpkg (vcpkg.json) and reads their
  # pkg-config files with vcpkg's pkgconf, so every consumer uses the same PkgConfig:: targets.
  set(_ninfer_vcpkg_prefix "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
  find_program(PKG_CONFIG_EXECUTABLE NAMES pkgconf pkg-config
    HINTS "${_ninfer_vcpkg_prefix}/tools/pkgconf" "${_ninfer_vcpkg_prefix}/../x64-windows/tools/pkgconf")
  set(ENV{PKG_CONFIG_PATH}
    "${_ninfer_vcpkg_prefix}/lib/pkgconfig;${_ninfer_vcpkg_prefix}/share/pkgconfig")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(HWLOC REQUIRED IMPORTED_TARGET hwloc>=2.5)
if(NOT WIN32)
  # Windows reads files through an I/O completion port (core/io/windows).
  pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing>=2.5)
endif()

# Operating-system link requirements shared by the product and serving libraries.
add_library(ninfer::platform INTERFACE IMPORTED GLOBAL)
if(WIN32)
  target_link_libraries(ninfer::platform INTERFACE ws2_32)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

if(NINFER_BUILD_ENGINE)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)
  add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)
endif()

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
