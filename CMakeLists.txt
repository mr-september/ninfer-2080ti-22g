cmake_minimum_required(VERSION 3.28)

include(cmake/NinferPlatform.cmake)

# The Windows preset takes its dependencies from vcpkg (vcpkg.json). The "x64 Native Tools" prompt of
# Visual Studio 2022 sets VCPKG_ROOT; without it the toolchain path is empty and configure fails with
# an unrelated "/scripts/buildsystems/vcpkg.cmake" error.
if(CMAKE_HOST_WIN32 AND NOT DEFINED ENV{VCPKG_ROOT} AND NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
  message(FATAL_ERROR
    "VCPKG_ROOT is not set. Configure from the \"x64 Native Tools Command Prompt for VS 2022\" "
    "(it sets VCPKG_ROOT), or set VCPKG_ROOT to a vcpkg checkout.")
endif()

project(ninfer LANGUAGES C CXX CUDA)
ninfer_validate_platform()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_ENGINE "Build models, frontend and Engine" ON)
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

if(NINFER_BUILD_APPS AND NOT NINFER_BUILD_ENGINE)
  message(FATAL_ERROR "NINFER_BUILD_APPS requires NINFER_BUILD_ENGINE")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

if(MSVC)
  # Sources and fmt use UTF-8 literals; MSVC otherwise assumes the ANSI code page.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()

if(WIN32)
  # <windows.h> arrives through the NVTX headers; keep std::min/max and numeric_limits usable.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Device code that differs by architecture keys on the capability macros defined here.
add_compile_definitions(NINFER_SM75=1)

# Debug build: record the largest magnitude entering Turing's BF16 -> FP16 conversion (see
# src/ops/common/range_probe.h). Off in every production build.
option(NINFER_SM75_RANGE_PROBE "Record the FP16 operand range on Turing (debug)" OFF)
if(NINFER_SM75_RANGE_PROBE)
  add_compile_definitions(NINFER_SM75_RANGE_PROBE=1)
endif()

# The shipped executables render prompts with this vendored template unless --artifact-chat-template.
# It is embedded in the binaries as a byte array so they run without the source tree.
set(NINFER_DEFAULT_CHAT_TEMPLATE_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/chat_templates/qwen3_8_sharp_v22_5.jinja")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NINFER_DEFAULT_CHAT_TEMPLATE_FILE}")
file(READ "${NINFER_DEFAULT_CHAT_TEMPLATE_FILE}" ninfer_chat_template_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ninfer_chat_template_bytes
       "${ninfer_chat_template_hex}")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/generated/product/default_chat_template.inc"
     CONTENT "${ninfer_chat_template_bytes}\n" NEWLINE_STYLE UNIX)
include_directories("${CMAKE_BINARY_DIR}/generated")

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_ENGINE AND (NINFER_BUILD_APPS OR BUILD_TESTING))
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# Project libraries are shared where the platform supports exporting every symbol (Linux) and
# static on native Windows, where a DLL would need an export annotation on every declaration.
if(WIN32)
  set(NINFER_LIBRARY_TYPE STATIC)
else()
  set(NINFER_LIBRARY_TYPE SHARED)
endif()

# Project binaries share the same libraries and CUDA runtime. These defaults are
# set after third-party targets so their build policies remain independent.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
