# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops ${NINFER_LIBRARY_TYPE})
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_library(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_ops_non_rdc)

# Non-RDC kernels keep instruction and register-allocation choices within one translation unit.
# This private PIC archive joins the shared library through host launchers.
add_library(ninfer_ops_non_rdc STATIC)
ninfer_internal_includes(ninfer_ops_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_ops_non_rdc)
target_link_libraries(ninfer_ops_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

# Debug build: NINFER_SM75_RANGE_PROBE (defined in the top-level CMakeLists.txt) records the
# largest magnitude entering Turing's BF16 -> FP16 conversion; see common/range_probe.h.
if(NINFER_SM75_RANGE_PROBE)
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/common/range_probe.cu")
endif()

target_sources(ninfer_ops PRIVATE weight_input.cpp)
include("${CMAKE_CURRENT_LIST_DIR}/common/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/elementwise/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/moe/sources.cmake")

# Blackwell-only kernels. FP8/NVFP4 weight paths, FP8/NVFP4/K8V4 KV caches and the TMA/tcgen
# kernels need sm_120a instructions. A Turing build drops these sources from the family manifests
# above (which stay identical to upstream's) and links throwing definitions of the same
# declarations instead; the BF16 and grouped-integer routes are the supported surface there.
if(NINFER_TARGET STREQUAL "turing")
  set(_ninfer_blackwell_only_regex
    "/(fp8|nvfp4|k8v4)/"
    "/(fp8|nvfp4|k8v4)[a-z0-9_]*[.](cu|cpp)$"
    "/gated_residual/mix/"
    "/kimi_delta_attention/"
    "/softmax_attention/qsa/")
  string(REPLACE ";" "|" _ninfer_blackwell_only_regex "${_ninfer_blackwell_only_regex}")
  foreach(_target ninfer_ops ninfer_ops_non_rdc)
    get_target_property(_sources ${_target} SOURCES)
    list(FILTER _sources EXCLUDE REGEX "${_ninfer_blackwell_only_regex}")
    set_property(TARGET ${_target} PROPERTY SOURCES ${_sources})
  endforeach()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/unsupported/turing_unsupported.cpp")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/unsupported/empty_non_rdc.cpp")
endif()
