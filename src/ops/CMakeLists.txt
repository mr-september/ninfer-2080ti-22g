# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_non_rdc)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
add_library(ninfer_nvfp4_non_rdc STATIC)
ninfer_internal_includes(ninfer_nvfp4_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core ninfer::cudart CUDA::cuda_driver)

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")

# Blackwell-only formats. The FP8/NVFP4 weight paths, FP8/NVFP4/K8V4 KV caches and the TMA/tcgen
# kernels need sm_120a instructions. Other targets drop these sources from the manifests above
# (keeping the upstream manifests untouched) and link throwing definitions of the same
# declarations instead; the BF16 and grouped-integer routes are the supported surface there.
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  set(_ninfer_sm120_only_regex
    "/(fp8|nvfp4)/"
    "/linear_topk/fp8(_m64)?[.]cu$"
    "/kv_cache/append/(nvfp4|k8v4)_launch[.]cu$"
    "/causal_cache/(fp8|nvfp4|k8v4)/")
  string(REPLACE ";" "|" _ninfer_sm120_only_regex "${_ninfer_sm120_only_regex}")
  foreach(_target ninfer_ops ninfer_nvfp4_non_rdc)
    get_target_property(_sources ${_target} SOURCES)
    list(FILTER _sources EXCLUDE REGEX "${_ninfer_sm120_only_regex}")
    set_property(TARGET ${_target} PROPERTY SOURCES ${_sources})
  endforeach()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/unsupported/sm120_formats_unsupported.cpp")
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/unsupported/empty_non_rdc.cpp")
endif()
