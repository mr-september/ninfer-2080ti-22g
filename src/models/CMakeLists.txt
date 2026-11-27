# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp common/bindings.cpp common/weights.cpp)
target_sources(ninfer_model_loading PRIVATE
  qwen4_exp/ple/parameters.cpp
  qwen4_exp/ple/disk/table.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

add_library(ninfer_model_runtime STATIC)
target_sources(ninfer_model_runtime PRIVATE
  qwen4_exp/ple/hash.cpp
  qwen4_exp/ple/upload.cpp
  qwen4_exp/ple/disk/row_reader.cpp
  qwen4_exp/ple/disk/prepare.cpp
)
ninfer_internal_includes(ninfer_model_runtime)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support
  PRIVATE ninfer_media_decode ninfer_jinja ninfer_grammar ninfer_xgrammar)
target_include_directories(ninfer_model_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_model_runtime SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)

target_sources(ninfer_model_loading PRIVATE common/moe/weights.cpp inkling2/port_config.cpp
    inkling2/load/text.cpp)
target_sources(ninfer_model_runtime PRIVATE
  common/context/state_store.cpp
  common/context/logical_kv_store.cpp
  common/context/host_kv_store.cpp
  common/context/kv_history.cpp
  common/context/work.cpp
  common/context/migration.cpp
  common/moe/execution.cpp
  common/moe/schedule.cpp)

include(qwen3_5/loading_sources.cmake)
include(qwen4_exp/loading_sources.cmake)
include(common/qwen/sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen4_exp/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)

# Inkling-2 lean port: the vendored kernels keep a private include root (their fork-local
# headers) that must not shadow the engine's own for every other model translation unit, so
# the executor and kernels compile in their own target.
add_library(ninfer_inkling2_execution STATIC)
target_sources(ninfer_inkling2_execution PRIVATE
  inkling2/execution/executor.cpp
  inkling2/execution/vendor/models/inkling2/kernels.cu)
target_include_directories(ninfer_inkling2_execution PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/execution/vendor)
ninfer_internal_includes(ninfer_inkling2_execution)
target_link_libraries(ninfer_inkling2_execution PUBLIC ninfer_ops ninfer_runtime_support ninfer::json CUDA::cublas)
target_link_libraries(ninfer_model_runtime PUBLIC ninfer_inkling2_execution)

# Inkling-2 donor graft: the donor executor compiled verbatim against the vendored kernel
# surface, with graft/include providing the donor-only headers the port lacks (feature headers
# copied from the donor tree; qwen3_5 model/resources shims). Not linked into ninfer_model_runtime
# yet — only the graft probe target consumes it, so the transcription path stays untouched.
add_library(ninfer_inkling2_graft STATIC)
# device.cu (donor) prints the architecture list via this define (donor core/CMakeLists.txt).
# NINFER_TIER_TURING declares the tier this binary carries (donor rule from CUDA arches; drives
# ops/common/device_tier.h dispatch — without it the tier-dispatch lambdas have no live branch).
target_compile_definitions(ninfer_inkling2_graft PRIVATE "NINFER_CUDA_ARCHITECTURES=\"75\"" NINFER_TIER_TURING)
target_sources(ninfer_inkling2_graft PRIVATE
  inkling2/graft/include/models/inkling2/executor.cpp
  inkling2/graft/include/models/inkling2/kv_format.cpp
  # graft bridge: destination-side definitions (joins, Tensor ctor) + disabled-feature stubs.
  inkling2/graft/src/graft_bridge.cpp
  # donor core implementations, compiled against the graft/include shim tree (destination owns
  # the shared types; see graft-checklist §7-§8).
  # device.cu NOT compiled: the port engine already owns the device-layer objects
  # (CudaCompletionEvent/CudaEventTimer/DeviceContext ctors in ninfer_core's device.cu) — same
  # mangled names collide 38x (LNK2005). Grafted code reads rank facts through the graft
  # CudaExecutionView; promoting donor DeviceContext into the port device layer is a pre-serve
  # task (graft-checklist §8 ledger).
  inkling2/graft/donor/arena.cu
  inkling2/graft/donor/device_wait.cu
  inkling2/graft/donor/growable_region.cu
  inkling2/graft/donor/vmm.cu
  inkling2/graft/donor/kv_planes.cu
  inkling2/graft/donor/device_route.cpp
  inkling2/graft/donor/device_tier.cu
  inkling2/graft/donor/cyclic_kv_cache.cpp
  inkling2/graft/donor/layout.cpp
  inkling2/graft/donor/mapped_host_copy.cu
  inkling2/graft/donor/config.cpp)
  # model.cpp staged in graft/donor/ but NOT compiled: it is the donor LOAD layer (artifact
  # binder/reader, qwen3_5 bindings/tokenizer, Residency) — an API seam against the port loader
  # that is serve-integration work (M4 step 2), not link closure. graft-checklist §8 ledger.
target_include_directories(ninfer_inkling2_graft PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/graft/include
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/execution/vendor
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
ninfer_internal_includes(ninfer_inkling2_graft)

# Donor ggml block-quant bridge (the ops::gguf layer the grafted executor calls), copied verbatim
# under graft/gguf. Its own archive: the ggml headers define CUDA_CHECK and friends, so no other
# NInfer translation unit may see them (donor rule, sources.cmake there).
set(NINFER_GGML_QUANTS_ROOT "E:/ninfer-all/repo/third_party/ggml-quants" CACHE PATH
    "ggml CUDA kernel headers for the inkling2 graft gguf bridge (donor tree)")
add_library(ninfer_inkling2_ggml STATIC)
target_compile_definitions(ninfer_inkling2_ggml PRIVATE "NINFER_CUDA_ARCHITECTURES=\"75\"" NINFER_TIER_TURING)
# Donor ggml target flags (donor src/ops/linear/gguf/sources.cmake): fattn-mma-f16.cuh uses
# __device__-annotated lambdas; 177 = unused-variable noise in the vendored ggml kernels.
target_compile_options(ninfer_inkling2_ggml PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
file(GLOB NINFER_INKLING2_GGML_KERNELS CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/graft/gguf/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/graft/sm70/*.cu)
target_sources(ninfer_inkling2_ggml PRIVATE
  ${NINFER_INKLING2_GGML_KERNELS}
  inkling2/graft/gguf/gguf_linear.cpp)
target_include_directories(ninfer_inkling2_ggml PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/graft/include
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/graft/gguf
  ${NINFER_GGML_QUANTS_ROOT}/include
  ${NINFER_GGML_QUANTS_ROOT}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
ninfer_internal_includes(ninfer_inkling2_ggml)
target_link_libraries(ninfer_inkling2_graft PUBLIC ninfer_inkling2_execution ninfer_ops ninfer_runtime_support ninfer::json CUDA::cublas CUDA::cuda_driver)
