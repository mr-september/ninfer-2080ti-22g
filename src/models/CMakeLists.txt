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
target_sources(ninfer_inkling2_graft PRIVATE
  inkling2/graft/include/models/inkling2/executor.cpp
  inkling2/graft/include/models/inkling2/kv_format.cpp)
target_include_directories(ninfer_inkling2_graft PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/graft/include
  ${CMAKE_CURRENT_SOURCE_DIR}/inkling2/execution/vendor)
ninfer_internal_includes(ninfer_inkling2_graft)
target_link_libraries(ninfer_inkling2_graft PUBLIC ninfer_inkling2_execution ninfer_ops ninfer_runtime_support ninfer::json CUDA::cublas)
