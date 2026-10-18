add_library(ninfer_cpu ${NINFER_LIBRARY_TYPE}
  cpu/system.cpp
  cpu/task_pool.cpp
)
ninfer_internal_includes(ninfer_cpu)
target_link_libraries(ninfer_cpu PRIVATE PkgConfig::HWLOC Threads::Threads)

if(NINFER_TARGET STREQUAL "rtx5090")
  add_library(ninfer_cpu_executor ${NINFER_LIBRARY_TYPE} cpu/executor.cpp cpu/x86.cpp)
  ninfer_internal_includes(ninfer_cpu_executor)
  target_link_libraries(ninfer_cpu_executor PUBLIC ninfer_cpu)
endif()

# Linux reads weights through io_uring; native Windows through an I/O completion port.
if(WIN32)
  set(NINFER_CORE_IO_SOURCES io/direct_read.cpp io/windows/file.cpp io/windows/read_queue.cpp)
else()
  set(NINFER_CORE_IO_SOURCES io/direct_read.cpp io/file.cpp io/read_queue.cpp)
endif()

add_library(ninfer_core ${NINFER_LIBRARY_TYPE}
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  device_memory.cpp
  dtype.cpp
  nf4.cpp
  ${NINFER_CORE_IO_SOURCES}
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  state_image.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  weight_bank.cpp
  expert_cache.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_library(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
if(NOT WIN32)
  target_link_libraries(ninfer_core PRIVATE PkgConfig::URING)
endif()
