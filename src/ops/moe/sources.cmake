target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/weights.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/block_scaled/shared.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda/routing/route.cu"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_route/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/topk_softmax_swiglu_moe/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/routed_swiglu/cuda/entry.cpp"
)
include("${CMAKE_CURRENT_LIST_DIR}/topk_softmax_route/bf16/sources.cmake")
