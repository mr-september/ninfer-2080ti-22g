target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/plain/softmax_attention.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/uniform.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/plan.cpp"
)

