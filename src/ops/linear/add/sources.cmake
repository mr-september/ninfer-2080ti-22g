target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_add.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n2048_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n2048_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n5120_k6144.cu"
)
