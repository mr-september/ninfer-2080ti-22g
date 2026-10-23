target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/small.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/small.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n6144_k5120.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/bf16/sources.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/q8/sources.cmake")
