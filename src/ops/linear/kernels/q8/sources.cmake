target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q8_sliced_k_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_simt.cu"
)
