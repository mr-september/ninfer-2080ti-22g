target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/norm_combine.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/norm_combine/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp")
