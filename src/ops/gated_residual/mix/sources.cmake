target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/implementation.cpp")

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/select.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu"
)
