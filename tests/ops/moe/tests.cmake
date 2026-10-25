ninfer_add_op_test(ninfer_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_weights.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_moe_block_scaled_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_block_scaled.cpp"
  LIBRARIES ninfer_ops)
