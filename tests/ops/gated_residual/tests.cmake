ninfer_add_op_test(ninfer_gated_residual_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/norm.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/fixtures.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/validation.cpp"
  LIBRARIES ninfer_ops)
