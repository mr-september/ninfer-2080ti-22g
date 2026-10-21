# This fork builds for NVIDIA Turing (sm_75) on x86-64 Linux and Windows.
set(NINFER_CUDA_ARCHITECTURE 75)
set(NINFER_PROCESSOR_PATTERN "^(x86_64|amd64|AMD64)$")
set(NINFER_SYSTEMS "Linux;Windows")

# CUDA compiler detection must see the architecture, not a host default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${NINFER_CUDA_ARCHITECTURE}
    CACHE STRING "CUDA architectures to build")
endif()
function(ninfer_validate_cuda_architecture)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL NINFER_CUDA_ARCHITECTURE)
    message(FATAL_ERROR
      "Turing requires CMAKE_CUDA_ARCHITECTURES=${NINFER_CUDA_ARCHITECTURE}; "
      "got '${CMAKE_CUDA_ARCHITECTURES}'")
  endif()
endfunction()
ninfer_validate_cuda_architecture()

function(ninfer_validate_platform)
  # The toolchain file can change CUDA architectures during project() initialization.
  ninfer_validate_cuda_architecture()
  if(NOT CMAKE_SYSTEM_NAME IN_LIST NINFER_SYSTEMS OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "${NINFER_PROCESSOR_PATTERN}")
    message(FATAL_ERROR
      "Turing does not support ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  configure_file(${PROJECT_SOURCE_DIR}/src/core/platform.h.in
    ${PROJECT_BINARY_DIR}/generated/core/platform.h @ONLY)
  message(STATUS "NInfer: Turing (${CMAKE_SYSTEM_PROCESSOR}, sm_${NINFER_CUDA_ARCHITECTURE})")
endfunction()
