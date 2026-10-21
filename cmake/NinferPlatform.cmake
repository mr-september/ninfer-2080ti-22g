# This fork targets Turing by default; rtx5090 keeps upstream's behavior.
set(NINFER_TARGET turing CACHE STRING "NInfer execution target")
set_property(CACHE NINFER_TARGET PROPERTY STRINGS turing rtx5090)

if(DEFINED NINFER_CONFIGURED_TARGET AND
   NOT NINFER_CONFIGURED_TARGET STREQUAL NINFER_TARGET)
  message(FATAL_ERROR
    "This build directory belongs to ${NINFER_CONFIGURED_TARGET}; "
    "use a separate directory for ${NINFER_TARGET}")
endif()

# Turing builds the portable kernels the source manifests list (the Blackwell-only ones are filtered out in
# src/ops/CMakeLists.txt); NINFER_TARGET keeps behavior that must differ (no CPU expert executor, no expert offload).
set(NINFER_TARGET_TURING 0)
set(NINFER_TARGET_RTX5090 0)
set(NINFER_SYSTEMS "Linux")
if(NINFER_TARGET STREQUAL "turing")
  set(NINFER_TARGET_TURING 1)
  set(NINFER_CUDA_ARCHITECTURE 75)
  set(NINFER_PROCESSOR_PATTERN "^(x86_64|amd64|AMD64)$")
  set(NINFER_SYSTEMS "Linux;Windows")
elseif(NINFER_TARGET STREQUAL "rtx5090")
  set(NINFER_TARGET_RTX5090 1)
  set(NINFER_CUDA_ARCHITECTURE 120a)
  set(NINFER_PROCESSOR_PATTERN "^(x86_64|amd64|AMD64)$")
else()
  message(FATAL_ERROR "Unknown NINFER_TARGET '${NINFER_TARGET}'")
endif()

# CUDA compiler detection must see the selected architecture, not a host default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${NINFER_CUDA_ARCHITECTURE}
    CACHE STRING "CUDA architectures to build")
endif()
function(ninfer_validate_cuda_architecture)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL NINFER_CUDA_ARCHITECTURE)
    message(FATAL_ERROR
      "${NINFER_TARGET} requires CMAKE_CUDA_ARCHITECTURES=${NINFER_CUDA_ARCHITECTURE}; "
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
      "${NINFER_TARGET} does not support ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  set(NINFER_CONFIGURED_TARGET ${NINFER_TARGET} CACHE INTERNAL "Configured NInfer target")
  configure_file(${PROJECT_SOURCE_DIR}/src/core/platform.h.in
    ${PROJECT_BINARY_DIR}/generated/core/platform.h @ONLY)
  message(STATUS "NInfer target: ${NINFER_TARGET} (${CMAKE_SYSTEM_PROCESSOR}, sm_${NINFER_CUDA_ARCHITECTURE})")
endfunction()
