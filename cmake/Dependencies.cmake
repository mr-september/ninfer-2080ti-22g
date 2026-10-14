find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# Operating-system link requirements shared by the product and serving libraries.
add_library(ninfer::platform INTERFACE IMPORTED GLOBAL)
if(WIN32)
  target_link_libraries(ninfer::platform INTERFACE ws2_32)
endif()

# FFmpeg and libcurl: pkg-config on Linux/WSL, vcpkg CMake packages on native Windows. Both paths
# expose the same ninfer::ffmpeg / ninfer::curl targets.
add_library(ninfer::ffmpeg INTERFACE IMPORTED GLOBAL)
if(WIN32)
  find_package(FFMPEG REQUIRED)
  target_include_directories(ninfer::ffmpeg INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer::ffmpeg INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer::ffmpeg INTERFACE ${FFMPEG_LIBRARIES})
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  target_link_libraries(ninfer::ffmpeg INTERFACE PkgConfig::FFMPEG)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# CPU grammar source base; consumers will link it explicitly.
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  add_library(ninfer::curl INTERFACE IMPORTED GLOBAL)
  if(WIN32)
    find_package(CURL 7.85 REQUIRED CONFIG)
    target_link_libraries(ninfer::curl INTERFACE CURL::libcurl)
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    target_link_libraries(ninfer::curl INTERFACE PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
