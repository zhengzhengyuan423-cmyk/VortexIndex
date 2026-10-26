cmake_minimum_required(VERSION 3.20)
project(VortexIndex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VORTEX_BUILD_TESTS "Build unit tests" ON)
option(VORTEX_BUILD_BENCH "Build benchmark harness" ON)
option(VORTEX_BUILD_CLI "Build vortex-cli tool" ON)

include(GNUInstallDirs)          # CMAKE_INSTALL_LIBDIR 等标准目录名，src 里也要用
add_subdirectory(third_party)
add_subdirectory(src)

if(VORTEX_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(VORTEX_BUILD_CLI)
  add_subdirectory(tools)
endif()

if(VORTEX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()


install(TARGETS vortex EXPORT vortexTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/vortex
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vortexTargets NAMESPACE vortex::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vortex)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/vortex-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/vortex-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vortex)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/vortex-config-version.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/vortex-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/vortex-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vortex)
