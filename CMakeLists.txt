cmake_minimum_required(VERSION 3.20)
project(VortexIndex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VORTEX_BUILD_TESTS "Build unit tests" ON)
option(VORTEX_BUILD_BENCH "Build benchmark harness" ON)

add_subdirectory(third_party)
add_subdirectory(src)

if(VORTEX_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(VORTEX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
