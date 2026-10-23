add_executable(vortex_tests
  foundation_test.cpp
  graph_test.cpp
  search_test.cpp
  index_test.cpp
  concurrency_test.cpp)

find_package(Threads REQUIRED)
target_include_directories(vortex_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_definitions(vortex_tests PRIVATE
  VORTEX_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
target_link_libraries(vortex_tests PRIVATE
  vortex::vortex GTest::gtest_main Threads::Threads)

include(GoogleTest)

option(VORTEX_NO_BUILD_DISCOVERY "skip gtest_discover_tests build-time run" OFF)
if(NOT VORTEX_NO_BUILD_DISCOVERY)
  gtest_discover_tests(vortex_tests)
endif()
