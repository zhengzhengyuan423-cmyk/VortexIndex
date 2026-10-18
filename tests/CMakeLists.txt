add_executable(vortex_tests
  foundation_test.cpp
  graph_test.cpp
  search_test.cpp)

target_include_directories(vortex_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
# 数据依赖的测试用绝对路径定位，不受 ctest 工作目录影响
target_compile_definitions(vortex_tests PRIVATE
  VORTEX_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
target_link_libraries(vortex_tests PRIVATE vortex::vortex GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(vortex_tests)
