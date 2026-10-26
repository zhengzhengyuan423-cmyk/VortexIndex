find_package(Threads REQUIRED)
add_library(vortex
  api/index.cpp
  api/slot_book.cpp
  core/distance.cpp
  core/bruteforce.cpp
  core/level_generator.cpp
  core/graph_store.cpp
  core/build_pipeline.cpp
  core/hnsw_index.cpp
  core/neighbor_select.cpp
  io/dataset.cpp
  io/serializer.cpp)
add_library(vortex::vortex ALIAS vortex)

target_link_libraries(vortex PUBLIC Threads::Threads)

# PUBLIC 头路径分两套：仓库内构建用源码绝对路径；install(EXPORT) 之后
# 用安装相对目录，不然 vortexTargets.cmake 里会焊死本机路径，换机器即废
target_include_directories(vortex PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# api 层要跨目录引 core（GraphStore 等内核类型），不外泄给使用方
target_include_directories(vortex PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(vortex PUBLIC cxx_std_17)
target_compile_options(vortex PRIVATE -Wall -Wextra -Werror)
