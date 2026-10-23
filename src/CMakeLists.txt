find_package(Threads REQUIRED)
add_library(vortex
  api/index.cpp
  core/distance.cpp
  core/bruteforce.cpp
  core/level_generator.cpp
  core/graph_store.cpp
  core/build_pipeline.cpp
  core/hnsw_index.cpp
  core/neighbor_select.cpp
  io/dataset.cpp)
add_library(vortex::vortex ALIAS vortex)
# shared_mutex 的锁操作背后是 pthread
target_link_libraries(vortex PUBLIC Threads::Threads)

target_include_directories(vortex PUBLIC ${PROJECT_SOURCE_DIR}/include)
# api 层要跨目录引 core（GraphStore 等内核类型），不外泄给使用方
target_include_directories(vortex PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(vortex PUBLIC cxx_std_17)
target_compile_options(vortex PRIVATE -Wall -Wextra -Werror)
