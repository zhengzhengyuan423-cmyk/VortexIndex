add_library(vortex
  api/index.cpp
  core/distance.cpp)
add_library(vortex::vortex ALIAS vortex)

target_include_directories(vortex PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(vortex PUBLIC cxx_std_17)
target_compile_options(vortex PRIVATE -Wall -Wextra -Werror)
