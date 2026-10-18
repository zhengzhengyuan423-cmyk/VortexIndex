add_executable(run_bench run_bench.cpp)

target_include_directories(run_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(run_bench PRIVATE vortex::vortex)
target_compile_options(run_bench PRIVATE -Wall -Wextra -Werror)
