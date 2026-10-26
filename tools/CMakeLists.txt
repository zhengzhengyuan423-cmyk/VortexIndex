find_package(Threads REQUIRED)

add_executable(vortex-cli vortex_cli.cpp)
target_include_directories(vortex-cli PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(vortex-cli PRIVATE vortex::vortex Threads::Threads)
target_compile_options(vortex-cli PRIVATE -Wall -Wextra -Werror)
