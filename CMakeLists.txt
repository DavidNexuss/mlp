cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ftree-vectorize -march=native")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
set(CMAKE_CXX_STANDARD 17)

project(mlp)

find_package(OpenMP REQUIRED)

include_directories(lib)
include_directories(src)
include_directories(lib/imgui)
add_subdirectory(lib/glew)
add_subdirectory(lib/imgui)


file(GLOB_RECURSE SRC 
  src/*.cpp
)

add_executable(tui ${SRC} src/tui.cc)
target_include_directories(tui PUBLIC src)
target_link_libraries(tui PUBLIC glfw GL glew imgui OpenMP::OpenMP_CXX)

add_executable(cli ${SRC} src/cli.cc)
target_include_directories(cli PUBLIC src)

target_link_libraries(cli PUBLIC glfw GL glew imgui OpenMP::OpenMP_CXX)
